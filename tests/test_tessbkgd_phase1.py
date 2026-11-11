"""Phase 1 test suite for tessbkgd packaging, fixes, precision, and FITS contract."""

from __future__ import annotations

import importlib.metadata
import warnings
from pathlib import Path

import astropy
import astropy.units as u
import jax
import jax.numpy as jnp
import lightkurve as lk
import numpy as np
import pytest
from astropy.io import fits
from astropy.units import Quantity

import tessbkgd
from conftest import SyntheticTPFTruth, make_synthetic_tpf
from tessbkgd import (
    HALOCVER,
    PixelQuality,
    bkgd_tpf,
    cast_tree,
    in_float64,
    pack_bkg_chol,
    read_bkgsub_state,
    resolve_cache_dir,
    run_in,
    unpack_bkg_chol,
    write_tessbkgd_provenance,
)

_X64_AT_IMPORT = bool(jax.config.jax_enable_x64)


def _as_bkgd_tpf(tpf: lk.TessTargetPixelFile) -> bkgd_tpf:
    tpf.__class__ = bkgd_tpf
    return tpf  # type: ignore[return-value]


def test_package_layout_and_imports() -> None:
    assert hasattr(tessbkgd, "__version__")
    assert tessbkgd.__version__ == importlib.metadata.version("tessbkgd")
    # Both modern top-level and legacy submodule paths must resolve to bkgd_tpf.
    assert tessbkgd.bkgd_tpf is bkgd_tpf
    assert tessbkgd.tessbkgd.bkgd_tpf is bkgd_tpf
    assert int(astropy.__version__.split(".")[0]) >= 6


def test_import_has_no_global_jax_x64_side_effect() -> None:
    assert _X64_AT_IMPORT is False
    assert bool(jax.config.jax_enable_x64) is False


def test_in_float64_scoping_and_exception_restoration() -> None:
    with run_in("float32"):
        assert bool(jax.config.jax_enable_x64) is False

        @in_float64
        def _accumulate(
            arr: object, meta: dict[str, object]
        ) -> tuple[jax.Array, object]:
            assert bool(jax.config.jax_enable_x64) is True
            assert isinstance(arr, jax.Array)
            assert arr.dtype == jnp.float64
            nested = meta["weights"]
            assert isinstance(nested, jax.Array)
            assert nested.dtype == jnp.float64
            # Integer leaves and strings must pass through untouched.
            assert meta["label"] == "test"
            assert np.asarray(meta["idx"]).dtype == np.int32
            return jnp.sum(arr) + jnp.sum(nested), meta["q"]

        x32 = np.array([1.0, 1e-12, -1.0], dtype=np.float32)
        q32 = Quantity(np.array([2.0, 3.0], dtype=np.float32), unit="electron/s")
        total, q_out = _accumulate(
            x32,
            {
                "weights": jnp.ones(2, dtype=jnp.float32) * 0.5,
                "idx": np.array([1, 2], dtype=np.int32),
                "label": "test",
                "q": q32,
            },
        )
        assert total.dtype == jnp.float64
        assert isinstance(q_out, Quantity)
        assert q_out.dtype == np.float64
        assert q_out.unit == u.Unit("electron/s")
        # Prior float32 state restored on exit.
        assert bool(jax.config.jax_enable_x64) is False

        @in_float64
        def _boom(x: object) -> None:
            assert bool(jax.config.jax_enable_x64) is True
            raise RuntimeError("deliberate error")

        with pytest.raises(RuntimeError, match="deliberate error"):
            _boom(np.ones(3, dtype=np.float32))
        assert bool(jax.config.jax_enable_x64) is False

    # Also verify that if caller had x64=True enabled prior, @in_float64 keeps it True.
    with run_in("float64"):
        assert bool(jax.config.jax_enable_x64) is True

        @in_float64
        def _identity(v: object) -> object:
            return v

        _identity(np.ones(2, dtype=np.float32))
        assert bool(jax.config.jax_enable_x64) is True

    with pytest.raises(ValueError, match="float32"):
        cast_tree(np.ones(2), dtype="float16")


def test_get_bkgd_verbose_and_spoc_linear_recovery(
    spoc_tpf_truth: SyntheticTPFTruth, capsys: pytest.CaptureFixture[str]
) -> None:
    tpf = _as_bkgd_tpf(spoc_tpf_truth.tpf)
    assert read_bkgsub_state(tpf) is True

    bkgd, mask = tpf.get_bkgd(
        bkgdmask=spoc_tpf_truth.sky_pixel_mask,
        order="linear",
        verbose=True,
    )
    captured = capsys.readouterr()
    assert "Fitting linear background" in captured.out
    assert isinstance(bkgd, Quantity)
    assert bkgd.unit == u.Unit("electron/s")
    assert mask.shape == spoc_tpf_truth.sky_pixel_mask.shape

    expected_sky = spoc_tpf_truth.sky_raw[spoc_tpf_truth.good_mask]
    np.testing.assert_allclose(
        bkgd.to_value("electron/s"), expected_sky, rtol=1e-4, atol=1e-2
    )


def test_get_bkgd_quadratic_recovery_and_invalid_order() -> None:
    truth = make_synthetic_tpf(mode="spoc", sky_order="quadratic")
    tpf = _as_bkgd_tpf(truth.tpf)

    bkgd, _ = tpf.get_bkgd(bkgdmask=truth.sky_pixel_mask, order="quadratic")
    expected_sky = truth.sky_raw[truth.good_mask]
    np.testing.assert_allclose(
        bkgd.to_value("electron/s"), expected_sky, rtol=1e-4, atol=1e-2
    )

    with pytest.raises(ValueError, match="Unrecognised order 'cubic'"):
        tpf.get_bkgd(order="cubic")


def test_tesscut_tpf_no_tunit5_and_zero_bkg(
    tesscut_tpf_truth: SyntheticTPFTruth,
) -> None:
    tpf = _as_bkgd_tpf(tesscut_tpf_truth.tpf)
    assert "TUNIT5" not in tpf.get_header(1)
    assert read_bkgsub_state(tpf) is False

    # Previously raised UnitConversionError at flux + oldbkgd and at setter assignment.
    bkgd, mask = tpf.get_bkgd(order="linear")
    assert mask.any()
    assert bkgd.unit == u.dimensionless_unscaled

    expected_sky = tesscut_tpf_truth.sky_raw[tesscut_tpf_truth.good_mask]
    np.testing.assert_allclose(bkgd.value, expected_sky, rtol=2e-2, atol=1.0)

    tpf.flux = tpf.flux - bkgd
    tpf.flux_bkg = bkgd
    expected_star = tesscut_tpf_truth.star_raw[tesscut_tpf_truth.good_mask]
    np.testing.assert_allclose(tpf.flux.value, expected_star, atol=1.0)


def test_mundey_v2_bkgsub_false_does_not_add_stale_oldbkgd(
    mundey_v2_tpf_truth: SyntheticTPFTruth,
) -> None:
    # Standard mundey v2.0 (BKGSUB=False, FLUX_BKG=0).
    tpf_clean = _as_bkgd_tpf(mundey_v2_tpf_truth.tpf)
    assert read_bkgsub_state(tpf_clean) is False
    bkgd_clean, _ = tpf_clean.get_bkgd(
        bkgdmask=mundey_v2_tpf_truth.sky_pixel_mask, order="linear"
    )
    expected_sky = mundey_v2_tpf_truth.sky_raw[mundey_v2_tpf_truth.good_mask]
    np.testing.assert_allclose(
        bkgd_clean.to_value("electron/s"), expected_sky, rtol=1e-4, atol=1e-2
    )

    # Even if stale non-zero FLUX_BKG is present, BKGSUB=False must NOT add oldbkgd
    # and must NOT emit the legacy mundey v0.1 UserWarning.
    stale_truth = make_synthetic_tpf(mode="mundey_v2_stale_bkg")
    tpf_stale = _as_bkgd_tpf(stale_truth.tpf)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert read_bkgsub_state(tpf_stale) is False
        bkgd_stale, _ = tpf_stale.get_bkgd(
            bkgdmask=stale_truth.sky_pixel_mask, order="linear"
        )
    assert len(record) == 0
    np.testing.assert_allclose(
        bkgd_stale.to_value("electron/s"),
        stale_truth.sky_raw[stale_truth.good_mask],
        rtol=1e-4,
        atol=1e-2,
    )


def test_legacy_mundey_v01_emits_userwarning(
    mundey_v01_tpf_truth: SyntheticTPFTruth,
) -> None:
    tpf = _as_bkgd_tpf(mundey_v01_tpf_truth.tpf)
    with pytest.warns(UserWarning, match="BKGSUB header card is absent"):
        state = read_bkgsub_state(tpf)
    assert state is True

    with pytest.warns(UserWarning, match="legacy mundey"):
        tpf.get_bkgd(bkgdmask=mundey_v01_tpf_truth.sky_pixel_mask, order="linear")


def test_flux_and_flux_bkg_setters_with_quantity_and_fits_roundtrip(
    spoc_tpf_truth: SyntheticTPFTruth, tmp_path: Path
) -> None:
    # Round-trip to FITS on disk first so hdu[1].data uses big-endian >f4 FITS_rec
    # buffers just like real MAST SPOC files on Astropy >= 6.
    fits_path = tmp_path / "synthetic_spoc_tp.fits"
    spoc_tpf_truth.tpf.to_fits(str(fits_path), overwrite=True)

    loaded = lk.read(str(fits_path))
    assert isinstance(loaded, lk.TessTargetPixelFile)
    tpf = _as_bkgd_tpf(loaded)

    newbkgd, _ = tpf.get_bkgd(
        bkgdmask=spoc_tpf_truth.sky_pixel_mask, order="linear"
    )
    # Exact notebook idiom from iot_Dra_tpf.ipynb & sirius_example.ipynb:
    updated_flux = tpf.flux + tpf.flux_bkg - newbkgd
    assert isinstance(updated_flux, Quantity)
    assert updated_flux.unit == u.Unit("electron/s")

    tpf.flux = updated_flux
    tpf.flux_bkg = newbkgd

    expected_star = spoc_tpf_truth.star_raw[spoc_tpf_truth.good_mask]
    expected_sky = spoc_tpf_truth.sky_raw[spoc_tpf_truth.good_mask]
    np.testing.assert_allclose(
        tpf.flux.to_value("electron/s"), expected_star, atol=0.25
    )
    np.testing.assert_allclose(
        tpf.flux_bkg.to_value("electron/s"), expected_sky, rtol=1e-4, atol=1e-2
    )

    # Also test compatible scaled unit (ke-/s) and incompatible unit rejection.
    tpf.flux_bkg = (expected_sky * 1e-3) * u.Unit("1000 electron / s")
    np.testing.assert_allclose(
        tpf.flux_bkg.to_value("electron/s"), expected_sky, rtol=1e-4, atol=1e-2
    )
    with pytest.raises(u.UnitConversionError):
        tpf.flux = np.ones_like(expected_sky) * u.meter


def test_single_pixel_nan_does_not_wipe_cadence_and_all_nan_cadence_is_nan() -> None:
    truth = make_synthetic_tpf(mode="mundey_v2", sky_order="linear")
    tpf = _as_bkgd_tpf(truth.tpf)

    # Inject single-pixel NaN on good cadence 5 and all-NaN frame on good cadence 10.
    good_rows = np.flatnonzero(truth.good_mask)
    flux_full = np.array(tpf.hdu[1].data["FLUX"], dtype=np.float32, copy=True)
    flux_full[good_rows[5], 0, 0] = np.nan
    flux_full[good_rows[10], :, :] = np.nan
    tpf.hdu[1].data["FLUX"] = flux_full

    bkgd, _ = tpf.get_bkgd(bkgdmask=truth.sky_pixel_mask, order="linear")
    bkgd_val = bkgd.to_value("electron/s")

    # Cadence 5 must NOT be zeroed out; it must match the true linear background!
    np.testing.assert_allclose(
        bkgd_val[5], truth.sky_raw[good_rows[5]], rtol=1e-4, atol=1e-2
    )
    # Cadence 10 (all-NaN) must be NaN, never 0.0!
    assert np.all(np.isnan(bkgd_val[10]))


def test_dark_and_bright_sky_default_mask_does_not_wipe_out() -> None:
    for sky_lvl in (4.0, 650.0):
        truth = make_synthetic_tpf(mode="mundey_v2", sky_base=sky_lvl)
        tpf = _as_bkgd_tpf(truth.tpf)
        bkgd, mask = tpf.get_bkgd(bkgdmask=None, order="linear")
        assert np.count_nonzero(mask) >= 3
        assert np.all(np.isfinite(bkgd.to_value("electron/s")))


def test_halocver_1_provenance_and_cadence_slicing_lockstep(
    spoc_tpf_truth: SyntheticTPFTruth, tmp_path: Path
) -> None:
    tpf = spoc_tpf_truth.tpf
    n_raw = len(tpf.hdu[1].data)
    _, ny, nx = tpf.flux.shape
    n_basis = 6  # quadratic basis (P = 6 -> n_tril = 21)
    n_tril = n_basis * (n_basis + 1) // 2

    # Pre-populate PIXEL_QUALITY with STRAP_COL and SAT_BLEED bits to test preservation.
    prior_pq = np.zeros((ny, nx), dtype=np.uint16)
    prior_pq[:, 2] |= np.uint16(PixelQuality.STRAP_COL)
    prior_pq[ny // 2, nx // 2] |= np.uint16(PixelQuality.SAT_BLEED)
    tpf.hdu.append(fits.ImageHDU(data=prior_pq, name="PIXEL_QUALITY"))

    rng = np.random.default_rng(123)
    bkg_basis = rng.normal(size=(n_basis, ny, nx)).astype(np.float32)

    # Construct distinct per-raw-cadence lower-triangular Cholesky factors L_t.
    chol_3d = np.zeros((n_raw, n_basis, n_basis), dtype=np.float32)
    tril_r, tril_c = np.tril_indices(n_basis)
    for r_idx in range(n_raw):
        vals = (r_idx + 1) * 100.0 + np.arange(1, n_tril + 1, dtype=np.float32)
        chol_3d[r_idx, tril_r, tril_c] = vals

    bkg_used = spoc_tpf_truth.sky_pixel_mask.copy()
    bkg_used[:, 2] = False  # disjoint from STRAP_COL

    write_tessbkgd_provenance(
        tpf.hdu,
        bkg_basis=bkg_basis,
        bkg_chol=chol_3d,
        bkg_used_mask=bkg_used,
    )
    updated_tpf = lk.TessTargetPixelFile(tpf.hdu)

    # 1. Header cards & read-only HduToMetaMapping exposure.
    assert tpf.hdu[0].header["HALOCVER"] == HALOCVER == 1
    assert tpf.hdu[1].header["HALOCVER"] == HALOCVER == 1
    assert tpf.hdu[0].header["BKGSUB"] is True
    assert tpf.hdu[1].header["BKGSUB"] is True
    assert updated_tpf.meta["HALOCVER"] == 1
    assert updated_tpf.meta["BKGSUB"] is True
    assert read_bkgsub_state(updated_tpf) is True

    # 2. PIXEL_QUALITY preserves STRAP_COL/SAT_BLEED and sets Bit 4 (BKG_USED).
    pq_out = tpf.hdu["PIXEL_QUALITY"].data
    assert pq_out.dtype == np.uint16
    assert np.all((pq_out[:, 2] & np.uint16(PixelQuality.STRAP_COL)) != 0)
    assert (pq_out[ny // 2, nx // 2] & np.uint16(PixelQuality.SAT_BLEED)) != 0
    assert np.array_equal(
        (pq_out & np.uint16(PixelQuality.BKG_USED)) != 0, bkg_used
    )

    # 3. BKG_BASIS ImageHDU shape & dtype.
    basis_out = tpf.hdu["BKG_BASIS"].data
    assert basis_out.dtype == np.float32
    assert basis_out.shape == (n_basis, ny, nx)
    np.testing.assert_allclose(basis_out, bkg_basis)

    # 4. Packed BKG_CHOL in hdu[1] and 1-to-1 lockstep slicing under tpf[5:25].
    chol_col = tpf.hdu[1].data["BKG_CHOL"]
    assert chol_col.shape == (n_raw, n_tril)
    np.testing.assert_allclose(chol_col, pack_bkg_chol(chol_3d))
    np.testing.assert_allclose(unpack_bkg_chol(chol_col), chol_3d)

    good_chol = pack_bkg_chol(chol_3d)[tpf.quality_mask]
    sliced = tpf[5:25]
    assert len(sliced.time) == 20
    assert sliced.flux.shape[0] == 20
    # In lightkurve, tpf[5:25] slices quality-masked cadences 5..25 of hdu[1].data!
    sliced_chol = sliced.hdu[1].data["BKG_CHOL"][sliced.quality_mask]
    assert sliced_chol.shape == (20, n_tril)
    np.testing.assert_array_equal(sliced_chol, good_chol[5:25])
    np.testing.assert_array_equal(sliced.time.value, tpf.time.value[5:25])

    # 5. FITS serialization round-trip preserves all extensions and columns.
    out_fits = tmp_path / "provenance_roundtrip.fits"
    tpf.to_fits(str(out_fits), overwrite=True)
    reloaded = lk.read(str(out_fits))
    assert reloaded.meta["HALOCVER"] == 1
    assert reloaded.meta["BKGSUB"] is True
    assert reloaded.hdu["BKG_BASIS"].data.shape == (n_basis, ny, nx)
    reloaded_slice = reloaded[5:25]
    np.testing.assert_array_equal(
        reloaded_slice.hdu[1].data["BKG_CHOL"][reloaded_slice.quality_mask],
        good_chol[5:25],
    )

    # 6. Repeated provenance write updates BKG_USED bit and BKG_CHOL in place.
    new_used = np.zeros_like(bkg_used)
    new_used[0, :4] = True
    write_tessbkgd_provenance(
        updated_tpf,
        bkg_basis=bkg_basis[:3],
        bkg_chol=chol_3d[:, :3, :3],
        bkg_used_mask=new_used,
    )
    assert updated_tpf.meta["HALOCVER"] == 1
    assert np.array_equal(
        (updated_tpf.hdu["PIXEL_QUALITY"].data & np.uint16(PixelQuality.BKG_USED))
        != 0,
        new_used,
    )
    assert updated_tpf.hdu[1].data["BKG_CHOL"].shape == (n_raw, 6)


def test_resolve_cache_dir_scratch_and_fred_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("TESS_BRIGHT_CACHE_DIR", raising=False)
    monkeypatch.delenv("TESSBKGD_CACHE_DIR", raising=False)
    scratch_dir = tmp_path / "aphid_scratch"
    monkeypatch.setenv("SCRATCH", str(scratch_dir))

    resolved = resolve_cache_dir()
    assert resolved == (scratch_dir / ".cache" / "tess_bright").resolve()

    with pytest.raises(ValueError, match="/fred/"):
        resolve_cache_dir("/fred/oz440/bpo/.cache/tess_bright")

    with pytest.warns(UserWarning, match="/fred/"):
        non_strict = resolve_cache_dir(
            "/fred/oz440/bpo/.cache/tess_bright", strict_hpc=False
        )
    assert str(non_strict).startswith("/fred/")
