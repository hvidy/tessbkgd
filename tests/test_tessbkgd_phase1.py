"""Phase 1 test suite for tessbkgd packaging, fixes, precision, and FITS contract."""

from __future__ import annotations

import collections
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
from tessbkgd.tessbkgd import _as_float64_ndarray, _strip_for_fits_column

_X64_AT_IMPORT = bool(jax.config.jax_enable_x64)


def _as_bkgd_tpf(tpf: lk.TessTargetPixelFile) -> bkgd_tpf:
    tpf.__class__ = bkgd_tpf
    return tpf  # type: ignore[return-value]


def test_package_layout_and_imports() -> None:
    assert hasattr(tessbkgd, "__version__")
    assert tessbkgd.__version__ == "2.0.0"
    try:
        dist_ver = importlib.metadata.version("tessbkgd")
    except importlib.metadata.PackageNotFoundError:
        dist_ver = None
    if dist_ver is not None:
        assert tessbkgd.__version__ == dist_ver
    # Both modern top-level and legacy submodule paths must resolve to bkgd_tpf.
    assert tessbkgd.bkgd_tpf is bkgd_tpf
    assert tessbkgd.tessbkgd.bkgd_tpf is bkgd_tpf
    assert int(astropy.__version__.split(".")[0]) >= 6


def test_import_has_no_global_jax_x64_side_effect() -> None:
    assert _X64_AT_IMPORT is False
    assert bool(jax.config.jax_enable_x64) is False


def test_in_float64_and_standalone_cast_tree_scoping() -> None:
    with run_in("float32"):
        assert bool(jax.config.jax_enable_x64) is False

        # Standalone cast_tree(..., 'float64') while x64=False must NOT emit a
        # JAX truncation warning or downcast float64 BJD timestamps, must
        # promote NumPy scalar leaves (np.float32, np.complex64), and must
        # restore jax_enable_x64=False on exit.
        bjd = np.array([2458468.1234567890123], dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cast_out = cast_tree(
                {
                    "bjd": bjd,
                    "scalar_f32": np.float32(1.25),
                    "scalar_c64": np.complex64(1.0 - 2.0j),
                    "scalar_i32": np.int32(7),
                },
                dtype="float64",
            )
        assert bool(jax.config.jax_enable_x64) is False
        assert cast_out["bjd"].dtype == jnp.float64
        assert float(cast_out["bjd"][0]) == float(bjd[0])
        assert cast_out["scalar_f32"].dtype == jnp.float64
        assert cast_out["scalar_c64"].dtype == jnp.complex128
        assert cast_out["scalar_i32"].dtype == np.int32

        @in_float64
        def _accumulate(
            arr: object, scale: object, meta: dict[str, object]
        ) -> tuple[jax.Array, object]:
            assert bool(jax.config.jax_enable_x64) is True
            assert isinstance(arr, jax.Array)
            assert arr.dtype == jnp.float64
            assert isinstance(scale, jax.Array)
            assert scale.dtype == jnp.float64
            nested = meta["weights"]
            assert isinstance(nested, jax.Array)
            assert nested.dtype == jnp.float64
            assert meta["label"] == "test"
            assert np.asarray(meta["idx"]).dtype == np.int32
            return scale * jnp.sum(arr) + jnp.sum(nested), meta["q"]

        x32 = np.array([1.0, 1e-12, -1.0], dtype=np.float32)
        q32 = Quantity(np.array([2.0, 3.0], dtype=np.float32), unit="electron/s")
        total, q_out = _accumulate(
            x32,
            np.float32(0.5),
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
        assert bool(jax.config.jax_enable_x64) is False

        @in_float64
        def _boom(x: object) -> None:
            assert bool(jax.config.jax_enable_x64) is True
            raise RuntimeError("deliberate error")

        with pytest.raises(RuntimeError, match="deliberate error"):
            _boom(np.ones(3, dtype=np.float32))
        assert bool(jax.config.jax_enable_x64) is False

    # Verify that if caller had x64=True enabled prior, @in_float64 keeps it True.
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

    bkgd_cubic, _ = tpf.get_bkgd(bkgdmask=truth.sky_pixel_mask, order="cubic")
    np.testing.assert_allclose(
        bkgd_cubic.to_value("electron/s"), expected_sky, rtol=1e-4, atol=1e-2
    )

    with pytest.raises(ValueError, match="Unrecognised order 'quartic'"):
        tpf.get_bkgd(order="quartic")


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
    np.testing.assert_allclose(bkgd.value, expected_sky, rtol=1e-4, atol=2e-2)

    tpf.flux = tpf.flux - bkgd
    tpf.flux_bkg = bkgd
    assert read_bkgsub_state(tpf) is True
    assert tpf.meta["BKGSUB"] is True
    expected_star = tesscut_tpf_truth.star_raw[tesscut_tpf_truth.good_mask]
    np.testing.assert_allclose(tpf.flux.value, expected_star, rtol=1e-4, atol=2e-2)


def test_mundey_v2_bkgsub_false_and_subtract_refit_idempotence(
    mundey_v2_tpf_truth: SyntheticTPFTruth,
) -> None:
    # 1. Standard mundey v2.0 (BKGSUB=False, FLUX_BKG=0): manual setter idiom
    #    must flip BKGSUB -> True and keep subsequent get_bkgd() idempotent.
    tpf_clean = _as_bkgd_tpf(mundey_v2_tpf_truth.tpf)
    assert read_bkgsub_state(tpf_clean) is False
    bkgd_clean, _ = tpf_clean.get_bkgd(
        bkgdmask=mundey_v2_tpf_truth.sky_pixel_mask, order="linear"
    )
    expected_sky = mundey_v2_tpf_truth.sky_raw[mundey_v2_tpf_truth.good_mask]
    expected_star = mundey_v2_tpf_truth.star_raw[mundey_v2_tpf_truth.good_mask]
    np.testing.assert_allclose(
        bkgd_clean.to_value("electron/s"), expected_sky, rtol=1e-4, atol=1e-2
    )

    tpf_clean.flux = tpf_clean.flux - bkgd_clean
    tpf_clean.flux_bkg = bkgd_clean
    assert tpf_clean.hdu[0].header["BKGSUB"] is True
    assert tpf_clean.hdu[1].header["BKGSUB"] is True
    assert tpf_clean.meta["BKGSUB"] is True
    assert read_bkgsub_state(tpf_clean) is True

    refit_bkgd, _ = tpf_clean.get_bkgd(
        bkgdmask=mundey_v2_tpf_truth.sky_pixel_mask, order="linear"
    )
    np.testing.assert_allclose(
        refit_bkgd.to_value("electron/s"), expected_sky, rtol=1e-4, atol=1e-2
    )

    # 2. Stale non-zero FLUX_BKG with BKGSUB=False: must NOT add oldbkgd, must
    #    NOT warn, and subtract_bkgd() must be idempotent across repeated calls.
    stale_truth = make_synthetic_tpf(mode="mundey_v2_stale_bkg")
    tpf_stale = _as_bkgd_tpf(stale_truth.tpf)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert read_bkgsub_state(tpf_stale) is False
        sub1, _ = tpf_stale.subtract_bkgd(
            bkgdmask=stale_truth.sky_pixel_mask, order="linear"
        )
        assert read_bkgsub_state(tpf_stale) is True
        sub2, _ = tpf_stale.subtract_bkgd(
            bkgdmask=stale_truth.sky_pixel_mask, order="linear"
        )
    user_warnings = [
        w for w in record if issubclass(w.category, UserWarning)
    ]
    dep_warnings = [
        w for w in record if issubclass(w.category, DeprecationWarning)
    ]
    assert len(user_warnings) == 0
    assert len(dep_warnings) == 2
    np.testing.assert_allclose(
        sub1.to_value("electron/s"),
        stale_truth.sky_raw[stale_truth.good_mask],
        rtol=1e-4,
        atol=1e-2,
    )
    np.testing.assert_allclose(
        sub2.to_value("electron/s"),
        stale_truth.sky_raw[stale_truth.good_mask],
        rtol=1e-4,
        atol=1e-2,
    )
    np.testing.assert_allclose(
        tpf_stale.flux.to_value("electron/s"),
        expected_star,
        rtol=1e-4,
        atol=2e-2,
    )


def test_legacy_mundey_v01_warning_and_no_false_positive_on_bright_spoc_halo(
    mundey_v01_tpf_truth: SyntheticTPFTruth,
) -> None:
    tpf = _as_bkgd_tpf(mundey_v01_tpf_truth.tpf)
    with pytest.warns(UserWarning, match="BKGSUB header card is absent"):
        state = read_bkgsub_state(tpf)
    assert state is True

    with pytest.warns(UserWarning, match="legacy mundey"):
        tpf.get_bkgd(bkgdmask=mundey_v01_tpf_truth.sky_pixel_mask, order="linear")

    # Genuine MAST SPOC TPF of a bright halo star (e.g. Sirius / iot Dra) where
    # static stellar halo lifts dark_q10 >> 0.5 * med_bkg, while FLUX_BKG was
    # already subtracted (so faint-pixel FLUX does NOT co-vary with FLUX_BKG):
    # must NOT emit a false-positive UserWarning!
    bright_spoc = make_synthetic_tpf(
        mode="spoc", sky_base=120.0, halo_pedestal=600.0
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert read_bkgsub_state(bright_spoc.tpf) is True
    assert len(record) == 0

    # np.bool_ header cards and conflicting hdu[0] vs hdu[1] BKGSUB guard.
    bright_spoc.tpf.hdu[0].header["BKGSUB"] = np.bool_(False)
    bright_spoc.tpf.hdu[1].header["BKGSUB"] = np.bool_(False)
    assert read_bkgsub_state(bright_spoc.tpf) is False

    bright_spoc.tpf.hdu[1].header["BKGSUB"] = True
    with pytest.raises(ValueError, match="Conflicting boolean BKGSUB"):
        read_bkgsub_state(bright_spoc.tpf)


def test_flux_and_flux_bkg_setters_units_and_tunit_variants(
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

    # Compatible scaled count-rate & dimensionless conversions, plus unit rejection.
    tpf.flux_bkg = (expected_sky * 1e-3) * u.Unit("1000 electron / s")
    np.testing.assert_allclose(
        tpf.flux_bkg.to_value("electron/s"), expected_sky, rtol=1e-4, atol=1e-2
    )
    np.testing.assert_allclose(
        _strip_for_fits_column(np.array([120.0]) * u.Unit("ct / min")),
        np.array([2.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        _as_float64_ndarray(np.array([50.0]) * u.percent),
        np.array([0.5], dtype=np.float64),
    )
    with pytest.raises(u.UnitConversionError):
        tpf.flux = np.ones_like(expected_sky) * u.meter
    with pytest.raises(u.UnitConversionError):
        _as_float64_ndarray(np.ones(4) * u.Jy)

    # Standard FITS TUNIT formatting variants ('electron s-1', 'e- s-1').
    for alt_tunit in ("electron s-1", "e- s-1"):
        alt_truth = make_synthetic_tpf(
            mode="spoc", flux_unit_override=alt_tunit
        )
        alt_tpf = _as_bkgd_tpf(alt_truth.tpf)
        assert alt_tpf.flux.unit == u.Unit("electron/s")
        assert alt_tpf.flux_bkg.unit == u.Unit("electron/s")


def test_single_pixel_nan_and_rank_deficient_polynomial_guards() -> None:
    truth = make_synthetic_tpf(mode="mundey_v2", sky_order="linear")
    tpf = _as_bkgd_tpf(truth.tpf)

    good_rows = np.flatnonzero(truth.good_mask)
    flux_full = np.array(tpf.hdu[1].data["FLUX"], dtype=np.float32, copy=True)
    # Cadence 5: single-pixel NaN -> must still solve cleanly.
    flux_full[good_rows[5], 0, 0] = np.nan
    # Cadence 10: all-NaN -> must produce NaN, never 0.0.
    flux_full[good_rows[10], :, :] = np.nan
    # Cadence 15: 12 finite sky pixels surviving ONLY on column 0 (collinear,
    # M = 12 >= 3 but 2D rank = 2 < 3) -> must produce NaN rather than
    # extrapolating a minimum-norm 50%/100% plane error!
    flux_full[good_rows[15], :, 1:] = np.nan
    tpf.hdu[1].data["FLUX"] = flux_full

    bkgd, _ = tpf.get_bkgd(bkgdmask=truth.sky_pixel_mask, order="linear")
    bkgd_val = bkgd.to_value("electron/s")

    np.testing.assert_allclose(
        bkgd_val[5], truth.sky_raw[good_rows[5]], rtol=1e-4, atol=1e-2
    )
    assert np.all(np.isnan(bkgd_val[10]))
    assert np.all(np.isnan(bkgd_val[15]))

    # Static collinear mask (single edge column, M = 12 >= 3, rank = 2 < 3)
    # must raise ValueError immediately.
    collinear_mask = np.zeros_like(truth.sky_pixel_mask)
    collinear_mask[:, 0] = True
    with pytest.raises(ValueError, match="rank-deficient"):
        tpf.get_bkgd(bkgdmask=collinear_mask, order="linear")


def test_dark_and_bright_sky_default_mask_accuracy_and_star_filled_rejection() -> None:
    # Include flat dark sky (< 4 e-/s with gentle gradient inside [0, 10) bin)
    # and bright Earthshine sky (650 e-/s) with smooth un-truncated PSF wings.
    cases = (
        {"sky_base": 3.0, "sky_grad_x": 0.4, "sky_grad_y": -0.3},
        {"sky_base": 650.0, "sky_grad_x": 4.5, "sky_grad_y": -3.5},
    )
    for params in cases:
        truth = make_synthetic_tpf(mode="mundey_v2", **params)
        tpf = _as_bkgd_tpf(truth.tpf)
        bkgd, mask = tpf.get_bkgd(bkgdmask=None, order="linear")
        assert np.count_nonzero(mask) >= 3
        expected_sky = truth.sky_raw[truth.good_mask]
        np.testing.assert_allclose(
            bkgd.to_value("electron/s"), expected_sky, rtol=1e-3, atol=2e-2
        )

    # Stamp where PIXEL_QUALITY flags leave fewer than P unflagged sky pixels
    # must raise ValueError (never force size // 4 stellar pixels into mask).
    flagged_truth = make_synthetic_tpf(mode="mundey_v2")
    flagged_tpf = _as_bkgd_tpf(flagged_truth.tpf)
    _, ny, nx = flagged_tpf.flux.shape
    pq = np.full((ny, nx), np.uint16(PixelQuality.SOURCE_MASK), dtype=np.uint16)
    pq[0, :2] = 0  # only 2 unflagged pixels (< P = 3)
    flagged_tpf.hdu.append(fits.ImageHDU(data=pq, name="PIXEL_QUALITY"))
    with pytest.raises(ValueError, match="Insufficient finite background pixels"):
        flagged_tpf.get_bkgd(bkgdmask=None, order="linear")


def test_halocver_1_provenance_chol_validation_and_slicing_lockstep(
    spoc_tpf_truth: SyntheticTPFTruth,
    tesscut_tpf_truth: SyntheticTPFTruth,
    tmp_path: Path,
) -> None:
    tpf = spoc_tpf_truth.tpf
    n_raw = len(tpf.hdu[1].data)
    _, ny, nx = tpf.flux.shape
    n_basis = 6  # quadratic basis (P = 6 -> n_tril = 21)
    n_tril = n_basis * (n_basis + 1) // 2

    prior_pq = np.zeros((ny, nx), dtype=np.uint16)
    prior_pq[:, 2] |= np.uint16(PixelQuality.STRAP_COL)
    prior_pq[ny // 2, nx // 2] |= np.uint16(PixelQuality.SAT_BLEED)
    tpf.hdu.append(fits.ImageHDU(data=prior_pq, name="PIXEL_QUALITY"))

    rng = np.random.default_rng(123)
    bkg_basis = rng.normal(size=(n_basis, ny, nx)).astype(np.float32)

    chol_3d = np.zeros((n_raw, n_basis, n_basis), dtype=np.float32)
    tril_r, tril_c = np.tril_indices(n_basis)
    for r_idx in range(n_raw):
        vals = (r_idx + 1) * 100.0 + np.arange(1, n_tril + 1, dtype=np.float32)
        chol_3d[r_idx, tril_r, tril_c] = vals

    bkg_used = spoc_tpf_truth.sky_pixel_mask.copy()
    bkg_used[:, 2] = False

    write_tessbkgd_provenance(
        tpf.hdu,
        bkg_basis=bkg_basis,
        bkg_chol=chol_3d,
        bkg_used_mask=bkg_used,
    )
    updated_tpf = lk.TessTargetPixelFile(tpf.hdu)

    assert tpf.hdu[0].header["HALOCVER"] == HALOCVER == 1
    assert tpf.hdu[1].header["HALOCVER"] == HALOCVER == 1
    assert tpf.hdu[0].header["BKGSUB"] is True
    assert tpf.hdu[1].header["BKGSUB"] is True
    assert updated_tpf.meta["HALOCVER"] == 1
    assert updated_tpf.meta["BKGSUB"] is True
    assert read_bkgsub_state(updated_tpf) is True
    assert tpf.hdu[1].columns["BKG_CHOL"].unit == "e-/s"

    pq_out = tpf.hdu["PIXEL_QUALITY"].data
    assert pq_out.dtype == np.uint16
    assert np.all((pq_out[:, 2] & np.uint16(PixelQuality.STRAP_COL)) != 0)
    assert (pq_out[ny // 2, nx // 2] & np.uint16(PixelQuality.SAT_BLEED)) != 0
    assert np.array_equal(
        (pq_out & np.uint16(PixelQuality.BKG_USED)) != 0, bkg_used
    )

    basis_out = tpf.hdu["BKG_BASIS"].data
    assert basis_out.dtype == np.float32
    assert basis_out.shape == (n_basis, ny, nx)
    np.testing.assert_allclose(basis_out, bkg_basis)

    chol_col = tpf.hdu[1].data["BKG_CHOL"]
    assert chol_col.shape == (n_raw, n_tril)
    np.testing.assert_allclose(chol_col, pack_bkg_chol(chol_3d))
    np.testing.assert_allclose(unpack_bkg_chol(chol_col), chol_3d)

    good_chol = pack_bkg_chol(chol_3d)[tpf.quality_mask]
    sliced = tpf[5:25]
    assert len(sliced.time) == 20
    assert sliced.flux.shape[0] == 20
    sliced_chol = sliced.hdu[1].data["BKG_CHOL"][sliced.quality_mask]
    assert sliced_chol.shape == (20, n_tril)
    np.testing.assert_array_equal(sliced_chol, good_chol[5:25])
    np.testing.assert_array_equal(sliced.time.value, tpf.time.value[5:25])

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

    # MutableMapping meta sync (e.g. OrderedDict) + in-place update.
    updated_tpf.meta = collections.OrderedDict()
    new_used = np.zeros_like(bkg_used)
    new_used[0, :4] = True
    write_tessbkgd_provenance(
        updated_tpf,
        bkg_basis=bkg_basis[:3],
        bkg_chol=chol_3d[:, :3, :3],
        bkg_used_mask=new_used,
    )
    assert updated_tpf.meta["HALOCVER"] == 1
    assert updated_tpf.meta["BKGSUB"] is True
    assert updated_tpf.hdu[1].data["BKG_CHOL"].shape == (n_raw, 6)

    # Dimensionless TessCut TPF must NOT receive a spurious 'e-/s' unit on BKG_CHOL.
    tc_tpf = tesscut_tpf_truth.tpf
    write_tessbkgd_provenance(
        tc_tpf,
        bkg_basis=bkg_basis[:3],
        bkg_chol=chol_3d[:, :3, :3],
        bkg_used_mask=new_used,
    )
    assert tc_tpf.hdu[1].columns["BKG_CHOL"].unit is None

    # Cholesky validation guards: upper-triangular U, invalid width, P <= 0.
    upper_u = np.swapaxes(chol_3d[:, :3, :3], 1, 2)
    with pytest.raises(ValueError, match="lower-triangular"):
        pack_bkg_chol(upper_u)
    with pytest.raises(ValueError, match="triangular number"):
        pack_bkg_chol(np.ones((n_raw, 4), dtype=np.float32))
    with pytest.raises(ValueError, match=">= 1"):
        pack_bkg_chol(np.zeros((n_raw, 0, 0), dtype=np.float32))
    with pytest.raises(ValueError, match=">= 1"):
        unpack_bkg_chol(np.zeros((n_raw, 0), dtype=np.float32))
    with pytest.raises(ValueError, match="P >= 1"):
        unpack_bkg_chol(np.zeros((n_raw, 1), dtype=np.float32), n_basis=-1)
    with pytest.raises(ValueError, match="P >= 1"):
        write_tessbkgd_provenance(
            tc_tpf,
            bkg_basis=np.zeros((0, ny, nx), dtype=np.float32),
            bkg_chol=np.zeros((n_raw, 0, 0), dtype=np.float32),
            bkg_used_mask=new_used,
        )


def test_resolve_cache_dir_scratch_fred_and_symlink_guard(
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

    # Symlink into /fred/ (e.g. ~/.cache -> /fred/<proj>/...) must be caught!
    fred_symlink = tmp_path / "home_cache_link"
    fred_symlink.symlink_to("/fred/oz440/bpo/.cache")
    with pytest.raises(ValueError, match="/fred/"):
        resolve_cache_dir(fred_symlink / "tess_bright")
