"""v2.0.0 release-polish and adversarial-audit regression tests for tessbkgd."""

from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path

import lightkurve as lk
import numpy as np
import pytest
from astropy.io import fits

import tessbkgd
from tessbkgd import (
    HALOCVER,
    BkgCovariance,
    PixelQuality,
    estimate_background,
    normalized_grid,
    read_bkg_covariance,
    read_smear_err,
    resolve_cache_dir,
    subtract_background,
)


def _build_mundey_tpf(
    flux: np.ndarray,
    *,
    flux_err: np.ndarray | None = None,
    smear_err: np.ndarray | None = None,
    quality: np.ndarray | None = None,
    pixel_quality: np.ndarray | None = None,
    bkgsub: bool = False,
) -> lk.TessTargetPixelFile:
    """Construct a synthetic ``HALOCVER = 1`` ``mundey``-calibrated TPF."""
    n_raw, ny, nx = flux.shape
    n_pix = ny * nx
    img_dim = f"({nx}, {ny})"
    flux_unit = "e-/s"

    time = np.linspace(2000.0, 2027.0, n_raw, dtype=np.float64)
    cadenceno = np.arange(10000, 10000 + n_raw, dtype=np.int32)
    timecorr = np.zeros(n_raw, dtype=np.float32)
    qual = (
        np.zeros(n_raw, dtype=np.int32)
        if quality is None
        else np.asarray(quality, dtype=np.int32)
    )
    raw_cnts = np.zeros((n_raw, ny, nx), dtype=np.int32)
    pos_corr1 = np.zeros(n_raw, dtype=np.float32)
    pos_corr2 = np.zeros(n_raw, dtype=np.float32)

    cols = [
        fits.Column(
            name="TIME", format="D", unit="BJD - 2457000, days", array=time
        ),
        fits.Column(name="TIMECORR", format="E", unit="d", array=timecorr),
        fits.Column(name="CADENCENO", format="J", array=cadenceno),
        fits.Column(
            name="RAW_CNTS",
            format=f"{n_pix}J",
            dim=img_dim,
            unit="count",
            array=raw_cnts,
        ),
        fits.Column(
            name="FLUX",
            format=f"{n_pix}E",
            dim=img_dim,
            unit=flux_unit,
            array=np.asarray(flux, dtype=np.float32),
        ),
    ]
    if flux_err is not None:
        cols.append(
            fits.Column(
                name="FLUX_ERR",
                format=f"{n_pix}E",
                dim=img_dim,
                unit=flux_unit,
                array=np.asarray(flux_err, dtype=np.float32),
            )
        )
    cols.extend(
        [
            fits.Column(
                name="FLUX_BKG",
                format=f"{n_pix}E",
                dim=img_dim,
                unit=flux_unit,
                array=np.zeros((n_raw, ny, nx), dtype=np.float32),
            ),
            fits.Column(
                name="FLUX_BKG_ERR",
                format=f"{n_pix}E",
                dim=img_dim,
                unit=flux_unit,
                array=np.zeros((n_raw, ny, nx), dtype=np.float32),
            ),
            fits.Column(name="QUALITY", format="J", array=qual),
            fits.Column(
                name="POS_CORR1", format="E", unit="pixel", array=pos_corr1
            ),
            fits.Column(
                name="POS_CORR2", format="E", unit="pixel", array=pos_corr2
            ),
        ]
    )
    if smear_err is not None:
        cols.append(
            fits.Column(
                name="SMEAR_ERR",
                format=f"{nx}E",
                dim=f"({nx})",
                unit=flux_unit,
                array=np.asarray(smear_err, dtype=np.float32),
            )
        )

    primary_hdr = fits.Header()
    primary_hdr["TELESCOP"] = "TESS"
    primary_hdr["INSTRUME"] = "TESS Photometer"
    primary_hdr["OBJECT"] = "TIC 246813579"
    primary_hdr["TICID"] = 246813579
    primary_hdr["SECTOR"] = 21
    primary_hdr["CAMERA"] = 1
    primary_hdr["CCD"] = 3
    primary_hdr["HALOCVER"] = HALOCVER
    primary_hdr["BKGSUB"] = bkgsub
    primary_hdu = fits.PrimaryHDU(header=primary_hdr)

    pixels_hdu = fits.BinTableHDU.from_columns(cols, name="PIXELS")
    pixels_hdu.header["HALOCVER"] = HALOCVER
    pixels_hdu.header["BKGSUB"] = bkgsub

    aperture_hdu = fits.ImageHDU(
        data=np.ones((ny, nx), dtype=np.int32), name="APERTURE"
    )
    hdus: list[fits.PrimaryHDU | fits.BinTableHDU | fits.ImageHDU] = [
        primary_hdu,
        pixels_hdu,
        aperture_hdu,
    ]
    if pixel_quality is not None:
        hdus.append(
            fits.ImageHDU(
                data=np.asarray(pixel_quality, dtype=np.uint16),
                name="PIXEL_QUALITY",
            )
        )
    return lk.TessTargetPixelFile(fits.HDUList(hdus))


def test_tbkg01_smear_err_monte_carlo_and_halocver_invariants() -> None:
    """Verify [TBKG-01] pull calibration, pixel-independent BKG_CHOL, & idempotency."""
    rng = np.random.default_rng(20261114)
    n_mc = 200
    ny, nx = 12, 14
    mask = np.ones((ny, nx), dtype=bool)
    mask[4:8, 5:9] = False  # central stellar aperture excluded

    eta, xi = normalized_grid(ny, nx)
    eta_g, xi_g = np.meshgrid(eta, xi, indexing="ij")
    sky_true = 140.0 + 11.0 * xi_g - 7.5 * eta_g
    star = 1200.0 * np.exp(-0.5 * ((eta_g / 0.16) ** 2 + (xi_g / 0.16) ** 2))
    star *= (~mask).astype(np.float64)

    sigma_indep = (0.75 + 0.35 * (xi_g + 1.0)).astype(np.float32).astype(
        np.float64
    )
    sigma_smear_1d = (
        (0.55 + 0.30 * np.cos(0.5 * np.pi * xi)).astype(np.float32).astype(
            np.float64
        )
    )
    smear_err = np.broadcast_to(sigma_smear_1d, (n_mc, nx)).copy()
    sigma_u = np.sqrt(sigma_indep**2 + sigma_smear_1d[None, :] ** 2)
    flux_err_in = np.broadcast_to(sigma_u, (n_mc, ny, nx)).copy()

    eps_indep = rng.normal(0.0, 1.0, size=(n_mc, ny, nx)) * sigma_indep
    eps_smear = rng.normal(0.0, 1.0, size=(n_mc, nx)) * sigma_smear_1d
    u_cube = (
        sky_true[None, :, :]
        + star[None, :, :]
        + eps_indep
        + eps_smear[:, None, :]
    )

    cal = _build_mundey_tpf(
        u_cube, flux_err=flux_err_in, smear_err=smear_err, bkgsub=False
    )
    cal_no_smear = _build_mundey_tpf(
        u_cube, flux_err=flux_err_in, smear_err=None, bkgsub=False
    )

    sub, diag = subtract_background(
        cal, order=1, mask=mask, return_diagnostics=True
    )
    sub_no_smear = subtract_background(cal_no_smear, order=1, mask=mask)
    est = estimate_background(cal, order=1, mask=mask)

    bkg_hat = sub.flux_bkg.to_value("electron/s")
    bkg_err = sub.flux_bkg_err.to_value("electron/s")
    bkg_err_indep_only = sub_no_smear.flux_bkg_err.to_value("electron/s")

    # (i) Standalone FLUX_BKG_ERR & estimate_background bkg_err include coherent
    #     column smear variance and calibrate the 200-realization pull std:
    assert np.all(bkg_err > bkg_err_indep_only)
    np.testing.assert_allclose(
        est.bkg_err.to_value("electron/s"), bkg_err, rtol=1e-6, atol=1e-6
    )
    pulls = (bkg_hat - sky_true[None, :, :]) / bkg_err
    pull_std = float(np.std(pulls, ddof=1))
    assert 0.93 <= pull_std <= 1.07, f"Uncalibrated pull_std={pull_std:.4f}"

    # Verify that ignoring off-diagonal column smear would have inflated pulls > 1.15:
    pulls_naive = (bkg_hat - sky_true[None, :, :]) / bkg_err_indep_only
    assert float(np.std(pulls_naive, ddof=1)) > 1.25

    # (ii) BKG_CHOL and sub.FLUX_ERR remain strictly pixel-independent (identical
    #      to the control TPF without SMEAR_ERR):
    np.testing.assert_array_equal(
        sub.hdu[1].data["BKG_CHOL"], sub_no_smear.hdu[1].data["BKG_CHOL"]
    )
    np.testing.assert_array_equal(
        sub.hdu[1].data["FLUX_ERR"], sub_no_smear.hdu[1].data["FLUX_ERR"]
    )

    # (iii) Downstream halophot invariant:
    #       sub.FLUX_ERR^2 - SMEAR_ERR^2 - ||L_t^T phi||^2 == sigma_indep^2
    cov = read_bkg_covariance(sub)
    se_vis = read_smear_err(sub)
    assert cov is not None and se_vis is not None
    np.testing.assert_allclose(cov.chol, diag.bkg_chol, rtol=1e-6, atol=1e-6)
    lt_phi = np.einsum("tqp,qyx->tpyx", cov.chol, cov.basis)
    bkg_indep_sq = np.sum(lt_phi**2, axis=1)
    fe_out_sq = sub.flux_err.to_value("electron/s") ** 2
    recovered_indep_sq = fe_out_sq - se_vis[:, None, :] ** 2 - bkg_indep_sq
    np.testing.assert_allclose(
        recovered_indep_sq,
        np.broadcast_to(sigma_indep**2, (n_mc, ny, nx)),
        rtol=2e-5,
        atol=2e-5,
    )

    # (iv) Idempotency of subtract_background(subtract_background(cal)) when
    #      SMEAR_ERR > 0, both with explicit mask and auto-masking + quality flags:
    sub2 = subtract_background(sub, order=1, mask=mask)
    for col in ("FLUX", "FLUX_BKG", "FLUX_ERR", "FLUX_BKG_ERR", "BKG_CHOL"):
        np.testing.assert_allclose(
            sub2.hdu[1].data[col],
            sub.hdu[1].data[col],
            rtol=2e-5,
            atol=2e-4,
        )

    qual = np.zeros(n_mc, dtype=np.int32)
    qual[5] = 128
    cal_auto = _build_mundey_tpf(
        u_cube[:32],
        flux_err=flux_err_in[:32],
        smear_err=smear_err[:32],
        quality=qual[:32],
        bkgsub=False,
    )
    auto1 = subtract_background(cal_auto, order=1)
    auto2 = subtract_background(auto1, order=1)
    for col in ("FLUX", "FLUX_BKG", "FLUX_ERR", "FLUX_BKG_ERR", "BKG_CHOL"):
        np.testing.assert_allclose(
            auto2.hdu[1].data[col],
            auto1.hdu[1].data[col],
            rtol=2e-5,
            atol=2e-4,
        )


def test_tbkg01_smear_err_nan_propagation_and_masked_columns() -> None:
    """Verify NaN SMEAR_ERR propagates iff column c has active influence a != 0."""
    rng = np.random.default_rng(919)
    n_cad, ny, nx = 10, 10, 12
    eta, xi = normalized_grid(ny, nx)
    eta_g, xi_g = np.meshgrid(eta, xi, indexing="ij")
    sky = (80.0 + 5.0 * xi_g - 3.0 * eta_g)[None, :, :] + rng.normal(
        0.0, 0.5, (n_cad, ny, nx)
    )
    flux_err = np.full((n_cad, ny, nx), 0.7, dtype=np.float64)
    smear_err = np.full((n_cad, nx), 0.4, dtype=np.float64)

    # Column 4 is flagged SMEAR_UNRESOLVED -> excluded from bkg_mask -> a_{t,j,4} == 0.
    pq = np.zeros((ny, nx), dtype=np.uint16)
    pq[:, 4] |= np.uint16(PixelQuality.SMEAR_UNRESOLVED)

    # Cadence 2: NaN SMEAR_ERR on active column 0 -> FLUX_BKG_ERR[2] must be NaN,
    # while FLUX_ERR[2] and BKG_CHOL[2] stay finite.
    smear_err[2, 0] = np.nan
    # Cadence 3: NaN SMEAR_ERR only on excluded column 4 -> FLUX_BKG_ERR[3] finite!
    smear_err[3, 4] = np.nan
    # Cadence 5: all-NaN FLUX -> everything NaN.
    sky[5, :, :] = np.nan

    tpf = _build_mundey_tpf(
        sky,
        flux_err=flux_err,
        smear_err=smear_err,
        pixel_quality=pq,
        bkgsub=False,
    )
    sub = subtract_background(tpf, order=1)

    fbe = sub.hdu[1].data["FLUX_BKG_ERR"]
    fe = sub.hdu[1].data["FLUX_ERR"]
    chol = sub.hdu[1].data["BKG_CHOL"]

    assert np.all(np.isnan(fbe[2]))
    assert np.all(np.isfinite(fe[2]))
    assert np.all(np.isfinite(chol[2]))

    assert np.all(np.isfinite(fbe[3]))
    assert np.all(np.isfinite(fe[3]))

    assert np.all(np.isnan(fbe[5]))
    assert np.all(np.isnan(fe[5]))
    assert np.all(np.isnan(chol[5]))


def test_tbkg02_tiny_2x2_pure_sky_cutout_downgrades_cleanly() -> None:
    """Verify [TBKG-02] (2, 2) pure-sky stamp downgrades to BKG_ORD = 0."""
    for seed in (1, 7, 42, 99):
        rng = np.random.default_rng(seed)
        n_cad, ny, nx = 32, 2, 2
        sky_level = 95.0 + 12.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, n_cad))
        u_tiny = sky_level[:, None, None] + rng.normal(
            0.0, 0.4, (n_cad, ny, nx)
        )
        tpf_tiny = _build_mundey_tpf(
            u_tiny,
            flux_err=np.full_like(u_tiny, 0.4),
            smear_err=np.full((n_cad, nx), 0.15),
            bkgsub=False,
        )
        sub, diag = subtract_background(
            tpf_tiny, order=2, allow_downgrade=True, return_diagnostics=True
        )
        est = estimate_background(tpf_tiny, order=2, allow_downgrade=True)

        assert diag.requested_order == 2
        assert diag.effective_order == 0
        assert int(np.count_nonzero(diag.bkg_mask)) >= 2
        assert sub.hdu["BKG_BASIS"].header["BKG_ORD"] == 0
        assert sub.hdu["BKG_BASIS"].data.shape == (1, 2, 2)
        np.testing.assert_allclose(
            np.median(est.bkg.to_value("electron/s"), axis=(1, 2)),
            sky_level,
            atol=1.2,
        )

    # With allow_downgrade=False, order=2 (P=6 > 4 pixels) still raises ValueError:
    with pytest.raises(ValueError, match="Insufficient finite background pixels"):
        subtract_background(tpf_tiny, order=2, allow_downgrade=False)


def test_tbkg03_version_changelog_and_sdist_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify [TBKG-03] version 2.0.0, CHANGELOG.md, and sdist exclusions."""
    assert tessbkgd.__version__ == "2.0.0"

    repo_root = Path(__file__).resolve().parents[1]
    pyproject_path = repo_root / "pyproject.toml"
    changelog_path = repo_root / "CHANGELOG.md"

    assert changelog_path.is_file()
    changelog_text = changelog_path.read_text(encoding="utf-8")
    assert "## [2.0.0] - 2026-11-14" in changelog_text
    assert "HALOCVER = 1" in changelog_text
    assert "SMEAR_ERR" in changelog_text

    with pyproject_path.open("rb") as fh:
        cfg = tomllib.load(fh)
    assert cfg["project"]["version"] == "2.0.0"
    sdist_excl = cfg["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"]
    for pattern in ("notebooks/*", "*.ipynb", "*.fits"):
        assert pattern in sdist_excl

    # Simulate PYTHONPATH-only import without installed .dist-info metadata:
    def _raise_not_found(pkg_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(pkg_name)

    monkeypatch.setattr(importlib.metadata, "version", _raise_not_found)
    from test_tessbkgd_phase1 import test_package_layout_and_imports

    test_package_layout_and_imports()


def test_xpkg01_and_xpkg03_cache_precedence_and_contract_readers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify [XPKG-01] resolve_cache_dir precedence/create and [XPKG-03] readers."""
    for name in (
        "TESSBKGD_CACHE_DIR",
        "TESS_BRIGHT_CACHE_DIR",
        "SCRATCH",
        "XDG_CACHE_HOME",
    ):
        monkeypatch.delenv(name, raising=False)

    fake_home = tmp_path / "home_dir"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    # 5. Fallback ~/.cache/tess_bright (create=False does not mkdir)
    p_home = resolve_cache_dir()
    assert p_home == (fake_home / ".cache" / "tess_bright").resolve()
    assert not p_home.exists()

    # 4. XDG_CACHE_HOME > ~/.cache
    xdg_dir = tmp_path / "xdg_cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_dir))
    assert resolve_cache_dir() == (xdg_dir / "tess_bright").resolve()

    # 3. SCRATCH > XDG_CACHE_HOME
    scratch_dir = tmp_path / "scratch_space"
    monkeypatch.setenv("SCRATCH", str(scratch_dir))
    assert (
        resolve_cache_dir()
        == (scratch_dir / ".cache" / "tess_bright").resolve()
    )

    # 2. TESS_BRIGHT_CACHE_DIR > SCRATCH
    shared_dir = tmp_path / "shared_tess_bright"
    monkeypatch.setenv("TESS_BRIGHT_CACHE_DIR", str(shared_dir))
    assert resolve_cache_dir() == shared_dir.resolve()

    # 1. TESSBKGD_CACHE_DIR > TESS_BRIGHT_CACHE_DIR (package-specific wins!)
    pkg_dir = tmp_path / "pkg_tessbkgd_cache"
    monkeypatch.setenv("TESSBKGD_CACHE_DIR", str(pkg_dir))
    resolved_pkg = resolve_cache_dir(create=True)
    assert resolved_pkg == pkg_dir.resolve()
    assert resolved_pkg.is_dir()

    # Explicit cache_dir overrides all environment variables
    explicit_dir = tmp_path / "explicit_override"
    resolved_exp = resolve_cache_dir(explicit_dir, create=True)
    assert resolved_exp == explicit_dir.resolve()
    assert resolved_exp.is_dir()

    # /fred policy: ValueError when strict_hpc=True, UserWarning when strict_hpc=False
    with pytest.raises(ValueError, match="/fred/"):
        resolve_cache_dir("/fred/oz440/bpo/cache", strict_hpc=True)
    with pytest.warns(UserWarning, match="/fred/"):
        p_fred = resolve_cache_dir("/fred/oz440/bpo/cache", strict_hpc=False)
    assert p_fred.parts[:2] == ("/", "fred")

    # Symlink into /fred caught via resolved.parts[:2] == ("/", "fred")
    link_fred = tmp_path / "symlink_to_fred"
    link_fred.symlink_to("/fred/oz440/bpo/cache")
    monkeypatch.setenv("TESSBKGD_CACHE_DIR", str(link_fred))
    with pytest.raises(ValueError, match="/fred/"):
        resolve_cache_dir(strict_hpc=True)
    with pytest.warns(UserWarning, match="/fred/"):
        resolve_cache_dir(strict_hpc=False)

    # Contract readers exported in __all__: read_smear_err & read_bkg_covariance
    for sym in ("BkgCovariance", "read_smear_err", "read_bkg_covariance"):
        assert sym in tessbkgd.__all__

    rng = np.random.default_rng(314)
    n_raw, ny, nx = 16, 8, 9
    u_arr = 60.0 + rng.normal(0.0, 0.5, (n_raw, ny, nx))
    u_arr[4, :, :] = np.nan  # all-NaN cadence -> all-NaN BKG_CHOL row
    qual = np.zeros(n_raw, dtype=np.int32)
    qual[2] = 128
    qual[11] = 128
    se_in = np.linspace(0.1, 0.9, n_raw * nx, dtype=np.float32).reshape(
        n_raw, nx
    )

    tpf_plain = _build_mundey_tpf(
        u_arr, flux_err=np.full_like(u_arr, 0.6), smear_err=None, quality=qual
    )
    assert read_smear_err(tpf_plain) is None
    assert read_bkg_covariance(tpf_plain) is None

    tpf_smear = _build_mundey_tpf(
        u_arr,
        flux_err=np.full_like(u_arr, 0.6),
        smear_err=se_in,
        quality=qual,
    )
    se_vis = read_smear_err(tpf_smear)
    se_all = read_smear_err(tpf_smear, apply_quality_mask=False)
    assert se_vis is not None and se_all is not None
    assert se_vis.dtype == np.float64
    assert se_vis.shape == (n_raw - 2, nx)
    assert se_all.shape == (n_raw, nx)
    np.testing.assert_allclose(se_vis, se_in[qual == 0], rtol=1e-6)
    np.testing.assert_allclose(
        read_smear_err(tpf_smear[3:9]), se_vis[3:9], rtol=1e-6
    )

    sub = subtract_background(tpf_smear, order=1)
    fits_file = tmp_path / "sub_contract.fits"
    sub.to_fits(str(fits_file), overwrite=True)

    cov_vis = read_bkg_covariance(sub)
    cov_raw = read_bkg_covariance(fits_file)
    assert isinstance(cov_vis, BkgCovariance)
    assert cov_raw is not None
    chol_vis, basis_vis = cov_vis
    assert chol_vis.shape == (n_raw - 2, 3, 3)
    assert cov_raw.chol.shape == (n_raw, 3, 3)
    assert basis_vis.shape == (3, ny, nx)
    assert np.all(np.isnan(cov_raw.chol[4]))
    np.testing.assert_allclose(cov_vis.chol, cov_raw.chol[qual == 0])

    # Incomplete covariance (only BKG_BASIS without BKG_CHOL) raises ValueError
    broken_hdul = fits.HDUList([h.copy() for h in tpf_plain.hdu])
    broken_hdul.append(
        fits.ImageHDU(
            data=np.ones((3, ny, nx), dtype=np.float32), name="BKG_BASIS"
        )
    )
    with pytest.raises(ValueError, match="Incomplete HALOCVER=1"):
        read_bkg_covariance(broken_hdul)
