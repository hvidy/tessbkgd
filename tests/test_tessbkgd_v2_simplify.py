"""Regression tests for v2 simplification, anisotropic orders, HC2, CLI & plotting."""

from __future__ import annotations

import warnings
from pathlib import Path

import astropy.units as u
import lightkurve as lk
import matplotlib
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

matplotlib.use("Agg")

import tessbkgd
from tessbkgd import (
    BackgroundConfig,
    BackgroundDiagnostics,
    BackgroundEstimate,
    PixelQuality,
    estimate_background,
    plot_diagnostics,
    subtract_background,
)
from tessbkgd.cli import main as cli_main
from tessbkgd.masking import (
    build_background_mask,
    n_basis_for_order,
    normalized_grid,
    parse_order,
    raw_chebyshev_2d,
)
from tessbkgd.solver import (
    _bucket_pixel_dim,
    equilibrate_chebyshev_basis,
    solve_chebyshev_background,
)


def _make_tpf(
    flux: np.ndarray,
    *,
    flux_err: np.ndarray | None = None,
    pixel_quality: np.ndarray | None = None,
    smear_err: np.ndarray | None = None,
    wcs: WCS | None = None,
    ra_obj: float | None = None,
    dec_obj: float | None = None,
) -> lk.TessTargetPixelFile:
    n_cad, ny, nx = flux.shape
    cols = [
        fits.Column(
            name="TIME",
            format="D",
            unit="BJD - 2457000, days",
            array=np.linspace(1500.0, 1505.0, n_cad),
        ),
        fits.Column(
            name="CADENCENO", format="J", array=np.arange(n_cad, dtype=np.int32)
        ),
        fits.Column(
            name="FLUX",
            format=f"{ny * nx}E",
            dim=f"({nx}, {ny})",
            unit="e-/s",
            array=flux.astype(np.float32),
        ),
        fits.Column(name="QUALITY", format="J", array=np.zeros(n_cad, dtype=np.int32)),
    ]
    if flux_err is not None:
        cols.append(
            fits.Column(
                name="FLUX_ERR",
                format=f"{ny * nx}E",
                dim=f"({nx}, {ny})",
                unit="e-/s",
                array=flux_err.astype(np.float32),
            )
        )
    if smear_err is not None:
        cols.append(
            fits.Column(
                name="SMEAR_ERR",
                format=f"{nx}E",
                unit="e-/s",
                array=smear_err.astype(np.float32),
            )
        )
    hdr0 = fits.Header({"TELESCOP": "TESS", "BKGSUB": False})
    if ra_obj is not None and dec_obj is not None:
        hdr0["RA_OBJ"] = ra_obj
        hdr0["DEC_OBJ"] = dec_obj
    tbl_hdr = fits.Header({"BKGSUB": False, "BJDREFI": 2457000, "BJDREFF": 0.0})
    ap_hdr = wcs.to_header() if wcs is not None else fits.Header()
    hdus: list[fits.hdu.base.ExtensionHDU | fits.PrimaryHDU] = [
        fits.PrimaryHDU(header=hdr0),
        fits.BinTableHDU.from_columns(cols, header=tbl_hdr, name="PIXELS"),
        fits.ImageHDU(
            np.ones((ny, nx), dtype=np.int32), header=ap_hdr, name="APERTURE"
        ),
    ]
    if pixel_quality is not None:
        hdus.append(
            fits.ImageHDU(
                np.asarray(pixel_quality, dtype=np.uint16), name="PIXEL_QUALITY"
            )
        )
    return lk.TessTargetPixelFile(fits.HDUList(hdus))


def test_public_all_and_terminology_hygiene() -> None:
    expected_all = [
        "BackgroundConfig",
        "BackgroundDiagnostics",
        "BackgroundEstimate",
        "PixelQuality",
        "__version__",
        "bkgd_tpf",
        "estimate_background",
        "plot_diagnostics",
        "subtract_background",
    ]
    assert tessbkgd.__all__ == expected_all
    repo_root = Path(__file__).resolve().parents[1]
    forbidden = "aur" + "eole"
    for folder in ("src", "tests"):
        for path in (repo_root / folder).rglob("*.py"):
            assert forbidden not in path.read_text(encoding="utf-8").lower(), (
                f"Forbidden word in {path}"
            )


def test_anisotropic_order_defaults_and_tall_bleed_strip() -> None:
    assert parse_order(None, 1287, 11) == (2, 1)
    assert parse_order(None, 135, 25) == (2, 1)
    assert parse_order(None, 40, 40) == (2, 2)
    assert parse_order(None, 12, 12) == (1, 1)
    assert n_basis_for_order((2, 1)) == 5
    assert raw_chebyshev_2d(10, 8, (2, 1)).shape == (5, 10, 8)

    # Tall 120x11 strip with 6-col central SAT_BLEED (cols 3..8, 55% > 50% in x)
    # and strong quadratic vertical sky curvature: default order=(2, 1) preserves
    # oy=2 without downgrading y!
    rng = np.random.default_rng(11)
    n_cad, ny, nx = 24, 120, 11
    eta, xi = normalized_grid(ny, nx)
    eg, xg = np.meshgrid(eta, xi, indexing="ij")
    sky_true = (120.0 + 35.0 * (2.0 * eg**2 - 1.0) + 8.0 * eg + 4.0 * xg)[
        None, :, :
    ] * (1.0 + 0.1 * np.sin(np.linspace(0.0, 2.0, n_cad))[:, None, None])
    u_cube = sky_true + rng.normal(0.0, 0.5, sky_true.shape)
    pq = np.zeros((ny, nx), dtype=np.uint16)
    pq[:, 3:9] = np.uint16(PixelQuality.SAT_BLEED)

    est, diag = estimate_background(
        u_cube,
        flux_err=np.full_like(u_cube, 0.5),
        pixel_quality=pq,
        exclusion_radius=0.0,
        return_diagnostics=True,
    )
    assert diag.requested_order == (2, 1)
    assert diag.effective_order == (2, 1)
    # Top and bottom rows survive (degree-2 vertical P_0 does not clip curved ends!)
    assert diag.bkg_mask[0, 0] and diag.bkg_mask[-1, -1]
    np.testing.assert_allclose(est.bkg, sky_true, rtol=1e-2, atol=0.6)


def test_exclusion_radius_bleed_buffer_fallback_and_target_corr_warning() -> None:
    rng = np.random.default_rng(22)
    n_cad, ny, nx = 30, 50, 50
    rr, cc = np.mgrid[0:ny, 0:nx]
    r2 = (rr - 25.0) ** 2 + (cc - 25.0) ** 2
    sky = 80.0 + 0.3 * cc - 0.2 * rr
    # Saturated central star (>= 7.5e4 e-/s) + vertical SAT_BLEED column at c=25
    star = 1.5e5 * np.exp(-0.5 * r2 / (2.0**2))
    u = sky[None, :, :] + star[None, :, :] + rng.normal(0.0, 0.4, (n_cad, ny, nx))
    pq = np.zeros((ny, nx), dtype=np.uint16)
    pq[:, 25] = np.uint16(PixelQuality.SAT_BLEED)

    m_def = build_background_mask(u, pixel_quality=pq, exclusion_radius=15.0)
    # Within 15 px of (25, 25) and +/-2 cols around col 25 (cols 23..27) is excluded
    assert np.all(m_def.exclusion_mask[r2 <= 14.5**2])
    assert np.all(m_def.exclusion_mask[:, 23:28])
    assert not np.any(m_def.mask[m_def.exclusion_mask])

    m_zero = build_background_mask(u, pixel_quality=pq, exclusion_radius=0.0)
    assert not np.any(m_zero.exclusion_mask)

    # Algol-like 57x11 variable eclipsing saturated star whose broad PSF wings reach
    # the outer flanks -> |target_flux_corr| > 0.5 MUST trigger a loud UserWarning!
    n_t, ny_a, nx_a = 40, 57, 11
    ra, ca = np.mgrid[0:ny_a, 0:nx_a]
    dist_a = np.hypot(ra - 28.0, ca - 5.0)
    eclipse = np.ones(n_t, dtype=np.float64)
    eclipse[15:25] = 0.4  # deep primary eclipse
    psf_wing = 2.0e5 / (1.0 + (dist_a / 2.5) ** 2) ** 1.3
    u_algol = (
        50.0
        + eclipse[:, None, None] * psf_wing[None, :, :]
        + rng.normal(0.0, 0.3, (n_t, ny_a, nx_a))
    )
    pq_algol = np.zeros((ny_a, nx_a), dtype=np.uint16)
    pq_algol[:, 5] = np.uint16(PixelQuality.SAT_BLEED)

    with pytest.warns(
        UserWarning, match="correlates strongly with target aperture flux"
    ):
        _, diag_algol = estimate_background(
            u_algol,
            flux_err=np.full_like(u_algol, 0.3),
            pixel_quality=pq_algol,
            exclusion_radius=10.0,
            return_diagnostics=True,
        )
    assert np.isfinite(diag_algol.target_flux_corr)
    assert abs(diag_algol.target_flux_corr) > 0.5

    # Control: constant star + time-varying Earthshine must NOT trigger false warning!
    earth = 40.0 + 30.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, n_t))
    u_ctrl = (
        earth[:, None, None]
        + 5000.0 * np.exp(-0.5 * dist_a[None, :, :] ** 2 / 1.5**2)
        + rng.normal(0.0, 0.4, (n_t, ny_a, nx_a))
    )
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        estimate_background(u_ctrl, flux_err=np.full_like(u_ctrl, 0.4), order=1)
    assert not any("correlates strongly" in str(w.message) for w in rec)


def test_median_seed_corner_breakdown_hc2_asymmetry_and_smear_f_ind_bound() -> None:
    # 1. Corner outlier cluster (18.8% pixels +1e6 e-/s in corner): median-seeded IRLS
    #    must recover true plane to < 1 e-/s (old unweighted Step-0 blew up to 1.26e6!).
    rng = np.random.default_rng(33)
    ny, nx = 8, 8
    mask_all = np.ones((ny, nx), dtype=bool)
    eta, xi = normalized_grid(ny, nx)
    eg, xg = np.meshgrid(eta, xi, indexing="ij")
    plane = 100.0 + 10.0 * xg - 6.0 * eg
    u_corner = plane[None, :, :] + rng.normal(0.0, 0.5, (10, ny, nx))
    u_corner[:, :3, :4] += 1.0e6  # 12/64 = 18.75% extreme corner cluster!
    est_c = estimate_background(
        u_corner, order=1, mask=mask_all, flux_err=np.full_like(u_corner, 0.5)
    )
    assert float(np.max(np.abs(est_c.bkg - plane[None, :, :]))) < 1.0

    # 2. Spatially asymmetric heteroscedastic noise (sigma=8 left, sigma=2 right) with
    #    uncalibrated uniform weights: exact HC2 matrix sandwich must report larger
    #    uncertainty on the noisy left side (ratio > 0.70 vs old scalar SCR's 0.48!).
    ny_h, nx_h = 10, 10
    mask_h = np.ones((ny_h, nx_h), dtype=bool)
    phi_full, phi_bg, _ = equilibrate_chebyshev_basis(ny_h, nx_h, 2, mask_h)
    sig_2d = np.where(np.arange(nx_h)[None, :] < 5, 8.0, 2.0)
    u_het = rng.normal(0.0, sig_2d[None, :, :], (400, ny_h, nx_h))
    sol = solve_chebyshev_background(u_het.reshape(400, -1), phi_bg)
    b_left = np.einsum("tp,p->t", sol.coeffs, phi_full[:, 5, 0])
    e_left = np.sqrt(
        np.sum(np.einsum("tpq,p->tq", sol.bkg_chol, phi_full[:, 5, 0]) ** 2, axis=1)
    )
    b_right = np.einsum("tp,p->t", sol.coeffs, phi_full[:, 5, 9])
    e_right = np.sqrt(
        np.sum(np.einsum("tpq,p->tq", sol.bkg_chol, phi_full[:, 5, 9]) ** 2, axis=1)
    )
    assert float(np.median(e_left)) / float(np.std(b_left)) > 0.75
    assert float(np.median(e_right)) / float(np.std(b_right)) > 0.75
    assert float(np.median(e_left)) > 1.8 * float(np.median(e_right))

    # 3. Single-row smear fallback at f_ind = 0.01 (repro [D]): NO 1/f_ind explosion!
    ny_s, nx_s = 12, 16
    mask_1row = np.zeros((ny_s, nx_s), dtype=bool)
    mask_1row[0, :] = True
    sig_ind, sig_sm = 1.0, 10.0  # f_ind = 1 / 101 ~ 0.0099
    sig_tot = np.hypot(sig_ind, sig_sm)
    u_sm = (
        100.0
        + rng.normal(0.0, sig_sm, (120, 1, nx_s))
        + rng.normal(0.0, sig_ind, (120, ny_s, nx_s))
    )
    est_s, diag_s = estimate_background(
        u_sm,
        order=(0, 1),
        mask=mask_1row,
        flux_err=np.full_like(u_sm, sig_tot),
        smear_err=np.full((120, nx_s), sig_sm),
        return_diagnostics=True,
    )
    # Formal independent uncertainty at mean is sigma_ind / sqrt(nx_s) = 0.25 e-/s;
    # old 1/f_ind blew up to > 1.5 e-/s (6-12x!), whereas bounded HC2 stays < 0.6!
    phi_m = np.mean(diag_s.bkg_basis.reshape(diag_s.bkg_basis.shape[0], -1), axis=1)
    se_ind_mean = float(
        np.median(
            np.sqrt(np.sum(np.einsum("tpq,p->tq", diag_s.bkg_chol, phi_m) ** 2, axis=1))
        )
    )
    assert se_ind_mean < 0.6
    assert np.all(np.isfinite(est_s.bkg_err))


def test_zero_mean_negative_sky_and_no_nan_flux_err_and_bucketing() -> None:
    rng = np.random.default_rng(44)
    n_cad, ny, nx = 40, 14, 14
    # Zero-mean Gaussian sky with dead constant 0.0 column and -0.2 row
    u_zero = rng.normal(0.0, 1.0, (n_cad, ny, nx))
    u_zero[:, :, 0] = 0.0
    u_zero[:, -1, :] = -0.2
    est_z = estimate_background(u_zero, order=1)
    assert not np.any(est_z.mask[:, 0])
    assert not np.any(est_z.mask[-1, :])
    assert int(np.count_nonzero(est_z.mask[:-1, 1:])) >= int(0.85 * (ny - 1) * (nx - 1))
    assert abs(float(np.mean(est_z.bkg))) < 0.15

    # Negative sky (B = -0.5) also survives unbiased
    u_neg = -0.5 + rng.normal(0.0, 0.4, (n_cad, ny, nx))
    est_n = estimate_background(u_neg, order=0)
    assert abs(float(np.mean(est_n.bkg)) - (-0.5)) < 0.05

    # TPF without FLUX_ERR: subtract_background never fabricates an all-NaN FLUX_ERR!
    tpf_noerr = _make_tpf(50.0 + rng.normal(0.0, 0.5, (20, 11, 11)), flux_err=None)
    clean_noerr = subtract_background(tpf_noerr, order=1)
    assert "FLUX_ERR" not in {n.upper() for n in clean_noerr.hdu[1].data.names}
    assert np.all(np.isfinite(clean_noerr.flux.value))
    assert np.all(np.isfinite(clean_noerr.flux_bkg.value))

    # Power-of-two pixel bucketing above 4096
    assert _bucket_pixel_dim(100) == 256
    assert _bucket_pixel_dim(4096) == 4096
    assert _bucket_pixel_dim(4100) == 8192
    assert _bucket_pixel_dim(8192) == 8192
    assert _bucket_pixel_dim(8193) == 16384


def test_catalog_table_config_raw_array_plotting_and_cli(tmp_path: Path) -> None:
    rng = np.random.default_rng(55)
    n_cad, ny, nx = 20, 16, 16
    w = WCS(naxis=2)
    w.wcs.crpix = [8.5, 8.5]
    w.wcs.cdelt = [-21.0 / 3600.0, 21.0 / 3600.0]
    w.wcs.crval = [120.0, -30.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    ra1, dec1 = w.pixel_to_world_values(4.0, 4.0)
    cat = Table({"ra": [float(ra1)], "dec": [float(dec1)], "mag": [13.5]})

    u_cube = 90.0 + rng.normal(0.0, 0.5, (n_cad, ny, nx))
    tpf = _make_tpf(u_cube, flux_err=np.full_like(u_cube, 0.5), wcs=w)

    cfg = BackgroundConfig(order=1, exclusion_radius=12.0)
    est_cat, diag_cat = estimate_background(
        tpf, config=cfg, catalog_table=cat, return_diagnostics=True
    )
    _, diag_gaia = estimate_background(
        tpf, config=cfg, gaia_sources=cat, return_diagnostics=True
    )
    assert isinstance(est_cat, BackgroundEstimate)
    assert isinstance(diag_cat, BackgroundDiagnostics)
    assert np.array_equal(diag_cat.source_mask, diag_gaia.source_mask)
    assert diag_cat.source_mask[4, 4]

    # Raw 3D ndarray and Quantity inputs
    est_raw = estimate_background(u_cube, order=(1, 1))
    assert isinstance(est_raw.bkg, np.ndarray) and not isinstance(
        est_raw.bkg, u.Quantity
    )
    est_qty = estimate_background(u_cube * u.Unit("electron/s"), order=(1, 1))
    assert isinstance(est_qty.bkg, u.Quantity)

    # plot_diagnostics (and diag.plot) on TPF and raw array
    import matplotlib.pyplot as plt

    png_path = tmp_path / "diag.png"
    fig, axes = diag_cat.plot(tpf, save_path=str(png_path))
    assert axes.shape == (2, 2)
    assert png_path.is_file() and png_path.stat().st_size > 1000
    plt.close(fig)

    fig2, axes2 = plot_diagnostics(u_cube, est_raw)
    assert axes2.shape == (2, 2)
    plt.close(fig2)

    # CLI end-to-end
    in_fits = tmp_path / "in_tpf.fits"
    out_fits = tmp_path / "out_tpf.fits"
    cli_png = tmp_path / "cli_diag.png"
    tpf.to_fits(str(in_fits), overwrite=True)
    rc = cli_main(
        [
            str(in_fits),
            "-o",
            str(out_fits),
            "--order",
            "1,1",
            "--exclusion-radius",
            "10",
            "--plot",
            str(cli_png),
            "--overwrite",
        ]
    )
    assert rc == 0
    assert out_fits.is_file()
    assert cli_png.is_file()
    out_tpf = lk.read(str(out_fits))
    assert out_tpf.hdu[0].header["BKGSUB"] is True
