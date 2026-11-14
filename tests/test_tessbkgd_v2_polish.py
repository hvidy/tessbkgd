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


def _coherent_halo_flux_err(
    weightmap_2d: np.ndarray,
    *,
    flux_err: np.ndarray,
    smear_err: np.ndarray,
    bkg_cov: BkgCovariance,
    bkg_order: int,
) -> np.ndarray:
    """Evaluate downstream HALOCVER=1 coherent halo flux error (halophot Eq. 6.4.1)."""
    w_clean = np.where(weightmap_2d > 0.0, weightmap_2d, 0.0)
    rows_h, cols_h = np.where(w_clean > 0.0)
    w_star = w_clean[rows_h, cols_h]
    _, _, nx = flux_err.shape
    phi_halo = bkg_cov.basis[:, rows_h, cols_h]

    fe_sq = np.square(flux_err[:, rows_h, cols_h])
    smear_pix_sq = np.square(smear_err[:, cols_h])
    lt_phi = np.einsum("tqp,qm->tpm", bkg_cov.chol, phi_halo)
    bkg_pix_sq = np.sum(np.square(lt_phi), axis=1)
    sigma_pix_sq = np.maximum(fe_sq - smear_pix_sq - bkg_pix_sq, 0.0)
    var_total = sigma_pix_sq @ np.square(w_star)

    col_g = np.sum(w_clean, axis=0)
    if bkg_order >= nx - 1:
        g_perp = np.zeros_like(col_g)
    else:
        xi = np.linspace(-1.0, 1.0, nx, dtype=np.float64)
        vander = np.polynomial.chebyshev.chebvander(xi, bkg_order)
        q_mat, _ = np.linalg.qr(vander, mode="reduced")
        g_perp = col_g - q_mat @ (q_mat.T @ col_g)
    var_total += np.square(smear_err) @ np.square(g_perp)

    phi_w = phi_halo @ w_star
    lt_phi_w = np.einsum("tqp,q->tp", bkg_cov.chol, phi_w)
    var_total += np.sum(np.square(lt_phi_w), axis=1)
    return np.sqrt(np.maximum(var_total, 0.0))


def test_tbkg01_smear_err_monte_carlo_and_halocver_invariants() -> None:
    """Verify [TBKG-01] pulls, conditional BKG_CHOL, halo sum, & idempotency."""
    rng = np.random.default_rng(20261114)
    n_mc = 250
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
    sub, diag = subtract_background(
        cal, order=1, mask=mask, return_diagnostics=True
    )
    est = estimate_background(cal, order=1, mask=mask)

    bkg_hat = sub.flux_bkg.to_value("electron/s")
    bkg_err = sub.flux_bkg_err.to_value("electron/s")

    cov = read_bkg_covariance(sub)
    se_vis = read_smear_err(sub)
    assert cov is not None and se_vis is not None
    np.testing.assert_allclose(cov.chol, diag.bkg_chol, rtol=1e-6, atol=1e-6)
    lt_phi = np.einsum("tqp,qyx->tpyx", cov.chol, cov.basis)
    bkg_indep_sq = np.sum(lt_phi**2, axis=1)
    bkg_err_indep_only = np.sqrt(bkg_indep_sq)

    # (i) Standalone FLUX_BKG_ERR & estimate_background bkg_err include coherent
    #     column smear variance and calibrate the Monte Carlo pull distribution:
    assert np.all(bkg_err > bkg_err_indep_only)
    np.testing.assert_allclose(
        est.bkg_err.to_value("electron/s"), bkg_err, rtol=1e-6, atol=1e-6
    )
    pulls = (bkg_hat - sky_true[None, :, :]) / bkg_err
    pull_std = float(np.std(pulls, ddof=1))
    assert 0.93 <= pull_std <= 1.07, f"Uncalibrated pull_std={pull_std:.4f}"
    pulls_indep_only = (bkg_hat - sky_true[None, :, :]) / bkg_err_indep_only
    assert float(np.std(pulls_indep_only, ddof=1)) > 1.50

    # (ii) Non-tautological BKG_CHOL verification:
    #      L_t @ L_t.T must match analytical WLS H_0 @ diag(sigma_indep^2) @ H_0.T
    #      (up to ~5% Tukey efficiency) AND conditional MC Cov(beta_hat | eps_smear)
    #      in the smear-dominated regime (sigma_smear=1.5, sigma_indep=0.4):
    s_ind_dom = 0.4
    s_sm_dom = 1.5
    se_dom = np.full((n_mc, nx), s_sm_dom, dtype=np.float64)
    fe_dom = np.full((n_mc, ny, nx), np.hypot(s_ind_dom, s_sm_dom))
    eps_pix_dom = rng.normal(0.0, s_ind_dom, size=(n_mc, ny, nx))
    eps_sm_dom = rng.normal(0.0, s_sm_dom, size=(n_mc, nx))
    u_dom = (
        sky_true[None, :, :]
        + star[None, :, :]
        + eps_pix_dom
        + eps_sm_dom[:, None, :]
    )
    cal_dom = _build_mundey_tpf(
        u_dom, flux_err=fe_dom, smear_err=se_dom, bkgsub=False
    )
    sub_dom, diag_dom = subtract_background(
        cal_dom, order=1, mask=mask, return_diagnostics=True
    )
    cov_dom = read_bkg_covariance(sub_dom)
    assert cov_dom is not None

    phi_bg_dom = diag_dom.bkg_basis[:, mask].T  # orthonormal on mask: Phi.T @ Phi = I_P
    # Uniform weights w0 across mask => H_0 = Phi_bg.T, so analytical WLS
    # conditional covariance is s_ind_dom^2 * I_P = 0.16 * I_3 (NOT 2.41 * I_3!):
    cov_pred_mean = np.mean(
        np.matmul(cov_dom.chol, np.swapaxes(cov_dom.chol, 1, 2)), axis=0
    )
    cov_wls_exact = (s_ind_dom**2) * np.eye(phi_bg_dom.shape[1])
    np.testing.assert_allclose(
        cov_pred_mean,
        cov_wls_exact * 1.053,
        rtol=0.06,
        atol=0.01,
    )

    # Conditional MC holding eps_sm_dom[0] fixed and varying only eps_pix:
    u_cond = (
        sky_true[None, :, :]
        + star[None, :, :]
        + rng.normal(0.0, s_ind_dom, size=(n_mc, ny, nx))
        + eps_sm_dom[0:1, None, :]
    )
    _, diag_cond = subtract_background(
        _build_mundey_tpf(
            u_cond, flux_err=fe_dom, smear_err=se_dom, bkgsub=False
        ),
        order=1,
        mask=mask,
        return_diagnostics=True,
    )
    cov_emp_cond = np.cov(diag_cond.coeffs.T, ddof=1)
    np.testing.assert_allclose(
        np.diag(cov_pred_mean), np.diag(cov_emp_cond), rtol=0.22
    )

    # Smear-dominated standalone FLUX_BKG_ERR pull calibration:
    pulls_dom = (
        sub_dom.flux_bkg.to_value("electron/s") - sky_true[None, :, :]
    ) / sub_dom.flux_bkg_err.to_value("electron/s")
    assert 0.93 <= float(np.std(pulls_dom, ddof=1)) <= 1.07

    # (iii) Downstream halophot coherent halo-aperture sum calibration AND
    #       exact zero-error cancellation in the sigma_indep -> 0 limit:
    mask_sym = np.ones((ny, nx), dtype=bool)
    mask_sym[4:8, :] = False  # middle 4 rows across all columns = halo
    w_sym_unif = (~mask_sym).astype(np.float64) / np.count_nonzero(~mask_sym)
    w_sym_vary = (~mask_sym).astype(np.float64) * (1.0 + 2.0 * (xi_g**2))
    w_sym_vary /= np.sum(w_sym_vary)

    # (iii.a) Multi-column halo with g_perp != 0 under full i.i.d. column smear
    #         in both moderate (0.9, 0.7) and smear-dominated (0.4, 1.5) regimes:
    for s_ind_h, s_sm_h in ((0.9, 0.7), (0.4, 1.5)):
        u_h = (
            sky_true[None, :, :]
            + rng.normal(0.0, s_ind_h, size=(n_mc, ny, nx))
            + rng.normal(0.0, s_sm_h, size=(n_mc, nx))[:, None, :]
        )
        sub_h = subtract_background(
            _build_mundey_tpf(
                u_h,
                flux_err=np.full((n_mc, ny, nx), np.hypot(s_ind_h, s_sm_h)),
                smear_err=np.full((n_mc, nx), s_sm_h),
                bkgsub=False,
            ),
            order=1,
            mask=mask_sym,
        )
        cov_h = read_bkg_covariance(sub_h)
        se_h = read_smear_err(sub_h)
        assert cov_h is not None and se_h is not None
        halo_err = _coherent_halo_flux_err(
            w_sym_vary,
            flux_err=sub_h.flux_err.to_value("electron/s"),
            smear_err=se_h,
            bkg_cov=cov_h,
            bkg_order=1,
        )
        halo_lc = np.sum(
            sub_h.flux.to_value("electron/s") * w_sym_vary[None, :, :],
            axis=(1, 2),
        )
        halo_pull_std = float(np.std(halo_lc / halo_err, ddof=1))
        assert 0.92 <= halo_pull_std <= 1.08, (
            f"Uncalibrated halo_pull_std={halo_pull_std:.4f}"
        )

    # (iii.b) Symmetric all-column halo (g_perp == 0) in smear-dominated regime
    #         (0.4, 1.5) with degree-<=1 smear: isolates sigma_indep + BKG_CHOL
    #         directly (smear-contaminated BKG_CHOL would deflate pull_std ~0.38):
    sm_deg1 = rng.normal(0.0, 1.5 / np.sqrt(2.0), size=(n_mc, 1)) + (
        rng.normal(0.0, 1.5 / np.sqrt(2.0), size=(n_mc, 1)) * xi[None, :]
    )
    u_sym_dom = (
        sky_true[None, :, :]
        + rng.normal(0.0, 0.4, size=(n_mc, ny, nx))
        + sm_deg1[:, None, :]
    )
    sub_sym_dom = subtract_background(
        _build_mundey_tpf(
            u_sym_dom,
            flux_err=np.full((n_mc, ny, nx), np.hypot(0.4, 1.5)),
            smear_err=np.full((n_mc, nx), 1.5),
            bkgsub=False,
        ),
        order=1,
        mask=mask_sym,
    )
    cov_sym_dom = read_bkg_covariance(sub_sym_dom)
    se_sym_dom = read_smear_err(sub_sym_dom)
    assert cov_sym_dom is not None and se_sym_dom is not None
    err_sym_dom = _coherent_halo_flux_err(
        w_sym_unif,
        flux_err=sub_sym_dom.flux_err.to_value("electron/s"),
        smear_err=se_sym_dom,
        bkg_cov=cov_sym_dom,
        bkg_order=1,
    )
    lc_sym_dom = np.sum(
        sub_sym_dom.flux.to_value("electron/s") * w_sym_unif[None, :, :],
        axis=(1, 2),
    )
    pull_sym_dom = float(np.std(lc_sym_dom / err_sym_dom, ddof=1))
    assert 0.92 <= pull_sym_dom <= 1.08

    # (iii.c) Noiseless-pixel limit (sigma_indep = 0, sigma_smear = 1.5) on
    #         symmetric all-column halo: L_t == 0, predicted error == 0, and
    #         empirical degree-<=1 smear residual == 0:
    se_zero = np.full((40, nx), 1.5, dtype=np.float64)
    fe_zero = np.full((40, ny, nx), 1.5, dtype=np.float64)
    u_zero = sky_true[None, :, :] + sm_deg1[:40, None, :]
    sub_zero = subtract_background(
        _build_mundey_tpf(
            u_zero, flux_err=fe_zero, smear_err=se_zero, bkgsub=False
        ),
        order=1,
        mask=mask_sym,
    )
    cov_zero = read_bkg_covariance(sub_zero)
    se_zero_vis = read_smear_err(sub_zero)
    assert cov_zero is not None and se_zero_vis is not None
    np.testing.assert_array_equal(cov_zero.chol, 0.0)
    halo_err_zero = _coherent_halo_flux_err(
        w_sym_unif,
        flux_err=sub_zero.flux_err.to_value("electron/s"),
        smear_err=se_zero_vis,
        bkg_cov=cov_zero,
        bkg_order=1,
    )
    lc_zero = np.sum(
        sub_zero.flux.to_value("electron/s") * w_sym_unif[None, :, :],
        axis=(1, 2),
    )
    np.testing.assert_allclose(halo_err_zero, 0.0, atol=1e-7)
    np.testing.assert_allclose(lc_zero, 0.0, atol=2e-5)

    # (iv) Idempotency of subtract_background(subtract_background(cal)) when
    #      SMEAR_ERR > 0, across moderate smear, sigma_indep=0, and quality flags:
    for first_sub, m_arg in ((sub, mask), (sub_zero, mask_sym)):
        second_sub = subtract_background(first_sub, order=1, mask=m_arg)
        for col in ("FLUX", "FLUX_BKG", "FLUX_ERR", "FLUX_BKG_ERR", "BKG_CHOL"):
            np.testing.assert_allclose(
                second_sub.hdu[1].data[col],
                first_sub.hdu[1].data[col],
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


def test_tbkg01_narrow_stamps_and_higher_orders_calibration() -> None:
    """Verify FLUX_BKG_ERR calibration on narrow d+1=nx stamps and order=3 edges."""
    rng = np.random.default_rng(314159)
    n_mc = 300

    # (a) Narrow stamps (16x2 order=1 and 15x3 order=2) where d + 1 == nx absorbs
    #     100% of column smear into B_hat (corner smear variance == 1.00):
    for ny, nx, order in ((16, 2, 1), (15, 3, 2)):
        mask = np.ones((ny, nx), dtype=bool)
        eta, xi = normalized_grid(ny, nx)
        eta_g, xi_g = np.meshgrid(eta, xi, indexing="ij")
        sky = 90.0 + 6.0 * xi_g - 4.0 * eta_g
        s_ind, s_sm = 0.1, 1.0
        u_narrow = (
            sky[None, :, :]
            + rng.normal(0.0, s_ind, size=(n_mc, ny, nx))
            + rng.normal(0.0, s_sm, size=(n_mc, nx))[:, None, :]
        )
        cal_n = _build_mundey_tpf(
            u_narrow,
            flux_err=np.full((n_mc, ny, nx), np.hypot(s_ind, s_sm)),
            smear_err=np.full((n_mc, nx), s_sm),
            bkgsub=False,
        )
        sub_n = subtract_background(cal_n, order=order, mask=mask)
        fbe_n = sub_n.flux_bkg_err.to_value("electron/s")
        bkg_n = sub_n.flux_bkg.to_value("electron/s")
        # Corner variance must be >= s_sm^2 = 1.00 (never deflated to 0.70 or 0.90):
        corner_var = float(np.mean(fbe_n[:, 0, 0] ** 2))
        assert 1.00 <= corner_var <= 1.03, f"Narrow {ny}x{nx} var={corner_var:.4f}"
        pull_std_n = float(np.std((bkg_n - sky[None, :, :]) / fbe_n, ddof=1))
        assert 0.93 <= pull_std_n <= 1.07

    # (b) Order=3 on 12x14 where vertical modes change sign down columns (testing
    #     absence of upward clamping bias at vertical edges r = 0, ny - 1):
    ny3, nx3 = 12, 14
    mask3 = np.ones((ny3, nx3), dtype=bool)
    mask3[4:8, 5:9] = False
    eta3, xi3 = normalized_grid(ny3, nx3)
    eta3_g, xi3_g = np.meshgrid(eta3, xi3, indexing="ij")
    sky3 = 110.0 + 7.0 * xi3_g - 5.0 * eta3_g + 2.0 * (xi3_g * eta3_g)
    s_ind3, s_sm3 = 0.4, 1.2
    u3 = (
        sky3[None, :, :]
        + rng.normal(0.0, s_ind3, size=(n_mc, ny3, nx3))
        + rng.normal(0.0, s_sm3, size=(n_mc, nx3))[:, None, :]
    )
    sub3 = subtract_background(
        _build_mundey_tpf(
            u3,
            flux_err=np.full((n_mc, ny3, nx3), np.hypot(s_ind3, s_sm3)),
            smear_err=np.full((n_mc, nx3), s_sm3),
            bkgsub=False,
        ),
        order=3,
        mask=mask3,
    )
    pulls3 = (
        sub3.flux_bkg.to_value("electron/s") - sky3[None, :, :]
    ) / sub3.flux_bkg_err.to_value("electron/s")
    assert 0.93 <= float(np.std(pulls3, ddof=1)) <= 1.07
    edge_pull_std = float(np.std(pulls3[:, [0, -1], :], ddof=1))
    assert 0.93 <= edge_pull_std <= 1.07


def test_tbkg01_smear_err_nan_propagation_and_masked_columns() -> None:
    """Verify cadence-wide NaN SMEAR_ERR propagation on active vs excluded columns."""
    rng = np.random.default_rng(919)
    n_cad, ny, nx = 10, 10, 12
    eta, xi = normalized_grid(ny, nx)
    eta_g, xi_g = np.meshgrid(eta, xi, indexing="ij")
    sky = (80.0 + 5.0 * xi_g - 3.0 * eta_g)[None, :, :] + rng.normal(
        0.0, 0.5, (n_cad, ny, nx)
    )
    flux_err = np.full((n_cad, ny, nx), 0.7, dtype=np.float64)
    smear_err = np.full((n_cad, nx), 0.4, dtype=np.float64)

    # Column 4 is flagged SMEAR_UNRESOLVED -> excluded from bkg_mask -> inactive.
    pq = np.zeros((ny, nx), dtype=np.uint16)
    pq[:, 4] |= np.uint16(PixelQuality.SMEAR_UNRESOLVED)

    # Cadence 2: NaN SMEAR_ERR on active column 0 -> FLUX_BKG_ERR[2] uniformly NaN,
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

    # Polynomial-node regression: on an 8x5 stamp (xi = [-1, -0.5, 0, 0.5, 1])
    # with order=1, active column c=1 has exact leverage 1/5 + (1)(-0.5)/2.5 == 0
    # on column c=4. NaN SMEAR_ERR[1, 1] must still mark ALL 5 columns of
    # FLUX_BKG_ERR[1] uniformly NaN regardless of BLAS rounding.
    u_node = 60.0 + rng.normal(0.0, 0.3, (4, 8, 5))
    se_node = np.full((4, 5), 0.3, dtype=np.float64)
    se_node[1, 1] = np.nan
    sub_node = subtract_background(
        _build_mundey_tpf(
            u_node,
            flux_err=np.full((4, 8, 5), 0.5),
            smear_err=se_node,
            bkgsub=False,
        ),
        order=1,
        mask=np.ones((8, 5), dtype=bool),
    )
    assert np.all(np.isnan(sub_node.hdu[1].data["FLUX_BKG_ERR"][1]))
    assert np.all(np.isfinite(sub_node.hdu[1].data["FLUX_BKG_ERR"][0]))


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
