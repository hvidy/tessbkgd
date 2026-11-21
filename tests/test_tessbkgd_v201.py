"""v2.0.1 bugfix regression tests ([TBKG-01], [TBKG-02], [TBKG-03b], [TBKG-P1])."""

from __future__ import annotations

import warnings
from pathlib import Path

import astropy.units as u
import lightkurve as lk
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import MaskedColumn, QTable, Table
from astropy.time import Time
from astropy.wcs import WCS
from erfa import ErfaWarning

import tessbkgd
from tessbkgd import (
    HALOCVER,
    PixelQuality,
    apply_curvature_and_rank_guards,
    bkgd_tpf,
    build_wcs_source_mask,
    read_bkg_covariance,
    read_bkgsub_state,
    read_smear_err,
    subtract_background,
    write_tessbkgd_provenance,
)


def _make_tpf(
    flux: np.ndarray,
    *,
    flux_err: np.ndarray | None = None,
    smear_err: np.ndarray | None = None,
    pixel_quality: np.ndarray | None = None,
    wcs: WCS | None = None,
    time_btjd: np.ndarray | None = None,
    bjdrefi: int | None = 2457000,
    date_obs: str | None = None,
    mjd_obs: float | None = None,
) -> lk.TessTargetPixelFile:
    n_raw, ny, nx = flux.shape
    n_pix = ny * nx
    img_dim = f"({nx}, {ny})"
    t_arr = (
        np.linspace(2000.0, 2027.0, n_raw, dtype=np.float64)
        if time_btjd is None
        else np.asarray(time_btjd, dtype=np.float64)
    )
    cols = [
        fits.Column(
            name="TIME", format="D", unit="BJD - 2457000, days", array=t_arr
        ),
        fits.Column(
            name="CADENCENO",
            format="J",
            array=np.arange(1, n_raw + 1, dtype=np.int32),
        ),
        fits.Column(
            name="FLUX",
            format=f"{n_pix}E",
            dim=img_dim,
            unit="e-/s",
            array=np.asarray(flux, dtype=np.float32),
        ),
    ]
    if flux_err is not None:
        cols.append(
            fits.Column(
                name="FLUX_ERR",
                format=f"{n_pix}E",
                dim=img_dim,
                unit="e-/s",
                array=np.asarray(flux_err, dtype=np.float32),
            )
        )
    cols.extend(
        [
            fits.Column(
                name="FLUX_BKG",
                format=f"{n_pix}E",
                dim=img_dim,
                unit="e-/s",
                array=np.zeros((n_raw, ny, nx), dtype=np.float32),
            ),
            fits.Column(
                name="FLUX_BKG_ERR",
                format=f"{n_pix}E",
                dim=img_dim,
                unit="e-/s",
                array=np.zeros((n_raw, ny, nx), dtype=np.float32),
            ),
            fits.Column(
                name="QUALITY",
                format="J",
                array=np.zeros(n_raw, dtype=np.int32),
            ),
        ]
    )
    if smear_err is not None:
        cols.append(
            fits.Column(
                name="SMEAR_ERR",
                format=f"{nx}E",
                dim=f"({nx})",
                unit="e-/s",
                array=np.asarray(smear_err, dtype=np.float32),
            )
        )

    h0 = fits.PrimaryHDU()
    h0.header["TELESCOP"] = "TESS"
    h0.header["INSTRUME"] = "TESS Photometer"
    h0.header["HALOCVER"] = HALOCVER
    h0.header["BKGSUB"] = False
    if date_obs is not None:
        h0.header["DATE-OBS"] = date_obs
    if mjd_obs is not None:
        h0.header["MJD-OBS"] = mjd_obs

    h1 = fits.BinTableHDU.from_columns(cols, name="PIXELS")
    h1.header["HALOCVER"] = HALOCVER
    h1.header["BKGSUB"] = False
    if bjdrefi is not None:
        h1.header["BJDREFI"] = bjdrefi
        h1.header["BJDREFF"] = 0.0

    ap_hdr = wcs.to_header() if wcs is not None else fits.Header()
    h2 = fits.ImageHDU(
        data=np.ones((ny, nx), dtype=np.int32), header=ap_hdr, name="APERTURE"
    )
    hdus: list[fits.PrimaryHDU | fits.BinTableHDU | fits.ImageHDU] = [
        h0,
        h1,
        h2,
    ]
    if pixel_quality is not None:
        hdus.append(
            fits.ImageHDU(
                data=np.asarray(pixel_quality, dtype=np.uint16),
                name="PIXEL_QUALITY",
            )
        )
    return lk.TessTargetPixelFile(fits.HDUList(hdus))


def test_tbkg01_single_pixel_per_column_smear_fallback() -> None:
    """Verify [TBKG-01] 2D residual fallback when cnt_c == 1 in every column."""
    rng = np.random.default_rng(20261120)
    n_t, ny, nx = 80, 10, 15

    # Exactly 1 unmasked background pixel per column along a 2D curve -> full
    # rank for order=1 (P=3) with n_el == 0 in every active column.
    mask_1pc = np.zeros((ny, nx), dtype=bool)
    for c in range(nx):
        mask_1pc[(c * 3) % ny, c] = True

    # (a) has_flux_err=False with SMEAR_ERR > 0: never NaN, continuous as se->0+,
    #     and calibrated pulls when both pixel and smear noise are present.
    s_ind, s_sm = 2.0, 1.5
    sky_true = 100.0
    u_1a = (
        sky_true
        + rng.normal(0.0, s_ind, size=(n_t, ny, nx))
        + rng.normal(0.0, s_sm, size=(n_t, nx))[:, None, :]
    )
    se_1a = np.full((n_t, nx), s_sm, dtype=np.float32)
    se_tiny = np.full((n_t, nx), 1e-6, dtype=np.float32)

    sub_nose, diag_nose = subtract_background(
        _make_tpf(u_1a, flux_err=None, smear_err=None),
        order=1,
        mask=mask_1pc,
        return_diagnostics=True,
    )
    sub_se, diag_se = subtract_background(
        _make_tpf(u_1a, flux_err=None, smear_err=se_1a),
        order=1,
        mask=mask_1pc,
        return_diagnostics=True,
    )
    sub_tiny, diag_tiny = subtract_background(
        _make_tpf(u_1a, flux_err=None, smear_err=se_tiny),
        order=1,
        mask=mask_1pc,
        return_diagnostics=True,
    )

    assert np.all(np.isfinite(diag_se.bkg_chol))
    assert np.all(np.isfinite(sub_se.hdu[1].data["FLUX_BKG_ERR"]))
    assert np.all(np.diagonal(diag_se.bkg_chol, axis1=1, axis2=2) >= 0.0)
    # Continuity at SMEAR_ERR -> 0+ (matches smear_err=None path closely):
    np.testing.assert_allclose(
        diag_tiny.bkg_chol, diag_nose.bkg_chol, rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        sub_tiny.hdu[1].data["FLUX_BKG_ERR"],
        sub_nose.hdu[1].data["FLUX_BKG_ERR"],
        rtol=1e-4,
        atol=1e-5,
    )
    # Calibrated total FLUX_BKG_ERR pulls on sky pixels:
    pulls_noerr = (
        sub_se.flux_bkg.to_value("electron/s")[:, mask_1pc] - sky_true
    ) / sub_se.flux_bkg_err.to_value("electron/s")[:, mask_1pc]
    assert 0.80 <= float(np.std(pulls_noerr, ddof=1)) <= 1.25

    # Smear-dominated has_flux_err=False: clamps c_rob >= 0 cleanly (no NaN)
    u_sm_dom = (
        sky_true
        + rng.normal(0.0, 0.05, size=(20, ny, nx))
        + rng.normal(0.0, 2.5, size=(20, nx))[:, None, :]
    )
    sub_dom, diag_dom = subtract_background(
        _make_tpf(
            u_sm_dom,
            flux_err=None,
            smear_err=np.full((20, nx), 2.5, dtype=np.float32),
        ),
        order=1,
        mask=mask_1pc,
        return_diagnostics=True,
    )
    assert np.all(np.isfinite(diag_dom.bkg_chol))
    assert np.all(np.isfinite(sub_dom.hdu[1].data["FLUX_BKG_ERR"]))
    assert np.all(sub_dom.hdu[1].data["FLUX_BKG_ERR"] > 0.0)

    # Vertical df_rem == 0 (1 column has 2 rows, 14 columns have 1 row, order=1):
    mask_1col2 = mask_1pc.copy()
    mask_1col2[(0 * 3 + 1) % ny, 0] = True
    sub_v0, diag_v0 = subtract_background(
        _make_tpf(u_1a[:20], flux_err=None, smear_err=se_1a[:20]),
        order=1,
        mask=mask_1col2,
        return_diagnostics=True,
    )
    assert np.all(np.isfinite(diag_v0.bkg_chol))
    assert np.all(np.isfinite(sub_v0.hdu[1].data["FLUX_BKG_ERR"]))

    # (b) has_flux_err=True with cnt_c == 1 + unmodelled spatial dome (tau >> 1):
    #     must NOT collapse c_rob to 1.0 when SMEAR_ERR is tiny or moderate!
    yy, xx = np.mgrid[0:ny, 0:nx]
    dome = 15.0 * np.sin(0.7 * yy + 0.5 * xx)
    u_1b = 100.0 + dome[None, :, :] + rng.normal(0.0, 1.0, size=(30, ny, nx))
    fe_1b = np.ones((30, ny, nx), dtype=np.float32)
    se_1b_tiny = np.full((30, nx), 1e-6, dtype=np.float32)

    _, d_1b_nose = subtract_background(
        _make_tpf(u_1b, flux_err=fe_1b, smear_err=None),
        order=1,
        mask=mask_1pc,
        return_diagnostics=True,
    )
    _, d_1b_tiny = subtract_background(
        _make_tpf(u_1b, flux_err=fe_1b, smear_err=se_1b_tiny),
        order=1,
        mask=mask_1pc,
        return_diagnostics=True,
    )
    assert float(d_1b_nose.bkg_chol[0, 0, 0]) > 8.0
    np.testing.assert_allclose(
        d_1b_tiny.bkg_chol, d_1b_nose.bkg_chol, rtol=1e-4, atol=1e-5
    )

    # Moderate SMEAR_ERR with spatial dome: c_rob rescales past 1.0
    se_1b_mod = np.full((30, nx), 1.0, dtype=np.float32)
    fe_1b_mod = np.full((30, ny, nx), np.sqrt(2.0), dtype=np.float32)
    sub_1b_mod, d_1b_mod = subtract_background(
        _make_tpf(u_1b, flux_err=fe_1b_mod, smear_err=se_1b_mod),
        order=1,
        mask=mask_1pc,
        return_diagnostics=True,
    )
    assert float(np.median(d_1b_mod.bkg_chol[:, 0, 0])) > 8.0
    assert np.all(np.isfinite(sub_1b_mod.hdu[1].data["FLUX_BKG_ERR"]))

    # Vanishing vertical DoF (df_rem == 0 on mask_1col2) with has_flux_err=True
    # and a purely horizontal unmodelled sky dome: must STILL trigger 2D
    # fallback rather than collapsing c_rob ~ 1.0 when SMEAR_ERR -> 0+!
    horiz_dome = 15.0 * np.sin(0.5 * xx)
    u_v0_dome = (
        100.0 + horiz_dome[None, :, :] + rng.normal(0.0, 1.0, size=(30, ny, nx))
    )
    _, d_v0_nose = subtract_background(
        _make_tpf(u_v0_dome, flux_err=fe_1b, smear_err=None),
        order=1,
        mask=mask_1col2,
        return_diagnostics=True,
    )
    _, d_v0_tiny = subtract_background(
        _make_tpf(u_v0_dome, flux_err=fe_1b, smear_err=se_1b_tiny),
        order=1,
        mask=mask_1col2,
        return_diagnostics=True,
    )
    assert float(d_v0_nose.bkg_chol[0, 0, 0]) > 6.0
    np.testing.assert_allclose(
        d_v0_tiny.bkg_chol, d_v0_nose.bkg_chol, rtol=1e-4, atol=1e-5
    )
    for se_val in (0.0, 1e-6, 0.5, 2.0):
        fe_sw = np.full(
            (16, ny, nx), np.hypot(1.0, se_val), dtype=np.float32
        )
        se_sw = np.full((16, nx), se_val, dtype=np.float32)
        sub_sw, d_sw = subtract_background(
            _make_tpf(u_v0_dome[:16], flux_err=fe_sw, smear_err=se_sw),
            order=1,
            mask=mask_1col2,
            return_diagnostics=True,
        )
        assert np.all(np.isfinite(d_sw.bkg_chol))
        assert float(np.median(d_sw.bkg_chol[:, 0, 0])) > 5.0
        assert np.all(np.isfinite(sub_sw.hdu[1].data["FLUX_BKG_ERR"]))

    # Noiseless-pixel limit (sigma_indep == 0) on cnt_c == 1 -> BKG_CHOL == 0
    u_zero = 100.0 + rng.normal(0.0, 1.5, size=(12, nx))[:, None, :] * np.ones(
        (1, ny, 1)
    )
    sub_z, d_z = subtract_background(
        _make_tpf(
            u_zero,
            flux_err=np.full((12, ny, nx), 1.5, dtype=np.float32),
            smear_err=np.full((12, nx), 1.5, dtype=np.float32),
        ),
        order=1,
        mask=mask_1pc,
        return_diagnostics=True,
    )
    np.testing.assert_array_equal(d_z.bkg_chol, 0.0)
    assert np.all(np.isfinite(sub_z.hdu[1].data["FLUX_BKG_ERR"]))


def test_tbkg02_barnards_star_proper_motion_propagation() -> None:
    """Verify [TBKG-02] PM propagation on Barnard's Star (10.4\"/yr) & TPF epochs."""
    ny, nx = 25, 15
    w = WCS(naxis=2)
    w.wcs.crpix = [8.0, 8.0]
    w.wcs.cdelt = [-21.0 / 3600.0, 21.0 / 3600.0]
    w.wcs.crval = [269.452075, 4.69339]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    pmra_barnard = -801.55
    pmdec_barnard = 10362.39  # ~10.4 arcsec/yr total PM

    # (a) 10-year propagation from Gaia DR3 (default ref_epoch=2016.0 inferred
    #     from source_id/phot_g_mean_mag) to J2026.0 (~5.0 TESS pixels shift):
    sc_2016 = SkyCoord(
        ra=269.452075 * u.deg,
        dec=4.69339 * u.deg,
        pm_ra_cosdec=pmra_barnard * (u.mas / u.yr),
        pm_dec=pmdec_barnard * (u.mas / u.yr),
        obstime=Time(2016.0, format="jyear", scale="tdb"),
        frame="icrs",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErfaWarning)
        sc_2026 = sc_2016.apply_space_motion(
            new_obstime=Time(2026.0, format="jyear", scale="tdb")
        )
    tx_2026, ty_2026 = w.world_to_pixel(
        SkyCoord(ra=sc_2026.ra, dec=sc_2026.dec, frame="icrs")
    )
    r_true_10yr = int(round(float(ty_2026)))
    c_true_10yr = int(round(float(tx_2026)))
    assert abs(r_true_10yr - 7) >= 4

    gaia_dr3_tbl = Table(
        {
            "source_id": [4472832130942575872],
            "ra": [269.452075],
            "dec": [4.69339],
            "pmra": [pmra_barnard],
            "pmdec": [pmdec_barnard],
            "phot_g_mean_mag": [8.2],
        }
    )
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        src_10yr = build_wcs_source_mask(
            ny, nx, w, gaia_dr3_tbl, obs_time=2026.0, max_radius_px=2.0
        )
    erfa_w = [x for x in rec if issubclass(x.category, ErfaWarning)]
    assert len(erfa_w) == 0
    assert bool(src_10yr[r_true_10yr, c_true_10yr]) is True
    assert bool(src_10yr[7, 7]) is False

    # (b) End-to-end subtract_background inferring mid-sector obs_time from TPF
    #     TIME + BJDREFI (J2026.0 == JD 2461041.25 -> BTJD 4041.25):
    rng = np.random.default_rng(77)
    sky = 120.0 + rng.normal(0.0, 0.5, size=(16, ny, nx))
    btjd_2026 = np.linspace(4028.0, 4054.5, 16)
    tpf_2026 = _make_tpf(
        sky,
        flux_err=np.full_like(sky, 0.5),
        wcs=w,
        time_btjd=btjd_2026,
        bjdrefi=2457000,
    )
    _, diag_2026 = subtract_background(
        tpf_2026,
        order=1,
        gaia_sources=gaia_dr3_tbl,
        max_source_radius=2.0,
        return_diagnostics=True,
    )
    assert bool(diag_2026.source_mask[r_true_10yr, c_true_10yr]) is True
    assert bool(diag_2026.source_mask[7, 7]) is False

    # (c) 26.5-year TIC propagation (default ref_epoch=2000.0 + explicit ref_epoch)
    #     with mixed 2-parameter (NaN PM) star & unit-bearing QTable:
    sc_2000 = SkyCoord(
        ra=269.452075 * u.deg,
        dec=4.69339 * u.deg,
        pm_ra_cosdec=pmra_barnard * (u.mas / u.yr),
        pm_dec=pmdec_barnard * (u.mas / u.yr),
        obstime=Time(2000.0, format="jyear", scale="tdb"),
        frame="icrs",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErfaWarning)
        sc_2026_5 = sc_2000.apply_space_motion(new_obstime=Time("J2026.5"))
    tx_265, ty_265 = w.world_to_pixel(
        SkyCoord(ra=sc_2026_5.ra, dec=sc_2026_5.dec, frame="icrs")
    )
    r_265, c_265 = int(round(float(ty_265))), int(round(float(tx_265)))
    assert abs(r_265 - 7) >= 12

    ra_static, dec_static = w.pixel_to_world_values(2.0, 2.0)
    mixed_qtab = QTable(
        {
            "ra": [269.452075, float(ra_static)] * u.deg,
            "dec": [4.69339, float(dec_static)] * u.deg,
            "pm_ra_cosdec": [pmra_barnard / 1000.0, np.nan] * (u.arcsec / u.yr),
            "pmdec": [pmdec_barnard / 1000.0, np.nan] * (u.arcsec / u.yr),
            "tmag": [8.0, 12.0],
        }
    )
    src_265 = build_wcs_source_mask(
        ny, nx, w, mixed_qtab, obs_time="J2026.5", max_radius_px=2.0
    )
    assert bool(src_265[r_265, c_265]) is True
    assert bool(src_265[7, 7]) is False
    # 2-parameter star without PM remains masked at its catalogue pixel (2, 2):
    assert bool(src_265[2, 2]) is True

    # (d) Raw HDUList DATE-OBS fallback when TIME is all-NaN:
    tpf_dateobs = _make_tpf(
        sky,
        flux_err=np.full_like(sky, 0.5),
        wcs=w,
        time_btjd=np.full(16, np.nan),
        bjdrefi=None,
        date_obs="2026-01-01T12:00:00",
    )
    _, diag_dateobs = subtract_background(
        tpf_dateobs.hdu,
        order=1,
        gaia_sources=gaia_dr3_tbl,
        max_source_radius=2.0,
        return_diagnostics=True,
    )
    assert bool(diag_dateobs.source_mask[r_true_10yr, c_true_10yr]) is True
    assert bool(diag_dateobs.source_mask[7, 7]) is False

    # (e) MaskedColumn pmra/pmdec does not drop static stars:
    masked_pm = Table(
        [
            MaskedColumn([269.452075, float(ra_static)], mask=[False, False]),
            MaskedColumn([4.69339, float(dec_static)], mask=[False, False]),
            MaskedColumn([pmra_barnard, 99999.0], mask=[False, True]),
            MaskedColumn([pmdec_barnard, 99999.0], mask=[False, True]),
            MaskedColumn([8.2, 12.0], mask=[False, False]),
        ],
        names=("ra", "dec", "pmra", "pmdec", "phot_g_mean_mag"),
    )
    src_masked_pm = build_wcs_source_mask(
        ny, nx, w, masked_pm, obs_time=2026.0, max_radius_px=2.0
    )
    assert bool(src_masked_pm[r_true_10yr, c_true_10yr]) is True
    assert bool(src_masked_pm[2, 2]) is True

    # (f) VizieR Gaia DR3 (RA_ICRS, DE_ICRS, pmRA, pmDE, Gmag, Source) infers
    #     2016.0 and propagates pmDE; RAJ2000 + source_id infers 2000.0; partial
    #     PM emits UserWarning:
    vizier_gaia = Table(
        {
            "Source": [4472832130942575872],
            "RA_ICRS": [269.452075],
            "DE_ICRS": [4.69339],
            "pmRA": [pmra_barnard],
            "pmDE": [pmdec_barnard],
            "Gmag": [8.2],
        }
    )
    src_vizier = build_wcs_source_mask(
        ny, nx, w, vizier_gaia, obs_time=tpf_2026.time, max_radius_px=2.0
    )
    assert bool(src_vizier[r_true_10yr, c_true_10yr]) is True
    assert bool(src_vizier[7, 7]) is False

    xmatch_j2000 = Table(
        {
            "source_id": [4472832130942575872],
            "RAJ2000": [269.452075],
            "DEJ2000": [4.69339],
            "pmra": [pmra_barnard],
            "pmdec": [pmdec_barnard],
            "phot_g_mean_mag": [8.2],
        }
    )
    src_xmatch = build_wcs_source_mask(
        ny, nx, w, xmatch_j2000, obs_time="J2026.5", max_radius_px=2.0
    )
    assert bool(src_xmatch[r_265, c_265]) is True
    assert bool(src_xmatch[7, 7]) is False

    with pytest.warns(UserWarning, match="only one proper-motion component"):
        build_wcs_source_mask(
            ny,
            nx,
            w,
            {"ra": [269.452075], "dec": [4.69339], "pmra": [pmra_barnard]},
            obs_time=2026.0,
        )

    # (g) MJD / JD / u.d ref_epoch columns convert cleanly & ambiguous bare
    #     BTJD float obs_time raises ValueError:
    mjd_2016 = float(Time(2016.0, format="jyear", scale="tdb").mjd)
    jd_2016 = float(Time(2016.0, format="jyear", scale="tdb").jd)
    for ep_col in (
        [mjd_2016],
        [mjd_2016] * u.d,
        [jd_2016],
    ):
        src_ep = build_wcs_source_mask(
            ny,
            nx,
            w,
            {
                "ra": [269.452075],
                "dec": [4.69339],
                "pmra": [pmra_barnard],
                "pmdec": [pmdec_barnard],
                "ref_epoch": ep_col,
                "tmag": [8.0],
            },
            obs_time=2026.0,
            max_radius_px=2.0,
        )
        assert bool(src_ep[r_true_10yr, c_true_10yr]) is True
        assert bool(src_ep[7, 7]) is False

    for bad_btjd in (1325.0, 4041.25):
        with pytest.raises(ValueError, match="format='btjd'"):
            build_wcs_source_mask(
                ny, nx, w, gaia_dr3_tbl, obs_time=bad_btjd
            )


def test_tbkg03b_flux_bkg_setter_strips_stale_covariance_and_bool_halocver(
    tmp_path: Path,
) -> None:
    """Verify [TBKG-03b] stale covariance stripping, variance restore & HALOCVER."""
    rng = np.random.default_rng(303)
    n_t, ny, nx = 16, 10, 10
    flux = 150.0 + rng.normal(0.0, 0.8, size=(n_t, ny, nx))
    orig_fe = np.full((n_t, ny, nx), 0.8, dtype=np.float32)
    pq_init = np.zeros((ny, nx), dtype=np.uint16)
    pq_init[:, 0] = np.uint16(PixelQuality.STRAP_COL)
    tpf = _make_tpf(flux, flux_err=orig_fe, pixel_quality=pq_init)

    sub = subtract_background(tpf, order=1)
    assert read_bkg_covariance(sub) is not None
    assert "BKG_BASIS" in sub.hdu
    assert "BKG_CHOL" in [c.name.upper() for c in sub.hdu[1].columns]
    assert np.any(
        (sub.hdu["PIXEL_QUALITY"].data & int(PixelQuality.BKG_USED)) != 0
    )

    # Legacy bkgd_tpf.flux_bkg = custom_val must warn, strip BKG_CHOL/BKG_BASIS/
    # BKG_USED/BKG_ORD, restore FLUX_ERR, and clear FLUX_BKG_ERR -> NaN:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        bt = bkgd_tpf(fits.HDUList([h.copy() for h in sub.hdu]))
    bt.hdu[0].header["BKG_ORD"] = 1

    with pytest.warns(UserWarning, match="BKG_CHOL"):
        bt.flux_bkg = np.full((n_t, ny, nx), 25.0, dtype=np.float32)

    assert read_bkg_covariance(bt) is None
    assert "BKG_BASIS" not in bt.hdu
    assert "BKG_CHOL" not in [c.name.upper() for c in bt.hdu[1].columns]
    assert "BKG_ORD" not in bt.hdu[0].header
    assert read_bkgsub_state(bt) is True
    assert not np.any(
        (bt.hdu["PIXEL_QUALITY"].data & int(PixelQuality.BKG_USED)) != 0
    )
    assert np.all(
        (bt.hdu["PIXEL_QUALITY"].data[:, 0] & int(PixelQuality.STRAP_COL)) != 0
    )
    assert np.all(np.isnan(bt.hdu[1].data["FLUX_BKG_ERR"]))
    np.testing.assert_allclose(
        bt.hdu[1].data["FLUX_ERR"], orig_fe, rtol=1e-4, atol=1e-4
    )

    # Round-trip stripped TPF to disk cleanly:
    out_fits = tmp_path / "stripped_bt.fits"
    bt.to_fits(str(out_fits), overwrite=True)
    reloaded = lk.read(str(out_fits))
    assert read_bkg_covariance(reloaded) is None

    # Assigning zero background strips stale covariance, zeroes FLUX_BKG_ERR,
    # restores FLUX_ERR, clears BKG_USED, sets BKGSUB=False, and avoids
    # double-counting variance on a subsequent subtract_background():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        bt_zero = bkgd_tpf(fits.HDUList([h.copy() for h in sub.hdu]))
    bt_zero.flux = bt_zero.flux + bt_zero.flux_bkg
    with pytest.warns(UserWarning, match="BKG_CHOL"):
        bt_zero.flux_bkg = np.zeros((n_t, ny, nx), dtype=np.float32)
    assert read_bkg_covariance(bt_zero) is None
    assert read_bkgsub_state(bt_zero) is False
    assert not np.any(
        (bt_zero.hdu["PIXEL_QUALITY"].data & int(PixelQuality.BKG_USED)) != 0
    )
    np.testing.assert_array_equal(bt_zero.hdu[1].data["FLUX_BKG_ERR"], 0.0)
    np.testing.assert_allclose(
        bt_zero.hdu[1].data["FLUX_ERR"], orig_fe, rtol=1e-4, atol=1e-4
    )
    resub = subtract_background(bt_zero, order=1)
    np.testing.assert_allclose(
        resub.hdu[1].data["FLUX_ERR"],
        sub.hdu[1].data["FLUX_ERR"],
        rtol=2e-4,
        atol=2e-4,
    )

    # HALOCVER boolean / float rejection even when the OTHER HDU has int 1
    # (in Python, 1 == True and 1 == 1.0, so each HDU card must be type-checked!):
    basis = np.ones((3, ny, nx), dtype=np.float32)
    chol = np.broadcast_to(np.eye(3, dtype=np.float32), (n_t, 3, 3)).copy()
    used = np.ones((ny, nx), dtype=bool)
    for h0_ver, h1_ver in ((1, True), (True, 1), (1, 1.0), (1.0, 1)):
        bad_hdul = fits.HDUList([h.copy() for h in sub.hdu])
        bad_hdul[0].header["HALOCVER"] = h0_ver
        bad_hdul[1].header["HALOCVER"] = h1_ver
        with pytest.raises(ValueError, match="Unsupported HALOCVER"):
            read_bkgsub_state(bad_hdul)
        with pytest.raises(ValueError, match="Unsupported HALOCVER"):
            read_bkg_covariance(bad_hdul)
        with pytest.raises(ValueError, match="Unsupported HALOCVER"):
            read_smear_err(bad_hdul)
        with pytest.raises(ValueError, match="Unsupported HALOCVER"):
            write_tessbkgd_provenance(bad_hdul, basis, chol, used)


def test_tbkgp1_wide_interior_bleed_gap_curvature_guard() -> None:
    """Verify [TBKG-P1] wide >50% central SAT_BLEED gap downgrades order >= 2."""
    rng = np.random.default_rng(404)
    ny, nx = 30, 15

    # Central columns 2..12 masked (11/15 = 73% > 50% interior gap), leaving
    # narrow 2-col flank strips 0..1 and 13..14 (kappa = 6.11 <= 100 for order=2):
    mask_wide_col_gap = np.zeros((ny, nx), dtype=bool)
    mask_wide_col_gap[:, :2] = True
    mask_wide_col_gap[:, 13:] = True

    eff_ord2, kap_ord2 = apply_curvature_and_rank_guards(
        mask_wide_col_gap, 2, allow_downgrade=True
    )
    assert eff_ord2 == 1 and kap_ord2 <= 100.0
    eff_ord3, _ = apply_curvature_and_rank_guards(
        mask_wide_col_gap, 3, allow_downgrade=True
    )
    assert eff_ord3 == 1

    # Explicit allow_downgrade=False still honours full-rank order=2 request:
    eff_forced, kap_forced = apply_curvature_and_rank_guards(
        mask_wide_col_gap, 2, allow_downgrade=False
    )
    assert eff_forced == 2 and kap_forced < 20.0

    # Wide interior row gap (> 50% of ny) also downgrades order=2 -> 1:
    mask_wide_row_gap = np.zeros((15, 30), dtype=bool)
    mask_wide_row_gap[:2, :] = True
    mask_wide_row_gap[13:, :] = True
    eff_row, _ = apply_curvature_and_rank_guards(
        mask_wide_row_gap, 2, allow_downgrade=True
    )
    assert eff_row == 1

    # Control: narrow 3-col bleed (3/15 = 20% <= 50%) and full border ring
    # retain order=2 cleanly:
    mask_narrow_gap = np.ones((ny, nx), dtype=bool)
    mask_narrow_gap[:, 6:9] = False
    eff_narrow, _ = apply_curvature_and_rank_guards(
        mask_narrow_gap, 2, allow_downgrade=True
    )
    assert eff_narrow == 2

    mask_ring = np.zeros((ny, nx), dtype=bool)
    mask_ring[0, :] = True
    mask_ring[-1, :] = True
    mask_ring[:, 0] = True
    mask_ring[:, -1] = True
    eff_ring, _ = apply_curvature_and_rank_guards(
        mask_ring, 2, allow_downgrade=True
    )
    assert eff_ring == 2

    # End-to-end subtract_background with SAT_BLEED cols 2..12 + residual flank
    # wing excess (+3 e-/s on inner flank cols 1 and 13): order=2 would arch
    # ~3.8x (+11.4 e-/s) at stamp centre, while guarded order=1 stays flat!
    pq = np.zeros((ny, nx), dtype=np.uint16)
    pq[:, 2:13] = np.uint16(PixelQuality.SAT_BLEED)
    yy, xx = np.mgrid[0:ny, 0:nx]
    true_plane = 150.0 + 0.8 * xx - 0.4 * yy
    wing_leak = np.zeros((ny, nx), dtype=np.float64)
    wing_leak[:, 1] = 3.0
    wing_leak[:, 13] = 3.0
    u_cube = (
        true_plane[None, :, :]
        + wing_leak[None, :, :]
        + rng.normal(0.0, 0.3, size=(16, ny, nx))
    )
    tpf = _make_tpf(u_cube, flux_err=np.full_like(u_cube, 0.3), pixel_quality=pq)

    sub_guarded, diag_guarded = subtract_background(
        tpf, order=2, allow_downgrade=True, return_diagnostics=True
    )
    sub_unguarded, diag_unguarded = subtract_background(
        tpf,
        order=2,
        mask=mask_wide_col_gap,
        allow_downgrade=False,
        return_diagnostics=True,
    )
    assert diag_guarded.effective_order == 1
    assert sub_guarded.hdu["BKG_BASIS"].header["BKG_ORD"] == 1
    assert diag_unguarded.effective_order == 2

    center_bias_guarded = float(
        np.mean(
            sub_guarded.flux_bkg.to_value("electron/s")[:, :, 7]
            - true_plane[:, 7]
        )
    )
    center_bias_unguarded = float(
        np.mean(
            sub_unguarded.flux_bkg.to_value("electron/s")[:, :, 7]
            - true_plane[:, 7]
        )
    )
    assert abs(center_bias_unguarded) > 10.0
    assert abs(center_bias_guarded) < 2.0
    assert tessbkgd.__version__ == "2.0.1"
