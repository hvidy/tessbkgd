"""Phase 3 scientific, numerical, and API test suite for tessbkgd v2.0."""

from __future__ import annotations

import warnings

import astropy.units as u
import jax
import lightkurve as lk
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import MaskedColumn, QTable, Table
from astropy.wcs import WCS

import tessbkgd
from conftest import make_synthetic_tpf
from tessbkgd import (
    HALOCVER,
    BackgroundDiagnostics,
    BackgroundEstimate,
    PixelQuality,
    apply_curvature_and_rank_guards,
    bkgd_tpf,
    build_background_mask,
    build_wcs_source_mask,
    equilibrate_chebyshev_basis,
    estimate_background,
    n_basis_for_order,
    normalized_grid,
    parse_order,
    raw_chebyshev_2d,
    read_bkgsub_state,
    reconstruct_background_chunked,
    solve_chebyshev_background,
    subtract_background,
    unpack_bkg_chol,
    write_tessbkgd_provenance,
)


def _build_custom_tpf(
    flux: np.ndarray,
    *,
    flux_err: np.ndarray | None = None,
    flux_bkg: np.ndarray | None = None,
    flux_bkg_err: np.ndarray | None = None,
    quality: np.ndarray | None = None,
    pixel_quality: np.ndarray | None = None,
    bkgsub: bool | None = False,
    backapp: bool | None = None,
    wcs: WCS | None = None,
    include_tunit: bool = True,
) -> lk.TessTargetPixelFile:
    """Build an in-memory ``TessTargetPixelFile`` with configurable columns/headers."""
    n_raw, ny, nx = flux.shape
    n_pix = ny * nx
    img_dim = f"({nx}, {ny})"
    flux_unit = "e-/s" if include_tunit else None

    time = np.linspace(1800.0, 1827.0, n_raw, dtype=np.float64)
    cadenceno = np.arange(5000, 5000 + n_raw, dtype=np.int32)
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
        fits.Column(name="TIME", format="D", unit="BJD - 2457000, days", array=time),
        fits.Column(name="TIMECORR", format="E", unit="d", array=timecorr),
        fits.Column(name="CADENCENO", format="J", array=cadenceno),
        fits.Column(
            name="RAW_CNTS",
            format=f"{n_pix}J",
            dim=img_dim,
            unit="count" if include_tunit else None,
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
    if flux_bkg is not None:
        cols.append(
            fits.Column(
                name="FLUX_BKG",
                format=f"{n_pix}E",
                dim=img_dim,
                unit=flux_unit,
                array=np.asarray(flux_bkg, dtype=np.float32),
            )
        )
    if flux_bkg_err is not None:
        cols.append(
            fits.Column(
                name="FLUX_BKG_ERR",
                format=f"{n_pix}E",
                dim=img_dim,
                unit=flux_unit,
                array=np.asarray(flux_bkg_err, dtype=np.float32),
            )
        )
    cols.extend(
        [
            fits.Column(name="QUALITY", format="J", array=qual),
            fits.Column(name="POS_CORR1", format="E", unit="pixel", array=pos_corr1),
            fits.Column(name="POS_CORR2", format="E", unit="pixel", array=pos_corr2),
        ]
    )

    primary_hdr = fits.Header()
    primary_hdr["TELESCOP"] = "TESS"
    primary_hdr["INSTRUME"] = "TESS Photometer"
    primary_hdr["OBJECT"] = "TIC 123456789"
    primary_hdr["TICID"] = 123456789
    primary_hdr["SECTOR"] = 14
    primary_hdr["CAMERA"] = 2
    primary_hdr["CCD"] = 1
    if bkgsub is not None:
        primary_hdr["BKGSUB"] = bkgsub
    if backapp is not None:
        primary_hdr["BACKAPP"] = backapp
    primary_hdu = fits.PrimaryHDU(header=primary_hdr)

    pixels_hdu = fits.BinTableHDU.from_columns(cols, name="PIXELS")
    if bkgsub is not None:
        pixels_hdu.header["BKGSUB"] = bkgsub
    if backapp is not None:
        pixels_hdu.header["BACKAPP"] = backapp

    ap_hdr = wcs.to_header() if wcs is not None else fits.Header()
    aperture_hdu = fits.ImageHDU(
        data=np.ones((ny, nx), dtype=np.int32), header=ap_hdr, name="APERTURE"
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


def _make_tan_wcs(ny: int, nx: int, ra0: float = 120.0, dec0: float = -30.0) -> WCS:
    """Create a standard TESS-scale (21 arcsec/px) celestial TAN WCS."""
    w = WCS(naxis=2)
    w.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]
    w.wcs.cdelt = np.array([-21.0 / 3600.0, 21.0 / 3600.0], dtype=np.float64)
    w.wcs.crval = [ra0, dec0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def test_chebyshev_grid_orders_and_qr_equilibration() -> None:
    # 1x1 and 1D degenerate grid guards (max(n - 1, 1) denominator)
    eta_1, xi_1 = normalized_grid(1, 1)
    assert eta_1.shape == (1,) and xi_1.shape == (1,)
    assert float(eta_1[0]) == 0.0 and float(xi_1[0]) == 0.0

    eta, xi = normalized_grid(11, 15)
    np.testing.assert_allclose([eta[0], eta[-1]], [-1.0, 1.0])
    np.testing.assert_allclose([xi[0], xi[-1]], [-1.0, 1.0])

    with pytest.raises(ValueError, match="Grid dimensions"):
        normalized_grid(0, 5)

    # Order resolution & validation
    assert parse_order(None, 12, 12) == 1
    assert parse_order(None, 39, 64) == 1
    assert parse_order(None, 40, 40) == 2
    for num, label, expected_p in [
        (0, "pedestal", 1),
        (1, "linear", 3),
        (2, "quadratic", 6),
        (3, "cubic", 10),
    ]:
        assert parse_order(num, 20, 20) == num
        assert parse_order(label, 20, 20) == num
        assert parse_order(label.upper(), 20, 20) == num
        assert n_basis_for_order(num) == expected_p

    for bad in (True, False, -1, 4, "quartic", 1.5):
        with pytest.raises(ValueError, match="Unrecognised order"):
            parse_order(bad, 20, 20)  # type: ignore[arg-type]

    # QR equilibration on asymmetric background mask: kappa(Phi_bg) == 1 & diag(R) > 0
    ny, nx = 16, 18
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    asym_mask = ((yy - 6) ** 2 + (xx - 5) ** 2 > 4.2**2) & (xx >= 2)
    for d in (0, 1, 2, 3):
        p_dim = n_basis_for_order(d)
        phi_full, phi_bg, r_bg = equilibrate_chebyshev_basis(ny, nx, d, asym_mask)
        assert phi_full.shape == (p_dim, ny, nx)
        assert phi_bg.shape == (int(np.count_nonzero(asym_mask)), p_dim)
        assert np.all(np.diag(r_bg) > 0.0)
        np.testing.assert_allclose(
            phi_bg.T @ phi_bg, np.eye(p_dim), rtol=0.0, atol=1e-13
        )
        np.testing.assert_allclose(
            phi_full[:, asym_mask].T, phi_bg, rtol=0.0, atol=1e-13
        )
        svals = np.linalg.svd(phi_bg, compute_uv=False)
        np.testing.assert_allclose(svals[0] / svals[-1], 1.0, rtol=0.0, atol=1e-13)

        # Verify exact span equivalence: raw = R_bg.T @ phi_full
        raw = raw_chebyshev_2d(ny, nx, d)
        recon_raw = np.tensordot(r_bg.T, phi_full, axes=(1, 0))
        np.testing.assert_allclose(recon_raw, raw, rtol=0.0, atol=1e-12)


def test_dark_sky_dead_edges_and_dark_25_cadence_ranking() -> None:
    rng = np.random.default_rng(101)
    ny, nx = 13, 15
    eta, xi = normalized_grid(ny, nx)
    eta_g, xi_g = np.meshgrid(eta, xi, indexing="ij")

    # Dark sky (< 5 e-/s) in darkest 25% of cadences, with transient scattered
    # light + localized bright patch in the brightest 60% of cadences.
    base_t = np.concatenate(
        [
            np.full(16, 2.3, dtype=np.float64),
            np.linspace(12.0, 95.0, 32, dtype=np.float64),
        ]
    )
    sky_true = (
        base_t[:, None, None]
        + 0.45 * xi_g[None, :, :]
        - 0.30 * eta_g[None, :, :]
    )
    psf = 900.0 * np.exp(-0.5 * ((eta_g / 0.17) ** 2 + (xi_g / 0.17) ** 2))
    noise_sigma = 0.55
    flux = sky_true + psf[None, :, :] + rng.normal(0.0, noise_sigma, sky_true.shape)
    flux_err = np.full_like(flux, noise_sigma)

    # Dead U <= 0 detector strap/edge column (c=0) and dead row (r=ny-1)
    flux[:, :, 0] = 0.0
    flux[:, -1, :] = -0.2

    tpf = _build_custom_tpf(flux, flux_err=flux_err, bkgsub=False, backapp=False)
    clean_tpf, diag = subtract_background(tpf, order=None, return_diagnostics=True)

    assert isinstance(clean_tpf, lk.TessTargetPixelFile)
    assert not isinstance(clean_tpf, bkgd_tpf)
    assert isinstance(diag, BackgroundDiagnostics)
    assert diag.effective_order == 1

    # Dead U <= 0 edges and bright central star core must be excluded, while dark sky
    # (< 5 e-/s, despite occasional negative single-cadence noise draws) survives.
    assert not np.any(diag.bkg_mask[:, 0])
    assert not np.any(diag.bkg_mask[-1, :])
    assert not diag.bkg_mask[ny // 2, nx // 2]
    assert int(np.count_nonzero(diag.bkg_mask)) >= 70

    # Verify BACKAPP = True synchronized alongside BKGSUB = True and BKG_ORD = 1
    assert clean_tpf.hdu[0].header["BKGSUB"] is True
    assert clean_tpf.hdu[1].header["BKGSUB"] is True
    assert clean_tpf.hdu[0].header["BACKAPP"] is True
    assert clean_tpf.hdu[1].header["BACKAPP"] is True
    assert clean_tpf.hdu["BKG_BASIS"].header["BKG_ORD"] == 1
    assert clean_tpf.hdu[0].header["HALOCVER"] == HALOCVER

    # Background recovery on live pixels matches ground truth closely
    live_interior = np.ones((ny, nx), dtype=bool)
    live_interior[:, 0] = False
    live_interior[-1, :] = False
    bkg_est = clean_tpf.flux_bkg.to_value("electron/s")
    err_med = float(
        np.median(np.abs(bkg_est[:, live_interior] - sky_true[:, live_interior]))
    )
    assert err_med < 0.12
    assert bool(jax.config.jax_enable_x64) is False


def test_bright_earthshine_steep_gradient_and_wide_quadratic_stamp() -> None:
    rng = np.random.default_rng(202)

    # Part A: Bright Earthshine (> 1,500 e-/s) with > 100% spatial gradient
    # (750 e-/s at left edge -> 2,950 e-/s at right edge, mean 1,850 e-/s).
    n_cad, ny, nx = 36, 15, 17
    eta, xi = normalized_grid(ny, nx)
    eta_g, xi_g = np.meshgrid(eta, xi, indexing="ij")
    phase = np.linspace(0.0, np.pi, n_cad)
    sky_bright = (
        (1850.0 + 350.0 * np.sin(phase))[:, None, None]
        + 1100.0 * xi_g[None, :, :]
        - 220.0 * eta_g[None, :, :]
    )
    star = 18000.0 * np.exp(-0.5 * ((eta_g / 0.2) ** 2 + (xi_g / 0.2) ** 2))
    sig = 2.0
    flux = sky_bright + star[None, :, :] + rng.normal(0.0, sig, sky_bright.shape)
    tpf = _build_custom_tpf(flux, flux_err=np.full_like(flux, sig), bkgsub=False)

    clean, diag = subtract_background(tpf, order=None, return_diagnostics=True)
    assert diag.effective_order == 1
    # Both low-sky left column and high-sky right column must survive masking!
    assert int(np.count_nonzero(diag.bkg_mask[:, 0])) >= ny - 2
    assert int(np.count_nonzero(diag.bkg_mask[:, -1])) >= ny - 2
    assert not diag.bkg_mask[ny // 2, nx // 2]
    np.testing.assert_allclose(
        clean.flux_bkg.to_value("electron/s"), sky_bright, rtol=1e-3, atol=1.2
    )
    # BACKAPP was absent on input -> must not be invented
    assert "BACKAPP" not in clean.hdu[0].header

    # Part B: Wide stamp (min(nx, ny) >= 40) defaults to order=2 ('quadratic'),
    # preserves curved sky corners via degree-2 P_0(r, c), and honours PixelQuality.
    ny_w, nx_w = 40, 42
    eta_w, xi_w = normalized_grid(ny_w, nx_w)
    ew_g, xw_g = np.meshgrid(eta_w, xi_w, indexing="ij")
    sky_quad = (
        400.0
        + 95.0 * xw_g
        - 70.0 * ew_g
        + 160.0 * (xw_g**2 + ew_g**2)
        - 45.0 * xw_g * ew_g
    )[None, :, :] * (1.0 + 0.15 * np.sin(np.linspace(0.0, 3.0, 24))[:, None, None])
    star_w = 35000.0 * np.exp(-0.5 * ((ew_g / 0.14) ** 2 + (xw_g / 0.14) ** 2))
    u_wide = sky_quad + star_w[None, :, :] + rng.normal(0.0, 0.8, sky_quad.shape)

    pq = np.zeros((ny_w, nx_w), dtype=np.uint16)
    pq[:, 5] |= np.uint16(PixelQuality.STRAP_COL)
    pq[18:23, 20:22] |= np.uint16(PixelQuality.SAT_BLEED)
    pq[:, 21] |= np.uint16(PixelQuality.SMEAR_UNRESOLVED)
    pq[3, 3] |= np.uint16(PixelQuality.SOURCE_MASK)

    mask_res = build_background_mask(u_wide, pixel_quality=pq, order=None)
    assert mask_res.requested_order == 2
    assert mask_res.effective_order == 2
    # All four curved-sky corners must survive (degree-2 P_0 avoids corner clipping!)
    for cr, cc in [(0, 0), (0, nx_w - 1), (ny_w - 1, 0), (ny_w - 1, nx_w - 1)]:
        assert mask_res.mask[cr, cc], f"Corner ({cr}, {cc}) was clipped!"
    # All four PixelQuality exclusion bits must be strictly excluded
    assert not np.any(mask_res.mask[:, 5])
    assert not np.any(mask_res.mask[18:23, 20:22])
    assert not np.any(mask_res.mask[:, 21])
    assert not mask_res.mask[3, 3]

    # Also run full subtract_background on 40x42 with a border-ring PixelQuality mask
    # (<= 256 surviving pixels so XLA reuses the cached kernel instantaneously)
    pq_ring = pq.copy()
    pq_ring[1:-1, 1:-1] |= np.uint16(PixelQuality.SAT_BLEED)
    tpf_wide = _build_custom_tpf(
        u_wide,
        flux_err=np.full_like(u_wide, 0.8),
        pixel_quality=pq_ring,
        bkgsub=False,
    )
    clean_wide, diag_wide = subtract_background(
        tpf_wide, order=None, return_diagnostics=True
    )
    assert diag_wide.effective_order == 2
    assert clean_wide.hdu["BKG_BASIS"].header["BKG_ORD"] == 2
    pq_out = clean_wide.hdu["PIXEL_QUALITY"].data
    assert np.array_equal(
        (pq_out & int(PixelQuality.BKG_USED)) != 0, diag_wide.bkg_mask
    )
    np.testing.assert_allclose(
        clean_wide.flux_bkg.to_value("electron/s"), sky_quad, rtol=2e-3, atol=0.8
    )


def test_sweeping_earthshine_fan_does_not_self_mask_temporal_mad() -> None:
    rng = np.random.default_rng(303)
    n_cad, ny, nx = 64, 14, 16
    eta, xi = normalized_grid(ny, nx)
    eta_g, xi_g = np.meshgrid(eta, xi, indexing="ij")
    t_grid = np.linspace(0.0, 2.0 * np.pi, n_cad)

    # Sweeping Earthshine fan: spatial gradient swings across [-950, +1350] e-/s,
    # making raw pixel temporal MAD at corners > 600 e-/s vs 0.9 e-/s photon noise.
    b0 = 1600.0 + 900.0 * (0.5 - 0.5 * np.cos(t_grid))
    gx = -950.0 + 2300.0 * (0.5 + 0.5 * np.sin(t_grid))
    gy = 800.0 * np.cos(1.5 * t_grid)
    sky_fan = (
        b0[:, None, None]
        + gx[:, None, None] * xi_g[None, :, :]
        + gy[:, None, None] * eta_g[None, :, :]
    )
    sig = 0.9
    u_cube = sky_fan + rng.normal(0.0, sig, sky_fan.shape)

    # Inject a genuinely variable contaminant star at (2, 3) that is quiescent during
    # the darkest 25% of cadences (so M_{r,c} spatial median cannot see it!) and
    # flares/eclipses strongly during brighter cadences.
    dark_cut = np.percentile(b0, 30.0)
    bright_cad = b0 > dark_cut
    u_cube[bright_cad, 2, 3] += 45.0 * np.sin(9.0 * t_grid[bright_cad])

    res = build_background_mask(
        u_cube, sigma_u=np.full_like(u_cube, sig), order="linear"
    )
    # All four corners (where Earthshine sweeps by > 2,000 e-/s!) must NOT self-mask!
    for cr, cc in [(0, 0), (0, nx - 1), (ny - 1, 0), (ny - 1, nx - 1)]:
        assert res.mask[cr, cc], f"Sweeping Earthshine self-masked corner ({cr}, {cc})!"
    assert int(np.count_nonzero(res.mask)) >= int(0.88 * ny * nx)
    # Meanwhile the genuinely variable contaminant at (2, 3) MUST be rejected!
    assert not res.mask[2, 3]


def test_wcs_gaia_sources_catalogue_formats_and_missing_wcs_guard() -> None:
    rng = np.random.default_rng(404)
    n_cad, ny, nx = 24, 15, 15
    wcs = _make_tan_wcs(ny, nx)
    sky = np.full((n_cad, ny, nx), 85.0, dtype=np.float64) + rng.normal(
        0.0, 0.4, (n_cad, ny, nx)
    )

    # Place two catalogue stars at pixel coordinates (3.0, 11.0) and (11.0, 4.0)
    ra1, dec1 = wcs.pixel_to_world_values(11.0, 3.0)
    ra2, dec2 = wcs.pixel_to_world_values(4.0, 11.0)
    ra_faint, dec_faint = wcs.pixel_to_world_values(7.0, 7.0)

    cat_table = Table(
        {
            "ra": [float(ra1), float(ra2), float(ra_faint)],
            "dec": [float(dec1), float(dec2), float(dec_faint)],
            "phot_g_mean_mag": [11.0, 14.5, 21.5],  # 21.5 > mag_limit=19 -> ignored
        }
    )
    tpf = _build_custom_tpf(
        sky, flux_err=np.full_like(sky, 0.4), wcs=wcs, bkgsub=False
    )
    clean_tpf, diag = subtract_background(
        tpf, order=1, gaia_sources=cat_table, return_diagnostics=True
    )
    assert diag.source_mask[3, 11]
    assert diag.source_mask[11, 4]
    assert not diag.source_mask[7, 7]
    assert not diag.bkg_mask[3, 11]
    assert not diag.bkg_mask[11, 4]
    pq_data = clean_tpf.hdu["PIXEL_QUALITY"].data
    assert (pq_data[3, 11] & int(PixelQuality.SOURCE_MASK)) != 0

    # Mapping and (N, 3) array inputs produce identical source masks
    est_dict, diag_dict = estimate_background(
        tpf,
        order=1,
        gaia_sources={
            "RA": [float(ra1), float(ra2)],
            "DEC": [float(dec1), float(dec2)],
            "Tmag": [11.0, 14.5],
        },
        return_diagnostics=True,
    )
    assert isinstance(est_dict, BackgroundEstimate)
    assert np.array_equal(diag_dict.source_mask, diag.source_mask)

    arr_sources = np.array(
        [[float(ra1), float(dec1), 11.0], [float(ra2), float(dec2), 14.5]]
    )
    _, diag_arr = estimate_background(
        tpf, order=1, gaia_sources=arr_sources, return_diagnostics=True
    )
    assert np.array_equal(diag_arr.source_mask, diag.source_mask)

    # Missing WCS when gaia_sources is non-empty must raise ValueError
    tpf_no_wcs = _build_custom_tpf(sky, bkgsub=False, wcs=None)
    with pytest.raises(ValueError, match="celestial WCS"):
        subtract_background(tpf_no_wcs, gaia_sources=arr_sources)

    # MaskedColumn (with large fill_value / integer masked entries) + QTable units
    # (radians, hourangle, ABmag) must unwrap masks and convert angular units!
    ra_bogus, dec_bogus = wcs.pixel_to_world_values(13.0, 13.0)
    masked_cat = Table(
        [
            MaskedColumn(
                [float(ra1), float(ra2), float(ra_bogus)],
                mask=[False, False, True],
                fill_value=float(ra_bogus),
                unit=u.deg,
            ),
            MaskedColumn(
                [float(dec1), float(dec2), float(dec_bogus)],
                mask=[False, False, True],
                fill_value=float(dec_bogus),
                unit=u.deg,
            ),
            MaskedColumn([11.0, 14.5, 8.0], mask=[False, False, False]),
        ],
        names=("ra", "dec", "mag"),
    )
    mask_from_masked = build_wcs_source_mask(
        ny, nx, wcs, masked_cat, sky_sigma=0.4
    )
    assert not mask_from_masked[13, 13]
    assert np.array_equal(mask_from_masked, diag.source_mask)

    qtab = QTable(
        {
            "ra": ([float(ra1), float(ra2)] * u.deg).to(u.rad),
            "dec": ([float(dec1), float(dec2)] * u.deg).to(u.rad),
            "mag": [11.0, 14.5] * u.ABmag,
        }
    )
    mask_from_qtab = build_wcs_source_mask(ny, nx, wcs, qtab, sky_sigma=0.4)
    assert np.array_equal(mask_from_qtab, diag.source_mask)

    dict_ha = {
        "ra": ([float(ra1), float(ra2)] * u.deg).to(u.hourangle),
        "dec": ([float(dec1), float(dec2)] * u.deg).to(u.arcmin),
        "tmag": [11.0, 14.5] * u.mag,
    }
    mask_from_ha = build_wcs_source_mask(ny, nx, wcs, dict_ha, sky_sigma=0.4)
    assert np.array_equal(mask_from_ha, diag.source_mask)

    # Swapped (DEC, RA) 2D WCS and 3D (RA, DEC, TIME) WCS must project via wcs.celestial
    wcs_swapped = WCS(naxis=2)
    wcs_swapped.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]
    wcs_swapped.wcs.cdelt = [21.0 / 3600.0, -21.0 / 3600.0]
    wcs_swapped.wcs.crval = [-30.0, 210.0]
    wcs_swapped.wcs.ctype = ["DEC--TAN", "RA---TAN"]
    s_ra1, s_dec1 = (
        wcs_swapped.celestial.pixel_to_world(11.0, 3.0).ra.deg,
        wcs_swapped.celestial.pixel_to_world(11.0, 3.0).dec.deg,
    )
    mask_swapped = build_wcs_source_mask(
        ny,
        nx,
        wcs_swapped,
        {"ra": [s_ra1], "dec": [s_dec1], "mag": [13.0]},
        sky_sigma=0.4,
    )
    assert mask_swapped[3, 11] and not mask_swapped[11, 3]

    wcs_3d = WCS(naxis=3)
    wcs_3d.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0, 1.0]
    wcs_3d.wcs.cdelt = [-21.0 / 3600.0, 21.0 / 3600.0, 120.0]
    wcs_3d.wcs.crval = [120.0, -30.0, 0.0]
    wcs_3d.wcs.ctype = ["RA---TAN", "DEC--TAN", "TIME"]
    mask_3d = build_wcs_source_mask(
        ny, nx, wcs_3d, arr_sources, sky_sigma=0.4
    )
    assert np.array_equal(mask_3d, diag.source_mask)

    # Bright-star (T = 4.5 and T = 8.0) targets on 11x11 SPOC TPF and 50x50 FFI cutout:
    # must NEVER wipe out 100% of pixels, relying on PIXEL_QUALITY + data masking!
    ny11, nx11 = 11, 11
    wcs11 = _make_tan_wcs(ny11, nx11)
    eta11, xi11 = normalized_grid(ny11, nx11)
    e11, x11 = np.meshgrid(eta11, xi11, indexing="ij")
    sky11 = (120.0 + 15.0 * x11 - 10.0 * e11)[None, :, :] + rng.normal(
        0.0, 0.6, (20, ny11, nx11)
    )
    rr11, cc11 = np.meshgrid(np.arange(ny11), np.arange(nx11), indexing="ij")
    r2_11 = (rr11 - 5.0) ** 2 + (cc11 - 5.0) ** 2
    halo11 = 1.5e5 * (1.0 + r2_11 / (0.85**2)) ** (-2.2)
    halo11[:, 5] += 2.0e4 * np.exp(-0.25 * np.abs(rr11[:, 5] - 5.0))
    pq11 = np.zeros((ny11, nx11), dtype=np.uint16)
    pq11[:, 5] |= np.uint16(PixelQuality.SAT_BLEED)
    ra_c11, dec_c11 = wcs11.pixel_to_world_values(5.0, 5.0)
    tpf_bright11 = _build_custom_tpf(
        sky11 + halo11[None, :, :],
        flux_err=np.sqrt((sky11 + halo11[None, :, :]) / 100.0 + 0.36),
        pixel_quality=pq11,
        wcs=wcs11,
        bkgsub=False,
    )
    for tmag_bright in (4.5, 8.0):
        sub_b11, diag_b11 = subtract_background(
            tpf_bright11,
            gaia_sources={
                "ra": [float(ra_c11)],
                "dec": [float(dec_c11)],
                "tmag": [tmag_bright],
            },
            return_diagnostics=True,
        )
        assert diag_b11.effective_order == 1
        assert int(np.count_nonzero(diag_b11.bkg_mask)) >= 15
        assert np.all(np.isfinite(sub_b11.flux_bkg.to_value("electron/s")))

    ny50, nx50 = 50, 50
    wcs50 = _make_tan_wcs(ny50, nx50)
    eta50, xi50 = normalized_grid(ny50, nx50)
    e50, x50 = np.meshgrid(eta50, xi50, indexing="ij")
    sky50_true = 200.0 + 25.0 * x50 - 18.0 * e50 + 12.0 * (2.0 * x50**2 - 1.0)
    rr50, cc50 = np.meshgrid(np.arange(ny50), np.arange(nx50), indexing="ij")
    r2_50 = (rr50 - 25.0) ** 2 + (cc50 - 25.0) ** 2
    halo50 = 9.0e5 * (1.0 + r2_50 / (0.85**2)) ** (-2.2)
    pq50 = np.zeros((ny50, nx50), dtype=np.uint16)
    pq50[:, 24:27] |= np.uint16(PixelQuality.SAT_BLEED)
    u50 = (
        sky50_true[None, :, :]
        + halo50[None, :, :]
        + rng.normal(0.0, 0.8, (16, ny50, nx50))
    )
    ra_c50, dec_c50 = wcs50.pixel_to_world_values(25.0, 25.0)
    tpf_bright50 = _build_custom_tpf(
        u50,
        flux_err=np.sqrt(np.maximum(u50, 1.0) / 120.0 + 0.64),
        pixel_quality=pq50,
        wcs=wcs50,
        bkgsub=False,
    )
    sub_b50, diag_b50 = subtract_background(
        tpf_bright50,
        gaia_sources={
            "ra": [float(ra_c50)],
            "dec": [float(dec_c50)],
            "tmag": [5.5],
        },
        return_diagnostics=True,
    )
    assert diag_b50.effective_order == 2
    np.testing.assert_allclose(
        np.median(sub_b50.flux_bkg.to_value("electron/s"), axis=0),
        sky50_true,
        rtol=2e-2,
        atol=1.5,
    )


def test_curvature_and_rank_guards_auto_downgrade() -> None:
    ny, nx = 20, 20

    # < 3 pixels -> ValueError
    m2 = np.zeros((ny, nx), dtype=bool)
    m2[0, 0] = True
    m2[19, 19] = True
    with pytest.raises(ValueError, match="Insufficient finite background pixels"):
        apply_curvature_and_rank_guards(m2, 1, allow_downgrade=True)

    # 3 <= |Omega_bg| < 10 -> downgrades to order=0 pedestal
    m6 = np.zeros((ny, nx), dtype=bool)
    for r, c in [(0, 0), (0, 19), (19, 0), (19, 19), (10, 5), (5, 14)]:
        m6[r, c] = True
    eff_6, kap_6 = apply_curvature_and_rank_guards(m6, 3, allow_downgrade=True)
    assert eff_6 == 0 and kap_6 == 1.0

    # 10 <= |Omega_bg| < 30 well-spread -> downgrades 3 -> 2 -> 1 (linear)
    rng = np.random.default_rng(505)
    pts_20 = np.zeros((ny, nx), dtype=bool)
    chosen = rng.choice(ny * nx, size=20, replace=False)
    pts_20.ravel()[chosen] = True
    eff_20, kap_20 = apply_curvature_and_rank_guards(pts_20, 3, allow_downgrade=True)
    assert eff_20 == 1 and kap_20 <= 100.0

    # 30 <= |Omega_bg| < 50 well-spread -> downgrades 3 -> 2 (quadratic)
    pts_38 = np.zeros((ny, nx), dtype=bool)
    pts_38.ravel()[rng.choice(ny * nx, size=38, replace=False)] = True
    eff_38, kap_38 = apply_curvature_and_rank_guards(pts_38, 3, allow_downgrade=True)
    assert eff_38 == 2 and kap_38 <= 100.0

    # >= 50 pixels clustered in a narrow 2-column strip (kappa > 100 for high orders,
    # and collinear single column kappa == inf -> downgrades all the way to 0!)
    col_only = np.zeros((60, 60), dtype=bool)
    col_only[:, 0] = True  # 60 pixels, all on c=0 -> rank-deficient in xi!
    eff_col, _ = apply_curvature_and_rank_guards(col_only, 2, allow_downgrade=True)
    assert eff_col == 0


def test_cosmic_rays_asteroids_flux_err_absent_and_partial_nan_cadences() -> None:
    rng = np.random.default_rng(606)
    n_cad, ny, nx = 40, 12, 13
    eta, xi = normalized_grid(ny, nx)
    eta_g, xi_g = np.meshgrid(eta, xi, indexing="ij")

    sky_true = (
        55.0 + 8.0 * xi_g - 5.0 * eta_g
    )[None, :, :] * np.ones((n_cad, 1, 1), dtype=np.float64)

    # 1. 20-s cosmic rays (50-sigma and 120-sigma) + asteroid streak in Omega_bg:
    #    Tukey biweight (c_B = 4.685) must completely reject >= 50-sigma hits!
    sig = 0.5
    u_cr = sky_true + rng.normal(0.0, sig, sky_true.shape)
    u_clean = u_cr.copy()
    # Inject 6 extreme cosmic rays (50*sigma = +25 e-/s, 120*sigma = +60 e-/s)
    # near the corner on cadence 5, plus an asteroid trail across 4 pixels on cad 12
    cr_coords = [(0, 0), (0, 1), (1, 0), (0, 2), (2, 0), (1, 1)]
    for idx, (r, c) in enumerate(cr_coords):
        u_cr[5, r, c] += 25.0 if idx < 4 else 60.0
    for c_ast in range(8, 12):
        u_cr[12, 10, c_ast] += 30.0

    all_sky_mask = np.ones((ny, nx), dtype=bool)
    tpf_clean = _build_custom_tpf(u_clean, flux_err=np.full_like(u_clean, sig))
    tpf_cr = _build_custom_tpf(u_cr, flux_err=np.full_like(u_cr, sig))

    unhit_mask = all_sky_mask.copy()
    for r, c in cr_coords:
        unhit_mask[r, c] = False
    est_oracle_unhit = estimate_background(tpf_clean, order=1, mask=unhit_mask)
    est_oracle = estimate_background(tpf_clean, order=1, mask=all_sky_mask)
    est_robust, diag_cr = estimate_background(
        tpf_cr, order=1, mask=all_sky_mask, return_diagnostics=True
    )

    # Compare against non-robust OLS bias on cadence 5:
    phi_full, phi_bg, _ = equilibrate_chebyshev_basis(ny, nx, 1, all_sky_mask)
    beta_ols = phi_bg.T @ u_cr[5].ravel()
    bkg_ols_5 = np.tensordot(beta_ols, phi_full, axes=(0, 0))
    ols_max_bias = float(np.max(np.abs(bkg_ols_5 - sky_true[5])))
    rob_max_diff_vs_unhit = float(
        np.max(
            np.abs(
                est_robust.bkg.to_value("electron/s")[5]
                - est_oracle_unhit.bkg.to_value("electron/s")[5]
            )
        )
    )
    rob_max_diff_vs_oracle = float(
        np.max(
            np.abs(
                est_robust.bkg.to_value("electron/s")[5]
                - est_oracle.bkg.to_value("electron/s")[5]
            )
        )
    )
    assert ols_max_bias > 2.5
    assert rob_max_diff_vs_unhit < 0.05
    assert rob_max_diff_vs_oracle < 0.02 * ols_max_bias
    assert rob_max_diff_vs_oracle < 0.04
    assert diag_cr.n_tukey_inliers[5] == ny * nx - len(cr_coords)

    # 2. FLUX_ERR-absent input with 0.3 e-/s noise: verify tau_scale ~ 0.3 and
    #    FLUX_BKG_ERR is NOT inflated by a 1.0 clamp!
    sig_small = 0.3
    u_noerr = sky_true + rng.normal(0.0, sig_small, sky_true.shape)
    tpf_noerr = _build_custom_tpf(u_noerr, flux_err=None, bkgsub=False)
    clean_noerr, diag_noerr = subtract_background(
        tpf_noerr, order=1, mask=all_sky_mask, return_diagnostics=True
    )
    med_tau = float(np.median(diag_noerr.tau_scale))
    assert 0.24 < med_tau < 0.36
    # Mean marginal background error across pixels is sqrt(P / M) * sigma:
    # sigma=0.3, P=3, M=156 -> ~0.0416 e-/s (whereas a 1.0 clamp gives ~0.139!)
    rms_bkg_err = float(
        np.sqrt(np.mean(clean_noerr.hdu[1].data["FLUX_BKG_ERR"] ** 2))
    )
    expected_rms = sig_small * np.sqrt(3.0 / (ny * nx))
    assert abs(rms_bkg_err - expected_rms) < 0.012
    assert rms_bkg_err < 0.06
    # Uncalibrated input (absent FLUX_ERR) must keep FLUX_ERR == NaN rather than
    # fabricating finite FLUX_ERR == FLUX_BKG_ERR!
    assert np.all(np.isnan(clean_noerr.hdu[1].data["FLUX_ERR"]))
    assert np.all(np.isfinite(clean_noerr.hdu[1].data["FLUX_BKG_ERR"]))
    assert np.all(np.isfinite(clean_noerr.hdu[1].data["BKG_CHOL"]))

    # 3. Single-pixel NaN vs < P valid pixels vs !quality_mask row updates:
    u_nan = u_cr.copy()
    u_nan[2, 4, 5] = np.nan  # 1 NaN pixel -> must succeed cleanly (NEVER 0.0!)
    u_nan[3, :, :] = np.nan
    u_nan[3, 0, 0] = 55.0
    u_nan[3, 11, 12] = 55.0  # only 2 < P=3 valid pixels -> must return NaN (NEVER 0.0!)
    u_nan[4, :, :] = np.nan  # 0 valid pixels -> must return NaN (NEVER 0.0!)
    qual = np.zeros(n_cad, dtype=np.int32)
    qual[7] = 128  # !quality_mask flagged cadence

    tpf_nan = _build_custom_tpf(
        u_nan,
        flux_err=np.full_like(u_nan, sig),
        flux_bkg=np.full_like(u_nan, 999.0),  # stale background on raw rows
        quality=qual,
        bkgsub=False,
    )
    sub_nan, diag_nan = subtract_background(
        tpf_nan, order=1, mask=all_sky_mask, return_diagnostics=True
    )
    raw_bkg_hdu = sub_nan.hdu[1].data["FLUX_BKG"]
    raw_flux_hdu = sub_nan.hdu[1].data["FLUX"]
    raw_chol_hdu = sub_nan.hdu[1].data["BKG_CHOL"]

    # Cadence 2 (1 NaN pixel): background reconstructed across ALL pixels, non-zero!
    assert np.all(np.isfinite(raw_bkg_hdu[2]))
    np.testing.assert_allclose(raw_bkg_hdu[2], sky_true[2], rtol=1e-2, atol=0.25)
    assert np.isnan(raw_flux_hdu[2, 4, 5])

    # Cadences 3 (< P valid) and 4 (0 valid): strictly NaN, NEVER 0.0!
    for bad_t in (3, 4):
        assert np.all(np.isnan(raw_bkg_hdu[bad_t]))
        assert np.all(np.isnan(raw_flux_hdu[bad_t]))
        assert np.all(np.isnan(raw_chol_hdu[bad_t]))
        assert not np.any(raw_bkg_hdu[bad_t] == 0.0)

    # Flagged !quality_mask row 7 must ALSO be updated (not left at stale 999.0)!
    assert diag_nan.n_valid_pixels[7] == ny * nx
    np.testing.assert_allclose(raw_bkg_hdu[7], sky_true[7], rtol=1e-2, atol=0.25)


def test_tukey_rank_collapse_fallback_to_convex_huber() -> None:
    # 1. Count guard (M_I < max(2*P, P+1) = 6 for order=1, P=3) with full-rank
    #    Huber quadratic core: M = 6 sky pixels (4 clean corners + 1 interior clean
    #    pixel + 1 interior +50 e-/s positive cosmic ray -> M_I = 5 < 6 -> Huber
    #    fallback while G_H over the 5 clean core pixels remains well-conditioned!).
    ny, nx = 6, 6
    mask = np.zeros((ny, nx), dtype=bool)
    clean_coords = [(0, 0), (0, 5), (5, 0), (5, 5), (2, 3)]
    outlier_coord = (3, 2)
    for r, c in [*clean_coords, outlier_coord]:
        mask[r, c] = True

    eta, xi = normalized_grid(ny, nx)
    eta_g, xi_g = np.meshgrid(eta, xi, indexing="ij")
    plane = 100.0 + 6.0 * xi_g - 4.0 * eta_g

    u_cube = np.repeat(plane[None, :, :], 4, axis=0)
    u_cube[1, outlier_coord[0], outlier_coord[1]] += 50.0

    tpf1 = _build_custom_tpf(u_cube, flux_err=np.full_like(u_cube, 0.5), bkgsub=False)
    clean1, diag1 = subtract_background(
        tpf1, order=1, mask=mask, allow_downgrade=False, return_diagnostics=True
    )
    assert bool(diag1.used_huber_fallback[1]) is True
    assert bool(diag1.used_huber_fallback[0]) is False
    assert np.all(np.isfinite(diag1.coeffs[1]))
    assert np.all(np.isfinite(diag1.bkg_chol[1]))
    np.testing.assert_allclose(
        clean1.flux_bkg.to_value("electron/s")[1], plane, rtol=1e-2, atol=1.0
    )

    # Also verify directional variance inflation in G_H^{-1} when a positive
    # cosmic ray strikes corner (0, 5) while its mirror corner (5, 5) stays clean:
    # unweighted gram0 is mirror-symmetric across r = 2.5 (giving equal errors at
    # (0, 5) and (5, 5)), whereas G_H excludes the (0, 5) outlier and inflates
    # FLUX_BKG_ERR specifically at the unanchored (0, 5) corner!
    mask_edge = np.zeros((ny, nx), dtype=bool)
    for r, c in [(0, 0), (5, 0), (2, 2), (3, 2), (0, 5), (5, 5)]:
        mask_edge[r, c] = True
    u_edge = np.repeat(plane[None, :, :], 3, axis=0)
    u_edge[1, 0, 5] += 80.0
    tpf_edge = _build_custom_tpf(
        u_edge, flux_err=np.full_like(u_edge, 0.5), bkgsub=False
    )
    clean_edge, diag_edge = subtract_background(
        tpf_edge,
        order=1,
        mask=mask_edge,
        allow_downgrade=False,
        return_diagnostics=True,
    )
    assert bool(diag_edge.used_huber_fallback[1]) is True
    be_edge_hit = clean_edge.flux_bkg_err.to_value("electron/s")[1]
    be_edge_clean = clean_edge.flux_bkg_err.to_value("electron/s")[0]
    assert abs(float(be_edge_clean[0, 5]) - float(be_edge_clean[5, 5])) < 1e-5
    assert float(be_edge_hit[0, 5]) > 1.35 * float(be_edge_hit[5, 5])

    # 2. Rank-deficient Huber core breakdown: all 7 clean pixels lie strictly on
    #    row 0 while the ONLY 2 off-row pixels on row 7 are struck by strictly
    #    positive cosmic rays (+500 and +150 e-/s), leaving 0 off-row pixels in
    #    |u_h| <= c_H (rank(G_H) == 2 < 3) -> MUST return NaN on cadence 1!
    mask_col = np.zeros((8, 8), dtype=bool)
    mask_col[0, :7] = True
    mask_col[7, 2] = True
    mask_col[7, 5] = True
    eta8, xi8 = normalized_grid(8, 8)
    e8, x8 = np.meshgrid(eta8, xi8, indexing="ij")
    plane8 = 80.0 + 3.0 * x8 - 2.0 * e8
    u_col = np.repeat(plane8[None, :, :], 3, axis=0)
    u_col[1, 7, 2] += 500.0
    u_col[1, 7, 5] += 150.0

    tpf2 = _build_custom_tpf(u_col, flux_err=np.full_like(u_col, 0.5), bkgsub=False)
    clean2, diag2 = subtract_background(
        tpf2, order=1, mask=mask_col, allow_downgrade=False, return_diagnostics=True
    )
    # Clean cadence 0 succeeds and matches plane8; rank-collapsed cadence 1 is NaN!
    np.testing.assert_allclose(
        clean2.flux_bkg.to_value("electron/s")[0], plane8, rtol=1e-4, atol=1e-4
    )
    assert bool(diag2.used_huber_fallback[1]) is False
    assert np.all(np.isnan(diag2.coeffs[1]))
    assert np.all(np.isnan(diag2.bkg_chol[1]))
    assert np.all(np.isnan(clean2.hdu[1].data["FLUX_BKG"][1]))
    assert np.all(np.isnan(clean2.hdu[1].data["FLUX_BKG_ERR"][1]))
    chol_unpacked = unpack_bkg_chol(clean2.hdu[1].data["BKG_CHOL"])
    assert np.all(np.isnan(chol_unpacked[1]))


def test_idempotency_non_mutation_and_chunked_gemm_equivalence() -> None:
    # Verify idempotency across SPOC, TessCut, and mundey_v2 TPFs:
    # subtract_background(subtract_background(tpf)) must match single pass!
    for mode in ("spoc", "tesscut", "mundey_v2"):
        truth = make_synthetic_tpf(mode=mode, sky_order="linear", seed=77)
        orig_flux = truth.tpf.hdu[1].data["FLUX"].copy()
        orig_hdr0 = truth.tpf.hdu[0].header.copy()

        pass1, diag1 = subtract_background(truth.tpf, order=1, return_diagnostics=True)
        # Input TPF must never be mutated in place!
        np.testing.assert_array_equal(truth.tpf.hdu[1].data["FLUX"], orig_flux)
        assert list(truth.tpf.hdu[0].header.items()) == list(orig_hdr0.items())

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            pass2, diag2 = subtract_background(pass1, order=1, return_diagnostics=True)
        user_w = [w for w in rec if issubclass(w.category, UserWarning)]
        assert len(user_w) == 0
        assert read_bkgsub_state(pass1) is True
        assert diag2.was_bkg_subtracted is True
        assert np.array_equal(diag1.bkg_mask, diag2.bkg_mask)

        for col in ("FLUX", "FLUX_BKG", "FLUX_ERR", "FLUX_BKG_ERR", "BKG_CHOL"):
            np.testing.assert_allclose(
                pass2.hdu[1].data[col],
                pass1.hdu[1].data[col],
                rtol=2e-5,
                atol=2e-4,
            )

    # Verify idempotency for uncalibrated FLUX_ERR inputs (absent, 0.0, and -1.0 FFI
    # sentinel) AND high-tau calibrated cadences where sigma_B_hat > 15 * sigma_U:
    rng_id = np.random.default_rng(708)
    eta_i, xi_i = normalized_grid(11, 11)
    ei, xii = np.meshgrid(eta_i, xi_i, indexing="ij")
    u_base = (90.0 + 7.0 * xii - 5.0 * ei)[None, :, :] + rng_id.normal(
        0.0, 0.4, (20, 11, 11)
    )
    for fe_sentinel in (
        None,
        np.zeros_like(u_base),
        np.full_like(u_base, -1.0),
    ):
        tpf_uncal = _build_custom_tpf(u_base, flux_err=fe_sentinel, bkgsub=False)
        p1_u = subtract_background(tpf_uncal, order=1)
        p2_u = subtract_background(p1_u, order=1)
        assert np.all(np.isnan(p1_u.hdu[1].data["FLUX_ERR"]))
        assert np.all(np.isnan(p2_u.hdu[1].data["FLUX_ERR"]))
        assert np.all(np.isfinite(p1_u.hdu[1].data["FLUX_BKG_ERR"]))
        for col in ("FLUX", "FLUX_BKG", "FLUX_ERR", "FLUX_BKG_ERR", "BKG_CHOL"):
            np.testing.assert_allclose(
                p2_u.hdu[1].data[col],
                p1_u.hdu[1].data[col],
                rtol=2e-5,
                atol=2e-4,
                equal_nan=True,
            )

    # High-tau calibrated cadence (sigma_B_hat > 15 * sigma_U via strong quadratic
    # scattered-light dome fit with order=1 on heteroscedastic calibrated weights):
    sig_het = np.broadcast_to(0.12 + 0.08 * (xii + 1.0), (16, 11, 11)).copy()
    u_hightau = u_base[:16].copy()
    u_hightau[3] += 45.0 * (2.0 * xii**2 - 1.0)
    mask_all11 = np.ones((11, 11), dtype=bool)
    tpf_ht = _build_custom_tpf(u_hightau, flux_err=sig_het, bkgsub=False)
    p1_ht, d1_ht = subtract_background(
        tpf_ht, order=1, mask=mask_all11, return_diagnostics=True
    )
    assert float(np.max(p1_ht.hdu[1].data["FLUX_BKG_ERR"][3] / sig_het[3])) > 15.0
    assert d1_ht.tau_scale[3] > 50.0
    p2_ht = subtract_background(p1_ht, order=1, mask=mask_all11)
    # Tolerance matches IEEE-754 float32 quadrature storage roundoff
    # eps_f32 * (sigma_B / sigma_U)^2 ~ 1.7e-4 (~0.0024 e-/s on a 45 e-/s dome,
    # vs > 2.5 e-/s bias if sigma_U were clamped to 0.1 * FLUX_ERR!):
    for col in ("FLUX", "FLUX_BKG", "FLUX_ERR", "FLUX_BKG_ERR", "BKG_CHOL"):
        np.testing.assert_allclose(
            p2_ht.hdu[1].data[col],
            p1_ht.hdu[1].data[col],
            rtol=5e-4,
            atol=4e-3,
        )

    # Overwriting BKG_BASIS with a non-triangular P=4 custom basis (bkg_order=None)
    # must delete stale BKG_ORD from BKG_BASIS.header!
    assert "BKG_ORD" in p1_ht.hdu["BKG_BASIS"].header
    custom_basis_p4 = np.ones((4, 11, 11), dtype=np.float32)
    custom_chol_p4 = np.broadcast_to(np.eye(4, dtype=np.float32), (16, 4, 4))
    write_tessbkgd_provenance(
        p1_ht.hdu,
        bkg_basis=custom_basis_p4,
        bkg_chol=custom_chol_p4,
        bkg_used_mask=mask_all11,
        bkg_order=None,
    )
    assert "BKG_ORD" not in p1_ht.hdu["BKG_BASIS"].header

    # Verify chunked GEMM reconstruction matches full-batch GEMM bit-for-bit/ULP
    rng = np.random.default_rng(707)
    phi_full, _, _ = equilibrate_chebyshev_basis(
        10, 12, 2, np.ones((10, 12), dtype=bool)
    )
    n_c, p_d = 53, phi_full.shape[0]
    coeffs = rng.normal(0.0, 5.0, size=(n_c, p_d))
    coeffs[19] = np.nan
    chol = np.zeros((n_c, p_d, p_d), dtype=np.float64)
    rows, cols = np.tril_indices(p_d)
    chol[:, rows, cols] = rng.normal(0.0, 0.2, size=(n_c, rows.size))
    chol[:, np.arange(p_d), np.arange(p_d)] = np.abs(
        chol[:, np.arange(p_d), np.arange(p_d)]
    ) + 0.05
    chol[19] = np.nan

    b_full, e_full = reconstruct_background_chunked(
        coeffs, chol, phi_full, chunk_size=8192
    )
    b_chunk, e_chunk = reconstruct_background_chunked(
        coeffs, chol, phi_full, chunk_size=11
    )
    np.testing.assert_allclose(b_chunk, b_full, rtol=0.0, atol=1e-13, equal_nan=True)
    np.testing.assert_allclose(e_chunk, e_full, rtol=0.0, atol=1e-13, equal_nan=True)


def test_monte_carlo_sandwich_covariance_calibration() -> None:
    """Verify empirical B_hat scatter matches FLUX_BKG_ERR & BKG_CHOL (<10%)."""
    rng = np.random.default_rng(20261112)
    n_mc = 2048
    ny, nx = 12, 13
    mask = np.ones((ny, nx), dtype=bool)
    mask[4:8, 5:8] = False  # central star aperture excluded (Omega_halo)
    phi_full, phi_bg, _ = equilibrate_chebyshev_basis(ny, nx, 1, mask)

    eta, xi = normalized_grid(ny, nx)
    eta_g, xi_g = np.meshgrid(eta, xi, indexing="ij")
    b_true = 150.0 + 14.0 * xi_g - 9.0 * eta_g
    b_true_bg = b_true[mask]

    # Heteroscedastic pixel uncertainties across the stamp (0.8 to 2.2 e-/s)
    sigma_img = 0.8 + 0.7 * (xi_g + 1.0)
    sigma_bg = np.broadcast_to(sigma_img[mask], (n_mc, int(np.count_nonzero(mask))))

    u_mc = b_true_bg[None, :] + rng.normal(0.0, 1.0, size=sigma_bg.shape) * sigma_bg
    # Inject 2% 40-sigma cosmic rays randomly across background pixels to test
    # sandwich covariance calibration IN THE PRESENCE OF OUTLIERS!
    cr_hits = rng.random(sigma_bg.shape) < 0.02
    u_mc = np.where(cr_hits, u_mc + 40.0 * sigma_bg, u_mc)

    # 1. Calibrated heteroscedastic FLUX_ERR case (chunk_size=128 reuses cached kernel)
    sol_w = solve_chebyshev_background(
        u_mc, phi_bg, sigma_u_bg=sigma_bg, chunk_size=128
    )
    b_hat_w, b_err_w = reconstruct_background_chunked(
        sol_w.coeffs, sol_w.bkg_chol, phi_full, chunk_size=512
    )

    emp_std_px = np.std(b_hat_w - b_true[None, :, :], axis=0, ddof=1)
    pred_std_px = np.sqrt(np.mean(b_err_w**2, axis=0))
    rel_err_px = np.abs(pred_std_px - emp_std_px) / emp_std_px
    assert float(np.max(rel_err_px)) < 0.08
    assert float(np.median(rel_err_px)) < 0.04

    # Also verify full joint covariance L_t on a coherent halo aperture sum:
    # Var(sum_{i in halo} w_i B_hat_i) == || L_t.T @ (sum_{i in halo} w_i phi_i) ||_2^2
    halo_mask = ~mask
    phi_halo_sum = np.sum(phi_full[:, halo_mask], axis=1)  # (P,)
    halo_bkg_sum_hat = np.sum(b_hat_w[:, halo_mask], axis=1)
    halo_bkg_sum_true = float(np.sum(b_true[halo_mask]))
    emp_halo_std = float(np.std(halo_bkg_sum_hat - halo_bkg_sum_true, ddof=1))
    proj = np.einsum("tpq,p->tq", sol_w.bkg_chol, phi_halo_sum)
    pred_halo_std = float(np.sqrt(np.mean(np.sum(proj**2, axis=1))))
    assert abs(pred_halo_std - emp_halo_std) / emp_halo_std < 0.08

    # 2. Unweighted (FLUX_ERR-absent) homoscedastic 0.35 e-/s noise calibration:
    sig_homo = 0.35
    u_unw = b_true_bg[None, :] + rng.normal(0.0, sig_homo, size=sigma_bg.shape)
    sol_unw = solve_chebyshev_background(
        u_unw, phi_bg, sigma_u_bg=None, chunk_size=128
    )
    b_hat_u, b_err_u = reconstruct_background_chunked(
        sol_unw.coeffs, sol_unw.bkg_chol, phi_full, chunk_size=512
    )
    emp_std_u = np.std(b_hat_u - b_true[None, :, :], axis=0, ddof=1)
    pred_std_u = np.sqrt(np.mean(b_err_u**2, axis=0))
    assert float(np.max(np.abs(pred_std_u - emp_std_u) / emp_std_u)) < 0.08

    # 3. End-to-end TPF roundtrip check of FLUX_ERR propagation & packed BKG_CHOL:
    full_cube = b_true[None, :, :] + rng.normal(0.0, 1.0, (64, ny, nx)) * sigma_img
    tpf_mc = _build_custom_tpf(
        full_cube,
        flux_err=np.broadcast_to(sigma_img, (64, ny, nx)),
        bkgsub=False,
    )
    sub_mc = subtract_background(tpf_mc, order=1, mask=mask)
    fe_out = sub_mc.flux_err.to_value("electron/s")
    be_out = sub_mc.flux_bkg_err.to_value("electron/s")
    np.testing.assert_allclose(
        fe_out, np.sqrt(sigma_img[None, :, :] ** 2 + be_out**2), rtol=1e-5, atol=1e-5
    )
    chol_hdu = unpack_bkg_chol(sub_mc.hdu[1].data["BKG_CHOL"]).astype(np.float64)
    # Check lower-triangular positive diagonal & agreement with FLUX_BKG_ERR
    assert np.all(np.diagonal(chol_hdu, axis1=1, axis2=2) > 0.0)
    _, be_from_chol = reconstruct_background_chunked(
        np.zeros((64, 3)), chol_hdu, sub_mc.hdu["BKG_BASIS"].data.astype(np.float64)
    )
    np.testing.assert_allclose(be_from_chol, be_out, rtol=1e-5, atol=1e-5)
    assert tessbkgd.__version__.startswith("2.0")
    assert sub_mc.flux.unit == u.Unit("electron/s")
