"""Synthetic TESS TargetPixelFile fixtures for the Phase 1 test suite."""

from __future__ import annotations

from dataclasses import dataclass

import lightkurve as lk
import numpy as np
import pytest
from astropy.io import fits


@dataclass(frozen=True)
class SyntheticTPFTruth:
    """Ground-truth components of a synthetic TESS TargetPixelFile."""

    tpf: lk.TessTargetPixelFile
    sky_raw: np.ndarray
    star_raw: np.ndarray
    good_mask: np.ndarray
    sky_pixel_mask: np.ndarray


def make_synthetic_tpf(
    *,
    n_raw: int = 40,
    ny: int = 12,
    nx: int = 14,
    mode: str = "spoc",
    sky_order: str = "linear",
    sky_base: float = 120.0,
    sky_grad_x: float = 4.5,
    sky_grad_y: float = -3.5,
    psf_sigma: float = 0.16,
    star_peak: float = 2500.0,
    halo_pedestal: float = 0.0,
    include_tunit: bool = True,
    flux_unit_override: str | None = None,
    quality_bad_indices: tuple[int, ...] = (3, 19),
    seed: int = 42,
) -> SyntheticTPFTruth:
    """Construct an in-memory ``lightkurve.TessTargetPixelFile`` with known truth.

    Parameters
    ----------
    mode : {"spoc", "tesscut", "mundey_v2", "mundey_v2_stale_bkg", "mundey_v01"}
        Provenance configuration:
        - ``"spoc"``: ``BKGSUB`` unset, ``FLUX = star + sky - spoc_bkg``,
          ``FLUX_BKG = spoc_bkg > 0``, ``TUNIT5 = 'e-/s'``.
        - ``"tesscut"``: ``BKGSUB`` unset, ``FLUX = star + sky``,
          ``FLUX_BKG = 0``, no ``TUNIT`` header cards.
        - ``"mundey_v2"``: ``BKGSUB = False``, ``FLUX = star + sky``,
          ``FLUX_BKG = 0``.
        - ``"mundey_v2_stale_bkg"``: ``BKGSUB = False``, ``FLUX = star + sky``,
          non-zero stale ``FLUX_BKG > 0`` (verifying ``BKGSUB = False`` prevents
          adding ``oldbkgd``).
        - ``"mundey_v01"``: ``BKGSUB`` unset, unsubtracted ``FLUX = star + sky``
          with stale ``FLUX_BKG > 0`` left in ``hdu[1]`` (triggers legacy
          ``UserWarning``).
    """
    rng = np.random.default_rng(seed)

    time = np.linspace(1468.0, 1495.0, n_raw, dtype=np.float64)
    # Introduce a mid-sector downlink gap in timestamps.
    time[n_raw // 2 :] += 1.2
    cadenceno = np.arange(1000, 1000 + n_raw, dtype=np.int32)
    timecorr = np.full(n_raw, 1.5e-3, dtype=np.float32)

    quality = np.zeros(n_raw, dtype=np.int32)
    for bad_idx in quality_bad_indices:
        if 0 <= bad_idx < n_raw:
            quality[bad_idx] = 128  # spacecraft attitude tweak / coarse point
    good_mask = quality == 0

    eta = np.linspace(-1.0, 1.0, ny, dtype=np.float64)
    xi = np.linspace(-1.0, 1.0, nx, dtype=np.float64)
    eta_g, xi_g = np.meshgrid(eta, xi, indexing="ij")

    # Smooth time-varying Earthshine modulation + spatial polynomial gradient.
    phase = np.linspace(0.0, 2.0 * np.pi, n_raw, dtype=np.float64)
    c0 = sky_base * (1.0 + 0.25 * np.sin(phase) + 0.10 * np.cos(2.0 * phase))
    cx = sky_grad_x * (1.0 + 0.2 * np.cos(phase))
    cy = sky_grad_y * (1.0 + 0.2 * np.sin(phase))

    sky_raw = (
        c0[:, None, None]
        + cy[:, None, None] * eta_g[None, :, :]
        + cx[:, None, None] * xi_g[None, :, :]
    )
    if sky_order == "quadratic":
        cxx = 2.2 * (1.0 + 0.1 * np.sin(phase))
        cxy = -1.4 * np.ones(n_raw, dtype=np.float64)
        cyy = 1.8 * (1.0 + 0.1 * np.cos(phase))
        sky_raw = (
            sky_raw
            + cyy[:, None, None] * (eta_g**2)[None, :, :]
            + cxy[:, None, None] * (eta_g * xi_g)[None, :, :]
            + cxx[:, None, None] * (xi_g**2)[None, :, :]
        )

    # Smooth, continuous (un-truncated) central stellar PSF + optional extended halo.
    r2 = (eta_g / psf_sigma) ** 2 + (xi_g / psf_sigma) ** 2
    psf = halo_pedestal + star_peak * np.exp(-0.5 * r2)
    pulsation = 1.0 + 0.015 * np.sin(7.0 * phase)
    star_raw = pulsation[:, None, None] * psf[None, :, :]
    sky_pixel_mask = psf < 1e-3

    total_flux = star_raw + sky_raw
    spoc_bkg = sky_raw + rng.normal(0.0, 0.05, size=sky_raw.shape)

    if mode == "spoc":
        flux_col_data = total_flux - spoc_bkg
        bkg_col_data = spoc_bkg
        bkgsub_card: bool | None = None
    elif mode == "tesscut":
        flux_col_data = total_flux
        bkg_col_data = np.zeros_like(total_flux)
        bkgsub_card = None
        include_tunit = False
    elif mode == "mundey_v2":
        flux_col_data = total_flux
        bkg_col_data = np.zeros_like(total_flux)
        bkgsub_card = False
    elif mode == "mundey_v2_stale_bkg":
        flux_col_data = total_flux
        bkg_col_data = spoc_bkg
        bkgsub_card = False
    elif mode == "mundey_v01":
        flux_col_data = total_flux
        bkg_col_data = spoc_bkg
        bkgsub_card = None
    else:
        raise ValueError(f"Unknown synthetic TPF mode {mode!r}.")

    flux_err_data = np.sqrt(np.maximum(total_flux, 1.0)) * 0.1
    bkg_err_data = np.full_like(total_flux, 0.2)
    raw_cnts_data = np.rint(np.maximum(total_flux, 0.0) * 120.0).astype(np.int32)
    pos_corr1 = np.zeros(n_raw, dtype=np.float32)
    pos_corr2 = np.zeros(n_raw, dtype=np.float32)

    primary_hdr = fits.Header()
    primary_hdr["TELESCOP"] = "TESS"
    primary_hdr["INSTRUME"] = "TESS Photometer"
    if mode == "tesscut":
        primary_hdr["ORIGIN"] = "STScI/MAST"
        primary_hdr["CREATOR"] = "TESScut"
    else:
        primary_hdr["ORIGIN"] = "NASA/Ames"
        primary_hdr["CREATOR"] = "8012 TargetPixelExporterPipelineModule"
    primary_hdr["OBJECT"] = "TIC 999999999"
    primary_hdr["TICID"] = 999999999
    primary_hdr["SECTOR"] = 6
    primary_hdr["CAMERA"] = 1
    primary_hdr["CCD"] = 2
    primary_hdr["RA_OBJ"] = 101.287
    primary_hdr["DEC_OBJ"] = -16.716
    if bkgsub_card is not None:
        primary_hdr["BKGSUB"] = bkgsub_card
    primary_hdu = fits.PrimaryHDU(header=primary_hdr)

    img_dim = f"({nx}, {ny})"
    n_pix = ny * nx
    if flux_unit_override is not None:
        flux_unit: str | None = flux_unit_override
    else:
        flux_unit = "e-/s" if include_tunit else None
    cnt_unit = "count" if include_tunit else None

    cols = [
        fits.Column(name="TIME", format="D", unit="BJD - 2457000, days", array=time),
        fits.Column(name="TIMECORR", format="E", unit="d", array=timecorr),
        fits.Column(name="CADENCENO", format="J", array=cadenceno),
        fits.Column(
            name="RAW_CNTS",
            format=f"{n_pix}J",
            dim=img_dim,
            unit=cnt_unit,
            array=raw_cnts_data,
        ),
        fits.Column(
            name="FLUX",
            format=f"{n_pix}E",
            dim=img_dim,
            unit=flux_unit,
            array=flux_col_data.astype(np.float32),
        ),
        fits.Column(
            name="FLUX_ERR",
            format=f"{n_pix}E",
            dim=img_dim,
            unit=flux_unit,
            array=flux_err_data.astype(np.float32),
        ),
        fits.Column(
            name="FLUX_BKG",
            format=f"{n_pix}E",
            dim=img_dim,
            unit=flux_unit,
            array=bkg_col_data.astype(np.float32),
        ),
        fits.Column(
            name="FLUX_BKG_ERR",
            format=f"{n_pix}E",
            dim=img_dim,
            unit=flux_unit,
            array=bkg_err_data.astype(np.float32),
        ),
        fits.Column(name="QUALITY", format="J", array=quality),
        fits.Column(name="POS_CORR1", format="E", unit="pixel", array=pos_corr1),
        fits.Column(name="POS_CORR2", format="E", unit="pixel", array=pos_corr2),
    ]
    pixels_hdu = fits.BinTableHDU.from_columns(cols, name="PIXELS")
    pixels_hdu.header["INHERIT"] = True
    if bkgsub_card is not None:
        pixels_hdu.header["BKGSUB"] = bkgsub_card

    aperture = np.ones((ny, nx), dtype=np.int32)
    aperture[~sky_pixel_mask] = 3
    aperture_hdu = fits.ImageHDU(data=aperture, name="APERTURE")

    hdulist = fits.HDUList([primary_hdu, pixels_hdu, aperture_hdu])
    tpf = lk.TessTargetPixelFile(hdulist)
    return SyntheticTPFTruth(
        tpf=tpf,
        sky_raw=sky_raw,
        star_raw=star_raw,
        good_mask=good_mask,
        sky_pixel_mask=sky_pixel_mask,
    )


@pytest.fixture
def spoc_tpf_truth() -> SyntheticTPFTruth:
    return make_synthetic_tpf(mode="spoc")


@pytest.fixture
def tesscut_tpf_truth() -> SyntheticTPFTruth:
    return make_synthetic_tpf(mode="tesscut")


@pytest.fixture
def mundey_v2_tpf_truth() -> SyntheticTPFTruth:
    return make_synthetic_tpf(mode="mundey_v2")


@pytest.fixture
def mundey_v01_tpf_truth() -> SyntheticTPFTruth:
    return make_synthetic_tpf(mode="mundey_v01")
