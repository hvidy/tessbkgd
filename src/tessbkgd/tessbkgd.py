"""Functional ``lightkurve`` v2.0 background API and deprecated ``bkgd_tpf`` shim."""

from __future__ import annotations

import functools
import math
import re
import warnings
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, overload

import astropy.units as u
import lightkurve as lk
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.units import Quantity
from astropy.wcs import WCS

from ._contract import (
    BkgCovariance,
    _flux_column_unit_str,
    _set_bkgsub_header,
    _strip_bkg_covariance,
    _sync_tpf_meta,
    _unwrap_hdulist,
    pack_bkg_chol,
    read_bkg_covariance,
    read_bkgsub_state,
    read_smear_err,
    write_tessbkgd_provenance,
)
from .masking import ORDER_NAMES, build_background_mask
from .solver import (
    equilibrate_chebyshev_basis,
    reconstruct_background_chunked,
    solve_chebyshev_background,
)

_CANONICAL_RATE_UNITS = (
    u.Unit("electron / s"),
    u.Unit("ct / s"),
    u.Unit("adu / s"),
    u.Unit("photon / s"),
)
_IO_CHUNK = 8192
_VAR_SUBCHUNK = 4096
_F32_EPS = float(np.finfo(np.float32).eps)
_F32_QUAD_VAR_TOL = 2.0 * _F32_EPS
_F32_QUAD_AMP_FLOOR = math.sqrt(2.0 * _F32_EPS)

_DEPRECATION_MSG = (
    "bkgd_tpf is deprecated in tessbkgd 2.0 and will be removed in a future "
    "release; use tessbkgd.subtract_background(tpf) or "
    "tessbkgd.estimate_background(tpf) instead."
)


def _deprecated(message: str):
    """PEP 702-compatible deprecation decorator supporting Python >= 3.11."""

    def decorator(obj: Any) -> Any:
        if isinstance(obj, type):
            orig_init = obj.__init__

            @functools.wraps(orig_init)
            def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                orig_init(self, *args, **kwargs)

            obj.__init__ = wrapped_init
            obj.__deprecated__ = message
            return obj

        @functools.wraps(obj)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return obj(*args, **kwargs)

        wrapped.__deprecated__ = message  # type: ignore[attr-defined]
        return wrapped

    return decorator


@dataclass(frozen=True)
class BackgroundDiagnostics:
    """Detailed diagnostics from the v2.0 Chebyshev Huber-Tukey background fit.

    Under ``HALOCVER = 1``, ``bkg_chol`` is the pixel-independent Huber–White
    sandwich Cholesky factor conditional on column smear (which is carried
    separately in ``SMEAR_ERR``).
    """

    bkg_mask: np.ndarray
    source_mask: np.ndarray
    requested_order: int
    effective_order: int
    condition_number: float
    bkg_basis: np.ndarray
    coeffs: np.ndarray
    bkg_chol: np.ndarray
    tau_scale: np.ndarray
    n_valid_pixels: np.ndarray
    n_tukey_inliers: np.ndarray
    used_huber_fallback: np.ndarray
    was_bkg_subtracted: bool


class BackgroundEstimate(NamedTuple):
    """Estimated background cube, 1-sigma marginal uncertainty, and pixel mask."""

    bkg: Quantity
    bkg_err: Quantity
    mask: np.ndarray


def _quantity_to_canonical_array(arr: Any, *, dtype: Any) -> np.ndarray:
    """Convert ``arr`` to a raw NumPy ``ndarray`` with validated TESS flux units."""
    if isinstance(arr, Quantity):
        unit = arr.unit
        if unit is None or unit == u.dimensionless_unscaled:
            return np.asarray(arr.value, dtype=dtype)
        if unit.is_equivalent(u.dimensionless_unscaled):
            return np.asarray(
                arr.to_value(u.dimensionless_unscaled), dtype=dtype
            )
        for canon_unit in _CANONICAL_RATE_UNITS:
            if unit.is_equivalent(canon_unit):
                return np.asarray(arr.to_value(canon_unit), dtype=dtype)
        raise u.UnitConversionError(
            f"Cannot convert Quantity with unit {unit!r} to TESS count-rate "
            "('electron/s') or dimensionless flux."
        )
    if hasattr(arr, "unit") and arr.unit is not None:
        return _quantity_to_canonical_array(Quantity(arr, copy=False), dtype=dtype)
    if hasattr(arr, "value"):
        return np.asarray(arr.value, dtype=dtype)
    return np.asarray(arr, dtype=dtype)


def _as_float64_ndarray(arr: Any) -> np.ndarray:
    """Extract a raw ``float64`` NumPy ``ndarray`` with unit conversion."""
    return _quantity_to_canonical_array(arr, dtype=np.float64)


def _strip_for_fits_column(value: Any) -> np.ndarray:
    """Strip/convert Astropy ``Quantity`` units and cast to ``float32``."""
    return _quantity_to_canonical_array(value, dtype=np.float32)


def _parse_tpf_unit(unit_raw: str | u.Unit | None) -> u.Unit | None:
    """Parse a FITS ``TUNITn`` string into a canonical Astropy :class:`Unit`."""
    if unit_raw is None:
        return None
    if isinstance(unit_raw, str):
        cleaned = unit_raw.strip()
        if not cleaned:
            return None
        cleaned = re.sub(r"\be-(?=\s|/|$)", "electron", cleaned)
        try:
            parsed = u.Unit(cleaned)
        except ValueError:
            return None
    else:
        parsed = u.Unit(unit_raw)
    if parsed == u.dimensionless_unscaled:
        return None
    if parsed == u.Unit("electron / s"):
        return u.Unit("electron/s")
    return parsed


def _hdulist_column_unit(
    hdulist: fits.HDUList, col_name: str = "FLUX"
) -> u.Unit | None:
    """Resolve the Astropy :class:`Unit` of ``col_name`` in ``hdulist[1]``."""
    if len(hdulist) < 2:
        return None
    hdr = hdulist[1].header
    target = col_name.upper()
    if hdulist[1].columns is not None:
        for idx, col in enumerate(hdulist[1].columns, start=1):
            if col.name.upper() == target:
                raw_unit = col.unit or hdr.get(f"TUNIT{idx}")
                return _parse_tpf_unit(raw_unit)
        return None
    if target == "FLUX":
        return _parse_tpf_unit(hdr.get("TUNIT5"))
    if target == "FLUX_BKG":
        return _parse_tpf_unit(hdr.get("TUNIT7"))
    return None


def _resolve_tpf_wcs(tpf_or_hdu: Any, explicit_wcs: WCS | None) -> WCS | None:
    """Extract a celestial :class:`astropy.wcs.WCS` from ``tpf_or_hdu`` if present."""
    if explicit_wcs is not None:
        return explicit_wcs
    try:
        w = getattr(tpf_or_hdu, "wcs", None)
        if w is not None and getattr(w, "has_celestial", False):
            return w
    except Exception:
        pass
    hdulist = _unwrap_hdulist(tpf_or_hdu)
    for ext in (2, 1, 0):
        if ext < len(hdulist):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    w = WCS(hdulist[ext].header)
                if getattr(w, "has_celestial", False):
                    return w
            except Exception:
                continue
    return None


def _resolve_tpf_obs_time(tpf_or_hdu: Any) -> Time | None:
    """Infer mid-sector observation epoch ``Time`` (JD, TDB) from a TPF or HDUList."""
    hdulist = _unwrap_hdulist(tpf_or_hdu)
    h0 = hdulist[0].header if len(hdulist) > 0 else fits.Header()
    h1 = hdulist[1].header if len(hdulist) > 1 else fits.Header()

    def _header_bjdref() -> float | None:
        for hdr in (h1, h0):
            if "BJDREFI" in hdr:
                bjd_i = hdr.get("BJDREFI")
                bjd_f = hdr.get("BJDREFF", 0.0)
                if bjd_i is not None and np.isfinite(float(bjd_i)):
                    f_val = (
                        float(bjd_f)
                        if (bjd_f is not None and np.isfinite(float(bjd_f)))
                        else 0.0
                    )
                    return float(bjd_i) + f_val
        return None

    bjdref = _header_bjdref()

    # 1. Explicit TIME + BJDREFI/BJDREFF takes highest precedence when present.
    med_time_raw: float | None = None
    time_unit_str: str | None = None
    if (
        len(hdulist) > 1
        and hdulist[1].data is not None
        and hdulist[1].data.names is not None
    ):
        col_map = {n.upper(): n for n in hdulist[1].data.names}
        if "TIME" in col_map:
            t_raw = np.asarray(
                hdulist[1].data[col_map["TIME"]], dtype=np.float64
            )
            qmask = getattr(tpf_or_hdu, "quality_mask", None)
            if (
                qmask is not None
                and len(qmask) == t_raw.size
                and np.any(qmask)
            ):
                t_sel = t_raw[np.asarray(qmask, dtype=bool)]
            else:
                t_sel = t_raw
            fin_t = t_sel[np.isfinite(t_sel)]
            if fin_t.size == 0:
                fin_t = t_raw[np.isfinite(t_raw)]
            if fin_t.size > 0:
                med_time_raw = float(np.median(fin_t))
            if hdulist[1].columns is not None:
                for idx, col in enumerate(hdulist[1].columns, start=1):
                    if col.name.upper() == "TIME":
                        raw_u = col.unit or h1.get(f"TUNIT{idx}")
                        if raw_u:
                            time_unit_str = str(raw_u)
                        break

    if med_time_raw is not None and bjdref is not None:
        return Time(bjdref + med_time_raw, format="jd", scale="tdb")

    # 2. Lightkurve TargetPixelFile.time (handles BTJD/BKJD defaults accurately).
    t_attr = getattr(tpf_or_hdu, "time", None)
    if t_attr is not None:
        try:
            if len(t_attr) > 0:
                jd_vals = np.asarray(t_attr.jd, dtype=np.float64)
                fin_jd = jd_vals[np.isfinite(jd_vals)]
                if fin_jd.size > 0:
                    return Time(
                        float(np.median(fin_jd)), format="jd", scale="tdb"
                    )
        except Exception:
            pass

    # 3. Raw HDUList TIME column via TUNIT "BJD - <offset>", TELESCOP, or raw JD.
    if med_time_raw is not None:
        if med_time_raw > 2.4e6:
            return Time(med_time_raw, format="jd", scale="tdb")
        if time_unit_str:
            m_off = re.search(
                r"BJD\s*-\s*([0-9]+(?:\.[0-9]+)?)",
                time_unit_str,
                flags=re.IGNORECASE,
            )
            if m_off:
                return Time(
                    float(m_off.group(1)) + med_time_raw,
                    format="jd",
                    scale="tdb",
                )
        telescop = (
            str(h0.get("TELESCOP", h1.get("TELESCOP", ""))).strip().upper()
        )
        if telescop == "TESS" and 0.0 <= med_time_raw < 1.0e5:
            return Time(2457000.0 + med_time_raw, format="jd", scale="tdb")

    # 4. TSTART / TSTOP + BJDREFI/BJDREFF fallback.
    if bjdref is not None:
        for hdr in (h1, h0):
            ts = hdr.get("TSTART")
            te = hdr.get("TSTOP")
            if (
                ts is not None
                and te is not None
                and np.isfinite(float(ts))
                and np.isfinite(float(te))
            ):
                return Time(
                    bjdref + 0.5 * (float(ts) + float(te)),
                    format="jd",
                    scale="tdb",
                )

    # 5. MJD-OBS / DATE-OBS fallback across HDU headers.
    for ext in (0, 1, 2):
        if ext < len(hdulist):
            hdr = hdulist[ext].header
            mjd_obs = hdr.get("MJD-OBS")
            if mjd_obs is not None:
                try:
                    mjd_val = float(mjd_obs)
                    if np.isfinite(mjd_val) and mjd_val > 0.0:
                        return Time(mjd_val, format="mjd", scale="tdb")
                except (TypeError, ValueError):
                    pass
            date_obs = hdr.get("DATE-OBS")
            if isinstance(date_obs, str) and date_obs.strip():
                try:
                    t_parsed = Time(date_obs.strip())
                    return Time(float(t_parsed.jd), format="jd", scale="tdb")
                except ValueError:
                    pass
    return None


def _reconstruct_indep_bkg_var_slice(
    chol_slice: np.ndarray,
    phi_flat: np.ndarray,
    ny: int,
    nx: int,
) -> np.ndarray:
    """Evaluate ``||L_t.T @ phi||_2^2`` across ``chol_slice`` in ``float64``."""
    n_cad, p_dim, _ = chol_slice.shape
    n_pix = ny * nx
    out = np.empty((n_cad, ny, nx), dtype=np.float64)
    if n_cad == 0:
        return out

    sub_cap = min(n_cad, _VAR_SUBCHUNK)
    acc = np.empty((sub_cap, n_pix), dtype=np.float64)
    tmp = np.empty((sub_cap, n_pix), dtype=np.float64)
    for start in range(0, n_cad, _VAR_SUBCHUNK):
        stop = min(start + _VAR_SUBCHUNK, n_cad)
        m_len = stop - start
        acc_v = acc[:m_len]
        tmp_v = tmp[:m_len]
        acc_v.fill(0.0)
        c_sub = chol_slice[start:stop]
        bad_l = ~np.all(np.isfinite(c_sub), axis=(1, 2))
        c_clean = np.where(bad_l[:, np.newaxis, np.newaxis], 0.0, c_sub)
        for j in range(p_dim):
            l_col = np.ascontiguousarray(c_clean[:, :, j], dtype=np.float64)
            np.matmul(l_col, phi_flat, out=tmp_v)
            np.square(tmp_v, out=tmp_v)
            acc_v += tmp_v
        if np.any(bad_l):
            acc_v[bad_l] = np.nan
        out[start:stop] = acc_v.reshape(m_len, ny, nx)
    return out


def _extract_unsubtracted_slice(
    data: Any,
    col_map: dict[str, str],
    was_subtracted: bool,
    slc: slice | np.ndarray,
    *,
    prev_bkg_cov: BkgCovariance | None = None,
    prev_phi_flat: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return ``(U, sigma_U_sq, raw_flux_err)`` in ``float64`` for row slice ``slc``."""
    flux_f64 = np.asarray(data[col_map["FLUX"]][slc], dtype=np.float64)
    if was_subtracted and "FLUX_BKG" in col_map:
        bkg_f64 = np.asarray(data[col_map["FLUX_BKG"]][slc], dtype=np.float64)
        u_slice = flux_f64 + bkg_f64
    else:
        u_slice = flux_f64

    if "FLUX_ERR" not in col_map:
        return u_slice, np.full_like(u_slice, np.nan), None

    fe_f64 = np.asarray(data[col_map["FLUX_ERR"]][slc], dtype=np.float64)
    valid_fe = np.isfinite(fe_f64) & (fe_f64 > 0.0)
    fe_sq = np.where(valid_fe, fe_f64 * fe_f64, np.nan)
    if was_subtracted:
        if prev_bkg_cov is not None and prev_phi_flat is not None:
            _, ny, nx = prev_bkg_cov.basis.shape
            be_indep_sq = _reconstruct_indep_bkg_var_slice(
                prev_bkg_cov.chol[slc], prev_phi_flat, ny, nx
            )
            valid_be = np.isfinite(be_indep_sq) & (be_indep_sq >= 0.0)
            sigma_u_sq = np.where(
                valid_fe & valid_be,
                np.maximum(fe_sq - be_indep_sq, 0.0),
                fe_sq,
            )
        elif "FLUX_BKG_ERR" in col_map:
            be_f64 = np.asarray(
                data[col_map["FLUX_BKG_ERR"]][slc], dtype=np.float64
            )
            valid_be = np.isfinite(be_f64) & (be_f64 >= 0.0)
            sigma_u_sq = np.where(
                valid_fe & valid_be,
                np.maximum(fe_sq - be_f64 * be_f64, 0.0),
                fe_sq,
            )
        else:
            sigma_u_sq = fe_sq
    else:
        sigma_u_sq = fe_sq

    fe_clean = np.where(valid_fe, fe_f64, np.nan)
    return u_slice, sigma_u_sq, fe_clean


def _prepare_prev_bkg_cov(
    hdulist: fits.HDUList, was_subtracted: bool
) -> tuple[BkgCovariance | None, np.ndarray | None]:
    """Load prior ``(BkgCovariance, phi_flat)`` when ``was_subtracted=True``."""
    if not was_subtracted:
        return None, None
    cov = read_bkg_covariance(hdulist, apply_quality_mask=False)
    if cov is None:
        return None, None
    p_dim, ny, nx = cov.basis.shape
    phi_flat = np.ascontiguousarray(
        cov.basis.reshape(p_dim, ny * nx), dtype=np.float64
    )
    return cov, phi_flat


def _fit_tpf_background_core(
    tpf: Any,
    *,
    order: int | str | None = None,
    mask: np.ndarray | None = None,
    gaia_sources: Any = None,
    wcs: WCS | None = None,
    source_mag_limit: float = 19.0,
    max_source_radius: float | None = None,
    obs_time: Time | str | float | None = None,
    allow_downgrade: bool = True,
    verbose: bool = False,
    reconstruct: bool = True,
) -> tuple[
    fits.HDUList,
    dict[str, str],
    np.ndarray,
    BackgroundDiagnostics,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Execute v2.0 masking, QR Chebyshev equilibration, IRLS, and reconstruction."""
    hdulist = _unwrap_hdulist(tpf)
    if len(hdulist) < 2 or hdulist[1].data is None:
        raise ValueError(
            "TargetPixelFile must contain a populated PIXELS BinTableHDU at hdu[1]."
        )

    was_subtracted = read_bkgsub_state(tpf)
    data = hdulist[1].data
    col_map = {name.upper(): name for name in (data.names or ())}
    if "FLUX" not in col_map:
        raise ValueError("PIXELS HDU (hdu[1]) does not contain a 'FLUX' column.")

    flux_col = data[col_map["FLUX"]]
    if flux_col.ndim != 3:
        raise ValueError(
            f"Expected 3D FLUX column (N_raw, ny, nx), got shape {flux_col.shape}."
        )
    n_raw, ny, nx = flux_col.shape

    prev_cov, prev_phi_flat = _prepare_prev_bkg_cov(hdulist, was_subtracted)

    qmask_attr = getattr(tpf, "quality_mask", None)
    if qmask_attr is not None and len(qmask_attr) == n_raw:
        good_mask = np.asarray(qmask_attr, dtype=bool)
    else:
        good_mask = np.ones(n_raw, dtype=bool)

    mask_rows = good_mask if np.any(good_mask) else np.ones(n_raw, dtype=bool)
    mask_idx = np.flatnonzero(mask_rows)
    n_mask_rows = int(mask_idx.size)
    mask_dtype = np.float32 if flux_col.dtype.itemsize <= 4 else np.float64
    u_mask_cube = np.empty((n_mask_rows, ny, nx), dtype=mask_dtype)
    has_fe_col = "FLUX_ERR" in col_map
    sigma_u_buf = (
        np.empty((n_mask_rows, ny, nx), dtype=mask_dtype)
        if has_fe_col
        else None
    )
    any_valid_fe_mask = False
    for m0 in range(0, n_mask_rows, _IO_CHUNK):
        m1 = min(m0 + _IO_CHUNK, n_mask_rows)
        u_c, sig_sq_c, fe_c = _extract_unsubtracted_slice(
            data,
            col_map,
            was_subtracted,
            mask_idx[m0:m1],
            prev_bkg_cov=prev_cov,
            prev_phi_flat=prev_phi_flat,
        )
        u_mask_cube[m0:m1] = u_c
        if sigma_u_buf is not None and fe_c is not None:
            valid_fe = (
                np.isfinite(u_c)
                & np.isfinite(fe_c)
                & (fe_c > 0.0)
                & (sig_sq_c > _F32_QUAD_VAR_TOL * fe_c * fe_c)
            )
            if np.any(valid_fe):
                any_valid_fe_mask = True
            sigma_u_buf[m0:m1] = np.where(
                np.isfinite(fe_c) & (fe_c > 0.0),
                np.maximum(np.sqrt(sig_sq_c), _F32_QUAD_AMP_FLOOR * fe_c),
                np.nan,
            )
    sigma_u_mask: np.ndarray | None = (
        sigma_u_buf if any_valid_fe_mask else None
    )
    del sigma_u_buf

    pq_arr = (
        np.asarray(hdulist["PIXEL_QUALITY"].data, dtype=np.uint16)
        if (
            "PIXEL_QUALITY" in hdulist
            and hdulist["PIXEL_QUALITY"].data is not None
        )
        else None
    )
    resolved_wcs = (
        _resolve_tpf_wcs(tpf, wcs) if gaia_sources is not None else wcs
    )
    resolved_obs_time = (
        obs_time
        if obs_time is not None
        else (_resolve_tpf_obs_time(tpf) if gaia_sources is not None else None)
    )

    mask_res = build_background_mask(
        u_mask_cube,
        sigma_u=sigma_u_mask,
        pixel_quality=pq_arr,
        order=order,
        mask=mask,
        wcs=resolved_wcs,
        gaia_sources=gaia_sources,
        source_mag_limit=source_mag_limit,
        max_source_radius=max_source_radius,
        obs_time=resolved_obs_time,
        allow_downgrade=allow_downgrade,
    )
    del u_mask_cube, sigma_u_mask

    bkg_mask = mask_res.mask
    eff_order = mask_res.effective_order
    n_used = int(np.count_nonzero(bkg_mask))

    phi_full, phi_bg, _ = equilibrate_chebyshev_basis(
        ny, nx, eff_order, bkg_mask
    )

    if verbose:
        n_vis = int(np.count_nonzero(good_mask))
        print(
            f"Fitting {ORDER_NAMES[eff_order]} background across {n_vis} "
            f"cadences using {n_used}/{ny * nx} pixels."
        )

    # Extract background pixels across all N_raw rows in <= 8,192-cadence chunks.
    u_bg = np.empty((n_raw, n_used), dtype=np.float64)
    has_fe_col = "FLUX_ERR" in col_map
    sig_bg = (
        np.empty((n_raw, n_used), dtype=np.float64) if has_fe_col else None
    )
    any_informative_err = False

    for start in range(0, n_raw, _IO_CHUNK):
        stop = min(start + _IO_CHUNK, n_raw)
        u_c, sig_sq_c, fe_c = _extract_unsubtracted_slice(
            data,
            col_map,
            was_subtracted,
            slice(start, stop),
            prev_bkg_cov=prev_cov,
            prev_phi_flat=prev_phi_flat,
        )
        u_bg_c = u_c[:, bkg_mask]
        u_bg[start:stop] = u_bg_c
        if sig_bg is not None and fe_c is not None:
            fe_bg_c = fe_c[:, bkg_mask]
            sq_bg_c = sig_sq_c[:, bkg_mask]
            informative = (
                np.isfinite(u_bg_c)
                & np.isfinite(fe_bg_c)
                & (fe_bg_c > 0.0)
                & (sq_bg_c > _F32_QUAD_VAR_TOL * fe_bg_c * fe_bg_c)
            )
            if np.any(informative):
                any_informative_err = True
            sig_bg[start:stop] = np.where(
                np.isfinite(fe_bg_c) & (fe_bg_c > 0.0),
                np.maximum(np.sqrt(sq_bg_c), _F32_QUAD_AMP_FLOOR * fe_bg_c),
                np.nan,
            )

    if not any_informative_err:
        sig_bg = None

    smear_raw = read_smear_err(hdulist, apply_quality_mask=False)

    sol = solve_chebyshev_background(
        u_bg,
        phi_bg,
        sigma_u_bg=sig_bg,
        chunk_size=_IO_CHUNK,
        smear_err=smear_raw,
        phi_full=phi_full,
        bkg_mask=bkg_mask,
    )
    del u_bg, sig_bg
    if reconstruct:
        bkg_raw_f64, bkg_err_indep_f64 = reconstruct_background_chunked(
            sol.coeffs,
            sol.bkg_chol,
            phi_full,
            chunk_size=_IO_CHUNK,
            out_dtype=np.float64,
        )

        if sol.delta_bkg_smear_sq is not None:
            bkg_err_total_f64 = np.empty_like(bkg_err_indep_f64)
            for start in range(0, n_raw, _IO_CHUNK):
                stop = min(start + _IO_CHUNK, n_raw)
                ind_c = bkg_err_indep_f64[start:stop]
                bkg_err_total_f64[start:stop] = np.sqrt(
                    ind_c * ind_c + sol.delta_bkg_smear_sq[start:stop]
                )
        else:
            bkg_err_total_f64 = bkg_err_indep_f64
    else:
        bkg_raw_f64 = None
        bkg_err_indep_f64 = None
        bkg_err_total_f64 = None

    diagnostics = BackgroundDiagnostics(
        bkg_mask=bkg_mask,
        source_mask=mask_res.source_mask,
        requested_order=mask_res.requested_order,
        effective_order=eff_order,
        condition_number=mask_res.condition_number,
        bkg_basis=phi_full,
        coeffs=sol.coeffs,
        bkg_chol=sol.bkg_chol,
        tau_scale=sol.tau_scale,
        n_valid_pixels=sol.n_valid_pixels,
        n_tukey_inliers=sol.n_tukey_inliers,
        used_huber_fallback=sol.used_huber_fallback,
        was_bkg_subtracted=was_subtracted,
    )
    return (
        hdulist,
        col_map,
        good_mask,
        diagnostics,
        bkg_raw_f64,
        bkg_err_indep_f64,
        bkg_err_total_f64,
        sol.delta_bkg_smear_sq,
    )


@overload
def estimate_background(
    tpf: Any,
    *,
    order: int | str | None = ...,
    mask: np.ndarray | None = ...,
    gaia_sources: Any = ...,
    wcs: WCS | None = ...,
    source_mag_limit: float = ...,
    max_source_radius: float | None = ...,
    obs_time: Time | str | float | None = ...,
    allow_downgrade: bool = ...,
    return_diagnostics: Literal[False] = ...,
    verbose: bool = ...,
) -> BackgroundEstimate: ...


@overload
def estimate_background(
    tpf: Any,
    *,
    order: int | str | None = ...,
    mask: np.ndarray | None = ...,
    gaia_sources: Any = ...,
    wcs: WCS | None = ...,
    source_mag_limit: float = ...,
    max_source_radius: float | None = ...,
    obs_time: Time | str | float | None = ...,
    allow_downgrade: bool = ...,
    return_diagnostics: Literal[True],
    verbose: bool = ...,
) -> tuple[BackgroundEstimate, BackgroundDiagnostics]: ...


def estimate_background(
    tpf: Any,
    *,
    order: int | str | None = None,
    mask: np.ndarray | None = None,
    gaia_sources: Any = None,
    wcs: WCS | None = None,
    source_mag_limit: float = 19.0,
    max_source_radius: float | None = None,
    obs_time: Time | str | float | None = None,
    allow_downgrade: bool = True,
    return_diagnostics: bool = False,
    verbose: bool = False,
) -> BackgroundEstimate | tuple[BackgroundEstimate, BackgroundDiagnostics]:
    """Estimate a 2D Chebyshev scattered-light model without mutating ``tpf``."""
    hdulist, _, good_mask, diag, bkg_raw, _, bkg_err_total, _ = (
        _fit_tpf_background_core(
            tpf,
            order=order,
            mask=mask,
            gaia_sources=gaia_sources,
            wcs=wcs,
            source_mag_limit=source_mag_limit,
            max_source_radius=max_source_radius,
            obs_time=obs_time,
            allow_downgrade=allow_downgrade,
            verbose=verbose,
        )
    )
    assert bkg_raw is not None and bkg_err_total is not None
    unit = _hdulist_column_unit(hdulist, "FLUX")
    est = BackgroundEstimate(
        bkg=Quantity(bkg_raw[good_mask], unit=unit),
        bkg_err=Quantity(bkg_err_total[good_mask], unit=unit),
        mask=diag.bkg_mask,
    )
    if return_diagnostics:
        return est, diag
    return est


def _write_subtracted_columns_into_hdulist(
    hdulist: fits.HDUList,
    col_map: dict[str, str],
    was_subtracted: bool,
    bkg_raw_f64: np.ndarray | None,
    bkg_err_indep_f64: np.ndarray | None,
    bkg_err_total_f64: np.ndarray | None,
    *,
    diag: BackgroundDiagnostics | None = None,
    delta_bkg_smear_sq: np.ndarray | None = None,
    prev_cov: BkgCovariance | None = None,
    prev_phi_flat: np.ndarray | None = None,
    use_existing_prev_cov: bool = True,
) -> fits.HDUList:
    """Update ``FLUX``, ``FLUX_ERR``, ``FLUX_BKG``, and ``FLUX_BKG_ERR`` in chunks."""
    data = hdulist[1].data
    if bkg_raw_f64 is not None:
        n_raw, ny, nx = bkg_raw_f64.shape
    else:
        assert diag is not None
        n_raw = diag.coeffs.shape[0]
        _, ny, nx = diag.bkg_basis.shape
    n_pix = ny * nx

    if use_existing_prev_cov:
        prev_cov, prev_phi_flat = _prepare_prev_bkg_cov(
            hdulist, was_subtracted
        )

    has_flux = "FLUX" in col_map
    has_flux_err = "FLUX_ERR" in col_map
    has_bkg = "FLUX_BKG" in col_map
    has_bkg_err = "FLUX_BKG_ERR" in col_map

    new_flux = (
        None if has_flux else np.empty((n_raw, ny, nx), dtype=np.float32)
    )
    new_flux_err = (
        None if has_flux_err else np.empty((n_raw, ny, nx), dtype=np.float32)
    )
    new_bkg = None if has_bkg else np.empty((n_raw, ny, nx), dtype=np.float32)
    new_bkg_err = (
        None if has_bkg_err else np.empty((n_raw, ny, nx), dtype=np.float32)
    )

    flux_col = data[col_map["FLUX"]] if has_flux else None
    flux_err_col = data[col_map["FLUX_ERR"]] if has_flux_err else None
    bkg_col = data[col_map["FLUX_BKG"]] if has_bkg else None
    bkg_err_col = data[col_map["FLUX_BKG_ERR"]] if has_bkg_err else None

    for start in range(0, n_raw, _IO_CHUNK):
        stop = min(start + _IO_CHUNK, n_raw)
        u_c, sig_sq_c, _ = _extract_unsubtracted_slice(
            data,
            col_map,
            was_subtracted,
            slice(start, stop),
            prev_bkg_cov=prev_cov,
            prev_phi_flat=prev_phi_flat,
        )
        if bkg_raw_f64 is not None:
            assert (
                bkg_err_indep_f64 is not None
                and bkg_err_total_f64 is not None
            )
            b_c = bkg_raw_f64[start:stop]
            be_indep_c = bkg_err_indep_f64[start:stop]
            be_total_c = bkg_err_total_f64[start:stop]
        else:
            assert diag is not None
            b_c, be_indep_c = reconstruct_background_chunked(
                diag.coeffs[start:stop],
                diag.bkg_chol[start:stop],
                diag.bkg_basis,
                chunk_size=_IO_CHUNK,
                out_dtype=np.float64,
            )
            if delta_bkg_smear_sq is not None:
                be_total_c = np.sqrt(
                    be_indep_c * be_indep_c + delta_bkg_smear_sq[start:stop]
                )
            else:
                be_total_c = be_indep_c
        flux_out = (u_c - b_c).astype(np.float32)
        tot_err = np.where(
            np.isfinite(u_c) & np.isfinite(sig_sq_c),
            np.sqrt(sig_sq_c + be_indep_c * be_indep_c),
            np.nan,
        ).astype(np.float32)
        bkg_out = b_c.astype(np.float32)
        bkg_err_out = be_total_c.astype(np.float32)

        if flux_col is not None:
            flux_col[start:stop] = flux_out
        else:
            assert new_flux is not None
            new_flux[start:stop] = flux_out

        if flux_err_col is not None:
            flux_err_col[start:stop] = tot_err
        else:
            assert new_flux_err is not None
            new_flux_err[start:stop] = tot_err

        if bkg_col is not None:
            bkg_col[start:stop] = bkg_out
        else:
            assert new_bkg is not None
            new_bkg[start:stop] = bkg_out

        if bkg_err_col is not None:
            bkg_err_col[start:stop] = bkg_err_out
        else:
            assert new_bkg_err is not None
            new_bkg_err[start:stop] = bkg_err_out

    # Append any missing standard columns while preserving existing column indices
    # (TUNIT5/WCS) before HALOCVER = 1 provenance is written.
    missing_cols: list[fits.Column] = []
    flux_unit_str = _flux_column_unit_str(hdulist[1])
    img_dim = f"({nx}, {ny})"

    for canonical_name, arr_f32 in (
        ("FLUX", new_flux),
        ("FLUX_ERR", new_flux_err),
        ("FLUX_BKG", new_bkg),
        ("FLUX_BKG_ERR", new_bkg_err),
    ):
        if arr_f32 is not None:
            missing_cols.append(
                fits.Column(
                    name=canonical_name,
                    format=f"{n_pix}E",
                    dim=img_dim,
                    unit=flux_unit_str,
                    array=arr_f32,
                )
            )

    if missing_cols:
        old_hdr = hdulist[1].header.copy()
        ext_name = hdulist[1].name or "PIXELS"
        hdulist[1] = fits.BinTableHDU.from_columns(
            fits.ColDefs([*hdulist[1].columns, *missing_cols]),
            header=old_hdr,
            name=ext_name,
        )

    return hdulist


def _construct_clean_tpf(
    source_tpf: Any, new_hdulist: fits.HDUList
) -> lk.targetpixelfile.TargetPixelFile:
    """Wrap ``new_hdulist`` in a standard ``lightkurve.TargetPixelFile``."""
    qbitmask = getattr(source_tpf, "quality_bitmask", "default")
    targetid = getattr(source_tpf, "targetid", None)
    if isinstance(source_tpf, lk.TessTargetPixelFile):
        return lk.TessTargetPixelFile(
            new_hdulist, quality_bitmask=qbitmask, targetid=targetid
        )
    if isinstance(source_tpf, lk.targetpixelfile.TargetPixelFile):
        return type(source_tpf)(
            new_hdulist, quality_bitmask=qbitmask, targetid=targetid
        )
    return lk.TessTargetPixelFile(new_hdulist, quality_bitmask=qbitmask)


@overload
def subtract_background(
    tpf: Any,
    *,
    order: int | str | None = ...,
    mask: np.ndarray | None = ...,
    gaia_sources: Any = ...,
    wcs: WCS | None = ...,
    source_mag_limit: float = ...,
    max_source_radius: float | None = ...,
    obs_time: Time | str | float | None = ...,
    allow_downgrade: bool = ...,
    return_diagnostics: Literal[False] = ...,
    verbose: bool = ...,
) -> lk.TargetPixelFile: ...


@overload
def subtract_background(
    tpf: Any,
    *,
    order: int | str | None = ...,
    mask: np.ndarray | None = ...,
    gaia_sources: Any = ...,
    wcs: WCS | None = ...,
    source_mag_limit: float = ...,
    max_source_radius: float | None = ...,
    obs_time: Time | str | float | None = ...,
    allow_downgrade: bool = ...,
    return_diagnostics: Literal[True],
    verbose: bool = ...,
) -> tuple[lk.TargetPixelFile, BackgroundDiagnostics]: ...


def subtract_background(
    tpf: Any,
    *,
    order: int | str | None = None,
    mask: np.ndarray | None = None,
    gaia_sources: Any = None,
    wcs: WCS | None = None,
    source_mag_limit: float = 19.0,
    max_source_radius: float | None = None,
    obs_time: Time | str | float | None = None,
    allow_downgrade: bool = True,
    return_diagnostics: bool = False,
    verbose: bool = False,
) -> lk.TargetPixelFile | tuple[lk.TargetPixelFile, BackgroundDiagnostics]:
    """Fit and subtract a 2D Chebyshev background, returning a new ``TargetPixelFile``.

    Processes all ``N_raw`` rows of ``hdu[1]`` under ``HALOCVER = 1`` so
    ``!quality_mask`` rows never retain stale background. Propagates pixel-
    independent Huber-Street-Carroll-Ruppert sandwich covariances into
    ``BKG_CHOL`` and ``FLUX_ERR`` (preserving downstream ``SMEAR_ERR``
    orthogonality for ``halophot``), includes extra column-coherent ``SMEAR_ERR``
    variance in standalone ``FLUX_BKG_ERR`` whenever ``SMEAR_ERR`` is present,
    and stamps ``BKGSUB = True`` (plus ``BACKAPP = True`` when present) so
    repeated calls are idempotent.
    """
    (
        src_hdulist,
        _,
        _,
        diag,
        _,
        _,
        _,
        delta_bkg_smear_sq,
    ) = _fit_tpf_background_core(
        tpf,
        order=order,
        mask=mask,
        gaia_sources=gaia_sources,
        wcs=wcs,
        source_mag_limit=source_mag_limit,
        max_source_radius=max_source_radius,
        obs_time=obs_time,
        allow_downgrade=allow_downgrade,
        verbose=verbose,
        reconstruct=False,
    )
    prev_cov, prev_phi_flat = _prepare_prev_bkg_cov(
        src_hdulist, diag.was_bkg_subtracted
    )
    # Copy HDUs once while pre-attaching BKG_CHOL if absent/mismatched so
    # write_tessbkgd_provenance avoids a second full BinTableHDU copy on large TPFs.
    chol_packed = pack_bkg_chol(diag.bkg_chol)
    n_raw_tpf, n_tril = chol_packed.shape
    src_hdu1 = src_hdulist[1]
    target_chol_unit = _flux_column_unit_str(src_hdu1)
    src_cols = src_hdu1.columns if src_hdu1.columns is not None else ()
    existing_chol = next(
        (c for c in src_cols if c.name.upper() == "BKG_CHOL"), None
    )
    existing_unit = (
        str(existing_chol.unit)
        if (existing_chol is not None and existing_chol.unit)
        else None
    )
    if (
        existing_chol is not None
        and existing_unit == target_chol_unit
        and (
            src_hdu1.data["BKG_CHOL"].shape == (n_raw_tpf, n_tril)
            or (
                n_tril == 1
                and src_hdu1.data["BKG_CHOL"].shape == (n_raw_tpf,)
            )
        )
    ):
        new_hdu1 = src_hdu1.copy()
    else:
        kept_cols = [c for c in src_cols if c.name.upper() != "BKG_CHOL"]
        chol_col = fits.Column(
            name="BKG_CHOL",
            format=f"{n_tril}E",
            dim=f"({n_tril})",
            unit=target_chol_unit,
            array=chol_packed,
        )
        new_hdu1 = fits.BinTableHDU.from_columns(
            fits.ColDefs([*kept_cols, chol_col]),
            header=src_hdu1.header.copy(),
            name=src_hdu1.name or "PIXELS",
        )
    new_hdulist = fits.HDUList(
        [
            src_hdulist[0].copy(),
            new_hdu1,
            *(h.copy() for h in src_hdulist[2:]),
        ]
    )
    new_col_map = {
        name.upper(): name for name in (new_hdulist[1].data.names or ())
    }
    _write_subtracted_columns_into_hdulist(
        new_hdulist,
        new_col_map,
        diag.was_bkg_subtracted,
        None,
        None,
        None,
        diag=diag,
        delta_bkg_smear_sq=delta_bkg_smear_sq,
        prev_cov=prev_cov,
        prev_phi_flat=prev_phi_flat,
        use_existing_prev_cov=False,
    )
    del delta_bkg_smear_sq
    src_mask_arg = diag.source_mask if np.any(diag.source_mask) else None
    write_tessbkgd_provenance(
        new_hdulist,
        bkg_basis=diag.bkg_basis,
        bkg_chol=chol_packed,
        bkg_used_mask=diag.bkg_mask,
        bkg_order=diag.effective_order,
        source_mask=src_mask_arg,
    )
    out_tpf = _construct_clean_tpf(tpf, new_hdulist)
    if return_diagnostics:
        return out_tpf, diag
    return out_tpf


@_deprecated(_DEPRECATION_MSG)
class bkgd_tpf(lk.TessTargetPixelFile):
    """Deprecated ``TessTargetPixelFile`` shim delegating onto the v2.0 solver."""

    def _column_unit(self, col_name: str = "FLUX") -> u.Unit | None:
        """Return the parsed Astropy :class:`Unit` for ``col_name`` if present."""
        return _hdulist_column_unit(self.hdu, col_name)

    def _assign_table_column(self, col_name: str, value: Any) -> None:
        target_unit = self._column_unit(col_name) or self._column_unit("FLUX")
        if (
            isinstance(value, Quantity)
            and value.unit is not None
            and value.unit != u.dimensionless_unscaled
            and target_unit is not None
            and value.unit.is_equivalent(target_unit)
        ):
            raw = np.asarray(value.to_value(target_unit), dtype=np.float32)
        else:
            raw = _strip_for_fits_column(value)
        col = np.array(self.hdu[1].data[col_name], dtype=np.float32, copy=True)
        if raw.ndim == 0 or raw.shape == col.shape:
            col[:] = raw
        else:
            col[self.quality_mask] = raw
        self.hdu[1].data[col_name] = col

    @_deprecated(_DEPRECATION_MSG)
    def get_bkgd(
        self,
        bkgdmask: np.ndarray | None = None,
        order: int | str | None = "linear",
        verbose: bool = False,
    ) -> tuple[Quantity, np.ndarray]:
        """Estimate a 2D Chebyshev background across good-quality cadences."""
        est = estimate_background(
            self,
            order=order,
            mask=bkgdmask,
            allow_downgrade=(bkgdmask is None),
            verbose=verbose,
        )
        return est.bkg, est.mask

    @_deprecated(_DEPRECATION_MSG)
    def subtract_bkgd(
        self,
        bkgdmask: np.ndarray | None = None,
        order: int | str | None = "linear",
        verbose: bool = False,
    ) -> tuple[Quantity, np.ndarray]:
        """Fit and subtract a 2D Chebyshev background in place on ``self``."""
        (
            hdulist,
            col_map,
            good_mask,
            diag,
            bkg_raw,
            bkg_err_indep,
            bkg_err_total,
            _,
        ) = _fit_tpf_background_core(
            self,
            order=order,
            mask=bkgdmask,
            allow_downgrade=(bkgdmask is None),
            verbose=verbose,
        )
        assert bkg_raw is not None
        _write_subtracted_columns_into_hdulist(
            hdulist,
            col_map,
            diag.was_bkg_subtracted,
            bkg_raw,
            bkg_err_indep,
            bkg_err_total,
        )
        del bkg_err_indep, bkg_err_total
        src_mask_arg = diag.source_mask if np.any(diag.source_mask) else None
        write_tessbkgd_provenance(
            hdulist,
            bkg_basis=diag.bkg_basis,
            bkg_chol=diag.bkg_chol,
            bkg_used_mask=diag.bkg_mask,
            bkg_order=diag.effective_order,
            source_mask=src_mask_arg,
        )
        _sync_tpf_meta(self, hdulist)
        unit = self._column_unit("FLUX")
        return Quantity(bkg_raw[good_mask], unit=unit), diag.bkg_mask

    @property
    def flux(self) -> Quantity:
        """Return the flux cube for all good-quality cadences."""
        return Quantity(
            self.hdu[1].data["FLUX"][self.quality_mask],
            unit=self._column_unit("FLUX"),
        )

    @flux.setter
    def flux(self, value: Any) -> None:
        self._assign_table_column("FLUX", value)

    @property
    def flux_bkg(self) -> Quantity:
        """Return the background flux cube for all good-quality cadences."""
        unit = self._column_unit("FLUX_BKG") or self._column_unit("FLUX")
        return Quantity(
            self.hdu[1].data["FLUX_BKG"][self.quality_mask],
            unit=unit,
        )

    @flux_bkg.setter
    def flux_bkg(self, value: Any) -> None:
        self._assign_table_column("FLUX_BKG", value)
        bkg_good = self.hdu[1].data["FLUX_BKG"][self.quality_mask]
        has_nonzero_bkg = bool(
            np.any(np.isfinite(bkg_good) & (bkg_good != 0.0))
        )
        if _strip_bkg_covariance(self, has_nonzero_bkg=has_nonzero_bkg):
            warnings.warn(
                "Assigning bkgd_tpf.flux_bkg directly invalidated and stripped "
                "stale BKG_CHOL / BKG_BASIS / BKG_USED covariance artefacts; "
                "use tessbkgd.subtract_background(tpf) to retain calibrated "
                "HALOCVER = 1 background covariance.",
                UserWarning,
                stacklevel=2,
            )
        _set_bkgsub_header(self, has_nonzero_bkg)

    @property
    def raw_cnts(self) -> np.ndarray:
        """Return the raw counts for all good-quality cadences."""
        return self.hdu[1].data["RAW_CNTS"][self.quality_mask]
