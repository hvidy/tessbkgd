"""Functional ``lightkurve`` v2 background API, raw-cube support, and shim."""

from __future__ import annotations

import functools
import math
import re
import warnings
from dataclasses import dataclass
from typing import Any, NamedTuple

import astropy.units as u
import lightkurve as lk
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.units import Quantity
from astropy.wcs import WCS

from ._contract import (
    BkgCovariance,
    PixelQuality,
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
from .masking import (
    DEFAULT_EXCLUSION_RADIUS,
    OrderSpec,
    build_background_mask,
    format_order,
)
from .solver import (
    equilibrate_chebyshev_basis,
    reconstruct_background_chunked,
    reconstruct_chol_var_slice,
    solve_chebyshev_background,
)

_CANONICAL_RATE_UNITS = (
    u.Unit("electron / s"), u.Unit("ct / s"), u.Unit("adu / s"), u.Unit("photon / s"),
)
_IO_CHUNK = 8192
_F32_EPS = float(np.finfo(np.float32).eps)
_F32_QUAD_VAR_TOL = 2.0 * _F32_EPS
_F32_QUAD_AMP_FLOOR = math.sqrt(2.0 * _F32_EPS)
_TARGET_CORR_WARN_THRESHOLD = 0.5
_DEPRECATION_MSG = (
    "bkgd_tpf is deprecated in tessbkgd 2.0; use tessbkgd.subtract_background(tpf) "
    "or tessbkgd.estimate_background(tpf) instead."
)


def _deprecated(msg: str):
    def dec(obj: Any) -> Any:
        if isinstance(obj, type):
            orig = obj.__init__

            @functools.wraps(orig)
            def w_init(self: Any, *a: Any, **kw: Any) -> None:
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
                orig(self, *a, **kw)

            obj.__init__, obj.__deprecated__ = w_init, msg
            return obj

        @functools.wraps(obj)
        def w_fn(*a: Any, **kw: Any) -> Any:
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return obj(*a, **kw)

        w_fn.__deprecated__ = msg  # type: ignore[attr-defined]
        return w_fn

    return dec


@dataclass(frozen=True)
class BackgroundConfig:
    """User-tunable configuration for 2D Chebyshev background estimation."""

    order: OrderSpec = None
    exclusion_radius: float = DEFAULT_EXCLUSION_RADIUS
    source_mag_limit: float = 19.0
    max_source_radius: float | None = None
    allow_downgrade: bool = True
    verbose: bool = False


@dataclass(frozen=True)
class BackgroundDiagnostics:
    """Detailed diagnostics from the 2D Chebyshev Huber-Tukey background fit."""

    bkg_mask: np.ndarray
    source_mask: np.ndarray
    exclusion_mask: np.ndarray
    requested_order: tuple[int, int]
    effective_order: tuple[int, int]
    condition_number: float
    bkg_basis: np.ndarray
    coeffs: np.ndarray
    bkg_chol: np.ndarray
    tau_scale: np.ndarray
    n_valid_pixels: np.ndarray
    n_tukey_inliers: np.ndarray
    used_huber_fallback: np.ndarray
    was_bkg_subtracted: bool
    target_flux_corr: float = float("nan")
    dark_median_image: np.ndarray | None = None
    smear_chol: np.ndarray | None = None

    def plot(self, tpf: Any, **kwargs: Any) -> Any:
        """Render the 4-panel diagnostic figure via ``plot_diagnostics``."""
        from .plotting import plot_diagnostics

        return plot_diagnostics(tpf, result=self, **kwargs)


class BackgroundEstimate(NamedTuple):
    """Estimated background cube, 1-sigma marginal uncertainty, and pixel mask."""

    bkg: Quantity | np.ndarray
    bkg_err: Quantity | np.ndarray
    mask: np.ndarray


def _quantity_to_canonical_array(arr: Any, *, dtype: Any) -> np.ndarray:
    if isinstance(arr, Quantity):
        unit = arr.unit
        if unit is None or unit == u.dimensionless_unscaled:
            return np.asarray(arr.value, dtype=dtype)
        if unit.is_equivalent(u.dimensionless_unscaled):
            return np.asarray(arr.to_value(u.dimensionless_unscaled), dtype=dtype)
        for canon in _CANONICAL_RATE_UNITS:
            if unit.is_equivalent(canon):
                return np.asarray(arr.to_value(canon), dtype=dtype)
        raise u.UnitConversionError(f"Incompatible unit {unit!r}.")
    if hasattr(arr, "unit") and arr.unit is not None:
        return _quantity_to_canonical_array(Quantity(arr, copy=False), dtype=dtype)
    return np.asarray(arr.value if hasattr(arr, "value") else arr, dtype=dtype)


def _strip_for_fits_column(value: Any) -> np.ndarray:
    return _quantity_to_canonical_array(value, dtype=np.float32)


def _parse_tpf_unit(raw: str | u.Unit | None) -> u.Unit | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        cl = re.sub(r"\be-(?=\s|/|$)", "electron", raw.strip())
        if not cl:
            return None
        try:
            parsed = u.Unit(cl)
        except ValueError:
            return None
    else:
        parsed = u.Unit(raw)
    if parsed == u.dimensionless_unscaled:
        return None
    return u.Unit("electron/s") if parsed == u.Unit("electron / s") else parsed


def _hdulist_column_unit(hdul: fits.HDUList, col_name: str = "FLUX") -> u.Unit | None:
    if len(hdul) < 2:
        return None
    hdr, tgt = hdul[1].header, col_name.upper()
    if hdul[1].columns is not None:
        for i, col in enumerate(hdul[1].columns, start=1):
            if col.name.upper() == tgt:
                return _parse_tpf_unit(col.unit or hdr.get(f"TUNIT{i}"))
        return None
    return _parse_tpf_unit(hdr.get("TUNIT5" if tgt == "FLUX" else "TUNIT7"))


def _resolve_tpf_wcs(tpf_or_hdu: Any, explicit: WCS | None) -> WCS | None:
    if explicit is not None:
        return explicit
    try:
        w = getattr(tpf_or_hdu, "wcs", None)
        if w is not None and getattr(w, "has_celestial", False):
            return w
    except Exception:
        pass
    hdul = _unwrap_hdulist(tpf_or_hdu)
    for ext in (2, 1, 0):
        if ext < len(hdul):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    w = WCS(hdul[ext].header)
                if getattr(w, "has_celestial", False):
                    return w
            except Exception:
                continue
    return None


def _resolve_target_coords(hdul: fits.HDUList) -> tuple[float, float] | None:
    for ext in (0, 1):
        if ext < len(hdul):
            ra, dec = hdul[ext].header.get("RA_OBJ"), hdul[ext].header.get("DEC_OBJ")
            if ra is not None and dec is not None:
                try:
                    rf, df = float(ra), float(dec)
                    if np.isfinite(rf) and np.isfinite(df):
                        return (rf, df)
                except (TypeError, ValueError):
                    pass
    return None


def _tpf_median_time(
    tpf_or_hdu: Any, hdul: fits.HDUList
) -> tuple[float | None, str | None]:
    if len(hdul) < 2 or hdul[1].data is None or not hdul[1].data.names:
        return None, None
    cmap = {n.upper(): n for n in hdul[1].data.names}
    if "TIME" not in cmap:
        return None, None
    tr = np.asarray(hdul[1].data[cmap["TIME"]], dtype=np.float64)
    qm = getattr(tpf_or_hdu, "quality_mask", None)
    use_q = qm is not None and len(qm) == tr.size and np.any(qm)
    ts = tr[np.asarray(qm, dtype=bool)] if use_q else tr
    fin = ts[np.isfinite(ts)] if np.any(np.isfinite(ts)) else tr[np.isfinite(tr)]
    med_t = float(np.median(fin)) if fin.size else None
    tu = next(
        (
            str(c.unit or hdul[1].header.get(f"TUNIT{i}") or "") or None
            for i, c in enumerate(hdul[1].columns or (), start=1)
            if c.name.upper() == "TIME"
        ),
        None,
    )
    return med_t, tu


def _resolve_tpf_obs_time(tpf_or_hdu: Any) -> Time | None:
    hdul = _unwrap_hdulist(tpf_or_hdu)
    hdrs = [hdul[i].header for i in (1, 0) if i < len(hdul)]
    bjdref = next(
        (
            float(h["BJDREFI"]) + float(h.get("BJDREFF") or 0.0)
            for h in hdrs
            if h.get("BJDREFI") is not None and np.isfinite(float(h["BJDREFI"]))
        ),
        None,
    )
    med_t, tu = _tpf_median_time(tpf_or_hdu, hdul)
    if med_t is not None and bjdref is not None:
        return Time(bjdref + med_t, format="jd", scale="tdb")
    if (ta := getattr(tpf_or_hdu, "time", None)) is not None:
        try:
            jv = np.asarray(ta.jd, dtype=np.float64)
            if np.any(np.isfinite(jv)):
                return Time(float(np.nanmedian(jv)), format="jd", scale="tdb")
        except Exception:
            pass
    if med_t is not None:
        if med_t > 2.4e6:
            return Time(med_t, format="jd", scale="tdb")
        if m := re.search(r"BJD\s*-\s*([0-9]+(?:\.[0-9]+)?)", tu or "", re.I):
            return Time(float(m.group(1)) + med_t, format="jd", scale="tdb")
        tel = str(
            (hdrs[-1].get("TELESCOP") if hdrs else "")
            or (hdrs[0].get("TELESCOP") if hdrs else "")
        ).strip().upper()
        if tel == "TESS" and 0.0 <= med_t < 1e5:
            return Time(2457000.0 + med_t, format="jd", scale="tdb")
    if bjdref is not None:
        for h in hdrs:
            ts, te = h.get("TSTART"), h.get("TSTOP")
            ok_ts = ts is not None and te is not None
            if ok_ts and np.isfinite(float(ts)) and np.isfinite(float(te)):
                mid = bjdref + 0.5 * (float(ts) + float(te))
                return Time(mid, format="jd", scale="tdb")
    for h in [hdul[i].header for i in range(min(3, len(hdul)))]:
        try:
            mjd = h.get("MJD-OBS")
            if mjd is not None and np.isfinite(float(mjd)) and float(mjd) > 0:
                return Time(float(mjd), format="mjd", scale="tdb")
            if isinstance(dobs := h.get("DATE-OBS"), str) and dobs.strip():
                return Time(float(Time(dobs.strip()).jd), format="jd", scale="tdb")
        except (TypeError, ValueError):
            continue
    return None


def _prepare_prev_bkg_cov(
    hdul: fits.HDUList, was_sub: bool
) -> tuple[BkgCovariance | None, np.ndarray | None]:
    cov = read_bkg_covariance(hdul, apply_quality_mask=False) if was_sub else None
    if cov is None:
        return None, None
    p, ny, nx = cov.basis.shape
    return cov, np.ascontiguousarray(cov.basis.reshape(p, ny * nx), dtype=np.float64)


def _extract_unsubtracted_slice(
    data: Any, cmap: dict[str, str], was_sub: bool, slc: slice | np.ndarray, *,
    prev_cov: BkgCovariance | None = None, prev_phi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    fl = np.asarray(data[cmap["FLUX"]][slc], dtype=np.float64)
    u_s = (
        fl + np.asarray(data[cmap["FLUX_BKG"]][slc], dtype=np.float64)
        if (was_sub and "FLUX_BKG" in cmap)
        else fl
    )
    if "FLUX_ERR" not in cmap:
        return u_s, np.full_like(u_s, np.nan), None
    fe = np.asarray(data[cmap["FLUX_ERR"]][slc], dtype=np.float64)
    vfe = np.isfinite(fe) & (fe > 0.0)
    fe2 = np.where(vfe, fe * fe, np.nan)
    if was_sub and prev_cov is not None and prev_phi is not None:
        be2 = reconstruct_chol_var_slice(
            prev_cov.chol[slc], prev_phi, *prev_cov.basis.shape[1:]
        )
        ok_b = vfe & np.isfinite(be2) & (be2 >= 0.0)
        sq = np.where(ok_b, np.maximum(fe2 - be2, 0.0), fe2)
    elif was_sub and "FLUX_BKG_ERR" in cmap:
        be = np.asarray(data[cmap["FLUX_BKG_ERR"]][slc], dtype=np.float64)
        ok_b = vfe & np.isfinite(be) & (be >= 0.0)
        sq = np.where(ok_b, np.maximum(fe2 - be * be, 0.0), fe2)
    else:
        sq = fe2
    return u_s, sq, np.where(vfe, fe, np.nan)


def _compute_target_flux_corr(
    u_fn: Any, gidx: np.ndarray, coeffs: np.ndarray, b_chol: np.ndarray,
    sm_chol: np.ndarray | None, phi: np.ndarray, bmask: np.ndarray,
    pq: np.ndarray | None, excl_r: float,
) -> float:
    """Measure Pearson ``r(median(B_hat_t), F_target_t)`` and warn if ``|r| > 0.5``."""
    if gidx.size < 6:
        return float("nan")
    p, ny, nx = phi.shape
    sel = gidx[:: max(1, gidx.size // 1024)]
    cs = coeffs[sel]
    okc = np.all(np.isfinite(cs), axis=1)
    if int(np.count_nonzero(okc)) < 6:
        return float("nan")
    sel_ok, cs = sel[okc], cs[okc]
    u_sub = np.asarray(u_fn(sel_ok), dtype=np.float64)
    pf = phi.reshape(p, ny * nx)
    b_sub = (cs @ pf).reshape(-1, ny, nx)
    sub = u_sub - b_sub

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
        mu, ms = np.nanmedian(u_sub, axis=0), np.nanmedian(sub, axis=0)
    unsat = (~bmask) & np.isfinite(ms) & (mu < 7.5e4)
    if pq is not None:
        bad = np.uint16(
            PixelQuality.SAT_BLEED
            | PixelQuality.STRAP_COL
            | PixelQuality.SMEAR_UNRESOLVED
        )
        unsat &= (np.asarray(pq, dtype=np.uint16) & bad) == 0
    pos = ms[unsat]
    pos = pos[pos > 0.0]
    if pos.size == 0:
        return float("nan")
    tap = unsat & (ms >= float(np.percentile(pos, 75.0))) & (ms > 0.0)
    n_ap = int(np.count_nonzero(tap))
    if n_ap == 0:
        return float("nan")

    bm, ft = np.nanmedian(b_sub, axis=(1, 2)), np.nansum(sub[:, tap], axis=1)
    fin = np.isfinite(bm) & np.isfinite(ft)
    n_fin = int(np.count_nonzero(fin))
    if n_fin < 6:
        return float("nan")
    sb, sf = float(np.std(bm[fin])), float(np.std(ft[fin]))
    if sb <= 1e-12 or sf <= 1e-12:
        return float("nan")

    r_c = float(np.corrcoef(bm[fin], ft[fin])[0, 1])
    pm, pa = np.mean(pf, axis=1), np.mean(phi[:, tap], axis=1)
    ls = b_chol[sel_ok][fin]
    okl = np.all(np.isfinite(ls), axis=(1, 2))
    s_bm = s_ba = 0.0
    if np.any(okl):
        vbm = np.sum(np.einsum("tpq,p->tq", ls[okl], pm) ** 2, axis=1)
        vba = np.sum(np.einsum("tpq,p->tq", ls[okl], pa) ** 2, axis=1)
        if sm_chol is not None:
            sm = sm_chol[sel_ok][fin][okl]
            okm = np.all(np.isfinite(sm), axis=(1, 2))
            if np.any(okm):
                vbm[okm] += np.sum(np.einsum("tpq,p->tq", sm[okm], pm) ** 2, axis=1)
                vba[okm] += np.sum(np.einsum("tpq,p->tq", sm[okm], pa) ** 2, axis=1)
        s_bm = float(np.nanmedian(np.sqrt(np.maximum(vbm, 0.0))))
        s_ba = float(np.nanmedian(np.sqrt(np.maximum(vba, 0.0))))

    t_st = abs(r_c) * math.sqrt(max(n_fin - 2, 1) / max(1.0 - r_c * r_c, 1e-12))
    bv = s_bm <= 0.0 or sb > 2.5 * s_bm
    tv = s_ba <= 0.0 or sf > 3.0 * n_ap * s_ba
    if (
        np.isfinite(r_c)
        and abs(r_c) > _TARGET_CORR_WARN_THRESHOLD
        and t_st > 3.0
        and (r_c > 0.0 or (bv and tv))
    ):
        warnings.warn(
            f"Fitted background correlates strongly with target aperture flux "
            f"(r = {r_c:+.3f}), indicating possible PSF-wing self-subtraction. "
            f"Inspect plot_diagnostics(), increase exclusion_radius (currently "
            f"{excl_r:g} px), use order=0, or use a wider cutout.",
            UserWarning, stacklevel=4,
        )
    return r_c


def _resolve_options(
    cfg: BackgroundConfig | None, *, order: OrderSpec, exclusion_radius: float | None,
    source_mag_limit: float | None, max_source_radius: float | None,
    allow_downgrade: bool | None, verbose: bool | None,
) -> BackgroundConfig:
    b = cfg if cfg is not None else BackgroundConfig()
    er = b.exclusion_radius if exclusion_radius is None else exclusion_radius
    ml = b.source_mag_limit if source_mag_limit is None else source_mag_limit
    mr = b.max_source_radius if max_source_radius is None else max_source_radius
    ad = b.allow_downgrade if allow_downgrade is None else allow_downgrade
    return BackgroundConfig(
        order=b.order if order is None else order,
        exclusion_radius=er, source_mag_limit=ml, max_source_radius=mr,
        allow_downgrade=ad, verbose=b.verbose if verbose is None else verbose,
    )


def _solve_and_diagnose(
    u_bg: np.ndarray, sig_bg: np.ndarray | None, mres: Any,
    smear_raw: np.ndarray | None, u_fn: Any, gidx: np.ndarray,
    pq: np.ndarray | None, cfg: BackgroundConfig, was_sub: bool,
) -> BackgroundDiagnostics:
    ny, nx = mres.mask.shape
    phi_f, phi_bg, _ = equilibrate_chebyshev_basis(
        ny, nx, mres.effective_order, mres.mask
    )
    if cfg.verbose:
        n_bg = int(np.count_nonzero(mres.mask))
        print(
            f"Fitting {format_order(mres.effective_order)} background across "
            f"{gidx.size} cadences using {n_bg}/{ny * nx} px."
        )
    sol = solve_chebyshev_background(
        u_bg, phi_bg, sigma_u_bg=sig_bg, chunk_size=_IO_CHUNK,
        smear_err=smear_raw, phi_full=phi_f, bkg_mask=mres.mask,
    )
    r_c = _compute_target_flux_corr(
        u_fn, gidx, sol.coeffs, sol.bkg_chol, sol.smear_chol,
        phi_f, mres.mask, pq, cfg.exclusion_radius,
    )
    return BackgroundDiagnostics(
        mres.mask, mres.source_mask, mres.exclusion_mask, mres.requested_order,
        mres.effective_order, mres.condition_number, phi_f, sol.coeffs, sol.bkg_chol,
        sol.tau_scale, sol.n_valid_pixels, sol.n_tukey_inliers, sol.used_huber_fallback,
        was_sub, r_c, mres.dark_median_image, sol.smear_chol,
    )


def _fit_tpf_background_core(
    tpf: Any, *, cfg: BackgroundConfig, mask: np.ndarray | None = None,
    catalog_table: Any = None, gaia_sources: Any = None,
    wcs: WCS | None = None, obs_time: Time | str | float | None = None,
) -> tuple[fits.HDUList, dict[str, str], np.ndarray, BackgroundDiagnostics]:
    hdul = _unwrap_hdulist(tpf)
    if len(hdul) < 2 or hdul[1].data is None:
        raise ValueError("TPF must contain a populated PIXELS table at hdu[1].")
    was_sub = read_bkgsub_state(tpf)
    data = hdul[1].data
    cmap = {n.upper(): n for n in (data.names or ())}
    if "FLUX" not in cmap or data[cmap["FLUX"]].ndim != 3:
        raise ValueError("PIXELS HDU (hdu[1]) must contain a 3D 'FLUX' column.")
    fcol = data[cmap["FLUX"]]
    n_raw, ny, nx = fcol.shape
    pcov, pphi = _prepare_prev_bkg_cov(hdul, was_sub)

    qm = getattr(tpf, "quality_mask", None)
    use_q = qm is not None and len(qm) == n_raw
    gmask = np.asarray(qm, dtype=bool) if use_q else np.ones(n_raw, dtype=bool)
    midx = np.flatnonzero(gmask if np.any(gmask) else np.ones(n_raw, dtype=bool))
    mdt = np.float32 if fcol.dtype.itemsize <= 4 else np.float64
    u_m = np.empty((midx.size, ny, nx), dtype=mdt)
    has_fe = "FLUX_ERR" in cmap
    s_buf = np.empty((midx.size, ny, nx), dtype=mdt) if has_fe else None
    any_fe = False
    for m0 in range(0, midx.size, _IO_CHUNK):
        m1 = min(m0 + _IO_CHUNK, midx.size)
        uc, sq, fe = _extract_unsubtracted_slice(
            data, cmap, was_sub, midx[m0:m1], prev_cov=pcov, prev_phi=pphi
        )
        u_m[m0:m1] = uc
        if s_buf is not None and fe is not None:
            vfe = np.isfinite(fe) & (fe > 0.0)
            ok_v = np.isfinite(uc) & vfe & (sq > _F32_QUAD_VAR_TOL * fe * fe)
            any_fe = any_fe or bool(np.any(ok_v))
            s_floor = np.maximum(np.sqrt(sq), _F32_QUAD_AMP_FLOOR * fe)
            s_buf[m0:m1] = np.where(vfe, s_floor, np.nan)

    has_pq = "PIXEL_QUALITY" in hdul and hdul["PIXEL_QUALITY"].data is not None
    pq = np.asarray(hdul["PIXEL_QUALITY"].data, dtype=np.uint16) if has_pq else None
    srcs = catalog_table if catalog_table is not None else gaia_sources
    tcoords = _resolve_target_coords(hdul)
    use_wcs = _resolve_tpf_wcs(tpf, wcs) if (srcs is not None or tcoords) else wcs
    use_t = obs_time if obs_time is not None else (
        _resolve_tpf_obs_time(tpf) if srcs is not None else None
    )
    mres = build_background_mask(
        u_m, sigma_u=s_buf if any_fe else None, pixel_quality=pq,
        order=cfg.order, mask=mask, wcs=use_wcs, catalog_table=srcs,
        exclusion_radius=cfg.exclusion_radius, target_coords=tcoords,
        source_mag_limit=cfg.source_mag_limit,
        max_source_radius=cfg.max_source_radius, obs_time=use_t,
        allow_downgrade=cfg.allow_downgrade,
    )
    del u_m, s_buf

    bm = mres.mask
    u_bg = np.empty((n_raw, int(np.count_nonzero(bm))), dtype=np.float64)
    s_bg = np.empty_like(u_bg) if has_fe else None
    any_i = False
    for s in range(0, n_raw, _IO_CHUNK):
        e = min(s + _IO_CHUNK, n_raw)
        uc, sq, fe = _extract_unsubtracted_slice(
            data, cmap, was_sub, slice(s, e), prev_cov=pcov, prev_phi=pphi
        )
        u_bg[s:e] = uc[:, bm]
        if s_bg is not None and fe is not None:
            fb, sb = fe[:, bm], sq[:, bm]
            vfb = np.isfinite(fb) & (fb > 0.0)
            ok_i = np.isfinite(u_bg[s:e]) & vfb & (sb > _F32_QUAD_VAR_TOL * fb * fb)
            any_i = any_i or bool(np.any(ok_i))
            s_bg[s:e] = np.where(
                vfb, np.maximum(np.sqrt(sb), _F32_QUAD_AMP_FLOOR * fb), np.nan
            )

    def _sample_u(idx: np.ndarray) -> np.ndarray:
        return _extract_unsubtracted_slice(
            data, cmap, was_sub, idx, prev_cov=pcov, prev_phi=pphi
        )[0]

    diag = _solve_and_diagnose(
        u_bg, s_bg if any_i else None, mres,
        read_smear_err(hdul, apply_quality_mask=False),
        _sample_u, np.flatnonzero(gmask), pq, cfg, was_sub,
    )
    return hdul, cmap, gmask, diag


def estimate_background(
    tpf: Any, *, config: BackgroundConfig | None = None, order: OrderSpec = None,
    mask: np.ndarray | None = None, catalog_table: Any = None, gaia_sources: Any = None,
    exclusion_radius: float | None = None, wcs: WCS | None = None,
    source_mag_limit: float | None = None, max_source_radius: float | None = None,
    obs_time: Time | str | float | None = None, allow_downgrade: bool | None = None,
    flux_err: np.ndarray | Quantity | None = None,
    pixel_quality: np.ndarray | None = None, smear_err: np.ndarray | None = None,
    return_diagnostics: bool = False, verbose: bool | None = None,
) -> BackgroundEstimate | tuple[BackgroundEstimate, BackgroundDiagnostics]:
    """Estimate a 2D Chebyshev background on a TPF, ``HDUList``, or 3D array."""
    cfg = _resolve_options(
        config, order=order, exclusion_radius=exclusion_radius,
        source_mag_limit=source_mag_limit, max_source_radius=max_source_radius,
        allow_downgrade=allow_downgrade, verbose=verbose,
    )
    if isinstance(tpf, np.ndarray | Quantity):
        is_q = isinstance(tpf, Quantity)
        unit = tpf.unit if is_q else None
        ua = np.asarray(tpf.value if is_q else tpf, dtype=np.float64)
        if ua.ndim != 3:
            raise ValueError(f"Expected 3D flux array (N_cad, ny, nx), got {ua.shape}.")
        sa = None
        if flux_err is not None:
            eq_u = (
                isinstance(flux_err, Quantity)
                and unit is not None
                and flux_err.unit.is_equivalent(unit)
            )
            raw_e = (
                flux_err.to_value(unit) if eq_u
                else (flux_err.value if isinstance(flux_err, Quantity) else flux_err)
            )
            sa = np.asarray(raw_e, dtype=np.float64)
            if sa.shape != ua.shape:
                raise ValueError(f"flux_err shape {sa.shape} != flux {ua.shape}.")
        mr = build_background_mask(
            ua, sigma_u=sa, pixel_quality=pixel_quality, order=cfg.order, mask=mask,
            wcs=wcs, catalog_table=catalog_table, gaia_sources=gaia_sources,
            exclusion_radius=cfg.exclusion_radius,
            source_mag_limit=cfg.source_mag_limit,
            max_source_radius=cfg.max_source_radius, obs_time=obs_time,
            allow_downgrade=cfg.allow_downgrade,
        )
        diag = _solve_and_diagnose(
            ua[:, mr.mask], sa[:, mr.mask] if sa is not None else None, mr, smear_err,
            lambda idx: ua[idx], np.arange(ua.shape[0], dtype=np.intp),
            pixel_quality, cfg, False,
        )
        odt = np.float32 if (not is_q and tpf.dtype.itemsize <= 4) else np.float64
        br, er = reconstruct_background_chunked(
            diag.coeffs, diag.bkg_chol, diag.bkg_basis,
            smear_chol=diag.smear_chol, chunk_size=_IO_CHUNK, out_dtype=odt,
        )
        est = BackgroundEstimate(
            Quantity(br, unit) if is_q else br,
            Quantity(er, unit) if is_q else er,
            diag.bkg_mask,
        )
        return (est, diag) if return_diagnostics else est

    hdul, _, gm, diag = _fit_tpf_background_core(
        tpf, cfg=cfg, mask=mask, catalog_table=catalog_table,
        gaia_sources=gaia_sources, wcs=wcs, obs_time=obs_time,
    )
    n_el = int(np.count_nonzero(gm)) * diag.bkg_mask.size
    odt = np.float32 if n_el > 20_000_000 else np.float64
    bg, eg = reconstruct_background_chunked(
        diag.coeffs[gm], diag.bkg_chol[gm], diag.bkg_basis,
        smear_chol=diag.smear_chol[gm] if diag.smear_chol is not None else None,
        chunk_size=_IO_CHUNK, out_dtype=odt,
    )
    unit = _hdulist_column_unit(hdul, "FLUX")
    est = BackgroundEstimate(
        Quantity(bg, unit=unit), Quantity(eg, unit=unit), diag.bkg_mask
    )
    return (est, diag) if return_diagnostics else est


def _write_subtracted_columns_into_hdulist(
    hdul: fits.HDUList, cmap: dict[str, str], diag: BackgroundDiagnostics, *,
    prev_cov: BkgCovariance | None = None, prev_phi: np.ndarray | None = None,
) -> fits.HDUList:
    data = hdul[1].data
    n_raw, (p, ny, nx) = diag.coeffs.shape[0], diag.bkg_basis.shape
    pf = np.ascontiguousarray(diag.bkg_basis.reshape(p, ny * nx), dtype=np.float64)
    keys = ("FLUX", "FLUX_ERR", "FLUX_BKG", "FLUX_BKG_ERR")
    cout = {k: (data[cmap[k]] if k in cmap else None) for k in keys}
    cnew = {
        k: (None if cout[k] is not None else np.empty((n_raw, ny, nx), np.float32))
        for k in ("FLUX", "FLUX_BKG", "FLUX_BKG_ERR")
    }

    for s in range(0, n_raw, _IO_CHUNK):
        e = min(s + _IO_CHUNK, n_raw)
        uc, sq, _ = _extract_unsubtracted_slice(
            data, cmap, diag.was_bkg_subtracted, slice(s, e),
            prev_cov=prev_cov, prev_phi=prev_phi,
        )
        bc, bei = reconstruct_background_chunked(
            diag.coeffs[s:e], diag.bkg_chol[s:e], diag.bkg_basis,
            chunk_size=_IO_CHUNK, out_dtype=np.float64,
        )
        if diag.smear_chol is not None:
            sm_v = reconstruct_chol_var_slice(diag.smear_chol[s:e], pf, ny, nx)
            bet = np.sqrt(bei * bei + sm_v)
        else:
            bet = bei
        for k, vals in (("FLUX", uc - bc), ("FLUX_BKG", bc), ("FLUX_BKG_ERR", bet)):
            target = cout[k] if cout[k] is not None else cnew[k]
            assert target is not None
            target[s:e] = vals.astype(np.float32)
        if cout["FLUX_ERR"] is not None:
            cout["FLUX_ERR"][s:e] = np.where(
                np.isfinite(uc) & np.isfinite(sq), np.sqrt(sq + bei * bei), np.nan
            ).astype(np.float32)

    u_str = _flux_column_unit_str(hdul[1])
    miss = [
        fits.Column(
            name=k, format=f"{ny * nx}E", dim=f"({nx}, {ny})", unit=u_str, array=a
        )
        for k, a in cnew.items()
        if a is not None
    ]
    if miss:
        hdul[1] = fits.BinTableHDU.from_columns(
            fits.ColDefs([*hdul[1].columns, *miss]),
            header=hdul[1].header.copy(), name=hdul[1].name or "PIXELS",
        )
    return hdul


def _construct_clean_tpf(
    src: Any, hdul: fits.HDUList
) -> lk.targetpixelfile.TargetPixelFile:
    qb, tid = getattr(src, "quality_bitmask", "default"), getattr(src, "targetid", None)
    if isinstance(src, lk.TessTargetPixelFile):
        return lk.TessTargetPixelFile(hdul, quality_bitmask=qb, targetid=tid)
    if isinstance(src, lk.targetpixelfile.TargetPixelFile):
        return type(src)(hdul, quality_bitmask=qb, targetid=tid)
    return lk.TessTargetPixelFile(hdul, quality_bitmask=qb)


def subtract_background(
    tpf: Any, *, config: BackgroundConfig | None = None, order: OrderSpec = None,
    mask: np.ndarray | None = None, catalog_table: Any = None, gaia_sources: Any = None,
    exclusion_radius: float | None = None, wcs: WCS | None = None,
    source_mag_limit: float | None = None, max_source_radius: float | None = None,
    obs_time: Time | str | float | None = None, allow_downgrade: bool | None = None,
    return_diagnostics: bool = False, verbose: bool | None = None,
) -> (
    lk.targetpixelfile.TargetPixelFile
    | tuple[lk.targetpixelfile.TargetPixelFile, BackgroundDiagnostics]
):
    """Fit and subtract a 2D Chebyshev background, returning a new TPF."""
    cfg = _resolve_options(
        config, order=order, exclusion_radius=exclusion_radius,
        source_mag_limit=source_mag_limit, max_source_radius=max_source_radius,
        allow_downgrade=allow_downgrade, verbose=verbose,
    )
    sh, _, _, diag = _fit_tpf_background_core(
        tpf, cfg=cfg, mask=mask, catalog_table=catalog_table,
        gaia_sources=gaia_sources, wcs=wcs, obs_time=obs_time,
    )
    pcov, pphi = _prepare_prev_bkg_cov(sh, diag.was_bkg_subtracted)
    nh = fits.HDUList([h.copy() for h in sh])
    _write_subtracted_columns_into_hdulist(
        nh, {n.upper(): n for n in (nh[1].data.names or ())}, diag,
        prev_cov=pcov, prev_phi=pphi,
    )
    write_tessbkgd_provenance(
        nh, bkg_basis=diag.bkg_basis, bkg_chol=pack_bkg_chol(diag.bkg_chol),
        bkg_used_mask=diag.bkg_mask, bkg_order=diag.effective_order,
        source_mask=diag.source_mask if np.any(diag.source_mask) else None,
    )
    out = _construct_clean_tpf(tpf, nh)
    return (out, diag) if return_diagnostics else out


@_deprecated(_DEPRECATION_MSG)
class bkgd_tpf(lk.TessTargetPixelFile):
    """Deprecated ``TessTargetPixelFile`` shim delegating onto the v2 solver."""

    def _column_unit(self, col_name: str = "FLUX") -> u.Unit | None:
        return _hdulist_column_unit(self.hdu, col_name)

    def _assign_table_column(self, col_name: str, val: Any) -> None:
        tu = self._column_unit(col_name) or self._column_unit("FLUX")
        eq = (
            isinstance(val, Quantity)
            and val.unit not in (None, u.dimensionless_unscaled)
            and tu is not None
            and val.unit.is_equivalent(tu)
        )
        raw = (
            np.asarray(val.to_value(tu), dtype=np.float32)
            if eq
            else _strip_for_fits_column(val)
        )
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
        order: OrderSpec = "linear",
        verbose: bool = False,
    ) -> tuple[Quantity, np.ndarray]:
        est = estimate_background(
            self, order=order, mask=bkgdmask,
            allow_downgrade=(bkgdmask is None), verbose=verbose,
        )
        assert isinstance(est.bkg, Quantity)
        return est.bkg, est.mask

    @_deprecated(_DEPRECATION_MSG)
    def subtract_bkgd(
        self,
        bkgdmask: np.ndarray | None = None,
        order: OrderSpec = "linear",
        verbose: bool = False,
    ) -> tuple[Quantity, np.ndarray]:
        cfg = BackgroundConfig(
            order=order, allow_downgrade=(bkgdmask is None), verbose=verbose
        )
        hdul, cmap, gm, diag = _fit_tpf_background_core(self, cfg=cfg, mask=bkgdmask)
        pcov, pphi = _prepare_prev_bkg_cov(hdul, diag.was_bkg_subtracted)
        _write_subtracted_columns_into_hdulist(
            hdul, cmap, diag, prev_cov=pcov, prev_phi=pphi
        )
        write_tessbkgd_provenance(
            hdul, bkg_basis=diag.bkg_basis, bkg_chol=pack_bkg_chol(diag.bkg_chol),
            bkg_used_mask=diag.bkg_mask, bkg_order=diag.effective_order,
            source_mask=diag.source_mask if np.any(diag.source_mask) else None,
        )
        _sync_tpf_meta(self, hdul)
        out = Quantity(hdul[1].data["FLUX_BKG"][gm], unit=self._column_unit("FLUX"))
        return out, diag.bkg_mask

    @property
    def flux(self) -> Quantity:
        return Quantity(
            self.hdu[1].data["FLUX"][self.quality_mask],
            unit=self._column_unit("FLUX"),
        )

    @flux.setter
    def flux(self, value: Any) -> None:
        self._assign_table_column("FLUX", value)

    @property
    def flux_bkg(self) -> Quantity:
        return Quantity(
            self.hdu[1].data["FLUX_BKG"][self.quality_mask],
            unit=self._column_unit("FLUX_BKG") or self._column_unit("FLUX"),
        )

    @flux_bkg.setter
    def flux_bkg(self, value: Any) -> None:
        self._assign_table_column("FLUX_BKG", value)
        bg = self.hdu[1].data["FLUX_BKG"][self.quality_mask]
        has_nz = bool(np.any(np.isfinite(bg) & (bg != 0.0)))
        if _strip_bkg_covariance(self, has_nonzero_bkg=has_nz):
            warnings.warn(
                "Assigning bkgd_tpf.flux_bkg directly stripped stale BKG_CHOL / "
                "BKG_BASIS / BKG_USED artefacts; use tessbkgd.subtract_background(tpf) "
                "to retain calibrated covariance.",
                UserWarning, stacklevel=2,
            )
        _set_bkgsub_header(self, has_nz)

    @property
    def raw_cnts(self) -> np.ndarray:
        return self.hdu[1].data["RAW_CNTS"][self.quality_mask]
