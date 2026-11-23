"""Dark-cadence detrended sky masking, exclusion zones, and anisotropic guards."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.units import Quantity
from astropy.wcs import WCS
from erfa import ErfaWarning
from scipy.ndimage import binary_dilation, label

from ._contract import PixelQuality

ORDER_NAMES: dict[int, str] = {0: "pedestal", 1: "linear", 2: "quadratic", 3: "cubic"}
_NAME_TO_ORDER: dict[str, int] = {v: k for k, v in ORDER_NAMES.items()}
OrderSpec = int | str | tuple[int, int] | None

DEFAULT_EXCLUSION_RADIUS = 15.0
_SATURATION_FLUX_THRESHOLD = 7.5e4
_BLEED_LATERAL_BUFFER = 2
_MAX_PRE_QR_KAPPA = 100.0
_MAX_INTERIOR_GAP_FRACTION = 0.5
_MIN_SURVIVING_PIXELS = 3
_MIN_WARN_PIXELS = 20
_CADENCE_CHUNK = 8192
_MAX_TEMPORAL_CADENCES = 4096

_MAD_TO_SIGMA = 1.482602218505602
_Q25_LOWER_TO_SIGMA = 2.101480909693451
_TESS_ZERO_POINT_MAG = 20.44
_TESS_PRF_RC_PX = 0.85
_TESS_PRF_BETA = 2.2
_PIXEL_DISK_RADIUS_PX = 1.0 / math.sqrt(math.pi)


@dataclass(frozen=True)
class BackgroundMaskResult:
    """Result of automatic or validated background pixel masking."""

    mask: np.ndarray
    source_mask: np.ndarray
    exclusion_mask: np.ndarray
    requested_order: tuple[int, int]
    effective_order: tuple[int, int]
    condition_number: float
    dark_median_image: np.ndarray | None
    coarse_surface: np.ndarray | None
    sky_sigma: float | None


def _parse_single_order(order: Any) -> int:
    if not isinstance(order, bool | np.bool_):
        if isinstance(order, int | np.integer) and int(order) in ORDER_NAMES:
            return int(order)
        if isinstance(order, str) and order.strip().lower() in _NAME_TO_ORDER:
            return _NAME_TO_ORDER[order.strip().lower()]
    raise ValueError(
        f"Unrecognised order {order!r}. Expected 0/'pedestal', "
        "1/'linear', 2/'quadratic', 3/'cubic', or (order_y, order_x)."
    )


def _normalize_order_pair(order: int | str | tuple[int, int]) -> tuple[int, int]:
    if isinstance(order, tuple | list) and not isinstance(order, str):
        if len(order) != 2:
            raise ValueError(f"Expected (order_y, order_x) 2-tuple, got {order!r}.")
        return _parse_single_order(order[0]), _parse_single_order(order[1])
    d = _parse_single_order(order)
    return (d, d)


def format_order(order: int | tuple[int, int]) -> str:
    oy, ox = _normalize_order_pair(order)
    return ORDER_NAMES[oy] if oy == ox else f"({oy}, {ox})"


def parse_order(order: OrderSpec, ny: int, nx: int) -> tuple[int, int]:
    """Resolve ``order`` to ``(order_y, order_x)``, defaulting per axis (>=40 -> 2)."""
    if order is None:
        return (2 if ny >= 40 else 1, 2 if nx >= 40 else 1)
    return _normalize_order_pair(order)


def _chebyshev_mode_pairs(oy: int, ox: int) -> list[tuple[int, int]]:
    d_max = max(oy, ox)
    return [
        (s - q, q)
        for s in range(d_max + 1)
        for q in range(s + 1)
        if (s - q) <= ox and q <= oy
    ]


def n_basis_for_order(order: int | str | tuple[int, int]) -> int:
    """Return 2D Chebyshev basis count ``P`` for ``order`` or ``(order_y, order_x)``."""
    return len(_chebyshev_mode_pairs(*_normalize_order_pair(order)))


def normalized_grid(ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized 1D coordinates ``(eta_r, xi_c)`` in ``[-1, 1]``."""
    if ny < 1 or nx < 1:
        raise ValueError(f"Grid dimensions must be >= 1, got ({ny}, {nx}).")
    eta = (2.0 * np.arange(ny, dtype=np.float64) - (ny - 1)) / max(ny - 1, 1)
    xi = (2.0 * np.arange(nx, dtype=np.float64) - (nx - 1)) / max(nx - 1, 1)
    return eta, xi


def _chebyshev_1d(x: np.ndarray, max_deg: int) -> np.ndarray:
    out = np.empty((max_deg + 1, x.size), dtype=np.float64)
    out[0] = 1.0
    if max_deg >= 1:
        out[1] = x
    for k in range(2, max_deg + 1):
        out[k] = 2.0 * x * out[k - 1] - out[k - 2]
    return out


def raw_chebyshev_2d(
    ny: int, nx: int, order: int | str | tuple[int, int]
) -> np.ndarray:
    """Evaluate raw 2D Chebyshev basis ``(P, ny, nx)`` for ``(order_y, order_x)``."""
    oy, ox = _normalize_order_pair(order)
    pairs = _chebyshev_mode_pairs(oy, ox)
    eta, xi = normalized_grid(ny, nx)
    te, tx = _chebyshev_1d(eta, oy), _chebyshev_1d(xi, ox)
    basis = np.empty((len(pairs), ny, nx), dtype=np.float64)
    for i, (p, q) in enumerate(pairs):
        basis[i] = np.outer(te[q], tx[p])
    return basis


def _pre_qr_condition_number(phi_bg: np.ndarray) -> float:
    if phi_bg.shape[0] < phi_bg.shape[1]:
        return float("inf")
    sv = np.linalg.svd(phi_bg, compute_uv=False)
    if sv.size == 0 or sv[-1] <= 0.0 or not np.all(np.isfinite(sv)):
        return float("inf")
    return float(sv[0] / sv[-1])


def _wide_interior_gaps(
    mask_bool: np.ndarray, *, max_gap_fraction: float = _MAX_INTERIOR_GAP_FRACTION
) -> tuple[bool, bool]:
    """Return ``(wide_y_gap, wide_x_gap)`` when an interior gap spans ``> 50%``."""
    ny, nx = mask_bool.shape
    out: list[bool] = []
    for occ, n_ax in ((np.any(mask_bool, 1), ny), (np.any(mask_bool, 0), nx)):
        idx = np.flatnonzero(occ)
        out.append(
            n_ax >= 4
            and idx.size >= 2
            and int(np.max(np.diff(idx) - 1)) > max_gap_fraction * n_ax
        )
    return out[0], out[1]


def _has_wide_interior_gap(
    mask_bool: np.ndarray, *, max_gap_fraction: float = _MAX_INTERIOR_GAP_FRACTION
) -> bool:
    wy, wx = _wide_interior_gaps(mask_bool, max_gap_fraction=max_gap_fraction)
    return wy or wx


def _step_down_order(oy: int, ox: int, mask_bool: np.ndarray) -> tuple[int, int]:
    if oy > ox:
        return (oy - 1, ox)
    if ox > oy:
        return (oy, ox - 1)
    if oy >= 1:
        ny, nx = mask_bool.shape
        ky = _pre_qr_condition_number(
            raw_chebyshev_2d(ny, nx, (oy, oy - 1))[:, mask_bool].T
        )
        kx = _pre_qr_condition_number(
            raw_chebyshev_2d(ny, nx, (oy - 1, oy))[:, mask_bool].T
        )
        if np.isfinite(ky) and ky <= _MAX_PRE_QR_KAPPA and (
            not np.isfinite(kx) or ky <= kx
        ):
            return (oy, oy - 1)
        if np.isfinite(kx) and kx <= _MAX_PRE_QR_KAPPA:
            return (oy - 1, oy)
    return (max(oy - 1, 0), max(ox - 1, 0))


def apply_curvature_and_rank_guards(
    mask: np.ndarray,
    requested_order: int | str | tuple[int, int],
    *,
    allow_downgrade: bool = True,
    warn_on_downgrade: bool = True,
) -> tuple[tuple[int, int], float]:
    """Validate pixel count and conditioning, downgrading per axis with a warning."""
    mb = np.asarray(mask, dtype=bool)
    ny, nx = mb.shape
    n_bg = int(np.count_nonzero(mb))
    ry, rx = _normalize_order_pair(requested_order)
    n_req = n_basis_for_order((ry, rx))

    if not allow_downgrade:
        min_n = max(_MIN_SURVIVING_PIXELS, n_req)
        if n_bg < min_n:
            raise ValueError(
                f"Insufficient finite background pixels ({n_bg}) for order "
                f"{format_order((ry, rx))} (requires >= {min_n})."
            )
        kap = _pre_qr_condition_number(raw_chebyshev_2d(ny, nx, (ry, rx))[:, mb].T)
        if not np.isfinite(kap) or kap > 1e10:
            raise ValueError(
                f"Background mask ({n_bg} px) is rank-deficient for order "
                f"{format_order((ry, rx))} (kappa={kap:.3g})."
            )
        return (ry, rx), kap

    min_s = 1 if ny * nx < 6 else _MIN_SURVIVING_PIXELS
    if n_bg < min_s:
        raise ValueError(f"Insufficient finite background pixels ({n_bg} < {min_s}).")

    wy, wx = _wide_interior_gaps(mb)
    oy = 0 if ny == 1 else (min(ry, 1) if wy else ry)
    ox = 0 if nx == 1 else (min(rx, 1) if wx else rx)
    eff, kappa = (0, 0), 1.0
    while oy > 0 or ox > 0:
        p = n_basis_for_order((oy, ox))
        if n_bg >= (max(5 * p, 30) if max(oy, ox) >= 2 else 10):
            kap = _pre_qr_condition_number(raw_chebyshev_2d(ny, nx, (oy, ox))[:, mb].T)
            if np.isfinite(kap) and kap <= _MAX_PRE_QR_KAPPA:
                eff, kappa = (oy, ox), kap
                break
        oy, ox = _step_down_order(oy, ox, mb)

    if warn_on_downgrade and eff != (ry, rx):
        warnings.warn(
            f"Background polynomial order downgraded from {(ry, rx)} to {eff} "
            f"({n_bg} px, kappa={kappa:.1f}; wide_gap_y={wy}, wide_gap_x={wx}).",
            UserWarning, stacklevel=3,
        )
    return eff, kappa


def _col_to_float64(
    col: Any, tgt: u.UnitBase | None = None, *, allow_mag: bool = False
) -> np.ndarray:
    unit, ma = getattr(col, "unit", None), getattr(col, "mask", None)
    raw = col.value if isinstance(col, Quantity) else col
    vals = np.asarray(np.ma.getdata(raw), dtype=np.float64)
    if ma is not None and ma is not np.ma.nomask:
        bm = np.broadcast_to(np.asarray(ma, dtype=bool), vals.shape)
        vals = np.where(bm, np.nan, vals)
    vals = vals.ravel()
    if tgt is None or unit is None or unit == u.dimensionless_unscaled:
        return vals
    if unit.is_equivalent(tgt):
        q = Quantity(vals, unit, copy=False).to_value(tgt)
        return np.asarray(q, dtype=np.float64)
    if allow_mag and isinstance(unit, u.MagUnit):
        return vals
    raise u.UnitConversionError(f"Column unit {unit!r} incompatible with {tgt}.")


def _parse_epoch_array(col: Any, default_ep: float) -> np.ndarray:
    raw = np.asarray(np.ma.getdata(col.value if isinstance(col, Quantity) else col))
    if raw.dtype.kind in ("U", "S", "O"):
        vals = np.full(raw.size, np.nan, dtype=np.float64)
        for i, it in enumerate(raw.ravel()):
            if it is not None:
                s = (
                    it.decode("utf-8", "ignore") if isinstance(it, bytes) else str(it)
                ).strip()
                try:
                    vals[i] = float(s[1:].strip() if s[:1] in "JjBb" else s)
                except ValueError:
                    pass
        ma = getattr(col, "mask", None)
        if ma is not None and ma is not np.ma.nomask:
            bm = np.broadcast_to(np.asarray(ma, dtype=bool), raw.shape).ravel()
            vals = np.where(bm, np.nan, vals)
    else:
        unit = getattr(col, "unit", None)
        if unit not in (None, u.dimensionless_unscaled, u.yr, u.a, u.d):
            raise u.UnitConversionError(f"ref_epoch unit {unit!r} is not year or day.")
        vals = _col_to_float64(col, u.yr if unit in (u.yr, u.a) else None)

    out = np.full(vals.shape, default_ep, dtype=np.float64)
    fin = np.isfinite(vals)
    jy = fin & (vals >= 1900.0) & (vals <= 2200.0)
    mjd = fin & (vals >= 3e4) & (vals <= 1e5)
    jd = fin & (vals > 2.4e6)
    out[jy] = vals[jy]
    if np.any(mjd):
        out[mjd] = Time(vals[mjd], format="mjd", scale="tdb").jyear
    if np.any(jd):
        out[jd] = Time(vals[jd], format="jd", scale="tdb").jyear
    return out


def _coerce_obs_time(
    obs_time: Time | str | float | None, wcs: WCS | None
) -> Time | None:
    if obs_time is not None:
        if isinstance(obs_time, Time):
            jd = np.asarray(obs_time.tdb.jd, dtype=np.float64).ravel()
            fin = jd[np.isfinite(jd)]
            if fin.size:
                return Time(float(np.median(fin)), format="jd", scale="tdb")
            return None
        if isinstance(obs_time, bool | np.bool_):
            raise ValueError(f"Invalid obs_time={obs_time!r}.")
        if isinstance(obs_time, int | float | np.integer | np.floating):
            v = float(obs_time)
            if not np.isfinite(v):
                return None
            ranges = ((1900, 2200, "jyear"), (3e4, 1e5, "mjd"), (2.4e6, np.inf, "jd"))
            for lo, hi, fmt in ranges:
                if lo <= v <= hi:
                    return Time(v, format=fmt, scale="tdb")
            raise ValueError(
                f"Ambiguous obs_time={v!r}; pass astropy.time.Time(..., format='btjd')."
            )
        if isinstance(obs_time, str):
            s = obs_time.strip()
            return Time(float(Time(s).tdb.jd), format="jd", scale="tdb") if s else None
        raise TypeError(f"Unsupported obs_time type {type(obs_time).__name__}.")

    if wcs is not None and getattr(wcs, "wcs", None) is not None:
        mjd = getattr(wcs.wcs, "mjdobs", np.nan)
        if np.isfinite(mjd) and float(mjd) > 0.0:
            return Time(float(mjd), format="mjd", scale="tdb")
        dobs = getattr(wcs.wcs, "dateobs", "")
        if isinstance(dobs, str) and dobs.strip():
            try:
                return Time(float(Time(dobs.strip()).tdb.jd), format="jd", scale="tdb")
            except ValueError:
                return None
    return None


def _extract_source_catalogue(
    cat: Any,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray | None, np.ndarray | None, np.ndarray | None,
]:
    emp = np.empty(0, dtype=np.float64)
    if cat is None:
        return emp, emp, emp, None, None, None

    cols: dict[str, Any] | None = None
    if isinstance(cat, Mapping):
        cols = {str(k).lower(): cat[k] for k in cat}
    elif hasattr(cat, "colnames"):
        cols = {str(k).lower(): cat[k] for k in cat.colnames}
    elif hasattr(cat, "columns") and hasattr(cat, "__getitem__"):
        cols = {str(k).lower(): cat[k] for k in cat.columns}
    elif isinstance(cat, np.ndarray) and cat.dtype.names:
        cols = {str(k).lower(): cat[k] for k in cat.dtype.names}

    if cols is not None:
        rk = next(
            (k for k in ("ra", "ra_icrs", "raj2000", "ra_deg") if k in cols), None
        )
        dk = next(
            (k for k in ("dec", "de_icrs", "dej2000", "dec_deg") if k in cols), None
        )
        if rk is None or dk is None:
            raise ValueError("catalog_table must include 'ra' and 'dec' columns.")
        ra, dec = _col_to_float64(cols[rk], u.deg), _col_to_float64(cols[dk], u.deg)
        mag_keys = (
            "mag", "tmag", "gmag", "phot_g_mean_mag", "phot_rp_mean_mag", "vmag", "v"
        )
        mk = next((k for k in mag_keys if k in cols), None)
        mag = (
            np.full(ra.shape, 14.0)
            if mk is None
            else _col_to_float64(cols[mk], u.mag, allow_mag=True)
        )
        if ra.shape != dec.shape or ra.shape != mag.shape:
            raise ValueError("catalog_table 'ra', 'dec', 'mag' lengths must match.")

        prk = next(
            (k for k in ("pmra", "pm_ra_cosdec", "pm_ra", "pmra_mas_yr") if k in cols),
            None,
        )
        pdk = next(
            (
                k
                for k in ("pmdec", "pm_dec", "pmde", "pm_de", "pmdec_mas_yr")
                if k in cols
            ),
            None,
        )
        pmra = pmdec = ref_ep = None
        if (prk is None) != (pdk is None):
            warnings.warn(
                "catalog_table has only one proper-motion component; skipping PM.",
                UserWarning, stacklevel=3,
            )
        elif prk is not None and pdk is not None:
            pmra = _col_to_float64(cols[prk], u.mas / u.yr)
            pmdec = _col_to_float64(cols[pdk], u.mas / u.yr)
            if pmra.shape != ra.shape or pmdec.shape != ra.shape:
                raise ValueError("catalog_table PM columns must match 'ra' length.")
            g_keys = ("phot_g_mean_mag", "phot_rp_mean_mag", "phot_bp_mean_mag")
            is_g = ("source_id" in cols or any(k in cols for k in g_keys)) and (
                "tmag" not in cols
            )
            if rk == "raj2000" or dk == "dej2000":
                dep = 2000.0
            elif rk == "ra_icrs" or dk == "de_icrs" or is_g:
                dep = 2016.0
            else:
                dep = 2000.0
            ek = next(
                (k for k in ("ref_epoch", "epoch", "epoch_ra") if k in cols), None
            )
            ref_ep = (
                _parse_epoch_array(cols[ek], dep)
                if ek is not None
                else np.full(ra.shape, dep)
            )
            if ref_ep.shape != ra.shape:
                raise ValueError("catalog_table 'ref_epoch' must match 'ra' length.")
        return ra, dec, mag, pmra, pmdec, ref_ep

    if isinstance(cat, Sequence | np.ndarray):
        raw_c = cat.value if isinstance(cat, Quantity) else cat
        arr = np.asarray(np.ma.getdata(raw_c), dtype=np.float64)
        ma = getattr(cat, "mask", None)
        if ma is not None and ma is not np.ma.nomask:
            bm = np.broadcast_to(np.asarray(ma, dtype=bool), arr.shape)
            arr = np.where(bm, np.nan, arr)
        if arr.size == 0:
            return emp, emp, emp, None, None, None
        arr = arr[None, :] if (arr.ndim == 1 and arr.size in (2, 3)) else arr
        if arr.ndim != 2 or arr.shape[1] not in (2, 3):
            raise ValueError(
                f"Sequence catalog_table must be (N, 2) or (N, 3), got {arr.shape}."
            )
        su = getattr(cat, "unit", None)
        if su is not None and su != u.dimensionless_unscaled:
            if arr.shape[1] == 3:
                raise u.UnitConversionError(
                    "Angular Quantity array cannot hold a magnitude column."
                )
            ra = _col_to_float64(Quantity(arr[:, 0], su), u.deg)
            dec = _col_to_float64(Quantity(arr[:, 1], su), u.deg)
        else:
            ra, dec = arr[:, 0], arr[:, 1]
        mag = arr[:, 2] if arr.shape[1] == 3 else np.full(ra.shape, 14.0)
        return ra, dec, mag, None, None, None

    raise TypeError(f"Unsupported catalog_table type {type(cat).__name__}.")


def build_wcs_source_mask(
    ny: int, nx: int, wcs: WCS | None, catalog_table: Any = None, *,
    gaia_sources: Any = None, mag_limit: float = 19.0, sky_sigma: float = 1.0,
    min_radius_px: float = 1.25, max_radius_px: float | None = None,
    exclusion_radius: float = DEFAULT_EXCLUSION_RADIUS,
    obs_time: Time | str | float | None = None,
) -> np.ndarray:
    """Project catalogue sources (with optional PM propagation) onto circular masks."""
    srcs = catalog_table if catalog_table is not None else gaia_sources
    ra, dec, mag, pmra, pmdec, rep = _extract_source_catalogue(srcs)
    mask = np.zeros((ny, nx), dtype=bool)
    if ra.size == 0:
        return mask
    if wcs is None or not getattr(wcs, "has_celestial", False):
        raise ValueError("A celestial WCS is required when catalog_table is given.")

    v = np.isfinite(ra) & np.isfinite(dec)
    if not np.any(v):
        return mask
    ra, dec = ra[v], dec[v]
    mag = np.where(np.isfinite(mag[v]), mag[v], 14.0)
    kp = mag <= mag_limit
    if not np.any(kp):
        return mask
    ra, dec, mag = ra[kp], dec[kp], mag[kp]

    tobs = _coerce_obs_time(obs_time, wcs)
    if pmra is not None and pmdec is not None and rep is not None and tobs is not None:
        pr, pd, re_a = pmra[v][kp], pmdec[v][kp], rep[v][kp]
        hp = np.isfinite(pr) & np.isfinite(pd) & np.isfinite(re_a)
        if np.any(hp):
            sc = SkyCoord(
                ra=ra[hp] * u.deg, dec=dec[hp] * u.deg,
                pm_ra_cosdec=pr[hp] * (u.mas / u.yr), pm_dec=pd[hp] * (u.mas / u.yr),
                obstime=Time(re_a[hp], format="jyear", scale="tdb"), frame="icrs",
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", ".*distance overridden.*", ErfaWarning
                )
                sc_o = sc.apply_space_motion(new_obstime=tobs)
            ra, dec = ra.copy(), dec.copy()
            ra[hp], dec[hp] = sc_o.ra.deg, sc_o.dec.deg

    sc_all = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    cx, ry = wcs.celestial.world_to_pixel(sc_all)
    cx, ry = np.asarray(cx, dtype=np.float64), np.asarray(ry, dtype=np.float64)
    gam = (
        (10.0 ** (-0.4 * (mag - _TESS_ZERO_POINT_MAG)))
        * (_TESS_PRF_BETA - 1.0)
        / (math.pi * _TESS_PRF_RC_PX**2 * max(float(sky_sigma), 1e-6))
    )
    moff = _TESS_PRF_RC_PX**2 * (np.maximum(gam, 1.0) ** (1.0 / _TESS_PRF_BETA) - 1.0)
    r2 = np.where(gam > 1.0, moff, 0.0)
    hi = (
        max(float(exclusion_radius), 25.0)
        if max_radius_px is None
        else max(float(max_radius_px), 0.0)
    )
    lo_r = min(max(min_radius_px, 0.0), hi)
    rads = np.clip(np.sqrt(r2 + _PIXEL_DISK_RADIUS_PX**2), lo_r, hi)

    near = (
        np.isfinite(cx) & np.isfinite(ry)
        & (cx >= -rads) & (cx <= nx - 1 + rads)
        & (ry >= -rads) & (ry <= ny - 1 + rads)
    )
    if np.any(near):
        rr, cc = np.meshgrid(
            np.arange(ny, dtype=np.float64),
            np.arange(nx, dtype=np.float64),
            indexing="ij",
        )
        for x0, y0, rad in zip(cx[near], ry[near], rads[near], strict=True):
            mask |= (cc - x0) ** 2 + (rr - y0) ** 2 <= rad * rad
    return mask


def _build_saturated_exclusion_mask(
    u_cube: np.ndarray, pq: np.ndarray | None, *, exclusion_radius: float,
    wcs: WCS | None = None, target_coords: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build circular ``exclusion_radius`` + ``+/-2``-col bleed exclusion mask."""
    n_cad, ny, nx = u_cube.shape
    excl = np.zeros((ny, nx), dtype=bool)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
        stride = max(1, n_cad // 256)
        med = np.asarray(np.nanmedian(u_cube[::stride], axis=0), dtype=np.float64)

    sat_f = np.isfinite(med) & (med >= _SATURATION_FLUX_THRESHOLD)
    sat_b = (
        (np.asarray(pq, dtype=np.uint16) & np.uint16(PixelQuality.SAT_BLEED)) != 0
        if pq is not None
        else np.zeros((ny, nx), dtype=bool)
    )
    sat_c = sat_f | sat_b
    if exclusion_radius <= 0.0 or not np.any(sat_c):
        return excl, sat_c

    if np.any(sat_b):
        st = np.ones((1, 2 * _BLEED_LATERAL_BUFFER + 1), dtype=bool)
        excl |= binary_dilation(sat_b, structure=st)

    rr, cc = np.meshgrid(
        np.arange(ny, dtype=np.float64), np.arange(nx, dtype=np.float64), indexing="ij"
    )
    r2 = float(exclusion_radius) ** 2
    lbl, n_comp = label(sat_c)
    for cid in range(1, n_comp + 1):
        comp = lbl == cid
        cp = sat_f & comp
        if np.any(cp):
            w = np.maximum(np.where(cp, med, 0.0), 0.0)
            ws = float(np.sum(w))
            if ws > 0:
                cy, cx = float(np.sum(rr * w) / ws), float(np.sum(cc * w) / ws)
            else:
                cy, cx = float(np.mean(rr[cp])), float(np.mean(cc[cp]))
        else:
            cy, cx = float(np.mean(rr[comp])), float(np.mean(cc[comp]))
        excl |= (rr - cy) ** 2 + (cc - cx) ** 2 <= r2

    has_w = target_coords is not None and wcs is not None and getattr(
        wcs, "has_celestial", False
    )
    if has_w:
        assert target_coords is not None and wcs is not None
        rt, dt = target_coords
        if np.isfinite(rt) and np.isfinite(dt):
            try:
                ct, yt = wcs.celestial.world_to_pixel(
                    SkyCoord(ra=rt * u.deg, dec=dt * u.deg, frame="icrs")
                )
                cf, yf = float(ct), float(yt)
                in_x = -exclusion_radius <= cf <= nx - 1 + exclusion_radius
                in_y = -exclusion_radius <= yf <= ny - 1 + exclusion_radius
                if np.isfinite(cf) and np.isfinite(yf) and in_x and in_y:
                    excl |= (rr - yf) ** 2 + (cc - cf) ** 2 <= r2
            except Exception:
                pass
    return excl, sat_c


def _qr_equilibrate(design: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if design.shape[0] < design.shape[1]:
        return None
    q, r = np.linalg.qr(design, mode="reduced")
    d = np.diag(r)
    md = float(np.max(np.abs(d))) if d.size else 0.0
    if md <= 0.0 or float(np.min(np.abs(d))) <= 1e-10 * md:
        return None
    s = np.where(d >= 0.0, 1.0, -1.0)
    return q * s[None, :], s[:, None] * r


def _quantile_irls(
    design: np.ndarray, y: np.ndarray, *, tau: float = 0.25
) -> np.ndarray | None:
    qr = _qr_equilibrate(design)
    if qr is None:
        return None
    q, rm = qr
    th = np.zeros(q.shape[1], dtype=np.float64)
    if abs(float(np.mean(q[:, 0]))) > 1e-12:
        th[0] = float(np.quantile(y, tau)) / float(np.mean(q[:, 0]))
    my = float(np.median(y))
    mad_y = _MAD_TO_SIGMA * float(np.median(np.abs(y - my)))
    dl = max(1e-3 * mad_y, 1e-8 * max(abs(my), 1.0), 1e-10)
    for _ in range(20):
        res = y - q @ th
        w = np.where(res >= 0.0, tau, 1.0 - tau) / np.maximum(np.abs(res), dl)
        try:
            th = np.linalg.solve((q * w[:, None]).T @ q, q.T @ (w * y))
        except np.linalg.LinAlgError:
            break
    return np.linalg.solve(rm, th)


def _select_tile_anchors(
    med_img: np.ndarray, elig: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ny, nx = med_img.shape
    md = min(ny, nx)
    tl = 3 if md < 20 else (4 if md < 40 else max(5, min(8, md // 8)))
    rbs = np.array_split(np.arange(ny), max(2, ny // tl))
    cbs = np.array_split(np.arange(nx), max(2, nx // tl))
    ar, ac, ed = [], [], []
    for iy, rb in enumerate(rbs):
        for ix, cb in enumerate(cbs):
            sub = elig[np.ix_(rb, cb)]
            if np.any(sub):
                lr, lc = np.nonzero(sub)
                vs = med_img[np.ix_(rb, cb)][lr, lc]
                pk = np.argsort(vs, kind="mergesort")[int(round(0.15 * (vs.size - 1)))]
                ar.append(int(rb[lr[pk]]))
                ac.append(int(cb[lc[pk]]))
                ed.append(iy in (0, len(rbs) - 1) or ix in (0, len(cbs) - 1))
    return (
        np.asarray(ar, dtype=np.intp),
        np.asarray(ac, dtype=np.intp),
        np.asarray(ed, dtype=bool),
    )


def _huber_detrend_batch(
    q: np.ndarray, y: np.ndarray, fok: np.ndarray,
    sok: np.ndarray, w0: np.ndarray, flr: float,
) -> np.ndarray:
    w, eye = w0, np.eye(q.shape[1], dtype=np.float64)[None, :, :]
    th = np.zeros((y.shape[0], q.shape[1]), dtype=np.float64)
    for st in range(3):
        try:
            th = np.linalg.solve(
                np.einsum("mp,tm,mq->tpq", q, w, q) + 1e-15 * eye,
                np.einsum("mp,tm->tp", q, w * y)[..., None],
            )[..., 0]
        except np.linalg.LinAlgError:
            break
        if st < 2:
            rs = (y - th @ q.T) / sok
            rn = np.where(fok, rs, np.nan)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
                mt = np.nanmedian(rn, axis=1, keepdims=True)
                mad = _MAD_TO_SIGMA * np.nanmedian(
                    np.abs(rn - mt), axis=1, keepdims=True
                )
                sc = np.maximum(mad, flr)
            w = w0 * np.minimum(1.0, 1.345 / np.maximum(np.abs(rs) / sc, 1e-30))
    return th


def _compute_aligned_dark_median(
    dcube: np.ndarray, elig: np.ndarray, deg_p0: tuple[int, int]
) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
        rmed = np.asarray(np.nanmedian(dcube, axis=0), dtype=np.float64)
    nd, ny, nx = dcube.shape
    live = elig & np.isfinite(rmed)
    if nd < 4 or int(np.count_nonzero(live)) < 12:
        return rmed
    ar, ac, _ = _select_tile_anchors(rmed, live)
    if ar.size < 6:
        return rmed
    ya = np.asarray(dcube[:, ar, ac], dtype=np.float64)
    af = np.all(np.isfinite(ya), axis=0)
    oy, ox = deg_p0
    while oy > 0 or ox > 0:
        if int(np.count_nonzero(af)) >= max(2 * n_basis_for_order((oy, ox)), 6):
            bas = raw_chebyshev_2d(ny, nx, (oy, ox))
            qr = _qr_equilibrate(bas[:, ar[af], ac[af]].T)
            if qr is not None:
                qa, ra_m = qr
                dy = ya[:, af] - np.median(ya[:, af], axis=0, keepdims=True)
                ones = np.ones_like(dy)
                th = _huber_detrend_batch(
                    qa, dy, np.ones(dy.shape, dtype=bool), ones, ones, 1e-6
                )
                th -= np.median(th, axis=0, keepdims=True)
                coef = np.linalg.solve(ra_m, th.T).T
                al = dcube - np.tensordot(coef, bas, axes=(1, 0))
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
                    return np.asarray(np.nanmedian(al, axis=0), dtype=np.float64)
        if oy > ox:
            oy -= 1
        elif ox > oy:
            ox -= 1
        else:
            oy, ox = oy - 1, ox - 1
    return rmed


def _fit_coarse_surface_p0(
    mimg: np.ndarray, elig: np.ndarray, deg_p0: tuple[int, int], *,
    allow_downgrade: bool = True,
) -> tuple[np.ndarray, float]:
    ny, nx = mimg.shape
    yel = mimg[elig]
    ulp = max(32.0 * float(np.median(np.abs(np.spacing(yel.astype(np.float32))))), 1e-8)
    if (allow_downgrade and ny * nx < 6) or yel.size < 6 or deg_p0 == (0, 0):
        med = float(np.median(yel))
        sig = max(_MAD_TO_SIGMA * float(np.median(np.abs(yel - med))), ulp)
        return np.full((ny, nx), med), sig

    ar, ac, ied = _select_tile_anchors(mimg, elig)
    oy, ox = deg_p0
    while oy > 0 or ox > 0:
        np_c = n_basis_for_order((oy, ox))
        bt = raw_chebyshev_2d(ny, nx, (oy, ox))
        if ar.size >= np_c and _qr_equilibrate(bt[:, ar, ac].T) is not None:
            break
        if yel.size >= np_c and _qr_equilibrate(bt[:, elig].T) is not None:
            ar, ac = np.nonzero(elig)
            ied = np.ones(ar.size, dtype=bool)
            break
        oy, ox = _step_down_order(oy, ox, elig)

    b0 = raw_chebyshev_2d(ny, nx, (oy, ox))
    del_m, ya = b0[:, elig].T, mimg[ar, ac]
    fa = np.ones(ar.size, dtype=bool)
    if max(oy, ox) >= 2 and ar.size >= 8:
        bp = raw_chebyshev_2d(ny, nx, (min(oy, 1), min(ox, 1)))
        bpl = _quantile_irls(bp[:, ar, ac].T, ya)
        if bpl is not None:
            rpl = ya - bp[:, ar, ac].T @ bpl
            er = rpl[ied] if np.any(ied) else rpl
            em = float(np.median(er))
            p85 = float(np.percentile(np.abs(er - em), 85.0))
            mad_e = 6.0 * _MAD_TO_SIGMA * float(np.median(np.abs(er - em)))
            sp = max(p85, mad_e, ulp)
            tr = (rpl - em) <= 4.0 * sp
            if (
                int(np.count_nonzero(tr)) >= b0.shape[0]
                and _qr_equilibrate(b0[:, ar[tr], ac[tr]].T) is not None
            ):
                fa = tr

    beta = _quantile_irls(b0[:, ar[fa], ac[fa]].T, ya[fa])
    if beta is None:
        beta = np.zeros(b0.shape[0], dtype=np.float64)
        beta[0] = float(np.quantile(yel, 0.25))

    r0 = yel - del_m @ beta
    q25 = float(np.percentile(r0, 25.0))
    l25 = q25 - r0[r0 <= q25]
    s0 = max(_Q25_LOWER_TO_SIGMA * float(np.median(l25)) if l25.size else 0.0, ulp)
    inl = (r0 >= q25 - 3.5 * s0) & (r0 <= q25 + 3.7 * s0)
    if int(np.count_nonzero(inl)) < b0.shape[0]:
        inl = r0 <= float(np.percentile(r0, 50.0))

    for _ in range(3):
        qi = _qr_equilibrate(del_m[inl])
        if qi is None:
            break
        beta = np.linalg.solve(qi[1], qi[0].T @ yel[inl])
        res = yel - del_m @ beta
        rm = float(np.median(res[inl]))
        lw = rm - res[inl][res[inl] <= rm]
        sg = max(_MAD_TO_SIGMA * float(np.median(lw)) if lw.size else 0.0, ulp)
        nxt = (res - rm >= -4.0 * sg) & (res - rm <= 3.0 * sg)
        if int(np.count_nonzero(nxt)) < b0.shape[0] or np.array_equal(nxt, inl):
            break
        inl = nxt

    rs = (yel - del_m @ beta)[inl]
    rm = float(np.median(rs))
    beta[0] += rm
    lf = -(rs - rm)[rs <= rm]
    sig_f = max(_MAD_TO_SIGMA * float(np.median(lf)) if lf.size else 0.0, ulp)
    return np.tensordot(beta, b0, axes=(0, 0)), sig_f


def _reject_anomalous_temporal_mad(
    u_cube: np.ndarray, sig_u: np.ndarray | None,
    cmask: np.ndarray, req: tuple[int, int],
) -> np.ndarray:
    n_cad, ny, nx = u_cube.shape
    nc = int(np.count_nonzero(cmask))
    if n_cad < 8 or nc < 6:
        return cmask
    st = max(1, n_cad // _MAX_TEMPORAL_CADENCES)
    us = np.asarray(u_cube[::st, cmask], dtype=np.float64)
    fin, ss = np.isfinite(us), None
    if sig_u is not None:
        sr = np.asarray(sig_u[::st, cmask], dtype=np.float64)
        sok = fin & np.isfinite(sr) & (sr > 0.0)
        if np.any(sok):
            fin, ss = sok, np.where(sok, sr, 1.0)

    oy, ox = min(req[0], 2), min(req[1], 2)
    qr = None
    while oy > 0 or ox > 0:
        if nc >= 2 * n_basis_for_order((oy, ox)):
            qr = _qr_equilibrate(raw_chebyshev_2d(ny, nx, (oy, ox))[:, cmask].T)
            if qr is not None:
                break
        oy, ox = _step_down_order(oy, ox, cmask)
    q = qr[0] if qr is not None else np.ones((nc, 1)) / np.sqrt(nc)
    cok = np.sum(fin, axis=1) >= max(q.shape[1] + 1, 3)
    if int(np.count_nonzero(cok)) < 8:
        return cmask

    fok = fin[cok]
    uok = np.where(fok, us[cok], 0.0)
    sok = ss[cok] if ss is not None else np.ones_like(uok)
    w0 = (
        np.where(fok, 1.0 / (sok * sok), 0.0)
        if ss is not None
        else fok.astype(np.float64)
    )
    th = _huber_detrend_batch(q, uok, fok, sok, w0, 1.0 if ss is not None else 1e-6)

    dn = np.where(fok, (uok - th @ q.T) / sok, np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
        med_dn = np.nanmedian(dn, axis=0)[None, :]
        pm = _MAD_TO_SIGMA * np.nanmedian(np.abs(dn - med_dn), axis=0)
    vm = np.isfinite(pm)
    if int(np.count_nonzero(vm)) < 6:
        return cmask

    pl = raw_chebyshev_2d(ny, nx, (1, 1))[:, cmask].T
    bm = _quantile_irls(pl[vm], pm[vm], tau=0.35)
    if bm is not None:
        base = pl @ bm
        med_pm = float(np.median(pm[vm]))
        sp = max(_MAD_TO_SIGMA * float(np.median(np.abs(pm[vm] - med_pm))), 1e-8)
        inb = vm & (np.abs(pm - base) <= 3.0 * sp)
        if int(np.count_nonzero(inb)) >= 3:
            base = base + float(np.median(pm[inb] - base[inb]))
    else:
        base = np.full(nc, float(np.median(pm[vm])))

    dm = pm - base
    ldm = -dm[vm & (dm <= 0.0)]
    uulp = float(np.median(np.abs(np.spacing(uok[fok].astype(np.float32))) / sok[fok]))
    sdm = max(
        _MAD_TO_SIGMA * float(np.median(ldm)) if ldm.size else 0.0,
        32.0 * uulp,
        0.05 * max(float(np.median(np.maximum(base[vm], 0.0))), 0.0),
        1e-8,
    )
    kp = vm & (dm <= 5.0 * sdm)
    if int(np.count_nonzero(kp)) < _MIN_SURVIVING_PIXELS:
        return cmask
    out = np.zeros((ny, nx), dtype=bool)
    out[cmask] = kp
    return out


def _estimate_cadence_sky_sigma(u_cube: np.ndarray, sig_u: np.ndarray | None) -> float:
    n_cad = u_cube.shape[0]
    if n_cad <= 0:
        return 1.0
    if sig_u is not None:
        sigs = [
            float(np.percentile(st[ok], 20.0))
            for t in np.linspace(0, n_cad - 1, num=min(n_cad, 64), dtype=np.intp)
            for st in (np.asarray(sig_u[t], dtype=np.float64),)
            for ok in (np.isfinite(st) & (st > 0.0) & np.isfinite(u_cube[t]),)
            if np.any(ok)
        ]
        if sigs:
            return max(float(np.percentile(sigs, 25.0)), 1e-6)
    if n_cad >= 2:
        bl = n_cad if n_cad < 8 else min(8, max(4, n_cad // 8))
        nb = 1 if n_cad < 8 else min(8, n_cad // bl)
        sigs = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
            for s in np.linspace(0, n_cad - bl, num=nb, dtype=np.intp):
                blk = np.asarray(u_cube[s : s + bl], dtype=np.float64)
                pm = (_MAD_TO_SIGMA / math.sqrt(2.0)) * np.nanmedian(
                    np.abs(np.diff(blk, axis=0)), axis=0
                )
                ok = np.isfinite(pm) & (pm > 0.0)
                if np.any(ok):
                    sigs.append(float(np.percentile(pm[ok], 20.0)))
        if sigs:
            return max(float(np.percentile(sigs, 25.0)), 1e-6)
    return 1.0


def build_background_mask(
    u_cube: np.ndarray, *, sigma_u: np.ndarray | None = None,
    pixel_quality: np.ndarray | None = None, order: OrderSpec = None,
    mask: np.ndarray | None = None, wcs: WCS | None = None,
    catalog_table: Any = None, gaia_sources: Any = None,
    exclusion_radius: float = DEFAULT_EXCLUSION_RADIUS,
    target_coords: tuple[float, float] | None = None,
    source_mag_limit: float = 19.0, max_source_radius: float | None = None,
    obs_time: Time | str | float | None = None, allow_downgrade: bool = True,
) -> BackgroundMaskResult:
    """Build or validate ``Omega_bg`` with exclusion zones and order guards."""
    if u_cube.ndim != 3:
        raise ValueError(f"u_cube must be 3D (N_cad, ny, nx), got {u_cube.shape}.")
    if exclusion_radius < 0.0 or not np.isfinite(exclusion_radius):
        raise ValueError(
            f"exclusion_radius must be finite >= 0, got {exclusion_radius!r}."
        )
    n_cad, ny, nx = u_cube.shape
    req = parse_order(order, ny, nx)

    srcs = catalog_table if catalog_table is not None else gaia_sources
    sig_est = _estimate_cadence_sky_sigma(u_cube, sigma_u) if srcs is not None else 1.0
    smask = build_wcs_source_mask(
        ny, nx, wcs, srcs, mag_limit=source_mag_limit, sky_sigma=sig_est,
        max_radius_px=max_source_radius, exclusion_radius=exclusion_radius,
        obs_time=obs_time,
    )

    if mask is not None:
        um = np.asarray(mask, dtype=bool)
        if um.shape != (ny, nx):
            raise ValueError(f"mask shape {um.shape} != ({ny}, {nx}).")
        fa = np.zeros((ny, nx), dtype=bool)
        for s in range(0, n_cad, _CADENCE_CHUNK):
            fa |= np.any(np.isfinite(u_cube[s : s + _CADENCE_CHUNK]), axis=0)
        fm = um & fa & (~smask)
        eff, kap = apply_curvature_and_rank_guards(
            fm, req, allow_downgrade=allow_downgrade
        )
        empty = np.zeros((ny, nx), dtype=bool)
        return BackgroundMaskResult(fm, smask, empty, req, eff, kap, None, None, None)

    epq = np.ones((ny, nx), dtype=bool)
    if pixel_quality is not None:
        pq = np.asarray(pixel_quality, dtype=np.uint16)
        if pq.shape != (ny, nx):
            raise ValueError(f"pixel_quality shape {pq.shape} != ({ny}, {nx}).")
        eb = np.uint16(
            PixelQuality.STRAP_COL
            | PixelQuality.SAT_BLEED
            | PixelQuality.SMEAR_UNRESOLVED
            | PixelQuality.SOURCE_MASK
        )
        epq &= (pq & eb) == 0

    fcnt = np.zeros((ny, nx), dtype=np.int64)
    lov, hiv = np.full((ny, nx), np.inf), np.full((ny, nx), -np.inf)
    for s in range(0, n_cad, _CADENCE_CHUNK):
        ch = np.asarray(u_cube[s : s + _CADENCE_CHUNK], dtype=np.float64)
        fc = np.isfinite(ch)
        fcnt += np.sum(fc, axis=0)
        if np.any(fc):
            lov = np.minimum(lov, np.min(np.where(fc, ch, np.inf), axis=0))
            hiv = np.maximum(hiv, np.max(np.where(fc, ch, -np.inf), axis=0))

    dead = (
        (fcnt == 0)
        | ((lov == 0.0) & (hiv == 0.0))
        | ((fcnt >= 2) & (lov == hiv) & (lov <= 0.0))
    )
    elive = epq & (~dead)
    min_s = 1 if (allow_downgrade and ny * nx < 6) else _MIN_SURVIVING_PIXELS
    nl = int(np.count_nonzero(elive))
    if nl < min_s:
        raise ValueError(f"Insufficient finite background pixels ({nl} < {min_s}).")

    excl, sat_c = _build_saturated_exclusion_mask(
        u_cube, pixel_quality, exclusion_radius=exclusion_radius,
        wcs=wcs, target_coords=target_coords,
    )
    elig = elive & (~smask) & (~excl)
    min_r = (
        min_s if allow_downgrade else max(_MIN_SURVIVING_PIXELS, n_basis_for_order(req))
    )
    nel = int(np.count_nonzero(elig))
    if nel < min_r:
        if not allow_downgrade:
            raise ValueError(
                f"Insufficient pixels ({nel} < {min_r}) after "
                f"exclusion_radius={exclusion_radius}."
            )
        er, ec = np.nonzero(elive)
        sr, sc = np.nonzero(sat_c)
        cy, cx = (
            (float(np.mean(sr)), float(np.mean(sc)))
            if sr.size
            else (0.5 * (ny - 1), 0.5 * (nx - 1))
        )
        n_pick = min(er.size, max(10, math.ceil(0.25 * er.size)))
        pk = np.argsort((er - cy) ** 2 + (ec - cx) ** 2, kind="mergesort")[-n_pick:]
        fbm = np.zeros((ny, nx), dtype=bool)
        fbm[er[pk], ec[pk]] = True
        n_fb = int(np.count_nonzero(fbm))
        warnings.warn(
            f"Only {nel} pixels survived exclusion_radius={exclusion_radius:g} px; "
            f"falling back to {n_fb} outer-edge pixels with order=(0, 0). "
            "Inspect plot_diagnostics(), adjust exclusion_radius, or widen cutout.",
            UserWarning, stacklevel=3,
        )
        return BackgroundMaskResult(
            fbm, smask, excl, req, (0, 0), 1.0, None, None, None
        )

    cq10 = np.full(n_cad, np.nan, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
        for s in range(0, n_cad, _CADENCE_CHUNK):
            sub = np.asarray(u_cube[s : s + _CADENCE_CHUNK][:, elig], dtype=np.float64)
            ok = np.any(np.isfinite(sub), axis=1)
            if np.any(ok):
                cq10[s : s + _CADENCE_CHUNK][ok] = np.nanpercentile(
                    sub[ok], 10.0, axis=1
                )
    vc = np.flatnonzero(np.isfinite(cq10))
    if vc.size == 0:
        raise ValueError(
            "Insufficient finite background pixels: no cadences have finite flux."
        )

    nd = max(1, math.ceil(0.25 * vc.size))
    drows = np.sort(vc[np.argpartition(cq10[vc], nd - 1)[:nd]])
    wy, wx = _wide_interior_gaps(elig) if allow_downgrade else (False, False)
    deg_p0 = (min(req[0], 1 if wy else 2), min(req[1], 1 if wx else 2))
    d_stride = max(1, drows.size // _MAX_TEMPORAL_CADENCES)
    dcube = np.asarray(u_cube[drows[::d_stride]], dtype=np.float64)
    mimg = _compute_aligned_dark_median(dcube, elig, deg_p0)
    elig &= np.isfinite(mimg)
    nel = int(np.count_nonzero(elig))
    if nel < min_s:
        raise ValueError(f"Insufficient finite background pixels ({nel} < {min_s}).")

    p0s, ssig = _fit_coarse_surface_p0(
        mimg, elig, deg_p0, allow_downgrade=allow_downgrade
    )
    rsp = mimg - p0s
    spm = elig & (rsp >= -5.0 * ssig) & (rsp <= 3.0 * ssig)
    nsp = int(np.count_nonzero(spm))
    if nsp < min_s:
        if allow_downgrade and nel >= min_s:
            spm = elig
        else:
            raise ValueError(
                f"Insufficient pixels ({nsp} < {min_s}) after spatial clipping."
            )

    fm = _reject_anomalous_temporal_mad(u_cube, sigma_u, spm, req)
    eff, kap = apply_curvature_and_rank_guards(
        fm, req, allow_downgrade=allow_downgrade
    )
    nfn = int(np.count_nonzero(fm))
    if ny * nx >= _MIN_WARN_PIXELS and nfn < _MIN_WARN_PIXELS:
        warnings.warn(
            f"Only {nfn} background pixels survived masking (< {_MIN_WARN_PIXELS}); "
            "inspect plot_diagnostics(), adjust exclusion_radius, use order=0, "
            "or use a wider cutout.",
            UserWarning, stacklevel=3,
        )
    return BackgroundMaskResult(fm, smask, excl, req, eff, kap, mimg, p0s, ssig)
