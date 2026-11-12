"""Dark-cadence detrended sky masking and curvature/rank guards for ``tessbkgd``."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astropy.wcs import WCS

from ._contract import PixelQuality

ORDER_NAMES: dict[int, str] = {
    0: "pedestal",
    1: "linear",
    2: "quadratic",
    3: "cubic",
}
_NAME_TO_ORDER: dict[str, int] = {v: k for k, v in ORDER_NAMES.items()}

_DARK_CADENCE_PERCENTILE = 10.0
_DARK_CADENCE_FRACTION = 0.25
_TILE_ANCHOR_QUANTILE = 0.15
_QUANTILE_IRLS_TAU = 0.25
_SPATIAL_UPPER_SIGMA = 3.0
_SPATIAL_LOWER_SIGMA = 5.0
_TEMPORAL_MAD_SIGMA = 5.0
_MAX_PRE_QR_KAPPA = 100.0
_MIN_SURVIVING_PIXELS = 3
_CADENCE_CHUNK = 8192
_MAX_TEMPORAL_MAD_CADENCES = 4096

# Gaussian consistency constants:
# - 1 / Phi^{-1}(0.75) = 1.4826022 (half-normal median below Q50)
# - 1 / (Phi^{-1}(0.875) - Phi^{-1}(0.75)) = 2.1014809 (Q25 - Q12.5 spacing)
_MAD_TO_SIGMA = 1.482602218505602
_Q25_LOWER_TO_SIGMA = 2.101480909693451

# TESS photometric zero-point (Sullivan et al. 2015, ApJ 809, 77) and isotropic
# Moffat PRF parameters reproducing Vanderspek et al. (2018, TESS Instrument
# Handbook Table 1.1) ensquared energy (~50% in 1x1 px, ~90% in 4x4 px):
_TESS_ZERO_POINT_MAG = 20.44
_TESS_PRF_RC_PX = 0.85
_TESS_PRF_BETA = 2.2
_PIXEL_DISK_RADIUS_PX = 1.0 / math.sqrt(math.pi)


@dataclass(frozen=True)
class BackgroundMaskResult:
    """Result of automatic or validated background pixel masking."""

    mask: np.ndarray
    source_mask: np.ndarray
    requested_order: int
    effective_order: int
    condition_number: float
    dark_median_image: np.ndarray | None
    coarse_surface: np.ndarray | None
    sky_sigma: float | None


def n_basis_for_order(order: int) -> int:
    """Return 2D total-degree polynomial basis count ``P = (d + 1)*(d + 2)//2``."""
    if isinstance(order, (bool, np.bool_)) or order < 0 or order > 3:
        raise ValueError(
            f"Unrecognised order {order!r}. Expected 0/'pedestal', "
            "1/'linear', 2/'quadratic', or 3/'cubic'."
        )
    return (int(order) + 1) * (int(order) + 2) // 2


def parse_order(order: int | str | None, ny: int, nx: int) -> int:
    """Resolve ``order`` to an integer polynomial degree in ``{0, 1, 2, 3}``.

    When ``order is None``, defaults to ``1`` (``'linear'``, harmonic plane) when
    ``min(nx, ny) < 40`` and ``2`` (``'quadratic'``) when ``min(nx, ny) >= 40``.
    """
    if order is None:
        return 1 if min(nx, ny) < 40 else 2
    if isinstance(order, (bool, np.bool_)):
        raise ValueError(
            f"Unrecognised order {order!r}. Expected 0/'pedestal', "
            "1/'linear', 2/'quadratic', or 3/'cubic'."
        )
    if isinstance(order, (int, np.integer)):
        d = int(order)
        if d in ORDER_NAMES:
            return d
        raise ValueError(
            f"Unrecognised order {order!r}. Expected 0/'pedestal', "
            "1/'linear', 2/'quadratic', or 3/'cubic'."
        )
    if isinstance(order, str):
        key = order.strip().lower()
        if key in _NAME_TO_ORDER:
            return _NAME_TO_ORDER[key]
        raise ValueError(
            f"Unrecognised order {order!r}. Expected 0/'pedestal', "
            "1/'linear', 2/'quadratic', or 3/'cubic'."
        )
    raise ValueError(
        f"Unrecognised order {order!r}. Expected 0/'pedestal', "
        "1/'linear', 2/'quadratic', or 3/'cubic'."
    )


def normalized_grid(ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized 1D coordinates ``(eta_r, xi_c)`` in ``[-1, 1]``."""
    if ny < 1 or nx < 1:
        raise ValueError(f"Grid dimensions must be >= 1, got ({ny}, {nx}).")
    denom_y = float(max(ny - 1, 1))
    denom_x = float(max(nx - 1, 1))
    r = np.arange(ny, dtype=np.float64)
    c = np.arange(nx, dtype=np.float64)
    eta = (2.0 * r - float(ny - 1)) / denom_y
    xi = (2.0 * c - float(nx - 1)) / denom_x
    return eta, xi


def _chebyshev_1d(x: np.ndarray, max_deg: int) -> np.ndarray:
    """Evaluate 1D Chebyshev polynomials ``T_0(x), ..., T_{max_deg}(x)``."""
    out = np.empty((max_deg + 1, x.size), dtype=np.float64)
    out[0] = 1.0
    if max_deg >= 1:
        out[1] = x
    for k in range(2, max_deg + 1):
        out[k] = 2.0 * x * out[k - 1] - out[k - 2]
    return out


def raw_chebyshev_2d(ny: int, nx: int, order: int) -> np.ndarray:
    """Evaluate raw total-degree ``order`` 2D Chebyshev polynomials ``(P, ny, nx)``.

    Uses ``Phi_tilde[(r, c), (p, q)] = T_p(xi_c) * T_q(eta_r)`` for ``p + q <= d``
    ordered by increasing total degree ``s = p + q`` and ``q = 0, ..., s``.
    """
    n_params = n_basis_for_order(order)
    eta, xi = normalized_grid(ny, nx)
    t_eta = _chebyshev_1d(eta, order)
    t_xi = _chebyshev_1d(xi, order)

    basis = np.empty((n_params, ny, nx), dtype=np.float64)
    idx = 0
    for s in range(order + 1):
        for q in range(s + 1):
            p = s - q
            basis[idx] = np.outer(t_eta[q], t_xi[p])
            idx += 1
    return basis


def _pre_qr_condition_number(phi_bg: np.ndarray) -> float:
    """Return the 2-norm condition number ``kappa(Phi_tilde_bg)``."""
    if phi_bg.shape[0] < phi_bg.shape[1]:
        return float("inf")
    svals = np.linalg.svd(phi_bg, compute_uv=False)
    if svals.size == 0 or svals[-1] <= 0.0 or not np.all(np.isfinite(svals)):
        return float("inf")
    return float(svals[0] / svals[-1])


def apply_curvature_and_rank_guards(
    mask: np.ndarray,
    requested_order: int,
    *,
    allow_downgrade: bool = True,
) -> tuple[int, float]:
    """Validate surviving pixel count and condition number, downgrading if needed.

    Follows Section 6.6.1:
    - Raises :class:`ValueError` if fewer than 3 background pixels survive.
    - When ``allow_downgrade=True``, automatically downgrades ``d -> d - 1``
      whenever ``|Omega_bg| < max(5 * P, 30)`` or ``kappa(Phi_tilde_bg) > 100``
      (down to ``order = 0`` pedestal when ``3 <= |Omega_bg| < 10``).
    - When ``allow_downgrade=False``, requires ``|Omega_bg| >= P`` and
      geometrically full rank (``kappa < 1e10``), raising :class:`ValueError`
      otherwise.
    """
    mask_bool = np.asarray(mask, dtype=bool)
    ny, nx = mask_bool.shape
    n_bg = int(np.count_nonzero(mask_bool))
    n_req = n_basis_for_order(requested_order)

    if not allow_downgrade:
        if n_bg < max(_MIN_SURVIVING_PIXELS, n_req):
            raise ValueError(
                f"Insufficient finite background pixels ({n_bg}) for "
                f"{ORDER_NAMES[requested_order]!r} background fit "
                f"(requires at least {max(_MIN_SURVIVING_PIXELS, n_req)})."
            )
        phi_req = raw_chebyshev_2d(ny, nx, requested_order)[:, mask_bool].T
        kappa_req = _pre_qr_condition_number(phi_req)
        if not np.isfinite(kappa_req) or kappa_req > 1e10:
            raise ValueError(
                f"Background mask of {n_bg} pixels is geometrically "
                f"rank-deficient for {ORDER_NAMES[requested_order]!r} 2D "
                f"polynomial fit (kappa={kappa_req:.3g})."
            )
        return requested_order, kappa_req

    if n_bg < _MIN_SURVIVING_PIXELS:
        raise ValueError(
            f"Insufficient finite background pixels ({n_bg} < "
            f"{_MIN_SURVIVING_PIXELS}) surviving background masking."
        )

    d = int(requested_order)
    while d > 0:
        p = n_basis_for_order(d)
        min_px = max(5 * p, 30) if d >= 2 else 10
        if n_bg < min_px:
            d -= 1
            continue
        phi_bg = raw_chebyshev_2d(ny, nx, d)[:, mask_bool].T
        kappa = _pre_qr_condition_number(phi_bg)
        if np.isfinite(kappa) and kappa <= _MAX_PRE_QR_KAPPA:
            return d, kappa
        d -= 1

    # d == 0 pedestal (P = 1, constant column of 1s -> kappa = 1.0)
    return 0, 1.0


def _unwrap_col_float64(col: Any) -> tuple[np.ndarray, u.UnitBase | None]:
    """Extract 1D float64 values (masking invalid/masked entries to NaN) and unit."""
    unit = getattr(col, "unit", None)
    mask_attr = getattr(col, "mask", None)
    raw_obj = col.value if isinstance(col, Quantity) else col
    vals = np.asarray(np.ma.getdata(raw_obj), dtype=np.float64)
    if mask_attr is not None and mask_attr is not np.ma.nomask:
        mask_arr = np.broadcast_to(np.asarray(mask_attr, dtype=bool), vals.shape)
        vals = np.where(mask_arr, np.nan, vals)
    return vals.ravel(), unit


def _extract_angle_deg(col: Any, col_name: str) -> np.ndarray:
    """Convert a sky-coordinate column to 1D float64 degrees."""
    vals, unit = _unwrap_col_float64(col)
    if unit is None or unit == u.dimensionless_unscaled:
        return vals
    if unit.is_equivalent(u.deg):
        return np.asarray(
            Quantity(vals, unit=unit, copy=False).to_value(u.deg),
            dtype=np.float64,
        )
    raise u.UnitConversionError(
        f"gaia_sources {col_name!r} column unit {unit!r} is not an angular unit."
    )


def _extract_mag_values(col: Any) -> np.ndarray:
    """Convert a magnitude column to 1D float64 magnitudes."""
    vals, unit = _unwrap_col_float64(col)
    if unit is None or unit == u.dimensionless_unscaled:
        return vals
    if unit.is_equivalent(u.mag):
        return np.asarray(
            Quantity(vals, unit=unit, copy=False).to_value(u.mag),
            dtype=np.float64,
        )
    if isinstance(unit, u.MagUnit):
        return vals
    raise u.UnitConversionError(
        f"gaia_sources magnitude column unit {unit!r} is not a magnitude unit."
    )


def _extract_source_catalogue(
    gaia_sources: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize ``gaia_sources`` into ``(ra_deg, dec_deg, mag)`` 1D float64 arrays."""
    if gaia_sources is None:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty

    col_lookup: dict[str, Any] | None = None
    if isinstance(gaia_sources, Mapping):
        col_lookup = {str(k).lower(): gaia_sources[k] for k in gaia_sources}
    elif hasattr(gaia_sources, "colnames"):
        col_lookup = {
            str(k).lower(): gaia_sources[k] for k in gaia_sources.colnames
        }
    elif hasattr(gaia_sources, "columns") and hasattr(gaia_sources, "__getitem__"):
        col_lookup = {
            str(k).lower(): gaia_sources[k] for k in gaia_sources.columns
        }
    elif isinstance(gaia_sources, np.ndarray) and gaia_sources.dtype.names:
        col_lookup = {
            str(k).lower(): gaia_sources[k] for k in gaia_sources.dtype.names
        }

    if col_lookup is not None:
        ra_col = next(
            (
                col_lookup[k]
                for k in ("ra", "ra_icrs", "raj2000", "ra_deg")
                if k in col_lookup
            ),
            None,
        )
        dec_col = next(
            (
                col_lookup[k]
                for k in ("dec", "de_icrs", "dej2000", "dec_deg")
                if k in col_lookup
            ),
            None,
        )
        if ra_col is None or dec_col is None:
            raise ValueError(
                "gaia_sources mapping/table must include 'ra' and 'dec' columns."
            )
        mag_col = next(
            (
                col_lookup[k]
                for k in (
                    "mag",
                    "tmag",
                    "gmag",
                    "phot_g_mean_mag",
                    "phot_rp_mean_mag",
                    "vmag",
                    "v",
                )
                if k in col_lookup
            ),
            None,
        )
        ra = _extract_angle_deg(ra_col, "ra")
        dec = _extract_angle_deg(dec_col, "dec")
        if mag_col is None:
            mag = np.full(ra.shape, 14.0, dtype=np.float64)
        else:
            mag = _extract_mag_values(mag_col)
        if ra.shape != dec.shape or ra.shape != mag.shape:
            raise ValueError(
                "gaia_sources 'ra', 'dec', and 'mag' columns must share "
                "the same length."
            )
        return ra, dec, mag

    if isinstance(gaia_sources, (Sequence, np.ndarray)):
        raw_seq = (
            gaia_sources.value
            if isinstance(gaia_sources, Quantity)
            else gaia_sources
        )
        arr = np.asarray(np.ma.getdata(raw_seq), dtype=np.float64)
        mask_attr = getattr(gaia_sources, "mask", None)
        if mask_attr is not None and mask_attr is not np.ma.nomask:
            mask_arr = np.broadcast_to(
                np.asarray(mask_attr, dtype=bool), arr.shape
            )
            arr = np.where(mask_arr, np.nan, arr)
        if arr.size == 0:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty, empty
        if arr.ndim == 1 and arr.size in (2, 3):
            arr = arr[np.newaxis, :]
        if arr.ndim != 2 or arr.shape[1] not in (2, 3):
            raise ValueError(
                "Sequence/array gaia_sources must have shape (N, 2) or (N, 3) "
                f"for (ra, dec[, mag]), got {arr.shape}."
            )
        seq_unit = getattr(gaia_sources, "unit", None)
        if seq_unit is not None and seq_unit != u.dimensionless_unscaled:
            if arr.shape[1] == 3:
                raise u.UnitConversionError(
                    "Homogeneous angular Quantity array cannot include a 3rd "
                    "magnitude column; pass a QTable, mapping, or (N, 2) Quantity."
                )
            ra = _extract_angle_deg(
                Quantity(arr[:, 0], unit=seq_unit, copy=False), "ra"
            )
            dec = _extract_angle_deg(
                Quantity(arr[:, 1], unit=seq_unit, copy=False), "dec"
            )
        else:
            ra = arr[:, 0]
            dec = arr[:, 1]
        mag = (
            arr[:, 2]
            if arr.shape[1] == 3
            else np.full(ra.shape, 14.0, dtype=np.float64)
        )
        return ra, dec, mag

    raise TypeError(
        "Unsupported gaia_sources type "
        f"{type(gaia_sources).__name__}; expected table, mapping, or (N, 3) array."
    )


def _tess_prf_source_radii(
    mag: np.ndarray,
    *,
    sky_sigma: float,
    min_radius_px: float,
    max_radius_px: float,
) -> np.ndarray:
    """Compute circular exclusion radii from TESS Moffat PRF wings vs ``sky_sigma``.

    Solves ``F_star(T) * P_Moffat(r_wing) = sky_sigma`` using ``T_zp = 20.44``
    (Sullivan et al. 2015) and ``(r_c, beta) = (0.85 px, 2.2)`` (Vanderspek et al.
    2018), convolved in quadrature with the equal-area pixel disk radius
    ``pi^(-1/2) px`` and clipped to ``[min(min_radius_px, max_radius_px),
    max_radius_px]``.
    """
    sig_eff = max(float(sky_sigma), 1e-6)
    flux_star = np.power(10.0, -0.4 * (mag - _TESS_ZERO_POINT_MAG))
    peak_scale = (_TESS_PRF_BETA - 1.0) / (
        math.pi * _TESS_PRF_RC_PX * _TESS_PRF_RC_PX * sig_eff
    )
    gamma = flux_star * peak_scale
    r_wing_sq = np.where(
        gamma > 1.0,
        (_TESS_PRF_RC_PX * _TESS_PRF_RC_PX)
        * (np.power(np.maximum(gamma, 1.0), 1.0 / _TESS_PRF_BETA) - 1.0),
        0.0,
    )
    r_prf = np.sqrt(r_wing_sq + _PIXEL_DISK_RADIUS_PX * _PIXEL_DISK_RADIUS_PX)
    hi = max(float(max_radius_px), 0.0)
    lo = min(max(float(min_radius_px), 0.0), hi)
    return np.clip(r_prf, lo, hi)


def build_wcs_source_mask(
    ny: int,
    nx: int,
    wcs: WCS | None,
    gaia_sources: Any,
    *,
    mag_limit: float = 19.0,
    sky_sigma: float = 1.0,
    min_radius_px: float = 1.25,
    max_radius_px: float | None = None,
) -> np.ndarray:
    """Project ``(ra, dec, mag)`` catalogue sources onto circular pixel masks.

    Projects ICRS coordinates via ``wcs.celestial`` (supporting swapped WCS axis
    orders and ``>2``D FITS cubes) and sizes source masks from the TESS Moffat PRF
    wing profile relative to ``sky_sigma``. When ``max_radius_px is None``, caps
    individual circular masks at ``sqrt(ny * nx / (4 * pi))`` (at most 25% of the
    stamp area) so bright targets on compact SPOC TPFs rely on ``PIXEL_QUALITY``
    and data-driven dark-median/temporal-MAD masking rather than wiping out 100%
    of pixels.
    """
    ra, dec, mag = _extract_source_catalogue(gaia_sources)
    mask = np.zeros((ny, nx), dtype=bool)
    if ra.size == 0:
        return mask

    if wcs is None or not getattr(wcs, "has_celestial", False):
        raise ValueError(
            "A valid celestial WCS is required when gaia_sources is provided."
        )

    valid_coord = np.isfinite(ra) & np.isfinite(dec)
    if not np.any(valid_coord):
        return mask
    ra = ra[valid_coord]
    dec = dec[valid_coord]
    mag = np.where(np.isfinite(mag[valid_coord]), mag[valid_coord], 14.0)

    keep = mag <= mag_limit
    if not np.any(keep):
        return mask
    ra = ra[keep]
    dec = dec[keep]
    mag = mag[keep]

    cel_wcs = wcs.celestial
    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    col_x, row_y = cel_wcs.world_to_pixel(coords)
    col_x = np.asarray(col_x, dtype=np.float64)
    row_y = np.asarray(row_y, dtype=np.float64)

    eff_max_rad = (
        max(1.0, math.sqrt((ny * nx) / (4.0 * math.pi)))
        if max_radius_px is None
        else float(max_radius_px)
    )
    radii = _tess_prf_source_radii(
        mag,
        sky_sigma=sky_sigma,
        min_radius_px=min_radius_px,
        max_radius_px=eff_max_rad,
    )

    near = (
        np.isfinite(col_x)
        & np.isfinite(row_y)
        & (col_x >= -radii)
        & (col_x <= (nx - 1) + radii)
        & (row_y >= -radii)
        & (row_y <= (ny - 1) + radii)
    )
    if not np.any(near):
        return mask

    rr, cc = np.meshgrid(
        np.arange(ny, dtype=np.float64),
        np.arange(nx, dtype=np.float64),
        indexing="ij",
    )
    for cx, ry, rad in zip(col_x[near], row_y[near], radii[near], strict=True):
        mask |= (cc - cx) ** 2 + (rr - ry) ** 2 <= rad * rad
    return mask


def _qr_equilibrate_matrix(
    design: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(Q, R)`` with positive diagonal on ``R``, or ``None`` if singular."""
    n_rows, n_cols = design.shape
    if n_rows < n_cols:
        return None
    q, r = np.linalg.qr(design, mode="reduced")
    diag_r = np.diag(r)
    max_diag = float(np.max(np.abs(diag_r))) if diag_r.size > 0 else 0.0
    if max_diag <= 0.0 or np.min(np.abs(diag_r)) <= 1e-10 * max_diag:
        return None
    signs = np.where(diag_r >= 0.0, 1.0, -1.0)
    return q * signs[np.newaxis, :], signs[:, np.newaxis] * r


def _quantile_irls_fit(
    design: np.ndarray,
    y: np.ndarray,
    *,
    tau: float = _QUANTILE_IRLS_TAU,
    n_iter: int = 20,
) -> np.ndarray | None:
    """Fit smoothed ``tau``-quantile regression via IRLS on QR-equilibrated basis."""
    qr_res = _qr_equilibrate_matrix(design)
    if qr_res is None:
        return None
    q, r_mat = qr_res
    n_cols = q.shape[1]

    # Seed at constant tau-quantile so initial step is immune to extreme stars.
    theta = np.zeros(n_cols, dtype=np.float64)
    q0_mean = float(np.mean(q[:, 0]))
    if abs(q0_mean) > 1e-12:
        theta[0] = float(np.quantile(y, tau)) / q0_mean

    med_y = float(np.median(y))
    mad_y = _MAD_TO_SIGMA * float(np.median(np.abs(y - med_y)))
    delta = max(1e-3 * mad_y, 1e-8 * max(abs(med_y), 1.0), 1e-10)

    for _ in range(n_iter):
        resid = y - q @ theta
        asym = np.where(resid >= 0.0, tau, 1.0 - tau)
        weights = asym / np.maximum(np.abs(resid), delta)
        gram = (q * weights[:, np.newaxis]).T @ q
        rhs = q.T @ (weights * y)
        try:
            theta = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            break

    return np.linalg.solve(r_mat, theta)


def _select_tile_15th_anchors(
    med_img: np.ndarray,
    eligible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select pixel coordinates and values at each spatial tile's 15th percentile."""
    ny, nx = med_img.shape
    min_dim = min(ny, nx)
    if min_dim < 20:
        tile_side = 3
    elif min_dim < 40:
        tile_side = 4
    else:
        tile_side = max(5, min(8, min_dim // 8))

    n_tiles_y = max(2, ny // tile_side)
    n_tiles_x = max(2, nx // tile_side)
    row_blocks = np.array_split(np.arange(ny), n_tiles_y)
    col_blocks = np.array_split(np.arange(nx), n_tiles_x)

    anchor_r: list[int] = []
    anchor_c: list[int] = []
    anchor_b_boundary: list[bool] = []

    for iy, r_idx in enumerate(row_blocks):
        is_y_edge = iy == 0 or iy == len(row_blocks) - 1
        for ix, c_idx in enumerate(col_blocks):
            is_edge = is_y_edge or ix == 0 or ix == len(col_blocks) - 1
            sub_elig = eligible[np.ix_(r_idx, c_idx)]
            if not np.any(sub_elig):
                continue
            sub_vals = med_img[np.ix_(r_idx, c_idx)]
            local_r, local_c = np.nonzero(sub_elig)
            vals = sub_vals[local_r, local_c]
            order = np.argsort(vals, kind="mergesort")
            pick = order[int(round(_TILE_ANCHOR_QUANTILE * (vals.size - 1)))]
            anchor_r.append(int(r_idx[local_r[pick]]))
            anchor_c.append(int(c_idx[local_c[pick]]))
            anchor_b_boundary.append(is_edge)

    return (
        np.asarray(anchor_r, dtype=np.intp),
        np.asarray(anchor_c, dtype=np.intp),
        np.asarray(anchor_b_boundary, dtype=bool),
    )


def _compute_aligned_dark_median(
    dark_cube: np.ndarray,
    eligible: np.ndarray,
    deg_p0: int,
) -> np.ndarray:
    """Compute ``M_{r,c} = nanmedian_{t in G_dark}(U_{t,r,c})`` aligned across orbits.

    Subtracts zero-median temporal sky-gradient fluctuations estimated on tile
    15th-percentile sky anchors before taking the pixel-wise temporal median so
    rotating scattered-light planes across separate orbits in ``G_dark`` cannot
    introduce order-statistic kinks at stamp corners.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="All-NaN slice encountered",
            category=RuntimeWarning,
        )
        raw_med = np.asarray(np.nanmedian(dark_cube, axis=0), dtype=np.float64)

    n_dark, ny, nx = dark_cube.shape
    live = eligible & np.isfinite(raw_med) & (raw_med > 0.0)
    if n_dark < 4 or int(np.count_nonzero(live)) < 12:
        return raw_med

    anchor_r, anchor_c, _ = _select_tile_15th_anchors(raw_med, live)
    if anchor_r.size < 6:
        return raw_med

    y_anc = np.asarray(dark_cube[:, anchor_r, anchor_c], dtype=np.float64)
    anc_fin = np.all(np.isfinite(y_anc), axis=0)
    deg = deg_p0
    while deg > 0:
        n_p = n_basis_for_order(deg)
        if int(np.count_nonzero(anc_fin)) >= max(2 * n_p, 6):
            basis_try = raw_chebyshev_2d(ny, nx, deg)
            ar = anchor_r[anc_fin]
            ac = anchor_c[anc_fin]
            qr_anc = _qr_equilibrate_matrix(basis_try[:, ar, ac].T)
            if qr_anc is not None:
                q_anc, r_anc = qr_anc
                ya = y_anc[:, anc_fin]
                dy = ya - np.median(ya, axis=0, keepdims=True)
                theta = dy @ q_anc
                eye_p = np.eye(q_anc.shape[1], dtype=np.float64)[
                    np.newaxis, :, :
                ]
                for _ in range(2):
                    res = dy - theta @ q_anc.T
                    med_r = np.median(res, axis=1, keepdims=True)
                    mad_r = np.maximum(
                        _MAD_TO_SIGMA
                        * np.median(
                            np.abs(res - med_r), axis=1, keepdims=True
                        ),
                        1e-6,
                    )
                    w = np.minimum(
                        1.0,
                        1.345 / np.maximum(np.abs(res - med_r) / mad_r, 1e-30),
                    )
                    gram = np.einsum("ap,ka,aq->kpq", q_anc, w, q_anc)
                    rhs = np.einsum("ap,ka->kp", q_anc, w * dy)
                    try:
                        theta = np.linalg.solve(
                            gram + 1e-14 * eye_p, rhs[..., np.newaxis]
                        )[..., 0]
                    except np.linalg.LinAlgError:
                        break
                theta -= np.median(theta, axis=0, keepdims=True)
                coeffs = np.linalg.solve(r_anc, theta.T).T
                aligned = dark_cube - np.tensordot(
                    coeffs, basis_try, axes=(1, 0)
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="All-NaN slice encountered",
                        category=RuntimeWarning,
                    )
                    return np.asarray(
                        np.nanmedian(aligned, axis=0), dtype=np.float64
                    )
        deg -= 1

    return raw_med


def _fit_coarse_surface_p0(
    med_img: np.ndarray,
    eligible: np.ndarray,
    deg_p0: int,
) -> tuple[np.ndarray, float]:
    """Fit coarse robust surface ``P_0(r, c)`` and lower-half sky MAD ``sigma_sky``."""
    ny, nx = med_img.shape
    y_elig = med_img[eligible]
    ulp_floor = max(
        32.0 * float(np.median(np.abs(np.spacing(y_elig.astype(np.float32))))),
        1e-8,
    )

    anchor_r, anchor_c, anchor_is_edge = _select_tile_15th_anchors(
        med_img, eligible
    )

    # Downgrade deg_p0 if anchor/pixel geometry cannot support it.
    deg = deg_p0
    while deg > 0:
        n_p = n_basis_for_order(deg)
        basis_try = raw_chebyshev_2d(ny, nx, deg)
        if anchor_r.size >= n_p:
            design_anchor = basis_try[:, anchor_r, anchor_c].T
            if _qr_equilibrate_matrix(design_anchor) is not None:
                break
        if y_elig.size >= n_p:
            design_all = basis_try[:, eligible].T
            if _qr_equilibrate_matrix(design_all) is not None:
                anchor_r, anchor_c = np.nonzero(eligible)
                anchor_is_edge = np.ones(anchor_r.size, dtype=bool)
                break
        deg -= 1

    basis_0 = raw_chebyshev_2d(ny, nx, deg)
    design_elig = basis_0[:, eligible].T
    y_anchor = med_img[anchor_r, anchor_c]

    # When deg == 2 on >= 40x40 cutouts, guard against a massive central saturated
    # star core bending an unconstrained quadratic into a concave star dome by
    # pre-screening gross interior core tiles relative to a harmonic degree-1 plane.
    fit_anchor_mask = np.ones(anchor_r.size, dtype=bool)
    if deg == 2 and anchor_r.size >= 8:
        basis_plane = basis_0[:3]
        beta_plane = _quantile_irls_fit(
            basis_plane[:, anchor_r, anchor_c].T, y_anchor
        )
        if beta_plane is not None:
            r_plane = y_anchor - (basis_plane[:, anchor_r, anchor_c].T @ beta_plane)
            edge_r = r_plane[anchor_is_edge] if np.any(anchor_is_edge) else r_plane
            edge_med = float(np.median(edge_r))
            edge_mad = _MAD_TO_SIGMA * float(np.median(np.abs(edge_r - edge_med)))
            bound_span = max(
                float(np.percentile(np.abs(edge_r - edge_med), 85.0)),
                6.0 * edge_mad,
                ulp_floor,
            )
            trimmed = r_plane - edge_med <= 4.0 * bound_span
            if (
                int(np.count_nonzero(trimmed)) >= 6
                and _qr_equilibrate_matrix(
                    basis_0[:, anchor_r[trimmed], anchor_c[trimmed]].T
                )
                is not None
            ):
                fit_anchor_mask = trimmed

    beta = _quantile_irls_fit(
        basis_0[:, anchor_r[fit_anchor_mask], anchor_c[fit_anchor_mask]].T,
        y_anchor[fit_anchor_mask],
    )
    if beta is None:
        beta = np.zeros(basis_0.shape[0], dtype=np.float64)
        beta[0] = float(np.quantile(y_elig, 0.25))

    # Stage 1: seed pixel-level sky inliers around Q25 using exact Q25-Q12.5 scale.
    r0 = y_elig - design_elig @ beta
    q25 = float(np.percentile(r0, 25.0))
    low25 = q25 - r0[r0 <= q25]
    sig0 = (
        _Q25_LOWER_TO_SIGMA * float(np.median(low25)) if low25.size > 0 else 0.0
    )
    sig0 = max(sig0, ulp_floor)
    inliers = (r0 >= q25 - 3.5 * sig0) & (r0 <= q25 + 3.7 * sig0)
    if int(np.count_nonzero(inliers)) < basis_0.shape[0]:
        inliers = r0 <= float(np.percentile(r0, 50.0))

    # Stage 2: refine P_0 via QR least squares on clean sky inliers and lower-half MAD.
    for _ in range(3):
        qr_in = _qr_equilibrate_matrix(design_elig[inliers])
        if qr_in is None:
            break
        q_in, r_in = qr_in
        beta = np.linalg.solve(r_in, q_in.T @ y_elig[inliers])
        resid = y_elig - design_elig @ beta
        r_sub = resid[inliers]
        r_med = float(np.median(r_sub))
        low_dev = r_med - r_sub[r_sub <= r_med]
        sig_low = (
            _MAD_TO_SIGMA * float(np.median(low_dev))
            if low_dev.size > 0
            else 0.0
        )
        sig = max(sig_low, ulp_floor)
        next_inliers = (resid - r_med >= -4.0 * sig) & (
            resid - r_med <= _SPATIAL_UPPER_SIGMA * sig
        )
        if int(np.count_nonzero(next_inliers)) < basis_0.shape[0]:
            break
        if np.array_equal(next_inliers, inliers):
            break
        inliers = next_inliers

    # Centre constant mode on final inlier median so clean sky residuals centre at 0.
    resid_final = y_elig - design_elig @ beta
    r_sub_final = resid_final[inliers]
    r_med_final = float(np.median(r_sub_final))
    beta[0] += r_med_final

    p0_surface = np.tensordot(beta, basis_0, axes=(0, 0))
    r_centred = r_sub_final - r_med_final
    low_final = -r_centred[r_centred <= 0.0]
    sigma_sky = (
        _MAD_TO_SIGMA * float(np.median(low_final))
        if low_final.size > 0
        else 0.0
    )
    sigma_sky = max(sigma_sky, ulp_floor)
    return p0_surface, sigma_sky


def _reject_anomalous_temporal_mad(
    u_cube: np.ndarray,
    sigma_u: np.ndarray | None,
    candidate_mask: np.ndarray,
    requested_order: int,
) -> np.ndarray:
    """Reject pixels with anomalous temporal MAD on detrended per-cadence residuals.

    Detrends each cadence by a preliminary robust background surface ``B_hat^(0)``
    (plane when ``min(nx, ny) < 40``; degree ``min(d, 2)`` when ``min(nx, ny) >= 40``
    and ``d >= 2``) BEFORE measuring pixel temporal MAD, preventing sweeping
    time-dependent Earthshine fans from self-masking stamp edges.
    """
    n_cad, ny, nx = u_cube.shape
    n_cand = int(np.count_nonzero(candidate_mask))
    if n_cad < 8 or n_cand < 6:
        return candidate_mask

    # Stride across sector only when N_t > 4096 (e.g. 120,000-cadence 20-s TPFs)
    # to bound temporary workspace well below 100 MB while sampling uniformly.
    stride = max(1, n_cad // _MAX_TEMPORAL_MAD_CADENCES)
    u_sub = np.asarray(u_cube[::stride, candidate_mask], dtype=np.float64)
    fin = np.isfinite(u_sub)

    sigma_sub: np.ndarray | None = None
    if sigma_u is not None:
        s_raw = np.asarray(sigma_u[::stride, candidate_mask], dtype=np.float64)
        s_ok = fin & np.isfinite(s_raw) & (s_raw > 0.0)
        if np.any(s_ok):
            fin = s_ok
            sigma_sub = np.where(fin, s_raw, 1.0)

    deg_det = 2 if (min(ny, nx) >= 40 and requested_order >= 2) else 1
    while deg_det > 0:
        n_p = n_basis_for_order(deg_det)
        if n_cand >= 2 * n_p:
            phi_try = raw_chebyshev_2d(ny, nx, deg_det)[:, candidate_mask].T
            if _qr_equilibrate_matrix(phi_try) is not None:
                break
        deg_det -= 1

    n_p = n_basis_for_order(deg_det)
    phi_cand = raw_chebyshev_2d(ny, nx, deg_det)[:, candidate_mask].T
    qr_cand = _qr_equilibrate_matrix(phi_cand)
    q_cand = (
        qr_cand[0]
        if qr_cand is not None
        else np.ones((n_cand, 1)) / np.sqrt(n_cand)
    )
    n_p = q_cand.shape[1]

    cad_ok = np.sum(fin, axis=1) >= max(n_p + 1, 3)
    if int(np.count_nonzero(cad_ok)) < 8:
        return candidate_mask

    u_ok = np.where(fin[cad_ok], u_sub[cad_ok], 0.0)
    f_ok = fin[cad_ok]
    if sigma_sub is not None:
        s_ok_arr = sigma_sub[cad_ok]
        w0 = np.where(f_ok, 1.0 / (s_ok_arr * s_ok_arr), 0.0)
    else:
        s_ok_arr = np.ones_like(u_ok)
        w0 = f_ok.astype(np.float64)

    # Batched 2-step Huber preliminary detrending over cadences.
    w = w0
    eye_p = np.eye(n_p, dtype=np.float64)[np.newaxis, :, :]
    theta = np.zeros((u_ok.shape[0], n_p), dtype=np.float64)
    for step in range(3):
        gram = np.einsum("mp,tm,mq->tpq", q_cand, w, q_cand)
        rhs = np.einsum("mp,tm->tp", q_cand, w * u_ok)
        try:
            theta = np.linalg.solve(gram + 1e-15 * eye_p, rhs[..., np.newaxis])[
                ..., 0
            ]
        except np.linalg.LinAlgError:
            break
        if step < 2:
            resid_step = (u_ok - theta @ q_cand.T) / s_ok_arr
            resid_nan = np.where(f_ok, resid_step, np.nan)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="All-NaN slice encountered",
                    category=RuntimeWarning,
                )
                med_t = np.nanmedian(resid_nan, axis=1, keepdims=True)
                mad_t = _MAD_TO_SIGMA * np.nanmedian(
                    np.abs(resid_nan - med_t), axis=1, keepdims=True
                )
            floor_t = 1.0 if sigma_sub is not None else 1e-6
            scale_t = np.maximum(mad_t, floor_t)
            u_std = np.abs(resid_step) / scale_t
            w_rob = np.minimum(1.0, 1.345 / np.maximum(u_std, 1e-30))
            w = w0 * w_rob

    detrended = (u_ok - theta @ q_cand.T) / s_ok_arr
    detrended_nan = np.where(f_ok, detrended, np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="All-NaN slice encountered",
            category=RuntimeWarning,
        )
        pix_med = np.nanmedian(detrended_nan, axis=0)
        pix_mad = _MAD_TO_SIGMA * np.nanmedian(
            np.abs(detrended_nan - pix_med[np.newaxis, :]), axis=0
        )

    valid_mad = np.isfinite(pix_mad)
    if int(np.count_nonzero(valid_mad)) < 6:
        return candidate_mask

    # Fit a smooth robust plane to pix_mad so legitimate sqrt(B) Poisson noise
    # variation across steep spatial gradients is never mistaken for a variable star
    # when FLUX_ERR is absent.
    plane_cand = raw_chebyshev_2d(ny, nx, 1)[:, candidate_mask].T
    beta_mad = _quantile_irls_fit(
        plane_cand[valid_mad], pix_mad[valid_mad], tau=0.35
    )
    if beta_mad is not None:
        mad_baseline = plane_cand @ beta_mad
        in_base = valid_mad & (
            np.abs(pix_mad - mad_baseline)
            <= 3.0
            * max(
                _MAD_TO_SIGMA
                * float(
                    np.median(
                        np.abs(pix_mad[valid_mad] - np.median(pix_mad[valid_mad]))
                    )
                ),
                1e-8,
            )
        )
        if int(np.count_nonzero(in_base)) >= 3:
            mad_baseline = mad_baseline + float(
                np.median(pix_mad[in_base] - mad_baseline[in_base])
            )
    else:
        mad_baseline = np.full(
            n_cand, float(np.median(pix_mad[valid_mad])), dtype=np.float64
        )

    delta_mad = pix_mad - mad_baseline
    low_dm = -delta_mad[valid_mad & (delta_mad <= 0.0)]
    sig_dm = _MAD_TO_SIGMA * float(np.median(low_dm)) if low_dm.size > 0 else 0.0
    u_ulp_scale = float(
        np.median(np.abs(np.spacing(u_ok[f_ok].astype(np.float32))) / s_ok_arr[f_ok])
    )
    ulp_dm = max(
        32.0 * u_ulp_scale,
        0.05 * max(float(np.median(np.maximum(mad_baseline[valid_mad], 0.0))), 0.0),
        1e-8,
    )
    sig_dm = max(sig_dm, ulp_dm)

    keep_cand = valid_mad & (delta_mad <= _TEMPORAL_MAD_SIGMA * sig_dm)
    if int(np.count_nonzero(keep_cand)) < _MIN_SURVIVING_PIXELS:
        return candidate_mask

    out_mask = np.zeros((ny, nx), dtype=bool)
    out_mask[candidate_mask] = keep_cand
    return out_mask


def _estimate_cadence_sky_sigma(
    u_cube: np.ndarray, sigma_u: np.ndarray | None
) -> float:
    """Estimate single-cadence sky noise (e-/s) for TESS PRF wing thresholding."""
    n_cad = u_cube.shape[0]
    n_sub = min(n_cad, 64)
    if sigma_u is not None:
        s_sub = np.asarray(sigma_u[:n_sub], dtype=np.float64)
        fin_s = np.isfinite(s_sub) & (s_sub > 0.0) & np.isfinite(u_cube[:n_sub])
        if np.any(fin_s):
            return max(float(np.percentile(s_sub[fin_s], 20.0)), 1e-6)
    if n_sub >= 2:
        u_sub = np.asarray(u_cube[:n_sub], dtype=np.float64)
        du = np.diff(u_sub, axis=0)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="All-NaN slice encountered",
                category=RuntimeWarning,
            )
            pix_mad = (
                _MAD_TO_SIGMA
                / math.sqrt(2.0)
                * np.nanmedian(np.abs(du), axis=0)
            )
        fin_m = np.isfinite(pix_mad) & (pix_mad > 0.0)
        if np.any(fin_m):
            return max(float(np.percentile(pix_mad[fin_m], 20.0)), 1e-6)
    return 1.0


def build_background_mask(
    u_cube: np.ndarray,
    *,
    sigma_u: np.ndarray | None = None,
    pixel_quality: np.ndarray | None = None,
    order: int | str | None = None,
    mask: np.ndarray | None = None,
    wcs: WCS | None = None,
    gaia_sources: Any = None,
    source_mag_limit: float = 19.0,
    max_source_radius: float | None = None,
    allow_downgrade: bool = True,
) -> BackgroundMaskResult:
    """Build or validate ``Omega_bg`` with dark-cadence detrended masking & guards."""
    if u_cube.ndim != 3:
        raise ValueError(
            f"u_cube must be 3D (N_cad, ny, nx), got shape {u_cube.shape}."
        )
    n_cad, ny, nx = u_cube.shape
    req_order = parse_order(order, ny, nx)
    cad_sky_sigma = (
        _estimate_cadence_sky_sigma(u_cube, sigma_u)
        if gaia_sources is not None
        else 1.0
    )
    src_mask = build_wcs_source_mask(
        ny,
        nx,
        wcs,
        gaia_sources,
        mag_limit=source_mag_limit,
        sky_sigma=cad_sky_sigma,
        max_radius_px=max_source_radius,
    )

    if mask is not None:
        user_mask = np.asarray(mask, dtype=bool)
        if user_mask.shape != (ny, nx):
            raise ValueError(
                f"mask must have shape ({ny}, {nx}), got {user_mask.shape}."
            )
        finite_any = np.zeros((ny, nx), dtype=bool)
        for start in range(0, n_cad, _CADENCE_CHUNK):
            finite_any |= np.any(
                np.isfinite(u_cube[start : start + _CADENCE_CHUNK]), axis=0
            )
        final_mask = user_mask & finite_any & (~src_mask)
        eff_order, kappa = apply_curvature_and_rank_guards(
            final_mask, req_order, allow_downgrade=allow_downgrade
        )
        return BackgroundMaskResult(
            mask=final_mask,
            source_mask=src_mask,
            requested_order=req_order,
            effective_order=eff_order,
            condition_number=kappa,
            dark_median_image=None,
            coarse_surface=None,
            sky_sigma=None,
        )

    # 1. Static PixelQuality & WCS catalogue exclusions + dead-edge rejection.
    eligible = ~src_mask.copy()
    if pixel_quality is not None:
        pq = np.asarray(pixel_quality, dtype=np.uint16)
        if pq.shape != (ny, nx):
            raise ValueError(
                f"pixel_quality shape {pq.shape} does not match ({ny}, {nx})."
            )
        excl = np.uint16(
            PixelQuality.STRAP_COL
            | PixelQuality.SAT_BLEED
            | PixelQuality.SMEAR_UNRESOLVED
            | PixelQuality.SOURCE_MASK
        )
        eligible &= (pq & excl) == 0

    # Chunked finite & positive-activity check to exclude dead U <= 0 edges
    # without allocating > 270 MB on 120,000-cadence 20-s TPFs.
    pos_counts = np.zeros((ny, nx), dtype=np.int64)
    fin_counts = np.zeros((ny, nx), dtype=np.int64)
    for start in range(0, n_cad, _CADENCE_CHUNK):
        chunk = u_cube[start : start + _CADENCE_CHUNK]
        fin_c = np.isfinite(chunk)
        fin_counts += np.sum(fin_c, axis=0)
        pos_counts += np.sum(fin_c & (chunk > 0.0), axis=0)

    # Live pixels have finite samples and strictly positive median (>50% > 0).
    eligible &= (fin_counts > 0) & (2 * pos_counts > fin_counts)
    if int(np.count_nonzero(eligible)) < _MIN_SURVIVING_PIXELS:
        raise ValueError(
            "Insufficient finite background pixels "
            f"({int(np.count_nonzero(eligible))} < {_MIN_SURVIVING_PIXELS}) "
            "surviving pixel-quality and U <= 0 dead-edge exclusions."
        )

    # 2. Rank cadences by spatial 10th-percentile flux across eligible live pixels.
    cad_q10 = np.full(n_cad, np.nan, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="All-NaN slice encountered",
            category=RuntimeWarning,
        )
        for start in range(0, n_cad, _CADENCE_CHUNK):
            sub = np.asarray(
                u_cube[start : start + _CADENCE_CHUNK][:, eligible],
                dtype=np.float64,
            )
            # Mask <= 0 per-cadence dropouts when computing spatial Q10.
            sub_pos = np.where(sub > 0.0, sub, np.nan)
            has_pos = np.any(np.isfinite(sub_pos), axis=1)
            if np.any(has_pos):
                cad_q10[start : start + _CADENCE_CHUNK][has_pos] = (
                    np.nanpercentile(
                        sub_pos[has_pos], _DARK_CADENCE_PERCENTILE, axis=1
                    )
                )

    valid_cad_idx = np.flatnonzero(np.isfinite(cad_q10))
    if valid_cad_idx.size == 0:
        raise ValueError(
            "Insufficient finite background pixels: no cadences contain "
            "positive finite unflagged flux."
        )

    n_dark = max(1, int(np.ceil(_DARK_CADENCE_FRACTION * valid_cad_idx.size)))
    dark_rel = np.argpartition(cad_q10[valid_cad_idx], n_dark - 1)[:n_dark]
    dark_rows = np.sort(valid_cad_idx[dark_rel])

    # If N_dark itself is huge (e.g. 30,000 cadences on 20-s TPF), stride
    # uniformly to <= 4096 dark cadences so nanmedian stays < 135 MB RAM.
    deg_p0 = 1 if min(nx, ny) < 40 else min(req_order, 2)
    dark_stride = max(1, dark_rows.size // _MAX_TEMPORAL_MAD_CADENCES)
    dark_cube = np.asarray(
        u_cube[dark_rows[::dark_stride]], dtype=np.float64
    )
    med_img = _compute_aligned_dark_median(dark_cube, eligible, deg_p0)
    del dark_cube

    eligible &= np.isfinite(med_img) & (med_img > 0.0)
    if int(np.count_nonzero(eligible)) < _MIN_SURVIVING_PIXELS:
        raise ValueError(
            "Insufficient finite background pixels "
            f"({int(np.count_nonzero(eligible))} < {_MIN_SURVIVING_PIXELS}) "
            "with positive dark-cadence median flux."
        )

    # 3. Fit coarse robust surface P_0(r, c) on local block-15th-percentile tiles:
    #    degree 1 when min(nx, ny) < 40; degree min(d, 2) when min(nx, ny) >= 40.
    p0_surface, sigma_sky = _fit_coarse_surface_p0(med_img, eligible, deg_p0)
    resid_spatial = med_img - p0_surface
    spatial_mask = (
        eligible
        & (resid_spatial >= -_SPATIAL_LOWER_SIGMA * sigma_sky)
        & (resid_spatial <= _SPATIAL_UPPER_SIGMA * sigma_sky)
    )
    if int(np.count_nonzero(spatial_mask)) < _MIN_SURVIVING_PIXELS:
        raise ValueError(
            "Insufficient finite background pixels "
            f"({int(np.count_nonzero(spatial_mask))} < {_MIN_SURVIVING_PIXELS}) "
            "surviving spatial sky residual clipping."
        )

    # 4. Reject anomalous temporal MAD on preliminary per-cadence detrended residuals.
    final_mask = _reject_anomalous_temporal_mad(
        u_cube=u_cube,
        sigma_u=sigma_u,
        candidate_mask=spatial_mask,
        requested_order=req_order,
    )

    # 5. Apply curvature & rank guards (auto-downgrade d -> d - 1 down to 0).
    eff_order, kappa = apply_curvature_and_rank_guards(
        final_mask, req_order, allow_downgrade=allow_downgrade
    )

    return BackgroundMaskResult(
        mask=final_mask,
        source_mask=src_mask,
        requested_order=req_order,
        effective_order=eff_order,
        condition_number=kappa,
        dark_median_image=med_img,
        coarse_surface=p0_surface,
        sky_sigma=sigma_sky,
    )
