"""Legacy ``bkgd_tpf`` TargetPixelFile shim with modernized numerics and setters."""

from __future__ import annotations

import re
import warnings
from typing import Any

import astropy.units as u
import lightkurve
import numpy as np
from astropy.units import Quantity
from scipy.linalg import lstsq

from ._contract import PixelQuality, _set_bkgsub_header, read_bkgsub_state

_CANONICAL_RATE_UNITS = (
    u.Unit("electron / s"),
    u.Unit("ct / s"),
    u.Unit("adu / s"),
    u.Unit("photon / s"),
)
_RANK_TOL = 1e-10


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
        # TESS FITS tables write 'e-/s' or 'e- s-1', whereas astropy.units
        # spells the electron count unit 'electron'.
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


def _estimate_sky_mask(
    rawflux: np.ndarray,
    basis_grid: np.ndarray,
    eta_grid: np.ndarray,
    xi_grid: np.ndarray,
    pq: np.ndarray | None,
    n_params: int,
) -> np.ndarray:
    """Select uncontaminated sky pixels via shift- and scale-invariant detrending.

    Forms a dark-cadence median image over the darkest 25% of cadences (excluding
    ``STRAP_COL | SAT_BLEED | SMEAR_UNRESOLVED | SOURCE_MASK`` when
    ``PIXEL_QUALITY`` is present), seeds from the lower quartile of each spatial
    quadrant, and refines inliers via asymmetric lower-half MAD clipping against
    the fitted 2D polynomial background surface.
    """
    _, ny, nx = rawflux.shape
    eligible = np.any(np.isfinite(rawflux), axis=0)
    if pq is not None and pq.shape == (ny, nx):
        excl = np.uint16(
            PixelQuality.STRAP_COL
            | PixelQuality.SAT_BLEED
            | PixelQuality.SMEAR_UNRESOLVED
            | PixelQuality.SOURCE_MASK
        )
        eligible = eligible & ((pq & excl) == 0)

    if int(np.count_nonzero(eligible)) < n_params:
        return np.zeros((ny, nx), dtype=bool)

    flux_elig = rawflux[:, eligible]
    valid_cad = np.any(np.isfinite(flux_elig), axis=1)
    if not np.any(valid_cad):
        return np.zeros((ny, nx), dtype=bool)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="All-NaN slice encountered",
            category=RuntimeWarning,
        )
        cad_q10 = np.nanpercentile(flux_elig[valid_cad], 10.0, axis=1)
        n_dark = max(1, int(np.ceil(0.25 * cad_q10.size)))
        dark_rel = np.argpartition(cad_q10, n_dark - 1)[:n_dark]
        dark_rows = np.flatnonzero(valid_cad)[dark_rel]
        med_img = np.nanmedian(rawflux[dark_rows], axis=0)

    eligible = eligible & np.isfinite(med_img)
    if int(np.count_nonzero(eligible)) < n_params:
        return np.zeros((ny, nx), dtype=bool)

    y_elig = med_img[eligible]
    basis_elig = basis_grid[:, eligible].T
    global_med = float(np.median(y_elig))

    seed_2d = np.zeros((ny, nx), dtype=bool)
    quadrants = (
        (eta_grid <= 0.0) & (xi_grid <= 0.0),
        (eta_grid <= 0.0) & (xi_grid > 0.0),
        (eta_grid > 0.0) & (xi_grid <= 0.0),
        (eta_grid > 0.0) & (xi_grid > 0.0),
    )
    for qm in quadrants:
        eq = eligible & qm
        if np.any(eq):
            q_cut = float(np.percentile(med_img[eq], 25.0))
            seed_2d |= eq & (med_img <= q_cut) & (med_img <= global_med)

    inliers = seed_2d[eligible]
    if (
        int(np.count_nonzero(inliers)) < n_params
        or int(np.linalg.matrix_rank(basis_elig[inliers], tol=_RANK_TOL))
        < n_params
    ):
        q25 = float(np.percentile(y_elig, 25.0))
        inliers = y_elig <= q25

    ulp_floor = 8.0 * float(
        np.median(np.abs(np.spacing(y_elig.astype(np.float32))))
    )
    for _ in range(5):
        if int(np.count_nonzero(inliers)) < n_params:
            break
        beta, _, rank, _ = lstsq(
            basis_elig[inliers], y_elig[inliers], cond=_RANK_TOL
        )
        if rank < n_params:
            break
        resid = y_elig - basis_elig @ beta
        r_in = resid[inliers]
        r_med = float(np.median(r_in))
        lower_dev = r_med - r_in[r_in <= r_med]
        sigma_low = (
            1.4826 * float(np.median(lower_dev)) if lower_dev.size > 0 else 0.0
        )
        sigma = max(sigma_low, ulp_floor)
        next_inliers = (resid - r_med >= -3.5 * sigma) & (
            resid - r_med <= 3.0 * sigma
        )
        if np.array_equal(next_inliers, inliers):
            break
        inliers = next_inliers

    mask = np.zeros((ny, nx), dtype=bool)
    mask[eligible] = inliers
    return mask


class bkgd_tpf(lightkurve.TessTargetPixelFile):
    """Subclass of :class:`lightkurve.TessTargetPixelFile` for 2D background fitting."""

    def _column_unit(self, col_name: str = "FLUX") -> u.Unit | None:
        """Return the parsed Astropy :class:`Unit` for ``col_name`` if present."""
        target = col_name.upper()
        hdr = self.get_header(1)
        if len(self.hdu) > 1 and self.hdu[1].columns is not None:
            for idx, col in enumerate(self.hdu[1].columns, start=1):
                if col.name.upper() == target:
                    raw_unit = col.unit or hdr.get(f"TUNIT{idx}")
                    return _parse_tpf_unit(raw_unit)
            return None
        if target == "FLUX":
            return _parse_tpf_unit(hdr.get("TUNIT5"))
        if target == "FLUX_BKG":
            return _parse_tpf_unit(hdr.get("TUNIT7"))
        return None

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

    def get_bkgd(
        self,
        bkgdmask: np.ndarray | None = None,
        order: str = "linear",
        verbose: bool = False,
    ) -> tuple[Quantity, np.ndarray]:
        """Calculate a spatially varying 2D polynomial background per cadence.

        Parameters
        ----------
        bkgdmask : ndarray of bool, shape ``(ny, nx)``, optional
            Boolean mask selecting background pixels. Computed automatically if
            ``None``.
        order : {"linear", "quadratic"}, default "linear"
            Spatial polynomial degree (3 parameters for ``'linear'``, 6 for
            ``'quadratic'``).
        verbose : bool, default False
            If ``True``, print a diagnostic summary of the background fit.

        Returns
        -------
        bkgd : astropy.units.Quantity, shape ``(N_t, ny, nx)``
            Estimated background flux cube across good-quality cadences.
        bkgdmask : numpy.ndarray of bool, shape ``(ny, nx)``
            Spatial pixel mask used for the background fit.
        """
        if order == "linear":
            n_params = 3
        elif order == "quadratic":
            n_params = 6
        else:
            raise ValueError(
                f"Unrecognised order {order!r}. Expected 'linear' or 'quadratic'."
            )

        flux = _as_float64_ndarray(self.flux)
        col_names_upper = {
            name.upper() for name in (self.hdu[1].data.names or ())
        }
        if read_bkgsub_state(self) and "FLUX_BKG" in col_names_upper:
            oldbkgd = _as_float64_ndarray(self.flux_bkg)
            rawflux = flux + np.where(np.isfinite(oldbkgd), oldbkgd, 0.0)
        else:
            rawflux = flux

        n_cad, ny, nx = rawflux.shape

        # Normalized [-1, 1] coordinates span the exact same polynomial space
        # while avoiding Vandermonde ill-conditioning on large cutouts.
        eta_1d = (
            np.linspace(-1.0, 1.0, ny, dtype=np.float64)
            if ny > 1
            else np.zeros(1, dtype=np.float64)
        )
        xi_1d = (
            np.linspace(-1.0, 1.0, nx, dtype=np.float64)
            if nx > 1
            else np.zeros(1, dtype=np.float64)
        )
        eta_grid, xi_grid = np.meshgrid(eta_1d, xi_1d, indexing="ij")

        if order == "quadratic":
            basis_grid = np.stack(
                [
                    np.ones((ny, nx), dtype=np.float64),
                    eta_grid,
                    xi_grid,
                    eta_grid**2,
                    eta_grid * xi_grid,
                    xi_grid**2,
                ],
                axis=0,
            )
        else:
            basis_grid = np.stack(
                [
                    np.ones((ny, nx), dtype=np.float64),
                    eta_grid,
                    xi_grid,
                ],
                axis=0,
            )

        if bkgdmask is None:
            pq_arr = (
                np.asarray(self.hdu["PIXEL_QUALITY"].data, dtype=np.uint16)
                if (
                    "PIXEL_QUALITY" in self.hdu
                    and self.hdu["PIXEL_QUALITY"].data is not None
                )
                else None
            )
            bkgdmask_out = _estimate_sky_mask(
                rawflux=rawflux,
                basis_grid=basis_grid,
                eta_grid=eta_grid,
                xi_grid=xi_grid,
                pq=pq_arr,
                n_params=n_params,
            )
        else:
            user_mask = np.asarray(bkgdmask, dtype=bool)
            if user_mask.shape != (ny, nx):
                raise ValueError(
                    f"bkgdmask must have shape ({ny}, {nx}), got {user_mask.shape}."
                )
            bkgdmask_out = user_mask & np.any(np.isfinite(rawflux), axis=0)

        n_used = int(np.count_nonzero(bkgdmask_out))
        if n_used < n_params:
            raise ValueError(
                f"Insufficient finite background pixels ({n_used}) for "
                f"{order!r} background fit (requires at least {n_params})."
            )

        design = basis_grid[:, bkgdmask_out].T  # (M, P)
        static_rank = int(np.linalg.matrix_rank(design, tol=_RANK_TOL))
        if static_rank < n_params:
            raise ValueError(
                f"Background mask of {n_used} pixels is geometrically "
                f"rank-deficient for {order!r} 2D polynomial fit "
                f"(rank {static_rank} < {n_params})."
            )

        if verbose:
            print(
                f"Fitting {order} background across {n_cad} cadences "
                f"using {n_used}/{ny * nx} pixels."
            )

        z = rawflux[:, bkgdmask_out]  # (N_t, M)
        finite_z = np.isfinite(z)
        all_finite_cad = np.all(finite_z, axis=1)

        coeffs = np.full((n_cad, n_params), np.nan, dtype=np.float64)
        if np.any(all_finite_cad):
            sol, _, rank_all, _ = lstsq(
                design, z[all_finite_cad].T, cond=_RANK_TOL
            )
            if rank_all < n_params:
                raise ValueError(
                    f"Rank-deficient background design matrix "
                    f"(rank {rank_all} < {n_params})."
                )
            coeffs[all_finite_cad] = sol.T

        # Per-cadence solve on remaining finite pixels whenever partial NaNs
        # occur, avoiding single-pixel NaN wipeouts to 0.0 and leaving cadences
        # whose surviving finite pixels are rank-deficient as NaN.
        for t in np.flatnonzero(~all_finite_cad):
            ok = finite_z[t]
            if int(np.count_nonzero(ok)) >= n_params:
                sol_t, _, rank_t, _ = lstsq(
                    design[ok], z[t, ok], cond=_RANK_TOL
                )
                if rank_t == n_params:
                    coeffs[t] = sol_t

        # Reconstruct via (N_t, P) @ (P, ny * nx) GEMM without 4D np.tile.
        bkgd = (coeffs @ basis_grid.reshape(n_params, ny * nx)).reshape(
            n_cad, ny, nx
        )

        unit = self._column_unit("FLUX")
        return Quantity(bkgd, unit=unit), bkgdmask_out

    def subtract_bkgd(
        self,
        bkgdmask: np.ndarray | None = None,
        order: str = "linear",
        verbose: bool = False,
    ) -> tuple[Quantity, np.ndarray]:
        """Fit and subtract a 2D polynomial background, updating ``BKGSUB`` provenance.

        Honours ``read_bkgsub_state(self)`` across SPOC, ``TessCut``, and
        ``mundey`` v2.0 TPFs and stamps ``BKGSUB = True`` onto ``hdu[0]``,
        ``hdu[1]``, and ``self.meta`` so repeated calls are idempotent.
        """
        was_subtracted = read_bkgsub_state(self)
        col_names_upper = {
            name.upper() for name in (self.hdu[1].data.names or ())
        }
        newbkgd, mask = self.get_bkgd(
            bkgdmask=bkgdmask, order=order, verbose=verbose
        )
        if was_subtracted and "FLUX_BKG" in col_names_upper:
            self.flux = self.flux + self.flux_bkg - newbkgd
        else:
            self.flux = self.flux - newbkgd
        self.flux_bkg = newbkgd
        return newbkgd, mask

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
        _set_bkgsub_header(self, has_nonzero_bkg)

    @property
    def raw_cnts(self) -> np.ndarray:
        """Return the raw counts for all good-quality cadences."""
        return self.hdu[1].data["RAW_CNTS"][self.quality_mask]
