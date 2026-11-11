"""Legacy ``bkgd_tpf`` TargetPixelFile shim with modernized numerics and setters."""

from __future__ import annotations

import warnings
from typing import Any

import astropy.units as u
import lightkurve
import numpy as np
from astropy.units import Quantity
from scipy.linalg import lstsq

from ._contract import read_bkgsub_state

_ALT_COUNT_RATE_UNITS = (
    u.Unit("ct / s"),
    u.Unit("adu / s"),
    u.Unit("photon / s"),
)


def _as_float64_ndarray(arr: Any) -> np.ndarray:
    """Extract a raw ``float64`` NumPy ``ndarray`` regardless of Astropy units."""
    if isinstance(arr, Quantity):
        if arr.unit is not None and arr.unit != u.dimensionless_unscaled:
            if arr.unit.is_equivalent(u.Unit("electron / s")):
                return np.asarray(
                    arr.to_value(u.Unit("electron / s")), dtype=np.float64
                )
        return np.asarray(arr.value, dtype=np.float64)
    if hasattr(arr, "value"):
        return np.asarray(arr.value, dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)


def _strip_for_fits_column(value: Any) -> np.ndarray:
    """Strip Astropy ``Quantity`` units and cast to ``float32`` for ``FITS_rec``."""
    if isinstance(value, Quantity):
        if value.unit is None or value.unit == u.dimensionless_unscaled:
            return np.asarray(value.value, dtype=np.float32)
        if value.unit.is_equivalent(u.Unit("electron / s")):
            return np.asarray(
                value.to_value(u.Unit("electron / s")), dtype=np.float32
            )
        if any(value.unit.is_equivalent(alt) for alt in _ALT_COUNT_RATE_UNITS):
            return np.asarray(value.value, dtype=np.float32)
        raise u.UnitConversionError(
            f"Cannot assign Quantity with unit {value.unit!r} to TESS flux column "
            "(expected 'electron/s' or dimensionless)."
        )
    if hasattr(value, "value"):
        return np.asarray(value.value, dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


class bkgd_tpf(lightkurve.TessTargetPixelFile):
    """Subclass of :class:`lightkurve.TessTargetPixelFile` for 2D background fitting."""

    def _flux_unit(self) -> str | None:
        """Return ``'electron/s'`` if the ``FLUX`` column carries electron/s units."""
        if len(self.hdu) > 1 and self.hdu[1].columns is not None:
            col_map = {c.name.upper(): c for c in self.hdu[1].columns}
            flux_col = col_map.get("FLUX")
            if flux_col is not None:
                if flux_col.unit in ("e-/s", "electron/s", "electron / s"):
                    return "electron/s"
                return None
        hdr = self.get_header(1)
        if hdr.get("TUNIT5") in ("e-/s", "electron/s", "electron / s"):
            return "electron/s"
        return None

    def _assign_table_column(self, col_name: str, value: Any) -> None:
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

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="All-NaN slice encountered",
                category=RuntimeWarning,
            )
            med_img = np.nanmedian(rawflux, axis=0)

        finite_med_mask = np.isfinite(med_img)

        if bkgdmask is None:
            finite_vals = med_img[finite_med_mask]
            hist, bins = np.histogram(finite_vals, bins=np.arange(0.0, 500.0, 10.0))
            if np.any(hist > 0):
                median_bkgd_lvl = float(bins[int(np.argmax(hist))] + 5.0)
                mask = (
                    finite_med_mask
                    & (med_img <= 1.1 * median_bkgd_lvl)
                    & (med_img >= 0.9 * median_bkgd_lvl)
                )
            else:
                mask = np.zeros((ny, nx), dtype=bool)

            # Guard dark (< 10 e-/s) and bright (> 500 e-/s) sky regimes where
            # the legacy [0, 500) histogram would leave fewer than P pixels.
            if np.count_nonzero(mask) < n_params and finite_vals.size >= n_params:
                q25 = float(np.percentile(finite_vals, 25.0))
                lower = finite_vals[finite_vals <= np.median(finite_vals)]
                mad = float(np.median(np.abs(lower - np.median(lower))))
                half_width = max(0.15 * abs(q25), 3.0 * 1.4826 * mad, 1e-3)
                mask = finite_med_mask & (np.abs(med_img - q25) <= half_width)
                if np.count_nonzero(mask) < n_params:
                    order_idx = np.argsort(np.abs(med_img - q25), axis=None)
                    mask = np.zeros((ny, nx), dtype=bool)
                    valid_flat = finite_med_mask.ravel()
                    chosen = [i for i in order_idx if valid_flat[i]][
                        : max(n_params, finite_vals.size // 4)
                    ]
                    mask.ravel()[chosen] = True
            bkgdmask_out = mask
        else:
            user_mask = np.asarray(bkgdmask, dtype=bool)
            if user_mask.shape != (ny, nx):
                raise ValueError(
                    f"bkgdmask must have shape ({ny}, {nx}), got {user_mask.shape}."
                )
            bkgdmask_out = user_mask & finite_med_mask

        n_used = int(np.count_nonzero(bkgdmask_out))
        if n_used < n_params:
            raise ValueError(
                f"Insufficient finite background pixels ({n_used}) for "
                f"{order!r} background fit (requires at least {n_params})."
            )

        if verbose:
            print(
                f"Fitting {order} background across {n_cad} cadences "
                f"using {n_used}/{ny * nx} pixels."
            )

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

        design = basis_grid[:, bkgdmask_out].T  # (M, P)
        z = rawflux[:, bkgdmask_out]  # (N_t, M)
        finite_z = np.isfinite(z)
        all_finite_cad = np.all(finite_z, axis=1)

        coeffs = np.full((n_cad, n_params), np.nan, dtype=np.float64)
        if np.any(all_finite_cad):
            sol, _, _, _ = lstsq(design, z[all_finite_cad].T)
            coeffs[all_finite_cad] = sol.T

        # Per-cadence solve on remaining finite pixels whenever partial NaNs
        # occur, avoiding single-pixel NaN wipeouts to 0.0.
        for t in np.flatnonzero(~all_finite_cad):
            ok = finite_z[t]
            if np.count_nonzero(ok) >= n_params:
                sol_t, _, _, _ = lstsq(design[ok], z[t, ok])
                coeffs[t] = sol_t

        # Reconstruct via (N_t, P) @ (P, ny * nx) GEMM without 4D np.tile.
        bkgd = (coeffs @ basis_grid.reshape(n_params, ny * nx)).reshape(
            n_cad, ny, nx
        )

        unit = self._flux_unit()
        return Quantity(bkgd, unit=unit), bkgdmask_out

    @property
    def flux(self) -> Quantity:
        """Return the flux cube for all good-quality cadences."""
        return Quantity(
            self.hdu[1].data["FLUX"][self.quality_mask], unit=self._flux_unit()
        )

    @flux.setter
    def flux(self, value: Any) -> None:
        self._assign_table_column("FLUX", value)

    @property
    def flux_bkg(self) -> Quantity:
        """Return the background flux cube for all good-quality cadences."""
        return Quantity(
            self.hdu[1].data["FLUX_BKG"][self.quality_mask],
            unit=self._flux_unit(),
        )

    @flux_bkg.setter
    def flux_bkg(self, value: Any) -> None:
        self._assign_table_column("FLUX_BKG", value)

    @property
    def raw_cnts(self) -> np.ndarray:
        """Return the raw counts for all good-quality cadences."""
        return self.hdu[1].data["RAW_CNTS"][self.quality_mask]
