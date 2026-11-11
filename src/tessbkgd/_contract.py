"""Versioned FITS contract (``HALOCVER = 1``) and HPC cache helpers.

This module has zero cross-package runtime imports: ``mundey``, ``tessbkgd``,
and ``halophot`` communicate strictly through standard
``lightkurve.TargetPixelFile`` / ``astropy.io.fits.HDUList`` structures under
the ``HALOCVER = 1`` specification.
"""

from __future__ import annotations

import math
import os
import warnings
from enum import IntFlag
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

HALOCVER: int = 1


class PixelQuality(IntFlag):
    """Spatial pixel-quality bitmask stored in ``PIXEL_QUALITY`` (``uint16``)."""

    STRAP_COL = 1 << 0
    SAT_BLEED = 1 << 1
    SMEAR_UNRESOLVED = 1 << 2
    SOURCE_MASK = 1 << 3
    BKG_USED = 1 << 4


def _unwrap_hdulist(tpf_or_hdu: Any) -> fits.HDUList:
    if isinstance(tpf_or_hdu, fits.HDUList):
        return tpf_or_hdu
    if hasattr(tpf_or_hdu, "hdu") and isinstance(tpf_or_hdu.hdu, fits.HDUList):
        return tpf_or_hdu.hdu
    raise TypeError(
        "Expected a lightkurve.TargetPixelFile or astropy.io.fits.HDUList, "
        f"got {type(tpf_or_hdu).__name__}."
    )


def _raw_float64(arr: Any) -> np.ndarray:
    if hasattr(arr, "value"):
        arr = arr.value
    return np.asarray(arr, dtype=np.float64)


def read_bkgsub_state(tpf: Any) -> bool:
    """Determine whether ``tpf`` currently stores background-subtracted ``FLUX``.

    Follows the ``HALOCVER = 1`` provenance rules:

    1. If ``BKGSUB`` is explicitly present as a boolean in ``hdu[0].header`` or
       ``hdu[1].header``, return its value directly (``False`` on ``mundey``
       v2.0 TPFs even if ``FLUX_BKG`` is non-zero; ``True`` after ``tessbkgd``
       subtraction).
    2. When ``BKGSUB`` is absent, inspect ``FLUX_BKG``: if absent, all-zero, or
       all-``NaN`` (as on ``TessCut`` FFI cutouts), return ``False``.
    3. When ``BKGSUB`` is absent and ``FLUX_BKG`` contains finite non-zero
       values (standard SPOC / TESS-SPOC HLSP TPFs), emit a defensive
       :class:`UserWarning` if the dark-cadence 10th-percentile of ``FLUX``
       exceeds ``0.5 * nanmedian(FLUX_BKG)`` (catching legacy ``mundey`` v0.1
       TPFs whose unsubtracted ``FLUX`` retained stale SPOC ``FLUX_BKG``), and
       return ``True``.
    """
    hdulist = _unwrap_hdulist(tpf)

    for ext in (0, 1):
        if ext < len(hdulist) and "BKGSUB" in hdulist[ext].header:
            card_val = hdulist[ext].header["BKGSUB"]
            if isinstance(card_val, bool):
                return card_val

    if len(hdulist) < 2 or hdulist[1].data is None:
        return False

    col_names = hdulist[1].data.names or ()
    col_map = {name.upper(): name for name in col_names}
    if "FLUX_BKG" not in col_map:
        return False

    raw_bkg_col = hdulist[1].data[col_map["FLUX_BKG"]]
    qmask = getattr(tpf, "quality_mask", None)
    use_qmask = (
        qmask is not None
        and len(qmask) == len(raw_bkg_col)
        and bool(np.any(qmask))
    )
    bkg_arr = _raw_float64(raw_bkg_col[qmask] if use_qmask else raw_bkg_col)

    if not np.any(np.isfinite(bkg_arr) & (bkg_arr != 0.0)):
        return False

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered", category=RuntimeWarning
        )
        med_bkg = float(np.nanmedian(bkg_arr))

    if np.isfinite(med_bkg) and med_bkg > 0.0 and "FLUX" in col_map:
        raw_flux_col = hdulist[1].data[col_map["FLUX"]]
        flux_arr = _raw_float64(raw_flux_col[qmask] if use_qmask else raw_flux_col)

        if (
            "PIXEL_QUALITY" in hdulist
            and hdulist["PIXEL_QUALITY"].data is not None
            and hdulist["PIXEL_QUALITY"].data.shape == flux_arr.shape[1:]
        ):
            pq = np.asarray(hdulist["PIXEL_QUALITY"].data, dtype=np.uint16)
            excl = np.uint16(
                PixelQuality.STRAP_COL
                | PixelQuality.SAT_BLEED
                | PixelQuality.SMEAR_UNRESOLVED
            )
            bad_px = (pq & excl) != 0
            if not np.all(bad_px):
                flux_spatial = flux_arr[:, ~bad_px]
            else:
                flux_spatial = flux_arr.reshape(flux_arr.shape[0], -1)
        else:
            flux_spatial = flux_arr.reshape(flux_arr.shape[0], -1)

        valid_cad = np.any(np.isfinite(flux_spatial), axis=1)
        if np.any(valid_cad):
            flux_valid = flux_spatial[valid_cad]
            cad_q10 = np.nanpercentile(flux_valid, 10.0, axis=1)
            n_dark = max(1, int(np.ceil(0.25 * cad_q10.size)))
            dark_idx = np.argpartition(cad_q10, n_dark - 1)[:n_dark]
            dark_q10 = float(np.nanpercentile(flux_valid[dark_idx], 10.0))
            if np.isfinite(dark_q10) and dark_q10 > 0.5 * med_bkg:
                warnings.warn(
                    "BKGSUB header card is absent while FLUX_BKG is non-zero "
                    f"(nanmedian={med_bkg:.3g}) and dark-cadence 10th-percentile "
                    f"FLUX ({dark_q10:.3g}) exceeds 0.5 * nanmedian(FLUX_BKG). "
                    "If this TPF was recalibrated by legacy mundey (< 2.0) "
                    "without zeroing FLUX_BKG, set hdu[0].header['BKGSUB'] = False "
                    "to avoid double-adding background.",
                    UserWarning,
                    stacklevel=2,
                )

    return True


def pack_bkg_chol(bkg_chol: np.ndarray) -> np.ndarray:
    """Pack lower-triangular Cholesky factors ``(N, P, P)`` into ``(N, P*(P+1)//2)``."""
    arr = np.asarray(bkg_chol, dtype=np.float32)
    if arr.ndim == 3:
        n_cad, p_rows, p_cols = arr.shape
        if p_rows != p_cols:
            raise ValueError(
                f"3D bkg_chol must have square trailing dimensions, got {arr.shape}."
            )
        rows, cols = np.tril_indices(p_rows)
        return np.ascontiguousarray(arr[:, rows, cols], dtype=np.float32)
    if arr.ndim == 2:
        return np.ascontiguousarray(arr, dtype=np.float32)
    raise ValueError(
        "bkg_chol must have shape (N_raw, P*(P+1)//2) or (N_raw, P, P), "
        f"got {arr.shape}."
    )


def unpack_bkg_chol(
    bkg_chol_packed: np.ndarray, n_basis: int | None = None
) -> np.ndarray:
    """Unpack ``(N, P*(P+1)//2)`` lower-triangular Cholesky factors to ``(N, P, P)``."""
    arr = np.asarray(bkg_chol_packed)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.ndim != 2:
        raise ValueError(
            f"Packed bkg_chol must be 2D (N, P*(P+1)//2), got shape {arr.shape}."
        )
    n_cad, n_tril = arr.shape
    if n_basis is None:
        inferred = int(round((-1.0 + math.sqrt(1.0 + 8.0 * n_tril)) / 2.0))
        n_basis = inferred
    if n_basis * (n_basis + 1) // 2 != n_tril:
        raise ValueError(
            f"Packed width {n_tril} does not match P*(P+1)//2 for P={n_basis}."
        )
    out = np.zeros((n_cad, n_basis, n_basis), dtype=arr.dtype)
    rows, cols = np.tril_indices(n_basis)
    out[:, rows, cols] = arr
    return out


def write_tessbkgd_provenance(
    hdu: Any,
    bkg_basis: np.ndarray,
    bkg_chol: np.ndarray,
    bkg_used_mask: np.ndarray,
) -> fits.HDUList:
    """Write ``HALOCVER = 1`` provenance cards, spatial HDUs, and ``BKG_CHOL`` column.

    Parameters
    ----------
    hdu : astropy.io.fits.HDUList or lightkurve.TargetPixelFile
        Target FITS ``HDUList`` (or ``TargetPixelFile`` wrapping ``.hdu``).
    bkg_basis : array-like, shape ``(P, ny, nx)``
        Static 2D background basis evaluated on the TPF pixel grid.
    bkg_chol : array-like, shape ``(N_raw, P*(P+1)//2)`` or ``(N_raw, P, P)``
        Per-cadence lower-triangular Cholesky factor of the basis-coefficient
        covariance across all ``N_raw`` rows of ``hdu[1]`` (packed via
        ``np.tril_indices(P)``).
    bkg_used_mask : array-like of bool, shape ``(ny, nx)``
        Spatial boolean mask identifying pixels used in the background fit
        (:attr:`PixelQuality.BKG_USED`, Bit 4).

    Returns
    -------
    astropy.io.fits.HDUList
        The updated ``HDUList``.
    """
    hdulist = _unwrap_hdulist(hdu)
    if len(hdulist) < 2 or hdulist[1].data is None:
        raise ValueError(
            "HDUList must contain a populated PIXELS BinTableHDU at hdu[1]."
        )

    used = np.asarray(bkg_used_mask, dtype=bool)
    if used.ndim != 2:
        raise ValueError(f"bkg_used_mask must be 2D (ny, nx), got shape {used.shape}.")
    ny, nx = used.shape

    basis_f32 = np.ascontiguousarray(bkg_basis, dtype=np.float32)
    if basis_f32.ndim != 3 or basis_f32.shape[1:] != (ny, nx):
        raise ValueError(
            f"bkg_basis must have shape (P, {ny}, {nx}), got {basis_f32.shape}."
        )
    n_basis = basis_f32.shape[0]
    n_tril = n_basis * (n_basis + 1) // 2

    n_raw = len(hdulist[1].data)
    chol_packed = pack_bkg_chol(bkg_chol)
    if chol_packed.shape != (n_raw, n_tril):
        raise ValueError(
            f"Packed bkg_chol must have shape ({n_raw}, {n_tril}) matching N_raw "
            f"and P={n_basis}, got {chol_packed.shape}."
        )

    for ext in (0, 1):
        hdulist[ext].header["HALOCVER"] = (
            HALOCVER,
            "halophot/mundey/tessbkgd FITS contract version",
        )
        hdulist[ext].header["BKGSUB"] = (
            True,
            "Sky background subtracted from FLUX",
        )

    # Update or create PIXEL_QUALITY ImageHDU (uint16, (ny, nx)).
    if "PIXEL_QUALITY" in hdulist and hdulist["PIXEL_QUALITY"].data is not None:
        pq = np.asarray(hdulist["PIXEL_QUALITY"].data, dtype=np.uint16).copy()
        if pq.shape != (ny, nx):
            raise ValueError(
                f"Existing PIXEL_QUALITY shape {pq.shape} does not match ({ny}, {nx})."
            )
        pq &= ~np.uint16(PixelQuality.BKG_USED)
        pq[used] |= np.uint16(PixelQuality.BKG_USED)
        hdulist["PIXEL_QUALITY"].data = pq
    else:
        pq = np.zeros((ny, nx), dtype=np.uint16)
        pq[used] |= np.uint16(PixelQuality.BKG_USED)
        hdulist.append(fits.ImageHDU(data=pq, name="PIXEL_QUALITY"))

    # Update or create BKG_BASIS ImageHDU (float32, (P, ny, nx)).
    if "BKG_BASIS" in hdulist:
        hdulist["BKG_BASIS"].data = basis_f32
    else:
        hdulist.append(fits.ImageHDU(data=basis_f32, name="BKG_BASIS"))

    # Write packed BKG_CHOL into hdu[1] (PIXELS BinTableHDU) for lockstep slicing.
    existing_names = hdulist[1].data.names or ()
    if "BKG_CHOL" in existing_names and (
        hdulist[1].data["BKG_CHOL"].shape == (n_raw, n_tril)
        or (n_tril == 1 and hdulist[1].data["BKG_CHOL"].shape == (n_raw,))
    ):
        if n_tril == 1 and hdulist[1].data["BKG_CHOL"].ndim == 1:
            hdulist[1].data["BKG_CHOL"] = chol_packed[:, 0]
        else:
            hdulist[1].data["BKG_CHOL"] = chol_packed
    else:
        kept_cols = [
            col for col in hdulist[1].columns if col.name.upper() != "BKG_CHOL"
        ]
        chol_col = fits.Column(
            name="BKG_CHOL",
            format=f"{n_tril}E",
            dim=f"({n_tril})",
            unit="e-/s",
            array=chol_packed,
        )
        old_header = hdulist[1].header.copy()
        ext_name = hdulist[1].name or "PIXELS"
        new_bintable = fits.BinTableHDU.from_columns(
            fits.ColDefs([*kept_cols, chol_col]),
            header=old_header,
            name=ext_name,
        )
        new_bintable.header["HALOCVER"] = (
            HALOCVER,
            "halophot/mundey/tessbkgd FITS contract version",
        )
        new_bintable.header["BKGSUB"] = (
            True,
            "Sky background subtracted from FLUX",
        )
        hdulist[1] = new_bintable

    if hasattr(hdu, "meta") and hasattr(hdu, "hdu"):
        try:
            hdu.meta = type(hdu.meta)(hdulist[0])
        except Exception:
            pass

    return hdulist


def _is_fred_path(path: Path) -> bool:
    for candidate in (path, Path(os.path.abspath(str(path)))):
        parts = candidate.parts
        if len(parts) >= 2 and parts[0] == "/" and parts[1] == "fred":
            return True
    return False


def resolve_cache_dir(
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    strict_hpc: bool = True,
) -> Path:
    """Resolve the package cache directory while guarding `/fred/` inode quotas.

    Defaults to ``$SCRATCH/.cache/tess_bright`` whenever ``SCRATCH`` is set in
    the environment (e.g. ``/aphid/scratch-3month/$USER/.cache/tess_bright`` on
    Swinburne Ngarrgu Tindebeek / OzSTAR), falling back to
    ``$XDG_CACHE_HOME/tess_bright`` or ``~/.cache/tess_bright`` otherwise.

    Parameters
    ----------
    cache_dir : path-like or None, optional
        Explicit cache directory override.
    strict_hpc : bool, default True
        When ``True``, raise :class:`ValueError` if the resolved cache path
        points under ``/fred``; when ``False``, emit a :class:`UserWarning`.
    """
    if cache_dir is not None:
        raw = Path(os.path.expanduser(str(cache_dir)))
    else:
        env_override = os.environ.get("TESS_BRIGHT_CACHE_DIR") or os.environ.get(
            "TESSBKGD_CACHE_DIR"
        )
        scratch = os.environ.get("SCRATCH")
        xdg = os.environ.get("XDG_CACHE_HOME")
        if env_override:
            raw = Path(os.path.expanduser(env_override))
        elif scratch:
            raw = Path(os.path.expanduser(scratch)) / ".cache" / "tess_bright"
        elif xdg:
            raw = Path(os.path.expanduser(xdg)) / "tess_bright"
        else:
            raw = Path.home() / ".cache" / "tess_bright"

    resolved = Path(os.path.abspath(str(raw)))
    if _is_fred_path(raw) or _is_fred_path(resolved):
        msg = (
            f"Cache directory {str(resolved)!r} points under /fred/, which "
            "violates Swinburne NT/OzSTAR project inode quotas. Route caches "
            "to $SCRATCH (/aphid/scratch-3month) or $JOBFS instead."
        )
        if strict_hpc:
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    return resolved
