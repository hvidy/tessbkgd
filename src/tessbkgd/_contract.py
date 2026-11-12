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
from collections.abc import MutableMapping
from enum import IntFlag
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

HALOCVER: int = 1

# HALOCVER = 1 (§5.2) heuristic constants for detecting legacy mundey (< 2.0)
# TPFs whose unsubtracted FLUX retained stale SPOC FLUX_BKG > 0:
_DARK_CADENCE_QUANTILE = 10.0
_DARK_CADENCE_FRACTION = 0.25
_FAINT_PIXEL_QUANTILE = 20.0
_LEGACY_MUNDEY_BKG_RATIO = 0.5
_LEGACY_MUNDEY_SLOPE_MIN = 0.5
_MIN_TEMPORAL_BKG_REL_STD = 1e-3


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


def _sync_tpf_meta(tpf_or_hdu: Any, hdulist: fits.HDUList) -> None:
    """Synchronize ``TargetPixelFile.meta`` with ``hdulist[0].header``."""
    if not (hasattr(tpf_or_hdu, "meta") and hasattr(tpf_or_hdu, "hdu")):
        return
    meta = tpf_or_hdu.meta
    if isinstance(meta, MutableMapping):
        meta.update(hdulist[0].header)
    elif type(meta).__name__ == "HduToMetaMapping":
        tpf_or_hdu.meta = type(meta)(hdulist[0])


def _set_bkgsub_header(tpf_or_hdu: Any, bkgsub: bool) -> None:
    """Set ``BKGSUB`` consistently on ``hdu[0]``, ``hdu[1]``, and ``tpf.meta``."""
    hdulist = _unwrap_hdulist(tpf_or_hdu)
    comment = (
        "Sky background subtracted from FLUX"
        if bkgsub
        else "Sky background not subtracted from FLUX"
    )
    for ext in (0, 1):
        if ext < len(hdulist):
            hdulist[ext].header["BKGSUB"] = (bool(bkgsub), comment)
    _sync_tpf_meta(tpf_or_hdu, hdulist)


def read_bkgsub_state(tpf: Any) -> bool:
    """Determine whether ``tpf`` currently stores background-subtracted ``FLUX``.

    Follows the ``HALOCVER = 1`` provenance rules:

    1. If ``BKGSUB`` is explicitly present as a boolean in ``hdu[0].header`` or
       ``hdu[1].header``, return its value directly (raising :class:`ValueError`
       if both headers define boolean ``BKGSUB`` cards that disagree).
    2. When ``BKGSUB`` is absent, inspect ``FLUX_BKG``: if absent, all-zero, or
       all-``NaN`` (as on ``TessCut`` FFI cutouts), return ``False``.
    3. When ``BKGSUB`` is absent and ``FLUX_BKG`` contains finite non-zero
       values (standard SPOC / TESS-SPOC HLSP TPFs), emit a defensive
       :class:`UserWarning` if the dark-cadence 10th-percentile of unflagged
       ``FLUX`` exceeds ``0.5 * nanmedian(FLUX_BKG)`` *and* faint-pixel
       ``FLUX(t)`` co-varies in time with ``FLUX_BKG(t)`` (catching legacy
       ``mundey`` v0.1 TPFs without false-triggering on static bright-star halos
       in genuine SPOC TPFs), and return ``True``.
    """
    hdulist = _unwrap_hdulist(tpf)

    bkgsub_cards: dict[int, bool] = {}
    for ext in (0, 1):
        if ext < len(hdulist) and "BKGSUB" in hdulist[ext].header:
            card_val = hdulist[ext].header["BKGSUB"]
            if isinstance(card_val, (bool, np.bool_)):
                bkgsub_cards[ext] = bool(card_val)

    if len(bkgsub_cards) == 2 and bkgsub_cards[0] != bkgsub_cards[1]:
        raise ValueError(
            f"Conflicting boolean BKGSUB header cards between hdu[0] "
            f"({bkgsub_cards[0]}) and hdu[1] ({bkgsub_cards[1]})."
        )
    if bkgsub_cards:
        return next(iter(bkgsub_cards.values()))

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
                | PixelQuality.SOURCE_MASK
            )
            bad_px = (pq & excl) != 0
            if not np.all(bad_px):
                flux_spatial = flux_arr[:, ~bad_px]
                bkg_spatial = bkg_arr[:, ~bad_px]
            else:
                flux_spatial = flux_arr.reshape(flux_arr.shape[0], -1)
                bkg_spatial = bkg_arr.reshape(bkg_arr.shape[0], -1)
        else:
            flux_spatial = flux_arr.reshape(flux_arr.shape[0], -1)
            bkg_spatial = bkg_arr.reshape(bkg_arr.shape[0], -1)

        valid_cad = np.any(np.isfinite(flux_spatial), axis=1) & np.any(
            np.isfinite(bkg_spatial), axis=1
        )
        if np.any(valid_cad):
            flux_valid = flux_spatial[valid_cad]
            bkg_valid = bkg_spatial[valid_cad]
            cad_q10 = np.nanpercentile(
                flux_valid, _DARK_CADENCE_QUANTILE, axis=1
            )
            n_dark = max(
                1, int(np.ceil(_DARK_CADENCE_FRACTION * cad_q10.size))
            )
            dark_idx = np.argpartition(cad_q10, n_dark - 1)[:n_dark]
            dark_q10 = float(
                np.nanpercentile(flux_valid[dark_idx], _DARK_CADENCE_QUANTILE)
            )
            if (
                np.isfinite(dark_q10)
                and dark_q10 > _LEGACY_MUNDEY_BKG_RATIO * med_bkg
            ):
                # Separate unsubtracted sky (faint-pixel FLUX co-varying ~1:1
                # with FLUX_BKG) from static bright-star halos on genuine SPOC
                # TPFs (where FLUX_BKG is already subtracted, so slope ~ 0).
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="All-NaN slice encountered",
                        category=RuntimeWarning,
                    )
                    pix_dark_med = np.nanmedian(flux_valid[dark_idx], axis=0)
                    fin_px = np.isfinite(pix_dark_med)
                    co_varies = True
                    if np.any(fin_px):
                        faint_cut = float(
                            np.percentile(
                                pix_dark_med[fin_px], _FAINT_PIXEL_QUANTILE
                            )
                        )
                        faint_px = fin_px & (pix_dark_med <= faint_cut)
                        f_ts = np.nanmedian(flux_valid[:, faint_px], axis=1)
                        b_ts = np.nanmedian(bkg_valid[:, faint_px], axis=1)
                        ok_ts = np.isfinite(f_ts) & np.isfinite(b_ts)
                        if np.count_nonzero(ok_ts) >= 3:
                            b_c = b_ts[ok_ts] - np.mean(b_ts[ok_ts])
                            b_std = float(np.std(b_c))
                            if b_std > _MIN_TEMPORAL_BKG_REL_STD * med_bkg:
                                f_c = f_ts[ok_ts] - np.mean(f_ts[ok_ts])
                                slope = float(
                                    np.dot(f_c, b_c) / np.dot(b_c, b_c)
                                )
                                co_varies = slope > _LEGACY_MUNDEY_SLOPE_MIN

                if co_varies:
                    warnings.warn(
                        "BKGSUB header card is absent while FLUX_BKG is non-zero "
                        f"(nanmedian={med_bkg:.3g}), dark-cadence 10th-percentile "
                        f"FLUX ({dark_q10:.3g}) exceeds 0.5 * nanmedian(FLUX_BKG), "
                        "and faint-pixel FLUX co-varies in time with FLUX_BKG. "
                        "If this TPF was recalibrated by legacy mundey (< 2.0) "
                        "without zeroing FLUX_BKG, set hdu[0].header['BKGSUB'] = "
                        "False and hdu[1].header['BKGSUB'] = False to avoid "
                        "double-adding background.",
                        UserWarning,
                        stacklevel=2,
                    )

    return True


def _infer_n_basis_from_tril(n_tril: int) -> int:
    if n_tril < 1:
        raise ValueError(
            f"Packed Cholesky width must be >= 1, got {n_tril}."
        )
    disc = 1 + 8 * n_tril
    root = math.isqrt(disc)
    if root * root != disc or (root - 1) % 2 != 0:
        raise ValueError(
            f"Packed width {n_tril} is not a valid triangular number "
            "P*(P+1)//2 for any P >= 1."
        )
    return (root - 1) // 2


def pack_bkg_chol(bkg_chol: np.ndarray) -> np.ndarray:
    """Pack lower-triangular Cholesky factors ``(N, P, P)`` into ``(N, P*(P+1)//2)``."""
    arr = np.asarray(bkg_chol, dtype=np.float32)
    if arr.ndim == 3:
        _, p_rows, p_cols = arr.shape
        if p_rows < 1 or p_rows != p_cols:
            raise ValueError(
                "3D bkg_chol must have square trailing dimensions with P >= 1, "
                f"got {arr.shape}."
            )
        if p_rows > 1:
            non_all_nan = ~np.all(np.isnan(arr), axis=(1, 2))
            if np.any(non_all_nan):
                u_rows, u_cols = np.triu_indices(p_rows, k=1)
                upper = arr[non_all_nan][:, u_rows, u_cols]
                if not np.all(upper == 0.0):
                    raise ValueError(
                        "3D bkg_chol must be lower-triangular with zero strict "
                        "upper triangle (pass lower=True when computing Cholesky "
                        "factors)."
                    )
        rows, cols = np.tril_indices(p_rows)
        return np.ascontiguousarray(arr[:, rows, cols], dtype=np.float32)
    if arr.ndim == 2:
        _infer_n_basis_from_tril(arr.shape[1])
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
    if n_tril < 1:
        raise ValueError(f"Packed bkg_chol width must be >= 1, got {n_tril}.")
    if n_basis is None:
        n_basis = _infer_n_basis_from_tril(n_tril)
    elif n_basis < 1 or n_basis * (n_basis + 1) // 2 != n_tril:
        raise ValueError(
            f"Packed width {n_tril} does not match P*(P+1)//2 for P={n_basis} "
            "(requires P >= 1)."
        )
    out = np.zeros((n_cad, n_basis, n_basis), dtype=arr.dtype)
    rows, cols = np.tril_indices(n_basis)
    out[:, rows, cols] = arr
    return out


def _flux_column_unit_str(pixels_hdu: fits.BinTableHDU) -> str | None:
    if pixels_hdu.columns is not None:
        for idx, col in enumerate(pixels_hdu.columns, start=1):
            if col.name.upper() == "FLUX":
                if col.unit:
                    return str(col.unit)
                tunit = pixels_hdu.header.get(f"TUNIT{idx}")
                return str(tunit) if tunit else None
    return None


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
        Static 2D background basis evaluated on the TPF pixel grid (``P >= 1``).
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
        raise ValueError(
            f"bkg_used_mask must be 2D (ny, nx), got shape {used.shape}."
        )
    ny, nx = used.shape

    basis_f32 = np.ascontiguousarray(bkg_basis, dtype=np.float32)
    if basis_f32.ndim != 3 or basis_f32.shape[1:] != (ny, nx):
        raise ValueError(
            f"bkg_basis must have shape (P, {ny}, {nx}), got {basis_f32.shape}."
        )
    n_basis = basis_f32.shape[0]
    if n_basis < 1:
        raise ValueError(f"bkg_basis must have P >= 1 basis modes, got P={n_basis}.")
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

    # Write packed BKG_CHOL into hdu[1] (PIXELS BinTableHDU) for lockstep slicing,
    # inheriting the FLUX column unit (e.g. 'e-/s' on SPOC/mundey, None on TessCut).
    target_chol_unit = _flux_column_unit_str(hdulist[1])
    existing_col_map = (
        {c.name.upper(): c for c in hdulist[1].columns}
        if hdulist[1].columns is not None
        else {}
    )
    existing_chol_col = existing_col_map.get("BKG_CHOL")
    existing_unit = (
        str(existing_chol_col.unit)
        if (existing_chol_col is not None and existing_chol_col.unit)
        else None
    )
    if (
        existing_chol_col is not None
        and existing_unit == target_chol_unit
        and (
            hdulist[1].data["BKG_CHOL"].shape == (n_raw, n_tril)
            or (n_tril == 1 and hdulist[1].data["BKG_CHOL"].shape == (n_raw,))
        )
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
            unit=target_chol_unit,
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

    _sync_tpf_meta(hdu, hdulist)
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
    Symlinks (such as ``~/.cache -> /fred/<project>/...``) are resolved prior
    to enforcing the ``/fred/`` check.

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

    resolved = raw.resolve(strict=False)
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
