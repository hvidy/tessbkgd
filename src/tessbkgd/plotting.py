"""4-panel visual diagnostics for ``tessbkgd`` background fits."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ._contract import PixelQuality, _unwrap_hdulist, read_bkgsub_state
from .masking import format_order


def _val(arr: Any) -> np.ndarray:
    return np.asarray(arr.value if hasattr(arr, "value") else arr, dtype=np.float64)


def _extract_tpf(
    tpf: Any, result: Any, is_est: bool, is_diag: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    hdul = _unwrap_hdulist(tpf)
    data = hdul[1].data
    cmap = {n.upper(): n for n in (data.names or ())}
    qm = getattr(tpf, "quality_mask", None)
    use_q = qm is not None and len(qm) == len(data) and np.any(qm)
    idx = np.flatnonzero(np.asarray(qm, dtype=bool) if use_q else np.ones(len(data)))
    idx = idx[:: max(1, idx.size // 512)]
    fl = np.asarray(data[cmap["FLUX"]][idx], dtype=np.float64)
    was_sub = read_bkgsub_state(tpf, warn_legacy_mundey=False)
    b_col = (
        np.asarray(data[cmap["FLUX_BKG"]][idx], dtype=np.float64)
        if (was_sub and "FLUX_BKG" in cmap)
        else None
    )
    u_c = fl + b_col if b_col is not None else fl
    e_c = (
        np.asarray(data[cmap["FLUX_ERR"]][idx], dtype=np.float64)
        if "FLUX_ERR" in cmap
        else None
    )
    if is_est:
        b_all = _val(result.bkg)
        b_c = b_all[:: max(1, b_all.shape[0] // 512)][: u_c.shape[0]]
        return u_c, b_c, e_c, np.asarray(result.mask, dtype=bool)
    if is_diag:
        p, ny, nx = result.bkg_basis.shape
        b_c = (result.coeffs[idx] @ result.bkg_basis.reshape(p, ny * nx)).reshape(
            -1, ny, nx
        )
        return u_c, b_c, e_c, np.asarray(result.bkg_mask, dtype=bool)
    if b_col is not None:
        if "PIXEL_QUALITY" in hdul and hdul["PIXEL_QUALITY"].data is not None:
            pq = np.asarray(hdul["PIXEL_QUALITY"].data, dtype=np.uint16)
            bm = (pq & int(PixelQuality.BKG_USED)) != 0
        else:
            bm = np.ones(u_c.shape[1:], dtype=bool)
        return u_c, b_col, e_c, bm
    raise ValueError("TPF lacks FLUX_BKG; run subtract_background() or pass result.")


def _extract_plot_arrays(
    tpf: Any, result: Any
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray | None, np.ndarray,
    np.ndarray | None, str, float, str,
]:
    is_diag = hasattr(result, "bkg_basis") and hasattr(result, "coeffs")
    is_est = hasattr(result, "bkg") and hasattr(result, "mask")
    unit_lbl = "e-/s"

    if isinstance(tpf, np.ndarray) or (
        hasattr(tpf, "unit") and not hasattr(tpf, "hdu")
    ):
        u_c, unit_lbl = _val(tpf), str(getattr(tpf, "unit", "") or "flux")
        e_c: np.ndarray | None = None
        if is_est:
            b_c, e_c = _val(result.bkg), _val(result.bkg_err)
            bm = np.asarray(result.mask, dtype=bool)
        elif is_diag:
            p, ny, nx = result.bkg_basis.shape
            b_c = (result.coeffs @ result.bkg_basis.reshape(p, ny * nx)).reshape(
                -1, ny, nx
            )
            bm = np.asarray(result.bkg_mask, dtype=bool)
        else:
            raise ValueError("Pass `result` when plotting a raw array.")
    else:
        u_c, b_c, e_c, bm = _extract_tpf(tpf, result, is_est, is_diag)

    excl = result.exclusion_mask if is_diag else None
    r_corr = float(result.target_flux_corr) if is_diag else float("nan")
    ord_lbl = format_order(result.effective_order) if is_diag else "fitted"
    return u_c, b_c, e_c, bm, excl, unit_lbl, r_corr, ord_lbl


def _lims(arr: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    fin = arr[np.isfinite(arr)]
    if not fin.size:
        return 0.0, 1.0
    v0, v1 = float(np.percentile(fin, lo)), float(np.percentile(fin, hi))
    return v0, (v1 if v1 > v0 else v0 + 1.0)


def _panel(
    fig: Any, ax: Any, img: np.ndarray, cmap: Any, v0: float, v1: float,
    aspect: str, title: str, label: str,
) -> None:
    im = ax.imshow(img, origin="lower", cmap=cmap, vmin=v0, vmax=v1, aspect=aspect)
    ax.set_title(title)
    ax.set_xlabel("Column (px)")
    ax.set_ylabel("Row (px)")
    fig.colorbar(im, ax=ax, label=label)


def plot_diagnostics(
    tpf: Any,
    result: Any = None,
    *,
    frame: int | None = None,
    figsize: tuple[float, float] = (14.0, 9.0),
    save_path: str | None = None,
) -> tuple[Any, np.ndarray]:
    """Render a 4-panel background diagnostic plot (image, model, residuals, r)."""
    import matplotlib.pyplot as plt

    u_c, b_c, e_c, bm, excl, unit_lbl, r_corr, ord_lbl = _extract_plot_arrays(
        tpf, result
    )
    ny, nx = u_c.shape[1:]
    asp = "auto" if max(ny, nx) / max(min(ny, nx), 1) > 4.0 else "equal"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
        if frame is None:
            med_u, med_b = np.nanmedian(u_c, axis=0), np.nanmedian(b_c, axis=0)
            res_img = np.nanmedian(u_c - b_c, axis=0)
            sig_img = np.nanmedian(e_c, axis=0) if e_c is not None else None
        else:
            fi = int(frame) % u_c.shape[0]
            med_u, med_b, res_img = u_c[fi], b_c[fi], u_c[fi] - b_c[fi]
            sig_img = e_c[fi] if e_c is not None else None

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    cm_v = plt.get_cmap("viridis").with_extremes(bad="0.75")
    cm_r = plt.get_cmap("RdBu_r").with_extremes(bad="0.75")

    _panel(
        fig, axes[0, 0], med_u, cm_v, *_lims(med_u, 5.0, 95.0), asp,
        f"Median flux + mask (cyan, N={int(np.count_nonzero(bm))})",
        f"Flux ({unit_lbl})",
    )
    if np.any(bm) and not np.all(bm):
        axes[0, 0].contour(bm.astype(float), levels=[0.5], colors="cyan", linewidths=1)
    if excl is not None and np.any(excl):
        axes[0, 0].contour(
            excl.astype(float), levels=[0.5], colors="red",
            linestyles="--", linewidths=1,
        )
    _panel(
        fig, axes[0, 1], med_b, cm_v, *_lims(med_b, 2.0, 98.0), asp,
        f"Fitted background model (order={ord_lbl})", f"Background ({unit_lbl})",
    )

    has_s = sig_img is not None and np.any(np.isfinite(sig_img) & (sig_img > 0.0))
    if has_s:
        assert sig_img is not None
        ok_s = np.isfinite(sig_img) & (sig_img > 0.0)
        z_img, z_lbl = np.where(ok_s, res_img / sig_img, np.nan), "Residual z-score"
    else:
        z_img, z_lbl = res_img, f"Signed residual (unnormalized, {unit_lbl})"
    z_bg = z_img[bm & np.isfinite(z_img)]
    vz = max(
        3.0 if has_s else 1e-6,
        float(np.percentile(np.abs(z_bg), 98.0)) if z_bg.size else 5.0,
    )
    _panel(
        fig, axes[1, 0], z_img, cm_r, -vz, vz, asp,
        "Residuals (grey=NaN/masked, contour=~bkg_mask)", z_lbl,
    )
    if np.any(~bm):
        axes[1, 0].contour(
            (~bm).astype(float), levels=[0.5], colors="k", linewidths=0.8
        )

    b_ts, sub = np.nanmedian(b_c, axis=(1, 2)), u_c - b_c
    f_ts = (
        np.nansum(sub[:, ~bm], axis=1) if np.any(~bm) else np.nansum(sub, axis=(1, 2))
    )
    if not np.isfinite(r_corr):
        fin = np.isfinite(b_ts) & np.isfinite(f_ts)
        if (
            int(np.count_nonzero(fin)) >= 6
            and np.std(b_ts[fin]) > 1e-12
            and np.std(f_ts[fin]) > 1e-12
        ):
            r_corr = float(np.corrcoef(b_ts[fin], f_ts[fin])[0, 1])

    ax_ts, cx = axes[1, 1], np.arange(b_ts.size)
    l1 = ax_ts.plot(cx, b_ts, color="tab:blue", lw=1.2, label="Median B_hat")
    ax_ts.set_xlabel("Sampled cadence index")
    ax_ts.set_ylabel(f"Median background ({unit_lbl})", color="tab:blue")
    ax_ts.tick_params(axis="y", labelcolor="tab:blue")
    ax_r = ax_ts.twinx()
    l2 = ax_r.plot(cx, f_ts, color="tab:orange", lw=1.0, alpha=0.8, label="Target flux")
    ax_r.set_ylabel(f"Target aperture flux ({unit_lbl})", color="tab:orange")
    ax_r.tick_params(axis="y", labelcolor="tab:orange")
    r_str = f"r = {r_corr:+.3f}" if np.isfinite(r_corr) else "r = N/A"
    ax_ts.set_title(f"Background vs target time series ({r_str})")
    ax_ts.legend(l1 + l2, [ln.get_label() for ln in l1 + l2], loc="best", frameon=False)
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig, axes
