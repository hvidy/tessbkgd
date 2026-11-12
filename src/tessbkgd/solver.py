"""QR-equilibrated 2D Chebyshev basis and batched JAX Huber-Tukey IRLS solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_triangular

from ._precision import in_float64
from .masking import n_basis_for_order, raw_chebyshev_2d

_HUBER_C = 1.345
_TUKEY_C = 4.685
_MAD_TO_SIGMA = 1.482602218505602
_P_STATIC = 10
_SOLVE_CHUNK = 8192
_RECON_CHUNK = 8192
_VAR_SUBCHUNK = 4096
_BATCH_BUCKETS = (128, 2048, 8192)
_PIXEL_BUCKETS = (256, 2048, 4096)


@dataclass(frozen=True)
class SolverOutput:
    """Per-cadence Chebyshev fit coefficients, sandwich Cholesky, and diagnostics."""

    coeffs: np.ndarray
    bkg_chol: np.ndarray
    tau_scale: np.ndarray
    n_valid_pixels: np.ndarray
    n_tukey_inliers: np.ndarray
    used_huber_fallback: np.ndarray


def equilibrate_chebyshev_basis(
    ny: int,
    nx: int,
    order: int,
    bkg_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct QR-equilibrated 2D Chebyshev basis ``Phi_full`` on ``bkg_mask``.

    Computes ``Phi_tilde_bg = Q_bg @ R_bg`` with strictly positive diagonal on
    ``R_bg`` and forms ``Phi_full = Phi_tilde_full @ inv(R_bg)`` via triangular
    solve so ``Phi_bg == Q_bg`` satisfies ``kappa(Phi_bg) == 1``.

    Returns
    -------
    phi_full : np.ndarray, shape ``(P, ny, nx)``
        QR-equilibrated basis evaluated over the full pixel grid (``float64``).
    phi_bg : np.ndarray, shape ``(M, P)``
        Orthonormal design matrix on ``bkg_mask`` (``Phi_bg.T @ Phi_bg = I_P``).
    r_bg : np.ndarray, shape ``(P, P)``
        Upper-triangular QR factor with strictly positive diagonal.
    """
    mask_bool = np.asarray(bkg_mask, dtype=bool)
    if mask_bool.shape != (ny, nx):
        raise ValueError(
            f"bkg_mask shape {mask_bool.shape} does not match ({ny}, {nx})."
        )
    n_params = n_basis_for_order(order)
    n_bg = int(np.count_nonzero(mask_bool))
    if n_bg < n_params:
        raise ValueError(
            f"Insufficient background pixels ({n_bg}) for P={n_params} basis modes."
        )

    phi_tilde = raw_chebyshev_2d(ny, nx, order)  # (P, ny, nx)
    phi_tilde_bg = phi_tilde[:, mask_bool].T  # (M, P)

    q_bg, r_bg = np.linalg.qr(phi_tilde_bg, mode="reduced")
    diag_r = np.diag(r_bg)
    max_diag = float(np.max(np.abs(diag_r))) if diag_r.size > 0 else 0.0
    if max_diag <= 0.0 or np.min(np.abs(diag_r)) <= 1e-12 * max_diag:
        raise ValueError(
            "Background design matrix is numerically rank-deficient during QR "
            f"equilibration (min|diag(R)|={float(np.min(np.abs(diag_r))):.3e})."
        )

    signs = np.where(diag_r >= 0.0, 1.0, -1.0)
    q_bg = q_bg * signs[np.newaxis, :]
    r_bg = signs[:, np.newaxis] * r_bg

    # Phi_full_flat = Phi_tilde_flat @ R_bg^{-1} via triangular solve:
    phi_tilde_flat = phi_tilde.reshape(n_params, ny * nx)  # (P, N_pix)
    phi_full_flat = solve_triangular(
        r_bg.T, phi_tilde_flat, lower=True, check_finite=False
    )
    phi_full = np.ascontiguousarray(
        phi_full_flat.reshape(n_params, ny, nx), dtype=np.float64
    )
    phi_bg = np.ascontiguousarray(q_bg, dtype=np.float64)
    return phi_full, phi_bg, r_bg


def _masked_median_2d(vals: jax.Array, mask: jax.Array) -> jax.Array:
    """Compute exact row-wise median of ``vals`` over ``mask`` without NaNs."""
    n = jnp.sum(mask, axis=1, dtype=jnp.int32)
    safe_vals = jnp.where(mask, vals, 0.0)
    filled = jnp.where(mask, safe_vals, jnp.inf)
    s = jnp.sort(filled, axis=1)
    max_idx = vals.shape[1] - 1
    lo = jnp.clip((n - 1) // 2, 0, max_idx)
    hi = jnp.clip(n // 2, 0, max_idx)
    b_idx = jnp.arange(vals.shape[0])
    med = 0.5 * (s[b_idx, lo] + s[b_idx, hi])
    return jnp.where(n > 0, med, 0.0)


def _adaptive_scale_2d(
    resid: jax.Array,
    sig: jax.Array,
    valid_px: jax.Array,
    has_flux_err: jax.Array,
) -> jax.Array:
    """Compute batched per-iteration adaptive MAD scale ``tau_t^(k)``."""
    z = resid / sig
    med_z = _masked_median_2d(z, valid_px)
    abs_dev = jnp.where(valid_px, jnp.abs(z - med_z[:, None]), 0.0)
    mad_sigma = _MAD_TO_SIGMA * _masked_median_2d(abs_dev, valid_px)
    floor = jnp.where(has_flux_err, 1.0, 1e-6)
    return jnp.maximum(floor, mad_sigma)


def _chol_factor_checked_1d(
    gram: jax.Array, active_p: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Compute lower Cholesky factor and check full-rank SPD on ``active_p``."""
    p_pad = gram.shape[0]
    gram_sym = 0.5 * (gram + gram.T)
    chol_raw = jnp.linalg.cholesky(gram_sym)
    diag_l = jnp.diag(chol_raw)
    finite_l = jnp.all(jnp.isfinite(chol_raw))
    act_diag = jnp.where(active_p & jnp.isfinite(diag_l), diag_l, 0.0)
    max_d = jnp.max(act_diag)
    min_d = jnp.min(
        jnp.where(active_p & jnp.isfinite(diag_l), diag_l, jnp.inf)
    )
    chol_ok = finite_l & (max_d > 0.0) & (min_d > 1e-6 * max_d)
    eye_p = jnp.eye(p_pad, dtype=gram.dtype)
    chol_safe = jnp.where(chol_ok, chol_raw, eye_p)
    return chol_safe, chol_ok


def _safe_spd_cho_solve_1d(
    gram: jax.Array,
    rhs: jax.Array,
    fallback_beta: jax.Array,
    active_p: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Solve ``gram @ beta = rhs`` via Cholesky if numerically full-rank SPD."""
    chol_l, chol_ok = _chol_factor_checked_1d(gram, active_p)
    sol = jax.scipy.linalg.cho_solve((chol_l, True), rhs)
    sol_ok = chol_ok & jnp.all(jnp.isfinite(sol))
    return jnp.where(sol_ok, sol, fallback_beta), sol_ok


def _inverse_lower_cholesky_1d(
    gram: jax.Array, active_p: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Compute lower-triangular ``L_base`` with ``L_base @ L_base.T == inv(gram)``."""
    p_pad = gram.shape[0]
    c_rev, ok = _chol_factor_checked_1d(gram[::-1, ::-1], active_p[::-1])
    eye_p = jnp.eye(p_pad, dtype=gram.dtype)
    z_inv = jax.scipy.linalg.solve_triangular(c_rev, eye_p, lower=True)
    l_base = jnp.tril(z_inv.T[::-1, ::-1])
    ok_all = ok & jnp.all(jnp.isfinite(l_base))
    return jnp.where(ok_all, l_base, eye_p), ok_all


_batched_cho_solve = jax.vmap(
    _safe_spd_cho_solve_1d, in_axes=(0, 0, 0, None)
)
_batched_inv_lower_chol = jax.vmap(
    _inverse_lower_cholesky_1d, in_axes=(0, None)
)


@jax.jit
def _solve_chunk_jit(
    u_chunk: jax.Array,
    sigma_chunk: jax.Array,
    phi_bg: jax.Array,
    p_actual: jax.Array,
    has_flux_err: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Solve a cadence batch via WLS + 4 Huber + 1 guarded Tukey + sandwich cov."""
    b_dim, m_dim = u_chunk.shape
    p_pad = phi_bg.shape[1]
    dtype = u_chunk.dtype
    p_act = p_actual.astype(jnp.int32)
    active_p = jnp.arange(p_pad, dtype=jnp.int32) < p_act
    inactive_eye = jnp.diag((~active_p).astype(dtype))[None, :, :]
    phi_outer = (phi_bg[:, :, None] * phi_bg[:, None, :]).reshape(
        m_dim, p_pad * p_pad
    )

    valid_err = jnp.isfinite(sigma_chunk) & (sigma_chunk > 0.0)
    valid_px = jnp.isfinite(u_chunk) & jnp.where(has_flux_err, valid_err, True)
    sig = jnp.where(has_flux_err & valid_px, sigma_chunk, 1.0)
    w0 = jnp.where(valid_px, 1.0 / (sig * sig), 0.0)

    n_valid = jnp.sum(valid_px, axis=1, dtype=jnp.int32)
    y = jnp.where(valid_px, u_chunk, 0.0)
    min_tukey_inliers = jnp.maximum(2 * p_act, p_act + 1)

    init_carry = (
        jnp.zeros((b_dim, p_pad), dtype=dtype),
        jnp.zeros((b_dim, p_pad), dtype=dtype),
        jnp.zeros((b_dim,), dtype=bool),
        jnp.zeros((b_dim,), dtype=jnp.int32),
        jnp.zeros((b_dim,), dtype=bool),
    )

    # Unified outer loop: step 0 = WLS, steps 1..4 = Huber, step 5 = guarded Tukey.
    def _irls_step(
        step: jax.Array,
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        beta_curr, beta_hub, cad_ok_c, m_step_c, tukey_ok_c = carry
        resid = y - beta_curr @ phi_bg.T
        tau = _adaptive_scale_2d(resid, sig, valid_px, has_flux_err)
        u_std = resid / (tau[:, None] * sig)

        w_hub = w0 * jnp.minimum(
            1.0, _HUBER_C / jnp.maximum(jnp.abs(u_std), 1e-30)
        )
        in_step = valid_px & (jnp.abs(u_std) < _TUKEY_C)
        q_step = jnp.where(in_step, (u_std / _TUKEY_C) ** 2, 1.0)
        w_tuk = w0 * ((1.0 - q_step) ** 2)

        w = jnp.where(step == 0, w0, jnp.where(step < 5, w_hub, w_tuk))
        gram = (w @ phi_outer).reshape(b_dim, p_pad, p_pad) + inactive_eye
        rhs = (w * y) @ phi_bg
        sol, sol_ok = _batched_cho_solve(gram, rhs, beta_curr, active_p)

        is_s0 = step == 0
        is_s4 = step == 4
        is_s5 = step == 5

        cad_ok_next = jnp.where(is_s0, (n_valid >= p_act) & sol_ok, cad_ok_c)
        beta_hub_next = jnp.where(is_s4, sol, beta_hub)
        m_step_now = jnp.sum(in_step, axis=1, dtype=jnp.int32)
        m_step_next = jnp.where(is_s5, m_step_now, m_step_c)
        t_ok_now = (
            cad_ok_next & (m_step_now >= min_tukey_inliers) & sol_ok
        )
        tukey_ok_next = jnp.where(is_s5, t_ok_now, tukey_ok_c)
        beta_tuk_guarded = jnp.where(t_ok_now[:, None], sol, beta_hub)
        beta_next = jnp.where(is_s5, beta_tuk_guarded, sol)
        return (
            beta_next,
            beta_hub_next,
            cad_ok_next,
            m_step_next,
            tukey_ok_next,
        )

    beta_hat, beta_hub, cad_ok, m_inliers_step, tukey_ok = jax.lax.fori_loop(
        0, 6, _irls_step, init_carry
    )

    # Huber-Street-Carroll-Ruppert sandwich covariance:
    # - Evaluate Tukey residuals & active inlier bread G_T at beta_hat.
    # - Evaluate Huber residuals & active quadratic-core bread G_H (|u_h| <= c_H)
    #   at beta_hub so flat-tail outliers (psi_H'(u) == 0) contribute zero curvature.
    resid_t = y - beta_hat @ phi_bg.T
    tau_t = _adaptive_scale_2d(resid_t, sig, valid_px, has_flux_err)
    u_t = resid_t / (tau_t[:, None] * sig)

    in_t = valid_px & (jnp.abs(u_t) < _TUKEY_C)
    m_t = jnp.sum(in_t, axis=1, dtype=jnp.int32)
    q_t = jnp.where(in_t, (u_t / _TUKEY_C) ** 2, 1.0)
    psi_t = jnp.where(in_t, u_t * ((1.0 - q_t) ** 2), 0.0)
    psip_t = jnp.where(in_t, (1.0 - q_t) * (1.0 - 5.0 * q_t), 0.0)

    resid_h = y - beta_hub @ phi_bg.T
    tau_h = _adaptive_scale_2d(resid_h, sig, valid_px, has_flux_err)
    u_h = resid_h / (tau_h[:, None] * sig)

    in_h = valid_px & (jnp.abs(u_h) <= _HUBER_C)
    m_h = jnp.sum(in_h, axis=1, dtype=jnp.int32)
    psi_h = jnp.where(valid_px, jnp.clip(u_h, -_HUBER_C, _HUBER_C), 0.0)

    w_in_t = jnp.where(in_t, w0, 0.0)
    w_in_h = jnp.where(in_h, w0, 0.0)
    gram_in_t = (w_in_t @ phi_outer).reshape(b_dim, p_pad, p_pad) + inactive_eye
    gram_in_h = (w_in_h @ phi_outer).reshape(b_dim, p_pad, p_pad) + inactive_eye

    # Invert active bread matrices [G_T, G_H] in a single batched Cholesky pass.
    grams_pair = jnp.concatenate([gram_in_t, gram_in_h], axis=0)
    l_pair, ok_pair = _batched_inv_lower_chol(grams_pair, active_p)
    l_base_t, l_base_h = l_pair[:b_dim], l_pair[b_dim:]
    bread_t_ok, bread_h_ok = ok_pair[:b_dim], ok_pair[b_dim:]

    tukey_bread_ok = (
        tukey_ok
        & (m_t >= min_tukey_inliers)
        & bread_t_ok
        & (
            jnp.sum(psip_t, axis=1)
            > 0.05 * jnp.maximum(m_t.astype(dtype), 1.0)
        )
    )
    huber_core_ok = cad_ok & (m_h >= p_act) & bread_h_ok
    cad_final_ok = jnp.where(tukey_bread_ok, cad_ok, huber_core_ok)

    beta_final = jnp.where(tukey_bread_ok[:, None], beta_hat, beta_hub)
    tau_final = jnp.where(tukey_bread_ok, tau_t, tau_h)
    l_base = jnp.where(tukey_bread_ok[:, None, None], l_base_t, l_base_h)

    exact_interp_alpha = jnp.where(has_flux_err, 1.0, jnp.nan)

    # Tukey SCR scalar factor on M_I inliers:
    m_t_f = jnp.maximum(m_t.astype(dtype), 1.0)
    df_t = m_t - p_act
    mean_psi2_t = jnp.sum(psi_t * psi_t, axis=1) / m_t_f
    mean_psip_t = jnp.maximum(jnp.sum(psip_t, axis=1) / m_t_f, 0.05)
    alpha_t_scr = (
        (tau_t * tau_t)
        * (m_t_f / jnp.maximum(df_t.astype(dtype), 1.0))
        * (mean_psi2_t / (mean_psip_t * mean_psip_t))
    )
    alpha_t = jnp.where(df_t > 0, alpha_t_scr, exact_interp_alpha)

    # Huber active-Hessian sandwich scalar factor:
    # Since G_H^{-1} already incorporates one factor of 1 / mean(psi_H'),
    # alpha_H = tau_h^2 * (N / (N - P)) * (mean(psi_H^2) / mean(psi_H'))
    #         = tau_h^2 * (N / (N - P)) * (sum(psi_H^2) / M_H).
    n_val_f = jnp.maximum(n_valid.astype(dtype), 1.0)
    m_h_f = jnp.maximum(m_h.astype(dtype), 1.0)
    df_h = n_valid - p_act
    alpha_h_scr = (
        (tau_h * tau_h)
        * (n_val_f / jnp.maximum(df_h.astype(dtype), 1.0))
        * (jnp.sum(psi_h * psi_h, axis=1) / m_h_f)
    )
    alpha_h = jnp.where(df_h > 0, alpha_h_scr, exact_interp_alpha)

    alpha = jnp.where(tukey_bread_ok, alpha_t, alpha_h)
    scale_sqrt = jnp.sqrt(jnp.maximum(alpha, 0.0))
    l_chol = scale_sqrt[:, None, None] * l_base

    nan_beta = jnp.full((b_dim, p_pad), jnp.nan, dtype=dtype)
    nan_chol = jnp.full((b_dim, p_pad, p_pad), jnp.nan, dtype=dtype)

    beta_out = jnp.where(cad_final_ok[:, None], beta_final, nan_beta)
    chol_valid = cad_final_ok & jnp.isfinite(alpha)
    chol_out = jnp.where(chol_valid[:, None, None], l_chol, nan_chol)
    tau_out = jnp.where(cad_final_ok, tau_final, jnp.nan)
    m_inliers_out = jnp.where(tukey_bread_ok, m_t, m_inliers_step)
    used_fallback = cad_final_ok & (~tukey_bread_ok)

    return (
        beta_out,
        chol_out,
        tau_out,
        n_valid,
        m_inliers_out,
        used_fallback,
    )


@in_float64
def _run_chunk_float64(
    u_chunk: np.ndarray,
    sigma_chunk: np.ndarray,
    phi_bg: np.ndarray,
    p_actual: np.ndarray,
    has_flux_err: np.ndarray,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    return jax.device_get(
        _solve_chunk_jit(
            u_chunk, sigma_chunk, phi_bg, p_actual, has_flux_err
        )
    )


def _bucket_pixel_dim(n_pix: int) -> int:
    """Bucket background pixel count ``M`` to reuse XLA kernels across cutouts."""
    for b in _PIXEL_BUCKETS:
        if n_pix <= b:
            return b
    return n_pix


def _bucket_batch_dim(m_len: int, chunk_size: int) -> int:
    """Bucket cadence batch count ``N`` to reuse XLA kernels across calls."""
    for b in _BATCH_BUCKETS:
        if m_len <= b <= chunk_size:
            return b
    return m_len


def solve_chebyshev_background(
    u_bg: np.ndarray,
    phi_bg: np.ndarray,
    *,
    sigma_u_bg: np.ndarray | None = None,
    chunk_size: int = _SOLVE_CHUNK,
) -> SolverOutput:
    """Solve batched Huber-Tukey 2D Chebyshev IRLS across all cadences in float64."""
    u_arr = np.asarray(u_bg, dtype=np.float64)
    phi_arr = np.asarray(phi_bg, dtype=np.float64)
    if u_arr.ndim != 2 or phi_arr.ndim != 2 or u_arr.shape[1] != phi_arr.shape[0]:
        raise ValueError(
            f"Incompatible shapes u_bg={u_arr.shape} and phi_bg={phi_arr.shape}."
        )

    n_cad, n_pix = u_arr.shape
    p_dim = phi_arr.shape[1]
    if p_dim < 1:
        raise ValueError(f"phi_bg must have P >= 1 basis columns, got {p_dim}.")

    has_flux_err = False
    if sigma_u_bg is not None:
        sig_arr = np.asarray(sigma_u_bg, dtype=np.float64)
        if sig_arr.shape != u_arr.shape:
            raise ValueError(
                f"sigma_u_bg shape {sig_arr.shape} must match u_bg {u_arr.shape}."
            )
        has_flux_err = bool(
            np.any(np.isfinite(u_arr) & np.isfinite(sig_arr) & (sig_arr > 0.0))
        )
    else:
        sig_arr = np.ones((1, n_pix), dtype=np.float64)

    coeffs = np.full((n_cad, p_dim), np.nan, dtype=np.float64)
    bkg_chol = np.full((n_cad, p_dim, p_dim), np.nan, dtype=np.float64)
    tau_scale = np.full(n_cad, np.nan, dtype=np.float64)
    n_valid = np.zeros(n_cad, dtype=np.int32)
    n_tukey = np.zeros(n_cad, dtype=np.int32)
    fallback = np.zeros(n_cad, dtype=bool)

    if n_cad == 0:
        return SolverOutput(
            coeffs=coeffs,
            bkg_chol=bkg_chol,
            tau_scale=tau_scale,
            n_valid_pixels=n_valid,
            n_tukey_inliers=n_tukey,
            used_huber_fallback=fallback,
        )

    m_pad = _bucket_pixel_dim(n_pix)
    p_pad = max(_P_STATIC, p_dim)
    if m_pad > n_pix or p_pad > p_dim:
        phi_pad = np.zeros((m_pad, p_pad), dtype=np.float64)
        phi_pad[:n_pix, :p_dim] = phi_arr
    else:
        phi_pad = phi_arr

    p_actual_arr = np.asarray(p_dim, dtype=np.int32)
    has_err_arr = np.asarray(has_flux_err, dtype=bool)

    for start in range(0, n_cad, chunk_size):
        stop = min(start + chunk_size, n_cad)
        m_len = stop - start
        b_pad = (
            chunk_size
            if n_cad > chunk_size
            else _bucket_batch_dim(m_len, chunk_size)
        )

        if b_pad == m_len and m_pad == n_pix:
            u_in = u_arr[start:stop]
            s_in = (
                sig_arr[start:stop]
                if has_flux_err
                else np.ones((m_len, n_pix), dtype=np.float64)
            )
        else:
            u_in = np.full((b_pad, m_pad), np.nan, dtype=np.float64)
            u_in[:m_len, :n_pix] = u_arr[start:stop]
            s_in = np.ones((b_pad, m_pad), dtype=np.float64)
            if has_flux_err:
                s_in[:m_len, :n_pix] = sig_arr[start:stop]

        c_np, l_np, t_np, nv_np, nt_np, fb_np = _run_chunk_float64(
            u_in, s_in, phi_pad, p_actual_arr, has_err_arr
        )
        coeffs[start:stop] = c_np[:m_len, :p_dim]
        bkg_chol[start:stop] = l_np[:m_len, :p_dim, :p_dim]
        tau_scale[start:stop] = t_np[:m_len]
        n_valid[start:stop] = nv_np[:m_len]
        n_tukey[start:stop] = nt_np[:m_len]
        fallback[start:stop] = fb_np[:m_len]

    return SolverOutput(
        coeffs=coeffs,
        bkg_chol=bkg_chol,
        tau_scale=tau_scale,
        n_valid_pixels=n_valid,
        n_tukey_inliers=n_tukey,
        used_huber_fallback=fallback,
    )


def reconstruct_background_chunked(
    coeffs: np.ndarray,
    bkg_chol: np.ndarray,
    phi_full: np.ndarray,
    *,
    chunk_size: int = _RECON_CHUNK,
    out_dtype: Any = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct ``B_hat`` and marginal ``sigma_B_hat`` in <= 8,192-cadence chunks.

    Evaluates ``B_hat[t] = Phi_full @ beta_hat[t]`` via 8,192-cadence GEMM chunks
    (``< 270 MB`` temporary workspace on 64x64 cutouts) and accumulates exact
    cancellation-free marginal variance ``((Phi_full @ L_t)**2 @ 1_P)`` without
    materializing 3D ``(chunk, ny*nx, P)`` tensors.
    """
    p_dim, ny, nx = phi_full.shape
    n_pix = ny * nx
    n_cad = coeffs.shape[0]
    phi_flat = np.ascontiguousarray(
        phi_full.reshape(p_dim, n_pix), dtype=np.float64
    )  # (P, N_pix)

    bkg_out = np.empty((n_cad, ny, nx), dtype=out_dtype)
    err_out = np.empty((n_cad, ny, nx), dtype=out_dtype)
    if n_cad == 0:
        return bkg_out, err_out

    # 1. Reconstruct B_hat in 8,192-cadence GEMM chunks (< 270 MB float64 buffer).
    buf_b = np.empty((min(n_cad, chunk_size), n_pix), dtype=np.float64)
    for start in range(0, n_cad, chunk_size):
        stop = min(start + chunk_size, n_cad)
        m_len = stop - start
        c_sub = coeffs[start:stop]
        bad_c = ~np.all(np.isfinite(c_sub), axis=1)
        view_b = buf_b[:m_len]
        np.matmul(
            np.where(bad_c[:, np.newaxis], 0.0, c_sub), phi_flat, out=view_b
        )
        if np.any(bad_c):
            view_b[bad_c] = np.nan
        bkg_out[start:stop] = view_b.reshape(m_len, ny, nx)
    del buf_b

    # 2. Reconstruct marginal variance sum_j ((L[:, :, j] @ Phi_full.T) ** 2)
    #    in 4,096-cadence sub-chunks so acc + tmp together stay < 270 MB.
    sub_chunk = min(chunk_size, _VAR_SUBCHUNK)
    acc = np.empty((min(n_cad, sub_chunk), n_pix), dtype=np.float64)
    tmp = np.empty((min(n_cad, sub_chunk), n_pix), dtype=np.float64)
    for start in range(0, n_cad, sub_chunk):
        stop = min(start + sub_chunk, n_cad)
        m_len = stop - start
        acc_v = acc[:m_len]
        tmp_v = tmp[:m_len]
        acc_v.fill(0.0)
        chol_sub = bkg_chol[start:stop]
        bad_l = ~np.all(np.isfinite(chol_sub), axis=(1, 2))
        chol_clean = np.where(bad_l[:, np.newaxis, np.newaxis], 0.0, chol_sub)
        for j in range(p_dim):
            l_col = np.ascontiguousarray(chol_clean[:, :, j], dtype=np.float64)
            np.matmul(l_col, phi_flat, out=tmp_v)
            np.square(tmp_v, out=tmp_v)
            acc_v += tmp_v
        np.sqrt(acc_v, out=acc_v)
        if np.any(bad_l):
            acc_v[bad_l] = np.nan
        err_out[start:stop] = acc_v.reshape(m_len, ny, nx)

    return bkg_out, err_out
