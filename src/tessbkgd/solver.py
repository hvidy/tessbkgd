"""QR-equilibrated 2D Chebyshev basis and batched JAX Huber-Tukey IRLS solver."""

from __future__ import annotations

import math
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
_SMEAR_SUBCHUNK = 512
_BATCH_BUCKETS = (128, 2048, 8192)
_PIXEL_BUCKETS = (256, 2048, 4096)


@dataclass(frozen=True)
class SolverOutput:
    """Per-cadence Chebyshev fit coefficients, sandwich Cholesky, and diagnostics.

    Under ``HALOCVER = 1``, ``bkg_chol`` is strictly the lower-triangular factor
    ``L_t`` of the pixel-independent Huber–Street–Carroll–Ruppert sandwich
    covariance ``Sigma_{beta,indep,t} = L_t @ L_t.T`` conditional on column smear
    (carried separately in ``SMEAR_ERR``). When ``smear_err`` is supplied to
    :func:`solve_chebyshev_background`, ``delta_bkg_smear_sq`` holds the positive
    semi-definite column-coherent smear background variance
    ``phi_j.T @ Sigma_{beta,smear,t} @ phi_j`` of shape ``(N_cad, ny, nx)`` so
    standalone ``FLUX_BKG_ERR = sqrt(||L_t.T @ phi_j||_2^2 + delta_bkg_smear_sq)``.
    """

    coeffs: np.ndarray
    bkg_chol: np.ndarray
    tau_scale: np.ndarray
    n_valid_pixels: np.ndarray
    n_tukey_inliers: np.ndarray
    used_huber_fallback: np.ndarray
    delta_bkg_smear_sq: np.ndarray | None = None


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
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
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
    zero_chol = jnp.zeros((b_dim, p_pad, p_pad), dtype=dtype)

    beta_out = jnp.where(cad_final_ok[:, None], beta_final, nan_beta)
    chol_valid = cad_final_ok & jnp.isfinite(alpha)
    chol_out = jnp.where(chol_valid[:, None, None], l_chol, nan_chol)
    tau_out = jnp.where(cad_final_ok, tau_final, jnp.nan)
    m_inliers_out = jnp.where(tukey_bread_ok, m_t, m_inliers_step)
    used_fallback = cad_final_ok & (~tukey_bread_ok)

    w_eff_raw = jnp.where(tukey_bread_ok[:, None], w_in_t, w_in_h)
    w_eff_out = jnp.where(cad_final_ok[:, None], w_eff_raw, 0.0)
    l_base_out = jnp.where(cad_final_ok[:, None, None], l_base, zero_chol)

    return (
        beta_out,
        chol_out,
        tau_out,
        n_valid,
        m_inliers_out,
        used_fallback,
        l_base_out,
        w_eff_out,
    )


@in_float64
def _run_chunk_float64(
    u_chunk: np.ndarray,
    sigma_chunk: np.ndarray,
    phi_bg: np.ndarray,
    p_actual: np.ndarray,
    has_flux_err: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
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


def _poly_vert_dof_from_p(p_dim: int) -> int:
    """Return number of 2D basis modes with vertical degree ``q >= 1``."""
    disc = 1 + 8 * p_dim
    root = int(math.isqrt(disc))
    if root * root == disc and root >= 3 and (root - 3) % 2 == 0:
        d = (root - 3) // 2
        return p_dim - (d + 1)
    return max(p_dim - 1, 0)


def _psd_lower_cholesky_batch(
    mat: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    """Factor symmetric PSD ``(B, P, P)`` matrices into lower-triangular ``L``.

    Uses eigendecomposition plus QR on ``B.T`` so rank-deficient or identically
    zero matrices (e.g. ``sigma_indep == 0``) yield exact non-negative lower
    factors satisfying ``L @ L.T == mat`` without Cholesky ``NaN``s.
    """
    b_len, p_dim, _ = mat.shape
    out = np.full((b_len, p_dim, p_dim), np.nan, dtype=np.float64)
    good = valid_mask & np.all(np.isfinite(mat), axis=(1, 2))
    if not np.any(good):
        return out

    sym = 0.5 * (mat[good] + np.swapaxes(mat[good], 1, 2))
    evals, evecs = np.linalg.eigh(sym)
    evals_pos = np.maximum(evals, 0.0)
    b_sqrt = evecs * np.sqrt(evals_pos)[:, np.newaxis, :]

    _, r_up = np.linalg.qr(np.swapaxes(b_sqrt, 1, 2), mode="reduced")
    l_low = np.swapaxes(r_up, 1, 2).copy()
    diag = np.diagonal(l_low, axis1=1, axis2=2)
    signs = np.where(diag >= 0.0, 1.0, -1.0)
    l_low *= signs[:, np.newaxis, :]
    tril_mask = np.tril(np.ones((p_dim, p_dim), dtype=bool))
    l_low = np.where(tril_mask[np.newaxis, :, :], l_low, 0.0)

    zero_mat = np.max(evals_pos, axis=1) <= 0.0
    if np.any(zero_mat):
        l_low[zero_mat] = 0.0

    out[good] = l_low
    return out


def _compute_smear_decomposed_chunk(
    u_chunk: np.ndarray,
    sig_chunk: np.ndarray,
    has_flux_err: bool,
    coeffs_chunk: np.ndarray,
    chol_orig_chunk: np.ndarray,
    fallback_chunk: np.ndarray,
    l_base_chunk: np.ndarray,
    w_eff_chunk: np.ndarray,
    phi_bg: np.ndarray,
    phi_flat: np.ndarray,
    active_col_groups: list[tuple[int, np.ndarray, np.ndarray]],
    smear_chunk: np.ndarray,
    ny: int,
    nx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate ``Sigma_{beta,indep,t}`` and ``Sigma_{beta,smear,t}`` on a chunk.

    Evaluates the Street–Carroll–Ruppert (1988) robust efficiency factor
    ``c_{rob,t}`` on smear-free within-column centred residuals, factors the
    pixel-independent sandwich covariance
    ``Sigma_{beta,indep,t} = c_{rob,t} * H_t @ diag(sigma_indep_sq) @ H_t.T``
    into ``L_t`` (for ``BKG_CHOL`` and ``sub.FLUX_ERR``), and evaluates the
    manifestly positive semi-definite column-coherent smear background variance
    ``phi_j.T @ Sigma_{beta,smear,t} @ phi_j`` with
    ``Sigma_{beta,smear,t} = sum_c smear_err[t, c]^2 * s_{t,c} @ s_{t,c}.T``.
    """
    m_len, p_dim = coeffs_chunk.shape
    n_bg = u_chunk.shape[1]
    n_pix_full = ny * nx
    chol_indep_out = np.full((m_len, p_dim, p_dim), np.nan, dtype=np.float64)
    smear_var_out = np.full((m_len, ny, nx), np.nan, dtype=np.float64)
    if m_len == 0 or not active_col_groups:
        return chol_indep_out, smear_var_out

    p_vert = _poly_vert_dof_from_p(p_dim)
    tau_floor = 1.0 if has_flux_err else 1e-6
    multi_row_cols = [
        (col_c, idx_c) for col_c, idx_c, _ in active_col_groups if idx_c.size >= 2
    ]

    sub_cap = min(m_len, _SMEAR_SUBCHUNK)
    acc_buf = np.empty((sub_cap, n_pix_full), dtype=np.float64)
    tmp_buf = np.empty((sub_cap, n_pix_full), dtype=np.float64)

    for s0 in range(0, m_len, _SMEAR_SUBCHUNK):
        s1 = min(s0 + _SMEAR_SUBCHUNK, m_len)
        sub_len = s1 - s0

        u_sub = u_chunk[s0:s1]
        sig_sub = sig_chunk[s0:s1]
        c_sub = coeffs_chunk[s0:s1]
        l_orig_sub = chol_orig_chunk[s0:s1]
        fb_sub = fallback_chunk[s0:s1]
        lb_sub = l_base_chunk[s0:s1]
        we_sub = w_eff_chunk[s0:s1]
        se_sub = smear_chunk[s0:s1]

        cad_ok = np.all(np.isfinite(c_sub), axis=1) & np.all(
            np.isfinite(l_orig_sub), axis=(1, 2)
        )
        if not np.any(cad_ok):
            continue

        valid_err = np.isfinite(sig_sub) & (sig_sub > 0.0)
        valid_px = np.isfinite(u_sub) & (valid_err if has_flux_err else True)
        c_safe = np.where(cad_ok[:, np.newaxis], c_sub, 0.0)
        resid_sub = np.where(
            cad_ok[:, np.newaxis] & valid_px,
            u_sub - c_safe @ phi_bg.T,
            0.0,
        )

        sig_indep_sq = np.zeros((sub_len, n_bg), dtype=np.float64)
        r_cent = np.zeros((sub_len, n_bg), dtype=np.float64)
        var_cent = np.zeros((sub_len, n_bg), dtype=np.float64)
        cent_ok = np.zeros((sub_len, n_bg), dtype=bool)

        for col_c, idx_c, _ in active_col_groups:
            se_c = se_sub[:, col_c]
            se_ok_c = np.isfinite(se_c) & (se_c >= 0.0)
            se_sq_c = np.where(se_ok_c, se_c * se_c, 0.0)[:, np.newaxis]

            v_c = valid_px[:, idx_c]
            if has_flux_err:
                si2_c = np.where(
                    v_c,
                    np.maximum(sig_sub[:, idx_c] ** 2 - se_sq_c, 0.0),
                    0.0,
                )
            else:
                si2_c = np.where(v_c, 1.0, 0.0)
            sig_indep_sq[:, idx_c] = si2_c

            if idx_c.size >= 2:
                cnt_c = np.sum(v_c, axis=1, keepdims=True)
                cnt_safe = np.maximum(cnt_c, 1)
                mean_r = (
                    np.sum(resid_sub[:, idx_c], axis=1, keepdims=True) / cnt_safe
                )
                rc_c = np.where(
                    v_c & (cnt_c >= 2), resid_sub[:, idx_c] - mean_r, 0.0
                )
                sum_si2 = np.sum(si2_c, axis=1, keepdims=True)
                vc_c = np.maximum(
                    (1.0 - 2.0 / cnt_safe) * si2_c
                    + sum_si2 / (cnt_safe * cnt_safe),
                    0.0,
                )
                r_cent[:, idx_c] = rc_c
                var_cent[:, idx_c] = vc_c
                cent_ok[:, idx_c] = v_c & (cnt_c >= 2) & (vc_c > 0.0)

        n_el = np.sum(cent_ok, axis=1)
        c_rob = np.where(cad_ok, 1.0 if has_flux_err else np.nan, np.nan)
        el_rows = cad_ok & (n_el > 0)
        if np.any(el_rows):
            ok_el = cent_ok[el_rows]
            denom_el = np.sqrt(np.where(ok_el, var_cent[el_rows], 1.0))
            z_el = np.where(ok_el, r_cent[el_rows] / denom_el, np.nan)
            med_z = np.nanmedian(z_el, axis=1, keepdims=True)
            mad_sig = _MAD_TO_SIGMA * np.nanmedian(
                np.abs(z_el - med_z), axis=1
            )
            tau_c = np.maximum(tau_floor, mad_sig)
            u_c = np.where(ok_el, z_el / tau_c[:, np.newaxis], 0.0)

            in_b = ok_el & (np.abs(u_c) < _TUKEY_C)
            m_b = np.sum(in_b, axis=1)
            c_b = np.zeros(in_b.shape[0], dtype=np.int32)
            c_all = np.zeros(ok_el.shape[0], dtype=np.int32)
            for _, idx_c in multi_row_cols:
                c_b += np.any(in_b[:, idx_c], axis=1)
                c_all += np.any(ok_el[:, idx_c], axis=1)

            df_col_b = np.maximum(m_b - c_b, 1)
            df_rem_b = df_col_b - p_vert
            q_b = np.where(in_b, (u_c / _TUKEY_C) ** 2, 1.0)
            psi_b = np.where(in_b, u_c * ((1.0 - q_b) ** 2), 0.0)
            psip_b = np.where(in_b, (1.0 - q_b) * (1.0 - 5.0 * q_b), 0.0)
            m_b_f = np.maximum(m_b.astype(np.float64), 1.0)
            mean_psi2_b = np.sum(psi_b * psi_b, axis=1) / m_b_f
            mean_psip_b = np.sum(psip_b, axis=1) / m_b_f
            psip_safe = np.maximum(mean_psip_b, 0.05)
            scr_ok_b = (m_b > 0) & (df_rem_b > 0) & (mean_psip_b > 0.05)
            fallback_scale = (tau_c * tau_c) if has_flux_err else np.nan
            c_rob_tuk = np.where(
                scr_ok_b,
                (tau_c * tau_c)
                * (df_col_b / np.maximum(df_rem_b, 1))
                * (mean_psi2_b / (psip_safe * psip_safe)),
                fallback_scale,
            )

            in_h = ok_el & (np.abs(u_c) <= _HUBER_C)
            m_h = np.sum(in_h, axis=1)
            df_col_h = np.maximum(n_el[el_rows] - c_all, 1)
            df_rem_h = df_col_h - p_vert
            psi_h = np.where(ok_el, np.clip(u_c, -_HUBER_C, _HUBER_C), 0.0)
            m_h_f = np.maximum(m_h.astype(np.float64), 1.0)
            scr_ok_h = (m_h > 0) & (df_rem_h > 0)
            c_rob_hub = np.where(
                scr_ok_h,
                (tau_c * tau_c)
                * (df_col_h / np.maximum(df_rem_h, 1))
                * (np.sum(psi_h * psi_h, axis=1) / m_h_f),
                fallback_scale,
            )
            c_rob[el_rows] = np.where(fb_sub[el_rows], c_rob_hub, c_rob_tuk)

        g_inv = np.matmul(lb_sub, np.swapaxes(lb_sub, 1, 2))
        cov_indep_raw = np.zeros((sub_len, p_dim, p_dim), dtype=np.float64)
        cov_smear = np.zeros((sub_len, p_dim, p_dim), dtype=np.float64)
        bad_smear_cad = ~cad_ok

        for col_c, idx_c, phi_c_t in active_col_groups:
            w_c = we_sub[:, idx_c]
            col_active = np.any(w_c > 0.0, axis=1)
            if not np.any(col_active):
                continue

            h_c = np.matmul(g_inv, phi_c_t) * w_c[:, np.newaxis, :]
            h_ind = h_c * np.sqrt(sig_indep_sq[:, idx_c])[:, np.newaxis, :]
            cov_indep_raw += np.matmul(h_ind, np.swapaxes(h_ind, 1, 2))

            se_c = se_sub[:, col_c]
            se_ok = np.isfinite(se_c) & (se_c >= 0.0)
            bad_smear_cad |= col_active & (~se_ok)

            se_sq_clean = np.where(col_active & se_ok, se_c * se_c, 0.0)
            if np.any(se_sq_clean > 0.0):
                s_c = np.sum(h_c, axis=2)
                cov_smear += (
                    se_sq_clean[:, np.newaxis, np.newaxis]
                    * s_c[:, :, np.newaxis]
                    * s_c[:, np.newaxis, :]
                )

        indep_valid = cad_ok & np.isfinite(c_rob)
        cov_indep = np.where(
            indep_valid[:, np.newaxis, np.newaxis],
            c_rob[:, np.newaxis, np.newaxis] * cov_indep_raw,
            np.nan,
        )
        chol_indep_out[s0:s1] = _psd_lower_cholesky_batch(
            cov_indep, indep_valid
        )

        smear_valid = ~bad_smear_cad
        if np.any(smear_valid):
            sym_s = 0.5 * (
                cov_smear[smear_valid]
                + np.swapaxes(cov_smear[smear_valid], 1, 2)
            )
            evals_s, evecs_s = np.linalg.eigh(sym_s)
            b_smear = evecs_s * np.sqrt(np.maximum(evals_s, 0.0))[:, None, :]
            n_ok = b_smear.shape[0]
            acc_v = acc_buf[:n_ok]
            tmp_v = tmp_buf[:n_ok]
            acc_v.fill(0.0)
            for q in range(p_dim):
                b_col = np.ascontiguousarray(
                    b_smear[:, :, q], dtype=np.float64
                )
                np.matmul(b_col, phi_flat, out=tmp_v)
                np.square(tmp_v, out=tmp_v)
                acc_v += tmp_v
            sub_out = smear_var_out[s0:s1]
            sub_out[smear_valid] = acc_v.reshape(n_ok, ny, nx)

    return chol_indep_out, smear_var_out


def solve_chebyshev_background(
    u_bg: np.ndarray,
    phi_bg: np.ndarray,
    *,
    sigma_u_bg: np.ndarray | None = None,
    chunk_size: int = _SOLVE_CHUNK,
    smear_err: np.ndarray | None = None,
    phi_full: np.ndarray | None = None,
    bkg_mask: np.ndarray | None = None,
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

    delta_smear_sq: np.ndarray | None = None
    smear_arr: np.ndarray | None = None
    phi_flat_smear: np.ndarray | None = None
    active_col_groups: list[tuple[int, np.ndarray, np.ndarray]] = []
    smear_ny = 0
    smear_nx = 0

    if smear_err is not None:
        if phi_full is None or bkg_mask is None:
            raise ValueError(
                "phi_full and bkg_mask are required when smear_err is provided."
            )
        phi_full_arr = np.asarray(phi_full, dtype=np.float64)
        mask_bool = np.asarray(bkg_mask, dtype=bool)
        if phi_full_arr.ndim != 3 or phi_full_arr.shape[0] != p_dim:
            raise ValueError(
                f"phi_full must have shape ({p_dim}, ny, nx), "
                f"got {phi_full_arr.shape}."
            )
        _, smear_ny, smear_nx = phi_full_arr.shape
        if mask_bool.shape != (smear_ny, smear_nx):
            raise ValueError(
                f"bkg_mask shape {mask_bool.shape} does not match "
                f"({smear_ny}, {smear_nx})."
            )
        if int(np.count_nonzero(mask_bool)) != n_pix:
            raise ValueError(
                f"bkg_mask nonzero count ({int(np.count_nonzero(mask_bool))}) "
                f"does not match n_pix ({n_pix})."
            )
        smear_arr = np.asarray(smear_err, dtype=np.float64)
        if smear_arr.shape != (n_cad, smear_nx):
            raise ValueError(
                f"smear_err shape {smear_arr.shape} must match "
                f"({n_cad}, {smear_nx})."
            )
        delta_smear_sq = np.zeros((n_cad, smear_ny, smear_nx), dtype=np.float64)
        phi_flat_smear = np.ascontiguousarray(
            phi_full_arr.reshape(p_dim, smear_ny * smear_nx), dtype=np.float64
        )
        _, bg_col_idx = np.nonzero(mask_bool)
        for col_c in range(smear_nx):
            idx_c = np.flatnonzero(bg_col_idx == col_c)
            if idx_c.size > 0:
                phi_c_t = np.ascontiguousarray(
                    phi_arr[idx_c, :].T, dtype=np.float64
                )
                active_col_groups.append((col_c, idx_c, phi_c_t))

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
            delta_bkg_smear_sq=delta_smear_sq,
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

        u_slice = u_arr[start:stop]
        s_slice = (
            sig_arr[start:stop]
            if has_flux_err
            else np.ones((m_len, n_pix), dtype=np.float64)
        )
        if b_pad == m_len and m_pad == n_pix:
            u_in = u_slice
            s_in = s_slice
        else:
            u_in = np.full((b_pad, m_pad), np.nan, dtype=np.float64)
            u_in[:m_len, :n_pix] = u_slice
            s_in = np.ones((b_pad, m_pad), dtype=np.float64)
            if has_flux_err:
                s_in[:m_len, :n_pix] = s_slice

        c_np, l_np, t_np, nv_np, nt_np, fb_np, lb_np, we_np = (
            _run_chunk_float64(
                u_in, s_in, phi_pad, p_actual_arr, has_err_arr
            )
        )
        c_chunk = c_np[:m_len, :p_dim]
        l_chunk = l_np[:m_len, :p_dim, :p_dim]
        fb_chunk = fb_np[:m_len]

        coeffs[start:stop] = c_chunk
        tau_scale[start:stop] = t_np[:m_len]
        n_valid[start:stop] = nv_np[:m_len]
        n_tukey[start:stop] = nt_np[:m_len]
        fallback[start:stop] = fb_chunk

        if (
            delta_smear_sq is not None
            and smear_arr is not None
            and phi_flat_smear is not None
        ):
            chol_ind_chunk, smear_var_chunk = _compute_smear_decomposed_chunk(
                u_chunk=u_slice,
                sig_chunk=s_slice,
                has_flux_err=has_flux_err,
                coeffs_chunk=c_chunk,
                chol_orig_chunk=l_chunk,
                fallback_chunk=fb_chunk,
                l_base_chunk=lb_np[:m_len, :p_dim, :p_dim],
                w_eff_chunk=we_np[:m_len, :n_pix],
                phi_bg=phi_arr,
                phi_flat=phi_flat_smear,
                active_col_groups=active_col_groups,
                smear_chunk=smear_arr[start:stop],
                ny=smear_ny,
                nx=smear_nx,
            )
            bkg_chol[start:stop] = chol_ind_chunk
            delta_smear_sq[start:stop] = smear_var_chunk
        else:
            bkg_chol[start:stop] = l_chunk

    return SolverOutput(
        coeffs=coeffs,
        bkg_chol=bkg_chol,
        tau_scale=tau_scale,
        n_valid_pixels=n_valid,
        n_tukey_inliers=n_tukey,
        used_huber_fallback=fallback,
        delta_bkg_smear_sq=delta_smear_sq,
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
