"""QR-equilibrated 2D Chebyshev basis and batched JAX Huber-Tukey IRLS solver."""

from __future__ import annotations

import warnings
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
_BATCH_BUCKETS = (128, 512, 2048, 8192)
_PIXEL_BUCKETS = (256, 2048, 4096)
_MAX_SOLVE_ELEMENTS = 2_097_152


@dataclass(frozen=True)
class SolverOutput:
    """Per-cadence Chebyshev coefficients, HC2 sandwich Cholesky, and diagnostics."""

    coeffs: np.ndarray
    bkg_chol: np.ndarray
    tau_scale: np.ndarray
    n_valid_pixels: np.ndarray
    n_tukey_inliers: np.ndarray
    used_huber_fallback: np.ndarray
    smear_chol: np.ndarray | None = None


def equilibrate_chebyshev_basis(
    ny: int, nx: int, order: int | str | tuple[int, int], bkg_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct QR-equilibrated 2D Chebyshev basis ``Phi_full`` on ``bkg_mask``."""
    mask_bool = np.asarray(bkg_mask, dtype=bool)
    if mask_bool.shape != (ny, nx):
        raise ValueError(f"bkg_mask shape {mask_bool.shape} != ({ny}, {nx}).")
    n_p = n_basis_for_order(order)
    n_bg = int(np.count_nonzero(mask_bool))
    if n_bg < n_p:
        raise ValueError(f"Insufficient background pixels ({n_bg}) for P={n_p}.")

    phi_tilde = raw_chebyshev_2d(ny, nx, order)
    q_bg, r_bg = np.linalg.qr(phi_tilde[:, mask_bool].T, mode="reduced")
    d_r = np.diag(r_bg)
    md = float(np.max(np.abs(d_r))) if d_r.size else 0.0
    if md <= 0.0 or float(np.min(np.abs(d_r))) <= 1e-12 * md:
        raise ValueError("Background design matrix is numerically rank-deficient.")

    sgn = np.where(d_r >= 0.0, 1.0, -1.0)
    q_bg, r_bg = q_bg * sgn[None, :], sgn[:, None] * r_bg
    phi_flat = solve_triangular(
        r_bg.T, phi_tilde.reshape(n_p, ny * nx), lower=True, check_finite=False
    )
    return (
        np.ascontiguousarray(phi_flat.reshape(n_p, ny, nx), dtype=np.float64),
        np.ascontiguousarray(q_bg, dtype=np.float64),
        r_bg,
    )


def _masked_median_2d(vals: jax.Array, mask: jax.Array) -> jax.Array:
    n = jnp.sum(mask, axis=1, dtype=jnp.int32)
    s = jnp.sort(jnp.where(mask, vals, jnp.inf), axis=1)
    lim, row = vals.shape[1] - 1, jnp.arange(vals.shape[0])
    lo_m = s[row, jnp.clip((n - 1) // 2, 0, lim)]
    hi_m = s[row, jnp.clip(n // 2, 0, lim)]
    return jnp.where(n > 0, 0.5 * (lo_m + hi_m), 0.0)


def _adaptive_scale_2d(
    resid: jax.Array, sig: jax.Array, valid_px: jax.Array, has_err: jax.Array
) -> jax.Array:
    z = resid / sig
    mz = _masked_median_2d(z, valid_px)
    mad = _MAD_TO_SIGMA * _masked_median_2d(
        jnp.where(valid_px, jnp.abs(z - mz[:, None]), 0.0), valid_px
    )
    return jnp.maximum(jnp.where(has_err, 1.0, 1e-6), mad)


def _chol_checked_1d(gram: jax.Array, act: jax.Array) -> tuple[jax.Array, jax.Array]:
    chol = jnp.linalg.cholesky(0.5 * (gram + gram.T))
    d = jnp.diag(chol)
    hi = jnp.max(jnp.where(act & jnp.isfinite(d), d, 0.0))
    lo = jnp.min(jnp.where(act & jnp.isfinite(d), d, jnp.inf))
    ok = jnp.all(jnp.isfinite(chol)) & (hi > 0.0) & (lo > 1e-6 * hi)
    return jnp.where(ok, chol, jnp.eye(gram.shape[0], dtype=gram.dtype)), ok


def _safe_cho_solve_1d(
    gram: jax.Array, rhs: jax.Array, fb: jax.Array, act: jax.Array
) -> tuple[jax.Array, jax.Array]:
    chol, ok = _chol_checked_1d(gram, act)
    sol = jax.scipy.linalg.cho_solve((chol, True), rhs)
    ok_a = ok & jnp.all(jnp.isfinite(sol))
    return jnp.where(ok_a, sol, fb), ok_a


def _invert_spd_1d(gram: jax.Array, act: jax.Array) -> tuple[jax.Array, jax.Array]:
    p = gram.shape[0]
    chol, ok = _chol_checked_1d(gram, act)
    inv = jax.scipy.linalg.cho_solve((chol, True), jnp.eye(p, dtype=gram.dtype))
    out = jnp.where(ok, inv * (act[:, None] & act[None, :]), 0.0)
    return out, ok & jnp.all(jnp.isfinite(out))


def _psd_lower_chol_1d(cov: jax.Array, act: jax.Array) -> tuple[jax.Array, jax.Array]:
    ev, ec = jnp.linalg.eigh(0.5 * (cov + cov.T))
    b_sq = ec * jnp.sqrt(jnp.maximum(ev, 0.0))[None, :]
    _, r_up = jnp.linalg.qr(b_sq.T, mode="reduced")
    l_m = r_up.T * jnp.where(jnp.diag(r_up.T) >= 0.0, 1.0, -1.0)[None, :]
    l_m = jnp.where(act[:, None] & act[None, :], jnp.tril(l_m), 0.0)
    return l_m, jnp.all(jnp.isfinite(cov)) & jnp.all(jnp.isfinite(l_m))


_vsolve = jax.vmap(_safe_cho_solve_1d, in_axes=(0, 0, 0, None))
_vchol = jax.vmap(_chol_checked_1d, in_axes=(0, None))
_vinv = jax.vmap(_invert_spd_1d, in_axes=(0, None))
_vpsd = jax.vmap(_psd_lower_chol_1d, in_axes=(0, None))


def _hc2_matrix_sandwich(
    phi_bg: jax.Array, phi_out: jax.Array, w0: jax.Array,
    psi: jax.Array, psip: jax.Array, tau: jax.Array,
    act: jax.Array, inact_eye: jax.Array, in_m: jax.Array,
    has_err: jax.Array, n_v: jax.Array, p_act: jax.Array,
) -> tuple[jax.Array, ...]:
    """Exact ``P x P`` HC2 Huber-Street-Carroll-Ruppert matrix sandwich."""
    b, p = psi.shape[0], phi_bg.shape[1]
    bread = (jnp.where(in_m, w0 * psip, 0.0) @ phi_out).reshape(b, p, p) + inact_eye
    a_inv, b_ok = _vinv(bread, act)

    w_lev = jnp.where(in_m, w0 * jnp.maximum(psip, 0.0), 0.0)
    a_phi = jnp.einsum("bpq,mq->bmp", a_inv, phi_bg)
    h_cl = jnp.clip(w_lev * jnp.sum(phi_bg[None, :, :] * a_phi, axis=2), 0.0, 0.90)

    w_m = jnp.where(in_m, (tau[:, None] ** 2) * w0 * (psi * psi) / (1.0 - h_cl), 0.0)
    meat = (w_m @ phi_out).reshape(b, p, p)
    cov_hc2 = a_inv @ meat @ jnp.swapaxes(a_inv, 1, 2)

    gram_in = (jnp.where(in_m, w0, 0.0) @ phi_out).reshape(b, p, p) + inact_eye
    g_inv, g_ok = _vinv(gram_in, act)

    df = n_v - p_act
    tr_m = jnp.trace(meat, axis1=1, axis2=2)
    tr_b = jnp.trace(bread - inact_eye, axis1=1, axis2=2)
    use_f = has_err & ((df <= 0) | (tr_m <= 1e-12 * jnp.maximum(tr_b, 1e-30)))
    l_cov, c_ok = _vpsd(jnp.where(use_f[:, None, None], g_inv, cov_hc2), act)
    ok_a = b_ok & g_ok & c_ok & ~((~has_err) & (df <= 0))
    return l_cov, ok_a, bread, gram_in, a_inv, g_inv, h_cl


@jax.jit
def _solve_chunk_jit(
    u_ch: jax.Array, sig_ch: jax.Array, phi_bg: jax.Array,
    p_actual: jax.Array, has_err: jax.Array,
) -> tuple[jax.Array, ...]:
    """Median-seeded 8-step Huber + 2-step guarded Tukey IRLS + HC2 sandwich."""
    b, m = u_ch.shape
    p, dt = phi_bg.shape[1], u_ch.dtype
    p_act = p_actual.astype(jnp.int32)
    act = jnp.arange(p, dtype=jnp.int32) < p_act
    inact = jnp.diag((~act).astype(dt))[None, :, :]
    phi_out = (phi_bg[:, :, None] * phi_bg[:, None, :]).reshape(m, p * p)

    v_err = jnp.isfinite(sig_ch) & (sig_ch > 0.0)
    v_px = jnp.isfinite(u_ch) & jnp.where(has_err, v_err, True)
    sig = jnp.where(has_err & v_px, sig_ch, 1.0)
    w0 = jnp.where(v_px, 1.0 / (sig * sig), 0.0)
    n_v = jnp.sum(v_px, axis=1, dtype=jnp.int32)
    y = jnp.where(v_px, u_ch, 0.0)
    min_tuk = jnp.maximum(2 * p_act, p_act + 1)

    # Step 0: median-pedestal Huber seed (immune to corner outlier clusters).
    r0 = y - _masked_median_2d(y, v_px)[:, None]
    t0 = _adaptive_scale_2d(r0, sig, v_px, has_err)
    u0 = jnp.maximum(jnp.abs(r0 / (t0[:, None] * sig)), 1e-30)
    wi = w0 * jnp.minimum(1.0, _HUBER_C / u0)
    g0 = (wi @ phi_out).reshape(b, p, p) + inact
    b0, ok0 = _vsolve(g0, (wi * y) @ phi_bg, jnp.zeros((b, p), dtype=dt), act)
    cad_ok = (n_v >= p_act) & ok0

    def _h_step(_: jax.Array, bc: jax.Array) -> jax.Array:
        res = y - bc @ phi_bg.T
        tau = _adaptive_scale_2d(res, sig, v_px, has_err)
        uh = jnp.maximum(jnp.abs(res / (tau[:, None] * sig)), 1e-30)
        wh = w0 * jnp.minimum(1.0, _HUBER_C / uh)
        gram = (wh @ phi_out).reshape(b, p, p) + inact
        return _vsolve(gram, (wh * y) @ phi_bg, bc, act)[0]

    b_hub = jax.lax.fori_loop(0, 7, _h_step, b0)

    def _t_step(_: jax.Array, carry: tuple[jax.Array, jax.Array, jax.Array]):
        bc, tok, _ = carry
        res = y - bc @ phi_bg.T
        tau = _adaptive_scale_2d(res, sig, v_px, has_err)
        u_s = res / (tau[:, None] * sig)
        in_s = v_px & (jnp.abs(u_s) < _TUKEY_C)
        m_s = jnp.sum(in_s, axis=1, dtype=jnp.int32)
        wt = w0 * ((1.0 - jnp.where(in_s, (u_s / _TUKEY_C) ** 2, 1.0)) ** 2)
        gram = (wt @ phi_out).reshape(b, p, p) + inact
        sol, sok = _vsolve(gram, (wt * y) @ phi_bg, b_hub, act)
        ok_n = tok & (m_s >= min_tuk) & sok
        return jnp.where(ok_n[:, None], sol, b_hub), ok_n, m_s

    b_tuk, tuk_ok, m_st = jax.lax.fori_loop(
        0, 2, _t_step, (b_hub, cad_ok, jnp.zeros((b,), dtype=jnp.int32))
    )

    res_t = y - b_tuk @ phi_bg.T
    tau_t = _adaptive_scale_2d(res_t, sig, v_px, has_err)
    u_t = res_t / (tau_t[:, None] * sig)
    in_t = v_px & (jnp.abs(u_t) < _TUKEY_C)
    m_t = jnp.sum(in_t, axis=1, dtype=jnp.int32)
    q_t = jnp.where(in_t, (u_t / _TUKEY_C) ** 2, 1.0)
    psi_t = jnp.where(in_t, u_t * ((1.0 - q_t) ** 2), 0.0)
    psip_t = jnp.where(in_t, (1.0 - q_t) * (1.0 - 5.0 * q_t), 0.0)
    l_t, ok_t, br_t, gr_t, ai_t, gi_t, h_t = _hc2_matrix_sandwich(
        phi_bg, phi_out, w0, psi_t, psip_t, tau_t, act, inact, in_t, has_err, m_t, p_act
    )
    _, curv_ok = _vchol(br_t - 0.15 * (gr_t - inact), act)
    use_t = tuk_ok & ok_t & curv_ok & (m_t >= min_tuk)

    res_h = y - b_hub @ phi_bg.T
    tau_h = _adaptive_scale_2d(res_h, sig, v_px, has_err)
    u_h = res_h / (tau_h[:, None] * sig)
    in_h = v_px & (jnp.abs(u_h) <= _HUBER_C)
    m_h = jnp.sum(in_h, axis=1, dtype=jnp.int32)
    psi_h = jnp.where(v_px, jnp.clip(u_h, -_HUBER_C, _HUBER_C), 0.0)
    l_h, ok_h, _, _, ai_h, _, h_h = _hc2_matrix_sandwich(
        phi_bg, phi_out, w0, psi_h, in_h.astype(dt), tau_h,
        act, inact, v_px, has_err, n_v, p_act,
    )
    w_ih = jnp.where(in_h, w0, 0.0)
    gi_ch, h_core_ok = _vinv((w_ih @ phi_out).reshape(b, p, p) + inact, act)
    hub_ok = cad_ok & (m_h >= p_act) & h_core_ok
    fin_ok = jnp.where(use_t, cad_ok, hub_ok)
    ch_ok = jnp.where(use_t, fin_ok, hub_ok & ok_h)

    nb, nc = jnp.full((b, p), jnp.nan, dtype=dt), jnp.full((b, p, p), jnp.nan, dtype=dt)
    zc = jnp.zeros((b, p, p), dtype=dt)
    ut2, ut3 = use_t[:, None], use_t[:, None, None]
    f2, f3 = fin_ok[:, None], fin_ok[:, None, None]
    w_t_in = jnp.where(in_t, w0, 0.0)
    return (
        jnp.where(f2, jnp.where(ut2, b_tuk, b_hub), nb),
        jnp.where(ch_ok[:, None, None], jnp.where(ut3, l_t, l_h), nc),
        jnp.where(fin_ok, jnp.where(use_t, tau_t, tau_h), jnp.nan),
        n_v,
        jnp.where(use_t, m_t, m_st),
        fin_ok & (~use_t),
        jnp.where(f3, jnp.where(ut3, gi_t, gi_ch), zc),
        jnp.where(f3, jnp.where(ut3, ai_t, ai_h), zc),
        jnp.where(f2, jnp.where(ut2, w_t_in, w_ih), 0.0),
        jnp.where(f2, jnp.where(ut2, psi_t, psi_h), 0.0),
        jnp.where(f2, jnp.where(ut2, h_t, h_h), 0.0),
    )


@in_float64
def _run_chunk_float64(
    u_c: np.ndarray, s_c: np.ndarray, phi: np.ndarray, p_a: np.ndarray, h_e: np.ndarray
) -> tuple[np.ndarray, ...]:
    return jax.device_get(_solve_chunk_jit(u_c, s_c, phi, p_a, h_e))


def _bucket_pixel_dim(n_pix: int) -> int:
    for b in _PIXEL_BUCKETS:
        if n_pix <= b:
            return b
    return 1 << (n_pix - 1).bit_length()


def _bucket_batch_dim(m_len: int, chunk_size: int) -> int:
    for b in _BATCH_BUCKETS:
        if m_len <= b <= chunk_size:
            return b
    return m_len


def _vertical_basis_dof(
    phi_bg: np.ndarray, col_groups: list[tuple[int, np.ndarray, np.ndarray]]
) -> int:
    cent = phi_bg.copy()
    for _, idx_c, _ in col_groups:
        cent[idx_c] -= np.mean(cent[idx_c], axis=0, keepdims=True)
    sv = np.linalg.svd(cent, compute_uv=False)
    return int(np.count_nonzero(sv > 1e-10 * sv[0])) if sv.size and sv[0] > 0 else 0


def _psd_lower_cholesky_batch(mat: np.ndarray, valid: np.ndarray) -> np.ndarray:
    b, p, _ = mat.shape
    out = np.full((b, p, p), np.nan, dtype=np.float64)
    ok = valid & np.all(np.isfinite(mat), axis=(1, 2))
    if not np.any(ok):
        return out
    ev, ec = np.linalg.eigh(0.5 * (mat[ok] + np.swapaxes(mat[ok], 1, 2)))
    evp = np.maximum(ev, 0.0)
    bsq = np.swapaxes(ec * np.sqrt(evp)[:, None, :], 1, 2)
    _, r_u = np.linalg.qr(bsq, mode="reduced")
    lm = np.swapaxes(r_u, 1, 2).copy()
    d = np.diagonal(lm, axis1=1, axis2=2)
    lm *= np.where(d >= 0.0, 1.0, -1.0)[:, None, :]
    lm = np.where(np.tril(np.ones((p, p), dtype=bool))[None, :, :], lm, 0.0)
    lm[np.max(evp, axis=1) <= 0.0] = 0.0
    out[ok] = lm
    return out


def _robust_col_scale(
    z_el: np.ndarray, ok_el: np.ndarray, is_hub: np.ndarray,
    mcols: list[tuple[int, np.ndarray]], p_vert: int, t_fl: float,
) -> np.ndarray:
    mz = np.nanmedian(z_el, axis=1, keepdims=True)
    tc = np.maximum(t_fl, _MAD_TO_SIGMA * np.nanmedian(np.abs(z_el - mz), axis=1))
    uc = np.where(ok_el, z_el / tc[:, None], 0.0)
    it = ok_el & (np.abs(uc) < _TUKEY_C)
    mt = np.sum(it, axis=1)
    ct = sum(np.any(it[:, idx], axis=1).astype(np.int32) for _, idx in mcols)
    ca = sum(np.any(ok_el[:, idx], axis=1).astype(np.int32) for _, idx in mcols)

    qt = np.where(it, (uc / _TUKEY_C) ** 2, 1.0)
    psit = np.where(it, uc * ((1.0 - qt) ** 2), 0.0)
    psipt = np.where(it, (1.0 - qt) * (1.0 - 5.0 * qt), 0.0)
    mtf = np.maximum(mt.astype(np.float64), 1.0)
    mp = np.sum(psipt, axis=1) / mtf
    df_c, df_r = mt - ct, mt - ct - p_vert
    c_t = np.where(
        (mt > 0) & (df_r > 0) & (mp > 0.05),
        (tc**2) * (df_c / np.maximum(df_r, 1)) * (np.sum(psit**2, axis=1) / mtf)
        / (np.maximum(mp, 0.05) ** 2),
        np.where((df_r > 0) & np.isfinite(tc), tc**2, np.nan),
    )
    mh = np.sum(ok_el & (np.abs(uc) <= _HUBER_C), axis=1)
    dh_c = np.sum(ok_el, axis=1) - ca
    dh_r = dh_c - p_vert
    psih = np.where(ok_el, np.clip(uc, -_HUBER_C, _HUBER_C), 0.0)
    c_h = np.where(
        (mh > 0) & (dh_r > 0),
        (tc**2) * (dh_c / np.maximum(dh_r, 1))
        * (np.sum(psih**2, axis=1) / np.maximum(mh.astype(np.float64), 1.0)),
        np.where((dh_r > 0) & np.isfinite(tc), tc**2, np.nan),
    )
    return np.where(is_hub, c_h, c_t)


def _decompose_smear_chunk(
    u_ch: np.ndarray, sig_ch: np.ndarray, has_err: bool, c_ch: np.ndarray,
    l_ch: np.ndarray, tau_ch: np.ndarray, fb_ch: np.ndarray, gi_ch: np.ndarray,
    ai_ch: np.ndarray, we_ch: np.ndarray, psi_ch: np.ndarray, h_ch: np.ndarray,
    phi_bg: np.ndarray, cgrps: list[tuple[int, np.ndarray, np.ndarray]],
    se_ch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate independent ``bkg_chol`` and compact ``(B, P, P)`` ``smear_chol``."""
    m_len, p = c_ch.shape
    n_bg = u_ch.shape[1]
    c_ind = np.full((m_len, p, p), np.nan, dtype=np.float64)
    c_sm = np.full((m_len, p, p), np.nan, dtype=np.float64)
    if m_len == 0 or not cgrps:
        return c_ind, c_sm

    pv = _vertical_basis_dof(phi_bg, cgrps)
    t_fl = 1.0 if has_err else 1e-6
    mcols = [(c, idx) for c, idx, _ in cgrps if idx.size >= 2]
    phi_o = (phi_bg[:, :, None] * phi_bg[:, None, :]).reshape(n_bg, p * p)

    for s0 in range(0, m_len, _SMEAR_SUBCHUNK):
        s1 = min(s0 + _SMEAR_SUBCHUNK, m_len)
        sl = s1 - s0
        us, ss, cs, lo = u_ch[s0:s1], sig_ch[s0:s1], c_ch[s0:s1], l_ch[s0:s1]
        ts, fbs, gis, ais = tau_ch[s0:s1], fb_ch[s0:s1], gi_ch[s0:s1], ai_ch[s0:s1]
        wes, psis, hs, ses = we_ch[s0:s1], psi_ch[s0:s1], h_ch[s0:s1], se_ch[s0:s1]

        ok_c = np.all(np.isfinite(cs), axis=1) & np.all(np.isfinite(lo), axis=(1, 2))
        if not np.any(ok_c):
            continue
        vpx = np.isfinite(us) & ((np.isfinite(ss) & (ss > 0)) if has_err else True)
        s2d = np.where(has_err & vpx, ss, 1.0)
        fit_s = np.where(ok_c[:, None], cs, 0.0) @ phi_bg.T
        res = np.where(ok_c[:, None] & vpx, us - fit_s, 0.0)
        tsf = np.where(ok_c & np.isfinite(ts) & (ts > 0), ts, 1.0)[:, None]
        inl = ok_c[:, None] & vpx & (np.abs(res) < _TUKEY_C * tsf * s2d)

        si2 = np.zeros((sl, n_bg), dtype=np.float64)
        cdev = np.zeros((sl, n_bg), dtype=np.float64)
        cmz = np.full((sl, n_bg), np.nan, dtype=np.float64)
        chm = np.zeros((sl, n_bg), dtype=bool)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
            for col_c, idx_c, _ in cgrps:
                sec = ses[:, col_c]
                se2 = np.where(np.isfinite(sec) & (sec >= 0), sec * sec, 0.0)[:, None]
                s_c = (
                    np.where(vpx[:, idx_c], np.maximum(ss[:, idx_c] ** 2 - se2, 0), 0.0)
                    if has_err
                    else np.where(vpx[:, idx_c], 1.0, 0.0)
                )
                si2[:, idx_c] = s_c
                if idx_c.size >= 3:
                    vi = inl[:, idx_c]
                    cnt = np.sum(vi, axis=1)
                    hm = cnt >= 3
                    if np.any(hm):
                        rm = np.where(vi[hm], res[hm][:, idx_c], np.nan)
                        med_rm = np.nanmedian(rm, axis=1, keepdims=True)
                        dm = np.where(vi[hm], np.abs(rm - med_rm), 0.0)
                        dc = np.sqrt(cnt[hm, None] / (cnt[hm, None] - 1.0))
                        sm_i = s_c[hm]
                        cdev[np.ix_(hm, idx_c)] = dm
                        denom = np.sqrt(np.where(sm_i > 0, sm_i, 1.0))
                        cmz[np.ix_(hm, idx_c)] = np.where(
                            vi[hm] & (sm_i > 0), dm * dc / denom, np.nan
                        )
                        chm[np.ix_(hm, idx_c)] = vi[hm]

            tmed = np.full(sl, t_fl, dtype=np.float64)
            az = np.any(np.isfinite(cmz), axis=1)
            if np.any(az):
                mad_z = _MAD_TO_SIGMA * np.nanmedian(cmz[az], axis=1)
                tmed[az] = np.maximum(t_fl, mad_z)

        lim_c = _TUKEY_C * tmed[:, None] * np.sqrt(si2)
        cinl = inl & ((~chm) | (si2 <= 0) | (cdev <= lim_c))
        rc = np.zeros((sl, n_bg))
        vc = np.zeros((sl, n_bg))
        cok = np.zeros((sl, n_bg), dtype=bool)
        for _, idx_c in mcols:
            v_m = cinl[:, idx_c]
            cnt = np.sum(v_m, axis=1, keepdims=True)
            csf = np.maximum(cnt, 1)
            mr = np.sum(np.where(v_m, res[:, idx_c], 0.0), axis=1, keepdims=True) / csf
            s_i = np.where(v_m, si2[:, idx_c], 0.0)
            s_sum = np.sum(s_i, axis=1, keepdims=True)
            vv = np.maximum((1.0 - 2.0 / csf) * s_i + s_sum / (csf * csf), 0.0)
            rc[:, idx_c] = np.where(v_m & (cnt >= 2), res[:, idx_c] - mr, 0.0)
            vc[:, idx_c], cok[:, idx_c] = vv, v_m & (cnt >= 2) & (vv > 0)

        cov_r = np.zeros((sl, p, p), dtype=np.float64)
        cov_s = np.zeros((sl, p, p), dtype=np.float64)
        bad_s = ~ok_c
        for col_c, idx_c, pct in cgrps:
            wc = wes[:, idx_c]
            ca = np.any(wc > 0, axis=1)
            if not np.any(ca):
                continue
            hc = np.matmul(gis, pct) * wc[:, None, :]
            hi = hc * np.sqrt(si2[:, idx_c])[:, None, :]
            cov_r += np.matmul(hi, np.swapaxes(hi, 1, 2))
            sec = ses[:, col_c]
            sok = np.isfinite(sec) & (sec >= 0)
            bad_s |= ca & (~sok)
            se2c = np.where(ca & sok, sec * sec, 0.0)
            if np.any(se2c > 0):
                sc = np.sum(hc, axis=2)
                cov_s += se2c[:, None, None] * sc[:, :, None] * sc[:, None, :]

        crob = np.full(sl, np.nan, dtype=np.float64)
        er = ok_c & (np.sum(cok, axis=1) > 0)
        if np.any(er):
            oe = cok[er]
            ze = np.where(oe, rc[er] / np.sqrt(np.where(oe, vc[er], 1.0)), np.nan)
            crob[er] = _robust_col_scale(ze, oe, fbs[er], mcols, pv, t_fl)

        cov_i = np.full((sl, p, p), np.nan, dtype=np.float64)
        done = ok_c & np.isfinite(crob)
        if np.any(done):
            cov_i[done] = crob[done, None, None] * cov_r[done]

        # Sparse 1-row fallback: exact HC2 matrix sandwich without 1/f_ind blowup!
        fbr = ok_c & (~done)
        if np.any(fbr):
            if has_err:
                w0f = np.where(vpx[fbr], 1.0 / (s2d[fbr] ** 2), 0.0)
                fi = np.where(vpx[fbr], si2[fbr] / (s2d[fbr] ** 2), 0.0)
                wm = (tsf[fbr] ** 2) * w0f * (psis[fbr] ** 2) * fi / (1.0 - hs[fbr])
                mf = (wm @ phi_o).reshape(-1, p, p)
                af = ais[fbr]
                cf = af @ mf @ np.swapaxes(af, 1, 2)
                zm = np.trace(mf, axis1=1, axis2=2) <= 1e-12 * np.maximum(
                    np.trace(cov_r[fbr], axis1=1, axis2=2), 1e-30
                )
                cov_i[fbr] = np.where(zm[:, None, None], cov_r[fbr], cf)
            else:
                cov_i[fbr] = lo[fbr] @ np.swapaxes(lo[fbr], 1, 2) - cov_s[fbr]

        c_ind[s0:s1] = _psd_lower_cholesky_batch(cov_i, ok_c)
        c_sm[s0:s1] = _psd_lower_cholesky_batch(cov_s, ~bad_s)
    return c_ind, c_sm


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
    ua, pa = np.asarray(u_bg, dtype=np.float64), np.asarray(phi_bg, dtype=np.float64)
    if ua.ndim != 2 or pa.ndim != 2 or ua.shape[1] != pa.shape[0]:
        raise ValueError(f"Incompatible shapes u_bg={ua.shape} and phi_bg={pa.shape}.")
    n_cad, n_pix, p = ua.shape[0], ua.shape[1], pa.shape[1]
    if p < 1:
        raise ValueError(f"phi_bg must have P >= 1 basis columns, got {p}.")

    has_err = False
    if sigma_u_bg is not None:
        sa = np.asarray(sigma_u_bg, dtype=np.float64)
        if sa.shape != ua.shape:
            raise ValueError(f"sigma_u_bg shape {sa.shape} != u_bg {ua.shape}.")
        has_err = bool(np.any(np.isfinite(ua) & np.isfinite(sa) & (sa > 0.0)))
    else:
        sa = np.ones((1, n_pix), dtype=np.float64)

    sm_arr: np.ndarray | None = None
    sm_ch: np.ndarray | None = None
    cgrps: list[tuple[int, np.ndarray, np.ndarray]] = []
    if smear_err is not None:
        if phi_full is None or bkg_mask is None:
            raise ValueError("phi_full and bkg_mask required with smear_err.")
        pf = np.asarray(phi_full, dtype=np.float64)
        mb = np.asarray(bkg_mask, dtype=bool)
        bad_sh = (
            pf.ndim != 3
            or pf.shape[0] != p
            or mb.shape != pf.shape[1:]
            or int(np.count_nonzero(mb)) != n_pix
        )
        if bad_sh:
            raise ValueError("phi_full / bkg_mask shapes inconsistent with u_bg.")
        sm_arr = np.asarray(smear_err, dtype=np.float64)
        if sm_arr.shape != (n_cad, pf.shape[2]):
            raise ValueError(
                f"smear_err shape {sm_arr.shape} != ({n_cad}, {pf.shape[2]})."
            )
        sm_ch = np.full((n_cad, p, p), np.nan, dtype=np.float64)
        _, bgc = np.nonzero(mb)
        for col_c in range(pf.shape[2]):
            idx_c = np.flatnonzero(bgc == col_c)
            if idx_c.size:
                cgrps.append((col_c, idx_c, np.ascontiguousarray(pa[idx_c, :].T)))

    coeffs = np.full((n_cad, p), np.nan, dtype=np.float64)
    b_chol = np.full((n_cad, p, p), np.nan, dtype=np.float64)
    tau = np.full(n_cad, np.nan, dtype=np.float64)
    nv, nt = np.zeros(n_cad, dtype=np.int32), np.zeros(n_cad, dtype=np.int32)
    fb = np.zeros(n_cad, dtype=bool)
    if n_cad == 0:
        return SolverOutput(coeffs, b_chol, tau, nv, nt, fb, sm_ch)

    m_pad, p_pad = _bucket_pixel_dim(n_pix), max(_P_STATIC, p)
    phi_pad = np.zeros((m_pad, p_pad), dtype=np.float64)
    phi_pad[:n_pix, :p] = pa
    eff_ch = min(chunk_size, max(64, _MAX_SOLVE_ELEMENTS // max(m_pad, 1)))
    p_aa, h_ea = np.asarray(p, dtype=np.int32), np.asarray(has_err, dtype=bool)

    for s in range(0, n_cad, eff_ch):
        e = min(s + eff_ch, n_cad)
        ml = e - s
        bp = eff_ch if n_cad > eff_ch else _bucket_batch_dim(ml, eff_ch)
        us = ua[s:e]
        ss = sa[s:e] if has_err else np.ones((ml, n_pix), dtype=np.float64)
        if bp == ml and m_pad == n_pix:
            ui, si = us, ss
        else:
            ui = np.full((bp, m_pad), np.nan, dtype=np.float64)
            ui[:ml, :n_pix] = us
            si = np.ones((bp, m_pad), dtype=np.float64)
            if has_err:
                si[:ml, :n_pix] = ss
        cn, ln, tn, nvn, ntn, fbn, gin, ain, wen, psin, hn = _run_chunk_float64(
            ui, si, phi_pad, p_aa, h_ea
        )
        c_c, l_c = cn[:ml, :p], ln[:ml, :p, :p]
        coeffs[s:e], tau[s:e] = c_c, tn[:ml]
        nv[s:e], nt[s:e], fb[s:e] = nvn[:ml], ntn[:ml], fbn[:ml]
        if sm_ch is not None and sm_arr is not None:
            b_chol[s:e], sm_ch[s:e] = _decompose_smear_chunk(
                us, ss, has_err, c_c, l_c, tn[:ml], fbn[:ml], gin[:ml, :p, :p],
                ain[:ml, :p, :p], wen[:ml, :n_pix], psin[:ml, :n_pix], hn[:ml, :n_pix],
                pa, cgrps, sm_arr[s:e],
            )
        else:
            b_chol[s:e] = l_c
    return SolverOutput(coeffs, b_chol, tau, nv, nt, fb, sm_ch)


def reconstruct_chol_var_slice(
    chol_sl: np.ndarray, phi_flat: np.ndarray, ny: int, nx: int
) -> np.ndarray:
    """Evaluate ``||L_t.T @ phi||_2^2`` across ``chol_sl`` in bounded RAM."""
    n_cad, p, _ = chol_sl.shape
    out = np.empty((n_cad, ny, nx), dtype=np.float64)
    if n_cad == 0:
        return out
    sc = min(n_cad, _VAR_SUBCHUNK)
    acc, tmp = np.empty((sc, ny * nx)), np.empty((sc, ny * nx))
    for s in range(0, n_cad, _VAR_SUBCHUNK):
        e = min(s + _VAR_SUBCHUNK, n_cad)
        av, tv = acc[: e - s], tmp[: e - s]
        av.fill(0.0)
        cs = chol_sl[s:e]
        bad = ~np.all(np.isfinite(cs), axis=(1, 2))
        cok = np.where(bad[:, None, None], 0.0, cs)
        for j in range(p):
            col_j = np.ascontiguousarray(cok[:, :, j], dtype=np.float64)
            np.matmul(col_j, phi_flat, out=tv)
            np.square(tv, out=tv)
            av += tv
        if np.any(bad):
            av[bad] = np.nan
        out[s:e] = av.reshape(e - s, ny, nx)
    return out


def reconstruct_background_chunked(
    coeffs: np.ndarray,
    bkg_chol: np.ndarray,
    phi_full: np.ndarray,
    *,
    smear_chol: np.ndarray | None = None,
    chunk_size: int = _RECON_CHUNK,
    out_dtype: Any = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct ``B_hat`` and marginal ``sigma_B_hat`` in bounded-RAM chunks."""
    p, ny, nx = phi_full.shape
    n_cad = coeffs.shape[0]
    pf = np.ascontiguousarray(phi_full.reshape(p, ny * nx), dtype=np.float64)
    bo = np.empty((n_cad, ny, nx), dtype=out_dtype)
    eo = np.empty((n_cad, ny, nx), dtype=out_dtype)
    if n_cad == 0:
        return bo, eo

    buf = np.empty((min(n_cad, chunk_size), ny * nx), dtype=np.float64)
    for s in range(0, n_cad, chunk_size):
        e = min(s + chunk_size, n_cad)
        cs = coeffs[s:e]
        bad = ~np.all(np.isfinite(cs), axis=1)
        vb = buf[: e - s]
        np.matmul(np.where(bad[:, None], 0.0, cs), pf, out=vb)
        if np.any(bad):
            vb[bad] = np.nan
        bo[s:e] = vb.reshape(e - s, ny, nx)
    del buf

    sc = min(chunk_size, _VAR_SUBCHUNK)
    for s in range(0, n_cad, sc):
        e = min(s + sc, n_cad)
        vc = reconstruct_chol_var_slice(bkg_chol[s:e], pf, ny, nx)
        if smear_chol is not None:
            vc += reconstruct_chol_var_slice(smear_chol[s:e], pf, ny, nx)
        eo[s:e] = np.sqrt(vc)
    return bo, eo
