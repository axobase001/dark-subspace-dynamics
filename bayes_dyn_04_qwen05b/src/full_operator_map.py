"""Complete 896x38 Target-Agnostic Full-Output Propagation Map & Deterministic Gaussian Conditioning (§3–§4, §6.3).

Pure Python + NumPy (zero torch dependency) for identical CPU/GPU/audit execution:
  - Full mean regression: phi(c) [53] -> M_0(c) [896, 38] via standardized multi-output ridge (unpenalized intercept).
  - Matrix-normal prior: A_c ~ MN(M_0(c), R, C),
      C in R^{38x38} shrunk toward (tr(C)/38) I_38, trace(C) = 38;
      R = U_e diag(lam_1..lam_64) U_e^T + lam_perp (I - U_e U_e^T) in R^{896x896}.
  - Exact deterministic Gaussian conditioning on Y = A_c X (X in R^{38xk}, Y in R^{896xk}):
      M_k = M_0 + (Y - M_0 X) (X^T C X)^{-1} X^T C
      C_k = C - C X (X^T C X)^{-1} X^T C
    solved stably via Cholesky (NO explicit matrix inverse; NO positive floor re-added to C_k!).
  - Target-agnostic functional projection from the exact same frozen (M_k, C_k):
      a_hat_g = (V + M_k)^T g,   Cov(a_g) = (t * g^T R g) * C_k.
  - Controls: PRIOR (M_0), UPDATE_k (M_k), PATCH_k (M_0 + (Y - M_0 X) X^T), MISPAIRED_8.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def _fit_ridge_multi(X: np.ndarray, Y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardized multi-output ridge regression with unpenalized intercept."""
    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    Xn = (X - x_mean) / x_std
    y_mean = np.mean(Y, axis=0)
    Yc = Y - y_mean
    p = Xn.shape[1]
    A = Xn.T @ Xn + float(alpha) * np.eye(p, dtype=np.float64)
    L = np.linalg.cholesky(0.5 * (A + A.T))
    W = np.linalg.solve(L.T, np.linalg.solve(L, Xn.T @ Yc))
    return W, x_mean, x_std, y_mean


def _predict_ridge_multi(X: np.ndarray, W: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray, y_mean: np.ndarray) -> np.ndarray:
    Xn = (X - x_mean) / x_std
    return Xn @ W + y_mean


def _apply_psd_floor(S: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Apply §3.3 prior PSD floor: max(1e-8 * mean_eig, 1e-12). Returns (S_psd, floor_val, min_eig_before)."""
    S_sym = 0.5 * (S + S.T)
    eigs, Q = np.linalg.eigh(S_sym)
    min_before = float(eigs[0])
    mean_eig = max(float(np.mean(np.maximum(eigs, 0.0))), 1e-12)
    floor_val = max(1e-8 * mean_eig, 1e-12)
    eigs_c = np.maximum(eigs, floor_val)
    S_psd = (Q * eigs_c[None, :]) @ Q.T
    S_psd = 0.5 * (S_psd + S_psd.T)
    return S_psd, floor_val, min_before


def _solve_spd_right(S: np.ndarray, B: np.ndarray, jitter: float = 1e-12) -> tuple[np.ndarray, float]:
    """Solve S @ W = B for SPD matrix S using Cholesky; adds solver_jitter only if needed (or fixed jitter).
    Returns (W, actual_jitter_used).
    """
    k = S.shape[0]
    S_sym = 0.5 * (S + S.T)
    used_jit = float(jitter)
    for mult in (1.0, 10.0, 100.0, 1e4):
        try:
            L = np.linalg.cholesky(S_sym + (used_jit * mult) * np.eye(k, dtype=np.float64))
            W = np.linalg.solve(L.T, np.linalg.solve(L, B))
            return W, used_jit * mult
        except np.linalg.LinAlgError:
            continue
    # Fallback eigensolve if numerically semi-definite
    eigs, Q = np.linalg.eigh(S_sym)
    jit_f = max(used_jit * 1e6, 1e-10 * max(float(eigs[-1]), 1.0))
    W = Q @ ((Q.T @ B) / np.maximum(eigs, jit_f)[:, None])
    return W, jit_f


@dataclass
class FullOperatorPrior:
    d: int                    # 896
    r: int                    # 38
    rank_R: int               # 64
    W_M: np.ndarray           # [p_phi, d*r]
    x_mean: np.ndarray        # [p_phi]
    x_std: np.ndarray         # [p_phi]
    y_mean: np.ndarray        # [d*r]
    alpha: float
    C: np.ndarray             # [r, r], trace(C) == r
    eta: float
    U_e: np.ndarray           # [d, 64], orthonormal columns
    lam_e: np.ndarray         # [64], > 0
    lam_perp: float           # > 0
    s_A_sq: float             # frozen DEV-fit mean squared L2 response scale per dark direction
    psd_floor_C: float
    psd_floor_R: float
    solver_jitter: float
    near_zero_norm_thresh: float  # frozen on DEV-tune (e.g. 10th percentile of ||y_true||_2)
    dev_diagnostics: dict[str, Any]


def predict_M0(prior: FullOperatorPrior, phi_c: np.ndarray) -> np.ndarray:
    """Predict full 896x38 active operator mean M_0(c) strictly from baseline source features phi_c [53]."""
    vec = _predict_ridge_multi(phi_c[None, :], prior.W_M, prior.x_mean, prior.x_std, prior.y_mean)[0]
    return vec.reshape(prior.d, prior.r).astype(np.float64)


def condition_exact_batch(
    prior: FullOperatorPrior,
    M_0: np.ndarray,               # [d, r]
    X_cols: np.ndarray,            # [r, k]
    Y_cols: np.ndarray,            # [d, k]  (Y = A X = Z - V X)
    context_id: str,
    evidence_ids: list[str],
    jitter_override: float | None = None,
) -> dict[str, Any]:
    """Exact batch deterministic Gaussian conditioning (§4.1):
        M_k = M_0 + (Y - M_0 X) (X^T C X)^{-1} X^T C
        C_k = C - C X (X^T C X)^{-1} X^T C
    Never re-adds a positive floor to C_k!
    """
    k = int(X_cols.shape[1])
    C = prior.C
    jit = prior.solver_jitter if jitter_override is None else float(jitter_override)

    if k == 0:
        M_k = M_0.astype(np.float64).copy()
        C_k = C.astype(np.float64).copy()
        used_jit = 0.0
        sup_res_frob = 0.0
        sup_res_rel = 0.0
        ck_sup_norm = 0.0
        innov_frob = 0.0
    else:
        X = np.asarray(X_cols, dtype=np.float64)
        Y = np.asarray(Y_cols, dtype=np.float64)
        CX = C @ X                          # [r, k]
        S = X.T @ CX                        # [k, k]
        # Solve S @ W = X^T C  -> W in [k, r]
        W, used_jit = _solve_spd_right(S, CX.T, jitter=jit)
        Innov = Y - M_0 @ X                 # [d, k]
        M_k = M_0 + Innov @ W               # [d, r]
        C_k = C - CX @ W                    # [r, r]
        C_k = 0.5 * (C_k + C_k.T)

        res_mat = Y - M_k @ X
        sup_res_frob = float(np.linalg.norm(res_mat, "fro"))
        y_frob = max(float(np.linalg.norm(Y, "fro")), 1e-12)
        sup_res_rel = sup_res_frob / y_frob
        ck_sup_norm = float(np.linalg.norm(C_k @ X, "fro"))
        innov_frob = float(np.linalg.norm(Innov, "fro"))

    min_eig_ck = float(np.linalg.eigvalsh(C_k)[0])
    # Deterministic posterior_id bound to exact float64 M_k & C_k & evidence_ids!
    h = hashlib.sha256()
    h.update(f"{context_id}|k={k}|ev={','.join(evidence_ids)}".encode("utf-8"))
    h.update(np.ascontiguousarray(M_k, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(C_k, dtype=np.float64).tobytes())
    pid = f"{context_id}:k{k}:{h.hexdigest()[:14]}"

    return {
        "posterior_id": pid,
        "k": k,
        "evidence_ids": list(evidence_ids),
        "M_k": M_k,
        "C_k": C_k,
        "used_jitter": float(used_jit),
        "support_residual_frob": sup_res_frob,
        "support_residual_rel": sup_res_rel,
        "C_k_support_norm": ck_sup_norm,
        "min_eig_C_k": min_eig_ck,
        "innovation_frob": innov_frob,
    }


def condition_exact_sequential_check(
    prior: FullOperatorPrior,
    M_0: np.ndarray,
    X_cols: np.ndarray,
    Y_cols: np.ndarray,
    jitter: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent sequential 1-by-1 Gaussian conditioning (for §4.1 numerical equivalence check ONLY)."""
    M = M_0.astype(np.float64).copy()
    C = prior.C.astype(np.float64).copy()
    k = X_cols.shape[1]
    for j in range(k):
        x = X_cols[:, j]
        y = Y_cols[:, j]
        cx = C @ x
        denom = float(np.dot(x, cx)) + jitter
        gain = cx / denom
        innov = y - M @ x
        M = M + np.outer(innov, gain)
        C = C - np.outer(cx, gain)
        C = 0.5 * (C + C.T)
    return M, C


def condition_patch_only(
    M_0: np.ndarray,      # [d, r]
    X_cols: np.ndarray,   # [r, k] orthonormal columns!
    Y_cols: np.ndarray,   # [d, k]
) -> np.ndarray:
    """PATCH_k (§4.3): M_patch = M_0 + (Y - M_0 X) X^T.
    Only modifies the observed span(X); strictly identical to M_0 on Q_perp!
    """
    X = np.asarray(X_cols, dtype=np.float64)
    Y = np.asarray(Y_cols, dtype=np.float64)
    return M_0 + (Y - M_0 @ X) @ X.T


def make_mispaired_Y(
    M_0: np.ndarray,         # [d, r]
    X_cols: np.ndarray,      # [r, k]
    Y_cols: np.ndarray,      # [d, k]
    S_mispair: np.ndarray,   # [k, k] pre-frozen non-identity signed permutation!
) -> np.ndarray:
    """MISPAIRED_8 (§4.3): Y_fake = M_0 X + (Y - M_0 X) S."""
    return M_0 @ X_cols + (Y_cols - M_0 @ X_cols) @ S_mispair


def structured_R_quad_form(prior: FullOperatorPrior, g: np.ndarray) -> float:
    """Exact g^T R g = sum_m lam_m (u_m^T g)^2 + lam_perp ||g - U_e U_e^T g||^2."""
    g64 = np.asarray(g, dtype=np.float64)
    proj = prior.U_e.T @ g64
    rem = g64 - prior.U_e @ proj
    return float(np.sum((proj ** 2) * prior.lam_e) + prior.lam_perp * np.sum(rem ** 2))


def structured_R_inv_quad_form(prior: FullOperatorPrior, e: np.ndarray) -> float:
    """Exact e^T R^{-1} e = sum_m (u_m^T e)^2 / lam_m + ||e - U_e U_e^T e||^2 / lam_perp (§6.3)."""
    e64 = np.asarray(e, dtype=np.float64)
    proj = prior.U_e.T @ e64
    rem = e64 - prior.U_e @ proj
    return float(np.sum((proj ** 2) / prior.lam_e) + np.sum(rem ** 2) / prior.lam_perp)


def project_function_from_posterior(
    prior: FullOperatorPrior,
    V_dark_np: np.ndarray,         # [d, r]
    M_k: np.ndarray,               # [d, r]
    C_k: np.ndarray,               # [r, r]
    g_vec: np.ndarray,             # [d]
    temp_t: float = 1.0,
) -> dict[str, Any]:
    """Project ANY endpoint functional gradient g onto frozen (M_k, C_k) (§4.2).
    Does NOT modify M_k, C_k, or posterior_id!
    """
    g64 = np.asarray(g_vec, dtype=np.float64)
    a_hat = (V_dark_np + M_k).T @ g64
    gRg = structured_R_quad_form(prior, g64)
    Cov_a = (float(temp_t) * gRg) * C_k
    Cov_a = 0.5 * (Cov_a + Cov_a.T)
    return {
        "a_hat": a_hat,
        "g_R_g": gRg,
        "Cov_a": Cov_a,
    }


def predict_scalar_direction_dist(
    a_hat: np.ndarray,       # [r]
    Cov_a: np.ndarray,       # [r, r]
    q_vec: np.ndarray,       # [r]
) -> tuple[float, float]:
    """Return (mean = a_hat^T q, std = sqrt(max(q^T Cov_a q, 0)))."""
    q = np.asarray(q_vec, dtype=np.float64)
    mean = float(np.dot(a_hat, q))
    var = max(float(q @ Cov_a @ q), 0.0)
    return mean, math.sqrt(var)


def fit_cal_temperature(
    prior: FullOperatorPrior,
    cal_errors_E: np.ndarray,    # [n_cal, 8, d]  e_{c,j} = M_8 q_j - y_true
    cal_denoms_qCq: np.ndarray,  # [n_cal, 8]     q_j^T C_{c,8} q_j
    denom_floor: float = 1e-12,
) -> dict[str, Any]:
    """Fit single global temperature t > 0 on CAL32 using §6.3 per-direction moment formula:
        t = (1 / (n_cal * n_q * d)) * sum_{c,j} (e_{c,j}^T R^{-1} e_{c,j}) / (q_j^T C_{c,8} q_j)
    """
    n_cal, n_q, d = cal_errors_E.shape
    ratios = np.zeros((n_cal, n_q), dtype=np.float64)
    quads = np.zeros((n_cal, n_q), dtype=np.float64)
    n_clamped = 0
    for c in range(n_cal):
        for j in range(n_q):
            q_val = structured_R_inv_quad_form(prior, cal_errors_E[c, j])
            quads[c, j] = q_val
            den = float(cal_denoms_qCq[c, j])
            if den < denom_floor:
                den = denom_floor
                n_clamped += 1
            ratios[c, j] = q_val / (den * d)
    t_val = float(np.mean(ratios))
    assert math.isfinite(t_val) and t_val > 0.0, f"Invalid CAL temperature t={t_val}"
    return {
        "temperature_t": t_val,
        "n_cal_contexts": n_cal,
        "n_q_perp": n_q,
        "d_dim": d,
        "denom_floor_applied": denom_floor,
        "n_denoms_clamped": n_clamped,
        "min_raw_denom": float(np.min(cal_denoms_qCq)),
        "median_raw_denom": float(np.median(cal_denoms_qCq)),
        "median_per_dir_ratio": float(np.median(ratios)),
        "max_per_dir_ratio": float(np.max(ratios)),
    }


def _build_prior_from_oof(
    W_M: np.ndarray, xm: np.ndarray, xs: np.ndarray, ym: np.ndarray, alpha: float,
    E_oof: np.ndarray,       # [N_fit, d, r]
    eta: float,
    rank_R: int,
    s_A_sq: float,
    solver_jitter: float,
) -> FullOperatorPrior:
    N_fit, d, r = E_oof.shape
    # Raw empirical input second moment C_emp = (1 / (N_fit * d)) sum_c E_c^T E_c
    C_emp = np.zeros((r, r), dtype=np.float64)
    for i in range(N_fit):
        C_emp += E_oof[i].T @ E_oof[i]
    C_emp /= (N_fit * d)
    tr_C = float(np.trace(C_emp))
    C_shrunk = (1.0 - float(eta)) * C_emp + float(eta) * (tr_C / r) * np.eye(r, dtype=np.float64)
    C_psd, floor_C, _ = _apply_psd_floor(C_shrunk)
    # Normalize trace(C) == r so scale lives cleanly in R (§3.3)
    C_norm = C_psd * (float(r) / float(np.trace(C_psd)))
    C_norm = 0.5 * (C_norm + C_norm.T)

    # Estimate R_raw = (1 / (N_fit * r)) sum_c E_c C^{-1} E_c^T using Cholesky solve (NO explicit inverse!)
    L_C = np.linalg.cholesky(C_norm)
    R_raw = np.zeros((d, d), dtype=np.float64)
    for i in range(N_fit):
        # Solve L_C @ Z = E_c^T  -> Z^T Z = E_c C^{-1} E_c^T!
        Z = np.linalg.solve(L_C, E_oof[i].T)
        R_raw += Z.T @ Z
    R_raw /= (N_fit * r)
    R_psd, floor_R, min_eig_R_raw = _apply_psd_floor(R_raw)

    eigs_R, Q_R = np.linalg.eigh(R_psd)
    # Descending order
    idx_desc = np.argsort(eigs_R)[::-1]
    eigs_desc = eigs_R[idx_desc]
    Q_desc = Q_R[:, idx_desc]
    U_e = Q_desc[:, :rank_R].copy()
    # Canonicalize U_e signs deterministically
    max_idx = np.argmax(np.abs(U_e), axis=0)
    signs = np.sign(U_e[max_idx, np.arange(rank_R)])
    signs[signs == 0] = 1.0
    U_e = U_e * signs[None, :]
    lam_e = np.maximum(eigs_desc[:rank_R], floor_R)
    lam_perp = float(max(np.mean(eigs_desc[rank_R:]), floor_R))

    return FullOperatorPrior(
        d=d, r=r, rank_R=rank_R,
        W_M=W_M, x_mean=xm, x_std=xs, y_mean=ym, alpha=float(alpha),
        C=C_norm, eta=float(eta),
        U_e=U_e, lam_e=lam_e, lam_perp=lam_perp,
        s_A_sq=float(s_A_sq), psd_floor_C=float(floor_C), psd_floor_R=float(floor_R),
        solver_jitter=float(solver_jitter), near_zero_norm_thresh=0.0,
        dev_diagnostics={"min_eig_R_raw_before_floor": min_eig_R_raw},
    )


def fit_and_select_full_operator_prior_on_dev(
    phi_fit: np.ndarray,         # [192, 53]
    A_fit: np.ndarray,           # [192, 896, 38]
    phi_tune: np.ndarray,        # [32, 53]
    A_tune: np.ndarray,          # [32, 896, 38]
    tune_banks_3: np.ndarray,    # [3, 32, 38, 38] three pre-fixed orthogonal direction banks!
    cfg: dict[str, Any],
) -> tuple[FullOperatorPrior, dict[str, Any]]:
    """Exact §3.3 DEV-fit192 / DEV-tune32 fitting & selection strictly by k=0 & k=8 orthogonal full-response error."""
    pm_cfg = cfg["prior_model"]
    alpha_grid = [float(a) for a in pm_cfg["ridge_alpha_grid"]]
    eta_grid = [float(e) for e in pm_cfg["cov_eta_grid"]]
    n_folds = int(pm_cfg["oof_folds"])
    rank_R = int(pm_cfg["output_cov_rank"])
    jitter = float(pm_cfg["solver_jitter"])
    oof_seed = int(cfg["derived_seeds"]["dev_oof_folds"])

    N_fit, d, r = A_fit.shape
    N_tune = A_tune.shape[0]
    Y_fit_flat = A_fit.reshape(N_fit, d * r).astype(np.float64)
    s_A_sq = float(np.mean(np.sum(A_fit.astype(np.float64) ** 2, axis=1)))  # (1/(N_fit*r)) sum_c ||A_c||_F^2

    # Precompute 5-fold context-grouped splits on DEV-fit192
    rng = np.random.default_rng(oof_seed)
    folds = np.array_split(rng.permutation(N_fit), n_folds)

    # Freeze near-zero response norm threshold on DEV-tune32 (10th percentile of ||A q||_2 over Q_perp)
    tune_ynorms = []
    for b_i in range(tune_banks_3.shape[0]):
        for c_i in range(N_tune):
            Q_p = tune_banks_3[b_i, c_i, :, 16:24]  # [38, 8]
            Y_p = A_tune[c_i].astype(np.float64) @ Q_p
            tune_ynorms.extend(np.linalg.norm(Y_p, axis=0).tolist())
    near_zero_thresh = float(np.percentile(tune_ynorms, 10))

    candidates_log: list[dict[str, Any]] = []
    best_score = float("inf")
    best_prior: FullOperatorPrior | None = None
    best_entry: dict[str, Any] = {}

    for alpha in alpha_grid:
        # Fit full DEV-fit192 ridge & 5-fold OOF residuals
        W_M, xm, xs, ym = _fit_ridge_multi(phi_fit.astype(np.float64), Y_fit_flat, alpha)
        E_oof = np.zeros((N_fit, d, r), dtype=np.float64)
        for f_i in range(n_folds):
            va_f = folds[f_i]
            tr_f = np.concatenate([folds[j] for j in range(n_folds) if j != f_i])
            wf, xmf, xsf, ymf = _fit_ridge_multi(phi_fit[tr_f].astype(np.float64), Y_fit_flat[tr_f], alpha)
            pred_f = _predict_ridge_multi(phi_fit[va_f].astype(np.float64), wf, xmf, xsf, ymf).reshape(-1, d, r)
            E_oof[va_f] = A_fit[va_f].astype(np.float64) - pred_f

        # Predict M_0 on DEV-tune32 once per alpha
        M0_tune = _predict_ridge_multi(phi_tune.astype(np.float64), W_M, xm, xs, ym).reshape(N_tune, d, r)
        # k=0 orthogonal error across all 3 banks
        l0_vals = []
        for b_i in range(tune_banks_3.shape[0]):
            for c_i in range(N_tune):
                Q_p = tune_banks_3[b_i, c_i, :, 16:24]
                diff0 = (M0_tune[c_i] - A_tune[c_i].astype(np.float64)) @ Q_p
                l0_vals.append(float(np.mean(np.sum(diff0 ** 2, axis=0)) / s_A_sq))
        mean_l0 = float(np.mean(l0_vals))

        for eta in eta_grid:
            prior_cand = _build_prior_from_oof(W_M, xm, xs, ym, alpha, E_oof, eta, rank_R, s_A_sq, jitter)
            prior_cand.near_zero_norm_thresh = near_zero_thresh
            # Evaluate k=8 strictly orthogonal error on DEV-tune32 across all 3 pre-fixed banks!
            l8_vals = []
            for b_i in range(tune_banks_3.shape[0]):
                for c_i in range(N_tune):
                    X8 = tune_banks_3[b_i, c_i, :, 0:8]
                    Q_p = tune_banks_3[b_i, c_i, :, 16:24]
                    Ac = A_tune[c_i].astype(np.float64)
                    Y8 = Ac @ X8
                    st8 = condition_exact_batch(prior_cand, M0_tune[c_i], X8, Y8, f"TUNE_{c_i}", [f"b{b_i}"])
                    diff8 = (st8["M_k"] - Ac) @ Q_p
                    l8_vals.append(float(np.mean(np.sum(diff8 ** 2, axis=0)) / s_A_sq))
            mean_l8 = float(np.mean(l8_vals))
            score = 0.5 * (mean_l0 + mean_l8)
            entry = {
                "alpha": alpha, "eta": eta,
                "tune_L0_perp_norm": mean_l0, "tune_L8_perp_norm": mean_l8,
                "tune_avg_L0_L8": score,
                "tune_rel_reduction_k8_vs_k0": float((mean_l0 - mean_l8) / max(mean_l0, 1e-12)),
                "tune_L0_raw_l2_sq": mean_l0 * s_A_sq,
                "tune_L8_raw_l2_sq": mean_l8 * s_A_sq,
                "C_cond_number": float(np.linalg.cond(prior_cand.C)),
                "R_top1_eig": float(prior_cand.lam_e[0]),
                "R_lam_perp": float(prior_cand.lam_perp),
            }
            candidates_log.append(entry)
            if score < best_score:
                best_score = score
                best_prior = prior_cand
                best_entry = entry

    assert best_prior is not None
    best_prior.dev_diagnostics["selected_candidate"] = best_entry
    best_prior.dev_diagnostics["near_zero_response_norm_p10_tune"] = near_zero_thresh
    sel_report = {
        "n_dev_fit": N_fit, "n_dev_tune": N_tune, "n_tune_banks": int(tune_banks_3.shape[0]),
        "s_A_sq_frozen": s_A_sq,
        "near_zero_norm_thresh_frozen_p10_tune": near_zero_thresh,
        "selected": best_entry,
        "all_candidates": candidates_log,
    }
    return best_prior, sel_report


def save_full_operator_prior_npz(path: Any, prior: FullOperatorPrior, V_dark_np: np.ndarray, U_hybrid_64_old: np.ndarray) -> None:
    np.savez_compressed(
        path,
        d=np.array(prior.d), r=np.array(prior.r), rank_R=np.array(prior.rank_R),
        W_M=prior.W_M.astype(np.float64), x_mean=prior.x_mean.astype(np.float64),
        x_std=prior.x_std.astype(np.float64), y_mean=prior.y_mean.astype(np.float64),
        alpha=np.array(prior.alpha), C=prior.C.astype(np.float64), eta=np.array(prior.eta),
        U_e=prior.U_e.astype(np.float64), lam_e=prior.lam_e.astype(np.float64),
        lam_perp=np.array(prior.lam_perp), s_A_sq=np.array(prior.s_A_sq),
        psd_floor_C=np.array(prior.psd_floor_C), psd_floor_R=np.array(prior.psd_floor_R),
        solver_jitter=np.array(prior.solver_jitter),
        near_zero_norm_thresh=np.array(prior.near_zero_norm_thresh),
        V_dark=V_dark_np.astype(np.float64),
        U_hybrid_64_old=U_hybrid_64_old.astype(np.float64),
    )


def load_full_operator_prior_npz(path: Any) -> tuple[FullOperatorPrior, np.ndarray, np.ndarray]:
    z = np.load(path)
    p = FullOperatorPrior(
        d=int(z["d"]), r=int(z["r"]), rank_R=int(z["rank_R"]),
        W_M=z["W_M"], x_mean=z["x_mean"], x_std=z["x_std"], y_mean=z["y_mean"],
        alpha=float(z["alpha"]), C=z["C"], eta=float(z["eta"]),
        U_e=z["U_e"], lam_e=z["lam_e"], lam_perp=float(z["lam_perp"]),
        s_A_sq=float(z["s_A_sq"]), psd_floor_C=float(z["psd_floor_C"]),
        psd_floor_R=float(z["psd_floor_R"]), solver_jitter=float(z["solver_jitter"]),
        near_zero_norm_thresh=float(z["near_zero_norm_thresh"]), dev_diagnostics={},
    )
    return p, z["V_dark"], z["U_hybrid_64_old"]


def verify_full_operator_synth_fp64(seed: int = 2026112004) -> dict[str, Any]:
    """Synthetic FP64 check (§9.2): support interpolation, C_k X == 0, sequential==batch, PATCH invariance on Q_perp,
    and R^{-1} exact inverse check — all must be <= 1e-9!
    """
    rng = np.random.default_rng(seed)
    d_s, r_s, k_s = 40, 12, 5
    F_c = rng.normal(size=(20, r_s)); C_s = F_c.T @ F_c / 20.0 + 0.1 * np.eye(r_s)
    C_s = C_s * (r_s / np.trace(C_s))
    Q_u, _ = np.linalg.qr(rng.normal(size=(d_s, 10)))
    lam_e = np.linspace(3.0, 0.5, 10)
    lam_p = 0.2
    prior_s = FullOperatorPrior(
        d=d_s, r=r_s, rank_R=10,
        W_M=np.zeros((1, d_s * r_s)), x_mean=np.zeros(1), x_std=np.ones(1), y_mean=np.zeros(d_s * r_s),
        alpha=1.0, C=C_s, eta=0.2, U_e=Q_u, lam_e=lam_e, lam_perp=lam_p,
        s_A_sq=1.0, psd_floor_C=1e-12, psd_floor_R=1e-12, solver_jitter=0.0,
        near_zero_norm_thresh=0.1, dev_diagnostics={},
    )
    M0 = rng.normal(size=(d_s, r_s))
    A_true = M0 + 0.5 * rng.normal(size=(d_s, r_s))
    B, _ = np.linalg.qr(rng.normal(size=(r_s, r_s)))
    X = B[:, :k_s]
    Q_perp = B[:, k_s : k_s + 4]
    Y = A_true @ X

    st = condition_exact_batch(prior_s, M0, X, Y, "SYNTH", [f"e{i}" for i in range(k_s)], jitter_override=0.0)
    M_seq, C_seq = condition_exact_sequential_check(prior_s, M0, X, Y, jitter=0.0)
    M_pat = condition_patch_only(M0, X, Y)

    # Check R * R^{-1} identity via dense R
    R_dense = Q_u @ np.diag(lam_e) @ Q_u.T + lam_p * (np.eye(d_s) - Q_u @ Q_u.T)
    g = rng.normal(size=d_s)
    e = rng.normal(size=d_s)
    r_quad_err = abs(structured_R_quad_form(prior_s, g) - float(g @ R_dense @ g)) / max(abs(float(g @ R_dense @ g)), 1.0)
    r_inv_err = abs(structured_R_inv_quad_form(prior_s, e) - float(e @ np.linalg.solve(R_dense, e))) / max(abs(float(e @ np.linalg.solve(R_dense, e))), 1.0)

    # Projection identity: a_hat^T x_j == g^T (V x_j + y_j) for support x_j!
    V_s = rng.normal(size=(d_s, r_s))
    proj = project_function_from_posterior(prior_s, V_s, st["M_k"], st["C_k"], g)
    proj_sup_err = float(np.max(np.abs(proj["a_hat"] @ X - g @ (V_s @ X + Y))) / max(float(np.linalg.norm(g @ (V_s @ X + Y))), 1.0))

    res = {
        "support_interp_rel_err": st["support_residual_rel"],
        "C_k_support_norm": st["C_k_support_norm"],
        "batch_vs_seq_M_rel_err": float(np.linalg.norm(st["M_k"] - M_seq, "fro") / max(float(np.linalg.norm(M_seq, "fro")), 1.0)),
        "batch_vs_seq_C_rel_err": float(np.linalg.norm(st["C_k"] - C_seq, "fro") / max(float(np.linalg.norm(C_seq, "fro")), 1.0)),
        "patch_q_perp_invariance_err": float(np.linalg.norm((M_pat - M0) @ Q_perp, "fro")),
        "proj_support_identity_rel_err": proj_sup_err,
        "structured_R_quad_rel_err": r_quad_err,
        "structured_R_inv_quad_rel_err": r_inv_err,
    }
    res["passed_1e9"] = bool(max(res.values()) <= 1e-9)
    return res
