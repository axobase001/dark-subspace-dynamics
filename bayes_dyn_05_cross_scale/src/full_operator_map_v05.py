"""Dimension-Agnostic Full-Output Active Propagation Map & Exact Deterministic Gaussian Conditioning (BAYES-DYN-05).

Pure Python + NumPy (zero torch dependency) for identical GPU/CPU/audit execution across d in {896..4096}, r = 38:
  - Full mean regression: phi(c) [53] -> M_0(c) [d, 38] via standardized multi-output ridge.
  - Matrix-normal prior: A_c ~ MN(M_0(c), R, C), trace(C) == r, R = U_e diag(lam_e) U_e^T + lam_perp (I - U_e U_e^T).
  - Grid search over (alpha, eta) evaluates ONLY C (r x r) and computes d x d R eigendecomposition ONCE for winner (20x faster on d=4096, mathematically identical!).
  - Exact deterministic Gaussian conditioning (Cholesky, zero positive floor re-added to C_k).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def _fit_ridge_multi(X: np.ndarray, Y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    eigs, Q = np.linalg.eigh(S_sym)
    jit_f = max(used_jit * 1e6, 1e-10 * max(float(eigs[-1]), 1.0))
    W = Q @ ((Q.T @ B) / np.maximum(eigs, jit_f)[:, None])
    return W, jit_f


@dataclass
class FullOperatorPrior:
    d: int
    r: int
    rank_R: int
    W_M: np.ndarray           # [p_phi, d*r]
    x_mean: np.ndarray        # [p_phi]
    x_std: np.ndarray         # [p_phi]
    y_mean: np.ndarray        # [d*r]
    alpha: float
    C: np.ndarray             # [r, r], trace(C) == r
    eta: float
    U_e: np.ndarray           # [d, rank_R]
    lam_e: np.ndarray         # [rank_R]
    lam_perp: float
    s_A_sq: float
    psd_floor_C: float
    psd_floor_R: float
    solver_jitter: float
    near_zero_norm_thresh: float
    dev_diagnostics: dict[str, Any]


def predict_M0(prior: FullOperatorPrior, phi_c: np.ndarray) -> np.ndarray:
    vec = _predict_ridge_multi(phi_c[None, :], prior.W_M, prior.x_mean, prior.x_std, prior.y_mean)[0]
    return vec.reshape(prior.d, prior.r).astype(np.float64)


def condition_exact_batch_with_C(
    C: np.ndarray,
    M_0: np.ndarray,
    X_cols: np.ndarray,
    Y_cols: np.ndarray,
    context_id: str,
    evidence_ids: list[str],
    jitter: float = 1e-12,
) -> dict[str, Any]:
    k = int(X_cols.shape[1])
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
        CX = C @ X
        S = X.T @ CX
        W, used_jit = _solve_spd_right(S, CX.T, jitter=jitter)
        Innov = Y - M_0 @ X
        M_k = M_0 + Innov @ W
        C_k = C - CX @ W
        C_k = 0.5 * (C_k + C_k.T)

        res_mat = Y - M_k @ X
        sup_res_frob = float(np.linalg.norm(res_mat, "fro"))
        y_frob = max(float(np.linalg.norm(Y, "fro")), 1e-12)
        sup_res_rel = sup_res_frob / y_frob
        ck_sup_norm = float(np.linalg.norm(C_k @ X, "fro"))
        innov_frob = float(np.linalg.norm(Innov, "fro"))

    min_eig_ck = float(np.linalg.eigvalsh(C_k)[0])
    h = hashlib.sha256()
    h.update(f"{context_id}|k={k}|ev={','.join(evidence_ids)}".encode("utf-8"))
    h.update(np.ascontiguousarray(M_k, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(C_k, dtype=np.float64).tobytes())
    pid = f"{context_id}:k{k}:{h.hexdigest()[:14]}"

    return {
        "posterior_id": pid, "k": k, "evidence_ids": list(evidence_ids),
        "M_k": M_k, "C_k": C_k, "used_jitter": float(used_jit),
        "support_residual_frob": sup_res_frob, "support_residual_rel": sup_res_rel,
        "C_k_support_norm": ck_sup_norm, "min_eig_C_k": min_eig_ck, "innovation_frob": innov_frob,
    }


def condition_exact_batch(
    prior: FullOperatorPrior,
    M_0: np.ndarray,
    X_cols: np.ndarray,
    Y_cols: np.ndarray,
    context_id: str,
    evidence_ids: list[str],
    jitter_override: float | None = None,
) -> dict[str, Any]:
    jit = prior.solver_jitter if jitter_override is None else float(jitter_override)
    return condition_exact_batch_with_C(prior.C, M_0, X_cols, Y_cols, context_id, evidence_ids, jitter=jit)


def condition_patch_only(M_0: np.ndarray, X_cols: np.ndarray, Y_cols: np.ndarray) -> np.ndarray:
    """PATCH_k: M_patch = M_0 + (Y - M_0 X) X^T (exact identity on orthogonal Q_perp)."""
    if X_cols.shape[1] == 0:
        return M_0.astype(np.float64).copy()
    X = np.asarray(X_cols, dtype=np.float64)
    Y = np.asarray(Y_cols, dtype=np.float64)
    return M_0 + (Y - M_0 @ X) @ X.T


def make_mispaired_Y(M_0: np.ndarray, X_cols: np.ndarray, Y_cols: np.ndarray, S_mispair: np.ndarray) -> np.ndarray:
    return M_0 @ X_cols + (Y_cols - M_0 @ X_cols) @ S_mispair


def structured_R_quad_form(prior: FullOperatorPrior, g: np.ndarray) -> float:
    g64 = np.asarray(g, dtype=np.float64)
    proj = prior.U_e.T @ g64
    rem = g64 - prior.U_e @ proj
    return float(np.sum((proj ** 2) * prior.lam_e) + prior.lam_perp * np.sum(rem ** 2))


def structured_R_inv_quad_form(prior: FullOperatorPrior, e: np.ndarray) -> float:
    e64 = np.asarray(e, dtype=np.float64)
    proj = prior.U_e.T @ e64
    rem = e64 - prior.U_e @ proj
    return float(np.sum((proj ** 2) / prior.lam_e) + np.sum(rem ** 2) / prior.lam_perp)


def project_function_from_posterior(
    prior: FullOperatorPrior,
    V_dark_np: np.ndarray,
    M_k: np.ndarray,
    C_k: np.ndarray,
    g_vec: np.ndarray,
    temp_t: float = 1.0,
) -> dict[str, Any]:
    g64 = np.asarray(g_vec, dtype=np.float64)
    a_hat = (V_dark_np + M_k).T @ g64
    gRg = structured_R_quad_form(prior, g64)
    Cov_a = (float(temp_t) * gRg) * C_k
    Cov_a = 0.5 * (Cov_a + Cov_a.T)
    return {"a_hat": a_hat, "g_R_g": gRg, "Cov_a": Cov_a}


def predict_scalar_direction_dist(a_hat: np.ndarray, Cov_a: np.ndarray, q_vec: np.ndarray) -> tuple[float, float]:
    q = np.asarray(q_vec, dtype=np.float64)
    mean = float(np.dot(a_hat, q))
    var = max(float(q @ Cov_a @ q), 0.0)
    return mean, math.sqrt(var)


def fit_cal_temperature(
    prior: FullOperatorPrior,
    cal_errors_E: np.ndarray,
    cal_denoms_qCq: np.ndarray,
    denom_floor: float = 1e-12,
) -> dict[str, Any]:
    n_cal, n_q, d = cal_errors_E.shape
    ratios = np.zeros((n_cal, n_q), dtype=np.float64)
    n_clamped = 0
    for c in range(n_cal):
        for j in range(n_q):
            q_val = structured_R_inv_quad_form(prior, cal_errors_E[c, j])
            den = float(cal_denoms_qCq[c, j])
            if den < denom_floor:
                den = denom_floor
                n_clamped += 1
            ratios[c, j] = q_val / (den * d)
    t_val = float(np.mean(ratios))
    assert math.isfinite(t_val) and t_val > 0.0, f"Invalid CAL temperature t={t_val}"
    return {
        "temperature_t": t_val, "n_cal_contexts": n_cal, "n_q_perp": n_q, "d_dim": d,
        "denom_floor_applied": denom_floor, "n_denoms_clamped": n_clamped,
        "min_raw_denom": float(np.min(cal_denoms_qCq)),
        "median_raw_denom": float(np.median(cal_denoms_qCq)),
        "median_per_dir_ratio": float(np.median(ratios)),
        "max_per_dir_ratio": float(np.max(ratios)),
    }


def _estimate_C_from_oof(E_oof: np.ndarray, eta: float) -> tuple[np.ndarray, float]:
    N_fit, d, r = E_oof.shape
    C_emp = np.zeros((r, r), dtype=np.float64)
    for i in range(N_fit):
        C_emp += E_oof[i].T @ E_oof[i]
    C_emp /= (N_fit * d)
    tr_C = float(np.trace(C_emp))
    C_shrunk = (1.0 - float(eta)) * C_emp + float(eta) * (tr_C / r) * np.eye(r, dtype=np.float64)
    C_psd, floor_C, _ = _apply_psd_floor(C_shrunk)
    C_norm = C_psd * (float(r) / float(np.trace(C_psd)))
    C_norm = 0.5 * (C_norm + C_norm.T)
    return C_norm, float(floor_C)


def _estimate_R_from_oof(E_oof: np.ndarray, C_norm: np.ndarray, rank_R: int) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    N_fit, d, r = E_oof.shape
    L_C = np.linalg.cholesky(C_norm)
    R_raw = np.zeros((d, d), dtype=np.float64)
    for i in range(N_fit):
        Z = np.linalg.solve(L_C, E_oof[i].T)
        R_raw += Z.T @ Z
    R_raw /= (N_fit * r)
    R_psd, floor_R, min_eig_raw = _apply_psd_floor(R_raw)
    eigs_R, Q_R = np.linalg.eigh(R_psd)
    idx_desc = np.argsort(eigs_R)[::-1]
    eigs_desc = eigs_R[idx_desc]
    Q_desc = Q_R[:, idx_desc]
    U_e = Q_desc[:, :rank_R].copy()
    max_idx = np.argmax(np.abs(U_e), axis=0)
    signs = np.sign(U_e[max_idx, np.arange(rank_R)])
    signs[signs == 0] = 1.0
    U_e = U_e * signs[None, :]
    lam_e = np.maximum(eigs_desc[:rank_R], floor_R)
    lam_perp = float(max(np.mean(eigs_desc[rank_R:]), floor_R))
    return U_e, lam_e, lam_perp, float(floor_R), float(min_eig_raw)


def build_top64_dynamic_output_basis(A_fit: np.ndarray, rank: int = 64) -> np.ndarray:
    """Build top-64 active output basis U_dyn_64 strictly on DEV-fit for 4-block error breakdown diagnostics."""
    N_fit, d, r = A_fit.shape
    G = np.zeros((d, d), dtype=np.float64)
    for i in range(N_fit):
        Ai = A_fit[i].astype(np.float64)
        G += Ai @ Ai.T
    eigs, Q = np.linalg.eigh(0.5 * (G + G.T))
    order = np.argsort(eigs)[::-1]
    U = Q[:, order[:rank]].copy()
    max_idx = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[max_idx, np.arange(rank)])
    signs[signs == 0] = 1.0
    return U * signs[None, :]


def fit_and_select_full_operator_prior_on_dev(
    phi_fit: np.ndarray,         # [N_fit, p_phi]
    A_fit: np.ndarray,           # [N_fit, d, r]
    phi_tune: np.ndarray,        # [N_tune, p_phi]
    A_tune: np.ndarray,          # [N_tune, d, r]
    tune_banks_3: np.ndarray,    # [3, N_tune, r, r]
    cfg: dict[str, Any],
) -> tuple[FullOperatorPrior, dict[str, Any]]:
    """Fast exact DEV-fit / DEV-tune selection over (alpha, eta):
    Computes C_norm (r x r) during grid search and d x d R eigendecomposition ONCE for winning (alpha, eta)!
    """
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
    s_A_sq = float(np.mean(np.sum(A_fit.astype(np.float64) ** 2, axis=1)))

    rng = np.random.default_rng(oof_seed)
    folds = np.array_split(rng.permutation(N_fit), n_folds)

    tune_ynorms = []
    for b_i in range(tune_banks_3.shape[0]):
        for c_i in range(N_tune):
            Q_p = tune_banks_3[b_i, c_i, :, 16:24]
            Y_p = A_tune[c_i].astype(np.float64) @ Q_p
            tune_ynorms.extend(np.linalg.norm(Y_p, axis=0).tolist())
    near_zero_thresh = float(np.percentile(tune_ynorms, 10))

    candidates_log: list[dict[str, Any]] = []
    best_score = float("inf")
    best_pack: tuple[Any, ...] | None = None
    best_entry: dict[str, Any] = {}

    for alpha in alpha_grid:
        W_M, xm, xs, ym = _fit_ridge_multi(phi_fit.astype(np.float64), Y_fit_flat, alpha)
        E_oof = np.zeros((N_fit, d, r), dtype=np.float64)
        for f_i in range(n_folds):
            va_f = folds[f_i]
            tr_f = np.concatenate([folds[j] for j in range(n_folds) if j != f_i])
            wf, xmf, xsf, ymf = _fit_ridge_multi(phi_fit[tr_f].astype(np.float64), Y_fit_flat[tr_f], alpha)
            pred_f = _predict_ridge_multi(phi_fit[va_f].astype(np.float64), wf, xmf, xsf, ymf).reshape(-1, d, r)
            E_oof[va_f] = A_fit[va_f].astype(np.float64) - pred_f

        M0_tune = _predict_ridge_multi(phi_tune.astype(np.float64), W_M, xm, xs, ym).reshape(N_tune, d, r)
        l0_vals = []
        for b_i in range(tune_banks_3.shape[0]):
            for c_i in range(N_tune):
                Q_p = tune_banks_3[b_i, c_i, :, 16:24]
                diff0 = (M0_tune[c_i] - A_tune[c_i].astype(np.float64)) @ Q_p
                l0_vals.append(float(np.mean(np.sum(diff0 ** 2, axis=0)) / s_A_sq))
        mean_l0 = float(np.mean(l0_vals))

        for eta in eta_grid:
            C_norm, floor_C = _estimate_C_from_oof(E_oof, eta)
            l8_vals = []
            for b_i in range(tune_banks_3.shape[0]):
                for c_i in range(N_tune):
                    X8 = tune_banks_3[b_i, c_i, :, 0:8]
                    Q_p = tune_banks_3[b_i, c_i, :, 16:24]
                    Ac = A_tune[c_i].astype(np.float64)
                    Y8 = Ac @ X8
                    st8 = condition_exact_batch_with_C(C_norm, M0_tune[c_i], X8, Y8, f"TUNE_{c_i}", [f"b{b_i}"], jitter=jitter)
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
                "C_cond_number": float(np.linalg.cond(C_norm)),
            }
            candidates_log.append(entry)
            if score < best_score:
                best_score = score
                best_entry = entry
                best_pack = (W_M, xm, xs, ym, alpha, E_oof.copy(), C_norm, floor_C, eta)

    assert best_pack is not None
    W_M, xm, xs, ym, alpha, E_oof_best, C_norm, floor_C, eta = best_pack
    U_e, lam_e, lam_perp, floor_R, min_eig_R_raw = _estimate_R_from_oof(E_oof_best, C_norm, rank_R)

    best_prior = FullOperatorPrior(
        d=d, r=r, rank_R=rank_R,
        W_M=W_M, x_mean=xm, x_std=xs, y_mean=ym, alpha=float(alpha),
        C=C_norm, eta=float(eta),
        U_e=U_e, lam_e=lam_e, lam_perp=lam_perp,
        s_A_sq=float(s_A_sq), psd_floor_C=float(floor_C), psd_floor_R=float(floor_R),
        solver_jitter=float(jitter), near_zero_norm_thresh=near_zero_thresh,
        dev_diagnostics={"min_eig_R_raw_before_floor": min_eig_R_raw},
    )
    best_entry["R_top1_eig"] = float(lam_e[0])
    best_entry["R_lam_perp"] = float(lam_perp)

    report = {
        "n_fit": N_fit, "n_tune": N_tune, "d": d, "r": r,
        "s_A_sq": s_A_sq, "near_zero_norm_thresh_p10": near_zero_thresh,
        "selected": best_entry, "all_candidates": candidates_log,
    }
    return best_prior, report


def save_full_operator_prior_npz(path: Any, prior: FullOperatorPrior, V_dark: np.ndarray, U_dyn64: np.ndarray) -> None:
    np.savez_compressed(
        path,
        d=np.array(prior.d), r=np.array(prior.r), rank_R=np.array(prior.rank_R),
        W_M=prior.W_M.astype(np.float32), x_mean=prior.x_mean, x_std=prior.x_std, y_mean=prior.y_mean.astype(np.float32),
        alpha=np.array(prior.alpha), C=prior.C, eta=np.array(prior.eta),
        U_e=prior.U_e, lam_e=prior.lam_e, lam_perp=np.array(prior.lam_perp),
        s_A_sq=np.array(prior.s_A_sq), psd_floor_C=np.array(prior.psd_floor_C),
        psd_floor_R=np.array(prior.psd_floor_R), solver_jitter=np.array(prior.solver_jitter),
        near_zero_norm_thresh=np.array(prior.near_zero_norm_thresh),
        V_dark=V_dark.astype(np.float64), U_dyn64=U_dyn64.astype(np.float64),
    )


def load_full_operator_prior_npz(path: Any) -> tuple[FullOperatorPrior, np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    prior = FullOperatorPrior(
        d=int(z["d"]), r=int(z["r"]), rank_R=int(z["rank_R"]),
        W_M=z["W_M"].astype(np.float64), x_mean=z["x_mean"].astype(np.float64),
        x_std=z["x_std"].astype(np.float64), y_mean=z["y_mean"].astype(np.float64),
        alpha=float(z["alpha"]), C=z["C"].astype(np.float64), eta=float(z["eta"]),
        U_e=z["U_e"].astype(np.float64), lam_e=z["lam_e"].astype(np.float64), lam_perp=float(z["lam_perp"]),
        s_A_sq=float(z["s_A_sq"]), psd_floor_C=float(z["psd_floor_C"]),
        psd_floor_R=float(z["psd_floor_R"]), solver_jitter=float(z["solver_jitter"]),
        near_zero_norm_thresh=float(z["near_zero_norm_thresh"]), dev_diagnostics={},
    )
    return prior, z["V_dark"].astype(np.float64), z["U_dyn64"].astype(np.float64)


def verify_full_operator_synth_fp64(seed: int = 2026112005) -> dict[str, Any]:
    """Exact synthetic FP64 conditioning self-test (<= 1e-9)."""
    rng = np.random.default_rng(seed)
    d, r, k = 64, 16, 6
    G = rng.normal(size=(r, r))
    C = G.T @ G + 0.5 * np.eye(r)
    C = C * (r / np.trace(C))
    Q_bank, _ = np.linalg.qr(rng.normal(size=(r, r)))
    X = Q_bank[:, :k]
    Qp = Q_bank[:, k : k + 4]
    M0 = rng.normal(size=(d, r))
    A_true = M0 + rng.normal(scale=0.5, size=(d, r))
    Y = A_true @ X

    st_b = condition_exact_batch_with_C(C, M0, X, Y, "SYNTH", [f"s{i}" for i in range(k)], jitter=0.0)
    # Sequential check
    Mc, Cc = M0.copy(), C.copy()
    for j in range(k):
        xj = X[:, j : j + 1]
        yj = Y[:, j : j + 1]
        cx = Cc @ xj
        s = float((xj.T @ cx)[0, 0])
        Mc = Mc + (yj - Mc @ xj) @ (cx.T / s)
        Cc = 0.5 * ((Cc - (cx @ cx.T) / s) + (Cc - (cx @ cx.T) / s).T)
    m_diff = float(np.linalg.norm(st_b["M_k"] - Mc, "fro") / max(np.linalg.norm(Mc, "fro"), 1e-12))
    c_diff = float(np.linalg.norm(st_b["C_k"] - Cc, "fro") / max(np.linalg.norm(Cc, "fro"), 1e-12))
    M_patch = condition_patch_only(M0, X, Y)
    patch_inv = float(np.linalg.norm((M_patch - M0) @ Qp, "fro"))
    passed = bool(
        st_b["support_residual_rel"] <= 1e-9
        and st_b["C_k_support_norm"] <= 1e-9
        and m_diff <= 1e-9 and c_diff <= 1e-9 and patch_inv <= 1e-9
    )
    return {
        "support_interp_rel_err": st_b["support_residual_rel"],
        "C_k_support_norm": st_b["C_k_support_norm"],
        "batch_vs_seq_M_rel_err": m_diff,
        "batch_vs_seq_C_rel_err": c_diff,
        "patch_q_perp_invariance_err": patch_inv,
        "passed_1e9": passed,
    }
