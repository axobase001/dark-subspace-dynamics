"""Target/Option Readouts, Identical Min-Norm Closed-Ball Control Rule, Matched Batch=1 Evaluator & Stratified Bootstrap (§6.3, §7)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from src.spectral_basis import compute_target_grad_vjp
from src.transport import run_segment_layers, non_target_kl_divergence, js_divergence_full


def extract_v04_readouts(
    logits: torch.Tensor,            # [V]
    h_L: torch.Tensor,               # [d] pre final RMSNorm
    W_U: torch.Tensor,               # [V, d]
    gamma32: torch.Tensor,           # [d] float32
    rms_eps: float,
    option_token_ids: dict[str, int],
    target_option: str,
) -> dict[str, Any]:
    """Construct primary target F0 (target - highest wrong option) and the 3 fixed option contrasts
    (z_A - z_D, z_B - z_D, z_C - z_D) using exact FP32 RMSNorm VJP (§6.3).
    """
    opt_ids = {o: int(option_token_ids[o]) for o in ("A", "B", "C", "D")}
    wrong_opts = [o for o in ("A", "B", "C", "D") if o != target_option]
    foil_opt = max(wrong_opts, key=lambda o: (float(logits[opt_ids[o]].item()), -ord(o)))

    g0, ana0 = compute_target_grad_vjp(h_L, W_U, opt_ids[target_option], opt_ids[foil_opt], gamma32, rms_eps)
    f0 = {
        "feature_id": "F0",
        "label": f"{target_option}_minus_{foil_opt}",
        "tid_a": opt_ids[target_option],
        "tid_b": opt_ids[foil_opt],
        "g": g0.detach().to(torch.float64).cpu().numpy(),
        "g_norm": float(torch.linalg.vector_norm(g0.to(torch.float64)).item()),
        "analytic_rel_diff": float(ana0),
        "baseline_margin": float((logits[opt_ids[target_option]] - logits[opt_ids[foil_opt]]).item()),
    }

    opt3 = []
    for oa in ("A", "B", "C"):
        g_o, ana_o = compute_target_grad_vjp(h_L, W_U, opt_ids[oa], opt_ids["D"], gamma32, rms_eps)
        opt3.append({
            "feature_id": f"OPT_{oa}_minus_D",
            "label": f"{oa}_minus_D",
            "tid_a": opt_ids[oa],
            "tid_b": opt_ids["D"],
            "g": g_o.detach().to(torch.float64).cpu().numpy(),
            "g_norm": float(torch.linalg.vector_norm(g_o.to(torch.float64)).item()),
            "analytic_rel_diff": float(ana_o),
            "baseline_margin": float((logits[opt_ids[oa]] - logits[opt_ids["D"]]).item()),
        })

    return {
        "target_option": target_option,
        "foil_option": foil_opt,
        "F0": f0,
        "opt3": opt3,
        "max_analytic_rel_diff": max([ana0] + [x["analytic_rel_diff"] for x in opt3]),
    }


def solve_min_norm_ball_action(
    a_hat: np.ndarray,       # [r]
    tau: float,
    rho_c: float,
    zero_rel_tol: float = 1e-9,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Identical simple minimum-norm closed-ball control rule (§7.1):
        x = min(tau / ||a_hat||, rho_c) * (a_hat / ||a_hat||)
    """
    a = np.asarray(a_hat, dtype=np.float64).reshape(-1)
    r = a.shape[0]
    a_norm = float(np.linalg.norm(a))
    if (not np.all(np.isfinite(a))) or a_norm <= zero_rel_tol * max(float(tau) / max(rho_c, 1e-12), 1.0):
        return np.zeros(r, dtype=np.float64), {
            "is_zero_a": True,
            "hit_boundary_unreachable": True,
            "a_hat_norm": a_norm,
            "step_norm": 0.0,
            "pred_delta_F0": 0.0,
            "tau": float(tau),
            "rho_c": float(rho_c),
        }
    req_norm = float(tau) / a_norm
    unreachable = bool(req_norm > rho_c)
    step_norm = min(req_norm, float(rho_c))
    x = step_norm * (a / a_norm)
    return x, {
        "is_zero_a": False,
        "hit_boundary_unreachable": unreachable,
        "a_hat_norm": a_norm,
        "step_norm": float(step_norm),
        "pred_delta_F0": float(np.dot(a, x)),
        "tau": float(tau),
        "rho_c": float(rho_c),
    }


def evaluate_v04_intervention_batch1(
    model: torch.nn.Module,
    h_start: torch.Tensor,          # [1, S, d]
    h_final_base: torch.Tensor,     # [d]
    logits_base: torch.Tensor,      # [V]
    start_layer: int,
    end_layer: int,
    t_source: int,
    V_in: torch.Tensor,             # [d, r]
    x_vec: np.ndarray,              # [r]
    readouts_v04: dict[str, Any],
    option_token_ids: dict[str, int],
    target_option: str,
    tau: float,
    target_tol: float = 0.10,
) -> dict[str, Any]:
    """Exact matched batch=1 nonlinear forward evaluation for BAYES-DYN-04 (§7.2–§7.3)."""
    device = h_start.device
    dtype = h_start.dtype
    x_arr = np.asarray(x_vec, dtype=np.float64).reshape(-1)
    x_norm = float(np.linalg.norm(x_arr))
    opt_ids = [int(option_token_ids[o]) for o in ("A", "B", "C", "D")]
    t_opt_id = int(option_token_ids[target_option])

    delta = torch.tensor(x_arr, dtype=dtype, device=device) @ V_in.to(dtype=dtype, device=device).T
    h_in = h_start.detach().clone()
    h_in[0, t_source, :] = h_in[0, t_source, :] + delta

    with torch.no_grad():
        h_L_seq = run_segment_layers(model, h_in, start_layer, end_layer)
        h_L_pert = h_L_seq[0, t_source, :]
        logits_pert = model.lm_head(model.model.norm(h_L_seq[:, t_source : t_source + 1, :]))[0, 0, :]

    f0 = readouts_v04["F0"]
    base_F0 = float((logits_base[f0["tid_a"]] - logits_base[f0["tid_b"]]).item())
    pert_F0 = float((logits_pert[f0["tid_a"]] - logits_pert[f0["tid_b"]]).item())
    dF0 = pert_F0 - base_F0
    track_err = abs(dF0 - float(tau)) / max(abs(float(tau)), 1e-12)
    within_tol = int(track_err <= target_tol)

    d_opt3 = []
    for r_o in readouts_v04["opt3"]:
        bo = float((logits_base[r_o["tid_a"]] - logits_base[r_o["tid_b"]]).item())
        po = float((logits_pert[r_o["tid_a"]] - logits_pert[r_o["tid_b"]]).item())
        d_opt3.append(po - bo)

    dh = (h_L_pert - h_final_base).detach().to(torch.float64).cpu().numpy()
    dh_l2 = float(np.linalg.norm(dh))
    dh_sq = float(np.sum(dh ** 2))
    g0 = np.asarray(f0["g"], dtype=np.float64)
    gb0 = g0 / max(float(np.linalg.norm(g0)), 1e-12)
    dh_perp = dh - float(np.dot(gb0, dh)) * gb0
    D_geom = float(np.sum(dh_perp ** 2))

    nt_kl = non_target_kl_divergence(logits_base, logits_pert, opt_ids)
    js_full = js_divergence_full(logits_base, logits_pert)
    base_pred = max(opt_ids, key=lambda tid: float(logits_base[tid].item()))
    pert_pred = max(opt_ids, key=lambda tid: float(logits_pert[tid].item()))

    return {
        "actual_delta_F0": float(dF0),
        "target_tracking_err": float(track_err),
        "within_10pct": within_tol,
        "actual_delta_opt3": [float(v) for v in d_opt3],
        "actual_dh_l2": dh_l2,
        "actual_dh_sq": dh_sq,
        "actual_D_geom": D_geom,
        "non_target_kl": float(nt_kl),
        "full_vocab_js": float(js_full),
        "step_norm": x_norm,
        "base_4choice_correct": int(base_pred == t_opt_id),
        "pert_4choice_correct": int(pert_pred == t_opt_id),
        "actual_dh_vec": dh,  # caller may drop before JSONL serialization!
    }


def family_stratified_paired_bootstrap(
    vals_a: np.ndarray,
    vals_b: np.ndarray | None,
    families: list[str],
    n_resamples: int = 20000,
    seed: int = 2026112049,
) -> dict[str, Any]:
    """Family-stratified context-paired bootstrap (20,000 resamples, §6.1)."""
    a = np.asarray(vals_a, dtype=np.float64).reshape(-1)
    d = a if vals_b is None else (a - np.asarray(vals_b, dtype=np.float64).reshape(-1))
    n = len(d)
    fams_uniq = sorted(set(families))
    fam_indices = [np.array([i for i, f in enumerate(families) if f == fm], dtype=np.int64) for fm in fams_uniq]

    rng = np.random.default_rng(seed)
    boot_sums = np.zeros(n_resamples, dtype=np.float64)
    for idxs in fam_indices:
        nf = len(idxs)
        draws = rng.integers(0, nf, size=(n_resamples, nf))
        boot_sums += np.sum(d[idxs][draws], axis=1)
    boot_means = boot_sums / float(n)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "std": float(np.std(d)),
        "ci95": [float(lo), float(hi)],
        "n_obs": int(n),
        "n_resamples": int(n_resamples),
        "ci95_strictly_positive": bool(lo > 0.0),
        "ci95_strictly_negative": bool(hi < 0.0),
        "improved_frac": float(np.mean(d > 0.0)),
        "worsened_frac": float(np.mean(d < 0.0)),
    }
