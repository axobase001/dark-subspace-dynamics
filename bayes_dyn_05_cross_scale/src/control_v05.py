"""Minimum-Norm Closed-Ball Control Solver & Matched batch=1 Nonlinear Forward Evaluator (BAYES-DYN-05)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


Z_SCORES = {0.50: 0.6744897501960817, 0.80: 1.2815515655446004, 0.90: 1.6448536269514722, 0.95: 1.959963984540054}


def scalar_gaussian_nll(y_true: float, mu: float, sd: float, sd_floor: float = 1e-8) -> float:
    s = max(float(sd), sd_floor)
    return 0.5 * math.log(2.0 * math.pi * s * s) + 0.5 * ((float(y_true) - float(mu)) / s) ** 2


def solve_min_norm_ball_action(
    a_hat: np.ndarray,
    tau: float,
    rho_c: float,
    zero_tol: float = 1e-8,
) -> tuple[np.ndarray, dict[str, Any]]:
    a = np.asarray(a_hat, dtype=np.float64).reshape(-1)
    a_norm = float(np.linalg.norm(a))
    if a_norm <= zero_tol:
        return np.zeros_like(a), {
            "is_zero_a": True, "hit_boundary_unreachable": True,
            "a_hat_norm": a_norm, "step_norm": 0.0, "pred_delta_F0": 0.0,
            "tau": float(tau), "rho_c": float(rho_c),
        }
    req_norm = float(tau) / a_norm
    unreachable = bool(req_norm > float(rho_c))
    step_norm = min(req_norm, float(rho_c))
    x = step_norm * (a / a_norm)
    return x, {
        "is_zero_a": False, "hit_boundary_unreachable": unreachable,
        "a_hat_norm": a_norm, "step_norm": float(step_norm),
        "pred_delta_F0": float(np.dot(a, x)), "tau": float(tau), "rho_c": float(rho_c),
    }


def evaluate_v05_intervention_batch1(
    model: Any,
    h_start: Any,
    h_final_base: Any,
    logits_base: Any,
    start_layer: int,
    end_layer: int,
    t_source: int,
    V_in: Any,
    x_vec: np.ndarray,
    readouts: dict[str, Any],
    option_token_ids: dict[str, int],
    target_option: str,
    tau: float,
    target_tol: float = 0.10,
) -> dict[str, Any]:
    import torch
    from src.transport_v05 import run_segment_layers

    device = h_start.device
    dtype = h_start.dtype
    x_arr = np.asarray(x_vec, dtype=np.float64).reshape(-1)
    opt_ids = [int(option_token_ids[o]) for o in ("A", "B", "C", "D")]
    t_opt_id = int(option_token_ids[target_option])

    delta = torch.tensor(x_arr, dtype=dtype, device=device) @ V_in.to(dtype=dtype, device=device).T
    h_in = h_start.detach().clone()
    h_in[0, t_source, :] = h_in[0, t_source, :] + delta

    with torch.no_grad():
        h_L_seq = run_segment_layers(model, h_in, start_layer, end_layer)
        h_L_pert = h_L_seq[0, t_source, :]
        logits_pert = model.lm_head(model.model.norm(h_L_seq[:, t_source : t_source + 1, :]))[0, 0, :]

    f0 = readouts["F0"]
    base_F0 = float((logits_base[f0["tid_a"]] - logits_base[f0["tid_b"]]).item())
    pert_F0 = float((logits_pert[f0["tid_a"]] - logits_pert[f0["tid_b"]]).item())
    dF0 = pert_F0 - base_F0
    track_err = abs(dF0 - float(tau)) / max(abs(float(tau)), 1e-12)
    within_tol = int(track_err <= target_tol)

    dh = (h_L_pert - h_final_base).detach().to(torch.float64).cpu().numpy()
    dh_l2 = float(np.linalg.norm(dh))
    h_base_l2 = max(float(torch.linalg.vector_norm(h_final_base.to(torch.float64)).item()), 1e-12)
    d_geom = dh_l2 / h_base_l2

    # Non-target vocab KL (masking out A/B/C/D option tokens)
    mask_nt = torch.ones(logits_base.shape[0], dtype=torch.bool, device=device)
    for oid in opt_ids:
        mask_nt[oid] = False
    lb_nt = logits_base[mask_nt].to(torch.float32)
    lp_nt = logits_pert[mask_nt].to(torch.float32)
    log_pb = torch.log_softmax(lb_nt, dim=-1)
    log_pp = torch.log_softmax(lp_nt, dim=-1)
    pb = torch.exp(log_pb)
    nt_kl = float(torch.sum(pb * (log_pb - log_pp)).item())

    base_4 = logits_base[opt_ids]
    pert_4 = logits_pert[opt_ids]
    base_pred = opt_ids[int(torch.argmax(base_4).item())]
    pert_pred = opt_ids[int(torch.argmax(pert_4).item())]

    return {
        "base_F0": base_F0, "pert_F0": pert_F0, "actual_delta_F0": dF0,
        "target_tracking_err": track_err, "within_10pct": within_tol,
        "actual_dh_l2": dh_l2, "actual_D_geom": d_geom, "non_target_kl": nt_kl,
        "base_4choice_correct": int(base_pred == t_opt_id),
        "pert_4choice_correct": int(pert_pred == t_opt_id),
    }
