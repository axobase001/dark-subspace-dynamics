"""Cross-Architecture Spectral Decomposition of W_U and Exact Autograd Endpoint Readouts (BAYES-DYN-05)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class SpectralBasisBundleV05:
    singular_values: torch.Tensor     # [d] float32
    V_full: torch.Tensor              # [d, d] float32 (CPU, move slices as needed)
    V_dark: torch.Tensor              # [d, r] float32 (bottom r singular vectors)
    V_bright4: torch.Tensor           # [d, 4] float32 (top 4 bright directions)
    V_legacy12: torch.Tensor          # [d, 12] float32 (4 bright, 4 mid, 4 extreme dark)
    dark_indices: list[int]
    n_dark_probe: int                 # r = 38
    n_bright_99: int
    n_dark_99: int
    rms_gamma: torch.Tensor           # [d] float32
    rms_eps: float
    diagnostics: dict[str, Any]


def canonicalize_columns(V: torch.Tensor) -> torch.Tensor:
    V_out = V.clone()
    max_idx = torch.argmax(torch.abs(V_out), dim=0)
    cols = torch.arange(V_out.shape[1], device=V_out.device)
    signs = torch.sign(V_out[max_idx, cols])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return V_out * signs.unsqueeze(0)


def rmsnorm_vjp_analytic(h: torch.Tensor, w: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    d = h.shape[-1]
    ms = torch.mean(h * h, dim=-1, keepdim=True) + eps
    s = torch.sqrt(ms)
    gw = gamma * w
    dot_hgw = torch.sum(h * gw, dim=-1, keepdim=True)
    return gw / s - h * (dot_hgw / (d * ms * s))


def compute_spectral_basis_pre_move(
    model_cpu: torch.nn.Module,
    gpu_device: torch.device,
    cfg: dict[str, Any],
) -> SpectralBasisBundleV05:
    """Compute exact full FP64 SVD of lm_head.weight BEFORE moving full model to GPU,
    so the empty GPU runs FP64 SVD cleanly in ~2-5s with zero peak-VRAM collision on 7B/8B!
    """
    W_U_cpu = model_cpu.lm_head.weight.detach().cpu()
    W_E_cpu = model_cpu.model.embed_tokens.weight.detach().cpu()
    gamma_cpu = model_cpu.model.norm.weight.detach().cpu().to(torch.float32)
    rms_eps = float(getattr(model_cpu.config, "rms_norm_eps", getattr(model_cpu.config, "layer_norm_eps", 1e-6)))

    same_ptr = (model_cpu.lm_head.weight.data_ptr() == model_cpu.model.embed_tokens.weight.data_ptr())
    tie_cfg = bool(getattr(model_cpu.config, "tie_word_embeddings", False))

    W_U64_cpu = W_U_cpu.to(torch.float64)
    W_E64_cpu = W_E_cpu.to(torch.float64)
    if W_U64_cpu.shape == W_E64_cpu.shape:
        diff_norm = float(torch.linalg.matrix_norm(W_U64_cpu - W_E64_cpu).item())
        wu_norm = float(torch.linalg.matrix_norm(W_U64_cpu).item())
        rel_we_wu_diff = diff_norm / max(wu_norm, 1e-12)
    else:
        rel_we_wu_diff = float("nan")
    del W_E64_cpu, W_E_cpu

    # Use empty GPU for fast exact FP64 thin SVD if free VRAM >= 3.5x matrix bytes
    svd_dev = torch.device("cpu")
    if gpu_device.type == "cuda":
        torch.cuda.empty_cache()
        free_b, _ = torch.cuda.mem_get_info(gpu_device)
        need_b = int(W_U64_cpu.numel() * 8 * 3.5)
        if free_b >= need_b:
            svd_dev = gpu_device

    W_svd = W_U64_cpu.to(device=svd_dev)
    del W_U64_cpu
    _, S64, Vh64 = torch.linalg.svd(W_svd, full_matrices=False)
    del W_svd
    V64 = canonicalize_columns(Vh64.T).cpu()
    S64 = S64.cpu()
    del Vh64
    if gpu_device.type == "cuda":
        torch.cuda.empty_cache()

    d = int(V64.shape[0])
    ortho_err = float(torch.linalg.matrix_norm(V64.T @ V64 - torch.eye(d, dtype=torch.float64)).item())

    s2 = S64 ** 2
    tot_s2 = torch.sum(s2)
    cum_energy = torch.cumsum(s2, dim=0) / tot_s2

    dark_cfg = cfg["dark_definition"]
    thresh_table: dict[str, Any] = {}
    for th in dark_cfg["report_thresholds"]:
        nb = int(torch.searchsorted(cum_energy, torch.tensor(float(th), dtype=torch.float64)).item()) + 1
        nb = min(nb, d)
        thresh_table[str(th)] = {"n_bright": nb, "n_dark": d - nb, "dark_dim_fraction": float((d - nb) / d)}

    primary_th = float(dark_cfg["bright_energy_threshold"])
    n_bright_99 = int(torch.searchsorted(cum_energy, torch.tensor(primary_th, dtype=torch.float64)).item()) + 1
    n_bright_99 = min(n_bright_99, d)
    n_dark_99 = d - n_bright_99

    r_probe = int(dark_cfg["probe_dim_r"])
    assert r_probe <= d, f"r_probe={r_probe} > d={d}"
    dark_indices = list(range(d - r_probe, d))
    tail_r_energy_frac = float(torch.sum(s2[dark_indices]).item() / float(tot_s2.item()))
    strictly_inside_99_tail = bool(n_dark_99 >= r_probe)

    V_full = V64.to(torch.float32)
    S_full = S64.to(torch.float32)
    V_dark = V_full[:, dark_indices].clone()
    V_bright4 = V_full[:, 0:4].clone()

    mid0 = d // 2 - 2
    leg_idx = [0, 1, 2, 3, mid0, mid0 + 1, mid0 + 2, mid0 + 3, d - 4, d - 3, d - 2, d - 1]
    V_legacy12 = V_full[:, leg_idx].clone()

    p_s = s2 / tot_s2
    entropy = -torch.sum(p_s * torch.log(torch.clamp(p_s, min=1e-30)))
    effective_rank = float(torch.exp(entropy).item())

    diagnostics = {
        "vocab_size": int(W_U_cpu.shape[0]),
        "hidden_size": d,
        "svd_device_used": str(svd_dev),
        "tie_word_embeddings_config": tie_cfg,
        "embed_unembed_same_data_ptr": same_ptr,
        "embed_unembed_relative_frobenius_diff": rel_we_wu_diff,
        "svd_orthonormality_error_fp64": ortho_err,
        "sigma_max": float(S_full[0].item()),
        "sigma_min": float(S_full[-1].item()),
        "sigma_median": float(torch.median(S_full).item()),
        "condition_number": float((S_full[0] / torch.clamp(S_full[-1], min=1e-12)).item()),
        "effective_rank": effective_rank,
        "n_bright_99": n_bright_99,
        "n_dark_99": n_dark_99,
        "dark_99_dim_fraction": float(n_dark_99 / d),
        "probe_dim_r": r_probe,
        "strictly_inside_99_tail": strictly_inside_99_tail,
        "tail_r_energy_frac": tail_r_energy_frac,
        "dark_indices_range": [dark_indices[0], dark_indices[-1]],
        "dark_probe_sigma_max": float(S_full[dark_indices[0]].item()),
        "dark_probe_sigma_min": float(S_full[dark_indices[-1]].item()),
        "threshold_table": thresh_table,
        "rmsnorm_gamma_mean": float(gamma_cpu.mean().item()),
        "rmsnorm_gamma_std": float(gamma_cpu.std().item()),
        "rmsnorm_eps": rms_eps,
        "singular_values_quantiles": {
            "p0": float(S_full[-1].item()),
            "p5": float(torch.quantile(S_full, 0.05).item()),
            "p25": float(torch.quantile(S_full, 0.25).item()),
            "p50": float(torch.quantile(S_full, 0.50).item()),
            "p75": float(torch.quantile(S_full, 0.75).item()),
            "p95": float(torch.quantile(S_full, 0.95).item()),
            "p100": float(S_full[0].item()),
        },
    }

    return SpectralBasisBundleV05(
        singular_values=S_full,
        V_full=V_full,
        V_dark=V_dark,
        V_bright4=V_bright4,
        V_legacy12=V_legacy12,
        dark_indices=dark_indices,
        n_dark_probe=r_probe,
        n_bright_99=n_bright_99,
        n_dark_99=n_dark_99,
        rms_gamma=gamma_cpu,
        rms_eps=rms_eps,
        diagnostics=diagnostics,
    )


def compute_target_grad_vjp_module(
    model: torch.nn.Module,
    h_final: torch.Tensor,
    target_id: int,
    foil_id: int,
    gamma: torch.Tensor,
    rms_eps: float,
) -> tuple[torch.Tensor, float]:
    """Compute exact grad_{h_L} (z_a - z_b) via autograd through the ACTUAL model.model.norm + lm_head,
    works faithfully on Qwen2RMSNorm, Phi3RMSNorm, Olmo2RMSNorm, LlamaRMSNorm, and LayerNorm!
    """
    device = h_final.device
    W_U = model.lm_head.weight.detach()
    h32 = h_final.detach().to(dtype=torch.float32, device=device)
    w_diff = (W_U[target_id] - W_U[foil_id]).detach().to(dtype=torch.float32, device=device)

    h_req = h32.clone().requires_grad_(True)
    normed = model.model.norm(h_req.view(1, 1, -1))[0, 0].to(torch.float32)
    f_val = torch.dot(w_diff, normed)
    (grad_auto,) = torch.autograd.grad(f_val, h_req, create_graph=False)

    g_an = rmsnorm_vjp_analytic(h32, w_diff, gamma.to(device=device, dtype=torch.float32), rms_eps)
    rel_diff = float(
        torch.linalg.vector_norm(grad_auto - g_an).item()
        / max(float(torch.linalg.vector_norm(grad_auto).item()), 1e-12)
    )
    return grad_auto.detach(), rel_diff


def extract_readouts_v05(
    model: torch.nn.Module,
    logits_last: torch.Tensor,
    h_final: torch.Tensor,
    gamma: torch.Tensor,
    rms_eps: float,
    option_token_ids: dict[str, int],
    target_option: str,
) -> dict[str, Any]:
    """Build primary readout F0 (target vs strongest wrong competitor) + 3 fixed option contrasts (A-D, B-D, C-D)."""
    opts = ("A", "B", "C", "D")
    other_opts = [o for o in opts if o != target_option]
    foil_opt = max(other_opts, key=lambda o: float(logits_last[option_token_ids[o]].item()))
    tid_a = int(option_token_ids[target_option])
    tid_b = int(option_token_ids[foil_opt])
    g0_t, rel_an = compute_target_grad_vjp_module(model, h_final, tid_a, tid_b, gamma, rms_eps)
    f0_val = float((logits_last[tid_a] - logits_last[tid_b]).item())

    opt3 = []
    for oa, ob in (("A", "D"), ("B", "D"), ("C", "D")):
        ta, tb = int(option_token_ids[oa]), int(option_token_ids[ob])
        go_t, _ = compute_target_grad_vjp_module(model, h_final, ta, tb, gamma, rms_eps)
        opt3.append({
            "name": f"{oa}-{ob}", "tid_a": ta, "tid_b": tb,
            "base_val": float((logits_last[ta] - logits_last[tb]).item()),
            "g": go_t.detach().to(torch.float64).cpu().numpy(),
        })

    return {
        "F0": {
            "target_option": target_option, "foil_option": foil_opt,
            "tid_a": tid_a, "tid_b": tid_b, "base_val": f0_val,
            "g": g0_t.detach().to(torch.float64).cpu().numpy(),
            "rmsnorm_analytic_rel_diff": rel_an,
        },
        "opt3": opt3,
    }
