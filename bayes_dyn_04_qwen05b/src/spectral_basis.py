"""Spectral decomposition of W_U, exact RMSNorm JVP & VJP, and frozen input/output basis builders."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch


@dataclass
class SpectralBasisBundle:
    singular_values: torch.Tensor          # [d] float32
    V_full: torch.Tensor                   # [d, d] float32, right singular vectors (columns)
    V_dark: torch.Tensor                   # [d, r_dark] float32, tail after 99% cum squared energy
    dark_indices: list[int]                # 0-based indices into 0..d-1
    n_bright: int
    n_dark: int
    V_legacy12: torch.Tensor               # [d, 12] float32 (4 bright, 4 middle, 4 extreme dark)
    V_bright_matched: torch.Tensor         # [d, r_dark] float32 top r_dark bright directions
    V_middle_matched: torch.Tensor         # [d, r_dark] float32 middle r_dark directions
    rms_gamma: torch.Tensor                # [d] float32
    rms_eps: float
    diagnostics: dict[str, Any]


def canonicalize_columns(V: torch.Tensor) -> torch.Tensor:
    """Fix SVD/QR column sign indeterminacy so each column's largest-magnitude entry is positive."""
    V_out = V.clone()
    max_idx = torch.argmax(torch.abs(V_out), dim=0)
    cols = torch.arange(V_out.shape[1], device=V_out.device)
    signs = torch.sign(V_out[max_idx, cols])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return V_out * signs.unsqueeze(0)


def canonicalize_columns_np(V: np.ndarray) -> np.ndarray:
    """NumPy version of column sign canonicalization."""
    V_out = V.copy()
    if V_out.size == 0:
        return V_out
    max_idx = np.argmax(np.abs(V_out), axis=0)
    cols = np.arange(V_out.shape[1])
    signs = np.sign(V_out[max_idx, cols])
    signs[signs == 0] = 1.0
    return V_out * signs[None, :]


def rmsnorm_forward(h: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """Exact RMSNorm N(h) = gamma * h / sqrt(mean(h^2) + eps)."""
    ms = torch.mean(h * h, dim=-1, keepdim=True) + eps
    s = torch.sqrt(ms)
    return gamma * (h / s)


def rmsnorm_jvp(h: torch.Tensor, v: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """Exact forward Jacobian-vector product J_N(h) @ v:
        J_N v = (gamma * v) / s - (gamma * h) * (h^T v) / (d * s^3)
    """
    d = h.shape[-1]
    ms = torch.mean(h * h, dim=-1, keepdim=True) + eps
    s = torch.sqrt(ms)
    dot_hv = torch.sum(h * v, dim=-1, keepdim=True)
    return (gamma * v) / s - (gamma * h) * (dot_hv / (d * ms * s))


def rmsnorm_vjp(h: torch.Tensor, w: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """Exact transpose Jacobian-vector product (VJP) J_N(h)^T @ w:
        J_N^T w = (gamma * w) / s - h * (h^T (gamma * w)) / (d * s^3)
    Required for target logit gradient grad_h(z_a - z_b) = J_N(h)^T (W_U[a] - W_U[b]).
    """
    d = h.shape[-1]
    ms = torch.mean(h * h, dim=-1, keepdim=True) + eps
    s = torch.sqrt(ms)
    gw = gamma * w
    dot_hgw = torch.sum(h * gw, dim=-1, keepdim=True)
    return gw / s - h * (dot_hgw / (d * ms * s))


def compute_target_grad_vjp(
    h_final: torch.Tensor,
    W_U: torch.Tensor,
    target_id: int,
    foil_id: int,
    gamma: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, float]:
    """Compute exact grad_{h_L} (z_a - z_b) via autograd VJP on RMSNorm + W_U in FP32
    (matching Qwen2RMSNorm's internal FP32 variance calculation), and cross-check against analytic rmsnorm_vjp.
    Returns (grad_h_L [d] in float32, analytic_rel_diff).
    """
    device = h_final.device
    h32 = h_final.detach().to(dtype=torch.float32, device=device)
    g32 = gamma.detach().to(dtype=torch.float32, device=device)
    w_diff = (W_U[target_id] - W_U[foil_id]).detach().to(dtype=torch.float32, device=device)
    h_req = h32.clone().requires_grad_(True)
    normed = rmsnorm_forward(h_req, g32, eps)
    f_val = torch.dot(w_diff, normed)
    (grad_auto,) = torch.autograd.grad(f_val, h_req, create_graph=False)

    grad_analytic = rmsnorm_vjp(h32, w_diff, g32, eps)
    rel_diff = float(
        torch.linalg.vector_norm(grad_auto - grad_analytic).item()
        / max(float(torch.linalg.vector_norm(grad_auto).item()), 1e-12)
    )
    return grad_auto.detach(), rel_diff


def compute_spectral_basis(
    model: torch.nn.Module,
    config: dict[str, Any],
    assert_expected_05b: bool = True,
) -> SpectralBasisBundle:
    """Compute full FP64 SVD of W_U, derive dark tail strictly from checkpoint spectrum,
    and build frozen spectral basis matrices.
    """
    device = model.lm_head.weight.device
    W_U = model.lm_head.weight.detach()
    W_E = model.model.embed_tokens.weight.detach()
    gamma = model.model.norm.weight.detach().to(torch.float32)
    rms_eps = float(getattr(model.config, "rms_norm_eps", 1e-6))

    same_ptr = (W_U.data_ptr() == W_E.data_ptr())
    tie_cfg = bool(getattr(model.config, "tie_word_embeddings", False))
    # Compute embedding/unembedding diff on CPU in FP64 to avoid double 4+ GB GPU allocations on 7B models
    W_U_cpu64 = W_U.detach().cpu().to(torch.float64)
    W_E_cpu64 = W_E.detach().cpu().to(torch.float64)
    diff_norm = float(torch.linalg.matrix_norm(W_U_cpu64 - W_E_cpu64).item())
    wu_norm = float(torch.linalg.matrix_norm(W_U_cpu64).item())
    rel_we_wu_diff = diff_norm / max(wu_norm, 1e-12)
    del W_E_cpu64

    # Full thin SVD in FP64 on GPU if enough free VRAM, else CPU
    svd_dev = device
    if device.type == "cuda":
        free_b, _ = torch.cuda.mem_get_info(device)
        need_b = int(W_U.numel() * 8 * 2.5)
        if free_b < need_b:
            svd_dev = torch.device("cpu")
    W_U64 = W_U_cpu64.to(device=svd_dev)
    del W_U_cpu64
    _, S64, Vh64 = torch.linalg.svd(W_U64, full_matrices=False)
    del W_U64
    V64 = canonicalize_columns(Vh64.T).to(device=device)  # [d, d]
    S64 = S64.to(device=device)
    del Vh64
    if device.type == "cuda":
        torch.cuda.empty_cache()
    d = int(V64.shape[0])
    ortho_err = float(
        torch.linalg.matrix_norm(V64.T @ V64 - torch.eye(d, dtype=torch.float64, device=device)).item()
    )

    s2 = S64 ** 2
    cum_energy = torch.cumsum(s2, dim=0) / torch.sum(s2)

    # Report all threshold cutoffs
    dark_cfg = config["dark_definition"]
    thresh_table = {}
    for th in dark_cfg["report_thresholds"]:
        # First index where cumulative squared energy >= th (0-based), so n_bright = idx + 1
        nb = int(torch.searchsorted(cum_energy, torch.tensor(float(th), dtype=torch.float64, device=device)).item()) + 1
        nb = min(nb, d)
        thresh_table[str(th)] = {"n_bright": nb, "n_dark": d - nb, "dark_dim_fraction": float((d - nb) / d)}

    primary_th = float(dark_cfg["bright_energy_threshold"])
    n_bright = int(torch.searchsorted(cum_energy, torch.tensor(primary_th, dtype=torch.float64, device=device)).item()) + 1
    n_dark = d - n_bright
    if assert_expected_05b:
        exp_nb = int(dark_cfg["expected_n_bright_05b"])
        exp_nd = int(dark_cfg["expected_n_dark_05b"])
        assert n_bright == exp_nb and n_dark == exp_nd, (
            f"Dark tail mismatch on checkpoint: got n_bright={n_bright}, n_dark={n_dark}; "
            f"expected {exp_nb}/{exp_nd}"
        )

    dark_indices = list(range(n_bright, d))
    V_full = V64.to(torch.float32)
    S_full = S64.to(torch.float32)
    V_dark = V_full[:, dark_indices].clone()

    # Legacy 12 indices (4 bright, 4 middle, 4 extreme dark) adapted to d
    if d == 896:
        leg = config["basis"]["legacy_indices_12"]
        leg_idx = list(leg["bright"]) + list(leg["middle"]) + list(leg["dark"])
    else:
        mid0 = d // 2 - 2
        leg_idx = [0, 1, 2, 3, mid0, mid0 + 1, mid0 + 2, mid0 + 3, d - 4, d - 3, d - 2, d - 1]
    V_legacy12 = V_full[:, leg_idx].clone()

    # Matched-rank bright and middle spectral controls (same dimension as n_dark)
    bright_ctrl_idx = list(range(0, n_dark))
    mid_start = (d - n_dark) // 2
    middle_ctrl_idx = list(range(mid_start, mid_start + n_dark))
    V_bright_matched = V_full[:, bright_ctrl_idx].clone()
    V_middle_matched = V_full[:, middle_ctrl_idx].clone()

    # Effective rank
    p_s = s2 / torch.sum(s2)
    entropy = -torch.sum(p_s * torch.log(torch.clamp(p_s, min=1e-30)))
    effective_rank = float(torch.exp(entropy).item())

    diagnostics = {
        "vocab_size": int(W_U.shape[0]),
        "hidden_size": d,
        "tie_word_embeddings_config": tie_cfg,
        "embed_unembed_same_data_ptr": same_ptr,
        "embed_unembed_relative_frobenius_diff": rel_we_wu_diff,
        "svd_orthonormality_error_fp64": ortho_err,
        "sigma_max": float(S_full[0].item()),
        "sigma_min": float(S_full[-1].item()),
        "sigma_median": float(torch.median(S_full).item()),
        "condition_number": float((S_full[0] / S_full[-1]).item()),
        "effective_rank": effective_rank,
        "primary_bright_energy_threshold": primary_th,
        "n_bright": n_bright,
        "n_dark": n_dark,
        "dark_dim_fraction": float(n_dark / d),
        "dark_indices_range": [dark_indices[0], dark_indices[-1]],
        "dark_sigma_min": float(S_full[dark_indices[-1]].item()),
        "dark_sigma_max": float(S_full[dark_indices[0]].item()),
        "bright_matched_indices_range": [bright_ctrl_idx[0], bright_ctrl_idx[-1]],
        "middle_matched_indices_range": [middle_ctrl_idx[0], middle_ctrl_idx[-1]],
        "legacy_12_indices": leg_idx,
        "threshold_table": thresh_table,
        "rmsnorm_gamma_mean": float(gamma.mean().item()),
        "rmsnorm_gamma_std": float(gamma.std().item()),
        "rmsnorm_gamma_min": float(gamma.min().item()),
        "rmsnorm_gamma_max": float(gamma.max().item()),
        "singular_values_all": [float(x) for x in S_full.detach().cpu().tolist()],
    }

    return SpectralBasisBundle(
        singular_values=S_full,
        V_full=V_full,
        V_dark=V_dark,
        dark_indices=dark_indices,
        n_bright=n_bright,
        n_dark=n_dark,
        V_legacy12=V_legacy12,
        V_bright_matched=V_bright_matched,
        V_middle_matched=V_middle_matched,
        rms_gamma=gamma,
        rms_eps=rms_eps,
        diagnostics=diagnostics,
    )


def build_static_spectral_output_basis(V_full_np: np.ndarray, q: int) -> tuple[np.ndarray, list[int]]:
    """Pre-registered static spectral output basis U_spec,q balanced across bright/middle/dark."""
    d = V_full_np.shape[0]
    if q == 12 and d == 896:
        idx = [0, 1, 2, 3, 446, 447, 448, 449, 892, 893, 894, 895]
    else:
        n_b = (q + 2) // 3
        n_d = (q + 1) // 3
        n_m = q - n_b - n_d
        mid0 = d // 2 - n_m // 2
        idx = list(range(0, n_b)) + list(range(mid0, mid0 + n_m)) + list(range(d - n_d, d))
    U = canonicalize_columns_np(V_full_np[:, idx].astype(np.float64))
    return U, idx


def build_output_bases_from_train(
    A_train: np.ndarray,      # [N_train, d, r]
    V_full_np: np.ndarray,    # [d, d]
    U_anchor12: np.ndarray,   # [d, 12]
    q_list: list[int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build U_spec,q, U_dyn,q, and U_hybrid,q strictly from training active responses A_train."""
    N_tr, d, r = A_train.shape
    # Stack active response columns: [d, N_tr * r]
    A_mat = np.transpose(A_train, (1, 0, 2)).reshape(d, N_tr * r).astype(np.float64)
    # Full left singular vectors via symmetric Gram A_mat @ A_mat.T (d x d, fast & exact in FP64!)
    G_left = A_mat @ A_mat.T
    eigvals, eigvecs = np.linalg.eigh(0.5 * (G_left + G_left.T))
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    U_dyn_full = canonicalize_columns_np(eigvecs[:, order])

    # Orthonormalize anchor12 and compute orthogonal residual active matrix for hybrid basis
    Q_anc, _ = np.linalg.qr(U_anchor12.astype(np.float64))
    Q_anc = canonicalize_columns_np(Q_anc)
    A_rem = A_mat - Q_anc @ (Q_anc.T @ A_mat)
    G_rem = A_rem @ A_rem.T
    ev_r, vec_r = np.linalg.eigh(0.5 * (G_rem + G_rem.T))
    ord_r = np.argsort(ev_r)[::-1]
    U_rem_full = canonicalize_columns_np(vec_r[:, ord_r])

    bases: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {
        "train_active_total_frobenius_sq": float(np.sum(A_mat ** 2)),
        "top_64_active_eigenvalues": [float(x) for x in eigvals[:64]],
        "basis_details": {},
    }

    for q in q_list:
        u_spec, spec_idx = build_static_spectral_output_basis(V_full_np, q)
        u_dyn = U_dyn_full[:, :q].copy()
        if q <= 12:
            u_hyb = Q_anc[:, :q].copy()
        else:
            # Ensure exact orthonormality between anchor and residual directions
            extra = U_rem_full[:, : (q - 12)]
            extra = extra - Q_anc @ (Q_anc.T @ extra)
            Q_ext, _ = np.linalg.qr(extra)
            u_hyb = np.concatenate([Q_anc, canonicalize_columns_np(Q_ext)], axis=1)

        bases[f"U_spec_{q}"] = u_spec
        bases[f"U_dyn_{q}"] = u_dyn
        bases[f"U_hybrid_{q}"] = u_hyb

        for bname, U_b in ((f"U_spec_{q}", u_spec), (f"U_dyn_{q}", u_dyn), (f"U_hybrid_{q}", u_hyb)):
            ortho_err = float(np.linalg.norm(U_b.T @ U_b - np.eye(q)))
            train_cov = float(np.sum((U_b.T @ A_mat) ** 2) / max(float(np.sum(A_mat ** 2)), 1e-12))
            info = {
                "rank": q,
                "orthonormality_err": ortho_err,
                "train_active_energy_coverage": train_cov,
            }
            if bname.startswith("U_spec"):
                info["spectral_indices"] = spec_idx
            meta["basis_details"][bname] = info

    return bases, meta


def build_dark_input_subspaces_from_train(
    A_train: np.ndarray,           # [N_tr, d, r_dark]
    h_dark_train: np.ndarray,      # [N_tr, r_dark] source-layer dark projection V_dark^T h_l
    families_train: list[str],
    options_train: list[str],
    pod_rank: int = 16,
    random_seeds: list[int] = (101, 102, 103),
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build coefficient subspace matrices W_sub [r_dark, k] inside Dark38 strictly from train:
      - Dark38: I_{r_dark}
      - Dark-POD16: top-16 right singular vectors of stacked A_train (transport directions)
      - Signed-rank-k: exact numerical rank k from pre-listed training task & option contrasts (NEVER padded with PCA!)
      - Dark-random-k: matched-rank random orthonormal subspaces inside Dark38 across 3 fixed seeds
    """
    N_tr, d, r = A_train.shape
    subspaces: dict[str, np.ndarray] = {"Dark38": np.eye(r, dtype=np.float64)}

    # 1. Dark-POD16: right singular vectors of stacked A_train [N_tr * d, r] via r x r Gram
    G_right = np.zeros((r, r), dtype=np.float64)
    for i in range(N_tr):
        Ai = A_train[i].astype(np.float64)
        G_right += Ai.T @ Ai
    ev_p, vec_p = np.linalg.eigh(0.5 * (G_right + G_right.T))
    ord_p = np.argsort(ev_p)[::-1]
    ev_p = np.maximum(ev_p[ord_p], 0.0)
    W_pod = canonicalize_columns_np(vec_p[:, ord_p[:pod_rank]])
    subspaces[f"Dark_POD{pod_rank}"] = W_pod

    # 2. Signed-rank-k: exact matrix of pre-listed training contrasts in dark projection space
    # Pre-listed contrasts:
    #   - 4 task family centroids minus grand mean (theoretical max rank 3)
    #   - 4 answer option centroids minus grand mean (theoretical max rank 3)
    # Total theoretical max rank <= 6; keep exact numerical rank (sv > 1e-6 * sv_max), zero PCA padding!
    h_mean = np.mean(h_dark_train, axis=0)
    contrast_cols = []
    contrast_names = []
    for fam in sorted(set(families_train)):
        mask = np.array([f == fam for f in families_train])
        contrast_cols.append(np.mean(h_dark_train[mask], axis=0) - h_mean)
        contrast_names.append(f"family_{fam}_vs_grand")
    for opt in sorted(set(options_train)):
        mask = np.array([o == opt for o in options_train])
        contrast_cols.append(np.mean(h_dark_train[mask], axis=0) - h_mean)
        contrast_names.append(f"option_{opt}_vs_grand")

    C_mat = np.stack(contrast_cols, axis=1).astype(np.float64)  # [r, 8]
    U_c, S_c, _ = np.linalg.svd(C_mat, full_matrices=False)
    rank_tol = 1e-6 * max(float(S_c[0]), 1e-12)
    signed_rank = int(np.sum(S_c > rank_tol))
    W_signed = canonicalize_columns_np(U_c[:, :signed_rank])
    subspaces[f"Signed_rank{signed_rank}"] = W_signed

    # 3. Matched-rank random orthonormal subspaces inside Dark38 across 3 fixed seeds
    for k_match in sorted({pod_rank, signed_rank}):
        for seed in random_seeds:
            rng = np.random.default_rng(int(seed) + 1000 * k_match)
            Q_r, _ = np.linalg.qr(rng.normal(size=(r, k_match)))
            subspaces[f"Dark_random_k{k_match}_seed{seed}"] = canonicalize_columns_np(Q_r)

    # Also build per-target-option training static signed direction library for C1 (§7.3)
    # For each option o in {A,B,C,D}: mu_o = mean(h_dark_train[opt==o])
    # So for context with target a and foil b, static direction = (mu_a - mu_b) / ||mu_a - mu_b||
    option_centroids = {}
    for opt in sorted(set(options_train)):
        mask = np.array([o == opt for o in options_train])
        option_centroids[opt] = np.mean(h_dark_train[mask], axis=0).astype(np.float64)

    pod_retention = float(np.sum(ev_p[:pod_rank]) / max(float(np.sum(ev_p)), 1e-12))
    signed_retention = float(np.trace(W_signed.T @ G_right @ W_signed) / max(float(np.sum(ev_p)), 1e-12))

    meta = {
        "dark_rank": r,
        "pod_rank": pod_rank,
        "pod_train_active_energy_retention_within_dark38": pod_retention,
        "pod_top_eigenvalues": [float(x) for x in ev_p[:pod_rank]],
        "signed_contrast_names": contrast_names,
        "signed_contrast_singular_values": [float(x) for x in S_c],
        "signed_true_rank": signed_rank,
        "signed_train_active_energy_retention_within_dark38": signed_retention,
        "option_centroids_dark": {k: v.tolist() for k, v in option_centroids.items()},
    }
    return subspaces, meta
