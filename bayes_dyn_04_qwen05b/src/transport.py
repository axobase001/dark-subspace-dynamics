"""Exact Residual-Stream Transport (JVP / VJP), Feature Extraction, Late-Layer Scan, and Real Forward Perturbations."""

from __future__ import annotations
from typing import Any
import math
import numpy as np
import torch
import torch.autograd.forward_ad as fwAD

from src.spectral_basis import (
    SpectralBasisBundle,
    rmsnorm_forward,
    rmsnorm_jvp,
    compute_target_grad_vjp,
)


def tokenize_chat_prompt(tokenizer: Any, user_prompt: str, device: torch.device) -> dict[str, Any]:
    """Tokenize with Qwen chat template (add_generation_prompt=True) and locate last non-padding prompt token."""
    messages = [
        {"role": "system", "content": "Answer the multiple-choice question with only the single correct capital letter."},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    seq_len = int(input_ids.shape[1])
    t_last = int(attention_mask[0].sum().item()) - 1
    return {
        "formatted_text": text,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "seq_len": seq_len,
        "t_last": t_last,
    }


def resolve_option_token_ids(tokenizer: Any, options: list[str] = ("A", "B", "C", "D")) -> dict[str, int]:
    """Verify each option A/B/C/D encodes cleanly to a single token at the assistant generation boundary."""
    mapping: dict[str, int] = {}
    for opt in options:
        ids = tokenizer.encode(opt, add_special_tokens=False)
        assert len(ids) == 1, f"Option '{opt}' does not encode to a single token: {ids}"
        mapping[opt] = int(ids[0])
    assert len(set(mapping.values())) == len(options), f"Option token ID collision: {mapping}"
    return mapping


def _build_position_embeddings(model: torch.nn.Module, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = hidden_states.shape[1]
    pos_ids = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long).unsqueeze(0)
    return model.model.rotary_emb(hidden_states, pos_ids)


def _build_causal_mask(seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    min_val = torch.finfo(dtype).min
    mask = torch.full((1, 1, seq_len, seq_len), min_val, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


def capture_residual_states(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Capture exact residual stream states under fixed coordinate convention:
      - states[0]  = h_0: post-embedding, pre-block-0
      - states[l]  = h_l: output after completing first l blocks (block l-1)
      - states[L]  = h_L: output after final block (L-1), PRE final RMSNorm
    Returns (states_dict, native_logits_at_last_token [V]).
    """
    num_layers = len(model.model.layers)
    states: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        h = model.model.embed_tokens(input_ids)
        states[0] = h.detach().clone()
        pos_emb = _build_position_embeddings(model, h)
        causal_mask = _build_causal_mask(h.shape[1], h.dtype, h.device)
        for idx in range(num_layers):
            out = model.model.layers[idx](
                h,
                attention_mask=causal_mask,
                position_embeddings=pos_emb,
                use_cache=False,
            )
            h = out[0] if isinstance(out, tuple) else out
            states[idx + 1] = h.detach().clone()

        t_last = int(attention_mask[0].sum().item()) - 1
        h_normed_last = model.model.norm(h[:, t_last : t_last + 1, :])
        logits_last = model.lm_head(h_normed_last)[0, 0, :].detach()
    return states, logits_last


def run_segment_layers(
    model: torch.nn.Module,
    h_start: torch.Tensor,
    start_layer: int,
    end_layer: int,
    causal_mask: torch.Tensor | None = None,
    pos_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Propagate residual state h_start [B, S, d] from layer `start_layer` through `end_layer - 1`
    producing `h_{end_layer}` (pre final RMSNorm if end_layer == num_layers).
    """
    h = h_start
    if pos_emb is None:
        pos_emb = _build_position_embeddings(model, h)
    if causal_mask is None:
        causal_mask = _build_causal_mask(h.shape[1], h.dtype, h.device)
    for idx in range(start_layer, end_layer):
        out = model.model.layers[idx](
            h,
            attention_mask=causal_mask,
            position_embeddings=pos_emb,
            use_cache=False,
        )
        h = out[0] if isinstance(out, tuple) else out
    return h


def compute_exact_jvp_columns_batched(
    model: torch.nn.Module,
    h_start: torch.Tensor,        # [1, S, d]
    start_layer: int,
    end_layer: int,
    t_source: int,
    V_dirs: torch.Tensor,         # [d, r]
    t_target: int | None = None,
    batch_size: int = 19,
) -> torch.Tensor:
    """Compute exact JVP columns Z[:, j] = (d h_{end_layer, t_target} / d h_{start_layer, t_source}) @ V_dirs[:, j]
    using PyTorch forward-mode AD (dual_level) batched over tangent columns!
    Returns Z of shape [d, r] on CPU/device matching h_start.
    """
    if t_target is None:
        t_target = t_source
    device = h_start.device
    dtype = h_start.dtype
    _, S, d = h_start.shape
    r = int(V_dirs.shape[1])
    V_t = V_dirs.to(device=device, dtype=dtype)  # [d, r]

    causal_mask = _build_causal_mask(S, dtype, device)
    pos_emb = _build_position_embeddings(model, h_start)

    cols = []
    with torch.no_grad():
        for start in range(0, r, batch_size):
            end = min(start + batch_size, r)
            b = end - start
            primal = h_start.expand(b, S, d).contiguous()
            tangent = torch.zeros((b, S, d), dtype=dtype, device=device)
            tangent[:, t_source, :] = V_t[:, start:end].T  # [b, d]

            with fwAD.dual_level():
                dual_in = fwAD.make_dual(primal, tangent)
                dual_out = run_segment_layers(
                    model,
                    dual_in,
                    start_layer,
                    end_layer,
                    causal_mask=causal_mask,
                    pos_emb=pos_emb,
                )
                tang_out = fwAD.unpack_dual(dual_out).tangent  # [b, S, d]
                if tang_out is None:
                    tang_out = torch.zeros_like(primal)
                cols.append(tang_out[:, t_target, :].T.detach().clone())  # [d, b]

    return torch.cat(cols, dim=1)  # [d, r]


def compute_exact_jvp_single_functional(
    model: torch.nn.Module,
    h_start: torch.Tensor,
    start_layer: int,
    end_layer: int,
    t_source: int,
    v_dir: torch.Tensor,
    t_target: int | None = None,
) -> torch.Tensor:
    """Reference single-vector JVP via torch.autograd.functional.jvp for P0 cross-checking."""
    if t_target is None:
        t_target = t_source
    device = h_start.device
    dtype = h_start.dtype
    _, S, d = h_start.shape
    causal_mask = _build_causal_mask(S, dtype, device)
    pos_emb = _build_position_embeddings(model, h_start)

    v_full = torch.zeros_like(h_start)
    v_full[0, t_source, :] = v_dir.to(device=device, dtype=dtype)

    def _fn(h_in: torch.Tensor) -> torch.Tensor:
        h_out = run_segment_layers(model, h_in, start_layer, end_layer, causal_mask=causal_mask, pos_emb=pos_emb)
        return h_out[0, t_target, :]

    _, jvp_out = torch.autograd.functional.jvp(_fn, (h_start.detach(),), (v_full,), strict=False)
    return jvp_out.detach()


def compute_exact_vjp_single(
    model: torch.nn.Module,
    h_start: torch.Tensor,
    start_layer: int,
    end_layer: int,
    t_source: int,
    w_dir: torch.Tensor,
    t_target: int | None = None,
) -> torch.Tensor:
    """Compute exact VJP (d h_{end_layer, t_target} / d h_{start_layer, t_source})^T @ w_dir via reverse-mode autograd."""
    if t_target is None:
        t_target = t_source
    device = h_start.device
    dtype = h_start.dtype
    _, S, _ = h_start.shape
    causal_mask = _build_causal_mask(S, dtype, device)
    pos_emb = _build_position_embeddings(model, h_start)

    h_req = h_start.detach().clone().requires_grad_(True)
    h_out = run_segment_layers(model, h_req, start_layer, end_layer, causal_mask=causal_mask, pos_emb=pos_emb)
    scalar = torch.dot(h_out[0, t_target, :], w_dir.to(device=device, dtype=dtype))
    (grad_h,) = torch.autograd.grad(scalar, h_req, create_graph=False)
    return grad_h[0, t_source, :].detach()


def extract_context_features(
    h_ell_t: np.ndarray,       # [d]
    V_in: np.ndarray,          # [d, r]
    V_legacy12: np.ndarray,    # [d, 12]
    seq_len: int,
) -> np.ndarray:
    """Extract source-layer context features phi(c) available strictly from a single baseline forward pass.
    Includes:
      - normalized projection onto V_in [r]
      - normalized projection onto legacy bright/middle/dark [12]
      - log(||h_ell_t||), dark energy fraction, log(seq_len)
    """
    h = h_ell_t.astype(np.float64)
    h_norm = max(float(np.linalg.norm(h)), 1e-12)
    h_unit = h / h_norm
    p_in = V_in.T @ h_unit              # [r]
    p_leg = V_legacy12.T @ h_unit       # [12]
    dark_frac = float(np.sum(p_in ** 2))
    scalars = np.array([math.log(h_norm), dark_frac, math.log(max(seq_len, 1))], dtype=np.float64)
    return np.concatenate([p_in, p_leg, scalars], axis=0)


def extract_target_gradient_features(
    g_c: np.ndarray,           # [d]
    V_in: np.ndarray,          # [d, r]
    U_out: np.ndarray,         # [d, q]
) -> np.ndarray:
    """Extract target gradient features psi(g_c) openly shared with all controllers for E[b_bar | phi, target]."""
    g = g_c.astype(np.float64)
    g_norm = max(float(np.linalg.norm(g)), 1e-12)
    g_bar = g / g_norm
    p_v = V_in.T @ g_bar                # [r]
    p_u = U_out.T @ g_bar               # [q]
    rem_norm = math.sqrt(max(1.0 - float(np.sum(p_u ** 2)), 0.0))
    return np.concatenate([p_v, p_u, np.array([math.log(g_norm), rem_norm], dtype=np.float64)], axis=0)


def non_target_kl_divergence(
    logits_base: torch.Tensor,
    logits_pert: torch.Tensor,
    exclude_token_ids: list[int],
) -> float:
    """Compute KL(p_base || p_pert) strictly over non-target tokens after excluding `exclude_token_ids`
    (the 4 choice tokens A/B/C/D) and renormalizing both distributions.
    """
    lb = logits_base.detach().to(torch.float64).clone()
    lp = logits_pert.detach().to(torch.float64).clone()
    for tid in exclude_token_ids:
        lb[tid] = -1e30
        lp[tid] = -1e30
    log_pb = torch.log_softmax(lb, dim=-1)
    log_pp = torch.log_softmax(lp, dim=-1)
    pb = torch.exp(log_pb)
    # Mask excluded tokens cleanly so 0 * (-inf) never produces NaN
    mask = torch.ones_like(pb, dtype=torch.bool)
    for tid in exclude_token_ids:
        mask[tid] = False
    kl = torch.sum(pb[mask] * (log_pb[mask] - log_pp[mask]))
    return float(max(kl.item(), 0.0))


def js_divergence_full(logits_base: torch.Tensor, logits_pert: torch.Tensor) -> float:
    """Full-vocabulary Jensen-Shannon divergence in nats."""
    pb = torch.softmax(logits_base.detach().to(torch.float64), dim=-1)
    pp = torch.softmax(logits_pert.detach().to(torch.float64), dim=-1)
    m = 0.5 * (pb + pp)
    kl1 = torch.sum(pb * (torch.log(torch.clamp(pb, min=1e-30)) - torch.log(torch.clamp(m, min=1e-30))))
    kl2 = torch.sum(pp * (torch.log(torch.clamp(pp, min=1e-30)) - torch.log(torch.clamp(m, min=1e-30))))
    return float(max(0.5 * (kl1 + kl2).item(), 0.0))


def run_batched_real_perturbations(
    model: torch.nn.Module,
    h_start: torch.Tensor,             # [1, S, d]
    h_final_base: torch.Tensor,        # [d]
    logits_base: torch.Tensor,         # [V]
    start_layer: int,
    end_layer: int,
    t_source: int,
    V_in: torch.Tensor,                # [d, r]
    X_coeffs: np.ndarray,              # [M, r] in physical delta_h = V_in @ x coordinates!
    g_c: np.ndarray,                   # [d]
    target_id: int,
    foil_id: int,
    option_token_ids: dict[str, int],
    target_option: str,
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    """Run REAL nonlinear forward passes for M perturbations delta_h = V_in @ X_coeffs[m]
    at (start_layer, t_source) and evaluate exact Delta_F, target probability, 4-choice accuracy,
    exact full-space geometric drift D_geom = ||P_{perp g} Delta h_L||^2, non-target KL, and JS divergence.
    """
    device = h_start.device
    dtype = h_start.dtype
    _, S, d = h_start.shape
    M = int(X_coeffs.shape[0])
    V_t = V_in.to(device=device, dtype=dtype)
    h_final_b = h_final_base.to(device=device, dtype=dtype)

    g_np = g_c.astype(np.float64)
    g_norm = max(float(np.linalg.norm(g_np)), 1e-12)
    g_bar = g_np / g_norm

    opt_ids_list = [option_token_ids[o] for o in ("A", "B", "C", "D")]
    opt_idx_target = ("A", "B", "C", "D").index(target_option)
    base_opt_logits = logits_base[opt_ids_list].detach().to(torch.float64)
    base_opt_probs = torch.softmax(base_opt_logits, dim=-1)
    base_F = float((logits_base[target_id] - logits_base[foil_id]).item())
    base_p_target = float(base_opt_probs[opt_idx_target].item())
    base_correct = bool(int(torch.argmax(base_opt_logits).item()) == opt_idx_target)

    causal_mask = _build_causal_mask(S, dtype, device)
    pos_emb = _build_position_embeddings(model, h_start)

    results: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, M, batch_size):
            end = min(start + batch_size, M)
            b = end - start
            X_b = torch.tensor(X_coeffs[start:end], dtype=dtype, device=device)  # [b, r]
            deltas = X_b @ V_t.T  # [b, d]

            H_b = h_start.expand(b, S, d).clone()
            H_b[:, t_source, :] = H_b[:, t_source, :] + deltas
            H_L_b = run_segment_layers(model, H_b, start_layer, end_layer, causal_mask=causal_mask, pos_emb=pos_emb)
            h_L_pert = H_L_b[:, t_source, :]  # [b, d]
            dh_L = (h_L_pert - h_final_b.unsqueeze(0)).to(torch.float64).cpu().numpy()  # [b, d]

            h_normed = model.model.norm(h_L_pert.unsqueeze(1))
            logits_b = model.lm_head(h_normed)[:, 0, :]  # [b, V]

            for j in range(b):
                dh = dh_L[j]
                proj_g = float(np.dot(g_bar, dh))
                dh_perp = dh - proj_g * g_bar
                d_geom = float(np.dot(dh_perp, dh_perp))
                dh_norm_sq = float(np.dot(dh, dh))

                log_j = logits_b[j]
                pert_F = float((log_j[target_id] - log_j[foil_id]).item())
                delta_F = pert_F - base_F

                pert_opt_logits = log_j[opt_ids_list].detach().to(torch.float64)
                pert_opt_probs = torch.softmax(pert_opt_logits, dim=-1)
                pert_p_target = float(pert_opt_probs[opt_idx_target].item())
                pert_correct = bool(int(torch.argmax(pert_opt_logits).item()) == opt_idx_target)

                nt_kl = non_target_kl_divergence(logits_base, log_j, opt_ids_list)
                js_d = js_divergence_full(logits_base, log_j)

                results.append({
                    "step_norm": float(np.linalg.norm(X_coeffs[start + j])),
                    "base_F": base_F,
                    "pert_F": pert_F,
                    "actual_delta_F": delta_F,
                    "base_p_target": base_p_target,
                    "pert_p_target": pert_p_target,
                    "delta_p_target": pert_p_target - base_p_target,
                    "base_4choice_correct": base_correct,
                    "pert_4choice_correct": pert_correct,
                    "actual_D_geom": d_geom,
                    "actual_dh_L_norm_sq": dh_norm_sq,
                    "non_target_kl": nt_kl,
                    "full_vocab_js": js_d,
                    "actual_dh_L_vec": dh,  # caller can strip before JSON serialization
                })
    return results


def run_late_layer_growth_and_rmsnorm_audit(
    model: torch.nn.Module,
    tokenizer: Any,
    basis: SpectralBasisBundle,
    train_records: list[dict[str, Any]],
    layers_to_scan: list[int],
    primary_source_layer: int,
    Z_dark_train: np.ndarray,  # [N_tr, d, r_dark] at primary_source_layer -> L
    max_contexts: int = 32,
) -> dict[str, Any]:
    """Audit §8.1:
      - Per-layer dark projection at L0/L4/L10/L16/L20/L21/L22/L23/L24
      - Explicitly separate centered variance, raw L2 norm, per-block residual write norm, and pre- vs post-RMSNorm
      - Same-terminal-state readout comparison: W_U J_N(h_L) (T v) vs W_U J_N(h_L) (I v) vs raw W_U v
    """
    device = next(model.parameters()).device
    V_dark = basis.V_dark.to(device=device, dtype=torch.float32)  # [d, r]
    W_U = model.lm_head.weight.detach().to(torch.float32)
    gamma = basis.rms_gamma.to(device=device, dtype=torch.float32)
    eps = basis.rms_eps
    num_layers = len(model.model.layers)

    n_use = min(max_contexts, len(train_records))
    dark_projs_by_layer: dict[int, list[np.ndarray]] = {l: [] for l in layers_to_scan}
    bright_projs_by_layer: dict[int, list[np.ndarray]] = {l: [] for l in layers_to_scan}
    block_write_dark_norms: dict[int, list[float]] = {l: [] for l in layers_to_scan if l > 0}
    block_write_total_norms: dict[int, list[float]] = {l: [] for l in layers_to_scan if l > 0}
    post_rmsnorm_dark_projs: list[np.ndarray] = []
    post_rmsnorm_bright_projs: list[np.ndarray] = []

    V_bright = basis.V_full[:, : basis.n_bright].to(device=device, dtype=torch.float32)

    # Same-terminal-state readout comparison on first 16 contexts across dark vs bright matched directions
    readout_transport_norms: list[float] = []
    readout_identity_norms: list[float] = []
    readout_raw_wu_norms: list[float] = []
    readout_source_direct_norms: list[float] = []

    with torch.no_grad():
        raw_wu_dark = float(torch.mean(torch.linalg.vector_norm(W_U @ V_dark, dim=0)).item())
        for i in range(n_use):
            tok = tokenize_chat_prompt(tokenizer, train_records[i]["user_prompt"], device)
            states, _ = capture_residual_states(model, tok["input_ids"], tok["attention_mask"])
            t = tok["t_last"]
            for l in layers_to_scan:
                h_l = states[l][0, t, :]
                pd = (V_dark.T @ h_l).cpu().numpy()
                pb = (V_bright.T @ h_l).cpu().numpy()
                dark_projs_by_layer[l].append(pd)
                bright_projs_by_layer[l].append(pb)
                if l > 0:
                    dh_blk = states[l][0, t, :] - states[l - 1][0, t, :]
                    block_write_dark_norms[l].append(float(torch.linalg.vector_norm(V_dark.T @ dh_blk).item()))
                    block_write_total_norms[l].append(float(torch.linalg.vector_norm(dh_blk).item()))

            h_L = states[num_layers][0, t, :]
            h_L_post = rmsnorm_forward(h_L, gamma, eps)
            post_rmsnorm_dark_projs.append((V_dark.T @ h_L_post).cpu().numpy())
            post_rmsnorm_bright_projs.append((V_bright.T @ h_L_post).cpu().numpy())

            if i < 16:
                Z_i = torch.tensor(Z_dark_train[i], dtype=torch.float32, device=device)  # [d, r]
                # Apply same terminal state's W_U J_N(h_L) to Z_i = T v and to I v = V_dark
                JN_Tv = rmsnorm_jvp(h_L.unsqueeze(0), Z_i.T, gamma, eps)       # [r, d]
                JN_Iv = rmsnorm_jvp(h_L.unsqueeze(0), V_dark.T, gamma, eps)    # [r, d]
                # W_U @ vec^T -> norm across vocab
                # Compute ||W_U u||_2 = ||S * (V_full^T u)||_2 cleanly in O(d^2) instead of O(V*d)!
                S_w = basis.singular_values.to(device=device, dtype=torch.float32)
                Vf = basis.V_full.to(device=device, dtype=torch.float32)

                tr_norms = torch.linalg.vector_norm((JN_Tv @ Vf) * S_w.unsqueeze(0), dim=1)
                id_norms = torch.linalg.vector_norm((JN_Iv @ Vf) * S_w.unsqueeze(0), dim=1)
                h_src = states[primary_source_layer][0, t, :]
                JN_src_v = rmsnorm_jvp(h_src.unsqueeze(0), V_dark.T, gamma, eps)
                src_norms = torch.linalg.vector_norm((JN_src_v @ Vf) * S_w.unsqueeze(0), dim=1)

                readout_transport_norms.append(float(tr_norms.mean().item()))
                readout_identity_norms.append(float(id_norms.mean().item()))
                readout_source_direct_norms.append(float(src_norms.mean().item()))
                readout_raw_wu_norms.append(raw_wu_dark)

    layer_stats = {}
    for l in layers_to_scan:
        D_arr = np.stack(dark_projs_by_layer[l], axis=0)    # [n, r_dark]
        B_arr = np.stack(bright_projs_by_layer[l], axis=0)  # [n, n_bright]
        d_var_total = float(np.sum(np.var(D_arr, axis=0)))
        b_var_total = float(np.sum(np.var(B_arr, axis=0)))
        d_var_per_dim = d_var_total / D_arr.shape[1]
        b_var_per_dim = b_var_total / B_arr.shape[1]
        d_norm_mean = float(np.mean(np.linalg.norm(D_arr, axis=1)))
        b_norm_mean = float(np.mean(np.linalg.norm(B_arr, axis=1)))
        entry = {
            "pre_rmsnorm_dark_l2_norm_mean": d_norm_mean,
            "pre_rmsnorm_bright_l2_norm_mean": b_norm_mean,
            "pre_rmsnorm_dark_to_bright_norm_ratio": d_norm_mean / max(b_norm_mean, 1e-12),
            "centered_dark_variance_total": d_var_total,
            "centered_bright_variance_total": b_var_total,
            "centered_dark_variance_fraction": d_var_total / max(d_var_total + b_var_total, 1e-12),
            "centered_per_dim_variance_density_ratio_dark_over_bright": d_var_per_dim / max(b_var_per_dim, 1e-12),
        }
        if l > 0:
            entry["block_residual_write_dark_norm_mean"] = float(np.mean(block_write_dark_norms[l]))
            entry["block_residual_write_total_norm_mean"] = float(np.mean(block_write_total_norms[l]))
        layer_stats[f"L{l}"] = entry

    D_post = np.stack(post_rmsnorm_dark_projs, axis=0)
    B_post = np.stack(post_rmsnorm_bright_projs, axis=0)
    dp_var = float(np.sum(np.var(D_post, axis=0)))
    bp_var = float(np.sum(np.var(B_post, axis=0)))
    post_stats = {
        "post_rmsnorm_dark_l2_norm_mean": float(np.mean(np.linalg.norm(D_post, axis=1))),
        "post_rmsnorm_bright_l2_norm_mean": float(np.mean(np.linalg.norm(B_post, axis=1))),
        "post_rmsnorm_centered_dark_variance_fraction": dp_var / max(dp_var + bp_var, 1e-12),
        "post_rmsnorm_per_dim_variance_density_ratio": (dp_var / D_post.shape[1]) / max(bp_var / B_post.shape[1], 1e-12),
    }

    mean_tr = float(np.mean(readout_transport_norms))
    mean_id = float(np.mean(readout_identity_norms))
    mean_raw = float(np.mean(readout_raw_wu_norms))
    mean_src = float(np.mean(readout_source_direct_norms))

    return {
        "n_train_contexts_scanned": n_use,
        "per_layer_pre_rmsnorm_stats": layer_stats,
        "final_layer_post_rmsnorm_stats": post_stats,
        "same_terminal_state_readout_comparison": {
            "source_layer": primary_source_layer,
            "target_layer": num_layers,
            "mean_norm_WU_JN_hL_Tv": mean_tr,
            "mean_norm_WU_JN_hL_Iv_identity_bypass": mean_id,
            "ratio_transport_over_identity_same_terminal_norm": mean_tr / max(mean_id, 1e-12),
            "mean_norm_raw_WU_v_unnormalized": mean_raw,
            "mean_norm_WU_JN_hSrc_v_source_layer_direct": mean_src,
        },
    }
