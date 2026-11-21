"""Cross-Architecture Residual Transport (JVP / VJP), Chat Tokenization & Option Token Resolution (BAYES-DYN-05).

Supports Qwen2, Phi-3, OLMo-2, and Llama-3.1 in transformers v5 under FP32 eager attention.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

import numpy as np
import torch
import torch.autograd.forward_ad as fwAD

SYS_PROMPT = "Answer the multiple-choice question with only the single correct capital letter."


def tokenize_chat_prompt_v05(tokenizer: Any, user_prompt: str, device: torch.device) -> dict[str, Any]:
    """Apply model's official chat template cleanly (with graceful system-role fallback)."""
    messages_sys = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        text = tokenizer.apply_chat_template(messages_sys, tokenize=False, add_generation_prompt=True)
    except Exception:
        messages_user = [{"role": "user", "content": f"{SYS_PROMPT}\n\n{user_prompt}"}]
        text = tokenizer.apply_chat_template(messages_user, tokenize=False, add_generation_prompt=True)

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


def _layer_call_kwargs(
    model: torch.nn.Module,
    h: torch.Tensor,
    causal_mask: torch.Tensor,
    pos_emb: tuple[torch.Tensor, torch.Tensor] | None,
    pos_ids: torch.Tensor,
) -> dict[str, Any]:
    """Inspect layer[0].forward signature (cached on model) to pass exact architecture-matched kwargs."""
    sig_params = getattr(model, "_cached_layer_sig_params", None)
    if sig_params is None:
        sig_params = set(inspect.signature(model.model.layers[0].forward).parameters.keys())
        setattr(model, "_cached_layer_sig_params", sig_params)

    kw: dict[str, Any] = {}
    if "attention_mask" in sig_params:
        kw["attention_mask"] = causal_mask
    if "position_embeddings" in sig_params and pos_emb is not None:
        kw["position_embeddings"] = pos_emb
    if "position_ids" in sig_params:
        kw["position_ids"] = pos_ids
    if "use_cache" in sig_params:
        kw["use_cache"] = False
    return kw


def _build_position_embeddings(model: torch.nn.Module, h: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor]:
    seq_len = h.shape[1]
    pos_ids = torch.arange(seq_len, device=h.device, dtype=torch.long).unsqueeze(0)
    rot = getattr(model.model, "rotary_emb", None)
    if rot is not None:
        return rot(h, pos_ids), pos_ids
    return None, pos_ids


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
      - states[0] = post-embedding, pre-block-0
      - states[l] = output after first l blocks
      - states[L] = output after final block, PRE final RMSNorm
    Computes logits_last on identical [1, 1, d] slice as evaluate_intervention_batch1 for 0.0 zero-pert diff!
    """
    num_layers = len(model.model.layers)
    states: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        h = model.model.embed_tokens(input_ids)
        states[0] = h.detach().clone()
        pos_emb, pos_ids = _build_position_embeddings(model, h)
        causal_mask = _build_causal_mask(h.shape[1], h.dtype, h.device)
        kw = _layer_call_kwargs(model, h, causal_mask, pos_emb, pos_ids)
        for idx in range(num_layers):
            out = model.model.layers[idx](h, **kw)
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
    pos_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    h = h_start
    if pos_ids is None or pos_emb is None:
        pe, pi = _build_position_embeddings(model, h)
        if pos_emb is None:
            pos_emb = pe
        if pos_ids is None:
            pos_ids = pi
    if causal_mask is None:
        causal_mask = _build_causal_mask(h.shape[1], h.dtype, h.device)
    kw = _layer_call_kwargs(model, h, causal_mask, pos_emb, pos_ids)
    for idx in range(start_layer, end_layer):
        out = model.model.layers[idx](h, **kw)
        h = out[0] if isinstance(out, tuple) else out
    return h


def compute_exact_jvp_columns_batched(
    model: torch.nn.Module,
    h_start: torch.Tensor,
    start_layer: int,
    end_layer: int,
    t_source: int,
    V_dirs: torch.Tensor,
    t_target: int | None = None,
    batch_size: int = 8,
) -> torch.Tensor:
    if t_target is None:
        t_target = t_source
    device = h_start.device
    dtype = h_start.dtype
    _, S, d = h_start.shape
    r = int(V_dirs.shape[1])
    V_t = V_dirs.to(device=device, dtype=dtype)

    causal_mask = _build_causal_mask(S, dtype, device)
    pos_emb, pos_ids = _build_position_embeddings(model, h_start)

    cols = []
    with torch.no_grad():
        for start in range(0, r, batch_size):
            end = min(start + batch_size, r)
            b = end - start
            primal = h_start.expand(b, S, d).contiguous()
            tangent = torch.zeros((b, S, d), dtype=dtype, device=device)
            tangent[:, t_source, :] = V_t[:, start:end].T

            with fwAD.dual_level():
                dual_in = fwAD.make_dual(primal, tangent)
                dual_out = run_segment_layers(
                    model, dual_in, start_layer, end_layer,
                    causal_mask=causal_mask, pos_emb=pos_emb, pos_ids=pos_ids,
                )
                tang_out = fwAD.unpack_dual(dual_out).tangent
                if tang_out is None:
                    tang_out = torch.zeros_like(primal)
                cols.append(tang_out[:, t_target, :].T.detach().clone())

    return torch.cat(cols, dim=1)


def compute_exact_jvp_dual_mid_and_end(
    model: torch.nn.Module,
    h_start: torch.Tensor,
    start_layer: int,
    mid_layer: int,
    end_layer: int,
    t_source: int,
    V_dirs: torch.Tensor,
    batch_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simultaneously capture exact JVP tangent at mid_layer and end_layer in ONE forward-AD pass."""
    device = h_start.device
    dtype = h_start.dtype
    _, S, d = h_start.shape
    m = int(V_dirs.shape[1])
    V_t = V_dirs.to(device=device, dtype=dtype)
    causal_mask = _build_causal_mask(S, dtype, device)
    pos_emb, pos_ids = _build_position_embeddings(model, h_start)
    kw = _layer_call_kwargs(model, h_start, causal_mask, pos_emb, pos_ids)

    cols_mid, cols_end = [], []
    with torch.no_grad():
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            b = end - start
            primal = h_start.expand(b, S, d).contiguous()
            tangent = torch.zeros((b, S, d), dtype=dtype, device=device)
            tangent[:, t_source, :] = V_t[:, start:end].T

            with fwAD.dual_level():
                h = fwAD.make_dual(primal, tangent)
                z_mid_b = None
                for idx in range(start_layer, end_layer):
                    out = model.model.layers[idx](h, **kw)
                    h = out[0] if isinstance(out, tuple) else out
                    if idx + 1 == mid_layer:
                        tm = fwAD.unpack_dual(h).tangent
                        z_mid_b = (tm if tm is not None else torch.zeros_like(primal))[:, t_source, :].T.detach().clone()
                te = fwAD.unpack_dual(h).tangent
                z_end_b = (te if te is not None else torch.zeros_like(primal))[:, t_source, :].T.detach().clone()
                assert z_mid_b is not None
                cols_mid.append(z_mid_b)
                cols_end.append(z_end_b)
    return torch.cat(cols_end, dim=1), torch.cat(cols_mid, dim=1)


def compute_exact_vjp_single(
    model: torch.nn.Module,
    h_start: torch.Tensor,
    start_layer: int,
    end_layer: int,
    t_source: int,
    w_dir: torch.Tensor,
    t_target: int | None = None,
) -> torch.Tensor:
    if t_target is None:
        t_target = t_source
    device = h_start.device
    dtype = h_start.dtype
    _, S, _ = h_start.shape
    causal_mask = _build_causal_mask(S, dtype, device)
    pos_emb, pos_ids = _build_position_embeddings(model, h_start)

    h_req = h_start.detach().clone().requires_grad_(True)
    h_out = run_segment_layers(
        model, h_req, start_layer, end_layer,
        causal_mask=causal_mask, pos_emb=pos_emb, pos_ids=pos_ids,
    )
    scalar = torch.dot(h_out[0, t_target, :], w_dir.to(device=device, dtype=dtype))
    (grad_h,) = torch.autograd.grad(scalar, h_req, create_graph=False)
    return grad_h[0, t_source, :].detach()


def resolve_option_token_ids_v05(
    model: torch.nn.Module,
    tokenizer: Any,
    pilot_4: list[dict[str, Any]],
    device: torch.device,
    options: tuple[str, ...] = ("A", "B", "C", "D"),
) -> tuple[dict[str, int], dict[str, Any]]:
    """Resolve single-token IDs for A/B/C/D across BPE and SentencePiece tokenizers,
    picking the candidate convention with highest baseline softmax mass on the 4 P0 pilot prompts (pre-Confirm!).
    """
    candidates: dict[str, dict[str, int]] = {}

    # Candidate 1: encode("A", add_special_tokens=False)
    c_enc: dict[str, int] = {}
    ok_enc = True
    for o in options:
        ids = tokenizer.encode(o, add_special_tokens=False)
        if len(ids) == 1:
            c_enc[o] = int(ids[0])
        elif len(ids) > 1:
            c_enc[o] = int(ids[-1])
        else:
            ok_enc = False
    if ok_enc and len(set(c_enc.values())) == len(options):
        candidates["encode_bare"] = c_enc

    # Candidate 2: encode(" A", add_special_tokens=False)
    c_sp: dict[str, int] = {}
    ok_sp = True
    for o in options:
        ids = tokenizer.encode(" " + o, add_special_tokens=False)
        if len(ids) == 1:
            c_sp[o] = int(ids[0])
        else:
            ok_sp = False
    if ok_sp and len(set(c_sp.values())) == len(options) and c_sp != c_enc:
        candidates["encode_spaced"] = c_sp

    # Candidate 3: raw vocab convert_tokens_to_ids("A") (distinguishes SentencePiece A vs ▁A)
    c_raw: dict[str, int] = {}
    ok_raw = True
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for o in options:
        tid = tokenizer.convert_tokens_to_ids(o)
        if tid is None or tid < 0 or (unk_id is not None and tid == unk_id):
            ok_raw = False
        else:
            c_raw[o] = int(tid)
    if ok_raw and len(set(c_raw.values())) == len(options) and c_raw not in candidates.values():
        candidates["raw_vocab"] = c_raw

    assert candidates, "Failed to resolve any valid 4-option token ID mapping!"

    # Score candidate conventions on 4 P0 pilot prompts via native model forward
    mass_scores: dict[str, list[float]] = {k: [] for k in candidates}
    for rec in pilot_4:
        tok = tokenize_chat_prompt_v05(tokenizer, rec["user_prompt"], device)
        with torch.no_grad():
            out = model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"], use_cache=False)
            probs = torch.softmax(out.logits[0, tok["t_last"], :].to(torch.float32), dim=-1)
        for cname, cmap in candidates.items():
            mass_scores[cname].append(float(sum(probs[tid].item() for tid in cmap.values())))

    mean_masses = {k: float(np.mean(v)) for k, v in mass_scores.items()}
    best_name = max(mean_masses.keys(), key=lambda k: mean_masses[k])
    chosen = candidates[best_name]
    meta = {
        "chosen_convention": best_name,
        "chosen_ids": chosen,
        "chosen_decoded": {o: repr(tokenizer.decode([tid])) for o, tid in chosen.items()},
        "candidate_mean_prob_mass_on_pilot4": mean_masses,
        "all_candidates": candidates,
    }
    return chosen, meta


def extract_context_features(
    h_ell_t: np.ndarray,
    V_in: np.ndarray,
    V_legacy12: np.ndarray,
    seq_len: int,
) -> np.ndarray:
    h = h_ell_t.astype(np.float64)
    h_norm = max(float(np.linalg.norm(h)), 1e-12)
    h_unit = h / h_norm
    p_in = V_in.T @ h_unit
    p_leg = V_legacy12.T @ h_unit
    dark_frac = float(np.sum(p_in ** 2))
    scalars = np.array([math.log(h_norm), dark_frac, math.log(max(seq_len, 1))], dtype=np.float64)
    return np.concatenate([p_in, p_leg, scalars], axis=0)
