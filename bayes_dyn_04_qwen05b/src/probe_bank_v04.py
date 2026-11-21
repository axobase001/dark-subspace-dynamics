"""Fixed 38x38 QR Probe Bank, Q_general, Frozen Mispairing Matrix, and Online ProbeSessionV04 (§5.2, §9.1).

Fixes BAYES-DYN-03's k_max direction-identity issue permanently:
  - Always generates ONE 38x38 Gaussian matrix from SHA256(f"{bank_seed}:{context_id}"), QR-decomposes,
    fixes column signs via diag(R) >= 0, and preserves float64!
  - Columns 0:16  = nested supports X_sup_16 (k=4 is 0:4, k=8 is 0:8, k=16 is 0:16)
  - Columns 16:24 = fixed Q_perp (8 directions strictly orthogonal to ALL 16 supports)
  - Columns 24:38 = B_rem_14 (completed ONLY for the 16-context trajectory panel AFTER all learning commits)
  - Separate frozen seed generates Q_general (38x4 unit columns).
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bytes_sha256_short(arr: np.ndarray) -> str:
    """Hash EXACT archived dtype bytes (no hidden float64 cast!)."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def generate_fixed_38x38_bank(context_id: str, bank_seed: int, r: int = 38) -> np.ndarray:
    """Always generate a single fixed [r, r] float64 orthonormal matrix B from (bank_seed, context_id)."""
    digest = hashlib.sha256(f"{bank_seed}:{context_id}".encode("utf-8")).digest()
    seed_int = int.from_bytes(digest[:8], "little") % (2 ** 32)
    rng = np.random.default_rng(seed_int)
    G = rng.normal(size=(r, r))
    Q, R = np.linalg.qr(G)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    B = (Q * signs[None, :]).astype(np.float64)
    err = float(np.max(np.abs(B.T @ B - np.eye(r))))
    assert err < 1e-11, f"Bank orthonormality error {err}"
    return B


def generate_q_general(context_id: str, qgen_seed: int, r: int = 38, n_dirs: int = 4) -> np.ndarray:
    """Generate [r, n_dirs] independent unit columns (NOT forced orthogonal to support)."""
    digest = hashlib.sha256(f"{qgen_seed}:{context_id}".encode("utf-8")).digest()
    seed_int = int.from_bytes(digest[:8], "little") % (2 ** 32)
    rng = np.random.default_rng(seed_int)
    G = rng.normal(size=(r, n_dirs)).astype(np.float64)
    G = G / np.linalg.norm(G, axis=0, keepdims=True)
    return G


def generate_frozen_mispair_matrix(mispair_seed: int, k: int = 8) -> np.ndarray:
    """Pre-frozen non-identity signed permutation matrix S in R^{k x k} (derangement + random signs)."""
    rng = np.random.default_rng(mispair_seed)
    # Cyclic shift + shuffle guaranteed to be a strict derangement (no fixed point!)
    perm = np.arange(k)
    while np.any(perm == np.arange(k)):
        perm = rng.permutation(k)
    signs = rng.choice([-1.0, 1.0], size=k)
    S = np.zeros((k, k), dtype=np.float64)
    for j in range(k):
        S[perm[j], j] = signs[j]
    assert np.allclose(S @ S.T, np.eye(k)) and not np.allclose(S, np.eye(k)) and not np.allclose(S, -np.eye(k))
    return S


class BudgetExceededError(RuntimeError):
    pass


class UnauthorizedQueryError(RuntimeError):
    pass


class ProbeSessionV04:
    """Strict online query service for BAYES-DYN-04 separating learning support queries from evaluator truth queries.
    Optionally records mid-layer z20 tangents inside the EXACT same forward-AD pass (zero extra JVP directions!).
    """

    def __init__(
        self,
        session_id: str,
        role: str,                 # 'learning_support' or 'evaluator_truth'
        context_id: str,
        model: Any,
        h_start: Any,              # [1, S, d] torch.Tensor
        V_in: Any,                 # [d, r] torch.Tensor
        src_l: int,
        mid_l: int,
        tgt_l: int,
        t_source: int,
        max_jvp_dirs: int,
        allow_vjp: bool = False,
        jvp_batch_size: int = 8,
    ) -> None:
        self.session_id = session_id
        self.role = role
        self.context_id = context_id
        self._model = model
        self._h_start = h_start
        self._V_in = V_in
        self._src_l = src_l
        self._mid_l = mid_l
        self._tgt_l = tgt_l
        self._t_source = t_source
        self.max_jvp_dirs = int(max_jvp_dirs)
        self.allow_vjp = bool(allow_vjp)
        self.jvp_batch_size = int(jvp_batch_size)

        self.jvp_dirs_used = 0
        self.jvp_batch_calls = 0
        self.vjp_dirs_used = 0
        self.created_utc = utc_now()
        self._seen_col_hashes: set[str] = set()
        self._events: list[dict[str, Any]] = []

    def jvp_columns(
        self,
        X_cols: np.ndarray,           # [r, k]
        tag: str,
        record_mid_z20: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Compute REAL batched forward-AD JVP Z_24 = J_{16->24} V X_cols [d, k] (and optionally Z_20 [d, k]).
        Charges k directions against max_jvp_dirs; rejects duplicates and over-budget queries.
        """
        import torch
        import torch.autograd.forward_ad as fwAD
        from src.transport import run_segment_layers, _build_position_embeddings, _build_causal_mask

        X = np.asarray(X_cols, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        r, k = X.shape
        if self.jvp_dirs_used + k > self.max_jvp_dirs:
            self._events.append({"event": "jvp_rejected_budget", "utc": utc_now(), "tag": tag, "req": k, "used": self.jvp_dirs_used, "max": self.max_jvp_dirs})
            raise BudgetExceededError(f"{self.session_id}: requested {k} dirs, used {self.jvp_dirs_used}/{self.max_jvp_dirs}")

        col_hashes = [bytes_sha256_short(X[:, j]) for j in range(k)]
        if self.role == "learning_support":
            if len(set(col_hashes)) != k or any(h in self._seen_col_hashes for h in col_hashes):
                raise UnauthorizedQueryError(f"{self.session_id}: duplicate support direction submitted!")

        device = self._h_start.device
        dtype = self._h_start.dtype
        V_dirs = self._V_in.to(dtype=dtype, device=device) @ torch.tensor(X, dtype=dtype, device=device)  # [d, k]

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        z24_chunks, z20_chunks = [], []
        seq_len = self._h_start.shape[1]
        n_calls = 0

        for c0 in range(0, k, self.jvp_batch_size):
            c1 = min(c0 + self.jvp_batch_size, k)
            b = c1 - c0
            n_calls += 1
            h_rep = self._h_start.detach().expand(b, -1, -1).clone()
            tang = torch.zeros_like(h_rep)
            tang[:, self._t_source, :] = V_dirs[:, c0:c1].T
            pos_emb = _build_position_embeddings(self._model, h_rep)
            causal_mask = _build_causal_mask(seq_len, dtype, device)
            with torch.no_grad(), fwAD.dual_level():
                dual_h = fwAD.make_dual(h_rep, tang)
                if record_mid_z20:
                    dual_mid = run_segment_layers(self._model, dual_h, self._src_l, self._mid_l, causal_mask=causal_mask, pos_emb=pos_emb)
                    _, t20 = fwAD.unpack_dual(dual_mid)
                    z20_chunks.append(t20[:, self._t_source, :].T.detach().to(torch.float64).cpu().numpy())
                    dual_end = run_segment_layers(self._model, dual_mid, self._mid_l, self._tgt_l, causal_mask=causal_mask, pos_emb=pos_emb)
                else:
                    dual_end = run_segment_layers(self._model, dual_h, self._src_l, self._tgt_l, causal_mask=causal_mask, pos_emb=pos_emb)
                _, t24 = fwAD.unpack_dual(dual_end)
                z24_chunks.append(t24[:, self._t_source, :].T.detach().to(torch.float64).cpu().numpy())

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        Z24 = np.concatenate(z24_chunks, axis=1)
        Z20 = np.concatenate(z20_chunks, axis=1) if record_mid_z20 else None

        before = self.jvp_dirs_used
        self.jvp_dirs_used += k
        self.jvp_batch_calls += n_calls
        for h in col_hashes:
            self._seen_col_hashes.add(h)
        self._events.append({
            "event": "jvp_columns", "utc": utc_now(), "tag": tag, "n_dirs": k,
            "batch_calls": n_calls, "record_mid_z20": record_mid_z20,
            "jvp_used_before": before, "jvp_used_after": self.jvp_dirs_used,
            "gpu_ms": round(elapsed_ms, 3),
            "X_sha256": bytes_sha256_short(X), "Z24_sha256": bytes_sha256_short(Z24),
        })
        return Z24, Z20

    def vjp_eval_reference(self, g_vec: np.ndarray, feature_id: str) -> np.ndarray:
        """Evaluator-only exact source VJP a_true = (JV)^T g_vec. Strictly forbidden for learning_support sessions!"""
        import torch
        from src.transport import compute_exact_vjp_single

        if not self.allow_vjp:
            self._events.append({"event": "vjp_rejected_unauthorized", "utc": utc_now(), "feature_id": feature_id})
            raise UnauthorizedQueryError(f"{self.session_id} ({self.role}): VJP strictly forbidden!")
        device = self._h_start.device
        dtype = self._h_start.dtype
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g_t = torch.tensor(np.asarray(g_vec, dtype=np.float64), dtype=dtype, device=device)
        vjp_h = compute_exact_vjp_single(self._model, self._h_start, self._src_l, self._tgt_l, self._t_source, g_t)
        a_true = (self._V_in.to(torch.float64).T @ vjp_h.to(torch.float64)).detach().cpu().numpy()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.vjp_dirs_used += 1
        self._events.append({
            "event": "vjp_eval_reference", "utc": utc_now(), "feature_id": feature_id,
            "vjp_dirs_after": self.vjp_dirs_used, "gpu_ms": round(elapsed_ms, 3),
            "a_sha256": bytes_sha256_short(a_true),
        })
        return a_true

    def receipt(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "role": self.role,
            "context_id": self.context_id,
            "created_utc": self.created_utc,
            "receipt_utc": utc_now(),
            "jvp_dirs_used": self.jvp_dirs_used,
            "jvp_batch_calls": self.jvp_batch_calls,
            "max_jvp_dirs": self.max_jvp_dirs,
            "vjp_dirs_used": self.vjp_dirs_used,
            "allow_vjp": self.allow_vjp,
            "total_gpu_ms": round(sum(e.get("gpu_ms", 0.0) for e in self._events), 3),
            "events": list(self._events),
        }
