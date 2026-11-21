"""Fixed 38x38 Probe Banks & Strict Isolation ProbeSessionV05 (BAYES-DYN-05)."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import numpy as np


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed_from_tag(tag: str, base_seed: int) -> int:
    h = hashlib.sha256(f"{base_seed}|{tag}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % (2 ** 32 - 1)


def generate_fixed_38x38_bank(context_id: str, bank_seed: int, r: int = 38) -> np.ndarray:
    """Deterministic orthonormal 38x38 QR bank per context:
      - cols 0:16  = nested learning supports (k=4, 8, 16)
      - cols 16:24 = Q_perp (8 strictly orthogonal sealed directions)
      - cols 24:38 = 14 remaining cols for 12-context exact 38-D trajectory completion
    """
    rng = np.random.default_rng(_seed_from_tag(f"bank38|{context_id}", bank_seed))
    Q, _ = np.linalg.qr(rng.normal(size=(r, r)))
    max_idx = np.argmax(np.abs(Q), axis=0)
    signs = np.sign(Q[max_idx, np.arange(r)])
    signs[signs == 0] = 1.0
    return (Q * signs[None, :]).astype(np.float64)


def generate_q_general(context_id: str, qgen_seed: int, r: int = 38, n_dirs: int = 4) -> np.ndarray:
    rng = np.random.default_rng(_seed_from_tag(f"qgen|{context_id}", qgen_seed))
    Q, _ = np.linalg.qr(rng.normal(size=(r, n_dirs)))
    max_idx = np.argmax(np.abs(Q), axis=0)
    signs = np.sign(Q[max_idx, np.arange(n_dirs)])
    signs[signs == 0] = 1.0
    return (Q * signs[None, :]).astype(np.float64)


class BudgetExceededError(RuntimeError):
    pass


class UnauthorizedQueryError(RuntimeError):
    pass


class ProbeSessionV05:
    """Strict isolation wrapper enforcing role ('learning_support' vs 'evaluator_truth'),
    hard JVP column budgets, and hard VJP prohibition for learning sessions.
    """

    def __init__(
        self,
        session_id: str,
        role: str,
        context_id: str,
        model: Any,
        h_start: Any,
        V_in: Any,
        start_layer: int,
        mid_layer: int,
        end_layer: int,
        t_source: int,
        max_jvp_dirs: int,
        allow_vjp: bool = False,
        jvp_batch_size: int = 8,
    ) -> None:
        assert role in ("learning_support", "evaluator_truth")
        if role == "learning_support":
            assert not allow_vjp, "Learning session CANNOT enable VJP!"
        self.session_id = session_id
        self.role = role
        self.context_id = context_id
        self.model = model
        self.h_start = h_start
        self.V_in = V_in
        self.start_layer = start_layer
        self.mid_layer = mid_layer
        self.end_layer = end_layer
        self.t_source = t_source
        self.max_jvp_dirs = int(max_jvp_dirs)
        self.allow_vjp = bool(allow_vjp)
        self.jvp_batch_size = int(jvp_batch_size)
        self.jvp_dirs_used = 0
        self.vjp_dirs_used = 0
        self.events: list[dict[str, Any]] = []

    def jvp_columns(self, X_cols: np.ndarray, tag: str, record_mid_z: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
        import torch
        from src.transport_v05 import compute_exact_jvp_columns_batched, compute_exact_jvp_dual_mid_and_end

        X = np.asarray(X_cols, dtype=np.float64)
        m = int(X.shape[1])
        if self.jvp_dirs_used + m > self.max_jvp_dirs:
            raise BudgetExceededError(f"[{self.session_id}] JVP budget exceeded: {self.jvp_dirs_used}+{m} > {self.max_jvp_dirs}")
        self.jvp_dirs_used += m

        device = self.h_start.device
        V_dirs = self.V_in.to(device=device, dtype=torch.float32) @ torch.tensor(X, dtype=torch.float32, device=device)
        if record_mid_z:
            Z_end_t, Z_mid_t = compute_exact_jvp_dual_mid_and_end(
                self.model, self.h_start, self.start_layer, self.mid_layer, self.end_layer,
                self.t_source, V_dirs, batch_size=self.jvp_batch_size,
            )
            Z_end = Z_end_t.detach().to(torch.float64).cpu().numpy()
            Z_mid = Z_mid_t.detach().to(torch.float64).cpu().numpy()
        else:
            Z_end_t = compute_exact_jvp_columns_batched(
                self.model, self.h_start, self.start_layer, self.end_layer,
                self.t_source, V_dirs, batch_size=self.jvp_batch_size,
            )
            Z_end = Z_end_t.detach().to(torch.float64).cpu().numpy()
            Z_mid = None

        self.events.append({
            "utc": _utc_now(), "kind": "JVP", "tag": tag, "n_dirs": m,
            "record_mid_z": bool(record_mid_z),
            "X_sha256": hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()[:16],
        })
        return Z_end, Z_mid

    def vjp_eval_reference(self, g_vec: np.ndarray, tag: str) -> np.ndarray:
        import torch
        from src.transport_v05 import compute_exact_vjp_single

        if not self.allow_vjp or self.role != "evaluator_truth":
            raise UnauthorizedQueryError(f"[{self.session_id}] VJP prohibited for role={self.role}!")
        self.vjp_dirs_used += 1
        device = self.h_start.device
        g_t = torch.tensor(np.asarray(g_vec, dtype=np.float32), dtype=torch.float32, device=device)
        grad_h = compute_exact_vjp_single(
            self.model, self.h_start, self.start_layer, self.end_layer, self.t_source, g_t,
        )
        a_dark = (self.V_in.to(device=device, dtype=torch.float32).T @ grad_h).detach().to(torch.float64).cpu().numpy()
        self.events.append({"utc": _utc_now(), "kind": "VJP", "tag": tag})
        return a_dark

    def receipt(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id, "role": self.role, "context_id": self.context_id,
            "max_jvp_dirs": self.max_jvp_dirs, "jvp_dirs_used": self.jvp_dirs_used,
            "allow_vjp": self.allow_vjp, "vjp_dirs_used": self.vjp_dirs_used,
            "events": self.events,
        }
