"""Independent Metric & Bootstrap Recomputation Auditor (Pure NumPy, ZERO report_generator import, §11).

Recomputes Link 1 (Q_perp 896-D normalized error), Link 2 (F0 Q_perp MSE), and Link 3 (Real intervention target tracking error)
directly from raw archived observations, posteriors, and intervention JSONLs!
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def strat_boot(a: np.ndarray, b: np.ndarray, fams: list[str], n_boot: int, seed: int) -> dict[str, Any]:
    d = a - b
    n = len(d)
    f_u = sorted(set(fams))
    f_idx = [np.array([i for i, f in enumerate(fams) if f == fm], dtype=np.int64) for fm in f_u]
    rng = np.random.default_rng(seed)
    sums = np.zeros(n_boot, dtype=np.float64)
    for idxs in f_idx:
        nf = len(idxs)
        draws = rng.integers(0, nf, size=(n_boot, nf))
        sums += np.sum(d[idxs][draws], axis=1)
    bm = sums / float(n)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return {"mean": float(np.mean(d)), "ci95": [float(lo), float(hi)], "rel_reduction": float((np.mean(a) - np.mean(b)) / max(float(np.mean(a)), 1e-12))}


def main() -> None:
    cfg = json.loads((ROOT / "configs" / "experiment_config.json").read_text(encoding="utf-8"))
    n_boot = int(cfg["stats"]["bootstrap_resamples"])
    b_seed = int(cfg["derived_seeds"]["bootstrap_stats"])
    pm = np.load(ROOT / "results" / "prior_model.npz")
    s_A_sq = float(pm["s_A_sq"])
    V_dark = pm["V_dark"].astype(np.float64)

    out: dict[str, Any] = {}
    for split_name in ("confirm_id", "confirm_ood"):
        obs = np.load(ROOT / "results" / "observations" / f"{split_name}_obs.npz", allow_pickle=True)
        post = np.load(ROOT / "results" / "posteriors" / f"{split_name}_posteriors.npz", allow_pickle=True)
        recs = load_jsonl(ROOT / "data" / f"{split_name}.jsonl")
        ints = load_jsonl(ROOT / "results" / "interventions" / f"{split_name}_interventions.jsonl")

        fams = [r["family"] for r in recs]
        cids = [r["context_id"] for r in recs]
        Bs = obs["B_38x38"].astype(np.float64)
        Yp = obs["Y_perp_true"].astype(np.float64)
        g0s = obs["g0"].astype(np.float64)
        Mk = post["M_k"].astype(np.float64)  # [N, 4, d, r], budgets [0, 4, 8, 16]

        L0, L8, F0_0, F0_8 = [], [], [], []
        for i in range(len(cids)):
            Qp = Bs[i, :, 16:24]
            d0 = Mk[i, 0] @ Qp - Yp[i]
            d8 = Mk[i, 2] @ Qp - Yp[i]
            L0.append(float(np.mean(np.sum(d0 ** 2, axis=0)) / s_A_sq))
            L8.append(float(np.mean(np.sum(d8 ** 2, axis=0)) / s_A_sq))
            f0_err0 = g0s[i] @ d0
            f0_err8 = g0s[i] @ d8
            F0_0.append(float(np.mean(f0_err0 ** 2)))
            F0_8.append(float(np.mean(f0_err8 ** 2)))

        by_mc = {(r["context_id"], r["method"]): r["target_tracking_err"] for r in ints}
        E0 = np.array([by_mc[(c, "PRIOR")] for c in cids])
        E8 = np.array([by_mc[(c, "UPDATE_8")] for c in cids])

        out[split_name] = {
            "link1_L0_minus_L8_perp": strat_boot(np.array(L0), np.array(L8), fams, n_boot, b_seed),
            "link2_F0_MSE0_minus_MSE8_perp": strat_boot(np.array(F0_0), np.array(F0_8), fams, n_boot, b_seed),
            "link3_CtrlErr0_minus_CtrlErr8": strat_boot(E0, E8, fams, n_boot, b_seed),
        }

    # Cross-check against results/metrics_summary.json!
    ms = json.loads((ROOT / "results" / "metrics_summary.json").read_text(encoding="utf-8"))
    for sn in ("confirm_id", "confirm_ood"):
        a1 = out[sn]["link1_L0_minus_L8_perp"]["mean"]
        m1 = ms[sn]["paired_links"]["PRIOR_minus_UPDATE_8_L_perp"]["mean"]
        a2 = out[sn]["link2_F0_MSE0_minus_MSE8_perp"]["mean"]
        m2 = ms[sn]["paired_links"]["PRIOR_minus_UPDATE_8_f0_mse_qperp"]["mean"]
        a3 = out[sn]["link3_CtrlErr0_minus_CtrlErr8"]["mean"]
        m3 = ms[sn]["paired_links"]["PRIOR_minus_UPDATE_8_ctrl_err"]["mean"]
        assert abs(a1 - m1) < 1e-5 and abs(a2 - m2) < 1e-6 and abs(a3 - m3) < 1e-10, f"Mismatch on {sn}: {(a1,m1),(a2,m2),(a3,m3)}"

    out_p = ROOT / "audit" / "recomputed_summary.json"
    out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[AUDIT recompute_metrics] VERIFIED exact match on ALL 3 links (ID & OOD): {out['confirm_id']}")


if __name__ == "__main__":
    main()
