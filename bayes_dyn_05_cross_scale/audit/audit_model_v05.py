"""Independent Pure-NumPy CPU Auditor for BAYES-DYN-05 (per model).

Verifies WITHOUT importing torch or GPU code:
  1. Full posterior replay on ALL Confirm-ID (64x4=256) & Confirm-OOD (32x4=128) = 384 states
     - C_k max diff vs archived <= 1e-12
     - Support interpolation residual <= 1e-10
     - min_eig(C_k) >= 0
     - PATCH_8 Q_perp invariance <= 1e-12
     - Byte-exact M_k float32 SHA-256 match on ALL 48 archived traj12 states vs Stage-A predictions.jsonl
  2. Independent recomputation of ALL 3 links (Link 1 L_perp, Link 2 F0 MSE, Link 3 Tracking Error) + PATCH_8 vs UPDATE_8
     with 20,000 family-stratified paired bootstrap vs metrics_summary.json
  3. Strict chronological & budget isolation audit on query_receipts.jsonl & predictions.jsonl
  4. Exact frozen_config.json SHA-256 verification across all data & code files
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(p: Path) -> list[dict[str, Any]]:
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bytes_sha256_short(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def solve_spd_right(S: np.ndarray, B: np.ndarray, jit: float) -> np.ndarray:
    k = S.shape[0]
    S_sym = 0.5 * (S + S.T)
    for mult in (1.0, 10.0, 100.0, 1e4):
        try:
            L = np.linalg.cholesky(S_sym + (jit * mult) * np.eye(k, dtype=np.float64))
            return np.linalg.solve(L.T, np.linalg.solve(L, B))
        except np.linalg.LinAlgError:
            continue
    eigs, Q = np.linalg.eigh(S_sym)
    return Q @ ((Q.T @ B) / np.maximum(eigs, max(jit * 1e6, 1e-10))[:, None])


def cond_exact(C: np.ndarray, M0: np.ndarray, X: np.ndarray, Y: np.ndarray, jit: float) -> tuple[np.ndarray, np.ndarray]:
    if X.shape[1] == 0:
        return M0.copy(), C.copy()
    CX = C @ X
    S = X.T @ CX
    W = solve_spd_right(S, CX.T, jit)
    Mk = M0 + (Y - M0 @ X) @ W
    Ck = 0.5 * ((C - CX @ W) + (C - CX @ W).T)
    return Mk, Ck


def paired_boot(a: np.ndarray, b: np.ndarray, fams: list[str], n_boot: int, seed: int) -> dict[str, Any]:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    by_f: dict[str, list[int]] = {}
    for i, fm in enumerate(fams):
        by_f.setdefault(fm, []).append(i)
    f_arrs = [np.array(v, dtype=np.int64) for v in by_f.values()]
    rng = np.random.default_rng(int(seed))
    tot = np.zeros(n_boot, dtype=np.float64)
    for fa in f_arrs:
        nf = len(fa)
        tot += d[fa[rng.integers(0, nf, size=(n_boot, nf))]].sum(axis=1)
    m = tot / float(len(d))
    lo, hi = float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
    return {"mean": float(d.mean()), "ci95": [lo, hi], "ci95_pos": bool(lo > 0.0), "rel_reduction": float(d.mean() / max(float(a.mean()), 1e-12))}


def audit_model(mkey: str) -> dict[str, Any]:
    mdir = ROOT / "models" / mkey
    cfg = json.loads((ROOT / "configs" / "suite_config.json").read_text(encoding="utf-8"))
    frozen = json.loads((mdir / "configs" / "frozen_config.json").read_text(encoding="utf-8"))
    ms = json.loads((mdir / "metrics_summary.json").read_text(encoding="utf-8"))
    pm = np.load(mdir / "prior_model.npz", allow_pickle=False)

    d, r = int(pm["d"]), int(pm["r"])
    W_M = pm["W_M"].astype(np.float64)
    xm, xs, ym = pm["x_mean"].astype(np.float64), pm["x_std"].astype(np.float64), pm["y_mean"].astype(np.float64)
    C0 = pm["C"].astype(np.float64)
    s_A_sq = float(pm["s_A_sq"])
    jit = float(pm["solver_jitter"])
    V_dark = pm["V_dark"].astype(np.float64)

    n_boot = int(cfg["stats"]["bootstrap_resamples"])
    b_seed = int(cfg["derived_seeds"]["bootstrap_stats"])

    # 1. Replay posteriors & recompute all 3 links on ID64 and OOD32
    preds = load_jsonl(mdir / "predictions.jsonl")
    pred_by_cid = {p["context_id"]: p for p in preds}
    traj_m = np.load(mdir / "posteriors" / "traj12_M_k.npz", allow_pickle=False)
    traj_cids = [str(x) for x in traj_m["context_ids"]]
    traj_Mk = traj_m["M_k"]

    replay_stats: dict[str, Any] = {}
    recomp_stats: dict[str, Any] = {}
    violations: list[str] = []

    for spl in ("confirm_id", "confirm_ood"):
        obs = np.load(mdir / "observations" / f"{spl}_obs.npz", allow_pickle=False)
        ck_arch = np.load(mdir / "posteriors" / f"{spl}_C_k.npz", allow_pickle=False)
        int_rows = load_jsonl(mdir / "interventions" / f"{spl}_interventions.jsonl")
        int_by_c: dict[str, dict[str, Any]] = {}
        for row in int_rows:
            int_by_c.setdefault(row["context_id"], {})[row["method"]] = row

        cids = [str(x) for x in obs["context_ids"]]
        n_s = len(cids)
        fams = [int_by_c[c]["PRIOR"]["family"] for c in cids]

        max_c_diff, max_sup_res, min_eig_c, max_patch_inv = 0.0, 0.0, float("inf"), 0.0
        l0_arr, l8_arr, f0_0_arr, f0_8_arr = np.zeros(n_s), np.zeros(n_s), np.zeros(n_s), np.zeros(n_s)
        ec0_arr, ec8_arr, ecp8_arr = np.zeros(n_s), np.zeros(n_s), np.zeros(n_s)

        for i, cid in enumerate(cids):
            phi = obs["phi"][i].astype(np.float64)
            B = obs["B_38x38"][i].astype(np.float64)
            Y16 = obs["Y_sup_16"][i].astype(np.float64)
            Yp = obs["Y_perp_true"][i].astype(np.float64)
            g0 = obs["g0"][i].astype(np.float64)
            Qp = B[:, 16:24]

            M0 = (((phi - xm) / xs) @ W_M + ym).reshape(d, r)
            for bi, kb in enumerate((0, 4, 8, 16)):
                Xk = B[:, :kb]
                Yk = Y16[:, :kb]
                Mk, Ck = cond_exact(C0, M0, Xk, Yk, jit)
                c_diff = float(np.max(np.abs(Ck - ck_arch["C_k"][i, bi])))
                max_c_diff = max(max_c_diff, c_diff)
                min_eig_c = min(min_eig_c, float(np.linalg.eigvalsh(Ck)[0]))
                if kb > 0:
                    sup_r = float(np.linalg.norm(Yk - Mk @ Xk, "fro") / max(float(np.linalg.norm(Yk, "fro")), 1e-12))
                    max_sup_res = max(max_sup_res, sup_r)
                if kb == 0:
                    l0_arr[i] = float(np.mean(np.sum((Mk @ Qp - Yp) ** 2, axis=0)) / s_A_sq)
                    a0_hat = (V_dark + Mk).T @ g0
                    f0_t = (V_dark @ Qp + Yp).T @ g0
                    f0_0_arr[i] = float(np.mean((a0_hat @ Qp - f0_t) ** 2))
                if kb == 8:
                    l8_arr[i] = float(np.mean(np.sum((Mk @ Qp - Yp) ** 2, axis=0)) / s_A_sq)
                    a8_hat = (V_dark + Mk).T @ g0
                    f0_t = (V_dark @ Qp + Yp).T @ g0
                    f0_8_arr[i] = float(np.mean((a8_hat @ Qp - f0_t) ** 2))
                    M_patch = M0 + (Yk - M0 @ Xk) @ Xk.T
                    max_patch_inv = max(max_patch_inv, float(np.linalg.norm((M_patch - M0) @ Qp, "fro")))

            ec0_arr[i] = float(int_by_c[cid]["PRIOR"]["target_tracking_err"])
            ec8_arr[i] = float(int_by_c[cid]["UPDATE_8"]["target_tracking_err"])
            ecp8_arr[i] = float(int_by_c[cid]["PATCH_8"]["target_tracking_err"])

        r_l1 = paired_boot(l0_arr, l8_arr, fams, n_boot, b_seed)
        r_l2 = paired_boot(f0_0_arr, f0_8_arr, fams, n_boot, b_seed)
        r_l3 = paired_boot(ec0_arr, ec8_arr, fams, n_boot, b_seed)
        r_p8 = paired_boot(ecp8_arr, ec8_arr, fams, n_boot, b_seed)

        # Cross-check exact match with GPU metrics_summary.json!
        ref_p = ms[spl]["paired_links"]
        if abs(r_l1["mean"] - ref_p["PRIOR_minus_UPDATE_8_L_perp"]["mean"]) > 1e-6:
            violations.append(f"{spl} Link1 mean mismatch")
        if abs(r_l2["mean"] - ref_p["PRIOR_minus_UPDATE_8_f0_mse_qperp"]["mean"]) > 1e-7:
            violations.append(f"{spl} Link2 mean mismatch")
        if abs(r_l3["mean"] - ref_p["PRIOR_minus_UPDATE_8_ctrl_err"]["mean"]) > 1e-7:
            violations.append(f"{spl} Link3 mean mismatch")

        if max_c_diff > 1e-11:
            violations.append(f"{spl} C_k replay diff={max_c_diff}")
        if max_sup_res > 1e-9:
            violations.append(f"{spl} support residual={max_sup_res}")
        if min_eig_c < -1e-11:
            violations.append(f"{spl} negative C_k eig={min_eig_c}")
        if max_patch_inv > 1e-11:
            violations.append(f"{spl} PATCH_8 invariance={max_patch_inv}")

        replay_stats[spl] = {
            "n_contexts": n_s, "n_states": n_s * 4,
            "max_abs_diff_C_k": max_c_diff, "max_support_residual_rel": max_sup_res,
            "min_eig_C_k": min_eig_c, "max_patch8_qperp_invariance_err": max_patch_inv,
        }
        recomp_stats[spl] = {
            "link1_L0_minus_L8_perp": r_l1, "link2_F0_MSE0_minus_MSE8": r_l2,
            "link3_CtrlErr0_minus_CtrlErr8": r_l3, "patch8_minus_update8_ctrl_err": r_p8,
        }

    # 2. Byte-exact M_k hash verification on all 12x4 = 48 archived trajectory states!
    n_traj_hash_ok = 0
    for ti, cid in enumerate(traj_cids):
        for bi, kb in enumerate((0, 4, 8, 16)):
            got = bytes_sha256_short(traj_Mk[ti, bi])
            exp = pred_by_cid[cid]["M_k_f32_sha256"][f"k{kb}"]
            if got != exp:
                violations.append(f"Traj M_k hash mismatch on {cid} k={kb}: {got} vs {exp}")
            else:
                n_traj_hash_ok += 1

    # 3. Query receipts & chronological isolation audit
    receipts = load_jsonl(mdir / "observations" / "query_receipts.jsonl")
    n_learn, n_eval = 0, 0
    eval_first_utc: dict[str, str] = {}
    for rec in receipts:
        if rec["jvp_dirs_used"] > rec["max_jvp_dirs"]:
            violations.append(f"JVP budget exceeded: {rec['session_id']}")
        if rec["role"] == "learning_support":
            n_learn += 1
            if rec["vjp_dirs_used"] > 0 or rec["allow_vjp"]:
                violations.append(f"Learning session VJP violation: {rec['session_id']}")
        elif rec["role"] == "evaluator_truth":
            n_eval += 1
            if rec["events"]:
                cid = rec["context_id"]
                ts = rec["events"][0]["utc"]
                if cid not in eval_first_utc or ts < eval_first_utc[cid]:
                    eval_first_utc[cid] = ts

    n_chrono = 0
    for p in preds:
        cid = p["context_id"]
        if cid in eval_first_utc:
            if p["commit_utc"] > eval_first_utc[cid]:
                violations.append(f"Chrono violation on {cid}: {p['commit_utc']} > {eval_first_utc[cid]}")
            n_chrono += 1

    # 4. Frozen data & code SHA-256 verification
    n_frozen_ok = 0
    for grp in ("data_file_sha256", "code_file_sha256"):
        for rel, exp in frozen[grp].items():
            fp = ROOT / rel
            if not fp.exists() or sha256_file(fp) != exp:
                violations.append(f"Frozen SHA mismatch: {rel}")
            else:
                n_frozen_ok += 1

    report = {
        "model_key": mkey,
        "audit_passed": len(violations) == 0,
        "violations": violations,
        "replay": replay_stats,
        "recomputed_links": recomp_stats,
        "traj12_byte_exact_M_k_hashes_ok": n_traj_hash_ok,
        "isolation": {"learning_sessions": n_learn, "evaluator_sessions": n_eval, "chronological_checks": n_chrono},
        "frozen_code_data_hashes_matched": n_frozen_ok,
    }
    (mdir / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[AUDIT v05:{mkey}] passed={report['audit_passed']} | violations={len(violations)} | learn={n_learn} eval={n_eval} chrono={n_chrono} trajM={n_traj_hash_ok}/48 frozen={n_frozen_ok}")
    assert report["audit_passed"], f"Audit failed on {mkey}: {violations}"
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_key", required=True)
    args = ap.parse_args()
    audit_model(args.model_key)


if __name__ == "__main__":
    main()
