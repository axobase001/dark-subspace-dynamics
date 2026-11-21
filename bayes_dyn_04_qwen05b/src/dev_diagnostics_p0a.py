"""P0A CPU Diagnostics (D1, D2, D3) & DEV-fit192 / DEV-tune32 Full-Operator Prior Fitting (§2, §3.3).

Runs 100% on CPU using archived local assets:
  - BAYES-DYN-02.5 results/raw_transport.npz (224 contexts, full 896x38 Z/A, 53-D phi, V_dark, U_hybrid_64)
  - BAYES-DYN-03 results/posterior_snapshots.npz, exact_reference.npz, intervention_vectors.npz (160 contexts)
Produces:
  - development/d1_d2_d3_diagnostics.json
  - development/d1_d2_d3_summary.md
  - development/dev_candidate_grid.json
  - development/dev_asset_hashes.json
  - configs/dev_selection_config.json
  - results/prior_model.npz
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.full_operator_map import (
    fit_and_select_full_operator_prior_on_dev,
    save_full_operator_prior_npz,
    predict_M0,
    condition_exact_batch,
    verify_full_operator_synth_fp64,
    _fit_ridge_multi,
    _predict_ridge_multi,
)
from src.probe_bank_v04 import generate_fixed_38x38_bank

PATH_025_RAW = pathlib.Path(r"C:\Users\User\.gemini\antigravity\scratch\bayes_dyn_02_5\results\raw_transport.npz")
PATH_025_PROMPTS = pathlib.Path(r"C:\Users\User\.gemini\antigravity\scratch\bayes_dyn_02_5\configs\prompts_224.jsonl")
PATH_03_SNAPS = pathlib.Path(r"C:\Users\User\.gemini\antigravity\scratch\BAYES_DYN_03\results\posterior_snapshots.npz")
PATH_03_EXACT = pathlib.Path(r"C:\Users\User\.gemini\antigravity\scratch\BAYES_DYN_03\results\exact_reference.npz")
PATH_03_VECS = pathlib.Path(r"C:\Users\User\.gemini\antigravity\scratch\BAYES_DYN_03\results\intervention_vectors.npz")
PATH_03_ACTUAL = pathlib.Path(r"C:\Users\User\.gemini\antigravity\scratch\BAYES_DYN_03\results\actual_interventions.jsonl")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_d1_and_d3_on_v03() -> tuple[dict[str, Any], dict[str, Any]]:
    """D1 (observed vs unobserved input decomposition + signed action error) & D3 (support absorption) on 03's 160 contexts."""
    snaps = np.load(PATH_03_SNAPS, allow_pickle=True)
    exact = np.load(PATH_03_EXACT, allow_pickle=True)
    vecs = np.load(PATH_03_VECS, allow_pickle=True)

    cids = [str(x) for x in snaps["context_ids"]]
    splits = [str(x) for x in snaps["splits"]]
    X_obs_8 = snaps["X_obs_8"].astype(np.float64)        # [160, 8, 38]
    Z_obs_8 = snaps["Z_obs_8"].astype(np.float64)        # [160, 896, 8]
    G_prot = snaps["G_prot"].astype(np.float64)          # [160, 3, 896]
    a_k4 = snaps["a_mean_k4"].astype(np.float64)         # [160, 3, 38]
    a_k8 = snaps["a_mean_k8"].astype(np.float64)         # [160, 3, 38]
    A_ex = exact["A_exact_M4"].astype(np.float64)        # [160, 4, 38]
    X_act = vecs["X_vectors"].astype(np.float64)         # [160, 11, 38]
    m_list = [str(x) for x in vecs["methods"]]

    # Load family mapping from 03 actual_interventions.jsonl
    fam_by_cid: dict[str, str] = {}
    with PATH_03_ACTUAL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            fam_by_cid[r["context_id"]] = r["family"]

    d1_out: dict[str, Any] = {}
    d3_out: dict[str, Any] = {"old_sigma_obs": 0.02, "by_k": {}}

    for k_val, a_hat_all, act_methods in (
        (4, a_k4, ("MAP_RISK_4", "MAP_MEAN_4")),
        (8, a_k8, ("MAP_RISK_8", "MAP_MEAN_8")),
    ):
        par_sq_list, perp_sq_list, tot_sq_list, par_frac_list = [], [], [], []
        by_split: dict[str, dict[str, list[float]]] = {}
        by_fam: dict[str, dict[str, list[float]]] = {}
        # Signed action error contributions e_par^T x, e_perp^T x, (e_par + e_perp)^T x
        signed_act: dict[str, list[dict[str, float]]] = {m: [] for m in act_methods}
        # D3 support observation residual on projected functional response: a_hat^T X - (G_prot @ Z_obs)
        sup_res_abs, sup_res_rel, sup_true_abs, unsup_rms = [], [], [], []

        for c in range(len(cids)):
            X_raw = X_obs_8[c, :k_val, :].T  # [38, k]
            Q_x, _ = np.linalg.qr(X_raw)
            P_X = Q_x @ Q_x.T
            I_m_P = np.eye(38) - P_X
            sp = splits[c]
            fm = fam_by_cid.get(cids[c], "unknown")
            by_split.setdefault(sp, {"par": [], "perp": [], "tot": []})
            by_fam.setdefault(fm, {"par": [], "perp": [], "tot": []})

            # Exact support observations projected to protected readouts: Y_func = G_prot[c] @ Z_obs_8[c, :, :k] [3, k]
            Y_func_obs = G_prot[c] @ Z_obs_8[c, :, :k_val]  # [3, k]
            Y_func_pred = a_hat_all[c] @ X_raw              # [3, k]
            diff_sup = Y_func_obs - Y_func_pred
            sup_res_abs.append(float(np.sqrt(np.mean(diff_sup ** 2))))
            sup_true_abs.append(float(np.sqrt(np.mean(Y_func_obs ** 2))))
            sup_res_rel.append(float(np.linalg.norm(diff_sup) / max(float(np.linalg.norm(Y_func_obs)), 1e-12)))

            c_par, c_perp, c_tot = 0.0, 0.0, 0.0
            for j in range(3):
                e = A_ex[c, j + 1] - a_hat_all[c, j]
                e_par = P_X @ e
                e_perp = I_m_P @ e
                psq = float(np.dot(e_par, e_par))
                qsq = float(np.dot(e_perp, e_perp))
                tsq = float(np.dot(e, e))
                assert abs((psq + qsq) - tsq) < 1e-9 * max(tsq, 1.0)
                c_par += psq
                c_perp += qsq
                c_tot += tsq
                unsup_rms.append(float(np.sqrt(qsq / (38 - k_val))))

                for mn in act_methods:
                    x_m = X_act[c, m_list.index(mn)]
                    s_par = float(np.dot(e_par, x_m))
                    s_perp = float(np.dot(e_perp, x_m))
                    s_tot = float(np.dot(e, x_m))
                    signed_act[mn].append({
                        "e_par_dot_x": s_par,
                        "e_perp_dot_x": s_perp,
                        "e_tot_dot_x": s_tot,
                        "abs_par": abs(s_par),
                        "abs_perp": abs(s_perp),
                        "abs_tot": abs(s_tot),
                        "opposite_sign": int((s_par * s_perp) < 0.0),
                    })

            par_sq_list.append(c_par); perp_sq_list.append(c_perp); tot_sq_list.append(c_tot)
            par_frac_list.append(c_par / max(c_tot, 1e-12))
            by_split[sp]["par"].append(c_par); by_split[sp]["perp"].append(c_perp); by_split[sp]["tot"].append(c_tot)
            by_fam[fm]["par"].append(c_par); by_fam[fm]["perp"].append(c_perp); by_fam[fm]["tot"].append(c_tot)

        act_summary = {}
        for mn, rows in signed_act.items():
            act_summary[mn] = {
                "mean_signed_e_par_dot_x": float(np.mean([r["e_par_dot_x"] for r in rows])),
                "mean_signed_e_perp_dot_x": float(np.mean([r["e_perp_dot_x"] for r in rows])),
                "mean_signed_e_tot_dot_x": float(np.mean([r["e_tot_dot_x"] for r in rows])),
                "mean_abs_e_par_dot_x": float(np.mean([r["abs_par"] for r in rows])),
                "mean_abs_e_perp_dot_x": float(np.mean([r["abs_perp"] for r in rows])),
                "mean_abs_e_tot_dot_x": float(np.mean([r["abs_tot"] for r in rows])),
                "opposite_sign_cancellation_rate": float(np.mean([r["opposite_sign"] for r in rows])),
            }

        d1_out[f"k{k_val}"] = {
            "n_contexts": len(cids),
            "overall": {
                "mean_e_par_sq": float(np.mean(par_sq_list)),
                "mean_e_perp_sq": float(np.mean(perp_sq_list)),
                "mean_e_tot_sq": float(np.mean(tot_sq_list)),
                "overall_par_energy_frac": float(np.sum(par_sq_list) / np.sum(tot_sq_list)),
                "overall_perp_energy_frac": float(np.sum(perp_sq_list) / np.sum(tot_sq_list)),
                "per_ctx_par_frac_quantiles_10_50_90": [float(q) for q in np.percentile(par_frac_list, [10, 50, 90])],
            },
            "by_split": {
                sp: {
                    "n": len(v["tot"]),
                    "mean_e_par_sq": float(np.mean(v["par"])),
                    "mean_e_perp_sq": float(np.mean(v["perp"])),
                    "par_frac": float(np.sum(v["par"]) / np.sum(v["tot"])),
                    "perp_frac": float(np.sum(v["perp"]) / np.sum(v["tot"])),
                } for sp, v in by_split.items()
            },
            "by_family": {
                fm: {
                    "n": len(v["tot"]),
                    "mean_e_par_sq": float(np.mean(v["par"])),
                    "mean_e_perp_sq": float(np.mean(v["perp"])),
                    "par_frac": float(np.sum(v["par"]) / np.sum(v["tot"])),
                    "perp_frac": float(np.sum(v["perp"]) / np.sum(v["tot"])),
                } for fm, v in by_fam.items()
            },
            "signed_action_error_contributions": act_summary,
        }
        d3_out["by_k"][f"k{k_val}"] = {
            "mean_support_func_residual_rms": float(np.mean(sup_res_abs)),
            "median_support_func_residual_rel": float(np.median(sup_res_rel)),
            "mean_support_func_residual_rel": float(np.mean(sup_res_rel)),
            "mean_support_func_signal_rms": float(np.mean(sup_true_abs)),
            "mean_unobserved_per_dim_rms": float(np.mean(unsup_rms)),
            "ratio_support_residual_to_unobs_rms": float(np.mean(sup_res_abs) / max(float(np.mean(unsup_rms)), 1e-12)),
            "jvp_numerical_fd_noise_ref": 1.37e-4,
        }

    return d1_out, d3_out


def run_d2_and_dev_prior_fit(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """D2 (4-block input x output error decomposition on old 02.5 A) + DEV-fit192/DEV-tune32 new full-operator prior."""
    raw = np.load(PATH_025_RAW, allow_pickle=True)
    A_all = raw["A"].astype(np.float64)            # [224, 896, 38]
    phi_all = raw["phi"].astype(np.float64)        # [224, 53]
    V_dark_np = raw["V_dark"].astype(np.float64)   # [896, 38]
    U_old_64 = raw["U_primary"].astype(np.float64) # [896, 64] (old U_hybrid_64!)
    splits_all = [str(x) for x in raw["splits"]]
    cids_all = [str(x) for x in raw["context_ids"]]

    tune_idx = np.array([i for i, s in enumerate(splits_all) if s == "validation"], dtype=np.int64)
    fit_idx = np.array([i for i, s in enumerate(splits_all) if s != "validation"], dtype=np.int64)
    assert len(tune_idx) == 32 and len(fit_idx) == 192, f"Expected 192/32, got {len(fit_idx)}/{len(tune_idx)}"

    phi_fit, A_fit = phi_all[fit_idx], A_all[fit_idx]
    phi_tune, A_tune = phi_all[tune_idx], A_all[tune_idx]

    # Pre-fix 3 orthogonal direction banks on DEV-tune32 using derived seeds (§3.3)
    bank_seeds = [int(s) for s in cfg["derived_seeds"]["dev_tune_banks"]]
    tune_banks_3 = np.zeros((3, len(tune_idx), 38, 38), dtype=np.float64)
    for b_i, b_seed in enumerate(bank_seeds):
        for c_i, idx_g in enumerate(tune_idx):
            tune_banks_3[b_i, c_i] = generate_fixed_38x38_bank(cids_all[idx_g], b_seed, r=38)

    # Fit & select new full-output 896x38 operator prior on DEV-fit192 / DEV-tune32!
    prior_new, sel_report = fit_and_select_full_operator_prior_on_dev(
        phi_fit, A_fit, phi_tune, A_tune, tune_banks_3, cfg
    )
    save_full_operator_prior_npz(ROOT / "results" / "prior_model.npz", prior_new, V_dark_np, U_old_64)

    # D2: 4-block error decomposition comparing OLD (M_old = U_old_64 K_hat, zero output complement) vs NEW (896x38 full)
    # Fit old K_hat [64, 38] on same DEV-fit192 with alpha=100 (matching 02.5/03) and apply old sigma_obs=0.02 update!
    K_fit = np.einsum("dq,ndr->nqr", U_old_64, A_fit)    # [192, 64, 38]
    W_K, xm_K, xs_K, ym_K = _fit_ridge_multi(phi_fit, K_fit.reshape(192, 64 * 38), alpha=100.0)
    K0_tune = _predict_ridge_multi(phi_tune, W_K, xm_K, xs_K, ym_K).reshape(32, 64, 38)

    # Old C_col on K_fit residuals (isotropic/shrunk proxy using same prior C_new for clean structural comparison,
    # PLUS old sigma_obs=0.02 vs exact conditioning!)
    P_U = U_old_64 @ U_old_64.T
    I_m_PU = np.eye(896) - P_U

    def _four_blocks(E: np.ndarray, P_X: np.ndarray) -> dict[str, float]:
        I_m_PX = np.eye(38) - P_X
        b_uu = float(np.sum((P_U @ E @ P_X) ** 2))
        b_up = float(np.sum((P_U @ E @ I_m_PX) ** 2))
        b_pu = float(np.sum((I_m_PU @ E @ P_X) ** 2))
        b_pp = float(np.sum((I_m_PU @ E @ I_m_PX) ** 2))
        tot = float(np.sum(E ** 2))
        assert abs((b_uu + b_up + b_pu + b_pp) - tot) < 1e-6 * max(tot, 1.0)
        return {"inU_obsX": b_uu, "inU_unobsX": b_up, "outU_obsX": b_pu, "outU_unobsX": b_pp, "total_frob_sq": tot}

    d2_out: dict[str, Any] = {}
    M0_new_tune = np.stack([predict_M0(prior_new, phi_tune[c]) for c in range(32)], axis=0)
    for k_val in (0, 4, 8, 16):
        old_blocks, new_blocks = [], []
        for c in range(32):
            Ac = A_tune[c]
            B = tune_banks_3[0, c]
            X_k = B[:, :k_val]
            P_X = X_k @ X_k.T if k_val > 0 else np.zeros((38, 38))
            # Old M_old = U K_k (with old sigma_obs=0.02 in K space!)
            if k_val == 0:
                M_old_k = U_old_64 @ K0_tune[c]
            else:
                Y_K = (U_old_64.T @ Ac) @ X_k
                C_k_prior = prior_new.C
                S_old = X_k.T @ C_k_prior @ X_k + (0.02 ** 2) * np.eye(k_val)
                W_old = np.linalg.solve(S_old, X_k.T @ C_k_prior)
                K_k_old = K0_tune[c] + (Y_K - K0_tune[c] @ X_k) @ W_old
                M_old_k = U_old_64 @ K_k_old
            old_blocks.append(_four_blocks(Ac - M_old_k, P_X))

            # New full 896x38 exact deterministic conditioning
            Y_full = Ac @ X_k if k_val > 0 else np.zeros((896, 0))
            st_new = condition_exact_batch(prior_new, M0_new_tune[c], X_k, Y_full, f"D2_{c}", [f"k{k_val}"])
            new_blocks.append(_four_blocks(Ac - st_new["M_k"], P_X))

        def _avg_blk(blks: list[dict[str, float]]) -> dict[str, float]:
            keys = ["inU_obsX", "inU_unobsX", "outU_obsX", "outU_unobsX", "total_frob_sq"]
            m = {kk: float(np.mean([b[kk] for b in blks])) for kk in keys}
            tot = max(m["total_frob_sq"], 1e-12)
            for kk in keys[:-1]:
                m[f"{kk}_frac"] = m[kk] / tot
            return m

        d2_out[f"k{k_val}"] = {
            "old_U_hybrid_64_zero_comp_sigma002": _avg_blk(old_blocks),
            "new_full_896D_exact_cond": _avg_blk(new_blocks),
        }

    return d2_out, sel_report


def main() -> None:
    cfg = json.loads((ROOT / "configs" / "experiment_config.json").read_text(encoding="utf-8"))

    # 0. Synthetic FP64 exactness verification (§9.2)
    synth_fp64 = verify_full_operator_synth_fp64(int(cfg["master_seed"]))
    assert synth_fp64["passed_1e9"], f"Synthetic FP64 check failed: {synth_fp64}"
    print(f"[P0A SYNTH] FP64 exact conditioning PASSED (max err = {max(v for k, v in synth_fp64.items() if k != 'passed_1e9'):.2e})")

    # 1. Asset hashes
    asset_hashes = {
        "bayes_dyn_02_5/results/raw_transport.npz": sha256_file(PATH_025_RAW),
        "bayes_dyn_02_5/configs/prompts_224.jsonl": sha256_file(PATH_025_PROMPTS),
        "BAYES_DYN_03/results/posterior_snapshots.npz": sha256_file(PATH_03_SNAPS),
        "BAYES_DYN_03/results/exact_reference.npz": sha256_file(PATH_03_EXACT),
        "BAYES_DYN_03/results/intervention_vectors.npz": sha256_file(PATH_03_VECS),
    }
    (ROOT / "development" / "dev_asset_hashes.json").write_text(json.dumps(asset_hashes, indent=2), encoding="utf-8")

    # 2. D1 & D3 on 03
    d1, d3 = run_d1_and_d3_on_v03()
    # 3. D2 & DEV-fit192/DEV-tune32 prior selection
    d2, sel_report = run_d2_and_dev_prior_fit(cfg)

    diag = {
        "experiment_id": "BAYES-DYN-04-P0A",
        "fp64_synthetic_check": synth_fp64,
        "D1_observed_vs_unobserved_input_03": d1,
        "D2_four_block_input_output_025": d2,
        "D3_support_absorption_03": d3,
        "dev_prior_selection": sel_report["selected"],
    }
    (ROOT / "development" / "d1_d2_d3_diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    (ROOT / "development" / "dev_candidate_grid.json").write_text(json.dumps(sel_report, indent=2), encoding="utf-8")
    (ROOT / "configs" / "dev_selection_config.json").write_text(json.dumps({
        "dev_fit_source": "old 224 excluding original validation32 (96 train + 64 test + 32 ood_test = 192)",
        "dev_tune_source": "old original validation32 (32 contexts)",
        "s_A_sq_frozen": sel_report["s_A_sq_frozen"],
        "near_zero_norm_thresh_frozen": sel_report["near_zero_norm_thresh_frozen_p10_tune"],
        "selected": sel_report["selected"],
        "fp64_synthetic_check": synth_fp64,
    }, indent=2), encoding="utf-8")

    # Write concise human-readable D1-D3 summary markdown
    md = [
        "# BAYES-DYN-04 P0A Diagnostics (CPU-Only on Archived 02.5 & 03 Dev Assets)",
        "",
        "## D1 — Observed vs Unobserved Input Subspace Error (BAYES-DYN-03, n=160 contexts)",
        f"- **k=4**: Observed `P_X` share = **{d1['k4']['overall']['overall_par_energy_frac']*100:.2f}%**, Unobserved `(I-P_X)` share = **{d1['k4']['overall']['overall_perp_energy_frac']*100:.2f}%** (per-ctx `P_X` 10/50/90% quantiles: `{[round(x*100,2) for x in d1['k4']['overall']['per_ctx_par_frac_quantiles_10_50_90']]}`).",
        f"- **k=8**: Observed `P_X` share = **{d1['k8']['overall']['overall_par_energy_frac']*100:.2f}%**, Unobserved `(I-P_X)` share = **{d1['k8']['overall']['overall_perp_energy_frac']*100:.2f}%**.",
        f"- **Signed action error (`MAP_RISK_4`)**: mean `|e_par.x| = {d1['k4']['signed_action_error_contributions']['MAP_RISK_4']['mean_abs_e_par_dot_x']:.5f}`, mean `|e_perp.x| = {d1['k4']['signed_action_error_contributions']['MAP_RISK_4']['mean_abs_e_perp_dot_x']:.5f}`, mean `|e_tot.x| = {d1['k4']['signed_action_error_contributions']['MAP_RISK_4']['mean_abs_e_tot_dot_x']:.5f}`, opposite-sign cancellation rate = `{d1['k4']['signed_action_error_contributions']['MAP_RISK_4']['opposite_sign_cancellation_rate']*100:.1f}%`.",
        "",
        "## D2 — Four-Block Input x Output Error Decomposition (DEV-tune32, full 896x38 active operator A)",
        "",
        "| Budget | Model | `inU_obsX` | `inU_unobsX` | `outU_obsX` | `outU_unobsX` | Total Frob² |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for kb in ("k0", "k4", "k8", "k16"):
        bo = d2[kb]["old_U_hybrid_64_zero_comp_sigma002"]
        bn = d2[kb]["new_full_896D_exact_cond"]
        md.append(f"| {kb} | Old `U_hybrid_64` (`sigma=0.02`, 0 comp) | {bo['inU_obsX']:.3f} ({bo['inU_obsX_frac']*100:.1f}%) | {bo['inU_unobsX']:.3f} ({bo['inU_unobsX_frac']*100:.1f}%) | {bo['outU_obsX']:.3f} ({bo['outU_obsX_frac']*100:.1f}%) | {bo['outU_unobsX']:.3f} ({bo['outU_unobsX_frac']*100:.1f}%) | {bo['total_frob_sq']:.3f} |")
        md.append(f"| {kb} | **New Full 896D Exact Cond** | {bn['inU_obsX']:.3f} ({bn['inU_obsX_frac']*100:.1f}%) | {bn['inU_unobsX']:.3f} ({bn['inU_unobsX_frac']*100:.1f}%) | {bn['outU_obsX']:.3f} ({bn['outU_obsX_frac']*100:.1f}%) | {bn['outU_unobsX']:.3f} ({bn['outU_unobsX_frac']*100:.1f}%) | **{bn['total_frob_sq']:.3f}** |")
    md += [
        "",
        "## D3 — Support Observation Absorption (BAYES-DYN-03, `sigma_obs=0.02`)",
        f"- **k=4**: Support residual RMS = `{d3['by_k']['k4']['mean_support_func_residual_rms']:.5f}` (median rel = `{d3['by_k']['k4']['median_support_func_residual_rel']*100:.2f}%`) vs numerical JVP FD error `1.37e-4` vs unobserved per-dim RMS `{d3['by_k']['k4']['mean_unobserved_per_dim_rms']:.5f}`.",
        f"- **k=8**: Support residual RMS = `{d3['by_k']['k8']['mean_support_func_residual_rms']:.5f}` (median rel = `{d3['by_k']['k8']['median_support_func_residual_rel']*100:.2f}%`).",
        "",
        "## DEV-fit192 / DEV-tune32 Full-Operator Selection (3 pre-fixed orthogonal banks)",
        f"- Selected: **`alpha = {sel_report['selected']['alpha']}`**, **`eta = {sel_report['selected']['eta']}`** (frozen `s_A^2 = {sel_report['s_A_sq_frozen']:.4f}`, `near_zero_p10 = {sel_report['near_zero_norm_thresh_frozen_p10_tune']:.4f}`).",
        f"- DEV-tune32 normalized `L_0^perp = {sel_report['selected']['tune_L0_perp_norm']:.5f}` -> `L_8^perp = {sel_report['selected']['tune_L8_perp_norm']:.5f}` (**`-{sel_report['selected']['tune_rel_reduction_k8_vs_k0']*100:.2f}%`** strictly orthogonal full-896D response error reduction!).",
        "",
    ]
    (ROOT / "development" / "d1_d2_d3_summary.md").write_text("\n".join(md), encoding="utf-8")

    print("\n".join(md))


if __name__ == "__main__":
    main()
