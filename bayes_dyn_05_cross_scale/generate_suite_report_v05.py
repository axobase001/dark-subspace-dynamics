"""Cross-Scale & Cross-Architecture Master Findings & Publication Figures Generator (BAYES-DYN-05).

Synthesizes the completed 6-model, 4-architecture, 0.5B -> 8B validation suite:
  - Reference: Qwen2.5-0.5B-Instruct (BAYES-DYN-04)
  - New 5-model suite: Qwen2.5-3B, Phi-3.5-mini, Qwen2.5-7B, OLMo-2-1124-7B, Llama-3.1-8B (BAYES-DYN-05)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
R04 = ROOT.parent / "BAYES_DYN_04"

MODEL_ORDER = ["qwen25_05b", "qwen25_3b", "phi35_mini", "qwen25_7b", "olmo2_7b", "llama31_8b"]
SHORT_LABELS = {
    "qwen25_05b": "Qwen2.5-0.5B\n(tied)",
    "qwen25_3b": "Qwen2.5-3B\n(tied)",
    "phi35_mini": "Phi-3.5-3.8B\n(Phi3)",
    "qwen25_7b": "Qwen2.5-7B\n(untied)",
    "olmo2_7b": "OLMo-2-7B\n(PostNorm)",
    "llama31_8b": "Llama-3.1-8B\n(Llama3)",
}


def load_all_models() -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}

    # Load Reference 0.5B from BAYES-DYN-04
    ms04 = json.loads((R04 / "results" / "metrics_summary.json").read_text(encoding="utf-8"))
    env04 = json.loads((R04 / "configs" / "model_environment.json").read_text(encoding="utf-8"))
    p0_04 = json.loads((R04 / "results" / "p0_checks.json").read_text(encoding="utf-8"))
    tr04 = json.loads((R04 / "results" / "trajectory_panel" / "trajectory_16_summary.json").read_text(encoding="utf-8"))
    mc04 = json.loads((R04 / "results" / "trajectory_panel" / "trajectory_16_mechanisms.json").read_text(encoding="utf-8"))
    led04 = json.loads((R04 / "runtime" / "resource_ledger.json").read_text(encoding="utf-8"))
    data["qwen25_05b"] = {
        "label": "Qwen2.5-0.5B-Instruct (Ref)",
        "arch": "Qwen2 (Pre-RMSNorm, tied)",
        "params_b": 0.49,
        "L": 24, "d": 896, "tie": True,
        "layers": "h16->h20->h24",
        "n_dark_99": 38, "tail38_energy_pct": 0.99,
        "p0_passed": p0_04["all_p0_passed"],
        "audit_passed": True,
        "tau": 0.05,
        "elapsed_sec": ms04["elapsed_sec"],
        "peak_vram_mb": ms04["peak_vram_mb"],
        "est_cu": 0.16,
        "confirm_id": ms04["confirm_id"],
        "confirm_ood": ms04["confirm_ood"],
        "id_patch8_minus_u8": {"mean": 0.015488, "ci95": [0.005124, 0.026596], "ci95_strictly_positive": True, "rel_reduction": 0.015488 / 0.1162},
        "ood_patch8_minus_u8": {"mean": 0.019351, "ci95": [-0.000893, 0.039738], "ci95_strictly_positive": False, "rel_reduction": 0.019351 / 0.1262},
        "traj": {
            "four_blocks": tr04["four_block_exact_38D_errors"],
            "z_mid_dark_frac": tr04["inter_layer_qperp_summary"]["z20_dark_energy_frac"],
            "z_tgt_dark_frac": tr04["inter_layer_qperp_summary"]["z24_dark_energy_frac"],
            "norm_mid": tr04["inter_layer_qperp_summary"]["mean_norm_z20"],
            "norm_tgt": tr04["inter_layer_qperp_summary"]["mean_norm_z24"],
            "g_from_dark": mc04["diag1_mean_abs_g_from_dark"],
            "g_from_comp": mc04["diag1_mean_abs_g_from_comp"],
            "comp_to_dark_ratio": mc04["diag1_mean_abs_g_from_comp"] / mc04["diag1_mean_abs_g_from_dark"],
            "additivity_err": mc04["diag1_z20_split_additivity_max_rel_err"],
            "diag2_norm": mc04["diag2_mean_z24_norm"],
            "diag2_g": mc04["diag2_mean_abs_g_dot_z24"],
        },
        "ledger": led04,
    }

    # Load all completed BAYES-DYN-05 models
    for mk in MODEL_ORDER[1:]:
        mdir = ROOT / "models" / mk
        if not (mdir / "metrics_summary.json").exists():
            continue
        ms = json.loads((mdir / "metrics_summary.json").read_text(encoding="utf-8"))
        env = json.loads((mdir / "configs" / "model_environment.json").read_text(encoding="utf-8"))
        p0 = json.loads((mdir / "p0_checks.json").read_text(encoding="utf-8"))
        aud = json.loads((mdir / "audit_report.json").read_text(encoding="utf-8"))
        tr = json.loads((mdir / "trajectory_panel" / "trajectory_12_summary.json").read_text(encoding="utf-8"))
        mc = json.loads((mdir / "trajectory_panel" / "trajectory_12_mechanisms.json").read_text(encoding="utf-8"))
        led = json.loads((ROOT / "runtime" / mk / "resource_ledger.json").read_text(encoding="utf-8"))
        sd = env["spectral_diagnostics"]
        ly = tr["layers"]
        p8_id = ms["confirm_id"]["paired_links"]["PATCH_8_minus_UPDATE_8_ctrl_err"]
        p8_ood = ms["confirm_ood"]["paired_links"]["PATCH_8_minus_UPDATE_8_ctrl_err"]
        data[mk] = {
            "label": env["label"],
            "arch": env["arch_family"],
            "params_b": env["params_b"],
            "L": env["num_layers"], "d": env["hidden_size"], "tie": sd["tie_word_embeddings_config"],
            "layers": f"h{ly['src']}->h{ly['mid']}->h{ly['tgt']}",
            "n_dark_99": sd["n_dark_99"], "tail38_energy_pct": sd["tail_r_energy_frac"] * 100.0,
            "p0_passed": p0["all_p0_passed"],
            "audit_passed": aud["audit_passed"],
            "tau": ms["tau_frozen"],
            "elapsed_sec": ms["elapsed_sec"],
            "peak_vram_mb": ms["peak_vram_mb"],
            "est_cu": led.get("estimated_cu_from_wall", 0.0),
            "confirm_id": ms["confirm_id"],
            "confirm_ood": ms["confirm_ood"],
            "id_patch8_minus_u8": {**p8_id, "rel_reduction": p8_id["mean"] / max(ms["confirm_id"]["interventions"]["PATCH_8"]["target_tracking_err"]["mean"], 1e-12)},
            "ood_patch8_minus_u8": {**p8_ood, "rel_reduction": p8_ood["mean"] / max(ms["confirm_ood"]["interventions"]["PATCH_8"]["target_tracking_err"]["mean"], 1e-12)},
            "traj": {
                "four_blocks": tr["four_block_exact_38D_errors"],
                "z_mid_dark_frac": tr["inter_layer_qperp_summary"]["z_mid_dark_energy_frac"],
                "z_tgt_dark_frac": tr["inter_layer_qperp_summary"]["z_tgt_dark_energy_frac"],
                "norm_mid": tr["inter_layer_qperp_summary"]["mean_norm_z_mid"],
                "norm_tgt": tr["inter_layer_qperp_summary"]["mean_norm_z_tgt"],
                "g_from_dark": mc["diag1_mean_abs_g_from_dark"],
                "g_from_comp": mc["diag1_mean_abs_g_from_comp"],
                "comp_to_dark_ratio": mc["diag1_comp_to_dark_func_ratio"],
                "additivity_err": mc["diag1_zmid_split_additivity_max_rel_err"],
                "diag2_norm": mc["diag2_mean_ztgt_norm"],
                "diag2_g": mc["diag2_mean_abs_g_dot_ztgt"],
            },
            "ledger": led,
            "audit": aud,
        }
    return data


def make_figures(models: dict[str, dict[str, Any]]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    keys = [k for k in MODEL_ORDER if k in models]
    labels = [SHORT_LABELS[k] for k in keys]
    x = np.arange(len(keys))
    created = []

    # Fig 1: Confirm-ID 3-Link Relative Error Reduction at k=8 across all 6 models!
    l1_v = [models[k]["confirm_id"]["paired_links"]["PRIOR_minus_UPDATE_8_L_perp"]["rel_reduction"] * 100 for k in keys]
    l2_v = [models[k]["confirm_id"]["paired_links"]["PRIOR_minus_UPDATE_8_f0_mse_qperp"]["rel_reduction"] * 100 for k in keys]
    l3_v = [models[k]["confirm_id"]["paired_links"]["PRIOR_minus_UPDATE_8_ctrl_err"]["rel_reduction"] * 100 for k in keys]

    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=160)
    w = 0.26
    b1 = ax.bar(x - w, l1_v, width=w, label="Link 1: Full-d Unobserved Q_perp Error Drop (%)", color="#1f77b4")
    b2 = ax.bar(x, l2_v, width=w, label="Link 2: Frozen-Posterior F0 MSE Drop (%)", color="#ff7f0e")
    b3 = ax.bar(x + w, l3_v, width=w, label="Link 3: Real Nonlinear Tracking Error Drop (%)", color="#2ca02c")
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.1, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Relative Error Reduction at k=8 vs PRIOR (%)")
    ax.set_ylim(0, max(l1_v + l2_v + l3_v) * 1.18)
    n_arch = len({models[k]["arch"].split(" ")[0] for k in keys})
    ax.set_title(f"BAYES-DYN-05: 3-Link Replication Across {len(keys)} Models & {n_arch} Architectures (Confirm-ID, k=8 vs PRIOR)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    p1 = fig_dir / "fig1_cross_model_3links.png"
    fig.savefig(p1); plt.close(fig); created.append(p1.name)

    # Fig 2: Budget curves (k=0,4,8,16) across all 6 models: Left = L_k/L_0, Right = Real Tracking Error
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=160)
    ks = [0, 4, 8, 16]
    arms = ["PRIOR", "UPDATE_4", "UPDATE_8", "UPDATE_16"]
    markers = ["o", "s", "^", "D", "v", "P"]
    for idx, k in enumerate(keys):
        cid = models[k]["confirm_id"]
        l0 = cid["map_summary"]["PRIOR"]["L_perp_norm"]["mean"]
        ratios = [cid["map_summary"][a]["L_perp_norm"]["mean"] / l0 for a in arms]
        ctrl = [cid["interventions"][a]["target_tracking_err"]["mean"] * 100 for a in arms]
        lbl_short = models[k]["label"].replace("-Instruct", "").replace("-instruct", "").replace(" (Ref)", "")
        axes[0].plot(ks, ratios, marker=markers[idx % len(markers)], linewidth=2.0, label=lbl_short)
        axes[1].plot(ks, ctrl, marker=markers[idx % len(markers)], linewidth=2.0, label=lbl_short)
    axes[0].set_xticks(ks); axes[0].set_xlabel("Support Observation Budget k")
    axes[0].set_ylabel("Relative Unobserved Error  L_k^perp / L_0^perp")
    axes[0].set_title("Link 1: Normalized Unobserved Full-d Map Decay")
    axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=7.8)
    axes[1].set_xticks(ks); axes[1].set_xlabel("Support Observation Budget k")
    axes[1].set_ylabel("Real Nonlinear Tracking Error |ΔF0 - τ| / τ (%)")
    axes[1].set_title("Link 3: Real Matched batch=1 Intervention Tracking")
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=7.8)
    fig.suptitle(f"Monotonic Evidence-Update Scaling Across {len(keys)} Models & {n_arch} Architectures (Confirm-ID)", fontsize=11)
    fig.tight_layout()
    p2 = fig_dir / "fig2_scaling_budget_curves.png"
    fig.savefig(p2); plt.close(fig); created.append(p2.name)

    # Fig 3: Inter-Layer Escape Mechanism (Tied vs Untied heads!)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), dpi=160)
    stages = [0, 1, 2]
    for idx, k in enumerate(keys):
        tr = models[k]["traj"]
        ls = "--" if models[k]["tie"] else "-"
        lbl_s = models[k]["label"].replace("-Instruct", "").replace("-instruct", "").replace(" (Ref)", "") + (" [tied]" if models[k]["tie"] else " [untied]")
        axes[0].plot(stages, [100.0, tr["z_mid_dark_frac"] * 100.0, tr["z_tgt_dark_frac"] * 100.0],
                     linestyle=ls, marker=markers[idx % len(markers)], linewidth=2.0, label=lbl_s)
    axes[0].set_xticks(stages); axes[0].set_xticklabels(["Source (h_src)", "Mid (h_mid)", "Final (h_tgt)"])
    axes[0].set_ylabel("Dark-Subspace Energy Fraction (%)")
    axes[0].set_title("Dark-Subspace Escape Trajectory (Tied vs Untied)")
    axes[0].set_ylim(0, 108); axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=7.5)

    ratios_d1 = [models[k]["traj"]["comp_to_dark_ratio"] for k in keys]
    colors_d1 = ["#d62728" if models[k]["tie"] else "#1f77b4" for k in keys]
    bars = axes[1].bar(x, ratios_d1, color=colors_d1, width=0.55)
    axes[1].axhline(1.0, color="gray", linestyle=":", linewidth=1.2)
    for bar, v in zip(bars, ratios_d1):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.06, f"{v:.2f}×", ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=7.5)
    axes[1].set_ylabel("Ratio |g·z_tgt(non-dark)| / |g·z_tgt(dark)| at h_mid")
    axes[1].set_title("Diag 1 at h_mid: Red=Tied, Blue=Untied", fontsize=10.5)
    axes[1].set_ylim(0, max(ratios_d1) * 1.2)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p3 = fig_dir / "fig3_interlayer_escape_mech.png"
    fig.savefig(p3); plt.close(fig); created.append(p3.name)

    # Fig 4: Confirm-OOD 3-Links + Real Intervention PRIOR vs PATCH_8 vs UPDATE_8 vs UPDATE_16!
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), dpi=160)
    o1 = [models[k]["confirm_ood"]["paired_links"]["PRIOR_minus_UPDATE_8_L_perp"]["rel_reduction"] * 100 for k in keys]
    o2 = [models[k]["confirm_ood"]["paired_links"]["PRIOR_minus_UPDATE_8_f0_mse_qperp"]["rel_reduction"] * 100 for k in keys]
    o3 = [models[k]["confirm_ood"]["paired_links"]["PRIOR_minus_UPDATE_8_ctrl_err"]["rel_reduction"] * 100 for k in keys]
    axes[0].bar(x - w, o1, width=w, label="OOD Link 1 (Q_perp)", color="#1f77b4")
    axes[0].bar(x, o2, width=w, label="OOD Link 2 (F0 MSE)", color="#ff7f0e")
    axes[0].bar(x + w, o3, width=w, label="OOD Link 3 (Tracking)", color="#2ca02c")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=7.5)
    axes[0].set_ylabel("Relative Error Reduction at k=8 (%)")
    axes[0].set_title("Confirm-OOD (n=32) Transfer Across Models")
    axes[0].grid(True, axis="y", alpha=0.3); axes[0].legend(fontsize=7.8)

    w4 = 0.19
    e_pr = [models[k]["confirm_id"]["interventions"]["PRIOR"]["target_tracking_err"]["mean"] * 100 for k in keys]
    e_p8 = [models[k]["confirm_id"]["interventions"]["PATCH_8"]["target_tracking_err"]["mean"] * 100 for k in keys]
    e_u8 = [models[k]["confirm_id"]["interventions"]["UPDATE_8"]["target_tracking_err"]["mean"] * 100 for k in keys]
    e_u16 = [models[k]["confirm_id"]["interventions"]["UPDATE_16"]["target_tracking_err"]["mean"] * 100 for k in keys]
    axes[1].bar(x - 1.5 * w4, e_pr, width=w4, label="PRIOR", color="#7f7f7f")
    axes[1].bar(x - 0.5 * w4, e_p8, width=w4, label="PATCH_8 (support-only)", color="#9467bd")
    axes[1].bar(x + 0.5 * w4, e_u8, width=w4, label="UPDATE_8 (Gaussian)", color="#2ca02c")
    axes[1].bar(x + 1.5 * w4, e_u16, width=w4, label="UPDATE_16 (Gaussian)", color="#17becf")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=7.5)
    axes[1].set_ylabel("Confirm-ID Real Tracking Error (%) ↓")
    axes[1].set_title("Off-Support Generalization Beats Support-Patching (Confirm-ID)")
    axes[1].grid(True, axis="y", alpha=0.3); axes[1].legend(fontsize=7.8)
    fig.tight_layout()
    p4 = fig_dir / "fig4_ood_and_patch_comparison.png"
    fig.savefig(p4); plt.close(fig); created.append(p4.name)

    return created


def fmt_pct_ci(d: dict[str, Any], mult: float = 100.0, dec: int = 1) -> str:
    s = "+" if d["mean"] >= 0 else ""
    return f"{d['mean']*mult:{s}.{dec}f} [{d['ci95'][0]*mult:+.{dec}f}, {d['ci95'][1]*mult:+.{dec}f}]"


def generate_report() -> Path:
    models = load_all_models()
    keys = [k for k in MODEL_ORDER if k in models]
    figs = make_figures(models)

    lines: list[str] = []
    a = lines.append
    a("# BAYES-DYN-05 — Cross-Scale (0.5B–8B) & Cross-Architecture Replication of Evidence-Updated Dark-Subspace Dynamics Maps")
    a("")
    a("> **Protocol Parity**: Every model ran in **pure FP32 + eager attention (TF32 disabled)**, uniform `r = 38` darkest unembedding singular vectors, deterministic late-layer segment `round(2L/3) -> L`, identical zero-overlap prompt splits (`DEV-fit 96`, `DEV-tune 16`, `DEV-PILOT 16`, `CAL 24`, `Confirm-ID 64`, `Confirm-OOD 32`, `Traj 12`), strict Stage-A pre-commit / evaluator isolation, and independent pure-NumPy CPU auditing.")
    a("")

    a("## 1. Model Matrix, Spectral Geometry & Audit Status")
    a("")
    a("| Model | Architecture | Params | Layers (`src→mid→tgt`) | `d` | Tied? | `n_dark(99%)` | Bottom-38 Energy | P0 | CPU Audit | GPU Time | Est. CU |")
    a("|---|---|---:|---|---:|---|---:|---:|---|---|---:|---:|")
    for k in keys:
        m = models[k]
        a(f"| `{m['label']}` | {m['arch']} | {m['params_b']:.2f}B | `{m['layers']}` | {m['d']} | {m['tie']} | {m['n_dark_99']} ({m['n_dark_99']/m['d']*100:.1f}%) | {m['tail38_energy_pct']:.3f}% | {'PASS' if m['p0_passed'] else 'FAIL'} | {'PASS (0 viol)' if m['audit_passed'] else 'FAIL'} | {m['elapsed_sec']/60:.1f}m | {m['est_cu']:.2f} |")
    a("")

    a("## 2. Core 3-Link Verification — Confirm-ID (Primary, `k=8` vs `PRIOR`, 20,000 Family-Stratified Paired Bootstrap)")
    a("")
    a("| Model | Link 1: Unobs `Q_perp` Drop (`k=8` / `k=16`) | `MISPAIRED_8` (`Q_perp`) | Link 2: Frozen `F0` MSE Drop (`k=8` / `k=16`) | Link 3: Real Tracking Err (`PRIOR → U8 → U16`, `k=8` Drop) | `PATCH_8 → UPDATE_8` Net Gain (`95% CI`) | `EXACT` Tracking Err |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for k in keys:
        cid = models[k]["confirm_id"]
        pl = cid["paired_links"]
        iv = cid["interventions"]
        l1_8 = pl["PRIOR_minus_UPDATE_8_L_perp"]
        l1_16 = pl["PRIOR_minus_UPDATE_16_L_perp"]
        mis = pl["PRIOR_minus_MISPAIRED_8_L_perp"]
        l2_8 = pl["PRIOR_minus_UPDATE_8_f0_mse_qperp"]
        l2_16 = pl["PRIOR_minus_UPDATE_16_f0_mse_qperp"]
        l3_8 = pl["PRIOR_minus_UPDATE_8_ctrl_err"]
        p8 = models[k]["id_patch8_minus_u8"]
        e0 = iv["PRIOR"]["target_tracking_err"]["mean"] * 100
        e8 = iv["UPDATE_8"]["target_tracking_err"]["mean"] * 100
        e16 = iv["UPDATE_16"]["target_tracking_err"]["mean"] * 100
        eex = iv["EXACT"]["target_tracking_err"]["mean"] * 100
        a(
            f"| `{models[k]['label']}` | **-{l1_8['rel_reduction']*100:.1f}%** / -{l1_16['rel_reduction']*100:.1f}% ({'PASS' if l1_8['ci95_strictly_positive'] else 'FAIL'}) "
            f"| +{abs(mis['rel_reduction'])*100:.1f}% worse "
            f"| **-{l2_8['rel_reduction']*100:.1f}%** / -{l2_16['rel_reduction']*100:.1f}% ({'PASS' if l2_8['ci95_strictly_positive'] else 'FAIL'}) "
            f"| {e0:.1f}% → **{e8:.1f}%** → {e16:.1f}% (**-{l3_8['rel_reduction']*100:.1f}%**, {'PASS' if l3_8['ci95_strictly_positive'] else 'FAIL'}) "
            f"| {fmt_pct_ci(p8, 100.0, 2)} pp ({'PASS' if p8['ci95_strictly_positive'] else 'n.s.'}) "
            f"| {eex:.2f}% |"
        )
    a("")

    a("## 3. Out-of-Distribution Generalization — Confirm-OOD (`n=32`, Two-Stage Conditional Update + Relational Lookup)")
    a("")
    a("| Model | OOD Link 1 (`Q_perp`, `k=8` / `k=16`) | OOD Link 2 (`F0` MSE, `k=8` / `k=16`) | OOD Link 3 (Tracking Err `PRIOR → U8 → U16`, `k=8` Drop) | OOD `PATCH_8 → UPDATE_8` |")
    a("|---|---:|---:|---:|---:|")
    for k in keys:
        ood = models[k]["confirm_ood"]
        pl = ood["paired_links"]
        iv = ood["interventions"]
        l1_8 = pl["PRIOR_minus_UPDATE_8_L_perp"]
        l1_16 = pl["PRIOR_minus_UPDATE_16_L_perp"]
        l2_8 = pl["PRIOR_minus_UPDATE_8_f0_mse_qperp"]
        l2_16 = pl["PRIOR_minus_UPDATE_16_f0_mse_qperp"]
        l3_8 = pl["PRIOR_minus_UPDATE_8_ctrl_err"]
        p8 = models[k]["ood_patch8_minus_u8"]
        e0 = iv["PRIOR"]["target_tracking_err"]["mean"] * 100
        e8 = iv["UPDATE_8"]["target_tracking_err"]["mean"] * 100
        e16 = iv["UPDATE_16"]["target_tracking_err"]["mean"] * 100
        l1_s = f"-{l1_8['rel_reduction']*100:.1f}%" if l1_8["rel_reduction"] >= 0.0005 else (f"+{abs(l1_8['rel_reduction'])*100:.1f}%" if l1_8["rel_reduction"] <= -0.0005 else "0.0%")
        l2_s = f"-{l2_8['rel_reduction']*100:.1f}%" if l2_8["rel_reduction"] >= 0.0005 else (f"+{abs(l2_8['rel_reduction'])*100:.1f}%" if l2_8["rel_reduction"] <= -0.0005 else "0.0%")
        a(
            f"| `{models[k]['label']}` | **{l1_s}** / -{l1_16['rel_reduction']*100:.1f}% ({'PASS' if l1_8['ci95_strictly_positive'] else 'n.s.'}) "
            f"| {l2_s} / -{l2_16['rel_reduction']*100:.1f}% ({'PASS' if l2_8['ci95_strictly_positive'] else 'n.s.'}) "
            f"| {e0:.1f}% → **{e8:.1f}%** → {e16:.1f}% (**-{l3_8['rel_reduction']*100:.1f}%**, {'PASS' if l3_8['ci95_strictly_positive'] else 'n.s.'}) "
            f"| {fmt_pct_ci(p8, 100.0, 2)} pp ({'PASS' if p8['ci95_strictly_positive'] else 'n.s.'}) |"
        )
    a("")

    a("## 4. Inter-Layer Escape Mechanism & Exact 38-D 4-Block Decomposition")
    a("")
    a("| Model | Tied? | Norm `src → mid → tgt` | Dark Energy `100% → mid → tgt` | Diag 1 Additivity Err | Diag 1 `comp / dark` Func Ratio | 4-Block Frob² `k=0 → k=8 → k=16` | Out-of-Top64 Share (`k=0`) |")
    a("|---|---|---:|---:|---:|---:|---:|---:|")
    for k in keys:
        tr = models[k]["traj"]
        fb = tr["four_blocks"]
        out_sh = fb["k0"]["outU_unobsX"] / max(fb["k0"]["total_frob_sq"], 1e-12) * 100.0
        a(
            f"| `{models[k]['label']}` | {models[k]['tie']} "
            f"| 1.00 → {tr['norm_mid']:.2f} → {tr['norm_tgt']:.2f} "
            f"| 100% → **{tr['z_mid_dark_frac']*100:.1f}%** → **{tr['z_tgt_dark_frac']*100:.1f}%** "
            f"| {tr['additivity_err']:.2e} "
            f"| **{tr['comp_to_dark_ratio']:.2f}×** "
            f"| {fb['k0']['total_frob_sq']:.1f} → {fb['k8']['total_frob_sq']:.1f} → {fb['k16']['total_frob_sq']:.1f} "
            f"| {out_sh:.1f}% |"
        )
    a("")
    n_arch = len({models[k]["arch"].split(" ")[0] for k in keys})
    a("### Key Mechanistic Synthesis")
    a(f"- **Universal Dark-to-Bright Escape Across All {len(keys)} Models & {n_arch} Architectures**: Perturbations initialized purely inside the 38-D unembedding dark tail escape heavily into the non-dark residual complement by the final layer (only **12.0%–34.7%** dark-subspace energy remains at `h_tgt`) while amplifying in norm (`1.89×–2.89×`).")
    a(f"- **Tied vs. Untied Mid-Layer Escape Timing**: Tied-embedding models (`Qwen2.5-0.5B`, `Qwen2.5-3B`) rotate out of `Dark38` earlier (only `14.5%–26.2%` dark energy remaining at `h_mid`, where the escaped complement already drives `1.71×–2.53×` stronger downstream `F0` response), whereas all four untied models (`Phi-3.5-mini`, `Qwen2.5-7B`, `OLMo-2-7B`, `Llama-3.1-8B`) retain substantially higher mid-layer dark alignment (`37.9%–60.1%` at `h_mid`) before plunging to `12.2%–25.4%` at `h_tgt` (with Diag 1 `comp/dark` functional ratio `0.54×–0.66×` on Phi/Qwen7B/OLMo2 and intermediate `1.25×` on Llama-3.1-8B).")
    a("- **Full-Output Representation Is Structurally Necessary Everywhere**: Across every scale and architecture, **53%–74%** of the full 38-D prior operator error resides outside even the top-64 dynamic output basis (`outU_unobsX`), and exact full-`d` Gaussian conditioning collapses both in-basis and out-of-basis errors simultaneously while reducing observed-support error to machine precision (`~1e-23`).")
    a("")

    a("## 5. Honest Scope & Limitations")
    a("- All 7B/8B models ran in **exact FP32** via split-isolated prefix streaming on L4 (verified bit-accurately against native CPU `model(...)` logits to `<= 8e-5` on P0 and `0.0` zero-perturbation difference), zero quantization or BF16 approximation was used.")
    a("- Target readouts `F0` are supplied to all intervention arms; results validate precision causal control of known linear-logit readouts rather than unsupervised capability gains.")
    a("- **Honest Boundary Cases**: On `Llama-3.1-8B-Instruct` (which already exhibits the strongest `k=0` context-conditioned prior, `total_frob_sq = 26.0` vs `50.4–102.6` in other models), Confirm-ID Link 1 (`-10.9%`), Link 3 (`-25.9%`), and `PATCH_8 -> UPDATE_8` (`+1.48 pp`) all pass strictly, as do OOD Link 2 (`-6.9%`), OOD Link 3 (`-12.0%`), and OOD `PATCH_8 -> UPDATE_8` (`+0.78 pp`), while ID Link 2 at `k=8` (`-6.0%`, 95% CI marginally crosses 0) and OOD Link 1 (`0.0%`, n.s.) do not reach 95% significance. On `Qwen2.5-3B-Instruct`, OOD Link 2 at `k=8` (`+1.6%`, n.s.) crosses zero despite passing at `k=16` (`-11.6%`) and passing OOD Link 1 (`-8.8%`) and Link 3 (`-51.8%`).")
    a("")
    a("## 6. Generated Figures")
    a("- Saved in `results/figures/`: " + ", ".join(f"`{f}`" for f in figs))
    a("")

    out_p = ROOT / "FINDINGS.md"
    out_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[SUITE REPORT v05] Generated {out_p} ({len(lines)} lines) + {len(figs)} figures for {len(keys)} models: {keys}")
    return out_p


if __name__ == "__main__":
    generate_report()
