"""BAYES-DYN-04 — Complete End-to-End GPU Experimental Runner (§5–§10).

Uses the pre-frozen CPU full-operator prior (`results/prior_model.npz`) and brand-new datasets:
  1. Strict P0 Verification (zero hook, matched batch=1, JVP FD curve, adjoint, linearity, mid-layer z20,
     archived conditioning <= 1e-6, ProbeSession isolation, and truth-replacement sentinel check).
  2. DEV-PILOT (16 contexts): freeze working target tau in {0.05, 0.02, 0.01} via EXACT reference + throughput check.
  3. CAL (32 contexts): fit single global temperature t > 0 on k=8 Q_perp full 896-D errors -> freeze configs/frozen_config.json.
  4. Confirm-ID (96 contexts) & Confirm-OOD (32 contexts):
       Stage A (zero truth / zero source VJP): k=0, 4, 8, 16 real JVP updates + PATCH_8 + MISPAIRED_8 -> commit maps,
       predictions on Q_perp & Q_general, and functional projections (F0 + A-D/B-D/C-D) + min-norm ball actions + common-action predictions.
       Stage B (strictly AFTER Stage A commit): reveal Q_perp (8), Q_general (4), 1 exact F0 VJP -> run matched batch=1
       real nonlinear forward passes for ZERO, PRIOR, UPDATE_4, UPDATE_8, UPDATE_16, PATCH_8, EXACT.
  5. Trajectory Panel (16 pre-registered Confirm-ID contexts):
       Simultaneous z20/z24 captured during Step 4 -> complete cols 25..38 after all commits -> exact 38-D 4-block decomposition ->
       within 45-min cutoff run Diag 1 (z20 dark/comp split -> h20->h24 JVP additivity) & Diag 2 (Bright4 / RandOrth4 vs Dark4).
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.spectral_basis import compute_spectral_basis
from src.transport import (
    tokenize_chat_prompt,
    resolve_option_token_ids,
    capture_residual_states,
    run_segment_layers,
    compute_exact_jvp_columns_batched,
    extract_context_features,
)
from src.full_operator_map import (
    load_full_operator_prior_npz,
    predict_M0,
    condition_exact_batch,
    condition_patch_only,
    make_mispaired_Y,
    project_function_from_posterior,
    predict_scalar_direction_dist,
    fit_cal_temperature,
    verify_full_operator_synth_fp64,
)
from src.probe_bank_v04 import (
    ProbeSessionV04,
    BudgetExceededError,
    UnauthorizedQueryError,
    generate_fixed_38x38_bank,
    generate_q_general,
    utc_now,
    bytes_sha256_short,
)
from src.control_v04 import (
    extract_v04_readouts,
    solve_min_norm_ball_action,
    evaluate_v04_intervention_batch1,
    family_stratified_paired_bootstrap,
)


Z_SCORES = {0.50: 0.6744897501960817, 0.80: 1.2815515655446004, 0.90: 1.6448536269514722, 0.95: 1.959963984540054}


class HeartbeatTracker:
    def __init__(self, path: Path, t0: float) -> None:
        self.path = path
        self.t0 = t0
        self.last_write = 0.0
        self.phase = "INIT"

    def beat(self, phase: str, step: int, total: int, status: str = "RUNNING", note: str = "", force: bool = False) -> None:
        now = time.time()
        if (not force) and (phase == self.phase) and (now - self.last_write < 15.0):
            return
        self.phase = phase
        self.last_write = now
        payload = {
            "utc": utc_now(), "epoch": now, "elapsed_sec": round(now - self.t0, 2),
            "phase": phase, "step": int(step), "total": int(total), "status": status, "note": note,
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def compute_full_weight_sha256(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        h.update(name.encode("utf-8"))
        arr = p.detach().cpu().contiguous()
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(str(tuple(arr.shape)).encode("utf-8"))
        h.update(arr.numpy().tobytes())
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scalar_gaussian_nll(y_true: float, mean: float, std: float, var_floor: float = 1e-12) -> float:
    var = max(std * std, var_floor)
    return float(0.5 * (math.log(2.0 * math.pi * var) + ((y_true - mean) ** 2) / var))


def vector_dir_metrics(y_pred: np.ndarray, y_true: np.ndarray, near_zero_thresh: float) -> dict[str, Any]:
    diff = y_pred - y_true
    sq_l2 = float(np.sum(diff ** 2))
    nt = float(np.linalg.norm(y_true))
    np_ = float(np.linalg.norm(y_pred))
    is_nz = bool(nt < near_zero_thresh)
    cos = float(np.dot(y_pred, y_true) / max(np_ * nt, 1e-12)) if (np_ > 1e-12 and nt > 1e-12) else None
    gain_ratio = float(np_ / nt) if (not is_nz and nt > 1e-12) else None
    return {
        "sq_l2": sq_l2, "abs_l2": math.sqrt(sq_l2), "true_norm": nt, "pred_norm": np_,
        "is_near_zero": is_nz, "cosine": cos, "gain_ratio": gain_ratio,
    }


def run_p0_checks_v04(
    model: torch.nn.Module,
    tokenizer: Any,
    basis: Any,
    prior: Any,
    V_dark_np: np.ndarray,
    U_leg12_np: np.ndarray,
    pilot_4: list[dict[str, Any]],
    option_token_ids: dict[str, int],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Comprehensive §9.2 P0 verification on 4 DEV-PILOT contexts (all 4 ID families)."""
    device = next(model.parameters()).device
    W_U = model.lm_head.weight.detach()
    gamma32 = basis.rms_gamma.to(device=device, dtype=torch.float32)
    V_dark = basis.V_dark.to(device=device, dtype=torch.float32)
    src_l, mid_l, tgt_l = int(cfg["layers"]["source_layer"]), int(cfg["layers"]["mid_layer"]), int(cfg["layers"]["target_layer"])
    th = cfg["p0_thresholds"]
    fd_steps = [float(s) for s in th["fd_step_sizes"]]
    abs_floor = float(th["near_zero_denom_abs_floor"])
    bank_seed = int(cfg["derived_seeds"]["probe_bank_38x38"])

    synth_fp64 = verify_full_operator_synth_fp64(int(cfg["master_seed"]))
    zero_D, zero_dF, zero_hook_diffs = [], [], []
    jvp_b8_vs_b1, lin_comb_errs, repeat_errs, mid_z20_errs, adjoint_errs = [], [], [], [], []
    jvp_fd_by_step: dict[str, list[dict[str, float]]] = {str(s): [] for s in fd_steps}
    archived_cond_errs = []

    for idx, rec in enumerate(pilot_4):
        tok = tokenize_chat_prompt(tokenizer, rec["user_prompt"], device)
        states, logits = capture_residual_states(model, tok["input_ids"], tok["attention_mask"])
        t = tok["t_last"]
        h_s, h_m, h_L = states[src_l], states[mid_l], states[tgt_l][0, t, :]

        # Zero-hook match with native model forward
        with torch.no_grad():
            nat = model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"], use_cache=False)
            zero_hook_diffs.append(float(torch.max(torch.abs(nat.logits[0, t, :] - logits)).item()))

        ro = extract_v04_readouts(logits, h_L, W_U, gamma32, basis.rms_eps, option_token_ids, rec["target_option"])
        ev0 = evaluate_v04_intervention_batch1(
            model, h_s, h_L, logits, src_l, tgt_l, t, V_dark,
            np.zeros(basis.n_dark, dtype=np.float64), ro, option_token_ids, rec["target_option"], tau=0.05,
        )
        zero_D.append(ev0["actual_D_geom"])
        zero_dF.append(abs(ev0["actual_delta_F0"]))

        B = generate_fixed_38x38_bank(rec["context_id"], bank_seed, r=38)
        X8 = B[:, :8]
        sess_p0 = ProbeSessionV04(f"p0_{idx}", "evaluator_truth", rec["context_id"], model, h_s, V_dark, src_l, mid_l, tgt_l, t, max_jvp_dirs=32, allow_vjp=True, jvp_batch_size=8)
        Z24_b8, Z20_b8 = sess_p0.jvp_columns(X8, "p0_b8", record_mid_z20=True)
        # Repeatability on first 2 supports
        Z24_rep, _ = sess_p0.jvp_columns(X8[:, :2], "p0_rep", record_mid_z20=False)
        repeat_errs.append(float(np.linalg.norm(Z24_rep - Z24_b8[:, :2], "fro") / max(float(np.linalg.norm(Z24_b8[:, :2], "fro")), abs_floor)))
        # Batch=1 reference on first 2 cols
        Z24_b1_0, _ = sess_p0.jvp_columns(X8[:, 0:1], "p0_b1_0", record_mid_z20=False)
        Z24_b1_1, _ = sess_p0.jvp_columns(X8[:, 1:2], "p0_b1_1", record_mid_z20=False)
        Z24_b1 = np.concatenate([Z24_b1_0, Z24_b1_1], axis=1)
        jvp_b8_vs_b1.append(float(np.linalg.norm(Z24_b8[:, :2] - Z24_b1, "fro") / max(float(np.linalg.norm(Z24_b1, "fro")), abs_floor)))

        # Linear combination consistency: J(0.6 v0 - 0.8 v1) == 0.6 J v0 - 0.8 J v1
        x_lc = 0.6 * X8[:, 0:1] - 0.8 * X8[:, 1:2]
        Z_lc, _ = sess_p0.jvp_columns(x_lc, "p0_lincomb", record_mid_z20=False)
        Z_lc_ref = 0.6 * Z24_b8[:, 0:1] - 0.8 * Z24_b8[:, 1:2]
        lin_comb_errs.append(float(np.linalg.norm(Z_lc - Z_lc_ref) / max(float(np.linalg.norm(Z_lc_ref)), abs_floor)))

        # Mid-layer z20 consistency: J_{20->24}(z20) == z24!
        z20_t = torch.tensor(Z20_b8[:, 0:1], dtype=torch.float32, device=device)
        z24_from_z20 = compute_exact_jvp_columns_batched(model, h_m, mid_l, tgt_l, t, z20_t, batch_size=1)[:, 0].to(torch.float64).cpu().numpy()
        mid_z20_errs.append(float(np.linalg.norm(z24_from_z20 - Z24_b8[:, 0]) / max(float(np.linalg.norm(Z24_b8[:, 0])), abs_floor)))

        # Central FD curve on col 0
        hs_scale = float(torch.linalg.vector_norm(h_s[0, t, :]).item())
        v0 = V_dark @ torch.tensor(X8[:, 0], dtype=torch.float32, device=device)
        for step in fd_steps:
            eps_s = step * hs_scale
            hp = h_s.clone(); hp[0, t, :] = hp[0, t, :] + eps_s * v0
            hm = h_s.clone(); hm[0, t, :] = hm[0, t, :] - eps_s * v0
            with torch.no_grad():
                yp = run_segment_layers(model, hp, src_l, tgt_l)[0, t, :].to(torch.float64).cpu().numpy()
                ym = run_segment_layers(model, hm, src_l, tgt_l)[0, t, :].to(torch.float64).cpu().numpy()
            z_fd = (yp - ym) / (2.0 * eps_s)
            ae = float(np.linalg.norm(Z24_b8[:, 0] - z_fd))
            dn = float(np.linalg.norm(z_fd))
            jvp_fd_by_step[str(step)].append({"abs_err": ae, "rel_err": ae / max(dn, abs_floor), "denom": dn})

        # Adjoint consistency across F0 + 3 option contrasts
        a0_vjp = sess_p0.vjp_eval_reference(ro["F0"]["g"], "F0")
        lhs = float(np.dot(Z24_b8[:, 0], ro["F0"]["g"]))
        rhs = float(np.dot(X8[:, 0], a0_vjp))
        adjoint_errs.append(abs(lhs - rhs) / max(abs(lhs), abs(rhs), abs_floor))

        # Archived conditioning exactness check (<= 1e-6!)
        phi_c = extract_context_features(h_s[0, t, :].detach().to(torch.float64).cpu().numpy(), V_dark_np, U_leg12_np, int(h_s.shape[1]))
        M0 = predict_M0(prior, phi_c)
        Y8 = Z24_b8 - V_dark_np @ X8
        st8 = condition_exact_batch(prior, M0, X8, Y8, rec["context_id"], [f"p{j}" for j in range(8)])
        archived_cond_errs.append(st8["support_residual_rel"])

    # ProbeSession isolation & budget rejection check + Truth-Replacement Sentinel!
    rec0 = pilot_4[0]
    tok0 = tokenize_chat_prompt(tokenizer, rec0["user_prompt"], device)
    st0, lg0 = capture_residual_states(model, tok0["input_ids"], tok0["attention_mask"])
    t0 = tok0["t_last"]
    B0 = generate_fixed_38x38_bank(rec0["context_id"], bank_seed, r=38)
    sess_iso = ProbeSessionV04("p0_iso", "learning_support", rec0["context_id"], model, st0[src_l], V_dark, src_l, mid_l, tgt_l, t0, max_jvp_dirs=4, allow_vjp=False, jvp_batch_size=4)
    Z4_iso, _ = sess_iso.jvp_columns(B0[:, :4], "iso_k4")
    budget_rejected, vjp_rejected = False, False
    try:
        sess_iso.jvp_columns(B0[:, 4:5], "iso_over")
    except BudgetExceededError:
        budget_rejected = True
    try:
        sess_iso.vjp_eval_reference(np.ones(V_dark_np.shape[0]), "F0")
    except UnauthorizedQueryError:
        vjp_rejected = True

    # Truth-replacement sentinel: compute Stage A M_4 & action under real vs NaN/scrambled evaluator truth dict!
    phi0 = extract_context_features(st0[src_l][0, t0, :].detach().to(torch.float64).cpu().numpy(), V_dark_np, U_leg12_np, int(st0[src_l].shape[1]))
    ro0 = extract_v04_readouts(lg0, st0[tgt_l][0, t0, :], W_U, gamma32, basis.rms_eps, option_token_ids, rec0["target_option"])
    def _stage_a_pure(eval_truth_unused: Any) -> np.ndarray:
        M0_l = predict_M0(prior, phi0)
        Y4_l = Z4_iso - V_dark_np @ B0[:, :4]
        st_l = condition_exact_batch(prior, M0_l, B0[:, :4], Y4_l, "SENT", ["0", "1", "2", "3"])
        pr_l = project_function_from_posterior(prior, V_dark_np, st_l["M_k"], st_l["C_k"], ro0["F0"]["g"])
        x_l, _ = solve_min_norm_ball_action(pr_l["a_hat"], 0.05, 0.01 * float(torch.linalg.vector_norm(st0[src_l][0, t0, :]).item()))
        return x_l
    x_s1 = _stage_a_pure({"Y_perp": np.zeros((896, 8))})
    x_s2 = _stage_a_pure({"Y_perp": np.full((896, 8), np.nan)})
    sentinel_diff = float(np.max(np.abs(x_s1 - x_s2)))

    fd_sum = {
        s: {
            "median_rel_err": float(np.median([x["rel_err"] for x in v])),
            "max_rel_err": float(np.max([x["rel_err"] for x in v])),
            "median_abs_err": float(np.median([x["abs_err"] for x in v])),
        } for s, v in jvp_fd_by_step.items()
    }
    best_fd_step = min(fd_sum.keys(), key=lambda s: fd_sum[s]["median_rel_err"])
    best_fd_med = fd_sum[best_fd_step]["median_rel_err"]

    passed = bool(
        synth_fp64["passed_1e9"]
        and max(zero_D) <= float(th["zero_pert_D_max"])
        and max(zero_dF) <= float(th["zero_pert_abs_dF_max"])
        and max(zero_hook_diffs) <= 1e-4
        and best_fd_med <= float(th["jvp_fd_median_max"])
        and max(adjoint_errs) <= float(th["adjoint_rel_max"])
        and max(lin_comb_errs) <= float(th["linearity_rel_max"])
        and max(jvp_b8_vs_b1) <= 1e-4
        and max(mid_z20_errs) <= 1e-4
        and max(archived_cond_errs) <= float(th["archived_cond_max"])
        and budget_rejected and vjp_rejected and sentinel_diff == 0.0
    )

    return {
        "all_p0_passed": passed,
        "fp64_synthetic_conditioning": synth_fp64,
        "zero_hook_max_logit_diff": float(max(zero_hook_diffs)),
        "zero_pert_matched_batch1": {"max_D_geom": float(max(zero_D)), "max_abs_dF": float(max(zero_dF))},
        "jvp_batched8_vs_batch1_max_rel_err": float(max(jvp_b8_vs_b1)),
        "jvp_repeatability_max_rel_err": float(max(repeat_errs)),
        "jvp_linear_comb_max_rel_err": float(max(lin_comb_errs)),
        "mid_layer_z20_chain_max_rel_err": float(max(mid_z20_errs)),
        "jvp_fd_curve": {"by_step": fd_sum, "best_step": best_fd_step, "best_median_rel_err": best_fd_med},
        "adjoint_consistency_max_rel_err": float(max(adjoint_errs)),
        "archived_conditioning_support_max_rel_err": float(max(archived_cond_errs)),
        "isolation_checks": {"budget_5th_rejected": budget_rejected, "learning_vjp_rejected": vjp_rejected, "truth_replacement_sentinel_diff": sentinel_diff},
    }


def main() -> None:
    t_start = time.time()
    utc_start = utc_now()
    cfg_dir = ROOT / "configs"
    res_dir = ROOT / "results"
    hb = HeartbeatTracker(ROOT / "heartbeat.json", t_start)
    hb.beat("BOOT", 0, 1, note="Starting BAYES-DYN-04", force=True)

    # Clean append-mode JSONL files
    for d_sub in ("observations", "posteriors", "interventions", "calibration", "trajectory_panel", "figures"):
        (res_dir / d_sub).mkdir(parents=True, exist_ok=True)

    for p in (
        res_dir / "observations" / "query_receipts.jsonl",
        res_dir / "predictions.jsonl",
        res_dir / "interventions" / "dev_pilot_interventions.jsonl",
        res_dir / "interventions" / "confirm_id_interventions.jsonl",
        res_dir / "interventions" / "confirm_ood_interventions.jsonl",
        ROOT / "atlas_index.jsonl",
    ):
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            p.unlink()

    cfg = json.loads((cfg_dir / "experiment_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((cfg_dir / "data_manifest.json").read_text(encoding="utf-8"))
    dev_sel = json.loads((cfg_dir / "dev_selection_config.json").read_text(encoding="utf-8"))
    prior, V_dark_prior_np, U_old_64 = load_full_operator_prior_npz(res_dir / "prior_model.npz")
    S_mispair = np.array(manifest["S_mispair_8x8"], dtype=np.float64)
    traj_16_set = set(manifest["pre_registered_trajectory_panel_16"])
    traj_16_order = list(manifest["pre_registered_trajectory_panel_16"])

    pip_out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    (res_dir / "pip_freeze.txt").write_text(pip_out.stdout, encoding="utf-8")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(int(cfg["master_seed"]))
    np.random.seed(int(cfg["master_seed"]))
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    model_id = cfg["model_id"]
    rev = cfg["model_revision"]
    src_l, mid_l, tgt_l = int(cfg["layers"]["source_layer"]), int(cfg["layers"]["mid_layer"]), int(cfg["layers"]["target_layer"])
    bank_seed = int(cfg["derived_seeds"]["probe_bank_38x38"])
    qgen_seed = int(cfg["derived_seeds"]["q_general_4"])
    rho_alpha = float(cfg["control"]["radius_alpha"])
    zero_tol = float(cfg["control"]["zero_a_rel_tol"])
    n_boot = int(cfg["stats"]["bootstrap_resamples"])
    b_seed = int(cfg["derived_seeds"]["bootstrap_stats"])

    hb.beat("LOAD_MODEL", 0, 1, note=f"Loading {model_id}@{rev[:12]}", force=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=rev, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=rev, dtype=torch.float32, attn_implementation="eager", trust_remote_code=False
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    full_sha = compute_full_weight_sha256(model)
    option_token_ids = resolve_option_token_ids(tokenizer, cfg["data"]["options"])
    basis = compute_spectral_basis(model, cfg, assert_expected_05b=True)
    V_dark = basis.V_dark.to(device=device, dtype=torch.float32)
    V_dark_np = V_dark.detach().to(torch.float64).cpu().numpy()
    V_full_np = basis.V_full.detach().to(torch.float64).cpu().numpy()
    U_leg12_np = basis.V_legacy12.detach().to(torch.float64).cpu().numpy()
    W_U = model.lm_head.weight.detach()
    gamma32 = basis.rms_gamma.to(device=device, dtype=torch.float32)
    r, d = basis.n_dark, basis.V_full.shape[0]
    # Verify exact match between live checkpoint V_dark and CPU-frozen prior V_dark!
    v_match_err = float(np.max(np.abs(V_dark_np - V_dark_prior_np)))
    assert v_match_err < 1e-5, f"Live V_dark differs from CPU prior V_dark: {v_match_err}"

    model_env = {
        "experiment_id": "BAYES-DYN-04",
        "timestamp_utc": utc_now(),
        "model_id": model_id, "model_revision": rev,
        "full_all_params_weight_sha256": full_sha,
        "dtype": "torch.float32", "attn_implementation": "eager", "allow_tf32": False,
        "hidden_size": d, "n_dark": r, "source_layer": src_l, "mid_layer": mid_l, "target_layer": tgt_l,
        "v_dark_prior_vs_live_max_diff": v_match_err,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "option_token_ids": option_token_ids,
    }
    save_json(cfg_dir / "model_environment.json", model_env)
    print(f"[04] Loaded {model_id}@{rev[:12]} on {model_env['gpu_name']} (V_dark match err={v_match_err:.2e})", flush=True)

    split_access_utc: dict[str, str] = {}
    pilot_recs = load_jsonl(ROOT / "data" / "dev_pilot.jsonl")
    split_access_utc["dev_pilot"] = utc_now()

    # =========================================================================
    # STEP 1: P0 VERIFICATION
    # =========================================================================
    hb.beat("P0_CHECKS", 0, 4, note="Running P0 numerical & isolation suite", force=True)
    p0_4 = [next(x for x in pilot_recs if x["family"] == fam) for fam in cfg["data"]["id_families"]]
    p0_rep = run_p0_checks_v04(model, tokenizer, basis, prior, V_dark_np, U_leg12_np, p0_4, option_token_ids, cfg)
    save_json(res_dir / "p0_checks.json", p0_rep)
    print(f"[04 P0] passed={p0_rep['all_p0_passed']} | FD={p0_rep['jvp_fd_curve']['best_median_rel_err']:.2e} | "
          f"adj={p0_rep['adjoint_consistency_max_rel_err']:.2e} | lin={p0_rep['jvp_linear_comb_max_rel_err']:.2e} | "
          f"z20_chain={p0_rep['mid_layer_z20_chain_max_rel_err']:.2e}", flush=True)
    assert p0_rep["all_p0_passed"], f"P0 FAILED: {p0_rep}"

    # =========================================================================
    # STEP 2: DEV-PILOT 16 -> FREEZE WORKING TARGET TAU & MEASURE THROUGHPUT
    # =========================================================================
    hb.beat("DEV_PILOT", 0, len(pilot_recs), note="Freezing tau on DEV-PILOT 16", force=True)
    t_pilot0 = time.time()
    pilot_cache = []
    for i, rec in enumerate(pilot_recs):
        tok = tokenize_chat_prompt(tokenizer, rec["user_prompt"], device)
        states, logits = capture_residual_states(model, tok["input_ids"], tok["attention_mask"])
        t = tok["t_last"]
        h_s, h_L = states[src_l], states[tgt_l][0, t, :]
        h_norm = float(torch.linalg.vector_norm(h_s[0, t, :]).item())
        ro = extract_v04_readouts(logits, h_L, W_U, gamma32, basis.rms_eps, option_token_ids, rec["target_option"])
        sess_ev = ProbeSessionV04(f"pilot_ev_{rec['context_id']}", "evaluator_truth", rec["context_id"], model, h_s, V_dark, src_l, mid_l, tgt_l, t, max_jvp_dirs=0, allow_vjp=True)
        a0_ex = sess_ev.vjp_eval_reference(ro["F0"]["g"], "F0")
        append_jsonl(res_dir / "observations" / "query_receipts.jsonl", [sess_ev.receipt()])
        pilot_cache.append((rec, tok, h_s, h_L, logits, h_norm, ro, a0_ex))

    tau_frozen = 0.01
    working_regime_established = False
    pilot_tau_diag: dict[str, Any] = {}
    for tau_c in [float(x) for x in cfg["control"]["tau_candidates_ordered"]]:
        hits = 0
        lin_rels = []
        rows_tau = []
        for (rec, tok, h_s, h_L, logits, h_norm, ro, a0_ex) in pilot_cache:
            rho_c = rho_alpha * h_norm
            x_ex, m_ex = solve_min_norm_ball_action(a0_ex, tau_c, rho_c, zero_tol)
            ev = evaluate_v04_intervention_batch1(
                model, h_s, h_L, logits, src_l, tgt_l, tok["t_last"], V_dark, x_ex, ro, option_token_ids, rec["target_option"], tau_c,
                target_tol=float(cfg["control"]["pilot_target_tol"]),
            )
            pred_lin = float(np.dot(a0_ex, x_ex))
            lin_rel = abs(ev["actual_delta_F0"] - pred_lin) / max(abs(pred_lin), 1e-6)
            hits += int(ev["within_10pct"])
            lin_rels.append(lin_rel)
            rows_tau.append({
                "context_id": rec["context_id"], "tau_cand": tau_c, "within_10pct": ev["within_10pct"],
                "actual_delta_F0": ev["actual_delta_F0"], "pred_lin_F0": pred_lin, "lin_rel_err": lin_rel,
                "target_tracking_err": ev["target_tracking_err"], "hit_boundary": m_ex["hit_boundary_unreachable"],
            })
        append_jsonl(res_dir / "interventions" / "dev_pilot_interventions.jsonl", rows_tau)
        med_lin = float(np.median(lin_rels))
        ok_tau = bool(hits >= int(cfg["control"]["pilot_hit_min_count"]) and med_lin <= float(cfg["control"]["pilot_lin_rel_err_max"]))
        pilot_tau_diag[str(tau_c)] = {"tau": tau_c, "hits_10pct": hits, "n": len(pilot_cache), "median_lin_rel_err": med_lin, "passed": ok_tau}
        print(f"[04 DEV-PILOT] tau={tau_c}: hits={hits}/16 (>=14), med_lin_rel_err={med_lin*100:.2f}% (<=5%), PASS={ok_tau}", flush=True)
        if ok_tau and not working_regime_established:
            tau_frozen = tau_c
            working_regime_established = True
            break

    # Quick throughput benchmark on 1 pilot context (28 JVP dirs + 1 VJP + 7 forwards)
    _, tok0, hs0, hL0, lg0, _, ro0, _ = pilot_cache[0]
    B_th = generate_fixed_38x38_bank("THRU", bank_seed, r=38)
    torch.cuda.synchronize(); t_th0 = time.perf_counter()
    s_th = ProbeSessionV04("thru", "evaluator_truth", "THRU", model, hs0, V_dark, src_l, mid_l, tgt_l, tok0["t_last"], max_jvp_dirs=28, allow_vjp=True, jvp_batch_size=8)
    _ = s_th.jvp_columns(B_th[:, :16], "sup16")
    _ = s_th.jvp_columns(B_th[:, 16:28], "eval12")
    _ = s_th.vjp_eval_reference(ro0["F0"]["g"], "F0")
    torch.cuda.synchronize()
    sec_per_ctx = time.perf_counter() - t_th0
    est_total_min = (sec_per_ctx * 160.0 + 45.0) / 60.0
    print(f"[04 THROUGHPUT] ~{sec_per_ctx:.2f}s/ctx -> estimated full panel (~160 ctx) = {est_total_min:.1f} min (well within 60 min / 2 CU!)", flush=True)
    del pilot_cache
    torch.cuda.empty_cache()

    # =========================================================================
    # STEP 3: CAL 32 -> FIT GLOBAL TEMPERATURE t ON k=8 Q_perp FULL 896-D ERRORS -> FREEZE CONFIG
    # =========================================================================
    cal_recs = load_jsonl(ROOT / "data" / "cal.jsonl")
    split_access_utc["cal"] = utc_now()
    hb.beat("CAL_32", 0, len(cal_recs), note="Running CAL 32 for global temperature t", force=True)

    cal_E_k8 = np.zeros((len(cal_recs), 8, d), dtype=np.float64)
    cal_den_k8 = np.zeros((len(cal_recs), 8), dtype=np.float64)
    cal_cids, cal_X16, cal_Y16, cal_Qp, cal_Yp, cal_M0, cal_M8, cal_C8, cal_phis = [], [], [], [], [], [], [], [], []

    for i, rec in enumerate(cal_recs):
        cid = rec["context_id"]
        tok = tokenize_chat_prompt(tokenizer, rec["user_prompt"], device)
        states, logits = capture_residual_states(model, tok["input_ids"], tok["attention_mask"])
        t = tok["t_last"]
        h_s = states[src_l]
        phi_c = extract_context_features(h_s[0, t, :].detach().to(torch.float64).cpu().numpy(), V_dark_np, U_leg12_np, int(h_s.shape[1]))
        M0 = predict_M0(prior, phi_c)
        B = generate_fixed_38x38_bank(cid, bank_seed, r=38)
        X8 = B[:, :8]
        Q_p = B[:, 16:24]

        # Stage A: 8 support JVPs
        sess_a = ProbeSessionV04(f"cal_a_{cid}", "learning_support", cid, model, h_s, V_dark, src_l, mid_l, tgt_l, t, max_jvp_dirs=8, allow_vjp=False, jvp_batch_size=8)
        Z8, _ = sess_a.jvp_columns(X8, "cal_sup8")
        Y8 = Z8 - V_dark_np @ X8
        st8 = condition_exact_batch(prior, M0, X8, Y8, cid, [f"{cid}:s{j}" for j in range(8)])
        append_jsonl(res_dir / "observations" / "query_receipts.jsonl", [sess_a.receipt()])

        # Stage B: 8 Q_perp evaluation JVPs
        sess_b = ProbeSessionV04(f"cal_b_{cid}", "evaluator_truth", cid, model, h_s, V_dark, src_l, mid_l, tgt_l, t, max_jvp_dirs=8, allow_vjp=False, jvp_batch_size=8)
        Zp, _ = sess_b.jvp_columns(Q_p, "cal_qperp8")
        Yp = Zp - V_dark_np @ Q_p
        append_jsonl(res_dir / "observations" / "query_receipts.jsonl", [sess_b.receipt()])

        for j in range(8):
            cal_E_k8[i, j] = st8["M_k"] @ Q_p[:, j] - Yp[:, j]
            cal_den_k8[i, j] = float(Q_p[:, j] @ st8["C_k"] @ Q_p[:, j])

        cal_cids.append(cid); cal_X16.append(X8); cal_Y16.append(Y8.astype(np.float32))
        cal_Qp.append(Q_p); cal_Yp.append(Yp.astype(np.float32))
        cal_M0.append(M0.astype(np.float32)); cal_M8.append(st8["M_k"].astype(np.float32))
        cal_C8.append(st8["C_k"]); cal_phis.append(phi_c)
        if (i + 1) % 8 == 0:
            hb.beat("CAL_32", i + 1, len(cal_recs), note=f"CAL {i+1}/{len(cal_recs)}")

    cal_fit_rep = fit_cal_temperature(prior, cal_E_k8, cal_den_k8)
    temp_t = float(cal_fit_rep["temperature_t"])
    save_json(res_dir / "calibration" / "cal32_temperature_fit.json", cal_fit_rep)
    np.savez_compressed(
        res_dir / "calibration" / "cal32_raw_errors.npz",
        context_ids=np.array(cal_cids), E_k8_qperp=cal_E_k8.astype(np.float32), qCq_k8_qperp=cal_den_k8.astype(np.float64),
    )
    np.savez_compressed(
        res_dir / "observations" / "cal_obs.npz",
        context_ids=np.array(cal_cids), X_sup_8=np.stack(cal_X16), Y_sup_8=np.stack(cal_Y16),
        Q_perp=np.stack(cal_Qp), Y_perp_true=np.stack(cal_Yp), phi=np.stack(cal_phis),
    )
    np.savez_compressed(
        res_dir / "posteriors" / "cal_posteriors.npz",
        context_ids=np.array(cal_cids), M_0=np.stack(cal_M0), M_8=np.stack(cal_M8), C_8=np.stack(cal_C8),
    )

    # WRITE IMMUTABLE frozen_config.json BEFORE ACCESSING CONFIRM!
    code_shas = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted((ROOT / "src").glob("*.py"))}
    code_shas["run_bayes_dyn_04.py"] = sha256_file(ROOT / "run_bayes_dyn_04.py")
    frozen_cfg = {
        "experiment_id": "BAYES-DYN-04",
        "frozen_utc": utc_now(),
        "model_id": model_id, "model_revision": rev, "full_all_params_weight_sha256": full_sha,
        "prior_alpha": prior.alpha, "prior_eta": prior.eta, "s_A_sq": prior.s_A_sq,
        "near_zero_norm_thresh": prior.near_zero_norm_thresh,
        "psd_floor_C": prior.psd_floor_C, "psd_floor_R": prior.psd_floor_R, "solver_jitter": prior.solver_jitter,
        "tau_frozen": tau_frozen, "working_regime_established": working_regime_established,
        "radius_alpha": rho_alpha, "pilot_tau_diagnostics": pilot_tau_diag,
        "cal32_temperature_t": temp_t, "cal32_fit_summary": cal_fit_rep,
        "throughput_sec_per_ctx": round(sec_per_ctx, 3),
        "data_file_sha256": manifest["file_sha256"], "code_file_sha256": code_shas,
    }
    save_json(cfg_dir / "frozen_config.json", frozen_cfg)
    print(f"[04 FROZEN] tau={tau_frozen} (working_regime={working_regime_established}), CAL32 temp t={temp_t:.4f} -> frozen_config.json saved!", flush=True)

    # =========================================================================
    # STEP 4: CONFIRM-ID 96 & CONFIRM-OOD 32 (Strict Stage A -> Commit -> Stage B!)
    # =========================================================================
    interv_methods = ["ZERO", "PRIOR", "UPDATE_4", "UPDATE_8", "UPDATE_16", "PATCH_8", "EXACT"]
    traj_store: dict[str, dict[str, Any]] = {}
    all_int_vecs: list[np.ndarray] = []
    all_int_cids: list[str] = []
    all_int_splits: list[str] = []

    def _run_confirm_split(split_name: str, recs: list[dict[str, Any]]) -> dict[str, Any]:
        split_access_utc[split_name] = utc_now()
        n_s = len(recs)
        hb.beat(f"CONFIRM_{split_name.upper()}", 0, n_s, note=f"Starting {split_name} (n={n_s})", force=True)
        print(f"[04 {split_name.upper()}] Running Stage A -> commit -> Stage B on n={n_s} contexts...", flush=True)

        # Observation & posterior archive buffers
        obs_cids, obs_phis, obs_B, obs_Qg, obs_Ysup16, obs_Yp, obs_Yg, obs_g0, obs_g3, obs_a0ex = [], [], [], [], [], [], [], [], [], []
        post_M = np.zeros((n_s, 4, d, r), dtype=np.float32)    # budgets [0, 4, 8, 16]
        post_C = np.zeros((n_s, 4, r, r), dtype=np.float64)
        post_ids_arr: list[list[str]] = []

        # Per-context summary tables for clean aggregation
        ctx_eval_records: list[dict[str, Any]] = []
        int_file = res_dir / "interventions" / f"{split_name}_interventions.jsonl"

        for i, rec in enumerate(recs):
            cid = rec["context_id"]
            fam = rec["family"]
            in_traj = (cid in traj_16_set)

            tok = tokenize_chat_prompt(tokenizer, rec["user_prompt"], device)
            states, logits = capture_residual_states(model, tok["input_ids"], tok["attention_mask"])
            t = tok["t_last"]
            h_s, h_m, h_L = states[src_l], states[mid_l], states[tgt_l][0, t, :]
            h_norm = float(torch.linalg.vector_norm(h_s[0, t, :]).item())
            rho_c = rho_alpha * h_norm
            phi_c = extract_context_features(h_s[0, t, :].detach().to(torch.float64).cpu().numpy(), V_dark_np, U_leg12_np, int(h_s.shape[1]))

            # Fixed 38x38 bank & Q_general
            B = generate_fixed_38x38_bank(cid, bank_seed, r=38)
            X_sup_16 = B[:, 0:16]
            Q_perp = B[:, 16:24]
            Q_gen = generate_q_general(cid, qgen_seed, r=38, n_dirs=4)

            # -------- STAGE A: TARGET-AGNOSTIC MAP UPDATES (ZERO truth, ZERO target info!) --------
            M0 = predict_M0(prior, phi_c)
            st0 = condition_exact_batch(prior, M0, np.zeros((r, 0)), np.zeros((d, 0)), cid, [])
            sess_a = ProbeSessionV04(f"{split_name}_a_{cid}", "learning_support", cid, model, h_s, V_dark, src_l, mid_l, tgt_l, t, max_jvp_dirs=16, allow_vjp=False, jvp_batch_size=8)

            Z4, Z20_4 = sess_a.jvp_columns(X_sup_16[:, 0:4], "sup_0_4", record_mid_z20=in_traj)
            Y4 = Z4 - V_dark_np @ X_sup_16[:, 0:4]
            st4 = condition_exact_batch(prior, M0, X_sup_16[:, 0:4], Y4, cid, [f"{cid}:s{j}" for j in range(4)])

            Z4_8, Z20_4_8 = sess_a.jvp_columns(X_sup_16[:, 4:8], "sup_4_8", record_mid_z20=in_traj)
            Z8 = np.concatenate([Z4, Z4_8], axis=1)
            Y8 = Z8 - V_dark_np @ X_sup_16[:, 0:8]
            st8 = condition_exact_batch(prior, M0, X_sup_16[:, 0:8], Y8, cid, [f"{cid}:s{j}" for j in range(8)])
            M_patch8 = condition_patch_only(M0, X_sup_16[:, 0:8], Y8)
            Y8_mis = make_mispaired_Y(M0, X_sup_16[:, 0:8], Y8, S_mispair)
            st8_mis = condition_exact_batch(prior, M0, X_sup_16[:, 0:8], Y8_mis, cid, [f"{cid}:mis{j}" for j in range(8)])

            Z8_16, Z20_8_16 = sess_a.jvp_columns(X_sup_16[:, 8:16], "sup_8_16", record_mid_z20=in_traj)
            Z16 = np.concatenate([Z8, Z8_16], axis=1)
            Y16 = Z16 - V_dark_np @ X_sup_16
            st16 = condition_exact_batch(prior, M0, X_sup_16, Y16, cid, [f"{cid}:s{j}" for j in range(16)])
            append_jsonl(res_dir / "observations" / "query_receipts.jsonl", [sess_a.receipt()])

            # Store frozen posteriors (exact archived float32 M_k hashed accurately!)
            sts = [st0, st4, st8, st16]
            for bi, stb in enumerate(sts):
                post_M[i, bi] = stb["M_k"].astype(np.float32)
                post_C[i, bi] = stb["C_k"]
            post_ids_arr.append([s["posterior_id"] for s in sts])

            # -------- STAGE A STEP 4: FREEZE MAPS -> CONSTRUCT READOUTS -> PROJECT & COMMIT ACTIONS! --------
            ro = extract_v04_readouts(logits, h_L, W_U, gamma32, basis.rms_eps, option_token_ids, rec["target_option"])
            g0 = ro["F0"]["g"]
            G_opt3 = np.stack([x["g"] for x in ro["opt3"]], axis=0)  # [3, d]

            # Project F0 & opt3 from EACH frozen posterior (posterior_id MUST remain identical!)
            proj_f0 = {}
            proj_opt3 = {}
            for lbl, stb in (("PRIOR", st0), ("UPDATE_4", st4), ("UPDATE_8", st8), ("UPDATE_16", st16)):
                pid_before = stb["posterior_id"]
                proj_f0[lbl] = project_function_from_posterior(prior, V_dark_np, stb["M_k"], stb["C_k"], g0, temp_t=temp_t)
                proj_opt3[lbl] = [project_function_from_posterior(prior, V_dark_np, stb["M_k"], stb["C_k"], G_opt3[oj], temp_t=temp_t) for oj in range(3)]
                assert stb["posterior_id"] == pid_before
            proj_f0["PATCH_8"] = {"a_hat": (V_dark_np + M_patch8).T @ g0, "Cov_a": proj_f0["UPDATE_8"]["Cov_a"]}
            proj_f0["MISPAIRED_8"] = project_function_from_posterior(prior, V_dark_np, st8_mis["M_k"], st8_mis["C_k"], g0, temp_t=temp_t)

            # Solve learning control actions BEFORE evaluator truth reveal!
            actions: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
            actions["ZERO"] = (np.zeros(r, dtype=np.float64), {"is_zero_a": False, "hit_boundary_unreachable": False, "a_hat_norm": 0.0, "step_norm": 0.0, "pred_delta_F0": 0.0, "tau": tau_frozen, "rho_c": rho_c})
            for lbl in ("PRIOR", "UPDATE_4", "UPDATE_8", "UPDATE_16", "PATCH_8"):
                xk, mk = solve_min_norm_ball_action(proj_f0[lbl]["a_hat"], tau_frozen, rho_c, zero_tol)
                # Ahead-of-time scalar std for chosen action xk
                _, sd_act = predict_scalar_direction_dist(proj_f0[lbl]["a_hat"], proj_f0[lbl]["Cov_a"], xk)
                mk["pred_delta_F0_std"] = sd_act
                actions[lbl] = (xk, mk)

            # Common-action diagnostic (§7.2): BOTH PRIOR and UPDATE_8 predict the SAME PRIOR action x_0!
            x_prior = actions["PRIOR"][0]
            ca_prior_mean, ca_prior_std = predict_scalar_direction_dist(proj_f0["PRIOR"]["a_hat"], proj_f0["PRIOR"]["Cov_a"], x_prior)
            ca_upd8_mean, ca_upd8_std = predict_scalar_direction_dist(proj_f0["UPDATE_8"]["a_hat"], proj_f0["UPDATE_8"]["Cov_a"], x_prior)

            # Commit Stage A predictions BEFORE revealing Stage B truth!
            stage_a_commit_utc = utc_now()
            pred_commit_row = {
                "split": split_name, "context_id": cid, "family": fam, "commit_utc": stage_a_commit_utc,
                "posterior_ids": {f"k{kb}": sts[bi]["posterior_id"] for bi, kb in enumerate((0, 4, 8, 16))},
                "M_k_f32_sha256": {f"k{kb}": bytes_sha256_short(post_M[i, bi]) for bi, kb in enumerate((0, 4, 8, 16))},
                "action_x_f64_sha256": {m: bytes_sha256_short(actions[m][0]) for m in ("ZERO", "PRIOR", "UPDATE_4", "UPDATE_8", "UPDATE_16", "PATCH_8")},
                "common_action_x0_pred": {
                    "PRIOR_mean": ca_prior_mean, "PRIOR_std": ca_prior_std,
                    "UPDATE_8_mean": ca_upd8_mean, "UPDATE_8_std": ca_upd8_std,
                },
                "pred_qperp_f0_k0": [float(x) for x in (proj_f0["PRIOR"]["a_hat"] @ Q_perp)],
                "pred_qperp_f0_k8": [float(x) for x in (proj_f0["UPDATE_8"]["a_hat"] @ Q_perp)],
            }
            append_jsonl(res_dir / "predictions.jsonl", [pred_commit_row])

            # -------- STAGE B: REVEAL Q_perp (8), Q_general (4), EXACT F0 VJP & RUN REAL FORWARDS --------
            sess_b = ProbeSessionV04(f"{split_name}_b_{cid}", "evaluator_truth", cid, model, h_s, V_dark, src_l, mid_l, tgt_l, t, max_jvp_dirs=12, allow_vjp=True, jvp_batch_size=8)
            Zp_true, Z20_p = sess_b.jvp_columns(Q_perp, "eval_qperp8", record_mid_z20=in_traj)
            Zg_true, Z20_g = sess_b.jvp_columns(Q_gen, "eval_qgen4", record_mid_z20=in_traj)
            Yp_true = Zp_true - V_dark_np @ Q_perp
            Yg_true = Zg_true - V_dark_np @ Q_gen
            a0_exact = sess_b.vjp_eval_reference(g0, "F0")
            append_jsonl(res_dir / "observations" / "query_receipts.jsonl", [sess_b.receipt()])

            # EXACT action using identical min-norm ball rule
            x_ex, m_ex = solve_min_norm_ball_action(a0_exact, tau_frozen, rho_c, zero_tol)
            m_ex["pred_delta_F0_std"] = 0.0
            actions["EXACT"] = (x_ex, m_ex)

            # Evaluate full-output 896-D errors on Q_perp & Q_general + in/out U_hybrid_64 + F0 & opt3 errors + scalar calibration!
            P_U_old = U_old_64
            eval_map_dict: dict[str, Any] = {}
            map_candidates = [
                ("PRIOR", st0["M_k"], st0["C_k"], st0["posterior_id"], 0),
                ("UPDATE_4", st4["M_k"], st4["C_k"], st4["posterior_id"], 4),
                ("UPDATE_8", st8["M_k"], st8["C_k"], st8["posterior_id"], 8),
                ("UPDATE_16", st16["M_k"], st16["C_k"], st16["posterior_id"], 16),
                ("PATCH_8", M_patch8, st8["C_k"], f"{cid}:patch8", 8),
                ("MISPAIRED_8", st8_mis["M_k"], st8_mis["C_k"], st8_mis["posterior_id"], 8),
            ]
            f0_true_qp = Zp_true.T @ g0      # [8]
            f0_true_qg = Zg_true.T @ g0      # [4]
            opt3_true_qp = G_opt3 @ Zp_true  # [3, 8]

            for lbl, Mk, Ck, pid, kb in map_candidates:
                # Q_perp 896-D
                diff_p = Mk @ Q_perp - Yp_true  # [896, 8]
                sq_p_each = np.sum(diff_p ** 2, axis=0)
                L_perp_norm = float(np.mean(sq_p_each) / prior.s_A_sq)
                # In / Out old U_hybrid_64 output breakdown (common full output denominator!)
                proj_u = P_U_old.T @ diff_p
                in_u_sq = float(np.mean(np.sum(proj_u ** 2, axis=0)))
                out_u_sq = float(np.mean(sq_p_each)) - in_u_sq
                # Near-zero stratified & direction/gain metrics on Q_perp
                dir_ms = [vector_dir_metrics(Mk @ Q_perp[:, j], Yp_true[:, j], prior.near_zero_norm_thresh) for j in range(8)]
                nz_errs = [dm["sq_l2"] for dm in dir_ms if dm["is_near_zero"]]
                non_nz_cos = [dm["cosine"] for dm in dir_ms if (not dm["is_near_zero"]) and dm["cosine"] is not None]
                non_nz_gain = [dm["gain_ratio"] for dm in dir_ms if (not dm["is_near_zero"]) and dm["gain_ratio"] is not None]

                # Q_general 896-D
                diff_g = Mk @ Q_gen - Yg_true
                L_gen_norm = float(np.mean(np.sum(diff_g ** 2, axis=0)) / prior.s_A_sq)

                # Functional F0 MSE on Q_perp & Q_general (physical logit^2 units, §6.3!)
                a_f0_hat = proj_f0[lbl]["a_hat"]
                f0_pred_qp = a_f0_hat @ Q_perp
                f0_pred_qg = a_f0_hat @ Q_gen
                f0_mse_qp = float(np.mean((f0_pred_qp - f0_true_qp) ** 2))
                f0_mse_qg = float(np.mean((f0_pred_qg - f0_true_qg) ** 2))
                f0_sign_acc_qp = float(np.mean(np.sign(f0_pred_qp) == np.sign(f0_true_qp)))

                # Opt3 MSE on Q_perp (same frozen posterior!)
                opt3_mse_qp = None
                if lbl in proj_opt3:
                    o3_preds = np.stack([proj_opt3[lbl][oj]["a_hat"] @ Q_perp for oj in range(3)], axis=0)
                    opt3_mse_qp = float(np.mean((o3_preds - opt3_true_qp) ** 2))

                # Scalar F0 uncertainty coverage (50/80/90/95%), width & NLL on Q_perp
                cov_hits = {lv: [] for lv in Z_SCORES}
                widths_90, nlls = [], []
                Cov_f0 = proj_f0[lbl]["Cov_a"]
                for j in range(8):
                    mu_j, sd_j = predict_scalar_direction_dist(a_f0_hat, Cov_f0, Q_perp[:, j])
                    err_j = abs(f0_true_qp[j] - mu_j)
                    for lv, zval in Z_SCORES.items():
                        cov_hits[lv].append(int(err_j <= zval * sd_j))
                    widths_90.append(2.0 * Z_SCORES[0.90] * sd_j)
                    nlls.append(scalar_gaussian_nll(f0_true_qp[j], mu_j, sd_j))

                eval_map_dict[lbl] = {
                    "posterior_id": pid, "k": kb,
                    "L_perp_norm": L_perp_norm, "L_perp_raw_l2_sq": float(np.mean(sq_p_each)),
                    "L_perp_in_Uold64_sq": in_u_sq, "L_perp_out_Uold64_sq": out_u_sq,
                    "L_gen_norm": L_gen_norm, "L_gen_raw_l2_sq": float(np.mean(np.sum(diff_g ** 2, axis=0))),
                    "mean_cosine_non_nz": float(np.mean(non_nz_cos)) if non_nz_cos else None,
                    "mean_gain_ratio_non_nz": float(np.mean(non_nz_gain)) if non_nz_gain else None,
                    "near_zero_count_in_8": len(nz_errs),
                    "near_zero_mean_sq_l2": float(np.mean(nz_errs)) if nz_errs else None,
                    "f0_mse_qperp": f0_mse_qp, "f0_mse_qgen": f0_mse_qg, "f0_sign_acc_qperp": f0_sign_acc_qp,
                    "opt3_mse_qperp": opt3_mse_qp,
                    "f0_qperp_cov": {str(int(lv * 100)): float(np.mean(v)) for lv, v in cov_hits.items()},
                    "f0_qperp_mean_width_90": float(np.mean(widths_90)),
                    "f0_qperp_mean_nll": float(np.mean(nlls)),
                }

            # Real matched batch=1 nonlinear interventions for ALL 7 arms!
            int_rows_ctx: list[dict[str, Any]] = []
            int_by_m: dict[str, dict[str, Any]] = {}
            for m_nm in interv_methods:
                xm, sm = actions[m_nm]
                ev = evaluate_v04_intervention_batch1(
                    model, h_s, h_L, logits, src_l, tgt_l, t, V_dark, xm, ro, option_token_ids, rec["target_option"], tau_frozen,
                    target_tol=float(cfg["control"]["pilot_target_tol"]),
                )
                dh_vec = ev.pop("actual_dh_vec")
                true_lin_f0 = float(np.dot(a0_exact, xm))
                pred_f0 = float(sm["pred_delta_F0"])
                sd_f0 = float(sm.get("pred_delta_F0_std", 0.0))
                # Coverage of chosen action: local linear a_true^T x vs end-to-end actual_delta_F0 (§6.3!)
                cov_lin_90 = int(abs(true_lin_f0 - pred_f0) <= Z_SCORES[0.90] * sd_f0) if sd_f0 > 0 else None
                cov_act_90 = int(abs(ev["actual_delta_F0"] - pred_f0) <= Z_SCORES[0.90] * sd_f0) if sd_f0 > 0 else None
                row_i = {
                    "split": split_name, "context_id": cid, "family": fam, "method": m_nm,
                    "tau": tau_frozen, "rho_c": rho_c, "x_sha256": bytes_sha256_short(xm),
                    "true_lin_delta_F0": true_lin_f0, "linearization_gap": float(ev["actual_delta_F0"] - true_lin_f0),
                    "action_cov_90_local_lin": cov_lin_90, "action_cov_90_actual_e2e": cov_act_90,
                    **sm, **ev,
                }
                int_rows_ctx.append(row_i)
                int_by_m[m_nm] = row_i
            append_jsonl(int_file, int_rows_ctx)

            # Common-action diagnostic truth on PRIOR action x_prior!
            ca_true_lin = float(np.dot(a0_exact, x_prior))
            ca_true_act = float(int_by_m["PRIOR"]["actual_delta_F0"])
            common_act_eval = {
                "true_lin_F0_at_x0": ca_true_lin, "actual_F0_at_x0": ca_true_act,
                "PRIOR_abs_err_lin": abs(ca_prior_mean - ca_true_lin),
                "UPDATE_8_abs_err_lin": abs(ca_upd8_mean - ca_true_lin),
                "PRIOR_abs_err_act": abs(ca_prior_mean - ca_true_act),
                "UPDATE_8_abs_err_act": abs(ca_upd8_mean - ca_true_act),
            }

            # Atlas index rows (§11)
            atlas_rows = []
            for kb, lbl in ((0, "PRIOR"), (4, "UPDATE_4"), (8, "UPDATE_8"), (16, "UPDATE_16")):
                em = eval_map_dict[lbl]
                im = int_by_m[lbl]
                atlas_rows.append({
                    "context_id": cid, "split": split_name, "family": fam,
                    "layers": f"h{src_l}->h{tgt_l}", "token_pos": t, "input_basis": "Dark38[858:896]",
                    "k": kb, "method": lbl, "posterior_id": em["posterior_id"],
                    "L_perp_norm": em["L_perp_norm"], "L_perp_raw_l2_sq": em["L_perp_raw_l2_sq"],
                    "f0_mse_qperp": em["f0_mse_qperp"],
                    "target_tracking_err": im["target_tracking_err"], "within_10pct": im["within_10pct"],
                    "actual_delta_F0": im["actual_delta_F0"], "non_target_kl": im["non_target_kl"],
                })
            append_jsonl(ROOT / "atlas_index.jsonl", atlas_rows)

            ctx_eval_records.append({
                "context_id": cid, "family": fam, "map_eval": eval_map_dict,
                "interventions": int_by_m, "common_action_x0": common_act_eval,
                "support_interp_rel_k8": st8["support_residual_rel"],
                "innovation_frob_k4": st4["innovation_frob"],
                "innovation_frob_k8": st8["innovation_frob"],
                "delta_M_k0_to_k8_frob": float(np.linalg.norm(st8["M_k"] - st0["M_k"], "fro")),
                "trace_C_k0": float(np.trace(st0["C_k"])),
                "trace_C_k8": float(np.trace(st8["C_k"])),
            })

            # Save observation & intervention vector archives
            obs_cids.append(cid); obs_phis.append(phi_c); obs_B.append(B); obs_Qg.append(Q_gen)
            obs_Ysup16.append(Y16.astype(np.float32)); obs_Yp.append(Yp_true.astype(np.float32))
            obs_Yg.append(Yg_true.astype(np.float32)); obs_g0.append(g0); obs_g3.append(G_opt3); obs_a0ex.append(a0_exact)
            all_int_cids.append(cid); all_int_splits.append(split_name)
            all_int_vecs.append(np.stack([actions[m][0] for m in interv_methods], axis=0).astype(np.float64))

            # If in 16-context trajectory panel, store h16/h20/h24 and z20/z24 for cols 0..24!
            if in_traj:
                Z20_sup16 = np.concatenate([Z20_4, Z20_4_8, Z20_8_16], axis=1)
                traj_store[cid] = {
                    "rec": rec, "t_last": t, "family": fam,
                    "h16_seq": h_s.detach().cpu(), "h20_seq": h_m.detach().cpu(),
                    "h16_tok": h_s[0, t, :].detach().to(torch.float64).cpu().numpy(),
                    "h20_tok": h_m[0, t, :].detach().to(torch.float64).cpu().numpy(),
                    "h24_tok": h_L.detach().to(torch.float64).cpu().numpy(),
                    "B_38x38": B, "g0": g0,
                    "Z20_0_24": np.concatenate([Z20_sup16, Z20_p], axis=1),
                    "Z24_0_24": np.concatenate([Z16, Zp_true], axis=1),
                    "M_by_k": {0: st0["M_k"], 4: st4["M_k"], 8: st8["M_k"], 16: st16["M_k"]},
                }

            if (i + 1) % 8 == 0 or i + 1 == n_s:
                hb.beat(f"CONFIRM_{split_name.upper()}", i + 1, n_s, note=f"{split_name} {i+1}/{n_s}")

        # Persist split observations & posteriors NPZ!
        np.savez_compressed(
            res_dir / "observations" / f"{split_name}_obs.npz",
            context_ids=np.array(obs_cids), phi=np.stack(obs_phis), B_38x38=np.stack(obs_B),
            Q_general=np.stack(obs_Qg), Y_sup_16=np.stack(obs_Ysup16), Y_perp_true=np.stack(obs_Yp),
            Y_gen_true=np.stack(obs_Yg), g0=np.stack(obs_g0), G_opt3=np.stack(obs_g3), a0_exact=np.stack(obs_a0ex),
        )
        np.savez_compressed(
            res_dir / "posteriors" / f"{split_name}_posteriors.npz",
            context_ids=np.array(obs_cids), budgets=np.array([0, 4, 8, 16]),
            M_k=post_M, C_k=post_C, posterior_ids=np.array(post_ids_arr, dtype=str),
        )

        # Aggregate split statistics with 20,000 family-stratified paired bootstrap!
        fams_s = [x["family"] for x in ctx_eval_records]
        map_summary = {}
        for lbl in ("PRIOR", "UPDATE_4", "UPDATE_8", "UPDATE_16", "PATCH_8", "MISPAIRED_8"):
            lp = np.array([x["map_eval"][lbl]["L_perp_norm"] for x in ctx_eval_records])
            lp_raw = np.array([x["map_eval"][lbl]["L_perp_raw_l2_sq"] for x in ctx_eval_records])
            lg = np.array([x["map_eval"][lbl]["L_gen_norm"] for x in ctx_eval_records])
            f0m = np.array([x["map_eval"][lbl]["f0_mse_qperp"] for x in ctx_eval_records])
            o3_vals = [x["map_eval"][lbl]["opt3_mse_qperp"] for x in ctx_eval_records if x["map_eval"][lbl]["opt3_mse_qperp"] is not None]
            map_summary[lbl] = {
                "L_perp_norm": family_stratified_paired_bootstrap(lp, None, fams_s, n_boot, b_seed),
                "L_perp_raw_l2_sq": family_stratified_paired_bootstrap(lp_raw, None, fams_s, n_boot, b_seed),
                "L_perp_in_Uold64_sq_mean": float(np.mean([x["map_eval"][lbl]["L_perp_in_Uold64_sq"] for x in ctx_eval_records])),
                "L_perp_out_Uold64_sq_mean": float(np.mean([x["map_eval"][lbl]["L_perp_out_Uold64_sq"] for x in ctx_eval_records])),
                "L_gen_norm": family_stratified_paired_bootstrap(lg, None, fams_s, n_boot, b_seed),
                "f0_mse_qperp": family_stratified_paired_bootstrap(f0m, None, fams_s, n_boot, b_seed),
                "f0_sign_acc_qperp_mean": float(np.mean([x["map_eval"][lbl]["f0_sign_acc_qperp"] for x in ctx_eval_records])),
                "opt3_mse_qperp_mean": float(np.mean(o3_vals)) if o3_vals else None,
                "f0_qperp_cov_mean": {lv: float(np.mean([x["map_eval"][lbl]["f0_qperp_cov"][lv] for x in ctx_eval_records])) for lv in ("50", "80", "90", "95")},
                "f0_qperp_mean_width_90": float(np.mean([x["map_eval"][lbl]["f0_qperp_mean_width_90"] for x in ctx_eval_records])),
                "f0_qperp_mean_nll": float(np.mean([x["map_eval"][lbl]["f0_qperp_mean_nll"] for x in ctx_eval_records])),
            }

        # Paired comparisons (Link 1 primary: L_0^perp - L_8^perp; Link 2: F0 MSE_0 - MSE_8; Link 3: E_ctrl_0 - E_ctrl_8!)
        paired_links = {}
        l0 = np.array([x["map_eval"]["PRIOR"]["L_perp_norm"] for x in ctx_eval_records])
        f0_0 = np.array([x["map_eval"]["PRIOR"]["f0_mse_qperp"] for x in ctx_eval_records])
        ec_0 = np.array([x["interventions"]["PRIOR"]["target_tracking_err"] for x in ctx_eval_records])
        for lbl in ("UPDATE_4", "UPDATE_8", "UPDATE_16", "PATCH_8", "MISPAIRED_8"):
            lk = np.array([x["map_eval"][lbl]["L_perp_norm"] for x in ctx_eval_records])
            fk = np.array([x["map_eval"][lbl]["f0_mse_qperp"] for x in ctx_eval_records])
            paired_links[f"PRIOR_minus_{lbl}_L_perp"] = family_stratified_paired_bootstrap(l0, lk, fams_s, n_boot, b_seed)
            paired_links[f"PRIOR_minus_{lbl}_L_perp"]["rel_reduction"] = float((np.mean(l0) - np.mean(lk)) / max(float(np.mean(l0)), 1e-12))
            paired_links[f"PRIOR_minus_{lbl}_f0_mse_qperp"] = family_stratified_paired_bootstrap(f0_0, fk, fams_s, n_boot, b_seed)
            paired_links[f"PRIOR_minus_{lbl}_f0_mse_qperp"]["rel_reduction"] = float((np.mean(f0_0) - np.mean(fk)) / max(float(np.mean(f0_0)), 1e-12))
            if lbl != "MISPAIRED_8":
                eck = np.array([x["interventions"][lbl]["target_tracking_err"] for x in ctx_eval_records])
                paired_links[f"PRIOR_minus_{lbl}_ctrl_err"] = family_stratified_paired_bootstrap(ec_0, eck, fams_s, n_boot, b_seed)
                paired_links[f"PRIOR_minus_{lbl}_ctrl_err"]["rel_reduction"] = float((np.mean(ec_0) - np.mean(eck)) / max(float(np.mean(ec_0)), 1e-12))

        # Per-family breakdown of primary Link 1 (PRIOR - UPDATE_8)
        fam_breakdown = {}
        for fm in sorted(set(fams_s)):
            sub_c = [x for x in ctx_eval_records if x["family"] == fm]
            d_l = np.array([x["map_eval"]["PRIOR"]["L_perp_norm"] - x["map_eval"]["UPDATE_8"]["L_perp_norm"] for x in sub_c])
            fam_breakdown[fm] = {
                "n": len(sub_c),
                "mean_L0_perp": float(np.mean([x["map_eval"]["PRIOR"]["L_perp_norm"] for x in sub_c])),
                "mean_L8_perp": float(np.mean([x["map_eval"]["UPDATE_8"]["L_perp_norm"] for x in sub_c])),
                "mean_improvement": float(np.mean(d_l)),
                "improved_frac": float(np.mean(d_l > 0.0)),
            }

        # Intervention summary per arm
        int_summary = {}
        for mn in interv_methods:
            ec = np.array([x["interventions"][mn]["target_tracking_err"] for x in ctx_eval_records])
            w10 = np.array([x["interventions"][mn]["within_10pct"] for x in ctx_eval_records], dtype=np.float64)
            c_lin = [x["interventions"][mn]["action_cov_90_local_lin"] for x in ctx_eval_records if x["interventions"][mn]["action_cov_90_local_lin"] is not None]
            c_act = [x["interventions"][mn]["action_cov_90_actual_e2e"] for x in ctx_eval_records if x["interventions"][mn]["action_cov_90_actual_e2e"] is not None]
            int_summary[mn] = {
                "target_tracking_err": family_stratified_paired_bootstrap(ec, None, fams_s, n_boot, b_seed),
                "within_10pct_rate": family_stratified_paired_bootstrap(w10, None, fams_s, n_boot, b_seed),
                "mean_actual_delta_F0": float(np.mean([x["interventions"][mn]["actual_delta_F0"] for x in ctx_eval_records])),
                "mean_step_norm": float(np.mean([x["interventions"][mn]["step_norm"] for x in ctx_eval_records])),
                "mean_actual_dh_l2": float(np.mean([x["interventions"][mn]["actual_dh_l2"] for x in ctx_eval_records])),
                "mean_non_target_kl": float(np.mean([x["interventions"][mn]["non_target_kl"] for x in ctx_eval_records])),
                "base_4choice_acc": float(np.mean([x["interventions"][mn]["base_4choice_correct"] for x in ctx_eval_records])),
                "pert_4choice_acc": float(np.mean([x["interventions"][mn]["pert_4choice_correct"] for x in ctx_eval_records])),
                "unreachable_rate": float(np.mean([x["interventions"][mn]["hit_boundary_unreachable"] for x in ctx_eval_records])),
                "action_cov_90_local_lin": float(np.mean(c_lin)) if c_lin else None,
                "action_cov_90_actual_e2e": float(np.mean(c_act)) if c_act else None,
                "mean_abs_linearization_gap": float(np.mean([abs(x["interventions"][mn]["linearization_gap"]) for x in ctx_eval_records])),
            }

        # Common-action diagnostic summary (§7.2)
        ca_p_lin = np.array([x["common_action_x0"]["PRIOR_abs_err_lin"] for x in ctx_eval_records])
        ca_u_lin = np.array([x["common_action_x0"]["UPDATE_8_abs_err_lin"] for x in ctx_eval_records])
        common_act_sum = {
            "PRIOR_mean_abs_err_lin": float(np.mean(ca_p_lin)),
            "UPDATE_8_mean_abs_err_lin": float(np.mean(ca_u_lin)),
            "PRIOR_minus_UPDATE_8_abs_err_lin": family_stratified_paired_bootstrap(ca_p_lin, ca_u_lin, fams_s, n_boot, b_seed),
        }

        return {
            "n_contexts": n_s,
            "map_summary": map_summary,
            "paired_links": paired_links,
            "family_breakdown_L0_minus_L8": fam_breakdown,
            "interventions": int_summary,
            "common_action_x0_diagnostic": common_act_sum,
            "mean_support_interp_rel_k8": float(np.mean([x["support_interp_rel_k8"] for x in ctx_eval_records])),
            "mean_delta_M_k0_to_k8_frob": float(np.mean([x["delta_M_k0_to_k8_frob"] for x in ctx_eval_records])),
            "mean_trace_C_k0_to_k8": [float(np.mean([x["trace_C_k0"] for x in ctx_eval_records])), float(np.mean([x["trace_C_k8"] for x in ctx_eval_records]))],
        }

    confirm_id_sum = _run_confirm_split("confirm_id", load_jsonl(ROOT / "data" / "confirm_id.jsonl"))
    confirm_ood_sum = _run_confirm_split("confirm_ood", load_jsonl(ROOT / "data" / "confirm_ood.jsonl"))

    np.savez_compressed(
        res_dir / "interventions" / "intervention_vectors.npz",
        context_ids=np.array(all_int_cids), splits=np.array(all_int_splits),
        methods=np.array(interv_methods), X_vectors=np.stack(all_int_vecs, axis=0),
    )

    # Print core 3-link Confirm-ID96 verdict immediately!
    l1 = confirm_id_sum["paired_links"]["PRIOR_minus_UPDATE_8_L_perp"]
    l2 = confirm_id_sum["paired_links"]["PRIOR_minus_UPDATE_8_f0_mse_qperp"]
    l3 = confirm_id_sum["paired_links"]["PRIOR_minus_UPDATE_8_ctrl_err"]
    print(f"[04 CONFIRM-ID96] Link1 L_perp: {l1['mean']:+.5f} {l1['ci95']} (-{l1['rel_reduction']*100:.2f}%, PASS={l1['ci95_strictly_positive']})", flush=True)
    print(f"[04 CONFIRM-ID96] Link2 F0 MSE: {l2['mean']:+.6f} {l2['ci95']} (-{l2['rel_reduction']*100:.2f}%, PASS={l2['ci95_strictly_positive']})", flush=True)
    print(f"[04 CONFIRM-ID96] Link3 CtrlEr: {l3['mean']:+.5f} {l3['ci95']} (-{l3['rel_reduction']*100:.2f}%, PASS={l3['ci95_strictly_positive']})", flush=True)

    # =========================================================================
    # STEP 5: 16-CONTEXT TRAJECTORY PANEL COMPLETION & OPTIONAL MECHANISM DIAGNOSTICS (§8)
    # =========================================================================
    hb.beat("TRAJECTORY_16", 0, 16, note="Completing 14 remaining bank cols for 16 trajectory contexts", force=True)
    P_U_old = U_old_64 @ U_old_64.T
    I_m_PU = np.eye(d) - P_U_old
    P_D = V_dark_np @ V_dark_np.T
    I_m_PD = np.eye(d) - P_D

    traj_cids, traj_h16, traj_h20, traj_h24, traj_Z20_full, traj_Z24_full = [], [], [], [], [], []
    four_blocks_by_k: dict[str, list[dict[str, float]]] = {f"k{kb}": [] for kb in (0, 4, 8, 16)}
    inter_layer_stats: list[dict[str, float]] = []
    mech_diag1_rows: list[dict[str, Any]] = []
    mech_diag2_rows: list[dict[str, Any]] = []

    run_optional_diags = ((time.time() - t_start) / 60.0) < float(cfg["budget"]["optional_diag_cutoff_min"])
    V_bright4_t = basis.V_full[:, 0:4].to(device=device, dtype=torch.float32)
    rng_rand = np.random.default_rng(int(cfg["derived_seeds"]["random_subspace_diag"]))

    for ti, cid in enumerate(traj_16_order):
        ts = traj_store[cid]
        h_s = ts["h16_seq"].to(device)
        h_m = ts["h20_seq"].to(device)
        t = ts["t_last"]
        B = ts["B_38x38"]
        B_rem_14 = B[:, 24:38]

        sess_t = ProbeSessionV04(f"traj_comp_{cid}", "evaluator_truth", cid, model, h_s, V_dark, src_l, mid_l, tgt_l, t, max_jvp_dirs=14, allow_vjp=False, jvp_batch_size=8)
        Z24_rem, Z20_rem = sess_t.jvp_columns(B_rem_14, "traj_rem14", record_mid_z20=True)
        append_jsonl(res_dir / "observations" / "query_receipts.jsonl", [sess_t.receipt()])

        Z20_bank = np.concatenate([ts["Z20_0_24"], Z20_rem], axis=1)  # [896, 38]
        Z24_bank = np.concatenate([ts["Z24_0_24"], Z24_rem], axis=1)  # [896, 38]
        Z20_full = Z20_bank @ B.T
        Z24_full = Z24_bank @ B.T
        A_full = Z24_full - V_dark_np

        traj_cids.append(cid); traj_h16.append(ts["h16_tok"]); traj_h20.append(ts["h20_tok"]); traj_h24.append(ts["h24_tok"])
        traj_Z20_full.append(Z20_full.astype(np.float32)); traj_Z24_full.append(Z24_full.astype(np.float32))

        # Exact 4-block error decomposition on full 38-D operator truth!
        for kb in (0, 4, 8, 16):
            X_k = B[:, :kb]
            P_X = X_k @ X_k.T if kb > 0 else np.zeros((r, r))
            I_m_PX = np.eye(r) - P_X
            E = A_full - ts["M_by_k"][kb]
            four_blocks_by_k[f"k{kb}"].append({
                "inU_obsX": float(np.sum((P_U_old @ E @ P_X) ** 2)),
                "inU_unobsX": float(np.sum((P_U_old @ E @ I_m_PX) ** 2)),
                "outU_obsX": float(np.sum((I_m_PU @ E @ P_X) ** 2)),
                "outU_unobsX": float(np.sum((I_m_PU @ E @ I_m_PX) ** 2)),
                "total_frob_sq": float(np.sum(E ** 2)),
            })

        # Inter-layer incremental tangent stats on Q_perp (cols 16..24 of bank)
        V_qp = V_dark_np @ B[:, 16:24]
        z20_qp = Z20_bank[:, 16:24]
        z24_qp = Z24_bank[:, 16:24]
        d16_20 = z20_qp - V_qp
        d20_24 = z24_qp - z20_qp
        g0 = ts["g0"]
        inter_layer_stats.append({
            "mean_norm_v16": float(np.mean(np.linalg.norm(V_qp, axis=0))),
            "mean_norm_z20": float(np.mean(np.linalg.norm(z20_qp, axis=0))),
            "mean_norm_z24": float(np.mean(np.linalg.norm(z24_qp, axis=0))),
            "mean_norm_delta_16_20": float(np.mean(np.linalg.norm(d16_20, axis=0))),
            "mean_norm_delta_20_24": float(np.mean(np.linalg.norm(d20_24, axis=0))),
            "z20_dark_energy_frac": float(np.sum((V_dark_np.T @ z20_qp) ** 2) / max(float(np.sum(z20_qp ** 2)), 1e-12)),
            "z24_dark_energy_frac": float(np.sum((V_dark_np.T @ z24_qp) ** 2) / max(float(np.sum(z24_qp ** 2)), 1e-12)),
            "mean_abs_g24_dot_z20_fixed_readout_proj": float(np.mean(np.abs(g0 @ z20_qp))),
            "mean_abs_g24_dot_z24_end_func_resp": float(np.mean(np.abs(g0 @ z24_qp))),
        })

        # Optional Mechanism Diagnostics (§8) if within 45 min cutoff
        if run_optional_diags:
            # Diag 1: First Q_perp dir's z20 split into P_D z20 + (I - P_D) z20 -> h20->h24 JVP!
            z20_0 = z20_qp[:, 0]
            z24_0 = z24_qp[:, 0]
            z20_dark = P_D @ z20_0
            z20_comp = I_m_PD @ z20_0
            pair_t = torch.tensor(np.stack([z20_dark, z20_comp], axis=1), dtype=torch.float32, device=device)
            out_pair = compute_exact_jvp_columns_batched(model, h_m, mid_l, tgt_l, t, pair_t, batch_size=2).to(torch.float64).cpu().numpy()
            z24_from_dark, z24_from_comp = out_pair[:, 0], out_pair[:, 1]
            sum_rel_err = float(np.linalg.norm((z24_from_dark + z24_from_comp) - z24_0) / max(float(np.linalg.norm(z24_0)), 1e-12))
            mech_diag1_rows.append({
                "context_id": cid, "family": ts["family"],
                "z20_dark_norm": float(np.linalg.norm(z20_dark)), "z20_comp_norm": float(np.linalg.norm(z20_comp)),
                "z24_from_dark_norm": float(np.linalg.norm(z24_from_dark)), "z24_from_comp_norm": float(np.linalg.norm(z24_from_comp)),
                "g_dot_z24_from_dark": float(np.dot(g0, z24_from_dark)), "g_dot_z24_from_comp": float(np.dot(g0, z24_from_comp)),
                "g_dot_z24_total": float(np.dot(g0, z24_0)),
                "exact_sum_rel_err": sum_rel_err,
            })

            # Diag 2: First 4 dark dirs vs Bright4 vs RandomOrth4 (equal norm=1, equal count=4)
            z24_dark4 = Z24_bank[:, 0:4]
            z24_bright4 = compute_exact_jvp_columns_batched(model, h_s, src_l, tgt_l, t, V_bright4_t, batch_size=4).to(torch.float64).cpu().numpy()
            Q_ro, _ = np.linalg.qr(rng_rand.normal(size=(d, 4)))
            z24_rand4 = compute_exact_jvp_columns_batched(model, h_s, src_l, tgt_l, t, torch.tensor(Q_ro, dtype=torch.float32, device=device), batch_size=4).to(torch.float64).cpu().numpy()
            mech_diag2_rows.append({
                "context_id": cid, "family": ts["family"],
                "dark4_mean_z24_norm": float(np.mean(np.linalg.norm(z24_dark4, axis=0))),
                "bright4_mean_z24_norm": float(np.mean(np.linalg.norm(z24_bright4, axis=0))),
                "rand4_mean_z24_norm": float(np.mean(np.linalg.norm(z24_rand4, axis=0))),
                "dark4_mean_abs_g_dot_z24": float(np.mean(np.abs(g0 @ z24_dark4))),
                "bright4_mean_abs_g_dot_z24": float(np.mean(np.abs(g0 @ z24_bright4))),
                "rand4_mean_abs_g_dot_z24": float(np.mean(np.abs(g0 @ z24_rand4))),
            })

        if (ti + 1) % 4 == 0:
            hb.beat("TRAJECTORY_16", ti + 1, 16, note=f"Trajectory panel {ti+1}/16")

    np.savez_compressed(
        res_dir / "trajectory_panel" / "trajectory_16_states.npz",
        context_ids=np.array(traj_cids), h16=np.stack(traj_h16), h20=np.stack(traj_h20), h24=np.stack(traj_h24),
        Z20_full=np.stack(traj_Z20_full), Z24_full=np.stack(traj_Z24_full),
    )
    four_blocks_summary = {
        kb: {k_f: float(np.mean([r[k_f] for r in rows])) for k_f in ("inU_obsX", "inU_unobsX", "outU_obsX", "outU_unobsX", "total_frob_sq")}
        for kb, rows in four_blocks_by_k.items()
    }
    traj_summary = {
        "n_contexts": 16, "context_ids": traj_cids,
        "four_block_exact_38D_errors": four_blocks_summary,
        "inter_layer_qperp_summary": {k_f: float(np.mean([r[k_f] for r in inter_layer_stats])) for k_f in inter_layer_stats[0]},
        "optional_diagnostics_executed": run_optional_diags,
    }
    save_json(res_dir / "trajectory_panel" / "trajectory_16_summary.json", traj_summary)
    if run_optional_diags:
        mech_sum = {
            "diag1_z20_split_additivity_max_rel_err": float(max(r["exact_sum_rel_err"] for r in mech_diag1_rows)),
            "diag1_mean_z24_from_dark_norm": float(np.mean([r["z24_from_dark_norm"] for r in mech_diag1_rows])),
            "diag1_mean_z24_from_comp_norm": float(np.mean([r["z24_from_comp_norm"] for r in mech_diag1_rows])),
            "diag1_mean_abs_g_from_dark": float(np.mean([abs(r["g_dot_z24_from_dark"]) for r in mech_diag1_rows])),
            "diag1_mean_abs_g_from_comp": float(np.mean([abs(r["g_dot_z24_from_comp"]) for r in mech_diag1_rows])),
            "diag1_per_context": mech_diag1_rows,
            "diag2_mean_z24_norm": {
                "dark4": float(np.mean([r["dark4_mean_z24_norm"] for r in mech_diag2_rows])),
                "bright4": float(np.mean([r["bright4_mean_z24_norm"] for r in mech_diag2_rows])),
                "rand4": float(np.mean([r["rand4_mean_z24_norm"] for r in mech_diag2_rows])),
            },
            "diag2_mean_abs_g_dot_z24": {
                "dark4": float(np.mean([r["dark4_mean_abs_g_dot_z24"] for r in mech_diag2_rows])),
                "bright4": float(np.mean([r["bright4_mean_abs_g_dot_z24"] for r in mech_diag2_rows])),
                "rand4": float(np.mean([r["rand4_mean_abs_g_dot_z24"] for r in mech_diag2_rows])),
            },
            "diag2_per_context": mech_diag2_rows,
        }
        save_json(res_dir / "trajectory_panel" / "trajectory_16_mechanisms.json", mech_sum)

    elapsed_sec = time.time() - t_start
    summary = {
        "experiment_id": "BAYES-DYN-04",
        "status": "COMPLETED",
        "started_utc": utc_start, "finished_utc": utc_now(),
        "elapsed_sec": round(elapsed_sec, 2), "elapsed_gpu_hours": round(elapsed_sec / 3600.0, 4),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1),
        "split_first_access_utc": split_access_utc,
        "confirm_id": confirm_id_sum,
        "confirm_ood": confirm_ood_sum,
        "trajectory_panel_16": traj_summary,
    }
    save_json(res_dir / "metrics_summary.json", summary)
    hb.beat("DONE", 1, 1, status="COMPLETED", note=f"Finished cleanly in {elapsed_sec:.1f}s", force=True)
    print("=" * 78)
    print(f"[04 COMPLETED] Elapsed: {elapsed_sec:.1f}s ({elapsed_sec/3600.0:.4f} GPU-hr) | Peak VRAM: {summary['peak_vram_mb']} MB")
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
