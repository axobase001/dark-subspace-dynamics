"""BAYES-DYN-05 Single-Model Cross-Scale & Cross-Architecture GPU Driver.

Runs the complete, pre-registered, strictly isolated 3-link verification protocol on any
configured model in configs/suite_config.json:
  Phase 0: Load FP32 eager model on CPU -> Exact FP64 SVD of W_U on empty GPU -> move to GPU
  Phase 1: P0 verification (FP64 synth, zero-hook, zero-pert, JVP batch/repeat/lin/zmid, FD, adjoint, isolation)
  Phase 2: Live DEV collection (DEV-fit 96 + DEV-tune 16, exact 38-D JVP) -> fast (alpha, eta) prior fit
  Phase 3: DEV-PILOT 16 (freeze working tau) + CAL 24 (fit global temperature t) -> write frozen_config.json
  Phase 4: Confirm-ID 64 & Confirm-OOD 32 (strict Stage A -> Commit -> Stage B, real batch=1 interventions)
  Phase 5: 12-context inter-layer trajectory panel (h_src -> h_mid -> h_tgt, 4-block 38-D + Diag 1 & Diag 2)
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
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

from src.control_v05 import (
    Z_SCORES,
    evaluate_v05_intervention_batch1,
    scalar_gaussian_nll,
    solve_min_norm_ball_action,
)
from src.full_operator_map_v05 import (
    build_top64_dynamic_output_basis,
    condition_exact_batch,
    condition_patch_only,
    fit_and_select_full_operator_prior_on_dev,
    fit_cal_temperature,
    make_mispaired_Y,
    predict_M0,
    predict_scalar_direction_dist,
    project_function_from_posterior,
    save_full_operator_prior_npz,
    verify_full_operator_synth_fp64,
)
from src.probe_bank_v05 import (
    BudgetExceededError,
    ProbeSessionV05,
    UnauthorizedQueryError,
    generate_fixed_38x38_bank,
    generate_q_general,
)
from src.spectral_basis_v05 import compute_spectral_basis_pre_move, extract_readouts_v05
from src.transport_v05 import (
    capture_residual_states,
    compute_exact_jvp_columns_batched,
    extract_context_features,
    resolve_option_token_ids_v05,
    run_segment_layers,
    tokenize_chat_prompt_v05,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bytes_sha256_short(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def compute_full_weight_sha256_cpu(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        h.update(name.encode("utf-8"))
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class HeartbeatWriter:
    def __init__(self, path: Path, model_key: str) -> None:
        self.path = path
        self.model_key = model_key
        self.last_write = 0.0

    def beat(self, phase: str, step: int, total: int, status: str = "RUNNING", note: str = "", force: bool = False) -> None:
        now = time.time()
        if (not force) and (now - self.last_write < 8.0):
            return
        self.last_write = now
        payload = {
            "experiment_id": "BAYES-DYN-05",
            "model_key": self.model_key,
            "pid": os.getpid(),
            "utc": utc_now(),
            "epoch": now,
            "status": status,
            "phase": phase,
            "step": step,
            "total": total,
            "note": note,
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, self.path)


def family_stratified_paired_bootstrap(
    vals_a: np.ndarray,
    vals_b: np.ndarray | None,
    families: list[str],
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    d = np.asarray(vals_a, dtype=np.float64) if vals_b is None else (np.asarray(vals_a, dtype=np.float64) - np.asarray(vals_b, dtype=np.float64))
    n = len(d)
    by_fam: dict[str, list[int]] = {}
    for i, fm in enumerate(families):
        by_fam.setdefault(fm, []).append(i)
    fam_idx = [np.array(v, dtype=np.int64) for v in by_fam.values()]

    rng = np.random.default_rng(int(seed))
    boot_sum = np.zeros(n_resamples, dtype=np.float64)
    for idxs in fam_idx:
        nf = len(idxs)
        sampled = rng.integers(0, nf, size=(n_resamples, nf))
        boot_sum += np.sum(d[idxs[sampled]], axis=1)
    boot_means = boot_sum / float(n)
    lo, hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
    return {
        "n": n,
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "std": float(np.std(d)),
        "ci95": [lo, hi],
        "ci95_strictly_positive": bool(lo > 0.0),
        "improved_frac": float(np.mean(d > 0.0)),
    }


def run_p0_checks_v05(
    model: torch.nn.Module,
    tokenizer: Any,
    basis: Any,
    V_dark_np: np.ndarray,
    pilot_4: list[dict[str, Any]],
    option_token_ids: dict[str, int],
    mcfg: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    device = next(model.parameters()).device
    gamma32 = basis.rms_gamma.to(device=device, dtype=torch.float32)
    V_dark = basis.V_dark.to(device=device, dtype=torch.float32)
    src_l, mid_l, tgt_l = int(mcfg["source_layer"]), int(mcfg["mid_layer"]), int(mcfg["target_layer"])
    jvp_bs = int(mcfg["jvp_batch_size"])
    th = cfg["p0_thresholds"]
    fd_steps = [float(s) for s in th["fd_step_sizes"]]
    abs_floor = float(th["near_zero_denom_abs_floor"])
    bank_seed = int(cfg["derived_seeds"]["probe_bank_38x38"])
    r = basis.n_dark_probe

    synth_fp64 = verify_full_operator_synth_fp64(int(cfg["master_seed"]))
    zero_D, zero_dF, zero_hook_diffs = [], [], []
    jvp_b_vs_b1, lin_comb_errs, repeat_errs, mid_z_errs, adjoint_errs, rms_an_diffs = [], [], [], [], [], []
    jvp_fd_by_step: dict[str, list[dict[str, float]]] = {str(s): [] for s in fd_steps}

    for idx, rec in enumerate(pilot_4):
        tok = tokenize_chat_prompt_v05(tokenizer, rec["user_prompt"], device)
        states, logits = capture_residual_states(model, tok["input_ids"], tok["attention_mask"])
        t = tok["t_last"]
        h_s, h_m, h_L = states[src_l], states[mid_l], states[tgt_l][0, t, :]

        with torch.no_grad():
            nat = model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"], use_cache=False)
            zero_hook_diffs.append(float(torch.max(torch.abs(nat.logits[0, t, :] - logits)).item()))

        ro = extract_readouts_v05(model, logits, h_L, gamma32, basis.rms_eps, option_token_ids, rec["target_option"])
        rms_an_diffs.append(float(ro["F0"]["rmsnorm_analytic_rel_diff"]))
        ev0 = evaluate_v05_intervention_batch1(
            model, h_s, h_L, logits, src_l, tgt_l, t, V_dark,
            np.zeros(r, dtype=np.float64), ro, option_token_ids, rec["target_option"], tau=0.05,
        )
        zero_D.append(ev0["actual_D_geom"])
        zero_dF.append(abs(ev0["actual_delta_F0"]))

        B = generate_fixed_38x38_bank(rec["context_id"], bank_seed, r=r)
        X4 = B[:, :4]
        sess_p0 = ProbeSessionV05(f"p0_{idx}", "evaluator_truth", rec["context_id"], model, h_s, V_dark, src_l, mid_l, tgt_l, t, max_jvp_dirs=24, allow_vjp=True, jvp_batch_size=jvp_bs)
        Z_b, Z_mid = sess_p0.jvp_columns(X4, "p0_batch", record_mid_z=True)
        Z_rep, _ = sess_p0.jvp_columns(X4[:, :2], "p0_rep", record_mid_z=False)
        repeat_errs.append(float(np.linalg.norm(Z_rep - Z_b[:, :2], "fro") / max(float(np.linalg.norm(Z_b[:, :2], "fro")), abs_floor)))

        Z_b1_0, _ = sess_p0.jvp_columns(X4[:, 0:1], "p0_b1_0", record_mid_z=False)
        Z_b1_1, _ = sess_p0.jvp_columns(X4[:, 1:2], "p0_b1_1", record_mid_z=False)
        Z_b1 = np.concatenate([Z_b1_0, Z_b1_1], axis=1)
        jvp_b_vs_b1.append(float(np.linalg.norm(Z_b[:, :2] - Z_b1, "fro") / max(float(np.linalg.norm(Z_b1, "fro")), abs_floor)))

        x_lc = 0.6 * X4[:, 0:1] - 0.8 * X4[:, 1:2]
        Z_lc, _ = sess_p0.jvp_columns(x_lc, "p0_lincomb", record_mid_z=False)
        Z_lc_ref = 0.6 * Z_b[:, 0:1] - 0.8 * Z_b[:, 1:2]
        lin_comb_errs.append(float(np.linalg.norm(Z_lc - Z_lc_ref) / max(float(np.linalg.norm(Z_lc_ref)), abs_floor)))

        # Mid-layer chain consistency: J_{mid->tgt}(z_mid) == z_tgt!
        zm_t = torch.tensor(Z_mid[:, 0:1], dtype=torch.float32, device=device)
        z_from_mid = compute_exact_jvp_columns_batched(model, h_m, mid_l, tgt_l, t, zm_t, batch_size=1)[:, 0].to(torch.float64).cpu().numpy()
        mid_z_errs.append(float(np.linalg.norm(z_from_mid - Z_b[:, 0]) / max(float(np.linalg.norm(Z_b[:, 0])), abs_floor)))

        # Central FD curve on col 0
        hs_scale = float(torch.linalg.vector_norm(h_s[0, t, :]).item())
        v0 = V_dark @ torch.tensor(X4[:, 0], dtype=torch.float32, device=device)
        for step in fd_steps:
            eps_s = step * hs_scale
            hp = h_s.clone(); hp[0, t, :] = hp[0, t, :] + eps_s * v0
            hm = h_s.clone(); hm[0, t, :] = hm[0, t, :] - eps_s * v0
            with torch.no_grad():
                yp = run_segment_layers(model, hp, src_l, tgt_l)[0, t, :].to(torch.float64).cpu().numpy()
                ym = run_segment_layers(model, hm, src_l, tgt_l)[0, t, :].to(torch.float64).cpu().numpy()
            z_fd = (yp - ym) / (2.0 * eps_s)
            ae = float(np.linalg.norm(Z_b[:, 0] - z_fd))
            dn = float(np.linalg.norm(z_fd))
            jvp_fd_by_step[str(step)].append({"abs_err": ae, "rel_err": ae / max(dn, abs_floor), "denom": dn})

        a0_vjp = sess_p0.vjp_eval_reference(ro["F0"]["g"], "F0")
        lhs = float(np.dot(Z_b[:, 0], ro["F0"]["g"]))
        rhs = float(np.dot(X4[:, 0], a0_vjp))
        adjoint_errs.append(abs(lhs - rhs) / max(abs(lhs), abs(rhs), abs_floor))

    # Isolation checks: budget rejection + VJP rejection
    rec0 = pilot_4[0]
    tok0 = tokenize_chat_prompt_v05(tokenizer, rec0["user_prompt"], device)
    st0, _ = capture_residual_states(model, tok0["input_ids"], tok0["attention_mask"])
    t0 = tok0["t_last"]
    B0 = generate_fixed_38x38_bank(rec0["context_id"], bank_seed, r=r)
    sess_iso = ProbeSessionV05("p0_iso", "learning_support", rec0["context_id"], model, st0[src_l], V_dark, src_l, mid_l, tgt_l, t0, max_jvp_dirs=2, allow_vjp=False, jvp_batch_size=2)
    sess_iso.jvp_columns(B0[:, :2], "iso_k2")
    budget_rejected, vjp_rejected = False, False
    try:
        sess_iso.jvp_columns(B0[:, 2:3], "iso_over")
    except BudgetExceededError:
        budget_rejected = True
    try:
        sess_iso.vjp_eval_reference(np.ones(V_dark_np.shape[0]), "F0")
    except UnauthorizedQueryError:
        vjp_rejected = True

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
        and max(zero_hook_diffs) <= float(th["zero_hook_max_logit_diff"])
        and best_fd_med <= float(th["jvp_fd_median_max"])
        and max(adjoint_errs) <= float(th["adjoint_rel_max"])
        and max(lin_comb_errs) <= float(th["linearity_rel_max"])
        and max(jvp_b_vs_b1) <= float(th["jvp_batched_rel_max"])
        and max(mid_z_errs) <= float(th["mid_chain_rel_max"])
        and budget_rejected and vjp_rejected
    )

    return {
        "all_p0_passed": passed,
        "fp64_synthetic_conditioning": synth_fp64,
        "zero_hook_max_logit_diff": float(max(zero_hook_diffs)),
        "zero_pert_matched_batch1": {"max_D_geom": float(max(zero_D)), "max_abs_dF": float(max(zero_dF))},
        "rmsnorm_analytic_vs_module_max_rel_diff": float(max(rms_an_diffs)),
        "jvp_batched_vs_batch1_max_rel_err": float(max(jvp_b_vs_b1)),
        "jvp_repeatability_max_rel_err": float(max(repeat_errs)),
        "jvp_linear_comb_max_rel_err": float(max(lin_comb_errs)),
        "mid_layer_chain_max_rel_err": float(max(mid_z_errs)),
        "jvp_fd_curve": {"by_step": fd_sum, "best_step": best_fd_step, "best_median_rel_err": best_fd_med},
        "adjoint_consistency_max_rel_err": float(max(adjoint_errs)),
        "isolation_checks": {"budget_rejected": budget_rejected, "learning_vjp_rejected": vjp_rejected},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_key", required=True)
    args = ap.parse_args()
    mkey = args.model_key

    t_start = time.time()
    utc_start = utc_now()

    cfg = json.loads((ROOT / "configs" / "suite_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "configs" / "data_manifest.json").read_text(encoding="utf-8"))
    assert mkey in cfg["models"], f"Unknown model_key={mkey}"
    mcfg = cfg["models"][mkey]

    out_dir = ROOT / "models" / mkey
    for sub in ("configs", "observations", "posteriors", "interventions", "calibration", "trajectory_panel"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    for p in (
        out_dir / "observations" / "query_receipts.jsonl",
        out_dir / "predictions.jsonl",
        out_dir / "interventions" / "dev_pilot_interventions.jsonl",
        out_dir / "interventions" / "confirm_id_interventions.jsonl",
        out_dir / "interventions" / "confirm_ood_interventions.jsonl",
    ):
        if p.exists():
            p.unlink()

    hb = HeartbeatWriter(ROOT / f"heartbeat_{mkey}.json", mkey)
    hb.beat("INIT", 0, 1, note=f"Starting BAYES-DYN-05 on {mcfg['label']}", force=True)

    pip_out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    (out_dir / "pip_freeze.txt").write_text(pip_out.stdout, encoding="utf-8")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(int(cfg["master_seed"]))
    np.random.seed(int(cfg["master_seed"]))
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    model_id = mcfg["hf_id"]
    rev = mcfg["revision"]
    src_l, mid_l, tgt_l = int(mcfg["source_layer"]), int(mcfg["mid_layer"]), int(mcfg["target_layer"])
    jvp_bs = int(mcfg["jvp_batch_size"])
    dev_jvp_bs = int(mcfg["dev_jvp_batch_size"])
    r = int(cfg["dark_definition"]["probe_dim_r"])
    bank_seed = int(cfg["derived_seeds"]["probe_bank_38x38"])
    qgen_seed = int(cfg["derived_seeds"]["q_general_4"])
    rho_alpha = float(cfg["control"]["radius_alpha"])
    zero_tol = float(cfg["control"]["zero_a_rel_tol"])
    n_boot = int(cfg["stats"]["bootstrap_resamples"])
    b_seed = int(cfg["derived_seeds"]["bootstrap_stats"])
    S_mispair = np.array(manifest["S_mispair_8x8"], dtype=np.float64)
    traj_12_order = list(manifest["pre_registered_trajectory_panel_12"])
    traj_12_set = set(traj_12_order)

    # =========================================================================
    # PHASE 0: LOAD MODEL ON CPU -> EXACT FP64 SVD ON EMPTY GPU -> MOVE TO GPU
    # =========================================================================
    hb.beat("LOAD_MODEL", 0, 1, note=f"Loading {model_id}@{rev[:12]} FP32 on CPU", force=True)
    print(f"[05:{mkey}] Loading {model_id}@{rev[:12]} (FP32, eager, low_cpu_mem_usage=True)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=rev, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=rev, dtype=torch.float32, attn_implementation="eager",
        low_cpu_mem_usage=True, trust_remote_code=False,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hb.beat("SPECTRAL_SVD", 0, 1, note="Computing exact FP64 SVD of lm_head on empty GPU", force=True)
    t_svd0 = time.time()
    basis = compute_spectral_basis_pre_move(model, device, cfg)
    svd_sec = time.time() - t_svd0
    print(
        f"[05:{mkey}] SVD done in {svd_sec:.1f}s ({basis.diagnostics['svd_device_used']}) | "
        f"d={basis.diagnostics['hidden_size']} V={basis.diagnostics['vocab_size']} "
        f"n_dark_99={basis.n_dark_99} ({basis.diagnostics['dark_99_dim_fraction']*100:.1f}%) "
        f"tail38_energy={basis.diagnostics['tail_r_energy_frac']*100:.4f}% tie={basis.diagnostics['tie_word_embeddings_config']}",
        flush=True,
    )
    full_sha = compute_full_weight_sha256_cpu(model)

    hb.beat("MOVE_TO_GPU", 0, 1, note="Moving FP32 model to GPU", force=True)
    model = model.to(device)
    gc.collect()
    torch.cuda.empty_cache()

    V_dark = basis.V_dark.to(device=device, dtype=torch.float32)
    V_dark_np = basis.V_dark.to(torch.float64).cpu().numpy()
    V_leg12_np = basis.V_legacy12.to(torch.float64).cpu().numpy()
    V_bright4_t = basis.V_bright4.to(device=device, dtype=torch.float32)
    gamma32 = basis.rms_gamma.to(device=device, dtype=torch.float32)
    d = int(basis.diagnostics["hidden_size"])
    assert len(model.model.layers) == int(mcfg["num_layers"]), f"Layer count mismatch: {len(model.model.layers)}"
    assert d == int(mcfg["hidden_size"]), f"Hidden size mismatch: {d}"

    pilot_recs = load_jsonl(ROOT / "data" / "dev_pilot_16.jsonl")
    p0_4 = [next(x for x in pilot_recs if x["family"] == fm) for fm in cfg["data"]["id_families"]]
    option_token_ids, opt_meta = resolve_option_token_ids_v05(model, tokenizer, p0_4, device, tuple(cfg["data"]["options"]))
    print(f"[05:{mkey}] Option tokens ({opt_meta['chosen_convention']}): {opt_meta['chosen_ids']} decoded={opt_meta['chosen_decoded']} mass={opt_meta['candidate_mean_prob_mass_on_pilot4'][opt_meta['chosen_convention']]:.4f}", flush=True)

    model_env = {
        "experiment_id": "BAYES-DYN-05", "model_key": mkey, "label": mcfg["label"],
        "arch_family": mcfg["arch_family"], "params_b": mcfg["params_b"],
        "timestamp_utc": utc_now(), "model_id": model_id, "model_revision": rev,
        "full_all_params_weight_sha256": full_sha,
        "dtype": "torch.float32", "attn_implementation": "eager", "allow_tf32": False,
        "num_layers": len(model.model.layers), "hidden_size": d, "probe_dim_r": r,
        "source_layer": src_l, "mid_layer": mid_l, "target_layer": tgt_l,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "option_token_resolution": opt_meta,
        "spectral_diagnostics": basis.diagnostics,
    }
    save_json(out_dir / "configs" / "model_environment.json", model_env)

    # =========================================================================
    # PHASE 1: P0 NUMERICAL & ISOLATION VERIFICATION
    # =========================================================================
    hb.beat("P0_CHECKS", 0, 4, note="Running P0 numerical & isolation checks", force=True)
    p0_rep = run_p0_checks_v05(model, tokenizer, basis, V_dark_np, p0_4, option_token_ids, mcfg, cfg)
    save_json(out_dir / "p0_checks.json", p0_rep)
    print(
        f"[05:{mkey} P0] passed={p0_rep['all_p0_passed']} | hook={p0_rep['zero_hook_max_logit_diff']:.2e} | "
        f"zeroD={p0_rep['zero_pert_matched_batch1']['max_D_geom']:.1e} zeroDF={p0_rep['zero_pert_matched_batch1']['max_abs_dF']:.1e} | "
        f"FD={p0_rep['jvp_fd_curve']['best_median_rel_err']:.2e} | adj={p0_rep['adjoint_consistency_max_rel_err']:.2e} | "
        f"lin={p0_rep['jvp_linear_comb_max_rel_err']:.2e} | zmid={p0_rep['mid_layer_chain_max_rel_err']:.2e}",
        flush=True,
    )
    assert p0_rep["all_p0_passed"], f"P0 FAILED on {mkey}: {p0_rep}"

    split_access_utc: dict[str, str] = {"dev_pilot_p0": utc_now()}

    # =========================================================================
    # PHASE 2: LIVE DEV COLLECTION (DEV-fit 96 + DEV-tune 16, exact 38-D JVP)
    # =========================================================================
    dev_fit_recs = load_jsonl(ROOT / "data" / "dev_fit_96.jsonl")
    dev_tune_recs = load_jsonl(ROOT / "data" / "dev_tune_16.jsonl")
    split_access_utc["dev_fit_96"] = utc_now()
    t_dev0 = time.time()

    def _collect_full38(recs: list[dict[str, Any]], tag: str) -> tuple[np.ndarray, np.ndarray]:
        n = len(recs)
        phis = np.zeros((n, r + 12 + 3), dtype=np.float64)
        As = np.zeros((n, d, r), dtype=np.float32)
        for i, rec in enumerate(recs):
            tok = tokenize_chat_prompt_v05(tokenizer, rec["user_prompt"], device)
            states, _ = capture_residual_states(model, tok["input_ids"], tok["attention_mask"])
            t = tok["t_last"]
            h_s = states[src_l]
            phis[i] = extract_context_features(h_s[0, t, :].detach().to(torch.float64).cpu().numpy(), V_dark_np, V_leg12_np, int(h_s.shape[1]))
            Z38 = compute_exact_jvp_columns_batched(model, h_s, src_l, tgt_l, t, V_dark, batch_size=dev_jvp_bs).detach().to(torch.float64).cpu().numpy()
            As[i] = (Z38 - V_dark_np).astype(np.float32)
            if (i + 1) % 8 == 0 or i + 1 == n:
                hb.beat(f"DEV_{tag}", i + 1, n, note=f"{tag} {i+1}/{n} ({(time.time()-t_dev0):.0f}s)")
        return phis, As

    hb.beat("DEV_FIT96", 0, len(dev_fit_recs), note="Collecting exact 38-D JVP on DEV-fit 96", force=True)
    phi_fit, A_fit = _collect_full38(dev_fit_recs, "FIT96")
    split_access_utc["dev_tune_16"] = utc_now()
    hb.beat("DEV_TUNE16", 0, len(dev_tune_recs), note="Collecting exact 38-D JVP on DEV-tune 16", force=True)
    phi_tune, A_tune = _collect_full38(dev_tune_recs, "TUNE16")
    dev_sec = time.time() - t_dev0
    print(f"[05:{mkey} DEV] Collected 112 exact 38-D Jacobians in {dev_sec:.1f}s ({dev_sec/112.0:.2f}s/ctx). Fitting prior...", flush=True)

    hb.beat("FIT_PRIOR", 0, 1, note="Fitting & selecting full d x 38 prior on DEV", force=True)
    tune_banks_3 = np.zeros((3, len(dev_tune_recs), r, r), dtype=np.float64)
    for b_i in range(3):
        for c_i, rec in enumerate(dev_tune_recs):
            tune_banks_3[b_i, c_i] = generate_fixed_38x38_bank(rec["context_id"], bank_seed + 100 * (b_i + 1), r=r)

    prior, dev_sel_rep = fit_and_select_full_operator_prior_on_dev(phi_fit, A_fit, phi_tune, A_tune, tune_banks_3, cfg)
    U_dyn64 = build_top64_dynamic_output_basis(A_fit, rank=64)
    save_full_operator_prior_npz(out_dir / "prior_model.npz", prior, V_dark_np, U_dyn64)
    save_json(out_dir / "configs" / "dev_selection.json", dev_sel_rep)
    sel = dev_sel_rep["selected"]
    print(
        f"[05:{mkey} PRIOR] Selected alpha={sel['alpha']}, eta={sel['eta']} | s_A^2={prior.s_A_sq:.3f} | "
        f"DEV-tune L0={sel['tune_L0_perp_norm']:.4f} -> L8={sel['tune_L8_perp_norm']:.4f} "
        f"(-{sel['tune_rel_reduction_k8_vs_k0']*100:.2f}%)",
        flush=True,
    )
    del A_fit, A_tune, phi_fit, phi_tune
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # PHASE 3: DEV-PILOT 16 (FREEZE TAU) & CAL 24 (FIT SCALAR TEMPERATURE t)
    # =========================================================================
    hb.beat("DEV_PILOT", 0, len(pilot_recs), note="Freezing tau on DEV-PILOT 16", force=True)
    t_p0 = time.time()
    pilot_cache = []
    for i, rec in enumerate(pilot_recs):
        tok = tokenize_chat_prompt_v05(tokenizer, rec["user_prompt"], device)
        states, logits = capture_residual_states(model, tok["input_ids"], tok["attention_mask"])
        t = tok["t_last"]
        h_s, h_L = states[src_l], states[tgt_l][0, t, :]
        h_norm = float(torch.linalg.vector_norm(h_s[0, t, :]).item())
        ro = extract_readouts_v05(model, logits, h_L, gamma32, basis.rms_eps, option_token_ids, rec["target_option"])
        sess_ev = ProbeSessionV05(f"pilot_ev_{rec['context_id']}", "evaluator_truth", rec["context_id"], model, h_s, V_dark, src_l, mid_l, tgt_l, t, max_jvp_dirs=0, allow_vjp=True)
        a0_ex = sess_ev.vjp_eval_reference(ro["F0"]["g"], "F0")
        append_jsonl(out_dir / "observations" / "query_receipts.jsonl", [sess_ev.receipt()])
        pilot_cache.append((rec, h_s, h_L, logits, t, rho_alpha * h_norm, ro, a0_ex))

    tau_cands = [float(x) for x in cfg["control"]["tau_candidates"]]
    tau_frozen = tau_cands[-1]
    working_regime_established = False
    pilot_tau_diag: dict[str, Any] = {}
    for tau_c in tau_cands:
        hits, lin_errs, rows_tau = 0, [], []
        for rec, h_s, h_L, logits, t, rho_c, ro, a0_ex in pilot_cache:
            x_ex, meta_ex = solve_min_norm_ball_action(a0_ex, tau_c, rho_c, zero_tol)
            ev = evaluate_v05_intervention_batch1(
                model, h_s, h_L, logits, src_l, tgt_l, t, V_dark, x_ex, ro, option_token_ids, rec["target_option"], tau_c,
                target_tol=float(cfg["control"]["pilot_target_tol"]),
            )
            true_lin = float(np.dot(a0_ex, x_ex))
            lin_rel = abs(ev["actual_delta_F0"] - true_lin) / max(abs(true_lin), 1e-8)
            hits += int(ev["within_10pct"])
            lin_errs.append(lin_rel)
            rows_tau.append({"context_id": rec["context_id"], "family": rec["family"], "tau": tau_c, "true_lin_dF0": true_lin, "lin_rel_err": lin_rel, **meta_ex, **ev})
        med_lin = float(np.median(lin_errs))
        ok = (hits >= int(cfg["control"]["pilot_min_hits"])) and (med_lin <= float(cfg["control"]["pilot_max_med_lin_err"]))
        pilot_tau_diag[str(tau_c)] = {"tau": tau_c, "hits_10pct": hits, "n": len(pilot_cache), "median_lin_rel_err": med_lin, "passed": ok}
        append_jsonl(out_dir / "interventions" / "dev_pilot_interventions.jsonl", rows_tau)
        print(f"[05:{mkey} PILOT] tau={tau_c}: hits={hits}/{len(pilot_cache)}, med_lin_err={med_lin*100:.2f}%, PASS={ok}", flush=True)
        if ok:
            tau_frozen = tau_c
            working_regime_established = True
            break

    # CAL-24: Fit single global temperature t > 0
    cal_recs = load_jsonl(ROOT / "data" / "cal_24.jsonl")
    split_access_utc["cal_24"] = utc_now()
    hb.beat("CAL_24", 0, len(cal_recs), note="Fitting scalar temperature t on CAL-24", force=True)
    cal_E_k8 = np.zeros((len(cal_recs), 8, d), dtype=np.float64)
    cal_den_k8 = np.zeros((len(cal_recs), 8), dtype=np.float64)
    cal_cids = []
    for i, rec in enumerate(cal_recs):
        cid = rec["context_id"]
        cal_cids.append(cid)
        tok = tokenize_chat_prompt_v05(tokenizer, rec["user_prompt"], device)
        states, _ = capture_residual_states(model, tok["input_ids"], tok["attention_mask"])
        t = tok["t_last"]
        h_s = states[src_l]
        phi_c = extract_context_features(h_s[0, t, :].detach().to(torch.float64).cpu().numpy(), V_dark_np, V_leg12_np, int(h_s.shape[1]))
        M0 = predict_M0(prior, phi_c)
        B = generate_fixed_38x38_bank(cid, bank_seed, r=r)
        X8, Qp = B[:, 0:8], B[:, 16:24]

        sess_a = ProbeSessionV05(f"cal_a_{cid}", "learning_support", cid, model, h_s, V_dark, src_l, mid_l, tgt_l, t, max_jvp_dirs=8, allow_vjp=False, jvp_batch_size=jvp_bs)
        Z8, _ = sess_a.jvp_columns(X8, "cal_sup8")
        Y8 = Z8 - V_dark_np @ X8
        st8 = condition_exact_batch(prior, M0, X8, Y8, cid, [f"{cid}:s{j}" for j in range(8)])
        append_jsonl(out_dir / "observations" / "query_receipts.jsonl", [sess_a.receipt()])

        sess_b = ProbeSessionV05(f"cal_b_{cid}", "evaluator_truth", cid, model, h_s, V_dark, src_l, mid_l, tgt_l, t, max_jvp_dirs=8, allow_vjp=False, jvp_batch_size=jvp_bs)
        Zp, _ = sess_b.jvp_columns(Qp, "cal_qperp8")
        Yp = Zp - V_dark_np @ Qp
        append_jsonl(out_dir / "observations" / "query_receipts.jsonl", [sess_b.receipt()])

        for j in range(8):
            cal_E_k8[i, j] = st8["M_k"] @ Qp[:, j] - Yp[:, j]
            cal_den_k8[i, j] = float(Qp[:, j] @ st8["C_k"] @ Qp[:, j])
        if (i + 1) % 8 == 0:
            hb.beat("CAL_24", i + 1, len(cal_recs), note=f"CAL {i+1}/{len(cal_recs)}")

    cal_fit_rep = fit_cal_temperature(prior, cal_E_k8, cal_den_k8, denom_floor=1e-12)
    temp_t = float(cal_fit_rep["temperature_t"])
    save_json(out_dir / "calibration" / "cal24_temperature_fit.json", cal_fit_rep)

    # WRITE IMMUTABLE frozen_config.json BEFORE ACCESSING CONFIRM!
    code_shas = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted((ROOT / "src").glob("*.py"))}
    code_shas["run_bayes_dyn_05_model.py"] = sha256_file(ROOT / "run_bayes_dyn_05_model.py")
    frozen_cfg = {
        "experiment_id": "BAYES-DYN-05", "model_key": mkey, "frozen_utc": utc_now(),
        "model_id": model_id, "model_revision": rev, "full_all_params_weight_sha256": full_sha,
        "prior_alpha": prior.alpha, "prior_eta": prior.eta, "s_A_sq": prior.s_A_sq,
        "near_zero_norm_thresh": prior.near_zero_norm_thresh,
        "psd_floor_C": prior.psd_floor_C, "psd_floor_R": prior.psd_floor_R, "solver_jitter": prior.solver_jitter,
        "tau_frozen": tau_frozen, "working_regime_established": working_regime_established,
        "radius_alpha": rho_alpha, "pilot_tau_diagnostics": pilot_tau_diag,
        "cal24_temperature_t": temp_t, "cal24_fit_summary": cal_fit_rep,
        "option_token_ids": option_token_ids, "option_token_convention": opt_meta["chosen_convention"],
        "data_file_sha256": manifest["file_sha256"], "code_file_sha256": code_shas,
    }
    save_json(out_dir / "configs" / "frozen_config.json", frozen_cfg)
    print(f"[05:{mkey} FROZEN] tau={tau_frozen} (working_regime={working_regime_established}), CAL24 temp t={temp_t:.4f}", flush=True)

    # =========================================================================
    # PHASE 4: CONFIRM-ID 64 & CONFIRM-OOD 32 (Strict Stage A -> Commit -> Stage B!)
    # =========================================================================
    interv_methods = ["ZERO", "PRIOR", "UPDATE_4", "UPDATE_8", "UPDATE_16", "PATCH_8", "EXACT"]
    traj_store: dict[str, dict[str, Any]] = {}
    all_int_vecs, all_int_cids, all_int_splits = [], [], []

    def _run_confirm_split(split_name: str, recs: list[dict[str, Any]]) -> dict[str, Any]:
        split_access_utc[split_name] = utc_now()
        n_s = len(recs)
        hb.beat(f"CONFIRM_{split_name.upper()}", 0, n_s, note=f"Starting {split_name} (n={n_s})", force=True)
        print(f"[05:{mkey} {split_name.upper()}] Running Stage A -> commit -> Stage B on n={n_s}...", flush=True)

        obs_cids, obs_phis, obs_B, obs_Qg, obs_Ysup16, obs_Yp, obs_Yg, obs_g0, obs_g3, obs_a0ex = [], [], [], [], [], [], [], [], [], []
        post_C = np.zeros((n_s, 4, r, r), dtype=np.float64)
        post_ids_arr: list[list[str]] = []
        ctx_eval_records: list[dict[str, Any]] = []
        int_file = out_dir / "interventions" / f"{split_name}_interventions.jsonl"

        for i, rec in enumerate(recs):
            cid = rec["context_id"]
            fam = rec["family"]
            in_traj = (cid in traj_12_set)

            tok = tokenize_chat_prompt_v05(tokenizer, rec["user_prompt"], device)
            states, logits = capture_residual_states(model, tok["input_ids"], tok["attention_mask"])
            t = tok["t_last"]
            h_s, h_m, h_L = states[src_l], states[mid_l], states[tgt_l][0, t, :]
            h_norm = float(torch.linalg.vector_norm(h_s[0, t, :]).item())
            rho_c = rho_alpha * h_norm
            phi_c = extract_context_features(h_s[0, t, :].detach().to(torch.float64).cpu().numpy(), V_dark_np, V_leg12_np, int(h_s.shape[1]))

            B = generate_fixed_38x38_bank(cid, bank_seed, r=r)
            X_sup_16 = B[:, 0:16]
            Q_perp = B[:, 16:24]
            Q_gen = generate_q_general(cid, qgen_seed, r=r, n_dirs=4)

            # --- STAGE A: TARGET-AGNOSTIC MAP UPDATES ---
            M0 = predict_M0(prior, phi_c)
            st0 = condition_exact_batch(prior, M0, np.zeros((r, 0)), np.zeros((d, 0)), cid, [])
            sess_a = ProbeSessionV05(f"{split_name}_a_{cid}", "learning_support", cid, model, h_s, V_dark, src_l, mid_l, tgt_l, t, max_jvp_dirs=16, allow_vjp=False, jvp_batch_size=jvp_bs)

            Z4, Zm_4 = sess_a.jvp_columns(X_sup_16[:, 0:4], "sup_0_4", record_mid_z=in_traj)
            Y4 = Z4 - V_dark_np @ X_sup_16[:, 0:4]
            st4 = condition_exact_batch(prior, M0, X_sup_16[:, 0:4], Y4, cid, [f"{cid}:s{j}" for j in range(4)])

            Z4_8, Zm_4_8 = sess_a.jvp_columns(X_sup_16[:, 4:8], "sup_4_8", record_mid_z=in_traj)
            Z8 = np.concatenate([Z4, Z4_8], axis=1)
            Y8 = Z8 - V_dark_np @ X_sup_16[:, 0:8]
            st8 = condition_exact_batch(prior, M0, X_sup_16[:, 0:8], Y8, cid, [f"{cid}:s{j}" for j in range(8)])
            M_patch8 = condition_patch_only(M0, X_sup_16[:, 0:8], Y8)
            Y8_mis = make_mispaired_Y(M0, X_sup_16[:, 0:8], Y8, S_mispair)
            st8_mis = condition_exact_batch(prior, M0, X_sup_16[:, 0:8], Y8_mis, cid, [f"{cid}:mis{j}" for j in range(8)])

            Z8_16, Zm_8_16 = sess_a.jvp_columns(X_sup_16[:, 8:16], "sup_8_16", record_mid_z=in_traj)
            Z16 = np.concatenate([Z8, Z8_16], axis=1)
            Y16 = Z16 - V_dark_np @ X_sup_16
            st16 = condition_exact_batch(prior, M0, X_sup_16, Y16, cid, [f"{cid}:s{j}" for j in range(16)])
            append_jsonl(out_dir / "observations" / "query_receipts.jsonl", [sess_a.receipt()])

            sts = [st0, st4, st8, st16]
            for bi, stb in enumerate(sts):
                post_C[i, bi] = stb["C_k"]
            post_ids_arr.append([s["posterior_id"] for s in sts])

            # Freeze maps -> construct readouts -> project & commit actions BEFORE Stage B truth reveal!
            ro = extract_readouts_v05(model, logits, h_L, gamma32, basis.rms_eps, option_token_ids, rec["target_option"])
            g0 = ro["F0"]["g"]
            G_opt3 = np.stack([x["g"] for x in ro["opt3"]], axis=0)

            proj_f0, proj_opt3 = {}, {}
            for lbl, stb in (("PRIOR", st0), ("UPDATE_4", st4), ("UPDATE_8", st8), ("UPDATE_16", st16)):
                proj_f0[lbl] = project_function_from_posterior(prior, V_dark_np, stb["M_k"], stb["C_k"], g0, temp_t=temp_t)
                proj_opt3[lbl] = [project_function_from_posterior(prior, V_dark_np, stb["M_k"], stb["C_k"], G_opt3[oj], temp_t=temp_t) for oj in range(3)]
            proj_f0["PATCH_8"] = {"a_hat": (V_dark_np + M_patch8).T @ g0, "Cov_a": proj_f0["UPDATE_8"]["Cov_a"]}
            proj_f0["MISPAIRED_8"] = project_function_from_posterior(prior, V_dark_np, st8_mis["M_k"], st8_mis["C_k"], g0, temp_t=temp_t)

            actions: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
            actions["ZERO"] = (np.zeros(r, dtype=np.float64), {"is_zero_a": False, "hit_boundary_unreachable": False, "a_hat_norm": 0.0, "step_norm": 0.0, "pred_delta_F0": 0.0, "tau": tau_frozen, "rho_c": rho_c, "pred_delta_F0_std": 0.0})
            for lbl in ("PRIOR", "UPDATE_4", "UPDATE_8", "UPDATE_16", "PATCH_8"):
                xk, mk = solve_min_norm_ball_action(proj_f0[lbl]["a_hat"], tau_frozen, rho_c, zero_tol)
                _, sd_act = predict_scalar_direction_dist(proj_f0[lbl]["a_hat"], proj_f0[lbl]["Cov_a"], xk)
                mk["pred_delta_F0_std"] = sd_act
                actions[lbl] = (xk, mk)

            x_prior = actions["PRIOR"][0]
            ca_prior_mean, ca_prior_std = predict_scalar_direction_dist(proj_f0["PRIOR"]["a_hat"], proj_f0["PRIOR"]["Cov_a"], x_prior)
            ca_upd8_mean, ca_upd8_std = predict_scalar_direction_dist(proj_f0["UPDATE_8"]["a_hat"], proj_f0["UPDATE_8"]["Cov_a"], x_prior)

            pred_commit_row = {
                "split": split_name, "context_id": cid, "family": fam, "commit_utc": utc_now(),
                "posterior_ids": {f"k{kb}": sts[bi]["posterior_id"] for bi, kb in enumerate((0, 4, 8, 16))},
                "M_k_f32_sha256": {f"k{kb}": bytes_sha256_short(sts[bi]["M_k"].astype(np.float32)) for bi, kb in enumerate((0, 4, 8, 16))},
                "action_x_f64_sha256": {m: bytes_sha256_short(actions[m][0]) for m in ("ZERO", "PRIOR", "UPDATE_4", "UPDATE_8", "UPDATE_16", "PATCH_8")},
                "common_action_x0_pred": {"PRIOR_mean": ca_prior_mean, "PRIOR_std": ca_prior_std, "UPDATE_8_mean": ca_upd8_mean, "UPDATE_8_std": ca_upd8_std},
                "pred_qperp_f0_k0": [float(x) for x in (proj_f0["PRIOR"]["a_hat"] @ Q_perp)],
                "pred_qperp_f0_k8": [float(x) for x in (proj_f0["UPDATE_8"]["a_hat"] @ Q_perp)],
            }
            append_jsonl(out_dir / "predictions.jsonl", [pred_commit_row])

            # --- STAGE B: REVEAL Q_perp (8), Q_general (4), EXACT F0 VJP & REAL FORWARDS ---
            sess_b = ProbeSessionV05(f"{split_name}_b_{cid}", "evaluator_truth", cid, model, h_s, V_dark, src_l, mid_l, tgt_l, t, max_jvp_dirs=12, allow_vjp=True, jvp_batch_size=jvp_bs)
            Zp_true, Zm_p = sess_b.jvp_columns(Q_perp, "eval_qperp8", record_mid_z=in_traj)
            Zg_true, _ = sess_b.jvp_columns(Q_gen, "eval_qgen4", record_mid_z=False)
            Yp_true = Zp_true - V_dark_np @ Q_perp
            Yg_true = Zg_true - V_dark_np @ Q_gen
            a0_exact = sess_b.vjp_eval_reference(g0, "F0")
            append_jsonl(out_dir / "observations" / "query_receipts.jsonl", [sess_b.receipt()])

            x_ex, m_ex = solve_min_norm_ball_action(a0_exact, tau_frozen, rho_c, zero_tol)
            m_ex["pred_delta_F0_std"] = 0.0
            actions["EXACT"] = (x_ex, m_ex)

            eval_map_dict: dict[str, Any] = {}
            map_candidates = [
                ("PRIOR", st0["M_k"], st0["C_k"], st0["posterior_id"], 0),
                ("UPDATE_4", st4["M_k"], st4["C_k"], st4["posterior_id"], 4),
                ("UPDATE_8", st8["M_k"], st8["C_k"], st8["posterior_id"], 8),
                ("UPDATE_16", st16["M_k"], st16["C_k"], st16["posterior_id"], 16),
                ("PATCH_8", M_patch8, st8["C_k"], f"{cid}:patch8", 8),
                ("MISPAIRED_8", st8_mis["M_k"], st8_mis["C_k"], st8_mis["posterior_id"], 8),
            ]
            f0_true_qp = Zp_true.T @ g0
            f0_true_qg = Zg_true.T @ g0
            opt3_true_qp = G_opt3 @ Zp_true

            for lbl, Mk, Ck, pid, kb in map_candidates:
                diff_p = Mk @ Q_perp - Yp_true
                sq_p_each = np.sum(diff_p ** 2, axis=0)
                L_perp_norm = float(np.mean(sq_p_each) / prior.s_A_sq)
                proj_u = U_dyn64.T @ diff_p
                in_u_sq = float(np.mean(np.sum(proj_u ** 2, axis=0)))
                out_u_sq = float(np.mean(sq_p_each)) - in_u_sq

                diff_g = Mk @ Q_gen - Yg_true
                L_gen_norm = float(np.mean(np.sum(diff_g ** 2, axis=0)) / prior.s_A_sq)

                a_f0_hat = proj_f0[lbl]["a_hat"]
                f0_pred_qp = a_f0_hat @ Q_perp
                f0_pred_qg = a_f0_hat @ Q_gen
                f0_mse_qp = float(np.mean((f0_pred_qp - f0_true_qp) ** 2))
                f0_mse_qg = float(np.mean((f0_pred_qg - f0_true_qg) ** 2))
                f0_sign_acc_qp = float(np.mean(np.sign(f0_pred_qp) == np.sign(f0_true_qp)))

                opt3_mse_qp = None
                if lbl in proj_opt3:
                    o3_preds = np.stack([proj_opt3[lbl][oj]["a_hat"] @ Q_perp for oj in range(3)], axis=0)
                    opt3_mse_qp = float(np.mean((o3_preds - opt3_true_qp) ** 2))

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
                    "L_perp_in_Udyn64_sq": in_u_sq, "L_perp_out_Udyn64_sq": out_u_sq,
                    "L_gen_norm": L_gen_norm,
                    "f0_mse_qperp": f0_mse_qp, "f0_mse_qgen": f0_mse_qg, "f0_sign_acc_qperp": f0_sign_acc_qp,
                    "opt3_mse_qperp": opt3_mse_qp,
                    "f0_qperp_cov": {str(int(lv * 100)): float(np.mean(v)) for lv, v in cov_hits.items()},
                    "f0_qperp_mean_width_90": float(np.mean(widths_90)),
                    "f0_qperp_mean_nll": float(np.mean(nlls)),
                }

            int_rows_ctx, int_by_m = [], {}
            for m_nm in interv_methods:
                xm, sm = actions[m_nm]
                ev = evaluate_v05_intervention_batch1(
                    model, h_s, h_L, logits, src_l, tgt_l, t, V_dark, xm, ro, option_token_ids, rec["target_option"], tau_frozen,
                    target_tol=float(cfg["control"]["pilot_target_tol"]),
                )
                true_lin_f0 = float(np.dot(a0_exact, xm))
                pred_f0 = float(sm["pred_delta_F0"])
                sd_f0 = float(sm.get("pred_delta_F0_std", 0.0))
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

            ca_true_lin = float(np.dot(a0_exact, x_prior))
            common_act_eval = {
                "true_lin_F0_at_x0": ca_true_lin,
                "PRIOR_abs_err_lin": abs(ca_prior_mean - ca_true_lin),
                "UPDATE_8_abs_err_lin": abs(ca_upd8_mean - ca_true_lin),
            }

            ctx_eval_records.append({
                "context_id": cid, "family": fam, "map_eval": eval_map_dict,
                "interventions": int_by_m, "common_action_x0": common_act_eval,
                "support_interp_rel_k8": st8["support_residual_rel"],
            })

            obs_cids.append(cid); obs_phis.append(phi_c); obs_B.append(B); obs_Qg.append(Q_gen)
            obs_Ysup16.append(Y16.astype(np.float32)); obs_Yp.append(Yp_true.astype(np.float32))
            obs_Yg.append(Yg_true.astype(np.float32)); obs_g0.append(g0); obs_g3.append(G_opt3); obs_a0ex.append(a0_exact)
            all_int_cids.append(cid); all_int_splits.append(split_name)
            all_int_vecs.append(np.stack([actions[m][0] for m in interv_methods], axis=0).astype(np.float64))

            if in_traj:
                Zm_sup16 = np.concatenate([Zm_4, Zm_4_8, Zm_8_16], axis=1)
                traj_store[cid] = {
                    "rec": rec, "t_last": t, "family": fam,
                    "h_src_seq": h_s.detach().cpu(), "h_mid_seq": h_m.detach().cpu(),
                    "B_38x38": B, "g0": g0,
                    "Zm_0_24": np.concatenate([Zm_sup16, Zm_p], axis=1),
                    "Zt_0_24": np.concatenate([Z16, Zp_true], axis=1),
                    "M_by_k": {0: st0["M_k"], 4: st4["M_k"], 8: st8["M_k"], 16: st16["M_k"]},
                }

            if (i + 1) % 8 == 0 or i + 1 == n_s:
                hb.beat(f"CONFIRM_{split_name.upper()}", i + 1, n_s, note=f"{split_name} {i+1}/{n_s}")

        np.savez_compressed(
            out_dir / "observations" / f"{split_name}_obs.npz",
            context_ids=np.array(obs_cids, dtype=str), phi=np.stack(obs_phis), B_38x38=np.stack(obs_B),
            Q_general=np.stack(obs_Qg), Y_sup_16=np.stack(obs_Ysup16), Y_perp_true=np.stack(obs_Yp),
            Y_gen_true=np.stack(obs_Yg), g0=np.stack(obs_g0), G_opt3=np.stack(obs_g3), a0_exact=np.stack(obs_a0ex),
        )
        np.savez_compressed(
            out_dir / "posteriors" / f"{split_name}_C_k.npz",
            context_ids=np.array(obs_cids, dtype=str), budgets=np.array([0, 4, 8, 16]),
            C_k=post_C, posterior_ids=np.array(post_ids_arr, dtype=str),
        )

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
                "L_perp_in_Udyn64_sq_mean": float(np.mean([x["map_eval"][lbl]["L_perp_in_Udyn64_sq"] for x in ctx_eval_records])),
                "L_perp_out_Udyn64_sq_mean": float(np.mean([x["map_eval"][lbl]["L_perp_out_Udyn64_sq"] for x in ctx_eval_records])),
                "L_gen_norm": family_stratified_paired_bootstrap(lg, None, fams_s, n_boot, b_seed),
                "f0_mse_qperp": family_stratified_paired_bootstrap(f0m, None, fams_s, n_boot, b_seed),
                "f0_sign_acc_qperp_mean": float(np.mean([x["map_eval"][lbl]["f0_sign_acc_qperp"] for x in ctx_eval_records])),
                "opt3_mse_qperp_mean": float(np.mean(o3_vals)) if o3_vals else None,
                "f0_qperp_cov_mean": {lv: float(np.mean([x["map_eval"][lbl]["f0_qperp_cov"][lv] for x in ctx_eval_records])) for lv in ("50", "80", "90", "95")},
                "f0_qperp_mean_width_90": float(np.mean([x["map_eval"][lbl]["f0_qperp_mean_width_90"] for x in ctx_eval_records])),
                "f0_qperp_mean_nll": float(np.mean([x["map_eval"][lbl]["f0_qperp_mean_nll"] for x in ctx_eval_records])),
            }

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

        ep8 = np.array([x["interventions"]["PATCH_8"]["target_tracking_err"] for x in ctx_eval_records])
        eu8 = np.array([x["interventions"]["UPDATE_8"]["target_tracking_err"] for x in ctx_eval_records])
        paired_links["PATCH_8_minus_UPDATE_8_ctrl_err"] = family_stratified_paired_bootstrap(ep8, eu8, fams_s, n_boot, b_seed)

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
            }

        ca_p = np.array([x["common_action_x0"]["PRIOR_abs_err_lin"] for x in ctx_eval_records])
        ca_u = np.array([x["common_action_x0"]["UPDATE_8_abs_err_lin"] for x in ctx_eval_records])
        common_act_sum = {
            "PRIOR_mean_abs_err_lin": float(np.mean(ca_p)),
            "UPDATE_8_mean_abs_err_lin": float(np.mean(ca_u)),
            "PRIOR_minus_UPDATE_8_abs_err_lin": family_stratified_paired_bootstrap(ca_p, ca_u, fams_s, n_boot, b_seed),
        }

        return {
            "n_contexts": n_s, "map_summary": map_summary, "paired_links": paired_links,
            "family_breakdown_L0_minus_L8": fam_breakdown, "interventions": int_summary,
            "common_action_x0_diagnostic": common_act_sum,
            "mean_support_interp_rel_k8": float(np.mean([x["support_interp_rel_k8"] for x in ctx_eval_records])),
        }

    confirm_id_sum = _run_confirm_split("confirm_id", load_jsonl(ROOT / "data" / "confirm_id_64.jsonl"))
    confirm_ood_sum = _run_confirm_split("confirm_ood", load_jsonl(ROOT / "data" / "confirm_ood_32.jsonl"))

    np.savez_compressed(
        out_dir / "interventions" / "intervention_vectors.npz",
        context_ids=np.array(all_int_cids, dtype=str), splits=np.array(all_int_splits, dtype=str),
        methods=np.array(interv_methods, dtype=str), X_vectors=np.stack(all_int_vecs, axis=0),
    )

    l1 = confirm_id_sum["paired_links"]["PRIOR_minus_UPDATE_8_L_perp"]
    l2 = confirm_id_sum["paired_links"]["PRIOR_minus_UPDATE_8_f0_mse_qperp"]
    l3 = confirm_id_sum["paired_links"]["PRIOR_minus_UPDATE_8_ctrl_err"]
    print(f"[05:{mkey} ID64] Link1 L_perp: {l1['mean']:+.5f} {l1['ci95']} (-{l1['rel_reduction']*100:.2f}%, PASS={l1['ci95_strictly_positive']})", flush=True)
    print(f"[05:{mkey} ID64] Link2 F0 MSE: {l2['mean']:+.6f} {l2['ci95']} (-{l2['rel_reduction']*100:.2f}%, PASS={l2['ci95_strictly_positive']})", flush=True)
    print(f"[05:{mkey} ID64] Link3 CtrlEr: {l3['mean']:+.5f} {l3['ci95']} (-{l3['rel_reduction']*100:.2f}%, PASS={l3['ci95_strictly_positive']})", flush=True)

    # =========================================================================
    # PHASE 5: 12-CONTEXT INTER-LAYER TRAJECTORY PANEL + DIAG 1 & DIAG 2
    # =========================================================================
    hb.beat("TRAJECTORY_12", 0, len(traj_12_order), note="Completing 12-ctx exact 38-D trajectory panel", force=True)
    P_U = U_dyn64 @ U_dyn64.T
    I_m_PU = np.eye(d) - P_U
    P_D = V_dark_np @ V_dark_np.T
    I_m_PD = np.eye(d) - P_D

    four_blocks_by_k: dict[str, list[dict[str, float]]] = {f"k{kb}": [] for kb in (0, 4, 8, 16)}
    inter_layer_stats: list[dict[str, float]] = []
    mech_diag1_rows: list[dict[str, Any]] = []
    mech_diag2_rows: list[dict[str, Any]] = []
    traj_M_archive = np.zeros((len(traj_12_order), 4, d, r), dtype=np.float32)

    run_opt = ((time.time() - t_start) / 60.0) < float(cfg["budget"]["optional_diag_cutoff_min"])
    rng_rand = np.random.default_rng(int(cfg["derived_seeds"]["random_subspace_diag"]))

    for ti, cid in enumerate(traj_12_order):
        ts = traj_store[cid]
        for bi, kb in enumerate((0, 4, 8, 16)):
            traj_M_archive[ti, bi] = ts["M_by_k"][kb].astype(np.float32)
        h_s = ts["h_src_seq"].to(device)
        h_m = ts["h_mid_seq"].to(device)
        t = ts["t_last"]
        B = ts["B_38x38"]
        B_rem_14 = B[:, 24:38]

        sess_t = ProbeSessionV05(f"traj_comp_{cid}", "evaluator_truth", cid, model, h_s, V_dark, src_l, mid_l, tgt_l, t, max_jvp_dirs=14, allow_vjp=False, jvp_batch_size=jvp_bs)
        Zt_rem, Zm_rem = sess_t.jvp_columns(B_rem_14, "traj_rem14", record_mid_z=True)
        append_jsonl(out_dir / "observations" / "query_receipts.jsonl", [sess_t.receipt()])

        Zm_bank = np.concatenate([ts["Zm_0_24"], Zm_rem], axis=1)
        Zt_bank = np.concatenate([ts["Zt_0_24"], Zt_rem], axis=1)
        Zt_full = Zt_bank @ B.T
        A_full = Zt_full - V_dark_np

        for kb in (0, 4, 8, 16):
            X_k = B[:, :kb]
            P_X = X_k @ X_k.T if kb > 0 else np.zeros((r, r))
            I_m_PX = np.eye(r) - P_X
            E = A_full - ts["M_by_k"][kb]
            four_blocks_by_k[f"k{kb}"].append({
                "inU_obsX": float(np.sum((P_U @ E @ P_X) ** 2)),
                "inU_unobsX": float(np.sum((P_U @ E @ I_m_PX) ** 2)),
                "outU_obsX": float(np.sum((I_m_PU @ E @ P_X) ** 2)),
                "outU_unobsX": float(np.sum((I_m_PU @ E @ I_m_PX) ** 2)),
                "total_frob_sq": float(np.sum(E ** 2)),
            })

        V_qp = V_dark_np @ B[:, 16:24]
        zm_qp = Zm_bank[:, 16:24]
        zt_qp = Zt_bank[:, 16:24]
        g0 = ts["g0"]
        inter_layer_stats.append({
            "mean_norm_v_src": float(np.mean(np.linalg.norm(V_qp, axis=0))),
            "mean_norm_z_mid": float(np.mean(np.linalg.norm(zm_qp, axis=0))),
            "mean_norm_z_tgt": float(np.mean(np.linalg.norm(zt_qp, axis=0))),
            "mean_norm_delta_src_mid": float(np.mean(np.linalg.norm(zm_qp - V_qp, axis=0))),
            "mean_norm_delta_mid_tgt": float(np.mean(np.linalg.norm(zt_qp - zm_qp, axis=0))),
            "z_mid_dark_energy_frac": float(np.sum((V_dark_np.T @ zm_qp) ** 2) / max(float(np.sum(zm_qp ** 2)), 1e-12)),
            "z_tgt_dark_energy_frac": float(np.sum((V_dark_np.T @ zt_qp) ** 2) / max(float(np.sum(zt_qp ** 2)), 1e-12)),
            "mean_abs_g_dot_z_mid": float(np.mean(np.abs(g0 @ zm_qp))),
            "mean_abs_g_dot_z_tgt": float(np.mean(np.abs(g0 @ zt_qp))),
        })

        if run_opt:
            zm_0, zt_0 = zm_qp[:, 0], zt_qp[:, 0]
            zm_dark, zm_comp = P_D @ zm_0, I_m_PD @ zm_0
            pair_t = torch.tensor(np.stack([zm_dark, zm_comp], axis=1), dtype=torch.float32, device=device)
            out_pair = compute_exact_jvp_columns_batched(model, h_m, mid_l, tgt_l, t, pair_t, batch_size=2).to(torch.float64).cpu().numpy()
            zt_from_dark, zt_from_comp = out_pair[:, 0], out_pair[:, 1]
            sum_rel_err = float(np.linalg.norm((zt_from_dark + zt_from_comp) - zt_0) / max(float(np.linalg.norm(zt_0)), 1e-12))
            mech_diag1_rows.append({
                "context_id": cid, "family": ts["family"],
                "z_mid_dark_norm": float(np.linalg.norm(zm_dark)), "z_mid_comp_norm": float(np.linalg.norm(zm_comp)),
                "z_tgt_from_dark_norm": float(np.linalg.norm(zt_from_dark)), "z_tgt_from_comp_norm": float(np.linalg.norm(zt_from_comp)),
                "g_dot_z_tgt_from_dark": float(np.dot(g0, zt_from_dark)), "g_dot_z_tgt_from_comp": float(np.dot(g0, zt_from_comp)),
                "g_dot_z_tgt_total": float(np.dot(g0, zt_0)), "exact_sum_rel_err": sum_rel_err,
            })

            zt_dark4 = Zt_bank[:, 0:4]
            zt_bright4 = compute_exact_jvp_columns_batched(model, h_s, src_l, tgt_l, t, V_bright4_t, batch_size=min(4, jvp_bs)).to(torch.float64).cpu().numpy()
            Q_ro, _ = np.linalg.qr(rng_rand.normal(size=(d, 4)))
            zt_rand4 = compute_exact_jvp_columns_batched(model, h_s, src_l, tgt_l, t, torch.tensor(Q_ro, dtype=torch.float32, device=device), batch_size=min(4, jvp_bs)).to(torch.float64).cpu().numpy()
            mech_diag2_rows.append({
                "context_id": cid, "family": ts["family"],
                "dark4_mean_z_tgt_norm": float(np.mean(np.linalg.norm(zt_dark4, axis=0))),
                "bright4_mean_z_tgt_norm": float(np.mean(np.linalg.norm(zt_bright4, axis=0))),
                "rand4_mean_z_tgt_norm": float(np.mean(np.linalg.norm(zt_rand4, axis=0))),
                "dark4_mean_abs_g_dot_z_tgt": float(np.mean(np.abs(g0 @ zt_dark4))),
                "bright4_mean_abs_g_dot_z_tgt": float(np.mean(np.abs(g0 @ zt_bright4))),
                "rand4_mean_abs_g_dot_z_tgt": float(np.mean(np.abs(g0 @ zt_rand4))),
            })

        if (ti + 1) % 4 == 0 or ti + 1 == len(traj_12_order):
            hb.beat("TRAJECTORY_12", ti + 1, len(traj_12_order), note=f"Trajectory {ti+1}/{len(traj_12_order)}")

    np.savez_compressed(
        out_dir / "posteriors" / "traj12_M_k.npz",
        context_ids=np.array(traj_12_order, dtype=str), budgets=np.array([0, 4, 8, 16]), M_k=traj_M_archive,
    )
    four_blocks_summary = {
        kb: {k_f: float(np.mean([r[k_f] for r in rows])) for k_f in ("inU_obsX", "inU_unobsX", "outU_obsX", "outU_unobsX", "total_frob_sq")}
        for kb, rows in four_blocks_by_k.items()
    }
    traj_summary = {
        "n_contexts": len(traj_12_order), "context_ids": traj_12_order,
        "layers": {"src": src_l, "mid": mid_l, "tgt": tgt_l},
        "four_block_exact_38D_errors": four_blocks_summary,
        "inter_layer_qperp_summary": {k_f: float(np.mean([r[k_f] for r in inter_layer_stats])) for k_f in inter_layer_stats[0]},
        "optional_diagnostics_executed": run_opt,
    }
    save_json(out_dir / "trajectory_panel" / "trajectory_12_summary.json", traj_summary)
    if run_opt:
        mech_sum = {
            "diag1_zmid_split_additivity_max_rel_err": float(max(r["exact_sum_rel_err"] for r in mech_diag1_rows)),
            "diag1_mean_ztgt_from_dark_norm": float(np.mean([r["z_tgt_from_dark_norm"] for r in mech_diag1_rows])),
            "diag1_mean_ztgt_from_comp_norm": float(np.mean([r["z_tgt_from_comp_norm"] for r in mech_diag1_rows])),
            "diag1_mean_abs_g_from_dark": float(np.mean([abs(r["g_dot_z_tgt_from_dark"]) for r in mech_diag1_rows])),
            "diag1_mean_abs_g_from_comp": float(np.mean([abs(r["g_dot_z_tgt_from_comp"]) for r in mech_diag1_rows])),
            "diag1_comp_to_dark_func_ratio": float(
                np.mean([abs(r["g_dot_z_tgt_from_comp"]) for r in mech_diag1_rows])
                / max(float(np.mean([abs(r["g_dot_z_tgt_from_dark"]) for r in mech_diag1_rows])), 1e-12)
            ),
            "diag1_per_context": mech_diag1_rows,
            "diag2_mean_ztgt_norm": {
                "dark4": float(np.mean([r["dark4_mean_z_tgt_norm"] for r in mech_diag2_rows])),
                "bright4": float(np.mean([r["bright4_mean_z_tgt_norm"] for r in mech_diag2_rows])),
                "rand4": float(np.mean([r["rand4_mean_z_tgt_norm"] for r in mech_diag2_rows])),
            },
            "diag2_mean_abs_g_dot_ztgt": {
                "dark4": float(np.mean([r["dark4_mean_abs_g_dot_z_tgt"] for r in mech_diag2_rows])),
                "bright4": float(np.mean([r["bright4_mean_abs_g_dot_z_tgt"] for r in mech_diag2_rows])),
                "rand4": float(np.mean([r["rand4_mean_abs_g_dot_z_tgt"] for r in mech_diag2_rows])),
            },
            "diag2_per_context": mech_diag2_rows,
        }
        save_json(out_dir / "trajectory_panel" / "trajectory_12_mechanisms.json", mech_sum)

    elapsed_sec = time.time() - t_start
    summary = {
        "experiment_id": "BAYES-DYN-05", "model_key": mkey, "label": mcfg["label"],
        "arch_family": mcfg["arch_family"], "params_b": mcfg["params_b"],
        "status": "COMPLETED", "started_utc": utc_start, "finished_utc": utc_now(),
        "elapsed_sec": round(elapsed_sec, 2), "elapsed_gpu_hours": round(elapsed_sec / 3600.0, 4),
        "dev_collection_sec": round(dev_sec, 2), "svd_sec": round(svd_sec, 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1),
        "split_first_access_utc": split_access_utc,
        "tau_frozen": tau_frozen, "working_regime_established": working_regime_established,
        "cal24_temperature_t": temp_t,
        "confirm_id": confirm_id_sum, "confirm_ood": confirm_ood_sum,
        "trajectory_panel_12": traj_summary,
    }
    save_json(out_dir / "metrics_summary.json", summary)
    hb.beat("DONE", 1, 1, status="COMPLETED", note=f"Finished {mkey} cleanly in {elapsed_sec:.1f}s", force=True)
    print("=" * 78)
    print(f"[05:{mkey} COMPLETED] Elapsed: {elapsed_sec:.1f}s ({elapsed_sec/3600.0:.4f} GPU-hr) | Peak VRAM: {summary['peak_vram_mb']} MB")
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
