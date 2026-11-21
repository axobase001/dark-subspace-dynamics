# BAYES-DYN-05 — Cross-Scale (0.5B–8B) & Cross-Architecture Replication of Evidence-Updated Dark-Subspace Dynamics Maps

> **Protocol Parity**: Every model ran in **pure FP32 + eager attention (TF32 disabled)**, uniform `r = 38` darkest unembedding singular vectors, deterministic late-layer segment `round(2L/3) -> L`, identical zero-overlap prompt splits (`DEV-fit 96`, `DEV-tune 16`, `DEV-PILOT 16`, `CAL 24`, `Confirm-ID 64`, `Confirm-OOD 32`, `Traj 12`), strict Stage-A pre-commit / evaluator isolation, and independent pure-NumPy CPU auditing.

## 1. Model Matrix, Spectral Geometry & Audit Status

| Model | Architecture | Params | Layers (`src→mid→tgt`) | `d` | Tied? | `n_dark(99%)` | Bottom-38 Energy | P0 | CPU Audit | GPU Time | Est. CU |
|---|---|---:|---|---:|---|---:|---:|---|---|---:|---:|
| `Qwen2.5-0.5B-Instruct (Ref)` | Qwen2 (Pre-RMSNorm, tied) | 0.49B | `h16->h20->h24` | 896 | True | 38 (4.2%) | 0.990% | PASS | PASS (0 viol) | 2.5m | 0.16 |
| `Qwen2.5-3B-Instruct` | Qwen2 (Pre-RMSNorm, GQA, tied) | 3.09B | `h24->h30->h36` | 2048 | True | 61 (3.0%) | 0.524% | PASS | PASS (0 viol) | 12.0m | 0.46 |
| `Phi-3.5-mini-instruct` | Phi-3 (Pre-RMSNorm, fused QKV/MLP, MHA, untied) | 3.82B | `h21->h26->h32` | 3072 | False | 114 (3.7%) | 0.254% | PASS | PASS (0 viol) | 17.1m | 0.59 |
| `Qwen2.5-7B-Instruct` | Qwen2 (Pre-RMSNorm, GQA, untied) | 7.62B | `h19->h23->h28` | 3584 | False | 145 (4.0%) | 0.125% | PASS | PASS (0 viol) | 29.2m | 1.03 |
| `OLMo-2-1124-7B-Instruct` | OLMo-2 (Post-RMSNorm + QK-Norm, untied) | 7.30B | `h21->h26->h32` | 4096 | False | 174 (4.2%) | 0.168% | PASS | PASS (0 viol) | 32.6m | 1.13 |
| `Llama-3.1-8B-Instruct` | Llama-3.1 (Pre-RMSNorm, GQA, untied) | 8.03B | `h21->h26->h32` | 4096 | False | 134 (3.3%) | 0.192% | PASS | PASS (0 viol) | 34.6m | 1.17 |

## 2. Core 3-Link Verification — Confirm-ID (Primary, `k=8` vs `PRIOR`, 20,000 Family-Stratified Paired Bootstrap)

| Model | Link 1: Unobs `Q_perp` Drop (`k=8` / `k=16`) | `MISPAIRED_8` (`Q_perp`) | Link 2: Frozen `F0` MSE Drop (`k=8` / `k=16`) | Link 3: Real Tracking Err (`PRIOR → U8 → U16`, `k=8` Drop) | `PATCH_8 → UPDATE_8` Net Gain (`95% CI`) | `EXACT` Tracking Err |
|---|---:|---:|---:|---:|---:|---:|
| `Qwen2.5-0.5B-Instruct (Ref)` | **-41.4%** / -49.5% (PASS) | +22.2% worse | **-15.0%** / -28.7% (PASS) | 14.2% → **10.1%** → 6.6% (**-29.1%**, PASS) | +1.55 [+0.51, +2.66] pp (PASS) | 0.40% |
| `Qwen2.5-3B-Instruct` | **-15.1%** / -24.2% (PASS) | +13.4% worse | **-21.5%** / -31.5% (PASS) | 54.4% → **23.8%** → 10.3% (**-56.2%**, PASS) | +9.95 [+5.67, +14.37] pp (PASS) | 0.12% |
| `Phi-3.5-mini-instruct` | **-22.2%** / -34.5% (PASS) | +19.6% worse | **-28.1%** / -48.8% (PASS) | 14.8% → **8.7%** → 5.4% (**-41.3%**, PASS) | +2.51 [+0.81, +4.30] pp (PASS) | 0.23% |
| `Qwen2.5-7B-Instruct` | **-18.6%** / -26.1% (PASS) | +9.8% worse | **-42.1%** / -52.5% (PASS) | 71.6% → **25.0%** → 13.1% (**-65.1%**, PASS) | +23.52 [+17.04, +30.32] pp (PASS) | 0.23% |
| `OLMo-2-1124-7B-Instruct` | **-37.9%** / -50.7% (PASS) | +25.4% worse | **-35.1%** / -54.2% (PASS) | 25.5% → **11.7%** → 6.3% (**-54.3%**, PASS) | +3.32 [+0.78, +5.95] pp (PASS) | 0.36% |
| `Llama-3.1-8B-Instruct` | **-10.9%** / -15.8% (PASS) | +9.0% worse | **-6.0%** / -14.9% (FAIL) | 24.6% → **18.2%** → 14.9% (**-25.9%**, PASS) | +1.48 [+0.79, +2.21] pp (PASS) | 7.19% |

## 3. Out-of-Distribution Generalization — Confirm-OOD (`n=32`, Two-Stage Conditional Update + Relational Lookup)

| Model | OOD Link 1 (`Q_perp`, `k=8` / `k=16`) | OOD Link 2 (`F0` MSE, `k=8` / `k=16`) | OOD Link 3 (Tracking Err `PRIOR → U8 → U16`, `k=8` Drop) | OOD `PATCH_8 → UPDATE_8` |
|---|---:|---:|---:|---:|
| `Qwen2.5-0.5B-Instruct (Ref)` | **-44.8%** / -53.6% (PASS) | -18.5% / -34.5% (PASS) | 14.4% → **10.7%** → 6.6% (**-25.7%**, PASS) | +1.94 [-0.09, +3.97] pp (n.s.) |
| `Qwen2.5-3B-Instruct` | **-8.8%** / -17.7% (PASS) | +1.6% / -11.6% (n.s.) | 81.5% → **39.3%** → 19.5% (**-51.8%**, PASS) | +1.09 [-6.57, +8.89] pp (n.s.) |
| `Phi-3.5-mini-instruct` | **-21.9%** / -32.8% (PASS) | -29.7% / -42.9% (PASS) | 23.2% → **11.0%** → 7.0% (**-52.5%**, PASS) | +1.91 [-0.41, +4.40] pp (n.s.) |
| `Qwen2.5-7B-Instruct` | **-23.6%** / -34.3% (PASS) | -12.6% / -23.3% (PASS) | 65.4% → **32.2%** → 17.1% (**-50.8%**, PASS) | +6.59 [-0.02, +13.13] pp (n.s.) |
| `OLMo-2-1124-7B-Instruct` | **-40.0%** / -52.6% (PASS) | -33.3% / -49.0% (PASS) | 44.0% → **19.8%** → 14.1% (**-54.9%**, PASS) | +9.91 [+5.67, +14.33] pp (PASS) |
| `Llama-3.1-8B-Instruct` | **0.0%** / -3.2% (n.s.) | -6.9% / -12.3% (PASS) | 48.2% → **42.4%** → 36.9% (**-12.0%**, PASS) | +0.78 [+0.19, +1.39] pp (PASS) |

## 4. Inter-Layer Escape Mechanism & Exact 38-D 4-Block Decomposition

| Model | Tied? | Norm `src → mid → tgt` | Dark Energy `100% → mid → tgt` | Diag 1 Additivity Err | Diag 1 `comp / dark` Func Ratio | 4-Block Frob² `k=0 → k=8 → k=16` | Out-of-Top64 Share (`k=0`) |
|---|---|---:|---:|---:|---:|---:|---:|
| `Qwen2.5-0.5B-Instruct (Ref)` | True | 1.00 → 1.30 → 2.74 | 100% → **26.2%** → **34.7%** | 1.74e-06 | **2.53×** | 64.3 → 29.1 → 18.3 | 66.8% |
| `Qwen2.5-3B-Instruct` | True | 1.00 → 1.14 → 2.18 | 100% → **14.5%** → **12.0%** | 6.99e-06 | **1.71×** | 102.6 → 69.3 → 44.3 | 53.4% |
| `Phi-3.5-mini-instruct` | False | 1.00 → 1.52 → 2.89 | 100% → **37.9%** → **12.2%** | 2.91e-06 | **0.66×** | 51.7 → 32.5 → 20.2 | 61.8% |
| `Qwen2.5-7B-Instruct` | False | 1.00 → 1.27 → 2.00 | 100% → **60.1%** → **16.2%** | 8.72e-05 | **0.54×** | 67.5 → 42.0 → 27.1 | 59.4% |
| `OLMo-2-1124-7B-Instruct` | False | 1.00 → 1.28 → 1.89 | 100% → **48.7%** → **25.4%** | 1.48e-06 | **0.58×** | 50.4 → 26.0 → 14.3 | 74.2% |
| `Llama-3.1-8B-Instruct` | False | 1.00 → 1.18 → 2.25 | 100% → **43.7%** → **13.7%** | 2.39e-06 | **1.25×** | 26.0 → 18.5 → 12.8 | 74.1% |

### Key Mechanistic Synthesis
- **Universal Dark-to-Bright Escape Across All 6 Models & 4 Architectures**: Perturbations initialized purely inside the 38-D unembedding dark tail escape heavily into the non-dark residual complement by the final layer (only **12.0%–34.7%** dark-subspace energy remains at `h_tgt`) while amplifying in norm (`1.89×–2.89×`).
- **Tied vs. Untied Mid-Layer Escape Timing**: Tied-embedding models (`Qwen2.5-0.5B`, `Qwen2.5-3B`) rotate out of `Dark38` earlier (only `14.5%–26.2%` dark energy remaining at `h_mid`, where the escaped complement already drives `1.71×–2.53×` stronger downstream `F0` response), whereas all four untied models (`Phi-3.5-mini`, `Qwen2.5-7B`, `OLMo-2-7B`, `Llama-3.1-8B`) retain substantially higher mid-layer dark alignment (`37.9%–60.1%` at `h_mid`) before plunging to `12.2%–25.4%` at `h_tgt` (with Diag 1 `comp/dark` functional ratio `0.54×–0.66×` on Phi/Qwen7B/OLMo2 and intermediate `1.25×` on Llama-3.1-8B).
- **Full-Output Representation Is Structurally Necessary Everywhere**: Across every scale and architecture, **53%–74%** of the full 38-D prior operator error resides outside even the top-64 dynamic output basis (`outU_unobsX`), and exact full-`d` Gaussian conditioning collapses both in-basis and out-of-basis errors simultaneously while reducing observed-support error to machine precision (`~1e-23`).

## 5. Honest Scope & Limitations
- All 7B/8B models ran in **exact FP32** via split-isolated prefix streaming on L4 (verified bit-accurately against native CPU `model(...)` logits to `<= 8e-5` on P0 and `0.0` zero-perturbation difference), zero quantization or BF16 approximation was used.
- Target readouts `F0` are supplied to all intervention arms; results validate precision causal control of known linear-logit readouts rather than unsupervised capability gains.
- **Honest Boundary Cases**: On `Llama-3.1-8B-Instruct` (which already exhibits the strongest `k=0` context-conditioned prior, `total_frob_sq = 26.0` vs `50.4–102.6` in other models), Confirm-ID Link 1 (`-10.9%`), Link 3 (`-25.9%`), and `PATCH_8 -> UPDATE_8` (`+1.48 pp`) all pass strictly, as do OOD Link 2 (`-6.9%`), OOD Link 3 (`-12.0%`), and OOD `PATCH_8 -> UPDATE_8` (`+0.78 pp`), while ID Link 2 at `k=8` (`-6.0%`, 95% CI marginally crosses 0) and OOD Link 1 (`0.0%`, n.s.) do not reach 95% significance. On `Qwen2.5-3B-Instruct`, OOD Link 2 at `k=8` (`+1.6%`, n.s.) crosses zero despite passing at `k=16` (`-11.6%`) and passing OOD Link 1 (`-8.8%`) and Link 3 (`-51.8%`).

## 6. Generated Figures
- Saved in `results/figures/`: `fig1_cross_model_3links.png`, `fig2_scaling_budget_curves.png`, `fig3_interlayer_escape_mech.png`, `fig4_ood_and_patch_comparison.png`

