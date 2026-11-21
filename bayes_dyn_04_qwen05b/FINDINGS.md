# BAYES-DYN-04 — Evidence-Updated Dark-Subspace Dynamics Map & Intervention Verification

> **Reporting Policy (§12)**: Each link of the `Evidence -> Full 896-D Map -> Functional Projection -> Real Intervention` chain is evaluated independently under pre-registered criteria. No blanket `GREEN` banner is used.

## 0. First-Screen Summary (Ordered 6-Part Verdict)

| # | Dimension | Empirical Evidence (Confirm-ID96 unless noted) | Verdict |
|---|---|---|---|
| **1** | **Map Revision (P0A -> Full 896-D Exact Cond)** | Old 03 D1: **99.68%** (`k=4`) of error sat in unobserved `(I-P_X)`; D2: **74.2%** of full-operator error sat in dropped output complement of `U_hybrid_64`; D3: `sigma_obs=0.02` left `9.12%` support residual. New model preserves full **896×38** mean + exact deterministic Gaussian conditioning (`alpha=100.0, eta=0.05`). | **IMPLEMENTED & P0 VALID** |
| **2** | **Unobserved 896-D Prediction (Primary Link 1)** | `PRIOR - UPDATE_8` on strictly orthogonal `Q_perp`: `+0.08897 [+0.07875, +0.10017]` (**`-41.35%`**, improved on `100.0%` of 96 contexts; raw `L_perp`: `1.6186 -> 0.9492`). `MISPAIRED_8` degrades error (`-0.04776`); `PATCH_8` equals `PRIOR` (`0.00000`). | **SUPPORTED (CI95 > 0)** |
| **3** | **Functional Projection & Uncertainty (Link 2)** | Same frozen posterior projected onto `F0` on `Q_perp`: MSE `0.003413 -> 0.002902` (`+0.000511 [+0.000244, +0.000783]`, **`-14.96%`**); 3 option contrasts MSE `0.005275 -> 0.004560`. CAL32 temp `t=2.0133` gives `UPDATE_8` 50/80/90/95% `Q_perp` coverage `71.2% / 94.5% / 97.5% / 99.2%`. | **SUPPORTED (CI95 > 0)** |
| **4** | **Real Intervention Tracking (Link 3, τ=0.05)** | Working regime on DEV-PILOT=`True`. `PRIOR - UPDATE_8` tracking error $|\Delta F_0 - \tau|/\tau$: `14.20% -> 10.07%` (diff `+4.13 pp [+2.63 pp, +5.71 pp]`, **`-29.06%`**); ±10% hit rate `47.9% -> 57.3%` (vs `EXACT` `100.0%`). | **SUPPORTED (FULL 3-LINK CHAIN)** |
| **5** | **16-Context Inter-Layer Trajectory Panel (`h16→h20→h24`)** | Exact 38-D 4-block total Frob² drops `64.29 (k=0) -> 29.07 (k=8) -> 18.29 (k=16)`; `z20` split additivity max rel err `1.74e-06`; Dark4 vs Bright4 vs Rand4 propagation documented below. | **COMPLETED (16/16 + MECH)** |
| **6** | **Actual Resources, Isolation & Independent Audit** | GPU `148.99s` (`0.0414 GPU-hr`, peak `6353.3 MB`), session wall-clock `330.0s`; Colab CU settled `110.36 -> 110.20` (`0.16 CU` this run; `0.32 CU` total across pre-checks + final run vs `2.0 CU` cap; stop confirmed=`True`, `0` active); independent replay=`True`, isolation=`True` (`0` violations, `14/14` frozen code/data SHA-256 matched). | **VALID & RELEASED** |

## 1. P0A Dev Diagnostics (CPU on 02.5 / 03) & Live P0 Verification

- **D1 (BAYES-DYN-03, n=160)**: `k=4` unobserved input `(I-P_X)` error share = `99.68%`, `k=8` = `99.14%`.
- **D2 (DEV-tune32, full 896×38 `A`)**: Old `U_hybrid_64` (`sigma=0.02`, zero complement) had `74.2%` of error outside `U_hybrid_64` (`total=56.62 -> 49.25` at `k=8`), whereas New Full 896-D exact conditioning drops `47.59 -> 21.73` at `k=8` (`-42.98%` strictly orthogonal error on 3 DEV-tune banks).
- **D3 (Support absorption)**: Old `sigma_obs=0.02` left `9.12%` median relative functional residual on observed supports (vs `1.37e-4` numerical JVP error).
- **Live P0 (`results/p0_checks.json`)**: `all_p0_passed=True`, FP64 synth `max_err <= 1e-9`, zero-pert `D=0.0e+00`, JVP FD `1.31e-04`, adjoint `2.35e-05`, linear comb `2.33e-06`, `z20->z24` chain `2.38e-06`, archived cond rel err `7.15e-13`, truth-replacement sentinel diff `0.0e+00`.

## 2.1 Unobserved Prediction & Functional Projection — Confirm-ID 96 (Primary Panel)

| Method | $L_k^\perp / s_A^2$ (95% CI) | Raw Sq-L2 ($Q_\perp$) | In / Out Old `U_64` | $L_k^{\text{gen}} / s_A^2$ | `F0` MSE ($Q_\perp$) | `F0` Sign Acc | `Opt3` MSE | `F0` 90%-Cov / Width / NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `PRIOR` | 0.21514 [0.20241, 0.22863] | 1.6186 | 0.536 / 1.083 | 0.24245 | 0.003413 | 84.9% | 0.005275 | `98.8% / 0.3215 / -1.252` |
| `UPDATE_4` | 0.14529 [0.13836, 0.15241] | 1.0931 | 0.389 / 0.704 | 0.13644 | 0.003045 | 87.2% | 0.004926 | `98.2% / 0.2711 / -1.378` |
| `UPDATE_8` | 0.12617 [0.12120, 0.13118] | 0.9492 | 0.337 / 0.612 | 0.10363 | 0.002902 | 86.8% | 0.004560 | `97.5% / 0.2548 / -1.424` |
| `UPDATE_16` | 0.10866 [0.10448, 0.11291] | 0.8175 | 0.284 / 0.533 | 0.06479 | 0.002432 | 88.5% | 0.003733 | `98.2% / 0.2373 / -1.497` |
| `PATCH_8` | 0.21514 [0.20241, 0.22863] | 1.6186 | 0.536 / 1.083 | 0.18413 | 0.003413 | 84.9% | — | `97.4% / 0.2548 / -1.377` |
| `MISPAIRED_8` | 0.26291 [0.24170, 0.28647] | 1.9780 | 0.667 / 1.311 | 0.34148 | 0.004191 | 83.5% | — | `94.7% / 0.2548 / -1.315` |

### Paired Improvements vs `PRIOR` (20,000 Family-Stratified Paired Bootstrap)

| Contrast | Link 1: $\Delta L^\perp$ (Rel Drop, % Improved) | Link 2: $\Delta \text{MSE}(F_0)$ (Rel Drop) | Link 3: $\Delta E^{\text{control}}$ (Rel Drop) |
|---|---:|---:|---:|
| `PRIOR - UPDATE_4` | +0.06985 [+0.06001, +0.08072] (-32.5%, 97%) | +0.000368 [+0.000104, +0.000620] (-10.8%) | +1.74 pp [+0.34 pp, +3.21 pp] (-12.3%) |
| `PRIOR - UPDATE_8` | +0.08897 [+0.07875, +0.10017] (-41.4%, 100%) | +0.000511 [+0.000244, +0.000783] (-15.0%) | +4.13 pp [+2.63 pp, +5.71 pp] (-29.1%) |
| `PRIOR - UPDATE_16` | +0.10648 [+0.09567, +0.11807] (-49.5%, 100%) | +0.000981 [+0.000702, +0.001271] (-28.7%) | +7.57 pp [+5.90 pp, +9.32 pp] (-53.3%) |
| `PRIOR - PATCH_8` | +0.00000 [-0.00000, +0.00000] (0.0%, 23%) | +0.000000 [-0.000000, +0.000000] (0.0%) | +2.58 pp [+1.76 pp, +3.47 pp] (-18.1%) |
| `PRIOR - MISPAIRED_8` | -0.04776 [-0.06303, -0.03360] (+22.2% worse, 20%) | -0.000778 [-0.001180, -0.000420] (+22.8% worse) | — |

### Per-Family Breakdown of Primary Link 1 (`PRIOR - UPDATE_8` on `Q_perp`)

- `attribute_binding` (n=24): `L0=0.26361` -> `L8=0.14904` (`+0.11457`, improved on `100.0%` of contexts)
- `instruction_priority` (n=24): `L0=0.22175` -> `L8=0.14977` (`+0.07198`, improved on `100.0%` of contexts)
- `simple_arithmetic` (n=24): `L0=0.12968` -> `L8=0.08844` (`+0.04124`, improved on `100.0%` of contexts)
- `state_tracking` (n=24): `L0=0.24553` -> `L8=0.11743` (`+0.12810`, improved on `100.0%` of contexts)

## 2.2 Unobserved Prediction & Functional Projection — Confirm-OOD 32 (Two-Stage Conditional Update + Relational Lookup — Secondary)

| Method | $L_k^\perp / s_A^2$ (95% CI) | Raw Sq-L2 ($Q_\perp$) | In / Out Old `U_64` | $L_k^{\text{gen}} / s_A^2$ | `F0` MSE ($Q_\perp$) | `F0` Sign Acc | `Opt3` MSE | `F0` 90%-Cov / Width / NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `PRIOR` | 0.27125 [0.24665, 0.29945] | 2.0408 | 0.849 / 1.192 | 0.26078 | 0.003202 | 84.4% | 0.007557 | `100.0% / 0.3289 / -1.274` |
| `UPDATE_4` | 0.17390 [0.16205, 0.18598] | 1.3083 | 0.526 / 0.782 | 0.15127 | 0.002795 | 85.9% | 0.005995 | `98.4% / 0.2775 / -1.390` |
| `UPDATE_8` | 0.14973 [0.14091, 0.15872] | 1.1265 | 0.430 / 0.696 | 0.11428 | 0.002608 | 87.1% | 0.005716 | `98.0% / 0.2608 / -1.436` |
| `UPDATE_16` | 0.12576 [0.11852, 0.13337] | 0.9462 | 0.344 / 0.602 | 0.07469 | 0.002096 | 85.5% | 0.004294 | `99.6% / 0.2415 / -1.517` |
| `PATCH_8` | 0.27125 [0.24665, 0.29945] | 2.0408 | 0.849 / 1.192 | 0.18449 | 0.003202 | 84.4% | — | `96.5% / 0.2608 / -1.392` |
| `MISPAIRED_8` | 0.33910 [0.30096, 0.38080] | 2.5512 | 1.076 / 1.476 | 0.40169 | 0.004341 | 81.2% | — | `96.1% / 0.2608 / -1.318` |

### Paired Improvements vs `PRIOR` (20,000 Family-Stratified Paired Bootstrap)

| Contrast | Link 1: $\Delta L^\perp$ (Rel Drop, % Improved) | Link 2: $\Delta \text{MSE}(F_0)$ (Rel Drop) | Link 3: $\Delta E^{\text{control}}$ (Rel Drop) |
|---|---:|---:|---:|
| `PRIOR - UPDATE_4` | +0.09735 [+0.07824, +0.11926] (-35.9%, 97%) | +0.000406 [-0.000124, +0.000922] (-12.7%) | +1.37 pp [-0.66 pp, +3.59 pp] (-9.5%) |
| `PRIOR - UPDATE_8` | +0.12151 [+0.10056, +0.14594] (-44.8%, 100%) | +0.000593 [+0.000028, +0.001142] (-18.5%) | +3.71 pp [+1.18 pp, +6.27 pp] (-25.7%) |
| `PRIOR - UPDATE_16` | +0.14549 [+0.12197, +0.17315] (-53.6%, 100%) | +0.001106 [+0.000553, +0.001682] (-34.5%) | +7.83 pp [+4.45 pp, +11.45 pp] (-54.4%) |
| `PRIOR - PATCH_8` | +0.00000 [-0.00000, +0.00000] (0.0%, 34%) | +0.000000 [-0.000000, +0.000000] (0.0%) | +1.77 pp [+0.47 pp, +3.08 pp] (-12.3%) |
| `PRIOR - MISPAIRED_8` | -0.06785 [-0.10095, -0.03763] (+25.0% worse, 25%) | -0.001139 [-0.002039, -0.000297] (+35.6% worse) | — |

### Per-Family Breakdown of Primary Link 1 (`PRIOR - UPDATE_8` on `Q_perp`)

- `two_stage_conditional_update_relational_lookup` (n=32): `L0=0.27125` -> `L8=0.14973` (`+0.12151`, improved on `100.0%` of contexts)

## 3.1 Real Matched-`batch=1` Interventions — Confirm-ID 96 (Frozen $\tau=0.05$, $\rho=0.01\|h_{16}\|$)

| Arm | Tracking Err $|\Delta F_0 - \tau|/\tau$ (95% CI) | ±10% Hit Rate (95% CI) | Mean $\Delta F_0$ | Mean $\|x\|$ | Mean $\|\Delta h_{24}\|$ | NT-KL | 4-Choice Acc (Base→Pert) | Action 90% Cov (Lin / E2E) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ZERO` | 100.00% [100.00%, 100.00%] | 0.0% [0.0%, 0.0%] | +0.0000 | 0.0000 | 0.0000 | 0.00000 | 67.7% → 67.7% | `— / —` |
| `PRIOR` | 14.20% [12.24%, 16.21%] | 47.9% [38.5%, 57.3%] | +0.0479 | 0.0766 | 0.2879 | 0.00028 | 67.7% → 67.7% | `90.6% / 90.6%` |
| `UPDATE_4` | 12.46% [10.55%, 14.42%] | 50.0% [40.6%, 59.4%] | +0.0480 | 0.0750 | 0.2918 | 0.00026 | 67.7% → 67.7% | `88.5% / 89.6%` |
| `UPDATE_8` | 10.07% [8.51%, 11.72%] | 57.3% [47.9%, 66.7%] | +0.0483 | 0.0744 | 0.2893 | 0.00026 | 67.7% → 67.7% | `86.5% / 87.5%` |
| `UPDATE_16` | 6.63% [5.58%, 7.76%] | 81.2% [74.0%, 88.5%] | +0.0490 | 0.0732 | 0.2858 | 0.00027 | 67.7% → 67.7% | `90.6% / 90.6%` |
| `PATCH_8` | 11.62% [9.92%, 13.38%] | 56.2% [46.9%, 65.6%] | +0.0482 | 0.0752 | 0.2815 | 0.00026 | 67.7% → 67.7% | `86.5% / 86.5%` |
| `EXACT` | 0.40% [0.36%, 0.44%] | 100.0% [100.0%, 100.0%] | +0.0501 | 0.0717 | 0.2863 | 0.00028 | 67.7% → 67.7% | `— / —` |

- **Common-Action Diagnostic (both `PRIOR` & `UPDATE_8` predicting the exact same `PRIOR` action $x_0$, §7.2)**: `PRIOR` mean $|\hat a_0^\top x_0 - a_{\text{true}}^\top x_0| = 0.00709` vs `UPDATE_8` `0.00530` (paired improvement `+0.00179 [+0.00104, +0.00255]`).
- **Cross-Direction Transfer vs Support-Patching Only (`PATCH_8 - UPDATE_8`, exploratory diagnostic)**: `+1.55 pp [+0.51 pp, +2.66 pp]` tracking-error reduction on Confirm-ID96 (improved on `62.5%` of contexts, 95% CI strictly positive), confirming that off-support conditioning contributes beyond patching the 8 observed directions alone.
- **Calibration Boundary Note**: Global scalar temperature `t = 2.0133` fitted on CAL-32 full-output normalized errors is slightly conservative on 1-D `Q_perp` `F0` projections (`97.5%` vs nominal `90%` for `UPDATE_8`), while chosen-action 90% coverage tracks nominal closely (`86.5%–90.6%` on Confirm-ID96).

## 3.2 Real Matched-`batch=1` Interventions — Confirm-OOD 32 (Frozen $\tau=0.05$, $\rho=0.01\|h_{16}\|$)

| Arm | Tracking Err $|\Delta F_0 - \tau|/\tau$ (95% CI) | ±10% Hit Rate (95% CI) | Mean $\Delta F_0$ | Mean $\|x\|$ | Mean $\|\Delta h_{24}\|$ | NT-KL | 4-Choice Acc (Base→Pert) | Action 90% Cov (Lin / E2E) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ZERO` | 100.00% [100.00%, 100.00%] | 0.0% [0.0%, 0.0%] | +0.0000 | 0.0000 | 0.0000 | 0.00000 | 50.0% → 50.0% | `— / —` |
| `PRIOR` | 14.39% [10.32%, 18.90%] | 43.8% [28.1%, 59.4%] | +0.0443 | 0.0795 | 0.2586 | 0.00024 | 50.0% → 50.0% | `81.2% / 81.2%` |
| `UPDATE_4` | 13.02% [9.96%, 16.53%] | 43.8% [28.1%, 59.4%] | +0.0447 | 0.0767 | 0.2728 | 0.00028 | 50.0% → 50.0% | `84.4% / 84.4%` |
| `UPDATE_8` | 10.69% [8.08%, 13.57%] | 50.0% [31.2%, 65.6%] | +0.0454 | 0.0768 | 0.2637 | 0.00025 | 50.0% → 50.0% | `87.5% / 87.5%` |
| `UPDATE_16` | 6.56% [5.10%, 8.17%] | 75.0% [59.4%, 87.5%] | +0.0476 | 0.0777 | 0.2688 | 0.00026 | 50.0% → 50.0% | `96.9% / 96.9%` |
| `PATCH_8` | 12.62% [9.26%, 16.39%] | 50.0% [31.2%, 65.6%] | +0.0452 | 0.0783 | 0.2505 | 0.00021 | 50.0% → 50.0% | `81.2% / 81.2%` |
| `EXACT` | 0.38% [0.30%, 0.46%] | 100.0% [100.0%, 100.0%] | +0.0500 | 0.0776 | 0.2681 | 0.00027 | 50.0% → 50.0% | `— / —` |

- **Common-Action Diagnostic (both `PRIOR` & `UPDATE_8` predicting the exact same `PRIOR` action $x_0$, §7.2)**: `PRIOR` mean $|\hat a_0^\top x_0 - a_{\text{true}}^\top x_0| = 0.00717` vs `UPDATE_8` `0.00487` (paired improvement `+0.00229 [+0.00093, +0.00377]`).
- **Cross-Direction Transfer vs Support-Patching Only (`PATCH_8 - UPDATE_8`, exploratory diagnostic)**: `+1.94 pp [-0.09 pp, +3.97 pp]` on Confirm-OOD32 (`53.1%` improved; same magnitude as ID, 95% CI marginally straddles zero at `n=32`).

## 4. Inter-Layer Dynamics Panel (`h16 -> h20 -> h24`, Pre-Registered 16 Confirm-ID Contexts)

- **Tangent norm growth along `Q_perp`**: `||v_16|| = 1.000` → `||z_20|| = 1.295` (`||z_20 - v_16|| = 1.230`) → `||z_24|| = 2.744` (`||z_24 - z_20|| = 2.480`).
- **Dark-subspace energy fraction**: drops from `100%` at `h16` → `26.22%` at `h20` → `34.71%` at `h24` (energy migrates strongly out of Dark38 into bright/middle output directions across layers!).
- **Fixed endpoint readout projection vs end response**: mean $|g_{24}^\top z_{20}| = 0.0671$ (fixed readout projection at `h20`, NOT downstream causal contribution) vs end response $|g_{24}^\top z_{24}| = 0.1086$.

### Exact 38-D Full-Operator 4-Block Error Decomposition (Confirm-ID Trajectory Panel, n=16)

| Budget | `inU_obsX` | `inU_unobsX` | `outU_obsX` | `outU_unobsX` | Total Frob² |
|---|---:|---:|---:|---:|---:|
| `k0` | 0.0000 | 21.3145 | 0.0000 | 42.9727 | **64.2872** |
| `k4` | 0.0000 | 13.4827 | 0.0000 | 24.9318 | **38.4145** |
| `k8` | 0.0000 | 10.1217 | 0.0000 | 18.9477 | **29.0694** |
| `k16` | 0.0000 | 6.2936 | 0.0000 | 12.0007 | **18.2943** |

### Optional Mechanism Diagnostics (§8)

- **Diag 1 (`z20 = P_D z20 + (I - P_D) z20` propagated through `h20 -> h24` JVP)**: Exact first-order additivity max relative error = **`1.74e-06`**. Mean downstream tangent norm from dark `z20` part = `1.610` (`|g.z24| = 0.0347`) vs from non-dark `z20` complement = `2.086` (`|g.z24| = 0.0878`).
- **Diag 2 (Matched unit-norm 4-direction input subspaces at `h16 -> h24`)**: Mean `||z24||`: Dark4=`2.707`, Bright4=`2.236`, RandOrth4=`2.205`; Mean `|g.z24|`: Dark4=`0.1072`, Bright4=`0.0962`, RandOrth4=`0.0883`. (Descriptive comparison only; does not establish unique Bayesian advantage in dark space.)

## 5. Generated Figures & Scope Boundaries

- Figures saved in `results/figures/`: `fig1_budget_curve.png`, `fig2_four_block_errors.png`, `fig3_context_before_after.png`
- This map covers only the 38-D dark unembedding tail input channel on `h16 -> h24` at the final prompt token of `Qwen/Qwen2.5-0.5B-Instruct`; it does not recover the full `896×896` Jacobian, multi-token interactions, or open-ended safety behavior.
- Target labels were provided to control arms to compute `F0`; changes in 4-choice accuracy are not unsupervised reasoning gains.

