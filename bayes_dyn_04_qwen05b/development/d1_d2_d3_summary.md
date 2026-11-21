# BAYES-DYN-04 P0A Diagnostics (CPU-Only on Archived 02.5 & 03 Dev Assets)

## D1 — Observed vs Unobserved Input Subspace Error (BAYES-DYN-03, n=160 contexts)
- **k=4**: Observed `P_X` share = **0.32%**, Unobserved `(I-P_X)` share = **99.68%** (per-ctx `P_X` 10/50/90% quantiles: `[0.12, 0.3, 0.62]`).
- **k=8**: Observed `P_X` share = **0.86%**, Unobserved `(I-P_X)` share = **99.14%**.
- **Signed action error (`MAP_RISK_4`)**: mean `|e_par.x| = 0.00079`, mean `|e_perp.x| = 0.01077`, mean `|e_tot.x| = 0.01075`, opposite-sign cancellation rate = `51.0%`.

## D2 — Four-Block Input x Output Error Decomposition (DEV-tune32, full 896x38 active operator A)

| Budget | Model | `inU_obsX` | `inU_unobsX` | `outU_obsX` | `outU_unobsX` | Total Frob² |
|---|---|---:|---:|---:|---:|---:|
| k0 | Old `U_hybrid_64` (`sigma=0.02`, 0 comp) | 0.000 (0.0%) | 14.588 (25.8%) | 0.000 (0.0%) | 42.032 (74.2%) | 56.619 |
| k0 | **New Full 896D Exact Cond** | 0.000 (0.0%) | 14.588 (30.7%) | 0.000 (0.0%) | 33.002 (69.3%) | **47.589** |
| k4 | Old `U_hybrid_64` (`sigma=0.02`, 0 comp) | 0.000 (0.0%) | 9.237 (18.0%) | 4.583 (8.9%) | 37.449 (73.0%) | 51.269 |
| k4 | **New Full 896D Exact Cond** | 0.000 (0.0%) | 9.237 (33.1%) | 0.000 (0.0%) | 18.708 (66.9%) | **27.945** |
| k8 | Old `U_hybrid_64` (`sigma=0.02`, 0 comp) | 0.000 (0.0%) | 7.214 (14.6%) | 8.887 (18.0%) | 33.145 (67.3%) | 49.246 |
| k8 | **New Full 896D Exact Cond** | 0.000 (0.0%) | 7.214 (33.2%) | 0.000 (0.0%) | 14.518 (66.8%) | **21.731** |
| k16 | Old `U_hybrid_64` (`sigma=0.02`, 0 comp) | 0.000 (0.0%) | 4.354 (9.4%) | 17.553 (37.8%) | 24.479 (52.8%) | 46.386 |
| k16 | **New Full 896D Exact Cond** | 0.000 (0.0%) | 4.354 (32.4%) | 0.000 (0.0%) | 9.084 (67.6%) | **13.438** |

## D3 — Support Observation Absorption (BAYES-DYN-03, `sigma_obs=0.02`)
- **k=4**: Support residual RMS = `0.01990` (median rel = `9.12%`) vs numerical JVP FD error `1.37e-4` vs unobserved per-dim RMS `0.11603`.
- **k=8**: Support residual RMS = `0.02091` (median rel = `9.32%`).

## DEV-fit192 / DEV-tune32 Full-Operator Selection (3 pre-fixed orthogonal banks)
- Selected: **`alpha = 100.0`**, **`eta = 0.05`** (frozen `s_A^2 = 7.5236`, `near_zero_p10 = 2.1345`).
- DEV-tune32 normalized `L_0^perp = 0.16699` -> `L_8^perp = 0.09522` (**`-42.98%`** strictly orthogonal full-896D response error reduction!).
