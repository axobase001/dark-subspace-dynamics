# BAYES-DYN-04 — Full-Output Dark-Subspace Dynamics Map, Evidence Updating & Real Intervention

## Overview
Tests whether exact deterministic Gaussian conditioning on a complete **896×38** target-agnostic propagation map $A_c = (J_c - I)V$ updates predictions on strictly unobserved orthogonal directions (`Q_perp`), transfers to functional readouts from the same frozen posterior (`F0` + 3 option contrasts), and reduces real matched-`batch=1` nonlinear target tracking error.

## Execution & Independent Replay
1. **CPU P0A Diagnostics & DEV-fit192/DEV-tune32 Prior Freeze**: `python src/dev_diagnostics_p0a.py`
2. **CPU New Data Generation & Independent Text-Only Verifier**: `python src/data_generator_v04.py`
3. **Colab L4 Execution with Independent 30s Heartbeat Watchdog**: `python runtime/run_and_watch_v04.py` (or `python run_bayes_dyn_04.py` directly on GPU)
4. **Independent CPU-Only Audit Suite (zero production update / report imports)**:
   - `python audit/replay_posterior.py`
   - `python audit/recompute_metrics.py`
   - `python audit/check_isolation_and_hashes.py`
5. **Report & Figures**: `python src/report_generator_v04.py`

See `FINDINGS.md` for all separated pre-registered verdicts and `atlas_index.jsonl` for the searchable per-context map index.
