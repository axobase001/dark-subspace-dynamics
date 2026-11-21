"""Prepare zero-overlap balanced dataset splits & frozen manifest for BAYES-DYN-05.

Reuses:
  - DEV-fit (96, 24/family) & DEV-tune (16, 4/family) from BAYES-DYN-02.5 prompts_224.jsonl (train & validation)
  - DEV-PILOT (16), CAL (24, 6/family), CONFIRM-ID (64, 16/family, 4x4 option-balanced),
    and CONFIRM-OOD (32, exact 2x2x4x2 balanced) from BAYES-DYN-04 data/*.jsonl.
Verifies strict 0 text overlap between DEV and EVAL splits and hashes all files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R025 = ROOT.parent / "BAYES_DYN_02_5"
R04 = ROOT.parent / "BAYES_DYN_04"


def load_jsonl(p: Path) -> list[dict[str, Any]]:
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(p: Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def norm_text(s: str) -> str:
    return " ".join(s.strip().split())


def main() -> None:
    cfg = json.loads((ROOT / "configs" / "suite_config.json").read_text(encoding="utf-8"))
    fams = cfg["data"]["id_families"]
    opts = cfg["data"]["options"]

    p224 = load_jsonl(R025 / "configs" / "prompts_224.jsonl")
    dev_fit = [r for r in p224 if r["split"] == "train"]
    val_32 = [r for r in p224 if r["split"] == "validation"]
    dev_tune = []
    for fm in fams:
        sub = [r for r in val_32 if r["family"] == fm]
        dev_tune.extend(sub[:4])

    assert len(dev_fit) == 96, f"Expected 96 dev_fit, got {len(dev_fit)}"
    assert len(dev_tune) == 16, f"Expected 16 dev_tune, got {len(dev_tune)}"

    pilot_16 = load_jsonl(R04 / "data" / "dev_pilot.jsonl")
    cal_all = load_jsonl(R04 / "data" / "cal.jsonl")
    cal_24 = []
    for fm in fams:
        sub = [r for r in cal_all if r["family"] == fm]
        cal_24.extend(sub[:6])

    cid_all = load_jsonl(R04 / "data" / "confirm_id.jsonl")
    # Select exact 16 per family (4 per A/B/C/D per family = 64 total!)
    cid_64 = []
    for fm in fams:
        for op in opts:
            sub = [r for r in cid_all if r["family"] == fm and r["target_option"] == op]
            assert len(sub) >= 4, f"Not enough {fm}/{op}: {len(sub)}"
            cid_64.extend(sub[:4])

    ood_32 = load_jsonl(R04 / "data" / "confirm_ood.jsonl")

    # Pre-register 12 trajectory contexts from Confirm-ID64 (3 per family, deterministic)
    traj_12 = []
    for fm in fams:
        sub = [r for r in cid_64 if r["family"] == fm]
        traj_12.extend([r["context_id"] for r in sub[:3]])

    # Verify strict 0 prompt overlap across ALL 6 splits!
    splits = {
        "dev_fit_96": dev_fit,
        "dev_tune_16": dev_tune,
        "dev_pilot_16": pilot_16,
        "cal_24": cal_24,
        "confirm_id_64": cid_64,
        "confirm_ood_32": ood_32,
    }
    seen_texts: dict[str, str] = {}
    seen_cids: dict[str, str] = {}
    for sn, rows in splits.items():
        for r in rows:
            t = norm_text(r["user_prompt"])
            cid = r["context_id"]
            assert t not in seen_texts, f"Prompt text overlap between {sn} and {seen_texts[t]}: {cid}"
            assert cid not in seen_cids, f"Context ID overlap between {sn} and {seen_cids[cid]}: {cid}"
            seen_texts[t] = sn
            seen_cids[cid] = sn

    # Frozen S_mispair 8x8 derangement + sign flip (same construction as BAYES-DYN-04)
    rng_m = np.random.default_rng(int(cfg["derived_seeds"]["mispair_perm"]))
    perm = np.array([1, 2, 3, 4, 5, 6, 7, 0], dtype=int)
    signs = rng_m.choice([-1.0, 1.0], size=8)
    S_mis = np.zeros((8, 8), dtype=np.float64)
    for j in range(8):
        S_mis[perm[j], j] = signs[j]
    assert np.allclose(S_mis.T @ S_mis, np.eye(8)) and np.allclose(np.diag(S_mis), 0.0)

    data_dir = ROOT / "data"
    file_hashes: dict[str, str] = {}
    counts: dict[str, Any] = {}
    for sn, rows in splits.items():
        p = data_dir / f"{sn}.jsonl"
        save_jsonl(p, rows)
        file_hashes[f"data/{sn}.jsonl"] = sha256_file(p)
        counts[sn] = {
            "n": len(rows),
            "families": {fm: sum(1 for r in rows if r["family"] == fm) for fm in sorted(set(r["family"] for r in rows))},
            "options": {op: sum(1 for r in rows if r["target_option"] == op) for op in opts},
        }

    manifest = {
        "experiment_id": "BAYES-DYN-05",
        "zero_overlap_verified": True,
        "total_unique_contexts": len(seen_cids),
        "split_counts": counts,
        "pre_registered_trajectory_panel_12": traj_12,
        "S_mispair_8x8": S_mis.tolist(),
        "file_sha256": file_hashes,
    }
    (ROOT / "configs" / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DATA v05] Prepared {len(seen_cids)} unique zero-overlap contexts across 6 splits. Traj12={traj_12}")


if __name__ == "__main__":
    main()
