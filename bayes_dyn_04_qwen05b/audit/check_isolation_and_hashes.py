"""Independent Isolation, Query-Budget & Byte-Exact File SHA-256 Auditor (§9.4–§9.5, §11)."""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bytes_sha256_short(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def main() -> None:
    res = ROOT / "results"
    receipts = load_jsonl(res / "observations" / "query_receipts.jsonl")
    preds = load_jsonl(res / "predictions.jsonl")

    violations: list[str] = []
    n_learn, n_eval = 0, 0
    eval_first_utc: dict[str, str] = {}

    for r in receipts:
        if r["jvp_dirs_used"] > r["max_jvp_dirs"]:
            violations.append(f"JVP exceeded in {r['session_id']}")
        if r["role"] == "learning_support":
            n_learn += 1
            if r["vjp_dirs_used"] > 0 or r["allow_vjp"]:
                violations.append(f"Learning session had VJP access in {r['session_id']}")
        elif r["role"] == "evaluator_truth":
            n_eval += 1
            if r["events"]:
                cid = r["context_id"]
                ts = r["events"][0]["utc"]
                if cid not in eval_first_utc or ts < eval_first_utc[cid]:
                    eval_first_utc[cid] = ts

    # Verify chronological isolation: Stage A commit_utc <= Stage B evaluator first query utc!
    n_chrono = 0
    for p in preds:
        cid = p["context_id"]
        if cid in eval_first_utc:
            if p["commit_utc"] > eval_first_utc[cid]:
                violations.append(f"Stage A commit {p['commit_utc']} > Stage B eval {eval_first_utc[cid]} on {cid}")
            n_chrono += 1

    # Verify byte-exact SHA-256 of archived float32 M_k matches predictions.jsonl!
    n_hash_matched = 0
    pred_by_cid = {p["context_id"]: p for p in preds}
    for split_name in ("confirm_id", "confirm_ood"):
        post_p = res / "posteriors" / f"{split_name}_posteriors.npz"
        if not post_p.exists():
            continue
        post = np.load(post_p, allow_pickle=True)
        cids = [str(x) for x in post["context_ids"]]
        M_k = post["M_k"]
        for i, cid in enumerate(cids):
            if cid in pred_by_cid:
                for bi, kb in enumerate((0, 4, 8, 16)):
                    got = bytes_sha256_short(M_k[i, bi])
                    exp = pred_by_cid[cid]["M_k_f32_sha256"][f"k{kb}"]
                    if got != exp:
                        violations.append(f"Byte-hash mismatch on {cid} k={kb}: {got} vs {exp}")
                    else:
                        n_hash_matched += 1

    # Hash ALL archived project files
    hash_map: dict[str, str] = {}
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        if "__pycache__" in rel or rel.endswith(".pyc") or rel.endswith(".tmp") or rel == "audit/input_output_sha256.json":
            continue
        hash_map[rel] = sha256_file(p)
    (ROOT / "audit" / "input_output_sha256.json").write_text(json.dumps(hash_map, indent=2, sort_keys=True), encoding="utf-8")

    rep = {
        "learning_sessions": n_learn,
        "evaluator_sessions": n_eval,
        "chronological_checks_verified": n_chrono,
        "byte_exact_M_k_hashes_verified": n_hash_matched,
        "violations": violations,
        "isolation_and_hashes_passed": len(violations) == 0,
        "total_files_hashed": len(hash_map),
    }
    (ROOT / "audit" / "isolation_and_hashes_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    assert rep["isolation_and_hashes_passed"], f"Isolation/hash violations: {violations[:10]}"
    print(f"[AUDIT check_isolation_and_hashes] PASSED: {n_learn} learn, {n_eval} eval, {n_chrono} chrono, {n_hash_matched} byte-exact M_k hashes, 0 violations.")


if __name__ == "__main__":
    main()
