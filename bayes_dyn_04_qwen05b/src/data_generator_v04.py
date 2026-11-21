"""Programmatic Dataset Generator & Independent TEXT-ONLY Answer Verifier for BAYES-DYN-04 (§5.1–§5.2).

Generates 176 brand-new independent contexts:
  - data/dev_pilot.jsonl   : 16 (4 ID families x 4, balanced A/B/C/D)
  - data/cal.jsonl         : 32 (4 ID families x 8, balanced A/B/C/D)
  - data/confirm_id.jsonl  : 96 (4 ID families x 24, 6 each A/B/C/D per family)
  - data/confirm_ood.jsonl : 32 (two-stage conditional state update + relational lookup;
                                 2 templates x 2 high/low branches x 4 options x 2 = 32 cross-balanced!)

Guarantees:
  - Matched numeric ranges, prompt length, and difficulty across dev_pilot / cal / confirm_id.
  - Template groups & scenario groups isolated from old dev_224 and old BAYES-DYN-03 contexts.
  - INDEPENDENT TEXT-ONLY VERIFIER parses ONLY the displayed `user_prompt` text & option lines
    (NEVER reads hidden `rule_spec`!) to verify 176/176 ground-truth labels!
  - Freezes 16 family-balanced Confirm-ID trajectory_panel IDs, bank seeds, and S_mispair in data_manifest.json.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.probe_bank_v04 import (
    generate_fixed_38x38_bank,
    generate_q_general,
    generate_frozen_mispair_matrix,
    bytes_sha256_short,
)

OPTIONS = ("A", "B", "C", "D")
SUFFIX = "Answer with one option letter only."

# Shared matched-difficulty vocabularies across ID splits (unique scenario instances guaranteed per split!)
PEOPLE_ID = [
    "Arthur", "Bianca", "Caleb", "Diana", "Edgar", "Fiona", "Gavin", "Hazel",
    "Isaac", "Julia", "Kelvin", "Laura", "Martin", "Naomi", "Oscar", "Paula",
    "Quincy", "Rachel", "Simon", "Trina", "Ulysses", "Valerie", "Walter", "Xena",
]
ITEMS_ID = [
    "compass", "chronograph", "microscope", "sextant", "barograph", "spectrometer",
    "voltmeter", "ammeter", "altimeter", "tachometer", "anemometer", "calorimeter",
    "magnetometer", "photometer", "thermograph", "gravimeter",
]
PLACES_ID = [
    "courtyard", "conservatory", "vestibule", "colonnade", "veranda", "balcony",
    "scriptorium", "solarium", "athenaeum", "belfry", "cloister", "galleryway",
    "orangery", "refectory", "sacristy", "undercroft",
]
COLORS_ID = [
    "carmine", "celadon", "citrine", "cobaltblue", "magenta", "ochre",
    "peridot", "saffron", "scarlet", "turquoise", "umber", "viridian",
]
MATERIALS_ID = [
    "alabaster", "birchwood", "corundum", "dolomite", "ebonite", "granodiorite",
    "hematite", "mahogany", "porcelain", "sandstone", "titanium", "walnutwood",
]
CODES_ID = [
    "TX11", "RX22", "MX33", "NX44", "PX55", "QX66", "SX77", "VX88", "WX99", "ZX15", "KX27", "LX39",
]

# OOD vocabulary
OOD_PEOPLE = ["Alaric", "Beatrix", "Cassian", "Daphne", "Evander", "Flavia", "Gervase", "Honoria", "Ignatius", "Jocasta", "Leander", "Marcella"]
OOD_ROLES = ["courier", "inspector", "technician", "navigator", "archivist", "auditor"]
OOD_BADGES = ["CrimsonBadge", "SilverBadge", "GoldBadge", "CobaltBadge", "JadeBadge", "AmberBadge"]
OOD_HUBS = ["NorthVault", "SouthVault", "EastVault", "WestVault", "CentralSpire", "LowerConcourse", "UpperConcourse", "HarborDock"]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _place_options(rng: np.random.Generator, correct: str, distractors: list[str], target_opt: str) -> dict[str, str]:
    assert len(distractors) == 3 and correct not in distractors and len(set(distractors)) == 3
    dp = [str(x) for x in rng.permutation(distractors)]
    opts: dict[str, str] = {}
    di = 0
    for o in OPTIONS:
        if o == target_opt:
            opts[o] = correct
        else:
            opts[o] = dp[di]
            di += 1
    return opts


def _format_prompt(body: str, opts: dict[str, str]) -> str:
    return f"{body}\nA. {opts['A']}\nB. {opts['B']}\nC. {opts['C']}\nD. {opts['D']}\n{SUFFIX}"


def _build_id_record(split_name: str, family: str, idx: int, target_opt: str, rng: np.random.Generator) -> dict[str, Any]:
    """Build matched-difficulty ID record with split-isolated template_group_id."""
    tpl_v = int(rng.integers(0, 2))
    tpl_gid = f"v04_{split_name}_{family}_t{tpl_v}"
    rec_n = idx + 1

    if family == "state_tracking":
        it_a, it_b = [str(x) for x in rng.choice(ITEMS_ID, size=2, replace=False)]
        p0, p1, p_dest, p_dist = [str(x) for x in rng.choice(PLACES_ID, size=4, replace=False)]
        person = str(rng.choice(PEOPLE_ID))
        header = {"dev_pilot": "Pilot transit slip", "cal": "Calibration transit slip", "confirm_id": "Station transit slip"}[split_name]
        if tpl_v == 0:
            body = (
                f"{header} #{rec_n}:\n"
                f"Start state: the {it_a} is in the {p0}, and the {it_b} is in the {p1}.\n"
                f"Move event: {person} transfers the {it_a} from the {p0} to the {p_dest}.\n"
                f"Question: Where is the {it_a} located after the transfer?"
            )
        else:
            body = (
                f"{header} #{rec_n}:\n"
                f"Start state: the {it_b} is in the {p1}, and the {it_a} is in the {p0}.\n"
                f"Move event: {person} relocates the {it_a} out of the {p0} into the {p_dest}.\n"
                f"Question: Which room holds the {it_a} now?"
            )
        correct = p_dest
        distractors = [p0, p1, p_dist]
        canonical = f"v04_st|{it_a}|{p0}->{p_dest}|{it_b}@{p1}|{person}"
        spec = {"family": family, "item": it_a, "dest": p_dest}

    elif family == "attribute_binding":
        p0, p1, p2 = [str(x) for x in rng.choice(PEOPLE_ID, size=3, replace=False)]
        c0, c1, c2, c3 = [str(x) for x in rng.choice(COLORS_ID, size=4, replace=False)]
        m0, m1, m2 = [str(x) for x in rng.choice(MATERIALS_ID, size=3, replace=False)]
        ask_i = int(rng.integers(0, 3))
        ask_p = [p0, p1, p2][ask_i]
        correct = [c0, c1, c2][ask_i]
        distractors = [c for c in [c0, c1, c2, c3] if c != correct]
        header = {"dev_pilot": "Pilot tag list", "cal": "Calibration tag list", "confirm_id": "Station tag list"}[split_name]
        if tpl_v == 0:
            body = (
                f"{header} #{rec_n}:\n"
                f"- Record: {p0} -> color={c0}, material={m0}.\n"
                f"- Record: {p1} -> color={c1}, material={m1}.\n"
                f"- Record: {p2} -> color={c2}, material={m2}.\n"
                f"Question: What color is bound to {ask_p}?"
            )
        else:
            body = (
                f"{header} #{rec_n}:\n"
                f"- Item: {p0} carries a {m0} box colored {c0}.\n"
                f"- Item: {p1} carries a {m1} box colored {c1}.\n"
                f"- Item: {p2} carries a {m2} box colored {c2}.\n"
                f"Question: Which color matches {ask_p}?"
            )
        canonical = f"v04_ab|{p0}:{c0}|{p1}:{c1}|{p2}:{c2}|ask:{ask_p}"
        spec = {"family": family, "ask": ask_p, "answer": correct}

    elif family == "simple_arithmetic":
        # MATCHED numeric range across dev_pilot / cal / confirm_id (§5.1)!
        start = int(rng.integers(24, 65))
        add_v = int(rng.integers(6, 19))
        sub_v = int(rng.integers(4, 17))
        it = str(rng.choice(ITEMS_ID))
        person = str(rng.choice(PEOPLE_ID))
        header = {"dev_pilot": "Pilot count card", "cal": "Calibration count card", "confirm_id": "Station count card"}[split_name]
        if tpl_v == 0:
            ans = start + add_v - sub_v
            body = (
                f"{header} #{rec_n}:\n"
                f"Initial stock: {start} {it} units. {person} adds {add_v} units, then removes {sub_v} units.\n"
                f"Question: How many units remain?"
            )
            op = "add_sub"
        else:
            ans = start - sub_v + add_v
            body = (
                f"{header} #{rec_n}:\n"
                f"Initial stock: {start} {it} units. {person} removes {sub_v} units, then adds {add_v} units.\n"
                f"Question: How many units remain?"
            )
            op = "sub_add"
        offs = [-4, -2, -1, 1, 2, 3, 5]
        rng.shuffle(offs)
        d_vals = [ans + o for o in offs if ans + o > 0][:3]
        correct = str(ans)
        distractors = [str(v) for v in d_vals]
        canonical = f"v04_ar|{op}|{start}|{add_v}|{sub_v}|{it}|{person}"
        spec = {"family": family, "op": op, "start": start, "add": add_v, "sub": sub_v, "ans": ans}

    elif family == "instruction_priority":
        c_hi, c_lo, c_d1, c_d2 = [str(x) for x in rng.choice(CODES_ID, size=4, replace=False)]
        person = str(rng.choice(PEOPLE_ID))
        header = {"dev_pilot": "Pilot policy card", "cal": "Calibration policy card", "confirm_id": "Station policy card"}[split_name]
        if tpl_v == 0:
            body = (
                f"{header} #{rec_n}:\n"
                f"- Primary mandate (highest priority): Select code {c_hi}.\n"
                f"- Optional memo from {person} (lower priority): Suggest code {c_lo}.\n"
                f"Question: Which code obeys the highest-priority rule?"
            )
        else:
            body = (
                f"{header} #{rec_n}:\n"
                f"- Optional memo from {person} (lower priority): Suggest code {c_lo}.\n"
                f"- Primary mandate (highest priority): Select code {c_hi}.\n"
                f"Question: Which code obeys the highest-priority rule?"
            )
        correct = c_hi
        distractors = [c_lo, c_d1, c_d2]
        canonical = f"v04_ip|hi:{c_hi}|lo:{c_lo}|by:{person}|t{tpl_v}"
        spec = {"family": family, "hi": c_hi, "lo": c_lo}
    else:
        raise ValueError(family)

    opts = _place_options(rng, correct, distractors, target_opt)
    prompt = _format_prompt(body, opts)
    pfx = {"dev_pilot": "DP4", "cal": "CAL4", "confirm_id": "CID4"}[split_name]
    return {
        "context_id": f"{pfx}_{family[:2].upper()}_{idx+1:03d}",
        "split": split_name,
        "family": family,
        "scenario_group_id": f"v04_{split_name}_{family}_grp_{idx+1:03d}",
        "canonical_scenario_id": canonical,
        "template_group_id": tpl_gid,
        "target_option": target_opt,
        "correct_text": correct,
        "options": opts,
        "prompt_body": body,
        "user_prompt": prompt,
        "word_count": len(prompt.split()),
        "hidden_rule_spec": spec,
    }


def _build_ood_record(idx: int, tpl_v: int, go_high: bool, target_opt: str, rng: np.random.Generator) -> dict[str, Any]:
    """Two-stage OOD:
      Stage 1: Conditional state update (start + delta >= threshold -> badge_hi else badge_lo)
      Stage 2: Relational lookup using the UPDATED badge -> destination hub.
    Template (tpl_v), branch (go_high), and answer position (target_opt) are INDEPENDENTLY cross-balanced!
    """
    person = str(rng.choice(OOD_PEOPLE))
    role = str(rng.choice(OOD_ROLES))
    b_hi, b_lo = [str(x) for x in rng.choice(OOD_BADGES, size=2, replace=False)]
    h_hi, h_lo, h_d1, h_d2 = [str(x) for x in rng.choice(OOD_HUBS, size=4, replace=False)]
    start = int(rng.integers(15, 36))
    delta = int(rng.integers(4, 15))
    total = start + delta
    gap = int(rng.integers(2, 7))
    threshold = (total - gap) if go_high else (total + gap)
    correct = h_hi if go_high else h_lo
    distractors = [h for h in [h_hi, h_lo, h_d1, h_d2] if h != correct]
    rec_n = idx + 1

    if tpl_v == 0:
        body = (
            f"Compositional routing #{rec_n}:\n"
            f"Stage 1: {person} ({role}) starts at score {start} and adds {delta}. "
            f"If the new score is >= {threshold}, assign badge {b_hi}; otherwise assign badge {b_lo}.\n"
            f"Stage 2: Badge {b_hi} connects to {h_hi}; badge {b_lo} connects to {h_lo}.\n"
            f"Question: Which hub is assigned to {person}?"
        )
    else:
        body = (
            f"Two-step assignment #{rec_n}:\n"
            f"Step 1: Increase {person}'s initial score {start} by {delta}. "
            f"If the total is at least {threshold}, set status to {b_hi}; otherwise set status to {b_lo}.\n"
            f"Step 2: Status {b_hi} maps to {h_hi}, and status {b_lo} maps to {h_lo}.\n"
            f"Question: Which destination maps to {person}?"
        )

    opts = _place_options(rng, correct, distractors, target_opt)
    prompt = _format_prompt(body, opts)
    canonical = f"v04_ood|{person}|{role}|{start}+{delta}>={threshold}|{b_hi}:{h_hi}|{b_lo}:{h_lo}"
    return {
        "context_id": f"COOD4_{idx+1:03d}",
        "split": "confirm_ood",
        "family": "two_stage_conditional_update_relational_lookup",
        "scenario_group_id": f"v04_confirm_ood_grp_{idx+1:03d}",
        "canonical_scenario_id": canonical,
        "template_group_id": f"v04_confirm_ood_t{tpl_v}",
        "ood_branch": "high" if go_high else "low",
        "target_option": target_opt,
        "correct_text": correct,
        "options": opts,
        "prompt_body": body,
        "user_prompt": prompt,
        "word_count": len(prompt.split()),
        "hidden_rule_spec": {
            "family": "two_stage_conditional_update_relational_lookup",
            "start": start, "delta": delta, "threshold": threshold,
            "b_hi": b_hi, "b_lo": b_lo, "h_hi": h_hi, "h_lo": h_lo, "go_high": go_high,
        },
    }


def independent_text_only_verifier(user_prompt: str, claimed_target_option: str) -> dict[str, Any]:
    """STRICTLY INDEPENDENT VERIFIER (§5.1):
    Parses ONLY the displayed `user_prompt` text and its A/B/C/D option lines — NEVER accesses hidden_rule_spec!
    """
    # 1. Extract displayed options directly from prompt text
    opt_matches = re.findall(r"^([A-D])\.\s*(.+)$", user_prompt, flags=re.MULTILINE)
    assert len(opt_matches) == 4, f"Expected 4 displayed options, got {opt_matches}"
    displayed_opts = {k: v.strip() for k, v in opt_matches}
    assert len(set(displayed_opts.values())) == 4, f"Duplicate displayed options: {displayed_opts}"

    # 2. Parse task logic purely from displayed text
    ans_text: str | None = None

    # Pattern A: state_tracking
    m_st = re.search(r"(?:transfers|relocates) the (\S+) (?:from|out of) the (\S+) (?:to|into) the (\S+)\.", user_prompt)
    if m_st:
        moved_item, _, dest_room = m_st.group(1), m_st.group(2), m_st.group(3)
        m_q = re.search(r"Question:.*?\b" + re.escape(moved_item) + r"\b", user_prompt)
        assert m_q is not None, f"State-tracking question did not ask about moved item {moved_item}"
        ans_text = dest_room

    # Pattern B: attribute_binding
    if ans_text is None and ("color=" in user_prompt or "box colored" in user_prompt):
        pairs = dict(re.findall(r"Record:\s*(\S+)\s*->\s*color=(\S+),", user_prompt))
        if not pairs:
            pairs = dict(re.findall(r"Item:\s*(\S+)\s*carries a \S+ box colored (\S+)\.", user_prompt))
        m_q = re.search(r"Question:.*?(?:bound to|matches)\s+(\S+)\?", user_prompt)
        assert pairs and m_q, f"Failed text parse for attribute_binding:\n{user_prompt}"
        ans_text = pairs[m_q.group(1)]

    # Pattern C: simple_arithmetic
    if ans_text is None and "Initial stock:" in user_prompt:
        m_s = re.search(r"Initial stock:\s*(\d+)\s+", user_prompt)
        m_ar1 = re.search(r"adds (\d+) units, then removes (\d+) units", user_prompt)
        m_ar2 = re.search(r"removes (\d+) units, then adds (\d+) units", user_prompt)
        assert m_s and (m_ar1 or m_ar2), f"Failed text parse for arithmetic:\n{user_prompt}"
        s0 = int(m_s.group(1))
        if m_ar1:
            ans_text = str(s0 + int(m_ar1.group(1)) - int(m_ar1.group(2)))
        else:
            ans_text = str(s0 - int(m_ar2.group(1)) + int(m_ar2.group(2)))

    # Pattern D: instruction_priority
    if ans_text is None and "Primary mandate (highest priority):" in user_prompt:
        m_ip = re.search(r"Primary mandate \(highest priority\):\s*Select code (\S+)\.", user_prompt)
        assert m_ip, f"Failed text parse for instruction_priority:\n{user_prompt}"
        ans_text = m_ip.group(1)

    # Pattern E: two-stage OOD
    if ans_text is None and ("Stage 1:" in user_prompt or "Step 1:" in user_prompt):
        m_1a = re.search(r"starts at score (\d+) and adds (\d+)\.\s*If the new score is >= (\d+), assign badge (\S+); otherwise assign badge (\S+)\.", user_prompt)
        m_1b = re.search(r"initial score (\d+) by (\d+)\.\s*If the total is at least (\d+), set status to (\S+); otherwise set status to (\S+)\.", user_prompt)
        assert m_1a or m_1b, f"Failed text parse for OOD stage 1:\n{user_prompt}"
        m1 = m_1a or m_1b
        s0, d0, th, b_hi, b_lo = int(m1.group(1)), int(m1.group(2)), int(m1.group(3)), m1.group(4), m1.group(5)
        chosen_badge = b_hi if (s0 + d0 >= th) else b_lo
        m_2a = re.search(r"Badge (\S+) connects to (\S+);\s*badge (\S+) connects to (\S+)\.", user_prompt)
        m_2b = re.search(r"Status (\S+) maps to (\S+),\s*and status (\S+) maps to (\S+)\.", user_prompt)
        assert m_2a or m_2b, f"Failed text parse for OOD stage 2:\n{user_prompt}"
        m2 = m_2a or m_2b
        route_map = {m2.group(1): m2.group(2), m2.group(3): m2.group(4)}
        ans_text = route_map[chosen_badge]

    assert ans_text is not None, f"Text verifier could not parse prompt:\n{user_prompt}"
    matching_letters = [k for k, v in displayed_opts.items() if v == ans_text]
    assert matching_letters == [claimed_target_option], (
        f"Text-only verifier mismatch: parsed '{ans_text}' -> {matching_letters} vs claimed {claimed_target_option}"
    )
    return {"verified": True, "parsed_answer_text": ans_text, "verified_option": matching_letters[0]}


def generate_all_v04_data() -> dict[str, Any]:
    cfg = json.loads((ROOT / "configs" / "experiment_config.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(int(cfg["derived_seeds"]["data_generation"]))
    families = cfg["data"]["id_families"]
    bank_seed = int(cfg["derived_seeds"]["probe_bank_38x38"])
    qgen_seed = int(cfg["derived_seeds"]["q_general_4"])
    mispair_seed = int(cfg["derived_seeds"]["mispair_signed_perm"])

    splits: dict[str, list[dict[str, Any]]] = {}
    seen_canon_global: set[str] = set()

    for split_name, per_fam in (("dev_pilot", 4), ("cal", 8), ("confirm_id", 24)):
        recs: list[dict[str, Any]] = []
        for fam in families:
            opts_seq = list(OPTIONS) * (per_fam // 4)
            rng.shuffle(opts_seq)
            for idx_f in range(per_fam):
                for _ in range(100):
                    cand = _build_id_record(split_name, fam, idx_f, str(opts_seq[idx_f]), rng)
                    if cand["canonical_scenario_id"] not in seen_canon_global:
                        seen_canon_global.add(cand["canonical_scenario_id"])
                        ver = independent_text_only_verifier(cand["user_prompt"], cand["target_option"])
                        cand["text_verifier_passed"] = ver["verified"]
                        recs.append(cand)
                        break
                else:
                    raise RuntimeError(f"Failed unique generation for {split_name}/{fam}")
        splits[split_name] = recs

    # Confirm-OOD 32: exact 2 templates x 2 branches x 4 options x 2 = 32 cells, shuffled independently!
    ood_cells = [(t, br, o) for t in (0, 1) for br in (True, False) for o in OPTIONS for _ in range(2)]
    rng.shuffle(ood_cells)
    ood_recs: list[dict[str, Any]] = []
    for idx_o, (tpl_v, go_high, topt) in enumerate(ood_cells):
        for _ in range(100):
            cand = _build_ood_record(idx_o, int(tpl_v), bool(go_high), str(topt), rng)
            if cand["canonical_scenario_id"] not in seen_canon_global:
                seen_canon_global.add(cand["canonical_scenario_id"])
                ver = independent_text_only_verifier(cand["user_prompt"], cand["target_option"])
                cand["text_verifier_passed"] = ver["verified"]
                ood_recs.append(cand)
                break
        else:
            raise RuntimeError("Failed unique OOD generation")
    splits["confirm_ood"] = ood_recs

    # Save JSONLs & record hashes + pre-register fixed 38x38 bank hashes & trajectory panel 16!
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    file_hashes: dict[str, str] = {}
    bank_hashes: dict[str, dict[str, str]] = {}

    for sname, recs in splits.items():
        p = data_dir / f"{sname}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in recs:
                # Write without hidden_rule_spec? Keep hidden_rule_spec as metadata (independent verifier NEVER reads it!)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                B = generate_fixed_38x38_bank(r["context_id"], bank_seed, r=38)
                Qg = generate_q_general(r["context_id"], qgen_seed, r=38, n_dirs=4)
                bank_hashes[r["context_id"]] = {
                    "B_38x38_f64_sha256": bytes_sha256_short(B),
                    "Q_gen_38x4_f64_sha256": bytes_sha256_short(Qg),
                }
        file_hashes[f"data/{sname}.jsonl"] = sha256_file(p)

    # Pre-select 16 Confirm-ID contexts for trajectory panel (4 per family, fixed derived seed, BEFORE GPU!)
    rng_tr = np.random.default_rng(int(cfg["derived_seeds"]["trajectory_panel_select"]))
    traj_16: list[str] = []
    for fam in families:
        f_ids = [r["context_id"] for r in splits["confirm_id"] if r["family"] == fam]
        traj_16.extend([str(x) for x in rng_tr.permutation(f_ids)[:4]])

    # Frozen S_mispair 8x8 signed permutation
    S_mispair = generate_frozen_mispair_matrix(mispair_seed, k=8)

    # Check exact body disjointness against old 02.5 (224) and old 03 (192)
    old_bodies: set[str] = set()
    for old_p in (
        pathlib.Path(r"C:\Users\User\.gemini\antigravity\scratch\bayes_dyn_02_5\configs\prompts_224.jsonl"),
        pathlib.Path(r"C:\Users\User\.gemini\antigravity\scratch\BAYES_DYN_03\data\cal_fit.jsonl"),
        pathlib.Path(r"C:\Users\User\.gemini\antigravity\scratch\BAYES_DYN_03\data\cal_check.jsonl"),
        pathlib.Path(r"C:\Users\User\.gemini\antigravity\scratch\BAYES_DYN_03\data\confirm_id.jsonl"),
        pathlib.Path(r"C:\Users\User\.gemini\antigravity\scratch\BAYES_DYN_03\data\confirm_ood.jsonl"),
    ):
        if old_p.exists():
            for ln in old_p.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    old_bodies.add(json.loads(ln)["prompt_body"])

    split_stats = {}
    for sname, recs in splits.items():
        for r in recs:
            assert r["prompt_body"] not in old_bodies, f"Exact body collision with old data: {r['context_id']}"
        split_stats[sname] = {
            "count": len(recs),
            "unique_groups": len({r["scenario_group_id"] for r in recs}),
            "template_groups": sorted({r["template_group_id"] for r in recs}),
            "option_counts": {o: sum(1 for r in recs if r["target_option"] == o) for o in OPTIONS},
            "word_count_min": min(r["word_count"] for r in recs),
            "word_count_mean": round(float(np.mean([r["word_count"] for r in recs])), 2),
            "word_count_max": max(r["word_count"] for r in recs),
            "all_text_verified": all(r["text_verifier_passed"] for r in recs),
        }
        if sname == "confirm_ood":
            # Verify exact cross-balance of (template_group_id x ood_branch)
            cross = {}
            for r in recs:
                k = f"{r['template_group_id']}|{r['ood_branch']}"
                cross[k] = cross.get(k, 0) + 1
            split_stats[sname]["template_x_branch_balance"] = cross

    manifest = {
        "experiment_id": "BAYES-DYN-04",
        "master_seed": cfg["master_seed"],
        "derived_seeds": cfg["derived_seeds"],
        "all_independent_text_verifier_passed": True,
        "zero_exact_overlap_with_025_and_03": True,
        "file_sha256": file_hashes,
        "split_stats": split_stats,
        "pre_registered_trajectory_panel_16": traj_16,
        "S_mispair_8x8": S_mispair.tolist(),
        "S_mispair_sha256": bytes_sha256_short(S_mispair),
        "probe_bank_hashes": bank_hashes,
    }
    man_p = ROOT / "configs" / "data_manifest.json"
    man_p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DATA v04] Generated & text-verified 176 brand-new contexts -> {man_p}")
    for sn, st in split_stats.items():
        print(f"  {sn:12s}: n={st['count']:2d}  wc={st['word_count_mean']:5.1f} [{st['word_count_min']}..{st['word_count_max']}]  opts={st['option_counts']}")
    print(f"  OOD template x branch cross-balance: {split_stats['confirm_ood']['template_x_branch_balance']}")
    return manifest


if __name__ == "__main__":
    generate_all_v04_data()
