#!/usr/bin/env python3
"""Reference-robustness analysis — SIV's reference-free advantage.

Auxiliary analysis (not one of the four primary paper experiments):
validates the severity-correlation headline ρ against FOLIO reference
errors. Olausson et al. report ~11% errors in FOLIO references;
hand-corrected pool to verify headline stability under correction.

Orchestrator with --step {1,2,3,4,5,6}.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from experiments.common import (
    load_test_suites,
    score_bleu,
    score_bertscore,
    score_propositional_le_raw,
    score_propositional_le_aligned,
    score_brunello_lt_raw,
    score_brunello_lt_aligned,
    score_siv_soft,
)
from siv.fol_utils import parse_fol, free_individual_variables, is_valid_fol

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXP3_DIR = _REPO_ROOT / "reports" / "experiments" / "reference_robustness"
TEST_SUITES_PATH = _REPO_ROOT / "test_suites" / "test_suites.jsonl"


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — Identify broken-gold premises
# ═══════════════════════════════════════════════════════════════════════════

def step1_broken_pool():
    """Build the broken-gold pool from per-premise evidence."""
    logger.info("Step 1: Identifying broken-gold premises")

    pool = []
    with open(TEST_SUITES_PATH) as f:
        for line in f:
            row = json.loads(line)
            gold = row.get("gold_fol", "")
            pid = row["premise_id"]
            story_id = row.get("story_id")
            nl = row.get("nl", "")

            if not gold:
                continue

            # (a) gold_fol fails parse_fol
            expr = parse_fol(gold)
            if expr is None:
                pool.append({
                    "premise_id": pid,
                    "story_id": story_id,
                    "nl": nl,
                    "gold_fol": gold,
                    "broken_reason": "syntax_error",
                    "broken_evidence": f"parse_fol returns None (hyphenated predicates, XOR operator, or unbalanced parens)",
                })
                continue

            # (b) gold_fol contains free individual variables
            fv = free_individual_variables(gold)
            if fv:
                pool.append({
                    "premise_id": pid,
                    "story_id": story_id,
                    "nl": nl,
                    "gold_fol": gold,
                    "broken_reason": "free_variable",
                    "broken_evidence": f"free individual variables: {sorted(fv)}",
                })

    EXP3_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXP3_DIR / "broken_gold_pool.jsonl"
    with open(out_path, "w") as f:
        for row in pool:
            f.write(json.dumps(row) + "\n")

    reasons = Counter(r["broken_reason"] for r in pool)
    logger.info("Broken-gold pool: %d premises", len(pool))
    logger.info("  By reason: %s", dict(reasons))
    logger.info("Wrote %s", out_path)

    # Update metadata
    _update_metadata("step1", {
        "pool_size": len(pool),
        "by_reason": dict(reasons),
        "criteria_used": ["syntax_error (parse_fol fails)", "free_variable (free_individual_variables non-empty)"],
        "criterion_c_note": "Criterion (c) 'unprovable_by_design' yielded 0 premises with per-premise evidence only. failure_analysis.json contains story-level buckets, not premise-level. Per spec, story-level heuristics not used.",
    })


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Select premises for hand-authoring
# ═══════════════════════════════════════════════════════════════════════════

def step2_select():
    """Select 30 premises from the broken-gold pool using deterministic filters."""
    logger.info("Step 2: Selecting premises for hand-authoring")

    pool_path = EXP3_DIR / "broken_gold_pool.jsonl"
    if not pool_path.exists():
        logger.error("Run step 1 first")
        sys.exit(1)

    pool = []
    for line in pool_path.read_text().splitlines():
        if line.strip():
            pool.append(json.loads(line))
    logger.info("Pool size: %d", len(pool))

    # Load test suites for filter checks
    suites = load_test_suites(TEST_SUITES_PATH)

    # Apply filters
    survivors = []
    filter_drops = Counter()

    for premise in pool:
        pid = premise["premise_id"]
        suite_row = suites.get(pid, {})

        # Filter 2: SIV extraction succeeded
        canonical = suite_row.get("canonical_fol", "")
        if not canonical:
            filter_drops["no_extraction"] += 1
            continue

        # Filter 3: >=2 positive probes
        n_pos = len(suite_row.get("positives") or [])
        if n_pos < 2:
            filter_drops["lt2_positives"] += 1
            continue

        # Filter 1: NL is unambiguous — <=30 words
        nl = premise["nl"]
        word_count = len(nl.split())
        if word_count > 30:
            filter_drops["gt30_words"] += 1
            continue

        # Filter 4: Skipped — SIV-soft handles vocabulary alignment at scoring time.
        # The human author will write c_correct in appropriate vocabulary.
        # Documented as deviation from spec.

        survivors.append({
            **premise,
            "n_positives": n_pos,
            "n_contrastives": len(suite_row.get("contrastives") or []),
            "siv_canonical_fol": canonical,
        })

    logger.info("After filters: %d (drops: %s)", len(survivors), dict(filter_drops))

    if len(survivors) < 20:
        logger.error("ABORT: fewer than 20 premises pass filters (%d)", len(survivors))
        sys.exit(1)

    # Sort by story_id (deterministic)
    survivors.sort(key=lambda x: (x["story_id"] or 0, x["premise_id"]))

    # Take first 30 (or all if fewer)
    target = min(30, len(survivors))
    selected = survivors[:target]

    # Assign selection order
    for i, s in enumerate(selected, 1):
        s["selection_order"] = i

    # Write output
    out_path = EXP3_DIR / "selected_premises.jsonl"
    with open(out_path, "w") as f:
        for row in selected:
            f.write(json.dumps(row) + "\n")

    reasons = Counter(s["broken_reason"] for s in selected)
    logger.info("Selected %d premises", len(selected))
    logger.info("  By reason: %s", dict(reasons))
    logger.info("Wrote %s", out_path)

    # Stratification check
    max_pct = max(reasons.values()) / len(selected) if selected else 0
    strat_ok = max_pct <= 0.5

    deviations = []
    if not strat_ok:
        deviations.append(
            f"Stratification: max bucket is {max(reasons.values())}/{len(selected)} "
            f"({max_pct:.0%}) from '{max(reasons, key=reasons.get)}'. "
            f"Exceeds 50% because criterion (c) yielded no per-premise cases, "
            f"leaving only 2 buckets."
        )
    if len(selected) < 30:
        deviations.append(
            f"Only {len(selected)} premises selected (target 30, minimum 20)."
        )

    _update_metadata("step2", {
        "pool_size": len(pool),
        "filter_drops": dict(filter_drops),
        "survivors_after_filters": len(survivors),
        "premises_selected": len(selected),
        "by_reason": dict(reasons),
        "stratification_ok": strat_ok,
        "deviations": deviations,
        "filter_4_note": "Filter 4 (predicate overlap) skipped — SIV-soft alignment handles vocabulary at scoring time.",
    })


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 — Generate authoring template
# ═══════════════════════════════════════════════════════════════════════════

def step3_template():
    """Generate the corrections template for human authoring."""
    logger.info("Step 3: Generating authoring template")

    selected_path = EXP3_DIR / "selected_premises.jsonl"
    if not selected_path.exists():
        logger.error("Run step 2 first")
        sys.exit(1)

    premises = []
    for line in selected_path.read_text().splitlines():
        if line.strip():
            premises.append(json.loads(line))

    suites = load_test_suites(TEST_SUITES_PATH)

    lines = ["# Experiment 3 — Corrections Template\n"]
    lines.append("Fill in `c_correct_fol` and `rationale` for each premise below.\n")
    lines.append("Use the same FOL convention as the SIV canonical: all/exists for quantifiers,")
    lines.append("-> for implication, & for conjunction, | for disjunction, - for negation.\n")
    lines.append("The correction must be MORE FAITHFUL to the NL than the broken gold.\n")
    lines.append("---\n")

    for premise in premises:
        pid = premise["premise_id"]
        suite_row = suites.get(pid, {})
        ej = suite_row.get("extraction_json", {})
        preds = ej.get("predicates", [])
        constants = ej.get("constants", [])

        pred_sigs = ", ".join(f"{p['name']}/{p['arity']}" for p in preds)
        const_list = ", ".join(c["id"] for c in constants) if constants else "(none)"

        lines.append(f"\n## {pid}\n")
        lines.append(f'NL: "{premise["nl"]}"\n')
        lines.append(f"Gold FOL (broken): {premise['gold_fol']}\n")
        lines.append(f"Broken reason: {premise['broken_reason']}")
        lines.append(f"Broken evidence: {premise['broken_evidence']}\n")
        lines.append(f"SIV canonical: {premise.get('siv_canonical_fol', '')}\n")
        lines.append(f"Predicate vocabulary (SIV): {pred_sigs}")
        lines.append(f"Constants (SIV): {const_list}\n")
        lines.append("### Your correction:")
        lines.append("c_correct_fol: <FILL IN>")
        lines.append("rationale: <FILL IN — one sentence, why this is more faithful to NL than gold>")
        lines.append("introduced_predicates: none")
        lines.append("")

    template_path = EXP3_DIR / "corrections_template.md"
    template_path.write_text("\n".join(lines))
    logger.info("Wrote template to %s (%d premises)", template_path, len(premises))
    logger.info("HAND OFF TO HUMAN: fill in corrections_template.md")


def step3_parse():
    """Parse the filled corrections template back into corrections.jsonl."""
    logger.info("Step 3 (parse): Reading filled corrections template")

    template_path = EXP3_DIR / "corrections_template.md"
    if not template_path.exists():
        logger.error("corrections_template.md not found")
        sys.exit(1)

    text = template_path.read_text()
    import re

    # Parse sections
    sections = re.split(r"\n## (P\d+)\n", text)
    corrections = []
    parse_failures = []

    # sections[0] is preamble, then alternating (pid, content)
    for i in range(1, len(sections), 2):
        pid = sections[i]
        content = sections[i + 1] if i + 1 < len(sections) else ""

        # Extract fields
        nl_match = re.search(r'NL: "(.+?)"', content)
        gold_match = re.search(r"Gold FOL \(broken\): (.+?)(?:\n|$)", content)
        reason_match = re.search(r"Broken reason: (\w+)", content)
        correct_match = re.search(r"c_correct_fol: (.+?)(?:\n|$)", content)
        rationale_match = re.search(r"rationale: (.+?)(?:\n|$)", content)
        intro_match = re.search(r"introduced_predicates: (.+?)(?:\n|$)", content)

        if not correct_match or "<FILL IN>" in correct_match.group(1):
            logger.warning("  %s: not yet filled in, skipping", pid)
            continue

        c_correct = correct_match.group(1).strip()
        rationale = rationale_match.group(1).strip() if rationale_match else ""
        intro_preds = intro_match.group(1).strip() if intro_match else "none"

        # Parse introduced_predicates
        if intro_preds.lower() in ("none", ""):
            intro_list = []
        else:
            intro_list = [p.strip() for p in intro_preds.split(",")]

        # Validate
        parses = parse_fol(c_correct) is not None
        valid = is_valid_fol(c_correct)

        if not parses:
            parse_failures.append(pid)
            logger.warning("  %s: c_correct_fol DOES NOT PARSE: %s", pid, c_correct[:60])

        corrections.append({
            "premise_id": pid,
            "nl": nl_match.group(1) if nl_match else "",
            "c_gold_fol": gold_match.group(1).strip() if gold_match else "",
            "c_correct_fol": c_correct,
            "rationale": rationale,
            "introduced_predicates": intro_list,
            "broken_reason": reason_match.group(1) if reason_match else "",
            "is_valid_fol": valid,
            "parses": parses,
        })

    # Write output
    out_path = EXP3_DIR / "corrections.jsonl"
    with open(out_path, "w") as f:
        for row in corrections:
            f.write(json.dumps(row) + "\n")

    logger.info("Parsed %d corrections, %d parse failures", len(corrections), len(parse_failures))
    if parse_failures:
        logger.error("PARSE FAILURES — must be fixed before step 4: %s", parse_failures)
    logger.info("Wrote %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════
# Step 4 — Scoring
# ═══════════════════════════════════════════════════════════════════════════

def step4_scoring():
    """Score c_gold and c_correct with all metrics."""
    logger.info("Step 4: Scoring")

    corrections_path = EXP3_DIR / "corrections.jsonl"
    if not corrections_path.exists():
        logger.error("Run step 3 (parse) first — corrections.jsonl not found")
        sys.exit(1)

    corrections = []
    for line in corrections_path.read_text().splitlines():
        if line.strip():
            corrections.append(json.loads(line))

    # Check all corrections parse
    unparsed = [c for c in corrections if not c["parses"]]
    if unparsed:
        logger.error("Cannot score: %d corrections don't parse. Fix first.", len(unparsed))
        sys.exit(1)

    suites = load_test_suites(TEST_SUITES_PATH)
    scored = []
    start = time.time()

    for corr in corrections:
        pid = corr["premise_id"]
        c_gold = corr["c_gold_fol"]
        c_correct = corr["c_correct_fol"]
        suite_row = suites.get(pid, {})

        if not suite_row:
            logger.warning("  %s: no test suite found, skipping", pid)
            continue

        # Score c_gold
        gold_scores = _score_exp3(suite_row, c_gold, c_gold)
        scored.append({
            "premise_id": pid,
            "candidate_label": "c_gold",
            "candidate_fol": c_gold,
            "scores": gold_scores,
        })

        # Score c_correct (reference for ref-based metrics is c_gold)
        correct_scores = _score_exp3(suite_row, c_correct, c_gold)
        scored.append({
            "premise_id": pid,
            "candidate_label": "c_correct",
            "candidate_fol": c_correct,
            "scores": correct_scores,
        })

    elapsed = time.time() - start
    logger.info("Scored %d rows in %.1fs", len(scored), elapsed)

    out_path = EXP3_DIR / "scored_candidates.jsonl"
    with open(out_path, "w") as f:
        for row in scored:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %s", out_path)

    _update_metadata("step4", {
        "scored_rows": len(scored),
        "premises_scored": len(corrections),
        "wall_time_s": round(elapsed, 1),
    })


def _score_exp3(suite_row: dict, candidate_fol: str, reference_fol: str) -> dict:
    """Score a candidate for Exp 3 (reference-based use reference_fol, SIV uses NL test suite)."""
    scores = {}

    # Reference-based (using reference_fol = c_gold as reference)
    scores["bleu_vs_gold"] = score_bleu(candidate_fol, reference_fol)
    scores["bertscore_vs_gold"] = score_bertscore(candidate_fol, reference_fol)
    scores["propositional_le_raw_vs_gold"] = score_propositional_le_raw(candidate_fol, reference_fol, timeout=10)
    scores["propositional_le_aligned_vs_gold"] = score_propositional_le_aligned(candidate_fol, reference_fol, timeout=10)
    scores["brunello_lt_raw_vs_gold"] = score_brunello_lt_raw(candidate_fol, reference_fol, timeout=10)
    scores["brunello_lt_aligned_vs_gold"] = score_brunello_lt_aligned(candidate_fol, reference_fol, timeout=10)

    # SIV (reference-free — uses NL via test suite)
    siv_report = score_siv_soft(suite_row, candidate_fol, timeout=10, threshold=0.6)
    if siv_report:
        scores["siv_soft_recall"] = siv_report.recall
        scores["siv_soft_precision"] = siv_report.precision
        scores["siv_soft_f1"] = siv_report.f1
    else:
        scores["siv_soft_recall"] = None
        scores["siv_soft_precision"] = None
        scores["siv_soft_f1"] = None

    return scores


# ═══════════════════════════════════════════════════════════════════════════
# Step 5 — Analysis
# ═══════════════════════════════════════════════════════════════════════════

def step5_analysis():
    """Produce tables and figures for Exp 3."""
    import numpy as np

    logger.info("Step 5: Analysis")

    scored_path = EXP3_DIR / "scored_candidates.jsonl"
    if not scored_path.exists():
        logger.error("Run step 4 first")
        sys.exit(1)

    scored = []
    for line in scored_path.read_text().splitlines():
        if line.strip():
            scored.append(json.loads(line))

    # Group by premise
    from collections import defaultdict
    by_premise = defaultdict(dict)
    for row in scored:
        by_premise[row["premise_id"]][row["candidate_label"]] = row["scores"]

    metrics_ref = ["bleu_vs_gold", "bertscore_vs_gold", "propositional_le_aligned_vs_gold", "brunello_lt_aligned_vs_gold"]
    metrics_siv = ["siv_soft_recall"]
    all_metrics = metrics_ref + metrics_siv

    # ── Table 3.5a — Mean scores ──
    table_a = {}
    for label in ["c_gold", "c_correct"]:
        table_a[label] = {}
        for m in all_metrics:
            vals = [by_premise[pid][label].get(m) for pid in by_premise
                    if label in by_premise[pid] and by_premise[pid][label].get(m) is not None]
            if vals:
                arr = np.array(vals)
                rng = np.random.RandomState(42)
                boot = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(1000)]
                table_a[label][m] = {
                    "mean": round(float(arr.mean()), 4),
                    "ci_lo": round(float(np.percentile(boot, 2.5)), 4),
                    "ci_hi": round(float(np.percentile(boot, 97.5)), 4),
                    "n": len(vals),
                }
            else:
                table_a[label][m] = {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0}

    csv_path = EXP3_DIR / "score_by_candidate.csv"
    with open(csv_path, "w") as f:
        f.write("candidate_label," + ",".join(f"{m}_mean,{m}_ci_lo,{m}_ci_hi" for m in all_metrics) + "\n")
        for label in ["c_gold", "c_correct"]:
            parts = [label]
            for m in all_metrics:
                d = table_a[label][m]
                parts.extend([
                    str(d["mean"]) if d["mean"] is not None else "",
                    str(d["ci_lo"]) if d["ci_lo"] is not None else "",
                    str(d["ci_hi"]) if d["ci_hi"] is not None else "",
                ])
            f.write(",".join(parts) + "\n")
    logger.info("Table 3.5a written to %s", csv_path)

    # ── Table 3.5b — Inversion rate ──
    inversion = {}
    for m in metrics_ref:
        inversions = 0
        total = 0
        for pid in by_premise:
            if "c_gold" in by_premise[pid] and "c_correct" in by_premise[pid]:
                g = by_premise[pid]["c_gold"].get(m)
                c = by_premise[pid]["c_correct"].get(m)
                if g is not None and c is not None:
                    total += 1
                    if g > c:
                        inversions += 1
        inversion[m] = {"rate": round(inversions / total, 4) if total else None, "n": total}

    inv_path = EXP3_DIR / "inversion_rate.json"
    inv_path.write_text(json.dumps(inversion, indent=2) + "\n")
    logger.info("Table 3.5b (inversion rate): %s",
                {m: d["rate"] for m, d in inversion.items()})

    # ── Table 3.5c — Correct-preference rate (HEADLINE) ──
    pref_results = {}

    # SIV preference
    siv_higher = 0
    siv_strict_higher = 0
    siv_tied = 0
    siv_lower = 0
    siv_total = 0

    for pid in by_premise:
        if "c_gold" in by_premise[pid] and "c_correct" in by_premise[pid]:
            g = by_premise[pid]["c_gold"].get("siv_soft_recall")
            c = by_premise[pid]["c_correct"].get("siv_soft_recall")
            if g is not None and c is not None:
                siv_total += 1
                if c > g:
                    siv_higher += 1
                    if c - g >= 0.1:
                        siv_strict_higher += 1
                elif abs(c - g) < 0.05:
                    siv_tied += 1
                else:
                    siv_lower += 1

    pref_results["siv_soft_recall"] = {
        "correct_preference_rate": round(siv_higher / siv_total, 4) if siv_total else None,
        "strict_higher_rate": round(siv_strict_higher / siv_total, 4) if siv_total else None,
        "tied_rate": round(siv_tied / siv_total, 4) if siv_total else None,
        "lower_rate": round(siv_lower / siv_total, 4) if siv_total else None,
        "n": siv_total,
    }

    # Equivalence-based preference (should be ~0%)
    for m in ["propositional_le_aligned_vs_gold", "brunello_lt_aligned_vs_gold"]:
        higher = 0
        total = 0
        for pid in by_premise:
            if "c_gold" in by_premise[pid] and "c_correct" in by_premise[pid]:
                g = by_premise[pid]["c_gold"].get(m)
                c = by_premise[pid]["c_correct"].get(m)
                if g is not None and c is not None:
                    total += 1
                    if c > g:
                        higher += 1
        pref_results[m] = {
            "correct_preference_rate": round(higher / total, 4) if total else None,
            "n": total,
        }

    pref_path = EXP3_DIR / "correct_preference.json"
    pref_path.write_text(json.dumps(pref_results, indent=2) + "\n")

    logger.info("\n=== HEADLINE: Correct-Preference Rate ===")
    siv_pref = pref_results["siv_soft_recall"]
    logger.info("  SIV correct_preference: %s", siv_pref["correct_preference_rate"])
    logger.info("  SIV strict_higher (margin>=0.1): %s", siv_pref["strict_higher_rate"])
    logger.info("  SIV tied (<0.05): %s", siv_pref["tied_rate"])
    logger.info("  SIV lower (c_gold wins): %s", siv_pref["lower_rate"])
    logger.info("  LE pref c_correct: %s", pref_results["propositional_le_aligned_vs_gold"]["correct_preference_rate"])
    logger.info("  Brunello pref c_correct: %s", pref_results["brunello_lt_aligned_vs_gold"]["correct_preference_rate"])

    # ── Figure 3.5d ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(all_metrics), figsize=(4 * len(all_metrics), 5))
        if len(all_metrics) == 1:
            axes = [axes]

        for ax, m in zip(axes, all_metrics):
            gold_vals = [by_premise[pid]["c_gold"].get(m) for pid in by_premise
                         if "c_gold" in by_premise[pid] and by_premise[pid]["c_gold"].get(m) is not None]
            corr_vals = [by_premise[pid]["c_correct"].get(m) for pid in by_premise
                         if "c_correct" in by_premise[pid] and by_premise[pid]["c_correct"].get(m) is not None]

            data = [gold_vals, corr_vals] if gold_vals or corr_vals else [[0], [0]]
            bp = ax.boxplot(data, labels=["c_gold", "c_correct"], patch_artist=True)
            bp["boxes"][0].set_facecolor("#e74c3c")
            bp["boxes"][0].set_alpha(0.7)
            bp["boxes"][1].set_facecolor("#2ecc71")
            bp["boxes"][1].set_alpha(0.7)

            ax.set_title(m.replace("_vs_gold", "").replace("_", " ").title(), fontsize=9)
            ax.set_ylabel("Score")
            ax.set_ylim(-0.05, 1.05)

        plt.suptitle("Exp 3: Broken Gold vs Corrected — Score Distributions", fontsize=12)
        plt.tight_layout()
        fig_path = EXP3_DIR / "score_distributions.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Figure written to %s", fig_path)
    except ImportError:
        logger.warning("matplotlib not available, skipping figure")

    # ── Acceptance check ──
    logger.info("\n=== ACCEPTANCE CHECK ===")
    all_inv_ok = all(d["rate"] is not None and d["rate"] >= 0.9 for d in inversion.values())
    logger.info("  Inversion rate >= 90%% for all ref-based: %s", all_inv_ok)
    logger.info("  SIV correct_preference >= 70%%: %s",
                siv_pref["correct_preference_rate"] is not None and siv_pref["correct_preference_rate"] >= 0.7)
    logger.info("  SIV strict_higher >= 50%%: %s",
                siv_pref["strict_higher_rate"] is not None and siv_pref["strict_higher_rate"] >= 0.5)

    equiv_pref_ok = all(
        pref_results[m]["correct_preference_rate"] is not None and pref_results[m]["correct_preference_rate"] <= 0.2
        for m in ["propositional_le_aligned_vs_gold", "brunello_lt_aligned_vs_gold"]
    )
    logger.info("  Equiv metrics pref c_correct <= 20%%: %s", equiv_pref_ok)

    _update_metadata("step5", {
        "inversion_rates": {m: d["rate"] for m, d in inversion.items()},
        "siv_preference": pref_results["siv_soft_recall"],
        "equiv_preference": {m: pref_results[m] for m in ["propositional_le_aligned_vs_gold", "brunello_lt_aligned_vs_gold"]},
        "acceptance": {
            "inversion_ok": all_inv_ok,
            "siv_pref_ok": siv_pref["correct_preference_rate"] is not None and siv_pref["correct_preference_rate"] >= 0.7,
            "siv_strict_ok": siv_pref["strict_higher_rate"] is not None and siv_pref["strict_higher_rate"] >= 0.5,
            "equiv_pref_ok": equiv_pref_ok,
        },
    })


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def _update_metadata(key: str, data: dict):
    meta_path = EXP3_DIR / "run_metadata.json"
    existing = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    existing[key] = data
    meta_path.write_text(json.dumps(existing, indent=2) + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Experiment 3 — Reference Failure")
    parser.add_argument("--step", type=str, required=True,
                        choices=["1", "2", "3-template", "3-parse", "4", "5", "6"])
    args = parser.parse_args()

    if args.step == "1":
        step1_broken_pool()
    elif args.step == "2":
        step2_select()
    elif args.step == "3-template":
        step3_template()
    elif args.step == "3-parse":
        step3_parse()
    elif args.step == "4":
        step4_scoring()
    elif args.step == "5":
        step5_analysis()
    elif args.step == "6":
        logger.info("Step 6 (error analysis) is manual — see siv_losses.md")


if __name__ == "__main__":
    main()
