#!/usr/bin/env python3
"""Worked example — every baseline misses the structural error, SIV catches it.

Reference: FOLIO premise P1138 ("Holding companies hold several companies").
  ∀x (HoldingCompany(x) → ∃y (Company(y) ∧ Holds(x, y)))

Attempt log:
  1. Scope swap (∀∃ → ∃∀) — produced an OVERSTRONG candidate; SIV-soft
     recall stayed at 1.0 because every gold-entailment is also entailed
     by the stronger candidate. Documented in `attempt_log` below; the
     headline result then pivots to the second attempt.
  2. Outer quantifier flip (∀x → ∃x) — produced an OVERWEAK candidate.
     SIV-soft recall < 1.0 (positive probe `∀v0.(HC(v0) → ∃v1...)`
     is no longer entailed). All other baselines miss it on different
     axes. This is the headline worked example.

The same predicate set and connective skeleton are preserved, so the
propositional LE metric (Yang-et-al-2024-style truth-table collapse)
returns 1.0, Smatch++ stays high (graph distance near zero), and
BLEU/BERTScore stay high (surface form barely changed). Brunello-LT
(Z3) correctly returns NOT_EQUIVALENT but only as a binary verdict.
SIV's recall < 1.0 with a specific positive probe rejected pinpoints
which quantifier property the candidate violates.

Output: reports/worked_example/worked_example.json
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from experiments.common import (
    load_test_suites,
    score_propositional_le_aligned,
    score_brunello_lt_aligned,
    score_smatchpp,
    score_siv_soft,
)
from siv.fol_utils import normalize_fol_string
from siv.vampire_interface import check_entailment, setup_vampire

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PREMISE_ID = "P1138"
NL = "Holding companies hold several companies."

REFERENCE_FOL_DISPLAY = "∀x (HoldingCompany(x) → ∃y (Company(y) ∧ Holds(x, y)))"

# Attempt 1 (documented as a near-miss): scope swap.
ATTEMPT1_FOL = "exists y.all x.(HoldingCompany(x) -> (Company(y) & Holds(x, y)))"
ATTEMPT1_DESC = (
    "Scope swap (∀∃ → ∃∀): pull the ∃y outside the ∀x. Logically "
    "OVERSTRONG (asserts a single y that every HC holds)."
)

# Attempt 2 (the headline): outer ∀→∃ flip.
CANDIDATE_FOL = "exists x.(HoldingCompany(x) -> exists y.(Company(y) & Holds(x, y)))"
CANDIDATE_DESC = (
    "Outer quantifier flip (∀x → ∃x). Logically OVERWEAK: only requires "
    "ONE x for which the implication holds. Material implication makes "
    "this near-trivially true if any non-HoldingCompany element exists."
)

OUT_DIR = _REPO_ROOT / "reports" / "worked_example"
TEST_SUITES_PATH = _REPO_ROOT / "test_suites" / "test_suites.jsonl"


# ── Inline BLEU / BERTScore (compute_baseline_metrics module is missing) ─

def compute_bleu(candidate: str, reference: str) -> Optional[float]:
    """Sentence-level BLEU; same tokenisation as scripts/generate_candidates.py."""
    try:
        import sacrebleu
    except ImportError:
        return None

    def _tok(fol: str) -> str:
        fol = re.sub(r"([(),&|<>!=\-])", r" \1 ", fol)
        return " ".join(fol.split())

    bleu = sacrebleu.sentence_bleu(_tok(candidate), [_tok(reference)])
    return round(bleu.score / 100.0, 4)


def compute_bertscore(candidate: str, reference: str) -> Optional[float]:
    """BERTScore F1 between FOL strings."""
    try:
        from bert_score import score as bert_score
    except ImportError:
        return None
    P, R, F1 = bert_score(
        [candidate], [reference],
        lang="en", model_type="microsoft/deberta-base-mnli", verbose=False,
    )
    return round(float(F1[0]), 4)




def vampire_label(forward: Optional[bool], reverse: Optional[bool]) -> str:
    """Forward = gold ⊨ candidate. Reverse = candidate ⊨ gold."""
    if forward is None or reverse is None:
        return "VERIFICATION_FAILED"
    if forward and reverse:
        return "EQUIVALENT"
    if forward and not reverse:
        return "OVERWEAK (gold ⊨ candidate, candidate ⊭ gold)"
    if not forward and reverse:
        return "OVERSTRONG (candidate ⊨ gold, gold ⊭ candidate)"
    return "INCOMPATIBLE"


def score_all(reference_fol: str, candidate_fol: str, suite_row: dict) -> dict:
    """Compute every metric on (candidate, reference)."""
    bleu = compute_bleu(candidate_fol, reference_fol)
    bertscore = compute_bertscore(candidate_fol, reference_fol)
    prop_le = score_propositional_le_aligned(candidate_fol, reference_fol, timeout=10)
    z3 = score_brunello_lt_aligned(candidate_fol, reference_fol, timeout=10)
    smatch = score_smatchpp(candidate_fol, reference_fol)
    siv = score_siv_soft(suite_row, candidate_fol, timeout=10, threshold=0.6)
    return {
        "bleu": bleu, "bertscore": bertscore,
        "propositional_le": prop_le,
        "brunello_lt_z3": z3,
        "smatchpp": smatch, "siv_report": siv,
    }


def vampire_pair(reference_fol: str, candidate_fol: str) -> tuple[Optional[bool], Optional[bool]]:
    return (
        check_entailment(reference_fol, candidate_fol, timeout=10),
        check_entailment(candidate_fol, reference_fol, timeout=10),
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_vampire()

    reference_fol = normalize_fol_string(REFERENCE_FOL_DISPLAY)
    logger.info("Reference (normalized): %s", reference_fol)

    suites = load_test_suites(TEST_SUITES_PATH)
    suite_row = suites.get(PREMISE_ID)
    if suite_row is None:
        logger.error("No test suite for %s", PREMISE_ID)
        sys.exit(1)

    # ── Attempt 1: scope swap (documented near-miss) ─────────────────────
    logger.info("\n=== ATTEMPT 1: scope swap (∀∃ → ∃∀) ===")
    a1_norm = normalize_fol_string(ATTEMPT1_FOL)
    a1_fwd, a1_rev = vampire_pair(reference_fol, a1_norm)
    a1_label = vampire_label(a1_fwd, a1_rev)
    a1_metrics = score_all(reference_fol, a1_norm, suite_row)
    a1_recall = a1_metrics["siv_report"].recall if a1_metrics["siv_report"] else None
    a1_precision = a1_metrics["siv_report"].precision if a1_metrics["siv_report"] else None
    logger.info("Vampire: %s", a1_label)
    logger.info("SIV-soft recall=%s precision=%s", a1_recall, a1_precision)
    a1_caught = (a1_recall is not None and a1_recall < 1.0) or (
        a1_precision is not None and a1_precision < 1.0
    )
    if a1_caught:
        logger.info("SIV caught attempt 1 — using it as the headline.")
        candidate_fol = a1_norm
        candidate_desc = ATTEMPT1_DESC
        attempt_used = 1
    else:
        logger.info(
            "Attempt 1 not caught by SIV (OVERSTRONG candidates pass recall by "
            "construction, and the contrastive family is conservative on this "
            "premise). Pivoting to attempt 2."
        )
        candidate_fol = None
        attempt_used = None

    # ── Attempt 2: outer ∀→∃ flip (the headline) ─────────────────────────
    logger.info("\n=== ATTEMPT 2: outer quantifier flip (∀x → ∃x) ===")
    a2_norm = normalize_fol_string(CANDIDATE_FOL)
    a2_fwd, a2_rev = vampire_pair(reference_fol, a2_norm)
    a2_label = vampire_label(a2_fwd, a2_rev)
    a2_metrics = score_all(reference_fol, a2_norm, suite_row)
    a2_recall = a2_metrics["siv_report"].recall if a2_metrics["siv_report"] else None
    a2_precision = a2_metrics["siv_report"].precision if a2_metrics["siv_report"] else None
    logger.info("Vampire: %s", a2_label)
    logger.info("SIV-soft recall=%s precision=%s", a2_recall, a2_precision)
    a2_caught = (a2_recall is not None and a2_recall < 1.0) or (
        a2_precision is not None and a2_precision < 1.0
    )

    if not a1_caught and a2_caught:
        candidate_fol = a2_norm
        candidate_desc = CANDIDATE_DESC
        attempt_used = 2
    elif not a1_caught and not a2_caught:
        logger.error("Neither attempt was caught by SIV. Bail out.")
        sys.exit(1)

    # ── Build the headline record ────────────────────────────────────────
    headline_label = a1_label if attempt_used == 1 else a2_label
    headline_metrics = a1_metrics if attempt_used == 1 else a2_metrics
    headline_fwd = a1_fwd if attempt_used == 1 else a2_fwd
    headline_rev = a1_rev if attempt_used == 1 else a2_rev
    siv_report = headline_metrics["siv_report"]

    per_probe = []
    for r in siv_report.per_test_results:
        passed = (
            (r.kind == "positive" and r.verdict == "entailed")
            or (r.kind == "contrastive" and r.verdict != "entailed")
        )
        per_probe.append({
            "kind": r.kind, "fol": r.fol, "verdict": r.verdict,
            "passed": passed,
            "probe_label": list(r.probe_label) if r.probe_label else None,
            "mutation_kind": r.mutation_kind,
        })

    logger.info("\n=== HEADLINE: attempt %d ===", attempt_used)
    logger.info("Candidate: %s", candidate_fol)
    logger.info("Vampire ground truth: %s", headline_label)
    for k, v in [
        ("BLEU", headline_metrics["bleu"]),
        ("BERTScore", headline_metrics["bertscore"]),
        ("LE (propositional, aligned)", headline_metrics["propositional_le"]),
        ("Brunello-LT (Z3)", headline_metrics["brunello_lt_z3"]),
        ("Smatch++", headline_metrics["smatchpp"]),
    ]:
        logger.info("  %-20s = %s", k, f"{v:.4f}" if v is not None else "n/a")
    logger.info(
        "  %-20s = %s (recall=%.3f, precision=%s, f1=%s)",
        "SIV-soft",
        "FAIL" if (siv_report.recall < 1.0 or (siv_report.precision is not None and siv_report.precision < 1.0)) else "PASS",
        siv_report.recall,
        f"{siv_report.precision:.3f}" if siv_report.precision is not None else "n/a",
        f"{siv_report.f1:.3f}" if siv_report.f1 is not None else "n/a",
    )
    logger.info("\nSIV per-probe trace:")
    for p in per_probe:
        marker = "PASS" if p["passed"] else "FAIL"
        logger.info("  [%s] %-12s %-12s %s", marker, p["kind"], p["verdict"], p["fol"])

    out = {
        "premise_id": PREMISE_ID,
        "natural_language": NL,
        "reference_fol_display": REFERENCE_FOL_DISPLAY,
        "reference_fol_normalized": reference_fol,
        "headline_attempt": attempt_used,
        "headline_candidate": {
            "fol": candidate_fol,
            "perturbation_description": candidate_desc,
            "vampire_ground_truth": {
                "label": headline_label,
                "gold_entails_candidate": headline_fwd,
                "candidate_entails_gold": headline_rev,
            },
            "metric_scores": {
                "bleu": headline_metrics["bleu"],
                "bertscore": headline_metrics["bertscore"],
                "propositional_le_aligned": headline_metrics["propositional_le"],
                "brunello_lt_aligned_z3": headline_metrics["brunello_lt_z3"],
                "smatchpp": headline_metrics["smatchpp"],
                "siv_soft_recall": siv_report.recall,
                "siv_soft_precision": siv_report.precision,
                "siv_soft_f1": siv_report.f1,
            },
            "siv_probe_trace": {
                "positives_entailed": siv_report.positives_entailed,
                "positives_total": siv_report.positives_total,
                "contrastives_rejected": siv_report.contrastives_rejected,
                "contrastives_total": siv_report.contrastives_total,
                "per_probe": per_probe,
            },
        },
        "attempt_log": [
            {
                "attempt": 1,
                "fol": a1_norm,
                "description": ATTEMPT1_DESC,
                "vampire_label": a1_label,
                "metric_scores": {
                    "bleu": a1_metrics["bleu"],
                    "bertscore": a1_metrics["bertscore"],
                    "propositional_le_aligned": a1_metrics["propositional_le"],
                    "brunello_lt_aligned_z3": a1_metrics["brunello_lt_z3"],
                    "smatchpp": a1_metrics["smatchpp"],
                    "siv_soft_recall": a1_recall,
                    "siv_soft_precision": a1_precision,
                },
                "caught_by_siv": a1_caught,
                "outcome": (
                    "Used as headline." if attempt_used == 1
                    else "OVERSTRONG; SIV-soft recall = 1.0 by construction "
                         "(candidate is logically stronger than gold, so it "
                         "entails every gold-derived positive probe). On this "
                         "premise the contrastive family also leaves precision "
                         "at 1.0. Pivoted to attempt 2."
                ),
            },
            {
                "attempt": 2,
                "fol": a2_norm,
                "description": CANDIDATE_DESC,
                "vampire_label": a2_label,
                "metric_scores": {
                    "bleu": a2_metrics["bleu"],
                    "bertscore": a2_metrics["bertscore"],
                    "propositional_le_aligned": a2_metrics["propositional_le"],
                    "brunello_lt_aligned_z3": a2_metrics["brunello_lt_z3"],
                    "smatchpp": a2_metrics["smatchpp"],
                    "siv_soft_recall": a2_recall,
                    "siv_soft_precision": a2_precision,
                },
                "caught_by_siv": a2_caught,
                "outcome": "Used as headline." if attempt_used == 2 else "Unused.",
            },
        ],
        "interpretation_one_liner": (
            "Each baseline misses the quantifier flip on a different axis: "
            "BLEU/BERTScore react to surface form only (one token "
            "changed); Smatch++ preserves graph similarity (the only "
            "difference is the outer quantifier label); propositional LE "
            "(the Yang-et-al-2024 style metric implemented in "
            "siv.propositional_le) returns 1.0 because stripping the "
            "outer quantifier leaves the same propositional formula on "
            "both sides; Brunello-LT (Z3) correctly returns 0.0 but only "
            "as a binary verdict. SIV's recall < 1.0 with the gold "
            "positive probe rejected pinpoints exactly which quantifier "
            "property the candidate violates."
        ),
    }

    out_path = OUT_DIR / "worked_example.json"
    out_path.write_text(json.dumps(out, indent=2, default=str) + "\n")
    logger.info("\nWrote %s", out_path)


if __name__ == "__main__":
    main()
