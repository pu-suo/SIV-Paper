"""Diagnostic-trace experiment (paper Exp 3) — recoverability of the
perturbation class from a labeled SIV trace.

Built on the frozen 427-predicate asymmetry-axiom label table at
``siv/asymmetry_axioms.py``, applied to a rescore of the
perturbation-detection (Exp 2) pool — produced by this script's
``rescore`` step.

Pipeline (each step writes an artifact under
``reports/experiments/diagnostic_trace/``):

  --step rescore  : rescore the PD reference/perturbed pool under the
                    asymmetry-axiom regime. Reuses the frozen PD pool's
                    (reference, perturbed) pairs; only re-runs SIV (BLEU,
                    BERTScore, Smatch++, LE, Brunello-LT scores are
                    re-used from the PD bundle). Outputs:
                    pd_rescore/{scored.jsonl, diff_vs_v2.json,
                                run_metadata.json, rescore.log}.
  --step extract  : extract per-pair trace features (failed_positives +
                    fired_contrastives + positive_recall + siv_f1) from
                    the rescored PD pool against freshly compiled
                    label-carrying SIV suites. Outputs:
                    trace_features.jsonl, extraction.log.
  --step classify : apply the frozen 7-rule classifier and the
                    score-only baseline; emit metrics + predictions +
                    confusion matrices + firing-recall sanity. Outputs:
                    metrics.json, predictions.jsonl,
                    contrastive_firing_matrix.json.
  --step all      : run rescore, extract, classify in order.

Usage:
    python scripts/experiments/run_diagnostic_trace.py --step all
    python scripts/experiments/run_diagnostic_trace.py --step rescore
    python scripts/experiments/run_diagnostic_trace.py --step extract --limit 100
    python scripts/experiments/run_diagnostic_trace.py --step classify

The classifier rules are frozen at
``reports/experiments/diagnostic_trace/predicted_patterns.md`` and
MUST NOT be edited post-hoc to improve recoverability.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import beta
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from siv.compiler import compile_sentence_test_suite
from siv.schema import SentenceExtraction, TestSuite
from siv.scorer import score


# ═══════════════════════════════════════════════════════════════════════════
# Paths and constants
# ═══════════════════════════════════════════════════════════════════════════

OUT_DIR = _REPO / "reports" / "experiments" / "diagnostic_trace"
RESCORE_DIR = OUT_DIR / "pd_rescore"
PD_DIR = _REPO / "reports" / "experiments" / "perturbation_detection"
TEST_SUITES_PATH = _REPO / "test_suites" / "test_suites.jsonl"

PD_SCORED_INPUT = PD_DIR / "scored.jsonl"           # rescore reads this
PD_RESCORED = RESCORE_DIR / "scored.jsonl"          # extract reads this
TRACE_PATH = OUT_DIR / "trace_features.jsonl"       # classify reads this
EXTRACT_LOG = OUT_DIR / "extraction.log"
RESCORE_LOG = RESCORE_DIR / "rescore.log"

VAMPIRE_TIMEOUT_S = 5

# Six perturbation classes (FROZEN to PD taxonomy).
CLASSES = [
    "arg_swap",
    "negation_drop",
    "restrictor_drop",
    "random_substitution",
    "flip_outer_quantifier",
    "strengthen_quantifier",
]
UNRECOGNIZED = "unrecognized"

logger = logging.getLogger("run_diagnostic_trace")


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

def _compile_fresh_labeled_suite(extraction_json: dict) -> TestSuite:
    """Compile a fresh labeled suite through the asymmetry-axiom
    admissibility regime. Includes any swap_binary_args contrastives
    admitted by the asymmetry-axiom regime."""
    extraction = SentenceExtraction(**extraction_json)
    return compile_sentence_test_suite(
        extraction, with_contrastives=True, timeout_s=VAMPIRE_TIMEOUT_S,
    )


def _load_test_suite_index() -> Dict[str, dict]:
    """premise_id → suite row dict (includes extraction_json + positives +
    contrastives)."""
    index: Dict[str, dict] = {}
    with TEST_SUITES_PATH.open() as f:
        for line in f:
            row = json.loads(line)
            index[row["premise_id"]] = row
    return index


# ═══════════════════════════════════════════════════════════════════════════
# Step: rescore — rerun SIV on the PD pool under the asymmetry-axiom regime
# ═══════════════════════════════════════════════════════════════════════════

def _load_pd_scored() -> List[dict]:
    rows: List[dict] = []
    with PD_SCORED_INPUT.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _siv_scores(suite, candidate_fol: str) -> Dict[str, Optional[float]]:
    report = score(suite, candidate_fol, timeout_s=VAMPIRE_TIMEOUT_S)
    return {
        "siv_strict_recall": report.recall,
        "siv_strict_f1": report.f1,
    }


def _compute_diff_vs_pd(pd_rows: List[dict], rescored_rows: List[dict]) -> dict:
    """Per-class diff: SIV-F1 drop magnitudes (rescored vs PD baseline)
    and detection rates."""
    rescored_by_key: Dict[Tuple[str, str], dict] = {
        (r["premise_id"], r["class"]): r for r in rescored_rows
    }

    by_class_pd_drops: Dict[str, List[float]] = defaultdict(list)
    by_class_rescored_drops: Dict[str, List[float]] = defaultdict(list)
    by_class_pd_detect: Dict[str, int] = defaultdict(int)
    by_class_rescored_detect: Dict[str, int] = defaultdict(int)
    by_class_n: Dict[str, int] = defaultdict(int)

    for r_pd in pd_rows:
        key = (r_pd["premise_id"], r_pd["class"])
        if key not in rescored_by_key:
            continue
        r_re = rescored_by_key[key]
        cls = r_pd["class"]
        by_class_n[cls] += 1

        g_pd = r_pd["gold_scores"].get("siv_strict_f1")
        p_pd = r_pd["perturbed_scores"].get("siv_strict_f1")
        g_re = r_re["gold_scores"].get("siv_strict_f1")
        p_re = r_re["perturbed_scores"].get("siv_strict_f1")
        if g_pd is not None and p_pd is not None:
            d = g_pd - p_pd
            by_class_pd_drops[cls].append(d)
            if d > 0:
                by_class_pd_detect[cls] += 1
        if g_re is not None and p_re is not None:
            d = g_re - p_re
            by_class_rescored_drops[cls].append(d)
            if d > 0:
                by_class_rescored_detect[cls] += 1

    def _stats(xs: List[float]) -> dict:
        if not xs:
            return {"mean": None, "std": None, "ci95_lo": None, "ci95_hi": None}
        m = statistics.fmean(xs)
        s = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        ci = 1.96 * (s / math.sqrt(len(xs))) if len(xs) > 0 else 0.0
        return {
            "mean": m, "std": s,
            "ci95_lo": m - ci, "ci95_hi": m + ci,
            "n": len(xs),
        }

    out: Dict[str, dict] = {}
    for cls in sorted(by_class_n.keys()):
        n = by_class_n[cls]
        pd_drops = by_class_pd_drops[cls]
        re_drops = by_class_rescored_drops[cls]
        delta = (
            _stats(re_drops)["mean"] - _stats(pd_drops)["mean"]
            if pd_drops and re_drops else None
        )
        out[cls] = {
            "n": n,
            "v2_drop_stats": _stats(pd_drops),
            "v3_drop_stats": _stats(re_drops),
            "v2_detection_rate": by_class_pd_detect[cls] / n if n else 0.0,
            "v3_detection_rate": by_class_rescored_detect[cls] / n if n else 0.0,
            "v2_to_v3_drop_delta": delta,
            "v2_to_v3_detection_delta": (
                by_class_rescored_detect[cls] / n - by_class_pd_detect[cls] / n
                if n else None
            ),
        }
    return out


def step_rescore(limit: Optional[int] = None) -> dict:
    """Rerun SIV on the PD pool under the asymmetry-axiom regime. Outputs
    pd_rescore/{scored.jsonl, diff_vs_v2.json, run_metadata.json}."""
    RESCORE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading PD scored rows from %s", PD_SCORED_INPUT)
    pd_rows = _load_pd_scored()
    logger.info("Loaded %d PD rows", len(pd_rows))

    logger.info("Loading test-suite extraction index")
    suite_index = _load_test_suite_index()

    needed = sorted({r["premise_id"] for r in pd_rows})
    logger.info("Compiling fresh-admissibility suites for %d unique premises…",
                len(needed))

    suites: Dict[str, object] = {}
    t0 = time.time()
    new_swap_count = 0
    for i, pid in enumerate(needed):
        if pid not in suite_index:
            logger.warning("No extraction for %s; skipping", pid)
            continue
        try:
            suite = _compile_fresh_labeled_suite(
                suite_index[pid]["extraction_json"]
            )
        except Exception as e:
            logger.warning("Suite compile failed for %s: %s", pid, e)
            continue
        pd_contrastives = suite_index[pid].get("contrastives", [])
        pd_mks = {(c["fol"], c.get("mutation_kind")) for c in pd_contrastives}
        new_mks = {(c.fol, c.mutation_kind) for c in suite.contrastives}
        added = new_mks - pd_mks
        added_swap = sum(1 for _, mk in added if mk == "swap_binary_args")
        new_swap_count += added_swap
        suites[pid] = suite
        if (i + 1) % 50 == 0:
            logger.info("  compiled %d/%d (+%d swap admits so far)",
                        i + 1, len(needed), new_swap_count)
    logger.info("Compilation done: %d suites in %.1fs (+%d swap admits)",
                len(suites), time.time() - t0, new_swap_count)

    # Score gold once per premise.
    gold_scores: Dict[str, Dict[str, Optional[float]]] = {}
    pid_to_gold: Dict[str, str] = {r["premise_id"]: r["gold_fol"] for r in pd_rows}
    t1 = time.time()
    for i, pid in enumerate(needed):
        if pid not in suites:
            continue
        gold_scores[pid] = _siv_scores(suites[pid], pid_to_gold[pid])
        if (i + 1) % 100 == 0:
            logger.info("  scored gold %d/%d (%.1fs)", i + 1, len(needed),
                        time.time() - t1)
    logger.info("Gold scoring done in %.1fs", time.time() - t1)

    # Score every perturbed candidate.
    out_rows: List[dict] = []
    t2 = time.time()
    for i, r in enumerate(pd_rows):
        if limit is not None and i >= limit:
            break
        pid = r["premise_id"]
        if pid not in suites:
            continue
        siv_new = _siv_scores(suites[pid], r["candidate_fol"])
        perturbed_new = dict(r["perturbed_scores"])
        perturbed_new.update(siv_new)
        gold_new = dict(r["gold_scores"])
        gold_new.update(gold_scores.get(pid, {}))
        out_rows.append({
            "premise_id": pid,
            "class": r["class"],
            "expected_entailment": r["expected_entailment"],
            "gold_fol": r["gold_fol"],
            "candidate_fol": r["candidate_fol"],
            "gold_scores": gold_new,
            "perturbed_scores": perturbed_new,
        })
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t2
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(pd_rows) - i - 1) / rate if rate > 0 else float("inf")
            logger.info("  rescored %d/%d (%.1fs, ETA %.0fs)",
                        i + 1, len(pd_rows), elapsed, eta)
    logger.info("Perturbed scoring done in %.1fs", time.time() - t2)

    with PD_RESCORED.open("w") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    logger.info("Wrote %d rows to %s", len(out_rows), PD_RESCORED)

    diff = _compute_diff_vs_pd(pd_rows, out_rows)
    with (RESCORE_DIR / "diff_vs_v2.json").open("w") as f:
        json.dump(diff, f, indent=2)
    logger.info("Diff vs PD written.")

    with (RESCORE_DIR / "run_metadata.json").open("w") as f:
        json.dump({
            "vampire_timeout_s": VAMPIRE_TIMEOUT_S,
            "n_premises_compiled": len(suites),
            "n_rows_rescored": len(out_rows),
            "total_added_swap_binary_args": new_swap_count,
            "suite_compile_seconds": round(time.time() - t0, 1),
        }, f, indent=2)

    return diff


# ═══════════════════════════════════════════════════════════════════════════
# Step: extract — per-pair trace features from the rescored pool
# ═══════════════════════════════════════════════════════════════════════════

def _extract_one_pair(pair: dict, labeled_suite: TestSuite) -> dict:
    candidate_fol = pair["candidate_fol"]
    report = score(labeled_suite, candidate_fol, timeout_s=VAMPIRE_TIMEOUT_S)

    failed_positives: List[list] = []
    fired_contrastives: List[list] = []
    for r in report.per_test_results:
        if r.kind == "positive" and r.verdict != "entailed":
            if r.probe_label is not None:
                pk, ft = r.probe_label
                failed_positives.append([pk, ft])
            else:
                failed_positives.append([None, None])
        elif r.kind == "contrastive" and r.verdict == "entailed":
            if r.probe_label is not None:
                pk, ft = r.probe_label
                fired_contrastives.append([pk, ft, r.mutation_kind])
            else:
                fired_contrastives.append([None, None, r.mutation_kind])

    return {
        "premise_id": pair["premise_id"],
        "perturbation_class": pair["class"],
        "expected_entailment": pair["expected_entailment"],
        "gold_fol": pair["gold_fol"],
        "candidate_fol": candidate_fol,
        "failed_positives": failed_positives,
        "fired_contrastives": fired_contrastives,
        "positive_recall": report.recall,
        "siv_f1": report.f1,
        "positives_total": report.positives_total,
        "contrastives_total": report.contrastives_total,
    }


def _load_rescored_pairs() -> List[dict]:
    pairs: List[dict] = []
    with PD_RESCORED.open() as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def step_extract(limit: Optional[int] = None) -> None:
    """Extract per-pair trace features from the rescored PD pool against
    freshly compiled label-carrying suites. Outputs trace_features.jsonl."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading rescored pairs from %s", PD_RESCORED)
    pairs = _load_rescored_pairs()
    logger.info("Loaded %d pairs", len(pairs))

    logger.info("Loading test-suite index from %s", TEST_SUITES_PATH)
    suite_index = _load_test_suite_index()
    logger.info("Loaded %d suites", len(suite_index))

    needed_premises = sorted({p["premise_id"] for p in pairs})
    logger.info("Pre-labeling %d unique premises", len(needed_premises))

    labeled: Dict[str, TestSuite] = {}
    t0 = time.time()
    for i, pid in enumerate(needed_premises):
        if pid not in suite_index:
            logger.warning("Premise %s not in suite index — skip", pid)
            continue
        try:
            labeled[pid] = _compile_fresh_labeled_suite(
                suite_index[pid]["extraction_json"]
            )
        except Exception as e:
            logger.warning("Labeling failed for %s: %s", pid, e)
        if (i + 1) % 50 == 0:
            logger.info("  labeled %d/%d premises (%.1fs)", i + 1,
                        len(needed_premises), time.time() - t0)
    logger.info("Pre-labeling done in %.1fs", time.time() - t0)

    logger.info("Scoring %d pairs (limit=%s) — strict mode, no alignment",
                len(pairs), limit)
    t1 = time.time()
    n_written = 0
    n_skip_no_suite = 0
    with TRACE_PATH.open("w") as f:
        for i, pair in enumerate(pairs):
            if limit is not None and i >= limit:
                break
            pid = pair["premise_id"]
            if pid not in labeled:
                n_skip_no_suite += 1
                continue
            try:
                row = _extract_one_pair(pair, labeled[pid])
            except Exception as e:
                logger.warning("Scoring failed for %s/%s: %s",
                               pid, pair["class"], e)
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t1
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(pairs) - i - 1) / rate if rate > 0 else float("inf")
                logger.info("  scored %d/%d pairs (%.1fs, ETA %.0fs)",
                            i + 1, len(pairs), elapsed, eta)

    logger.info("Wrote %d rows to %s (skipped %d for missing suite)",
                n_written, TRACE_PATH, n_skip_no_suite)
    logger.info("Total wall: %.1fs", time.time() - t1)


# ═══════════════════════════════════════════════════════════════════════════
# Step: classify — frozen rule classifier + score-only baseline
# ═══════════════════════════════════════════════════════════════════════════

def classify_trace(
    failed_positives: List[list],
    fired_contrastives: List[list],
) -> str:
    """Recover the perturbation class from a labeled SIV trace.

    Rules — first match wins. Each rule is grounded in a structural
    prediction from reports/experiments/diagnostic_trace/predicted_patterns.md.
    DO NOT add rules without a corresponding structural prediction.
    """
    fired_ops = {c[2] for c in fired_contrastives if c[2] is not None}
    has_failed_pos = len(failed_positives) > 0

    if "drop_restrictor_conjunct" in fired_ops and not has_failed_pos:
        return "restrictor_drop"
    if "flip_quantifier" in fired_ops and not has_failed_pos:
        return "strengthen_quantifier"
    if "flip_quantifier" in fired_ops and has_failed_pos:
        return "flip_outer_quantifier"
    if "swap_binary_args" in fired_ops:
        return "arg_swap"
    if "negate_atom" in fired_ops or "replace_subformula_with_negation" in fired_ops:
        return "negation_drop"
    if has_failed_pos and len(fired_ops) == 0:
        return "random_substitution"
    return UNRECOGNIZED


def score_only_baseline(
    siv_f1: np.ndarray, y_true: np.ndarray, n_folds: int = 5, seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """5-fold stratified CV with a 1-feature DecisionTreeClassifier on
    siv_f1 → perturbation class."""
    X = siv_f1.reshape(-1, 1).astype(float)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_pred = np.empty_like(y_true)
    for train_idx, test_idx in skf.split(X, y_true):
        clf = DecisionTreeClassifier(random_state=seed)
        clf.fit(X[train_idx], y_true[train_idx])
        y_pred[test_idx] = clf.predict(X[test_idx])
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=CLASSES,
                        zero_division=0)
    return y_pred, macro_f1


def _clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    lo = float(beta.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
    return lo, hi


def _per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: List[str],
) -> Dict[str, dict]:
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )
    out: Dict[str, dict] = {}
    for i, lbl in enumerate(labels):
        n = int(support[i])
        k = 0 if n == 0 else int(round(rec[i] * n))
        lo, hi = _clopper_pearson_ci(k, n)
        out[lbl] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "n": n,
            "recall_ci_95": [lo, hi],
        }
    return out


def _confusion_matrix_dict(
    y_true: np.ndarray, y_pred: np.ndarray, labels: List[str],
) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {"labels": labels, "matrix": cm.tolist()}


def _build_contrastive_firing_matrix(rows: List[dict]) -> dict:
    operators: set = set()
    counts: Dict[str, Counter] = defaultdict(Counter)
    n_per_class: Counter = Counter()
    for row in rows:
        cls = row["perturbation_class"]
        n_per_class[cls] += 1
        fired_ops = {c[2] for c in row["fired_contrastives"] if c[2] is not None}
        operators.update(fired_ops)
        for op in fired_ops:
            counts[cls][op] += 1
    operator_list = sorted(operators)
    return {
        "classes": CLASSES,
        "operators": operator_list,
        "n_per_class": {cls: int(n_per_class[cls]) for cls in CLASSES},
        "counts": {cls: {op: int(counts[cls].get(op, 0)) for op in operator_list}
                   for cls in CLASSES},
        "rates": {
            cls: {op: (counts[cls].get(op, 0) / n_per_class[cls] if n_per_class[cls] else 0.0)
                  for op in operator_list}
            for cls in CLASSES
        },
    }


# Predicted firings per class (sanity check; not a recoverability signal).
PREDICTED_FIRING = {
    "arg_swap":              {"contrastive_any": ["swap_binary_args"],          "positive_pattern": None},
    "negation_drop":         {"contrastive_any": ["negate_atom",
                                                  "replace_subformula_with_negation"],
                              "positive_pattern": "non_empty"},
    "restrictor_drop":       {"contrastive_any": ["drop_restrictor_conjunct"],  "positive_pattern": "empty"},
    "random_substitution":   {"contrastive_any": None,                          "positive_pattern": "non_empty"},
    "flip_outer_quantifier": {"contrastive_any": ["flip_quantifier"],           "positive_pattern": "non_empty"},
    "strengthen_quantifier": {"contrastive_any": ["flip_quantifier"],           "positive_pattern": "empty"},
}


def _per_class_firing_recall(rows: List[dict]) -> dict:
    summary: Dict[str, dict] = {}
    for cls in CLASSES:
        pred = PREDICTED_FIRING[cls]
        class_rows = [r for r in rows if r["perturbation_class"] == cls]
        n = len(class_rows)
        contrast_ok = positive_ok = joint_ok = 0
        for row in class_rows:
            fired_ops = {c[2] for c in row["fired_contrastives"] if c[2] is not None}
            has_failed = len(row["failed_positives"]) > 0

            if pred["contrastive_any"] is None:
                c_ok = (len(fired_ops) == 0)
            else:
                c_ok = any(op in fired_ops for op in pred["contrastive_any"])
            if pred["positive_pattern"] is None:
                p_ok = True
            elif pred["positive_pattern"] == "empty":
                p_ok = not has_failed
            else:
                p_ok = has_failed

            contrast_ok += int(c_ok)
            positive_ok += int(p_ok)
            joint_ok += int(c_ok and p_ok)

        summary[cls] = {
            "n": n,
            "contrastive_predicted_fired_rate": (contrast_ok / n) if n else 0.0,
            "positive_pattern_held_rate":       (positive_ok / n) if n else 0.0,
            "joint_pattern_held_rate":          (joint_ok / n) if n else 0.0,
        }
    return summary


def step_classify() -> dict:
    """Apply the frozen rule classifier + the score-only baseline, dump
    metrics + predictions + firing matrix + sanity check."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    with TRACE_PATH.open() as f:
        for line in f:
            rows.append(json.loads(line))
    assert len(rows) == 1865, f"expected 1865 pairs, got {len(rows)}"

    y_true = np.array([r["perturbation_class"] for r in rows])
    siv_f1 = np.array([r["siv_f1"] for r in rows], dtype=float)

    # Labeled-trace rule classifier.
    y_pred_rule = np.array([
        classify_trace(r["failed_positives"], r["fired_contrastives"])
        for r in rows
    ])
    unrecognized_n = int((y_pred_rule == UNRECOGNIZED).sum())
    rule_labels_for_metrics = CLASSES + [UNRECOGNIZED]
    rule_macro_f1 = f1_score(
        y_true, y_pred_rule, average="macro", labels=CLASSES, zero_division=0,
    )
    rule_per_class = _per_class_metrics(y_true, y_pred_rule, CLASSES)
    rule_confusion = _confusion_matrix_dict(y_true, y_pred_rule, rule_labels_for_metrics)

    # Score-only baseline.
    y_pred_score, baseline_macro_f1 = score_only_baseline(siv_f1, y_true)
    baseline_per_class = _per_class_metrics(y_true, y_pred_score, CLASSES)
    baseline_confusion = _confusion_matrix_dict(y_true, y_pred_score, CLASSES)

    # Contrastive firing matrix.
    firing_matrix = _build_contrastive_firing_matrix(rows)
    with (OUT_DIR / "contrastive_firing_matrix.json").open("w") as f:
        json.dump(firing_matrix, f, indent=2, ensure_ascii=False)

    # Per-class firing recall sanity.
    firing_recall = _per_class_firing_recall(rows)

    results = {
        "n_pairs": len(rows),
        "rule_classifier": {
            "macro_f1_6class": float(rule_macro_f1),
            "unrecognized_n": unrecognized_n,
            "unrecognized_rate": unrecognized_n / len(rows),
            "per_class": rule_per_class,
            "confusion_matrix": rule_confusion,
        },
        "score_only_baseline": {
            "macro_f1_6class": float(baseline_macro_f1),
            "per_class": baseline_per_class,
            "confusion_matrix": baseline_confusion,
        },
        "recoverability_delta_macro_f1": float(rule_macro_f1 - baseline_macro_f1),
        "phase6_firing_recall_sanity": firing_recall,
    }
    with (OUT_DIR / "metrics.json").open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with (OUT_DIR / "predictions.jsonl").open("w") as f:
        for r, y_r, y_s in zip(rows, y_pred_rule, y_pred_score):
            f.write(json.dumps({
                "premise_id": r["premise_id"],
                "true_class": r["perturbation_class"],
                "rule_pred": str(y_r),
                "score_pred": str(y_s),
                "siv_f1": r["siv_f1"],
                "n_failed_positives": len(r["failed_positives"]),
                "fired_mutation_kinds": sorted({
                    c[2] for c in r["fired_contrastives"] if c[2] is not None
                }),
            }, ensure_ascii=False) + "\n")

    return results


def _print_classify_report(r: dict) -> None:
    rc = r["rule_classifier"]
    sb = r["score_only_baseline"]
    print("=" * 72)
    print("Diagnostic-trace experiment — classifier results")
    print("=" * 72)
    print(f"Pairs scored: {r['n_pairs']}")
    print()
    print(f"Labeled-trace classifier macro-F1 (6-class): {rc['macro_f1_6class']:.4f}")
    print(f"Score-only baseline   macro-F1 (6-class):    {sb['macro_f1_6class']:.4f}")
    print(f"Recoverability delta (rule − baseline):      "
          f"{r['recoverability_delta_macro_f1']:+.4f}")
    print()
    print(f"Unrecognized rate: {rc['unrecognized_rate']:.4f} "
          f"({rc['unrecognized_n']}/{r['n_pairs']})")
    print()
    print("Per-class (labeled-trace classifier):")
    print(f"  {'class':<25} {'n':>5}  {'prec':>6}  {'rec':>6}  "
          f"{'F1':>6}  rec_CI95")
    for cls in CLASSES:
        m = rc["per_class"][cls]
        lo, hi = m["recall_ci_95"]
        print(f"  {cls:<25} {m['n']:>5}  {m['precision']:>6.3f}  "
              f"{m['recall']:>6.3f}  {m['f1']:>6.3f}  ({lo:.3f}, {hi:.3f})")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="w")],
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--step", required=True,
        choices=["rescore", "extract", "classify", "all"],
        help="Pipeline step to run.",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="(rescore/extract only) Process only the first N pairs.",
    )
    args = ap.parse_args()

    if args.step in ("rescore", "all"):
        _setup_logging(RESCORE_LOG)
        diff = step_rescore(limit=args.limit)
        print("\n=== Per-class diff vs PD baseline ===")
        for cls, d in diff.items():
            pdm = d["v2_drop_stats"].get("mean")
            nem = d["v3_drop_stats"].get("mean")
            d_drop = d["v2_to_v3_drop_delta"]
            d_det = d["v2_to_v3_detection_delta"]
            print(f"  {cls:<25} n={d['n']:>4}  drop pd={pdm:.4f} rescored={nem:.4f} "
                  f"Δ={d_drop:+.4f}  detect pd={d['v2_detection_rate']:.3f} "
                  f"rescored={d['v3_detection_rate']:.3f} Δ={d_det:+.3f}")

    if args.step in ("extract", "all"):
        _setup_logging(EXTRACT_LOG)
        step_extract(limit=args.limit)

    if args.step in ("classify", "all"):
        results = step_classify()
        _print_classify_report(results)


if __name__ == "__main__":
    main()
