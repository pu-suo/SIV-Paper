"""Common utilities shared across the three paper experiments.

Provides loaders for cached artifacts, subset filters, metric runners,
and statistical helpers.

File naming note:
  - Cache file is ``test_suites.jsonl`` (not ``test_suites_train.jsonl``).
  - Failures are in ``test_suites.failures.json`` (JSON array, not JSONL).
  - Experiment directories use full names (``severity_correlation/``,
    ``perturbation_detection/``, ``diagnostic_trace/``) under
    ``reports/experiments/``.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from siv.aligner import (
    align_symbols,
    extract_symbols_from_fol,
    rewrite_fol_strings,
    rewrite_test_suite,
)
from siv.contrastive_generator import derive_witness_axioms
from siv.fol_utils import free_individual_variables, parse_fol
from siv.schema import SentenceExtraction, TestSuite, UnitTest
from siv.scorer import ScoreReport, score

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Section 1 — Loaders
# ═══════════════════════════════════════════════════════════════════════════

def load_test_suites(path: str | Path) -> Dict[str, dict]:
    """Load test suites keyed by premise_id from a JSONL file."""
    suites: Dict[str, dict] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        suites[entry["premise_id"]] = entry
    return suites


def load_candidates(path: str | Path) -> List[dict]:
    """Load candidate list from a JSONL file."""
    candidates: List[dict] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        candidates.append(json.loads(line))
    return candidates


# ═══════════════════════════════════════════════════════════════════════════
# Section 2 — Subset filters (includes Pre-work C)
# ═══════════════════════════════════════════════════════════════════════════

def is_contrastive_eligible(test_suite_row: dict) -> bool:
    """Return True iff the test suite row has non-empty contrastives (Pre-work C)."""
    return len(test_suite_row.get("contrastives", [])) > 0


def _extract_predicate_names(fol_string: str) -> Set[str]:
    """Extract predicate names from a FOL string, lowercased and stripped."""
    symbols = extract_symbols_from_fol(fol_string)
    return {re.sub(r"[^a-z0-9]", "", name.lower()) for name in symbols["predicates"]}


def _extract_predicates_with_arity(fol_string: str) -> Dict[str, int]:
    """Extract predicate name -> arity mapping from a FOL string."""
    symbols = extract_symbols_from_fol(fol_string)
    return symbols["predicates"]


def _count_quantifiers(expr, depth: int = 0) -> Tuple[int, int, int]:
    """Count universals, existentials, and max nesting depth in an NLTK expression."""
    from nltk.sem.logic import AllExpression, ExistsExpression, BinaryExpression, NegatedExpression

    universals, existentials, max_depth = 0, 0, depth

    if isinstance(expr, AllExpression):
        universals = 1
        u, e, d = _count_quantifiers(expr.term, depth + 1)
        return universals + u, existentials + e, max(max_depth, d)
    elif isinstance(expr, ExistsExpression):
        existentials = 1
        u, e, d = _count_quantifiers(expr.term, depth + 1)
        return universals + u, existentials + e, max(max_depth, d)
    elif isinstance(expr, NegatedExpression):
        return _count_quantifiers(expr.term, depth)
    elif isinstance(expr, BinaryExpression):
        u1, e1, d1 = _count_quantifiers(expr.first, depth)
        u2, e2, d2 = _count_quantifiers(expr.second, depth)
        return u1 + u2, e1 + e2, max(d1, d2)
    else:
        return 0, 0, depth


# ── Structural-richness criteria (shared by Exp 1 and Exp 2) ──────────────
#
# The premise pool is filtered to formulas with at least one of:
#   - antecedent has ≥2 conjuncts (compound restrictor)
#   - consequent / nucleus has ≥2 conjuncts (conjunctive consequent)
#   - has a nested existential (e.g. ∀x. P(x) → ∃y. Q(x,y))
#   - has ≥2 binary predicates
#
# This replaces the legacy Jaccard-based "aligned subset" filter, which
# was vacuous under the deterministic v3 pipeline (canonical_fol IS the
# parsed gold_fol, so Jaccard is always 1.0). The structural-richness
# criteria identify premises whose logical structure is rich enough to
# support meaningful perturbation candidates (Exp 1) and graded
# contrastive tiers (Exp 2).


def _has_nested_existential(formula_dict: Optional[dict]) -> bool:
    """Recursively check for a nested existential anywhere in the formula."""
    if not formula_dict:
        return False
    quant = formula_dict.get("quantification")
    if quant:
        if quant.get("inner_quantifications"):
            return True
        nucleus = quant.get("nucleus") or {}
        if nucleus.get("quantification"):
            return True
        if _has_nested_existential(nucleus):
            return True
    for op in formula_dict.get("operands") or []:
        if _has_nested_existential(op):
            return True
    return False


def _count_consequent_conjuncts(formula_dict: dict) -> int:
    """Count conjuncts in consequent / nucleus position."""
    quant = formula_dict.get("quantification")
    if quant:
        nucleus = quant.get("nucleus") or {}
        if nucleus.get("connective") == "and":
            return len(nucleus.get("operands") or [])
        nq = nucleus.get("quantification")
        if nq:
            inner_restr = nq.get("restrictor") or []
            inner_nuc = nq.get("nucleus") or {}
            if inner_nuc.get("atomic"):
                return 1 + len(inner_restr)
            if inner_nuc.get("connective") == "and":
                return len(inner_nuc.get("operands") or [])
    if formula_dict.get("connective") == "and":
        return len(formula_dict.get("operands") or [])
    return 1


def _count_antecedent_conjuncts(formula_dict: dict) -> int:
    """Count restrictor atoms (antecedent conjuncts) for top-level universal."""
    quant = formula_dict.get("quantification")
    if quant:
        return len(quant.get("restrictor") or [])
    return 0


def extract_structural_features(extraction_json: dict) -> dict:
    """Public: extract structural features from an extraction_json."""
    formula = extraction_json.get("formula") or {}
    preds = extraction_json.get("predicates") or []
    binary_preds = [p for p in preds if p.get("arity", 0) >= 2]
    return {
        "antecedent_conjunct_count": _count_antecedent_conjuncts(formula),
        "consequent_conjunct_count": _count_consequent_conjuncts(formula),
        "has_nested_existential": _has_nested_existential(formula),
        "n_binary_predicates": len(binary_preds),
    }


def passes_structural_criteria(features: dict) -> bool:
    """True iff at least one structural-richness criterion is met."""
    return (
        features["antecedent_conjunct_count"] >= 2
        or features["consequent_conjunct_count"] >= 2
        or features["has_nested_existential"]
        or features["n_binary_predicates"] >= 2
    )


def passes_premise_filter(
    test_suite_row: dict,
    broken_gold_ids: FrozenSet[str] = frozenset(),
) -> Tuple[bool, dict]:
    """Premise-level filter for the experimental pool.

    A premise passes iff:
      1. Gold FOL parses (deterministic — already implied by being in
         test_suites.jsonl, which only contains successfully-parsed gold).
      2. Premise is not in the broken-gold set (no free vars; not flagged
         for hand correction in Exp 3).
      3. Structural richness — formula has antecedent / consequent
         conjuncts, nested existential, or ≥2 binary predicates (so it
         can support both meaningful perturbations and graded
         contrastive tiers).

    Returns (passes, criteria_dict).
    """
    criteria: Dict[str, Any] = {}

    gold_fol = test_suite_row.get("gold_fol", "")
    gold_expr = parse_fol(gold_fol)
    criteria["gold_parses"] = gold_expr is not None
    if not criteria["gold_parses"]:
        return False, criteria

    pid = test_suite_row.get("premise_id", "")
    criteria["not_broken_gold"] = pid not in broken_gold_ids
    if not criteria["not_broken_gold"]:
        return False, criteria

    ej = test_suite_row.get("extraction_json", {}) or {}
    features = extract_structural_features(ej)
    criteria["structural_features"] = features
    criteria["structurally_rich"] = passes_structural_criteria(features)
    if not criteria["structurally_rich"]:
        return False, criteria

    return True, criteria


def _passes_aligned_subset_filter__deprecated(
    test_suite_row: dict,
    gold_fol: str,
    broken_gold_ids: FrozenSet[str] = frozenset(),
    jaccard_threshold: float = 0.6,
    require_full_alignment: bool = False,
) -> Tuple[bool, dict]:
    """Deprecated: legacy v2 filter. Kept private only to verify no callers
    survive elsewhere; all production code now uses ``passes_premise_filter``."""
    criteria: Dict[str, Any] = {}
    canonical_fol = test_suite_row.get("canonical_fol", "")

    # 1. SIV extraction succeeded — canonical_fol is non-null, no free vars
    if not canonical_fol:
        criteria["extraction_ok"] = False
        return False, criteria
    ej = test_suite_row.get("extraction_json", {})
    declared_consts = frozenset(c["id"] for c in ej.get("constants", []))
    fv = free_individual_variables(canonical_fol, declared_consts)
    criteria["extraction_ok"] = len(fv) == 0
    if not criteria["extraction_ok"]:
        return False, criteria

    # 2. Gold parses cleanly
    gold_expr = parse_fol(gold_fol)
    criteria["gold_parses"] = gold_expr is not None
    if not criteria["gold_parses"]:
        return False, criteria

    # 3. Predicate-name Jaccard >= threshold
    siv_preds = _extract_predicate_names(canonical_fol)
    gold_preds = _extract_predicate_names(gold_fol)
    if siv_preds or gold_preds:
        jaccard = len(siv_preds & gold_preds) / len(siv_preds | gold_preds)
    else:
        jaccard = 0.0
    criteria["jaccard"] = round(jaccard, 3)
    if jaccard < jaccard_threshold:
        return False, criteria

    # 4. Arity match for shared predicates
    siv_pred_arity = _extract_predicates_with_arity(canonical_fol)
    gold_pred_arity = _extract_predicates_with_arity(gold_fol)
    shared = set(siv_pred_arity) & set(gold_pred_arity)
    arity_match = all(siv_pred_arity[p] == gold_pred_arity[p] for p in shared)
    criteria["arity_match"] = arity_match
    if not arity_match:
        return False, criteria

    # 5. Quantifier-skeleton match
    siv_expr = parse_fol(canonical_fol)
    if siv_expr is not None and gold_expr is not None:
        su, se, sd = _count_quantifiers(siv_expr)
        gu, ge, gd = _count_quantifiers(gold_expr)
        quant_match = (su == gu) and (se == ge) and (abs(sd - gd) <= 1)
    else:
        quant_match = False
    criteria["quant_skeleton_match"] = quant_match
    if not quant_match:
        return False, criteria

    # 6. Not in broken-gold list
    premise_id = test_suite_row.get("premise_id", "")
    criteria["not_broken_gold"] = premise_id not in broken_gold_ids
    if not criteria["not_broken_gold"]:
        return False, criteria

    # 7. Full predicate alignment (optional, for SIV-soft scoring quality)
    if require_full_alignment:
        siv_syms = extract_symbols_from_fol(canonical_fol)
        gold_syms = extract_symbols_from_fol(gold_fol)
        alignment = align_symbols(siv_syms, gold_syms)
        full_pred = (not alignment.unaligned_siv_predicates
                     and not alignment.unaligned_candidate_predicates)
        criteria["full_predicate_alignment"] = full_pred
        if not full_pred:
            return False, criteria

    return True, criteria


# ═══════════════════════════════════════════════════════════════════════════
# Section 3 — Metric runners
# ═══════════════════════════════════════════════════════════════════════════

# ── BLEU / BERTScore (sacrebleu + bert_score, inline) ─────────────────────

def score_bleu(candidate_fol: str, gold_fol: str) -> Optional[float]:
    """Sentence-level BLEU on FOL strings via sacrebleu."""
    try:
        import sacrebleu
        bleu = sacrebleu.sentence_bleu(candidate_fol, [gold_fol])
        return round(float(bleu.score) / 100.0, 4)
    except Exception as e:
        logger.warning("BLEU scoring failed: %s", e)
        return None


def score_bertscore(candidate_fol: str, gold_fol: str) -> Optional[float]:
    """BERTScore F1 on FOL strings via bert_score with deberta-base-mnli."""
    try:
        from bert_score import score as bert_score
        P, R, F1 = bert_score(
            [candidate_fol], [gold_fol],
            lang="en",
            model_type="microsoft/deberta-base-mnli",
            verbose=False,
        )
        return round(float(F1[0]), 4)
    except Exception as e:
        logger.warning("BERTScore scoring failed: %s", e)
        return None


# ── Propositional LE (Yang et al. 2024 §4.3) ──────────────────────────────
#
# Continuous [0,1] score: strip quantifiers, treat each unique
# predicate-application as one propositional atom, compute the fraction of
# truth assignments under which the two formulas agree.

def score_propositional_le_raw(
    candidate_fol: str, gold_fol: str, timeout: int = 10,
) -> Optional[float]:
    """Propositional LE without vocabulary alignment."""
    try:
        from siv.propositional_le import propositional_le
        return propositional_le(candidate_fol, gold_fol, timeout=timeout)
    except Exception as e:
        logger.warning("propositional_le-raw failed: %s", e)
        return None


def score_propositional_le_aligned(
    candidate_fol: str, gold_fol: str, timeout: int = 10,
) -> Optional[float]:
    """Propositional LE with vocabulary alignment."""
    try:
        from siv.propositional_le import propositional_le_aligned
        return propositional_le_aligned(candidate_fol, gold_fol, timeout=timeout)
    except Exception as e:
        logger.warning("propositional_le-aligned failed: %s", e)
        return None


# ── Brunello-LT ────────────────────────────────────────────────────────────

def score_brunello_lt_raw(
    candidate_fol: str, gold_fol: str, timeout: int = 10,
) -> Optional[float]:
    """Brunello-LT equivalence via Z3, without vocabulary alignment."""
    try:
        from siv.brunello_lt import brunello_lt_equivalence
        return brunello_lt_equivalence(candidate_fol, gold_fol, timeout=timeout)
    except Exception as e:
        logger.warning("Brunello-LT-raw failed: %s", e)
        return None


def score_brunello_lt_aligned(
    candidate_fol: str, gold_fol: str, timeout: int = 10,
) -> Optional[float]:
    """Brunello-LT equivalence via Z3, with vocabulary alignment."""
    try:
        from siv.brunello_lt import brunello_lt_equivalence_aligned
        return brunello_lt_equivalence_aligned(candidate_fol, gold_fol, timeout=timeout)
    except Exception as e:
        logger.warning("Brunello-LT-aligned failed: %s", e)
        return None


# ── Smatch++ ───────────────────────────────────────────────────────────────

def score_smatchpp(
    candidate_fol: str, gold_fol: str, timeout: int = 10,
) -> Optional[float]:
    """Smatch++ structural-graph F1 between candidate and gold FOL.

    The ``timeout`` argument is accepted for signature compatibility with
    the other reference-based scorers but is not used (Smatch++ has its
    own internal ILP timeout).
    """
    del timeout  # Smatch++ ILP solver manages its own timeout
    try:
        from siv.smatchpp_baseline import smatchpp_score
        return smatchpp_score(candidate_fol, gold_fol)
    except Exception as e:
        logger.warning("Smatch++ failed: %s", e)
        return None


# ── SIV ────────────────────────────────────────────────────────────────────

def _reconstruct_test_suite(suite_dict: dict) -> TestSuite:
    """Rebuild a TestSuite Pydantic model from a saved JSON dict."""
    extraction = SentenceExtraction(**suite_dict["extraction_json"])
    positives = [
        UnitTest(fol=t["fol"], kind=t["kind"])
        for t in suite_dict["positives"]
    ]
    contrastives = [
        UnitTest(fol=t["fol"], kind=t["kind"], mutation_kind=t.get("mutation_kind"))
        for t in suite_dict["contrastives"]
    ]
    return TestSuite(
        extraction=extraction,
        positives=positives,
        contrastives=contrastives,
    )


def score_siv_strict(
    test_suite_row: dict,
    candidate_fol: str,
    timeout: int = 10,
) -> Optional[ScoreReport]:
    """Score candidate in SIV strict mode (no vocabulary alignment)."""
    try:
        suite = _reconstruct_test_suite(test_suite_row)
        return score(suite, candidate_fol, timeout_s=timeout)
    except Exception as e:
        logger.warning("SIV-strict failed for %s: %s",
                       test_suite_row.get("premise_id"), e)
        return None


def score_siv_soft(
    test_suite_row: dict,
    candidate_fol: str,
    timeout: int = 10,
    threshold: float = 0.6,
) -> Optional[ScoreReport]:
    """Score candidate in SIV soft mode (aligned vocabulary).

    Pipeline: extract symbols from candidate -> align_symbols(threshold) ->
    rewrite_test_suite() + rewrite_fol_strings(witness_axioms) ->
    score(rewritten_suite, candidate, witness_axioms_override=rewritten_axioms).

    Witness axioms come from derive_witness_axioms(extraction), derived from
    the original extraction — NOT from the test suite.
    """
    try:
        suite = _reconstruct_test_suite(test_suite_row)
        canonical_fol = test_suite_row["canonical_fol"]

        siv_symbols = extract_symbols_from_fol(canonical_fol)
        cand_symbols = extract_symbols_from_fol(candidate_fol)
        alignment = align_symbols(siv_symbols, cand_symbols, threshold=threshold)

        rewritten_suite = rewrite_test_suite(suite, alignment)

        raw_witnesses = derive_witness_axioms(suite.extraction)
        rewritten_witnesses = rewrite_fol_strings(raw_witnesses, alignment)

        return score(
            rewritten_suite, candidate_fol, timeout_s=timeout,
            witness_axioms_override=rewritten_witnesses,
        )
    except Exception as e:
        logger.warning("SIV-soft failed for %s: %s",
                       test_suite_row.get("premise_id"), e)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Section 4 — Statistical helpers
# ═══════════════════════════════════════════════════════════════════════════

def paired_bootstrap_ci(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    rng_seed: int = 42,
) -> Tuple[float, float]:
    """Paired bootstrap 95% CI for the difference mean(a) - mean(b).

    Returns (ci_lower, ci_upper).
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    assert len(scores_a) == len(scores_b), "Arrays must have equal length"

    rng = np.random.RandomState(rng_seed)
    n = len(scores_a)
    diffs = np.empty(n_resamples)

    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        diffs[i] = scores_a[idx].mean() - scores_b[idx].mean()

    lo = np.percentile(diffs, 100 * alpha / 2)
    hi = np.percentile(diffs, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def paired_permutation_p(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_permutations: int = 10000,
    rng_seed: int = 42,
) -> float:
    """Two-sided paired permutation test p-value for mean(a) == mean(b)."""
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    assert len(scores_a) == len(scores_b), "Arrays must have equal length"

    rng = np.random.RandomState(rng_seed)
    n = len(scores_a)
    observed = abs(scores_a.mean() - scores_b.mean())

    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=n)
        diff = scores_a - scores_b
        perm_diff = (diff * signs).mean()
        if abs(perm_diff) >= observed:
            count += 1

    return count / n_permutations


def auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve.

    Uses sklearn if available, otherwise falls back to a manual
    trapezoidal-rule implementation.

    Args:
        scores: predicted scores (higher = more likely positive)
        labels: binary labels (0 or 1)
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except ImportError:
        pass

    # Manual implementation: sort by descending score, sweep threshold
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5  # Undefined; return chance level

    tp, fp = 0, 0
    prev_tp, prev_fp = 0, 0
    auc = 0.0

    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
            # Trapezoid: area added when FP increases
            auc += (tp + prev_tp) / 2.0
            prev_tp = tp

    # Final trapezoid
    auc += (tp + prev_tp) / 2.0 * (0 if fp == prev_fp else 1)

    return auc / (n_pos * n_neg) if (n_pos * n_neg) > 0 else 0.5
