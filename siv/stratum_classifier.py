"""
Structural stratum classifier for gold FOL expressions.

Classifies a parsed NLTK Expression into one of eight phenomenon strata
(S1–S8) based on the formula's structural shape.  Used by the human
annotation study to ensure per-phenomenon coverage.

This classifier operates on NLTK ``Expression`` objects (parsed from gold
FOL strings) — it is independent of SIV's ``SentenceExtraction`` schema
and the existing ``classify_structure()`` in ``contrastive_generator.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from siv.fol_utils import parse_fol, normalize_fol_string, NLTK_AVAILABLE

if NLTK_AVAILABLE:
    from nltk.sem.logic import (
        Expression,
        AllExpression,
        ExistsExpression,
        NegatedExpression,
        ApplicationExpression,
        AndExpression,
        OrExpression,
        ImpExpression,
        IffExpression,
        BinaryExpression,
        ConstantExpression,
        IndividualVariableExpression,
    )

STRATUM_LABELS = [
    "S1_atomic",
    "S2_universal_simple",
    "S3_universal_multi_restrictor",
    "S4_nested_quantifier",
    "S5_relational",
    "S6_negation",
    "S7_existential",
    "S8_other",
]


@dataclass
class _Features:
    """Structural features collected from walking the AST."""

    quantifiers: List[Tuple[str, int]] = field(default_factory=list)
    # (type, depth) — type is "all" or "exists"

    max_pred_arity: int = 0
    has_constant_args: bool = False
    # True if any predicate has a ConstantExpression argument that is NOT
    # inside a quantifier binding that variable.

    has_nontrivial_negation: bool = False
    # True if any NegatedExpression wraps something other than a single
    # ApplicationExpression (i.e., negation at wider-than-atomic scope).

    antecedent_conjunct_count: int = 0
    # For a top-level AllExpression whose body is an ImpExpression, the
    # number of top-level conjuncts in the antecedent.

    has_complex_connective: bool = False
    # True if the formula uses or/implies/iff at non-trivial positions.

    top_is_forall: bool = False
    top_is_exists: bool = False
    has_exists_not_under_forall: bool = False


def _collect_features(expr, depth: int = 0, under_forall: bool = False) -> _Features:
    """Walk the NLTK AST and collect structural features."""
    if not NLTK_AVAILABLE:
        return _Features()

    feat = _Features()

    if isinstance(expr, AllExpression):
        feat.quantifiers.append(("all", depth))
        if depth == 0:
            feat.top_is_forall = True
            # Check for implication body → count antecedent conjuncts
            body = expr.term
            if isinstance(body, ImpExpression):
                feat.antecedent_conjunct_count = _count_conjuncts(body.first)

        inner = _collect_features(expr.term, depth + 1, under_forall=True)
        _merge(feat, inner)

    elif isinstance(expr, ExistsExpression):
        feat.quantifiers.append(("exists", depth))
        if depth == 0:
            feat.top_is_exists = True
        if not under_forall:
            feat.has_exists_not_under_forall = True

        inner = _collect_features(expr.term, depth + 1, under_forall=under_forall)
        _merge(feat, inner)

    elif isinstance(expr, NegatedExpression):
        if not isinstance(expr.term, ApplicationExpression):
            feat.has_nontrivial_negation = True
        inner = _collect_features(expr.term, depth, under_forall=under_forall)
        _merge(feat, inner)

    elif isinstance(expr, ApplicationExpression):
        # Uncurry to get predicate arity and constant args
        func = expr
        args = []
        while isinstance(func, ApplicationExpression):
            args.insert(0, func.argument)
            func = func.function
        arity = len(args)
        feat.max_pred_arity = max(feat.max_pred_arity, arity)
        for arg in args:
            if isinstance(arg, ConstantExpression):
                feat.has_constant_args = True

    elif isinstance(expr, BinaryExpression):
        if isinstance(expr, (OrExpression, ImpExpression, IffExpression)):
            feat.has_complex_connective = True

        left = _collect_features(expr.first, depth, under_forall=under_forall)
        right = _collect_features(expr.second, depth, under_forall=under_forall)
        _merge(feat, left)
        _merge(feat, right)

    elif hasattr(expr, "term"):
        inner = _collect_features(expr.term, depth, under_forall=under_forall)
        _merge(feat, inner)

    return feat


def _merge(target: _Features, source: _Features) -> None:
    """Merge source features into target."""
    target.quantifiers.extend(source.quantifiers)
    target.max_pred_arity = max(target.max_pred_arity, source.max_pred_arity)
    target.has_constant_args = target.has_constant_args or source.has_constant_args
    target.has_nontrivial_negation = (
        target.has_nontrivial_negation or source.has_nontrivial_negation
    )
    target.has_complex_connective = (
        target.has_complex_connective or source.has_complex_connective
    )
    target.has_exists_not_under_forall = (
        target.has_exists_not_under_forall or source.has_exists_not_under_forall
    )
    # antecedent_conjunct_count only set at top level, don't overwrite
    if target.antecedent_conjunct_count == 0:
        target.antecedent_conjunct_count = source.antecedent_conjunct_count
    # top_is_* only set at depth 0
    target.top_is_forall = target.top_is_forall or source.top_is_forall
    target.top_is_exists = target.top_is_exists or source.top_is_exists


def _count_conjuncts(expr) -> int:
    """Count top-level conjuncts by flattening AndExpression."""
    if not NLTK_AVAILABLE:
        return 0
    if isinstance(expr, AndExpression):
        return _count_conjuncts(expr.first) + _count_conjuncts(expr.second)
    return 1


def classify_stratum(expr) -> str:
    """Classify a parsed NLTK Expression into one of S1-S8.

    Tie-break priority: S3 > S4 > S6 > S2 > S7 > S5 > S1.
    """
    feat = _collect_features(expr)

    n_quant = len(feat.quantifiers)
    has_nested = any(d > 0 for _, d in feat.quantifiers)
    has_forall = any(t == "all" for t, _ in feat.quantifiers)

    # S3: forall with multi-conjunct antecedent, no nesting
    if has_forall and feat.antecedent_conjunct_count >= 2 and not has_nested:
        return "S3_universal_multi_restrictor"

    # S4: >=2 quantifiers with at least one nested
    if n_quant >= 2 and has_nested:
        return "S4_nested_quantifier"

    # S6: non-trivial negation
    if feat.has_nontrivial_negation:
        return "S6_negation"

    # S2: single forall with 1 antecedent conjunct, no nesting
    if has_forall and feat.antecedent_conjunct_count == 1 and not has_nested:
        return "S2_universal_simple"

    # S7: top-level exists or exists not dominated by forall
    if feat.top_is_exists or feat.has_exists_not_under_forall:
        return "S7_existential"

    # S5: arity >= 2 with constant args, no quantifiers
    if (
        feat.max_pred_arity >= 2
        and feat.has_constant_args
        and n_quant == 0
    ):
        return "S5_relational"

    # S1: no quantifiers, no complex connectives
    if n_quant == 0 and not feat.has_complex_connective:
        return "S1_atomic"

    return "S8_other"


def classify_stratum_from_fol(fol_string: str) -> Optional[str]:
    """Convenience: parse *fol_string*, classify.  Returns None on parse failure."""
    normalized = normalize_fol_string(fol_string)
    expr = parse_fol(normalized)
    if expr is None:
        return None
    return classify_stratum(expr)
