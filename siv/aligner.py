"""
Cross-vocabulary symbol alignment for soft SIV scoring.

Aligns SIV-extracted predicate and constant symbols to a candidate FOL's
vocabulary via semantic embedding similarity, then rewrites the test suite
so that Vampire scoring operates in the candidate's symbol space.

Public API
----------
extract_symbols_from_fol(fol_string) -> dict
align_symbols(siv_symbols, candidate_symbols, threshold) -> AlignmentResult
rewrite_test_suite(test_suite, alignment) -> TestSuite
rewrite_fol_strings(fol_strings, alignment) -> List[str]
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from siv.fol_utils import parse_fol, normalize_fol_string, NLTK_AVAILABLE
from siv.schema import TestSuite, UnitTest

if NLTK_AVAILABLE:
    from nltk.sem.logic import (
        ApplicationExpression, NegatedExpression, BinaryExpression,
        AllExpression, ExistsExpression, EqualityExpression,
        IndividualVariableExpression, ConstantExpression,
    )


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class AlignmentResult:
    """Result of aligning SIV symbols to candidate symbols.

    ``predicate_map`` and ``constant_map`` include identity mappings
    (score 1.0) so the audit trail shows which symbols matched exactly
    vs. which required semantic alignment.  ``rewrite_test_suite`` skips
    identity mappings during regex substitution.
    """

    predicate_map: Dict[str, str]
    constant_map: Dict[str, str]
    predicate_scores: Dict[str, float]
    constant_scores: Dict[str, float]
    unaligned_siv_predicates: List[str]
    unaligned_candidate_predicates: List[str]
    unaligned_siv_constants: List[str]
    unaligned_candidate_constants: List[str]
    threshold: float


# ── Symbol extraction ────────────────────────────────────────────────────────

def extract_symbols_from_fol(fol_string: str) -> dict:
    """Parse a FOL string and return its symbol inventory.

    Returns ``{"predicates": {name: arity, ...}, "constants": set(...)}``.
    On parse failure, falls back to regex extraction.  On total failure,
    returns empty collections (graceful degradation).
    """
    if not fol_string or not isinstance(fol_string, str):
        return {"predicates": {}, "constants": set()}

    normalized = normalize_fol_string(fol_string)
    expr = parse_fol(normalized) if normalized else None

    if expr is not None:
        predicates: Dict[str, int] = {}
        constants: Set[str] = set()
        _walk_expr(expr, predicates, constants)
        return {"predicates": predicates, "constants": constants}

    return _extract_symbols_regex(fol_string)


def _walk_expr(expr, predicates: Dict[str, int], constants: Set[str]) -> None:
    """Recursively walk an NLTK Expression, collecting predicates and constants."""
    if not NLTK_AVAILABLE:
        return

    if isinstance(expr, ApplicationExpression):
        # Uncurry: f(a)(b)(c) → head, [a, b, c]
        func = expr
        args = []
        while isinstance(func, ApplicationExpression):
            args.insert(0, func.argument)
            func = func.function

        # Head is the predicate
        if hasattr(func, "variable"):
            pred_name = func.variable.name
            arity = len(args)
            if pred_name not in predicates:
                predicates[pred_name] = arity

        # Collect constants from arguments
        for arg in args:
            if isinstance(arg, ConstantExpression):
                constants.add(str(arg))
            elif isinstance(arg, ApplicationExpression):
                # Nested application — recurse
                _walk_expr(arg, predicates, constants)

    elif isinstance(expr, BinaryExpression):
        _walk_expr(expr.first, predicates, constants)
        _walk_expr(expr.second, predicates, constants)

    elif isinstance(expr, NegatedExpression):
        _walk_expr(expr.term, predicates, constants)

    elif isinstance(expr, EqualityExpression):
        for side in (expr.first, expr.second):
            if isinstance(side, ConstantExpression):
                constants.add(str(side))

    elif isinstance(expr, (AllExpression, ExistsExpression)):
        _walk_expr(expr.term, predicates, constants)

    elif hasattr(expr, "term"):
        _walk_expr(expr.term, predicates, constants)


def _extract_symbols_regex(fol_string: str) -> dict:
    """Regex fallback for strings that NLTK cannot parse."""
    normalized = normalize_fol_string(fol_string) or fol_string

    predicates: Dict[str, int] = {}
    keywords = {"All", "Exists", "Forall", "And", "Or", "Not", "Implies"}

    for match in re.finditer(r"([A-Z][A-Za-z0-9_]*)\s*\(([^)]*)\)", normalized):
        name = match.group(1)
        if name in keywords:
            continue
        args = [a.strip() for a in match.group(2).split(",") if a.strip()]
        if name not in predicates:
            predicates[name] = len(args)

    # Also scan the raw string (handles Unicode-context predicates)
    for match in re.finditer(r"([A-Z][A-Za-z0-9_]*)\s*\(([^)]*)\)", fol_string):
        name = match.group(1)
        if name in keywords:
            continue
        args = [a.strip() for a in match.group(2).split(",") if a.strip()]
        if name not in predicates:
            predicates[name] = len(args)

    # Constants: multi-char lowercase identifiers inside predicate args,
    # excluding NLTK variable patterns ([a-z][0-9]*)
    constants: Set[str] = set()
    for match in re.finditer(r"[A-Z][A-Za-z0-9_]*\(([^)]*)\)", normalized):
        for arg in match.group(1).split(","):
            arg = arg.strip()
            if (
                re.match(r"^[a-z][a-zA-Z0-9_]*$", arg)
                and not re.match(r"^[a-z]\d*$", arg)
            ):
                constants.add(arg)

    return {"predicates": predicates, "constants": constants}


# ── Name preprocessing ───────────────────────────────────────────────────────

def _preprocess_name(name: str) -> str:
    """Convert a symbol name to space-separated words for embedding.

    Splits camelCase and underscore_case, lowercases.
    Example: "HasTeeth" → "has teeth", "school_talent_show" → "school talent show"
    """
    # CamelCase split
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    # Underscore split
    s = s.replace("_", " ")
    return s.lower().strip()


# ── Embedding infrastructure ─────────────────────────────────────────────────

_EMBEDDER = None
_EMBEDDING_CACHE: Dict[str, np.ndarray] = {}


def _get_embedder():
    """Lazy-load the sentence-transformers model (singleton)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def _embed(texts: List[str]) -> np.ndarray:
    """Embed a list of texts, caching results for determinism."""
    embedder = _get_embedder()
    uncached = [t for t in texts if t not in _EMBEDDING_CACHE]
    if uncached:
        vecs = embedder.encode(uncached, normalize_embeddings=True)
        for t, v in zip(uncached, vecs):
            _EMBEDDING_CACHE[t] = v
    return np.array([_EMBEDDING_CACHE[t] for t in texts])


# ── Bipartite alignment ─────────────────────────────────────────────────────

def align_symbols(
    siv_symbols: dict,
    candidate_symbols: dict,
    threshold: float = 0.6,
) -> AlignmentResult:
    """Compute one-to-one bipartite alignment between SIV and candidate symbols.

    Predicate alignment is arity-partitioned: a SIV predicate with arity k
    can only align to a candidate predicate with arity k.  Constant alignment
    has no arity constraint.  Both use ``scipy.optimize.linear_sum_assignment``
    on an embedding-similarity cost matrix with a configurable threshold.
    """
    # TODO: threshold sensitivity sweep -- evaluate alignment quality across
    # threshold values [0.4, 0.5, 0.6, 0.7, 0.8] on the FOLIO validation set
    # to calibrate the default.

    siv_preds = siv_symbols.get("predicates", {})
    cand_preds = candidate_symbols.get("predicates", {})
    siv_consts = siv_symbols.get("constants", set())
    cand_consts = candidate_symbols.get("constants", set())

    # ── Predicate alignment (arity-partitioned) ──
    pred_map: Dict[str, str] = {}
    pred_scores: Dict[str, float] = {}
    unaligned_siv_preds: List[str] = []
    unaligned_cand_preds: List[str] = []

    # Group by arity
    siv_by_arity: Dict[int, List[str]] = defaultdict(list)
    cand_by_arity: Dict[int, List[str]] = defaultdict(list)
    for name, arity in siv_preds.items():
        siv_by_arity[arity].append(name)
    for name, arity in cand_preds.items():
        cand_by_arity[arity].append(name)

    # Sort name lists for determinism
    for names in siv_by_arity.values():
        names.sort()
    for names in cand_by_arity.values():
        names.sort()

    all_arities = set(siv_by_arity.keys()) | set(cand_by_arity.keys())
    for arity in sorted(all_arities):
        siv_names = siv_by_arity.get(arity, [])
        cand_names = cand_by_arity.get(arity, [])

        if not siv_names:
            unaligned_cand_preds.extend(
                f"{n}/{arity}" for n in cand_names
            )
            continue
        if not cand_names:
            unaligned_siv_preds.extend(
                f"{n}/{arity}" for n in siv_names
            )
            continue

        matched_siv, matched_cand = _bipartite_match(
            siv_names, cand_names, threshold,
        )
        for s, c, score in matched_siv:
            pred_map[s] = c
            pred_scores[f"{s}/{arity}->{c}/{arity}"] = round(score, 4)

        # Unmatched
        matched_siv_set = {s for s, _, _ in matched_siv}
        matched_cand_set = {c for _, c, _ in matched_siv}
        for n in siv_names:
            if n not in matched_siv_set:
                unaligned_siv_preds.append(f"{n}/{arity}")
        for n in cand_names:
            if n not in matched_cand_set:
                unaligned_cand_preds.append(f"{n}/{arity}")

    # ── Constant alignment ──
    const_map: Dict[str, str] = {}
    const_scores: Dict[str, float] = {}
    unaligned_siv_consts: List[str] = []
    unaligned_cand_consts: List[str] = []

    siv_const_list = sorted(siv_consts)
    cand_const_list = sorted(cand_consts)

    if siv_const_list and cand_const_list:
        matched_consts, _ = _bipartite_match(
            siv_const_list, cand_const_list, threshold,
        )
        matched_siv_set = set()
        matched_cand_set = set()
        for s, c, score in matched_consts:
            const_map[s] = c
            const_scores[f"{s}->{c}"] = round(score, 4)
            matched_siv_set.add(s)
            matched_cand_set.add(c)
        for n in siv_const_list:
            if n not in matched_siv_set:
                unaligned_siv_consts.append(n)
        for n in cand_const_list:
            if n not in matched_cand_set:
                unaligned_cand_consts.append(n)
    else:
        unaligned_siv_consts = list(siv_const_list)
        unaligned_cand_consts = list(cand_const_list)

    return AlignmentResult(
        predicate_map=pred_map,
        constant_map=const_map,
        predicate_scores=pred_scores,
        constant_scores=const_scores,
        unaligned_siv_predicates=unaligned_siv_preds,
        unaligned_candidate_predicates=unaligned_cand_preds,
        unaligned_siv_constants=unaligned_siv_consts,
        unaligned_candidate_constants=unaligned_cand_consts,
        threshold=threshold,
    )


def _bipartite_match(
    siv_names: List[str],
    cand_names: List[str],
    threshold: float,
) -> Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]]]:
    """Run optimal bipartite matching on two name lists.

    Returns (accepted_pairs, all_pairs) where accepted_pairs have
    similarity >= threshold.
    """
    siv_processed = [_preprocess_name(n) for n in siv_names]
    cand_processed = [_preprocess_name(n) for n in cand_names]

    siv_vecs = _embed(siv_processed)
    cand_vecs = _embed(cand_processed)

    # Cosine similarity (vecs are L2-normalized)
    sim_matrix = siv_vecs @ cand_vecs.T
    cost_matrix = 1.0 - sim_matrix

    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    accepted: List[Tuple[str, str, float]] = []
    for r, c in zip(row_idx, col_idx):
        score = float(sim_matrix[r, c])
        if score >= threshold:
            accepted.append((siv_names[r], cand_names[c], score))

    return accepted, []


# ── Test suite rewriting ─────────────────────────────────────────────────────

def rewrite_test_suite(
    test_suite: TestSuite,
    alignment: AlignmentResult,
) -> TestSuite:
    """Apply alignment substitutions to test suite FOL strings.

    Returns a new ``TestSuite`` with rewritten positives and contrastives.
    The ``extraction`` field is preserved unchanged.  Identity mappings
    (where old == new) are kept in the alignment audit but skipped in the
    regex substitution.
    """
    pattern, subs = _build_substitution(alignment)

    if pattern is None:
        # No non-identity substitutions needed
        return test_suite

    new_positives = [
        UnitTest(
            fol=pattern.sub(lambda m: subs[m.group(0)], t.fol),
            kind=t.kind,
            mutation_kind=t.mutation_kind,
        )
        for t in test_suite.positives
    ]
    new_contrastives = [
        UnitTest(
            fol=pattern.sub(lambda m: subs[m.group(0)], t.fol),
            kind=t.kind,
            mutation_kind=t.mutation_kind,
        )
        for t in test_suite.contrastives
    ]

    return TestSuite(
        extraction=test_suite.extraction,
        positives=new_positives,
        contrastives=new_contrastives,
    )


def rewrite_fol_strings(
    fol_strings: List[str],
    alignment: AlignmentResult,
) -> List[str]:
    """Apply alignment substitutions to a list of FOL strings.

    Used for rewriting witness axioms with the same substitution map
    as the test suite.
    """
    pattern, subs = _build_substitution(alignment)
    if pattern is None:
        return list(fol_strings)
    return [pattern.sub(lambda m: subs[m.group(0)], s) for s in fol_strings]


def _build_substitution(
    alignment: AlignmentResult,
) -> Tuple[Optional["re.Pattern"], Dict[str, str]]:
    """Build a compiled regex and substitution map from an alignment.

    Skips identity mappings (old == new).  Returns (None, {}) if no
    non-identity substitutions exist.
    """
    subs: Dict[str, str] = {}
    pred_patterns: List[str] = []
    const_patterns: List[str] = []

    for old, new in alignment.predicate_map.items():
        if old != new:
            subs[old] = new
            pred_patterns.append(re.escape(old))

    for old, new in alignment.constant_map.items():
        if old != new:
            subs[old] = new
            const_patterns.append(re.escape(old))

    if not subs:
        return None, {}

    # Sort longest-first within each category to avoid partial matches
    pred_patterns.sort(key=len, reverse=True)
    const_patterns.sort(key=len, reverse=True)

    parts = []
    # Predicates: match before '('
    for p in pred_patterns:
        parts.append(rf"{p}(?=\s*\()")
    # Constants: match as whole word
    for c in const_patterns:
        parts.append(rf"\b{c}\b")

    combined = "|".join(parts)
    pattern = re.compile(combined)

    return pattern, subs


def alignment_to_dict(alignment: AlignmentResult) -> dict:
    """Serialize an AlignmentResult for JSON output."""
    return {
        "predicate_map": {
            k: {"candidate": v, "score": alignment.predicate_scores.get(
                next(
                    (key for key in alignment.predicate_scores if key.startswith(f"{k}/")),
                    "",
                ),
                0.0,
            )}
            for k, v in alignment.predicate_map.items()
        },
        "constant_map": {
            k: {"candidate": v, "score": alignment.constant_scores.get(
                f"{k}->{v}", 0.0,
            )}
            for k, v in alignment.constant_map.items()
        },
        "unaligned_siv_predicates": alignment.unaligned_siv_predicates,
        "unaligned_candidate_predicates": alignment.unaligned_candidate_predicates,
        "unaligned_siv_constants": alignment.unaligned_siv_constants,
        "unaligned_candidate_constants": alignment.unaligned_candidate_constants,
        "threshold": alignment.threshold,
    }
