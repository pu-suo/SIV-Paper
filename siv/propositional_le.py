"""Propositional Logical-Equivalence (LE) score.

The metric introduced by Yang et al. 2024 (MALLS, §4.3) collapses two
first-order formulas to propositional logic by stripping all quantifiers
and treating each unique predicate-application as a single propositional
atom. It then reports the fraction of truth assignments under which the
two propositional formulas evaluate to the same truth value.

Definition implemented here:

    1.  Parse both formulas to NLTK Expression objects.
    2.  Recursively strip every All/Exists, keeping the body. Bound
        variables are renamed to positional canonical names (v0, v1, ...)
        as we recurse, so alpha-equivalent formulas produce identical
        atom sets.
    3.  Each unique (predicate_name, arg_tuple) — *after* canonical
        renaming — becomes a single propositional variable, shared
        across both formulas where the canonical signature matches.
    4.  Enumerate the 2^k truth assignments over the union of atoms.
        LE = (rows where cand and gold evaluate equally) / 2^k.
        For k > 15 we fall back to Monte-Carlo sampling (10 000 random
        assignments) and report agreement rate.

Notes on design choices (worth recording because they diverge from a
literal reading of the MALLS paper):

  - Predicate-applications keep argument identity. Loves(a,b) and
    Loves(b,a) propositionalize to *distinct* atoms; they are NOT
    merged via edit-distance binding on the predicate symbol. This
    means the metric is sensitive to argument-order swaps. The MALLS
    paper's "greedy literal binding minimising predicate-name edit
    distance" is omitted because it would merge arg_swap candidates
    into the gold and report LE = 1.0 — a degenerate behaviour the
    user's test specification explicitly rejects.

  - Identical formulas → LE = 1.0; alpha-equivalent formulas → LE = 1.0
    (because bound vars are canonicalised before atom signatures are
    computed).

  - Returns the metric value in [0, 1] as a continuous score, *not*
    a binary verdict.
"""
from __future__ import annotations

import logging
from itertools import product
from typing import Dict, List, Optional, Tuple

from siv.fol_utils import NLTK_AVAILABLE, normalize_fol_string, parse_fol

logger = logging.getLogger(__name__)

if NLTK_AVAILABLE:
    from nltk.sem.logic import (
        AllExpression,
        AndExpression,
        ApplicationExpression,
        EqualityExpression,
        ExistsExpression,
        IffExpression,
        ImpExpression,
        IndividualVariableExpression,
        NegatedExpression,
        OrExpression,
    )

_TRUTH_TABLE_LIMIT = 15  # 2^15 = 32 768 rows; cheap enough
_MONTE_CARLO_SAMPLES = 10_000


# ── Argument canonicalisation ─────────────────────────────────────────────────


def _canonical_arg_string(arg, scope: Dict[str, str]) -> str:
    """Render an NLTK argument as a canonical string, renaming bound vars."""
    if isinstance(arg, IndividualVariableExpression):
        name = str(arg.variable)
        return scope.get(name, name)
    # Constant / free var: take the literal name. NLTK's str() renders
    # ConstantExpression as the bare symbol.
    s = str(arg)
    return scope.get(s, s)


def _atom_signature(app_expr, scope: Dict[str, str]) -> str:
    """Canonical signature for a predicate-application.

    NLTK represents P(a, b) as ApplicationExpression(P_at_a, b) i.e. curried,
    so we uncurry by walking outward until we hit the predicate head.
    """
    args: List = []
    head = app_expr
    while isinstance(head, ApplicationExpression):
        args.insert(0, head.argument)
        head = head.function

    if hasattr(head, "variable"):
        pred_name = head.variable.name
    else:
        pred_name = str(head)

    arg_strs = [_canonical_arg_string(a, scope) for a in args]
    return f"{pred_name}({','.join(arg_strs)})"


# ── Recursive: NLTK Expression → atom-id tree ─────────────────────────────────
#
# We represent the propositional reduction as a nested tuple:
#   ("atom", atom_id)
#   ("not", child)
#   ("and", child1, child2)
#   ("or", child1, child2)
#   ("imp", child1, child2)
#   ("iff", child1, child2)
#
# This keeps the tree small and avoids depending on sympy for evaluation.


def _build_prop_tree(
    expr,
    shared_atoms: Dict[str, int],
    scope: Dict[str, str],
) -> tuple:
    """Recursively convert an NLTK expression to a propositional AST.

    Quantifiers are stripped (their bodies replace them). Bound variables
    are renamed to v0, v1, ... in `scope`, so alpha-equivalent formulas
    produce identical atom signatures.
    """
    if isinstance(expr, (AllExpression, ExistsExpression)):
        old_name = str(expr.variable)
        new_name = f"_v{len(scope)}"
        new_scope = {**scope, old_name: new_name}
        return _build_prop_tree(expr.term, shared_atoms, new_scope)

    if isinstance(expr, NegatedExpression):
        return ("not", _build_prop_tree(expr.term, shared_atoms, scope))

    if isinstance(expr, AndExpression):
        return (
            "and",
            _build_prop_tree(expr.first, shared_atoms, scope),
            _build_prop_tree(expr.second, shared_atoms, scope),
        )

    if isinstance(expr, OrExpression):
        return (
            "or",
            _build_prop_tree(expr.first, shared_atoms, scope),
            _build_prop_tree(expr.second, shared_atoms, scope),
        )

    if isinstance(expr, ImpExpression):
        return (
            "imp",
            _build_prop_tree(expr.first, shared_atoms, scope),
            _build_prop_tree(expr.second, shared_atoms, scope),
        )

    if isinstance(expr, IffExpression):
        return (
            "iff",
            _build_prop_tree(expr.first, shared_atoms, scope),
            _build_prop_tree(expr.second, shared_atoms, scope),
        )

    if isinstance(expr, EqualityExpression):
        # Treat equality (a = b) as a distinct propositional atom keyed on
        # the canonical operand strings.
        lhs = _canonical_arg_string(expr.first, scope)
        rhs = _canonical_arg_string(expr.second, scope)
        sig = f"=({lhs},{rhs})"
        if sig not in shared_atoms:
            shared_atoms[sig] = len(shared_atoms)
        return ("atom", shared_atoms[sig])

    if isinstance(expr, ApplicationExpression):
        sig = _atom_signature(expr, scope)
        if sig not in shared_atoms:
            shared_atoms[sig] = len(shared_atoms)
        return ("atom", shared_atoms[sig])

    raise ValueError(f"Unsupported NLTK expression type: {type(expr).__name__}")


def _eval_tree(tree: tuple, assignment: tuple) -> bool:
    """Evaluate a propositional tree under a tuple-indexed truth assignment."""
    tag = tree[0]
    if tag == "atom":
        return assignment[tree[1]]
    if tag == "not":
        return not _eval_tree(tree[1], assignment)
    if tag == "and":
        return _eval_tree(tree[1], assignment) and _eval_tree(tree[2], assignment)
    if tag == "or":
        return _eval_tree(tree[1], assignment) or _eval_tree(tree[2], assignment)
    if tag == "imp":
        return (not _eval_tree(tree[1], assignment)) or _eval_tree(tree[2], assignment)
    if tag == "iff":
        return _eval_tree(tree[1], assignment) == _eval_tree(tree[2], assignment)
    raise ValueError(f"Unknown prop-tree tag: {tag}")


def _agreement_rate(
    cand_tree: tuple,
    gold_tree: tuple,
    k: int,
) -> float:
    """Fraction of truth assignments under which both trees evaluate equally."""
    if k == 0:
        # Both formulas reduced to constants — they're equivalent iff they
        # evaluate the same way (which they will, since there are no atoms
        # to vary). Return 1.0 to avoid 0/0.
        return 1.0

    if k <= _TRUTH_TABLE_LIMIT:
        total = 1 << k
        agree = 0
        for assignment in product((False, True), repeat=k):
            if _eval_tree(cand_tree, assignment) == _eval_tree(gold_tree, assignment):
                agree += 1
        return agree / total

    # Monte-Carlo fallback for large atom unions.
    logger.warning(
        "propositional_le: %d atoms; using %d-sample Monte-Carlo agreement",
        k, _MONTE_CARLO_SAMPLES,
    )
    import random
    rng = random.Random(42)
    agree = 0
    for _ in range(_MONTE_CARLO_SAMPLES):
        assignment = tuple(rng.random() < 0.5 for _ in range(k))
        if _eval_tree(cand_tree, assignment) == _eval_tree(gold_tree, assignment):
            agree += 1
    return agree / _MONTE_CARLO_SAMPLES


# ── Public API ────────────────────────────────────────────────────────────────


def propositional_le(
    candidate_fol: str,
    gold_fol: str,
    timeout: int = 10,  # noqa: ARG001  (kept for wrapper-signature parity)
) -> Optional[float]:
    """Compute the propositional LE score (Yang-style) in [0, 1].

    Returns None on parse failure on either input.
    """
    if not NLTK_AVAILABLE:
        return None

    cand_norm = normalize_fol_string(candidate_fol)
    gold_norm = normalize_fol_string(gold_fol)
    if not cand_norm or not gold_norm:
        return None

    cand_expr = parse_fol(cand_norm)
    gold_expr = parse_fol(gold_norm)
    if cand_expr is None or gold_expr is None:
        return None

    try:
        shared_atoms: Dict[str, int] = {}
        cand_tree = _build_prop_tree(cand_expr, shared_atoms, scope={})
        gold_tree = _build_prop_tree(gold_expr, shared_atoms, scope={})
    except ValueError as e:
        logger.warning("propositional_le: build failed (%s)", e)
        return None

    k = len(shared_atoms)
    return _agreement_rate(cand_tree, gold_tree, k)


def propositional_le_aligned(
    candidate_fol: str,
    gold_fol: str,
    timeout: int = 10,
) -> Optional[float]:
    """LE score after applying symbol alignment to the candidate vocabulary."""
    from siv.aligner import align_symbols, extract_symbols_from_fol

    cand_norm = normalize_fol_string(candidate_fol)
    gold_norm = normalize_fol_string(gold_fol)
    if not cand_norm or not gold_norm:
        return None

    gold_symbols = extract_symbols_from_fol(gold_norm)
    cand_symbols = extract_symbols_from_fol(cand_norm)
    alignment = align_symbols(gold_symbols, cand_symbols)

    rename_map: Dict[str, str] = {}
    for gold_name, cand_name in alignment.predicate_map.items():
        if gold_name != cand_name:
            rename_map[cand_name] = gold_name
    for gold_name, cand_name in alignment.constant_map.items():
        if gold_name != cand_name:
            rename_map[cand_name] = gold_name

    aligned_cand = cand_norm
    if rename_map:
        import re
        pattern = re.compile(
            r"|".join(
                rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])"
                for old in sorted(rename_map, key=len, reverse=True)
            )
        )
        aligned_cand = pattern.sub(
            lambda m: rename_map[m.group(0)], aligned_cand
        )

    return propositional_le(aligned_cand, gold_norm, timeout=timeout)


def propositional_le_batch(
    candidates: List[str],
    golds: List[str],
    timeout: int = 10,
    aligned: bool = False,
) -> Dict[str, object]:
    """Batch propositional LE over parallel premise lists.

    Returns {"mean": float, "per_premise": [float|None, ...]}.
    """
    if len(candidates) != len(golds):
        raise ValueError("candidates and golds must have the same length")

    fn = propositional_le_aligned if aligned else propositional_le
    per_premise = [fn(c, g, timeout=timeout) for c, g in zip(candidates, golds)]

    scored = [v for v in per_premise if v is not None]
    mean = sum(scored) / len(scored) if scored else 0.0

    return {"mean": mean, "per_premise": per_premise}
