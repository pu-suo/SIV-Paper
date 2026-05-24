"""
AST-level perturbation functions for NLTK FOL Expressions.

Each perturbation takes a parsed NLTK ``Expression`` and returns a modified
``Expression``, or raises ``NotApplicable`` if the transformation does not
fit the formula's structure.  Perturbations are grouped into four tiers:

  - **Tier A**: Subtle, debatable (reasonable annotators may disagree).
  - **Tier B**: Meaning-altering, lexically close (paper's Exhibit A).
  - **Tier C**: Clearly wrong but fluent.
  - **Tier D**: Nonsense.

All perturbations operate on the parsed *gold* FOL — never on SIV's
``SentenceExtraction`` schema.  They are deterministic given a fixed RNG seed.
"""
from __future__ import annotations

import random
import re
import string
from typing import Dict, List, Optional, Set, Tuple

from siv.fol_utils import parse_fol, NLTK_AVAILABLE

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
        Variable,
    )

    read_expr = Expression.fromstring


# ── Exception ────────────────────────────────────────────────────────────────


class NotApplicable(Exception):
    """Raised when a perturbation cannot be applied to the given expression."""


# ── Helpers ──────────────────────────────────────────────────────────────────


def _uncurry(expr: "ApplicationExpression") -> Tuple:
    """Uncurry a chain of ApplicationExpressions into (head, [args])."""
    func = expr
    args = []
    while isinstance(func, ApplicationExpression):
        args.insert(0, func.argument)
        func = func.function
    return func, args


def _curry(head, args: list) -> "Expression":
    """Rebuild a curried ApplicationExpression from head and args."""
    result = head
    for arg in args:
        result = ApplicationExpression(result, arg)
    return result


def _pred_name(expr: "ApplicationExpression") -> str:
    """Get the predicate name from an ApplicationExpression."""
    head, _ = _uncurry(expr)
    return str(head)


def _find_predicates(expr) -> List[Tuple[str, int]]:
    """Walk AST and return (pred_name, arity) pairs found."""
    results = []
    if isinstance(expr, ApplicationExpression):
        head, args = _uncurry(expr)
        results.append((str(head), len(args)))
        for a in args:
            results.extend(_find_predicates(a))
    elif isinstance(expr, BinaryExpression):
        results.extend(_find_predicates(expr.first))
        results.extend(_find_predicates(expr.second))
    elif isinstance(expr, NegatedExpression):
        results.extend(_find_predicates(expr.term))
    elif isinstance(expr, (AllExpression, ExistsExpression)):
        results.extend(_find_predicates(expr.term))
    elif hasattr(expr, "term"):
        results.extend(_find_predicates(expr.term))
    return results


def _find_constants(expr) -> Set[str]:
    """Walk AST and return set of constant names."""
    results = set()
    if isinstance(expr, ApplicationExpression):
        _, args = _uncurry(expr)
        for a in args:
            if isinstance(a, ConstantExpression):
                results.add(str(a))
            else:
                results.update(_find_constants(a))
    elif isinstance(expr, BinaryExpression):
        results.update(_find_constants(expr.first))
        results.update(_find_constants(expr.second))
    elif isinstance(expr, NegatedExpression):
        results.update(_find_constants(expr.term))
    elif isinstance(expr, (AllExpression, ExistsExpression)):
        results.update(_find_constants(expr.term))
    elif isinstance(expr, ConstantExpression):
        results.add(str(expr))
    elif hasattr(expr, "term"):
        results.update(_find_constants(expr.term))
    return results


def _camel_split(name: str) -> List[str]:
    """Split CamelCase into components: 'ProfessionalTennisPlayer' -> ['Professional', 'Tennis', 'Player']."""
    parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    parts = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", parts)
    return parts.split()


def _replace_pred_name(expr, old_name: str, new_name: str):
    """Replace predicate name throughout the AST."""
    if isinstance(expr, ApplicationExpression):
        head, args = _uncurry(expr)
        if str(head) == old_name:
            new_head = read_expr(new_name)
            new_args = [_replace_pred_name(a, old_name, new_name) for a in args]
            return _curry(new_head, new_args)
        new_args = [_replace_pred_name(a, old_name, new_name) for a in args]
        return _curry(head, new_args)
    elif isinstance(expr, AndExpression):
        return AndExpression(
            _replace_pred_name(expr.first, old_name, new_name),
            _replace_pred_name(expr.second, old_name, new_name),
        )
    elif isinstance(expr, OrExpression):
        return OrExpression(
            _replace_pred_name(expr.first, old_name, new_name),
            _replace_pred_name(expr.second, old_name, new_name),
        )
    elif isinstance(expr, ImpExpression):
        return ImpExpression(
            _replace_pred_name(expr.first, old_name, new_name),
            _replace_pred_name(expr.second, old_name, new_name),
        )
    elif isinstance(expr, IffExpression):
        return IffExpression(
            _replace_pred_name(expr.first, old_name, new_name),
            _replace_pred_name(expr.second, old_name, new_name),
        )
    elif isinstance(expr, NegatedExpression):
        return NegatedExpression(_replace_pred_name(expr.term, old_name, new_name))
    elif isinstance(expr, AllExpression):
        return AllExpression(expr.variable, _replace_pred_name(expr.term, old_name, new_name))
    elif isinstance(expr, ExistsExpression):
        return ExistsExpression(expr.variable, _replace_pred_name(expr.term, old_name, new_name))
    return expr


def _replace_constant(expr, old_name: str, new_name: str):
    """Replace a constant name throughout the AST."""
    if isinstance(expr, ConstantExpression) and str(expr) == old_name:
        return read_expr(new_name)
    elif isinstance(expr, ApplicationExpression):
        head, args = _uncurry(expr)
        new_args = [_replace_constant(a, old_name, new_name) for a in args]
        new_head = _replace_constant(head, old_name, new_name)
        return _curry(new_head, new_args)
    elif isinstance(expr, AndExpression):
        return AndExpression(
            _replace_constant(expr.first, old_name, new_name),
            _replace_constant(expr.second, old_name, new_name),
        )
    elif isinstance(expr, OrExpression):
        return OrExpression(
            _replace_constant(expr.first, old_name, new_name),
            _replace_constant(expr.second, old_name, new_name),
        )
    elif isinstance(expr, ImpExpression):
        return ImpExpression(
            _replace_constant(expr.first, old_name, new_name),
            _replace_constant(expr.second, old_name, new_name),
        )
    elif isinstance(expr, IffExpression):
        return IffExpression(
            _replace_constant(expr.first, old_name, new_name),
            _replace_constant(expr.second, old_name, new_name),
        )
    elif isinstance(expr, NegatedExpression):
        return NegatedExpression(_replace_constant(expr.term, old_name, new_name))
    elif isinstance(expr, AllExpression):
        return AllExpression(expr.variable, _replace_constant(expr.term, old_name, new_name))
    elif isinstance(expr, ExistsExpression):
        return ExistsExpression(expr.variable, _replace_constant(expr.term, old_name, new_name))
    return expr


# ── Tier A — subtle ──────────────────────────────────────────────────────────


def A_arity_decompose(expr: "Expression") -> "Expression":
    """``P(x, c)`` → ``P_c(x)``: fold a constant argument into the predicate name.

    Finds the first binary predicate with a ``ConstantExpression`` argument,
    creates a new unary predicate incorporating the constant, and removes
    the constant argument.
    """
    found = _find_binary_with_constant(expr)
    if found is None:
        raise NotApplicable("No binary predicate with constant argument")
    pred_name, const_name, const_is_second = found
    new_pred = pred_name + const_name[0].upper() + const_name[1:]
    return _apply_arity_decompose(expr, pred_name, const_name, new_pred, const_is_second)


def _find_binary_with_constant(expr) -> Optional[Tuple[str, str, bool]]:
    """Find first binary predicate with a constant arg. Returns (pred, const, const_is_second)."""
    if isinstance(expr, ApplicationExpression):
        head, args = _uncurry(expr)
        if len(args) == 2:
            if isinstance(args[1], ConstantExpression):
                return (str(head), str(args[1]), True)
            if isinstance(args[0], ConstantExpression):
                return (str(head), str(args[0]), False)
        for a in args:
            r = _find_binary_with_constant(a)
            if r:
                return r
    elif isinstance(expr, BinaryExpression):
        r = _find_binary_with_constant(expr.first)
        if r:
            return r
        return _find_binary_with_constant(expr.second)
    elif isinstance(expr, NegatedExpression):
        return _find_binary_with_constant(expr.term)
    elif isinstance(expr, (AllExpression, ExistsExpression)):
        return _find_binary_with_constant(expr.term)
    return None


def _apply_arity_decompose(expr, pred_name, const_name, new_pred, const_is_second):
    """Recursively apply the arity decomposition."""
    if isinstance(expr, ApplicationExpression):
        head, args = _uncurry(expr)
        if str(head) == pred_name and len(args) == 2:
            if const_is_second and isinstance(args[1], ConstantExpression) and str(args[1]) == const_name:
                return _curry(read_expr(new_pred), [args[0]])
            if not const_is_second and isinstance(args[0], ConstantExpression) and str(args[0]) == const_name:
                return _curry(read_expr(new_pred), [args[1]])
        new_args = [_apply_arity_decompose(a, pred_name, const_name, new_pred, const_is_second) for a in args]
        return _curry(head, new_args)
    elif isinstance(expr, AndExpression):
        return AndExpression(
            _apply_arity_decompose(expr.first, pred_name, const_name, new_pred, const_is_second),
            _apply_arity_decompose(expr.second, pred_name, const_name, new_pred, const_is_second),
        )
    elif isinstance(expr, OrExpression):
        return OrExpression(
            _apply_arity_decompose(expr.first, pred_name, const_name, new_pred, const_is_second),
            _apply_arity_decompose(expr.second, pred_name, const_name, new_pred, const_is_second),
        )
    elif isinstance(expr, ImpExpression):
        return ImpExpression(
            _apply_arity_decompose(expr.first, pred_name, const_name, new_pred, const_is_second),
            _apply_arity_decompose(expr.second, pred_name, const_name, new_pred, const_is_second),
        )
    elif isinstance(expr, IffExpression):
        return IffExpression(
            _apply_arity_decompose(expr.first, pred_name, const_name, new_pred, const_is_second),
            _apply_arity_decompose(expr.second, pred_name, const_name, new_pred, const_is_second),
        )
    elif isinstance(expr, NegatedExpression):
        return NegatedExpression(
            _apply_arity_decompose(expr.term, pred_name, const_name, new_pred, const_is_second)
        )
    elif isinstance(expr, AllExpression):
        return AllExpression(
            expr.variable,
            _apply_arity_decompose(expr.term, pred_name, const_name, new_pred, const_is_second),
        )
    elif isinstance(expr, ExistsExpression):
        return ExistsExpression(
            expr.variable,
            _apply_arity_decompose(expr.term, pred_name, const_name, new_pred, const_is_second),
        )
    return expr


def A_const_to_unary(expr: "Expression") -> "Expression":
    """``Has(x, fever)`` → ``HasFever(x)``: merge constant into predicate name as unary."""
    found = _find_binary_with_constant(expr)
    if found is None:
        raise NotApplicable("No predicate with constant argument")
    pred_name, const_name, const_is_second = found
    new_pred = pred_name + const_name[0].upper() + const_name[1:]
    return _apply_arity_decompose(expr, pred_name, const_name, new_pred, const_is_second)


def A_compound_decompose(expr: "Expression") -> "Expression":
    """``ProfessionalTennisPlayer(x)`` → ``(Professional(x) & TennisPlayer(x))``.

    Splits a CamelCase compound predicate name into component predicates.
    Only applies to unary predicates with ≥2 CamelCase components.
    """
    preds = _find_predicates(expr)
    for name, arity in preds:
        if arity != 1:
            continue
        parts = _camel_split(name)
        if len(parts) >= 2:
            return _apply_compound_decompose(expr, name, parts)
    raise NotApplicable("No compound CamelCase unary predicate found")


def _apply_compound_decompose(expr, old_name: str, parts: List[str]):
    """Replace unary pred with conjunction of component predicates."""
    if isinstance(expr, ApplicationExpression):
        head, args = _uncurry(expr)
        if str(head) == old_name and len(args) == 1:
            arg = args[0]
            # Build conjunction: Part1(arg) & Part2(arg) & ...
            conjuncts = [_curry(read_expr(p), [arg]) for p in parts]
            result = conjuncts[0]
            for c in conjuncts[1:]:
                result = AndExpression(result, c)
            return result
        new_args = [_apply_compound_decompose(a, old_name, parts) for a in args]
        return _curry(head, new_args)
    elif isinstance(expr, AndExpression):
        return AndExpression(
            _apply_compound_decompose(expr.first, old_name, parts),
            _apply_compound_decompose(expr.second, old_name, parts),
        )
    elif isinstance(expr, OrExpression):
        return OrExpression(
            _apply_compound_decompose(expr.first, old_name, parts),
            _apply_compound_decompose(expr.second, old_name, parts),
        )
    elif isinstance(expr, ImpExpression):
        return ImpExpression(
            _apply_compound_decompose(expr.first, old_name, parts),
            _apply_compound_decompose(expr.second, old_name, parts),
        )
    elif isinstance(expr, IffExpression):
        return IffExpression(
            _apply_compound_decompose(expr.first, old_name, parts),
            _apply_compound_decompose(expr.second, old_name, parts),
        )
    elif isinstance(expr, NegatedExpression):
        return NegatedExpression(_apply_compound_decompose(expr.term, old_name, parts))
    elif isinstance(expr, AllExpression):
        return AllExpression(expr.variable, _apply_compound_decompose(expr.term, old_name, parts))
    elif isinstance(expr, ExistsExpression):
        return ExistsExpression(expr.variable, _apply_compound_decompose(expr.term, old_name, parts))
    return expr


def A_const_rename(expr: "Expression", rng: random.Random) -> "Expression":
    """Stylistic constant rename: ``theMixer`` → ``mixer``, ``summerOlympics2008`` → ``olym2008``."""
    consts = sorted(_find_constants(expr))
    if not consts:
        raise NotApplicable("No constants to rename")
    target = consts[0]
    # Generate a plausible rename: take first 4 chars + optional suffix
    base = target[:4].lower()
    suffix = str(rng.randint(1, 99))
    new_name = base + suffix
    # Ensure it's a valid NLTK constant (starts with lowercase letter)
    if not new_name[0].isalpha():
        new_name = "c" + new_name
    return _replace_constant(expr, target, new_name)


# ── Tier B — meaning-altering ────────────────────────────────────────────────


SYMMETRIC_PREDICATES = {"Equal", "SameAs", "Similar", "Adjacent", "Married", "Sibling"}


def B_arg_swap(expr: "Expression") -> "Expression":
    """``P(a, b)`` → ``P(b, a)``: swap arguments of the first binary predicate."""
    swapped, did_swap = _swap_first_binary_args(expr)
    if not did_swap:
        raise NotApplicable("No binary predicate to swap")
    return swapped


def _swap_first_binary_args(expr, done=False):
    """Find and swap the first binary predicate's arguments."""
    if done:
        return expr, True
    if isinstance(expr, ApplicationExpression):
        head, args = _uncurry(expr)
        if len(args) == 2 and str(head) not in SYMMETRIC_PREDICATES:
            swapped = _curry(head, [args[1], args[0]])
            return swapped, True
        return expr, False
    elif isinstance(expr, AndExpression):
        new_first, d = _swap_first_binary_args(expr.first)
        if d:
            return AndExpression(new_first, expr.second), True
        new_second, d = _swap_first_binary_args(expr.second)
        return AndExpression(expr.first, new_second), d
    elif isinstance(expr, OrExpression):
        new_first, d = _swap_first_binary_args(expr.first)
        if d:
            return OrExpression(new_first, expr.second), True
        new_second, d = _swap_first_binary_args(expr.second)
        return OrExpression(expr.first, new_second), d
    elif isinstance(expr, ImpExpression):
        new_first, d = _swap_first_binary_args(expr.first)
        if d:
            return ImpExpression(new_first, expr.second), True
        new_second, d = _swap_first_binary_args(expr.second)
        return ImpExpression(expr.first, new_second), d
    elif isinstance(expr, IffExpression):
        new_first, d = _swap_first_binary_args(expr.first)
        if d:
            return IffExpression(new_first, expr.second), True
        new_second, d = _swap_first_binary_args(expr.second)
        return IffExpression(expr.first, new_second), d
    elif isinstance(expr, NegatedExpression):
        new_term, d = _swap_first_binary_args(expr.term)
        return NegatedExpression(new_term), d
    elif isinstance(expr, AllExpression):
        new_term, d = _swap_first_binary_args(expr.term)
        return AllExpression(expr.variable, new_term), d
    elif isinstance(expr, ExistsExpression):
        new_term, d = _swap_first_binary_args(expr.term)
        return ExistsExpression(expr.variable, new_term), d
    return expr, False


def B_restrictor_drop(expr: "Expression") -> "Expression":
    """Drop one conjunct from the antecedent of a universal implication.

    ``all x.((A(x) & B(x)) -> C(x))`` → ``all x.(A(x) -> C(x))``
    """
    if not isinstance(expr, AllExpression):
        raise NotApplicable("Not a universal formula")
    body = expr.term
    if not isinstance(body, ImpExpression):
        raise NotApplicable("Universal body is not an implication")
    antecedent = body.first
    conjuncts = _flatten_and(antecedent)
    if len(conjuncts) < 2:
        raise NotApplicable("Antecedent has fewer than 2 conjuncts")
    # Drop the last conjunct
    remaining = conjuncts[:-1]
    new_ante = remaining[0]
    for c in remaining[1:]:
        new_ante = AndExpression(new_ante, c)
    return AllExpression(expr.variable, ImpExpression(new_ante, body.second))


def _flatten_and(expr) -> list:
    """Flatten nested AndExpressions into a list of conjuncts."""
    if isinstance(expr, AndExpression):
        return _flatten_and(expr.first) + _flatten_and(expr.second)
    return [expr]


def B_restrictor_add(expr: "Expression", story_predicates: List[str]) -> "Expression":
    """Add an extra conjunct to the antecedent from another story predicate.

    ``all x.(A(x) -> C(x))`` → ``all x.((A(x) & Extra(x)) -> C(x))``
    """
    if not isinstance(expr, AllExpression):
        raise NotApplicable("Not a universal formula")
    body = expr.term
    if not isinstance(body, ImpExpression):
        raise NotApplicable("Universal body is not an implication")

    existing_preds = {name for name, _ in _find_predicates(expr)}
    available = [p for p in story_predicates if p not in existing_preds]
    if not available:
        raise NotApplicable("No available predicates from story context")

    extra_pred = available[0]
    # Build Extra(bound_var) using the universal's bound variable
    bound_var = expr.variable
    extra_atom = read_expr(f"{extra_pred}({bound_var})")
    new_ante = AndExpression(body.first, extra_atom)
    return AllExpression(expr.variable, ImpExpression(new_ante, body.second))


def B_scope_flip(expr: "Expression") -> "Expression":
    """Swap the order of two nested quantifiers.

    ``all x.(exists y.R(x,y))`` → ``exists y.(all x.R(x,y))``
    """
    if isinstance(expr, AllExpression) and isinstance(expr.term, ExistsExpression):
        inner = expr.term
        return ExistsExpression(inner.variable, AllExpression(expr.variable, inner.term))
    if isinstance(expr, ExistsExpression) and isinstance(expr.term, AllExpression):
        inner = expr.term
        return AllExpression(inner.variable, ExistsExpression(expr.variable, inner.term))
    # Also handle: all x.(P(x) -> exists y.Q(x,y)) → exists y.(all x.(P(x) -> Q(x,y)))
    if isinstance(expr, AllExpression) and isinstance(expr.term, ImpExpression):
        consequent = expr.term.second
        if isinstance(consequent, ExistsExpression):
            new_body = ImpExpression(expr.term.first, consequent.term)
            return ExistsExpression(
                consequent.variable,
                AllExpression(expr.variable, new_body),
            )
    raise NotApplicable("No two nested quantifiers to flip")


def B_quantifier_swap(expr: "Expression") -> "Expression":
    """Swap the outermost quantifier type: ``∀`` → ``∃`` or vice versa."""
    if isinstance(expr, AllExpression):
        return ExistsExpression(expr.variable, expr.term)
    if isinstance(expr, ExistsExpression):
        return AllExpression(expr.variable, expr.term)
    raise NotApplicable("No top-level quantifier to swap")


# ── Tier C — clearly wrong ──────────────────────────────────────────────────


ANTONYM_LEXICON: Dict[str, str] = {
    "Tall": "Short", "Short": "Tall",
    "Happy": "Sad", "Sad": "Happy",
    "Rich": "Poor", "Poor": "Rich",
    "Strong": "Weak", "Weak": "Strong",
    "Love": "Hate", "Hate": "Love",
    "Loves": "Hates", "Hates": "Loves",
    "Like": "Dislike", "Dislike": "Like",
    "Likes": "Dislikes", "Dislikes": "Likes",
    "Before": "After", "After": "Before",
    "Above": "Below", "Below": "Above",
    "Taller": "Shorter", "Shorter": "Taller",
    "Larger": "Smaller", "Smaller": "Larger",
    "LocatedIn": "NotIn",
    "Accept": "Reject", "Reject": "Accept",
    "Win": "Lose", "Lose": "Win",
    "True": "False", "False": "True",
    "Good": "Bad", "Bad": "Good",
    "Fast": "Slow", "Slow": "Fast",
    "Hot": "Cold", "Cold": "Hot",
    "Old": "Young", "Young": "Old",
    "Cheap": "Expensive", "Expensive": "Cheap",
    "Safe": "Dangerous", "Dangerous": "Safe",
    "Legal": "Illegal", "Illegal": "Legal",
    "Dependent": "Independent", "Independent": "Dependent",
    "Aware": "Unaware", "Unaware": "Aware",
}


def C_predicate_substitute(
    expr: "Expression",
    antonym_lexicon: Optional[Dict[str, str]] = None,
) -> "Expression":
    """Swap one predicate for its antonym from the lexicon."""
    lexicon = antonym_lexicon or ANTONYM_LEXICON
    preds = _find_predicates(expr)
    for name, _ in preds:
        if name in lexicon:
            return _replace_pred_name(expr, name, lexicon[name])
    raise NotApplicable("No predicate has a known antonym")


def C_negation_drop(expr: "Expression") -> "Expression":
    """Remove the first ``NegatedExpression`` found in the AST."""
    result, did_drop = _drop_first_negation(expr)
    if not did_drop:
        raise NotApplicable("No negation to drop")
    return result


def _drop_first_negation(expr, done=False):
    if done:
        return expr, True
    if isinstance(expr, NegatedExpression):
        return expr.term, True
    elif isinstance(expr, AndExpression):
        new_first, d = _drop_first_negation(expr.first)
        if d:
            return AndExpression(new_first, expr.second), True
        new_second, d = _drop_first_negation(expr.second)
        return AndExpression(expr.first, new_second), d
    elif isinstance(expr, OrExpression):
        new_first, d = _drop_first_negation(expr.first)
        if d:
            return OrExpression(new_first, expr.second), True
        new_second, d = _drop_first_negation(expr.second)
        return OrExpression(expr.first, new_second), d
    elif isinstance(expr, ImpExpression):
        new_first, d = _drop_first_negation(expr.first)
        if d:
            return ImpExpression(new_first, expr.second), True
        new_second, d = _drop_first_negation(expr.second)
        return ImpExpression(expr.first, new_second), d
    elif isinstance(expr, IffExpression):
        new_first, d = _drop_first_negation(expr.first)
        if d:
            return IffExpression(new_first, expr.second), True
        new_second, d = _drop_first_negation(expr.second)
        return IffExpression(expr.first, new_second), d
    elif isinstance(expr, AllExpression):
        new_term, d = _drop_first_negation(expr.term)
        return AllExpression(expr.variable, new_term), d
    elif isinstance(expr, ExistsExpression):
        new_term, d = _drop_first_negation(expr.term)
        return ExistsExpression(expr.variable, new_term), d
    return expr, False


def C_entity_swap(expr: "Expression", story_constants: List[str]) -> "Expression":
    """Replace one constant with a different constant from the same story."""
    existing = sorted(_find_constants(expr))
    if not existing:
        raise NotApplicable("No constants in expression")
    target = existing[0]
    available = [c for c in story_constants if c != target and c not in existing]
    if not available:
        raise NotApplicable("No different constant available in story")
    replacement = available[0]
    return _replace_constant(expr, target, replacement)


# ── Tier D — nonsense ────────────────────────────────────────────────────────


def D_random_predicates(expr: "Expression", rng: random.Random) -> "Expression":
    """Replace all predicate names with random 6-character strings."""
    preds = _find_predicates(expr)
    if not preds:
        raise NotApplicable("No predicates in expression")
    # Build deterministic mapping
    name_map: Dict[str, str] = {}
    for name, _ in preds:
        if name not in name_map:
            rand_name = "".join(rng.choices(string.ascii_uppercase, k=1)) + \
                        "".join(rng.choices(string.ascii_lowercase + string.digits, k=5))
            name_map[name] = rand_name
    result = expr
    for old, new in name_map.items():
        result = _replace_pred_name(result, old, new)
    return result


# ── Dispatch ─────────────────────────────────────────────────────────────────

TIER_A_OPS = [A_arity_decompose, A_const_to_unary, A_compound_decompose, A_const_rename]
TIER_B_OPS = [B_arg_swap, B_restrictor_drop, B_restrictor_add, B_scope_flip, B_quantifier_swap]
TIER_C_OPS = [C_predicate_substitute, C_negation_drop, C_entity_swap]
TIER_D_OPS = [D_random_predicates]

_TIER_MAP = {"A": TIER_A_OPS, "B": TIER_B_OPS, "C": TIER_C_OPS, "D": TIER_D_OPS}

# Operators that need extra keyword args
_NEEDS_RNG = {A_const_rename, D_random_predicates}
_NEEDS_STORY_PREDS = {B_restrictor_add}
_NEEDS_STORY_CONSTS = {C_entity_swap}
_NEEDS_LEXICON = {C_predicate_substitute}


def select_perturbation(
    tier: str,
    expr: "Expression",
    rng: random.Random,
    story_predicates: Optional[List[str]] = None,
    story_constants: Optional[List[str]] = None,
    antonym_lexicon: Optional[Dict[str, str]] = None,
    exclude_ops: Optional[Set[str]] = None,
) -> Tuple["Expression", str]:
    """Try each operator in the tier (shuffled by rng) until one succeeds.

    Returns ``(perturbed_expr, operator_name)``.
    Raises ``NotApplicable`` if no operator in the tier can apply.

    *exclude_ops* can be used to skip specific operators (e.g., when
    generating a second Tier B perturbation different from the first).
    """
    ops = list(_TIER_MAP.get(tier, []))
    if not ops:
        raise NotApplicable(f"Unknown tier: {tier}")

    rng_copy = random.Random(rng.randint(0, 2**31))  # Don't mutate caller's rng state unpredictably
    rng_copy.shuffle(ops)

    excluded = exclude_ops or set()

    for op in ops:
        if op.__name__ in excluded:
            continue
        try:
            kwargs = {}
            if op in _NEEDS_RNG:
                kwargs["rng"] = random.Random(rng.randint(0, 2**31))
            if op in _NEEDS_STORY_PREDS:
                kwargs["story_predicates"] = story_predicates or []
            if op in _NEEDS_STORY_CONSTS:
                kwargs["story_constants"] = story_constants or []
            if op in _NEEDS_LEXICON:
                kwargs["antonym_lexicon"] = antonym_lexicon

            result = op(expr, **kwargs)

            # Validate round-trip
            result_str = str(result)
            reparsed = parse_fol(result_str)
            if reparsed is None:
                continue  # Skip this operator if output doesn't reparse

            return result, op.__name__
        except NotApplicable:
            continue

    raise NotApplicable(f"No Tier {tier} operator applicable")


# ════════════════════════════════════════════════════════════════════════════
# severity_correlation_v1 operators (FROZEN — pre-registration design)
# ════════════════════════════════════════════════════════════════════════════
#
# 15 deterministic AST-transform operators for paper Exp 1's redesigned
# severity-correlation experiment. Catalog is the authoritative spec:
# ``configs/severity_correlation_v1.yaml``.
#
# These stubs are committed at the design-freeze commit. Bodies raise
# ``NotImplementedError`` until the implementation phase fills them in.
# Any change to operator semantics, tier mapping, or applicability after
# the freeze requires a documented amendment commit.
#
# Each operator has a paired ``<NAME>_applies_to(expr) -> bool`` predicate
# for filtering before transform. Tests verify both transform behavior
# and applicability filtering.

# ── Internal helpers for severity_correlation_v1 ────────────────────────────


def _synthesize_aux_predicate(expr: "Expression") -> str:
    """Pick a deterministic ``Aux<n>`` name not in the formula's predicate set."""
    existing = {name for name, _ in _find_predicates(expr)}
    i = 1
    while f"Aux{i}" in existing:
        i += 1
    return f"Aux{i}"


def _flatten_or(expr) -> list:
    """Flatten nested OrExpressions into a list of disjuncts."""
    if isinstance(expr, OrExpression):
        return _flatten_or(expr.first) + _flatten_or(expr.second)
    return [expr]


def _build_and_from_list(atoms: list) -> "Expression":
    """Build a left-associated AndExpression from a non-empty list of atoms."""
    result = atoms[0]
    for a in atoms[1:]:
        result = AndExpression(result, a)
    return result


def _build_or_from_list(atoms: list) -> "Expression":
    """Build a left-associated OrExpression from a non-empty list of atoms."""
    result = atoms[0]
    for a in atoms[1:]:
        result = OrExpression(result, a)
    return result


# ── OS — Overstrong (cand ⊨ gold ∧ gold ⊭ cand) ─────────────────────────────

def OS_add_nucleus_conjunct(expr: "Expression") -> "Expression":
    """Add ONE new atomic conjunct to a consequent / nucleus / propositional
    conjunction.

    Example: ``∀x.(D(x) → M(x))`` → ``∀x.(D(x) → (M(x) ∧ Aux1(x)))``

    Strengthens because cand has more conjunctive content than gold.

    Tier: overstrong. Expected entailment: cand ⊨ gold. Witness axioms: none.
    """
    aux = _synthesize_aux_predicate(expr)

    # Case A: ∀x.(R(x) → C(x)) — add to consequent, use bound variable
    if isinstance(expr, AllExpression) and isinstance(expr.term, ImpExpression):
        bv = expr.variable
        body = expr.term
        new_atom = read_expr(f"{aux}({bv})")
        new_consequent = AndExpression(body.second, new_atom)
        return AllExpression(bv, ImpExpression(body.first, new_consequent))

    # Case B: ∀x.body or ∃x.body where body is not an implication
    if isinstance(expr, (AllExpression, ExistsExpression)):
        bv = expr.variable
        new_atom = read_expr(f"{aux}({bv})")
        new_body = AndExpression(expr.term, new_atom)
        return type(expr)(bv, new_body)

    # Case C/D: top-level And or atomic (propositional) — append to conjunction
    if isinstance(expr, (AndExpression, ApplicationExpression)):
        consts = sorted(_find_constants(expr))
        if not consts:
            raise NotApplicable("no constant to ground the new atom")
        new_atom = read_expr(f"{aux}({consts[0]})")
        return AndExpression(expr, new_atom)

    raise NotApplicable("no target conjunction position")


def OS_add_nucleus_conjunct_applies_to(expr: "Expression") -> bool:
    if isinstance(expr, (AllExpression, ExistsExpression)):
        return True
    if isinstance(expr, (AndExpression, ApplicationExpression)):
        return len(_find_constants(expr)) > 0
    return False


def OS_strengthen_quantifier(expr: "Expression") -> "Expression":
    """Replace an inner ∃ inside a ∀'s scope with a ∀ (strengthens).

    Example: ``∀x.(H(x) → ∃y.(C(y) ∧ Hold(x, y)))`` →
             ``∀x.(H(x) → ∀y.(C(y) ∧ Hold(x, y)))``

    Tier: overstrong. Expected entailment: cand ⊨ gold. Witness axioms: none.
    """
    if not isinstance(expr, AllExpression):
        raise NotApplicable("not a top-level universal")
    body = expr.term
    # Case A: ∀x.(P(x) → ∃y.Q(...)) — flip the consequent existential
    if isinstance(body, ImpExpression) and isinstance(body.second, ExistsExpression):
        inner = body.second
        new_consequent = AllExpression(inner.variable, inner.term)
        return AllExpression(expr.variable, ImpExpression(body.first, new_consequent))
    # Case B: ∀x.∃y.Q(...) — flip the directly-nested existential
    if isinstance(body, ExistsExpression):
        new_body = AllExpression(body.variable, body.term)
        return AllExpression(expr.variable, new_body)
    raise NotApplicable("no inner existential in universal scope")


def OS_strengthen_quantifier_applies_to(expr: "Expression") -> bool:
    if not isinstance(expr, AllExpression):
        return False
    body = expr.term
    if isinstance(body, ImpExpression) and isinstance(body.second, ExistsExpression):
        return True
    if isinstance(body, ExistsExpression):
        return True
    return False


def OS_narrow_consequent(expr: "Expression") -> "Expression":
    """Add an extra atomic constraint to an existential conjunction body.

    Example: ``∃x.(D(x) ∧ M(x))`` → ``∃x.(D(x) ∧ M(x) ∧ Aux1(x))``

    Applies only when the existential's body is already a conjunction
    (i.e., we are *narrowing* an existing constraint set, not introducing
    conjunction from scratch — that case is handled by OS_add_nucleus_conjunct).

    Tier: overstrong. Expected entailment: cand ⊨ gold. Witness axioms: none.
    """
    if not isinstance(expr, ExistsExpression):
        raise NotApplicable("not an existential")
    body = expr.term
    if not isinstance(body, AndExpression):
        raise NotApplicable("existential body is not a conjunction")
    aux = _synthesize_aux_predicate(expr)
    bv = expr.variable
    new_atom = read_expr(f"{aux}({bv})")
    return ExistsExpression(bv, AndExpression(body, new_atom))


def OS_narrow_consequent_applies_to(expr: "Expression") -> bool:
    return (
        isinstance(expr, ExistsExpression)
        and isinstance(expr.term, AndExpression)
    )


def OS_strengthen_predicate(expr: "Expression") -> "Expression":
    """Replace a consequent predicate with a 1-hop-stronger subtype from the
    curated FOLIO hierarchy. Supertype → subtype rewrite in a consequent
    strengthens the formula (requiring the subtype is more demanding).

    Example: ``∀x.(D(x) → Animal(x))`` → ``∀x.(D(x) → Mammal(x))``

    Tier: overstrong. Expected entailment: cand ⊨ gold. Witness axioms: none.
    """
    from siv.predicate_hierarchy import strict_subtype_of

    target = _find_consequent_predicate_with_subtype(expr)
    if target is None:
        raise NotApplicable(
            "no consequent predicate has a strict subtype in the hierarchy"
        )
    pred_name = target
    subtype = strict_subtype_of(pred_name)
    return _replace_pred_name(expr, pred_name, subtype)


def OS_strengthen_predicate_applies_to(expr: "Expression") -> bool:
    return _find_consequent_predicate_with_subtype(expr) is not None


def _find_consequent_predicate_with_subtype(expr) -> Optional[str]:
    """Return the first consequent / nucleus predicate that has a strict
    subtype in the curated hierarchy, alphabetically. ``None`` if none."""
    from siv.predicate_hierarchy import strict_subtype_of

    consequent_preds = sorted(_collect_consequent_predicates(expr))
    for p in consequent_preds:
        if strict_subtype_of(p) is not None:
            return p
    return None


def _collect_consequent_predicates(expr) -> Set[str]:
    """Collect predicate names that appear in 'consequent / nucleus' positions
    (consequent of ∀/∃-Imp, body of bare ∀/∃, top-level And/atomic).

    For severity_v1 the operative position is the part of gold being
    weakened/strengthened; that's the consequent of an implication or the
    body of a quantifier or the top-level conjunction itself.
    """
    if isinstance(expr, AllExpression) and isinstance(expr.term, ImpExpression):
        return {n for n, _ in _find_predicates(expr.term.second)}
    if isinstance(expr, (AllExpression, ExistsExpression)):
        return {n for n, _ in _find_predicates(expr.term)}
    return {n for n, _ in _find_predicates(expr)}


def OS_drop_conjunctive_restrictor(expr: "Expression") -> "Expression":
    """Drop the last conjunct from a multi-conjunct restrictor of a universal
    implication. Broadens the universal's range → strengthens the claim.

    Example: ``∀x.((D(x) ∧ Dom(x)) → M(x))`` → ``∀x.(D(x) → M(x))``

    Tier: overstrong. Expected entailment: cand ⊨ gold. Witness axioms: none.
    """
    if not isinstance(expr, AllExpression) or not isinstance(expr.term, ImpExpression):
        raise NotApplicable("not a universal implication")
    body = expr.term
    conjuncts = _flatten_and(body.first)
    if len(conjuncts) < 2:
        raise NotApplicable("restrictor has fewer than 2 conjuncts")
    new_ante = _build_and_from_list(conjuncts[:-1])
    return AllExpression(expr.variable, ImpExpression(new_ante, body.second))


def OS_drop_conjunctive_restrictor_applies_to(expr: "Expression") -> bool:
    if not isinstance(expr, AllExpression) or not isinstance(expr.term, ImpExpression):
        return False
    return len(_flatten_and(expr.term.first)) >= 2


# ── P — Partial (gold ⊨ cand ∧ cand ⊭ gold) ─────────────────────────────────

def P_drop_conjunct(expr: "Expression") -> "Expression":
    """Drop the last conjunct from a multi-conjunct nucleus / consequent /
    top-level propositional conjunction.

    Example: ``D(c) ∧ M(c) ∧ H(c)`` → ``D(c) ∧ M(c)``

    Tier: partial. Expected entailment: gold ⊨ cand. Witness axioms: none.
    """
    # Case A: ∀x.(R(x) → C(x)) where C is a multi-atom conjunction
    if isinstance(expr, AllExpression) and isinstance(expr.term, ImpExpression):
        body = expr.term
        conjuncts = _flatten_and(body.second)
        if len(conjuncts) >= 2:
            new_consequent = _build_and_from_list(conjuncts[:-1])
            return AllExpression(expr.variable, ImpExpression(body.first, new_consequent))
        raise NotApplicable("consequent has fewer than 2 conjuncts")
    # Case B: ∃x.body / ∀x.body where body is a multi-atom conjunction
    if isinstance(expr, (AllExpression, ExistsExpression)) and isinstance(expr.term, AndExpression):
        conjuncts = _flatten_and(expr.term)
        if len(conjuncts) >= 2:
            new_body = _build_and_from_list(conjuncts[:-1])
            return type(expr)(expr.variable, new_body)
        raise NotApplicable("body has fewer than 2 conjuncts")
    # Case C: top-level And (propositional)
    if isinstance(expr, AndExpression):
        conjuncts = _flatten_and(expr)
        if len(conjuncts) >= 2:
            return _build_and_from_list(conjuncts[:-1])
        raise NotApplicable("conjunction has fewer than 2 conjuncts")
    raise NotApplicable("no multi-conjunct target")


def P_drop_conjunct_applies_to(expr: "Expression") -> bool:
    if isinstance(expr, AllExpression) and isinstance(expr.term, ImpExpression):
        return len(_flatten_and(expr.term.second)) >= 2
    if isinstance(expr, (AllExpression, ExistsExpression)) and isinstance(expr.term, AndExpression):
        return len(_flatten_and(expr.term)) >= 2
    if isinstance(expr, AndExpression):
        return len(_flatten_and(expr)) >= 2
    return False


def P_drop_consequent_atom(expr: "Expression") -> "Expression":
    """Drop the last atom from a multi-atom implication consequent.

    Example: ``∀x.(R(x) → (P(x) ∧ Q(x)))`` → ``∀x.(R(x) → P(x))``

    Tier: partial. Expected entailment: gold ⊨ cand. Witness axioms: none.
    """
    if not isinstance(expr, AllExpression) or not isinstance(expr.term, ImpExpression):
        raise NotApplicable("not a universal implication")
    body = expr.term
    conjuncts = _flatten_and(body.second)
    if len(conjuncts) < 2:
        raise NotApplicable("consequent has fewer than 2 atoms")
    new_consequent = _build_and_from_list(conjuncts[:-1])
    return AllExpression(expr.variable, ImpExpression(body.first, new_consequent))


def P_drop_consequent_atom_applies_to(expr: "Expression") -> bool:
    if not isinstance(expr, AllExpression) or not isinstance(expr.term, ImpExpression):
        return False
    return len(_flatten_and(expr.term.second)) >= 2


def P_weaken_predicate(expr: "Expression") -> "Expression":
    """Replace a consequent predicate with its 1-hop supertype from the
    curated FOLIO hierarchy.

    Example: ``∀x.(D(x) → Mammal(x))`` → ``∀x.(D(x) → Animal(x))``

    Tier: partial. Expected entailment: gold ⊨ cand. Witness axioms: none.
    """
    from siv.predicate_hierarchy import parent_of

    target = _find_consequent_predicate_with_parent(expr)
    if target is None:
        raise NotApplicable(
            "no consequent predicate has a parent in the hierarchy"
        )
    new_pred = parent_of(target)
    return _replace_pred_name(expr, target, new_pred)


def P_weaken_predicate_applies_to(expr: "Expression") -> bool:
    return _find_consequent_predicate_with_parent(expr) is not None


def _find_consequent_predicate_with_parent(expr) -> Optional[str]:
    """First consequent predicate that has a 1-hop parent in the hierarchy."""
    from siv.predicate_hierarchy import parent_of
    consequent_preds = sorted(_collect_consequent_predicates(expr))
    for p in consequent_preds:
        if parent_of(p) is not None:
            return p
    return None


def P_drop_disjunctive_restrictor(expr: "Expression") -> "Expression":
    """Drop the last disjunct from a disjunctive restrictor of a universal
    implication. Narrows the universal's range → weakens.

    Example: ``∀x.((D(x) ∨ C(x)) → M(x))`` → ``∀x.(D(x) → M(x))``

    Tier: partial. Expected entailment: gold ⊨ cand. Witness axioms: none.
    """
    if not isinstance(expr, AllExpression) or not isinstance(expr.term, ImpExpression):
        raise NotApplicable("not a universal implication")
    body = expr.term
    if not isinstance(body.first, OrExpression):
        raise NotApplicable("restrictor is not disjunctive")
    disjuncts = _flatten_or(body.first)
    if len(disjuncts) < 2:
        raise NotApplicable("restrictor has fewer than 2 disjuncts")
    new_ante = _build_or_from_list(disjuncts[:-1])
    # If only one disjunct remains, the restrictor becomes an atomic atom
    if len(disjuncts) - 1 == 1:
        new_ante = disjuncts[0]
    return AllExpression(expr.variable, ImpExpression(new_ante, body.second))


def P_drop_disjunctive_restrictor_applies_to(expr: "Expression") -> bool:
    if not isinstance(expr, AllExpression) or not isinstance(expr.term, ImpExpression):
        return False
    if not isinstance(expr.term.first, OrExpression):
        return False
    return len(_flatten_or(expr.term.first)) >= 2


# ── OW — Overweak (gold ⊨ cand ∧ cand ⊭ gold; drastic by construction) ─────

def OW_drop_consequent_severely(expr: "Expression") -> "Expression":
    """Keep ONLY the first atom of a multi-atom consequent / conjunction;
    drop the rest. Drastic by entailment-content loss.

    Example: ``∀x.(R(x) → (P(x) ∧ Q(x) ∧ S(x)))`` → ``∀x.(R(x) → P(x))``

    Tier: overweak. Expected entailment: gold ⊨ cand. Witness axioms: none.
    """
    if isinstance(expr, AllExpression) and isinstance(expr.term, ImpExpression):
        body = expr.term
        conjuncts = _flatten_and(body.second)
        if len(conjuncts) >= 2:
            return AllExpression(expr.variable, ImpExpression(body.first, conjuncts[0]))
        raise NotApplicable("consequent has fewer than 2 atoms")
    if isinstance(expr, (AllExpression, ExistsExpression)) and isinstance(expr.term, AndExpression):
        conjuncts = _flatten_and(expr.term)
        if len(conjuncts) >= 2:
            return type(expr)(expr.variable, conjuncts[0])
        raise NotApplicable("body has fewer than 2 atoms")
    if isinstance(expr, AndExpression):
        conjuncts = _flatten_and(expr)
        if len(conjuncts) >= 2:
            return conjuncts[0]
        raise NotApplicable("conjunction has fewer than 2 atoms")
    raise NotApplicable("no multi-atom conjunction target")


def OW_drop_consequent_severely_applies_to(expr: "Expression") -> bool:
    return P_drop_conjunct_applies_to(expr)


def OW_weaken_predicate_severely(expr: "Expression") -> "Expression":
    """Replace a consequent predicate with its ≥2-hop ancestor in the curated
    hierarchy. Drastic by semantic distance (skips the 1-hop weakening that
    P_weaken_predicate would produce).

    Example: ``∀x.(D(x) → Dog(x))`` → ``∀x.(D(x) → Animal(x))``
             (Dog ⊏ Mammal ⊏ Animal — 2 hops)

    Tier: overweak. Expected entailment: gold ⊨ cand. Witness axioms: none.
    """
    from siv.predicate_hierarchy import ancestor_chain

    target = _find_consequent_predicate_with_2hop_ancestor(expr)
    if target is None:
        raise NotApplicable(
            "no consequent predicate has a ≥2-hop ancestor in the hierarchy"
        )
    chain = ancestor_chain(target)
    new_pred = chain[-1]  # the root / highest ancestor available
    return _replace_pred_name(expr, target, new_pred)


def OW_weaken_predicate_severely_applies_to(expr: "Expression") -> bool:
    return _find_consequent_predicate_with_2hop_ancestor(expr) is not None


def _find_consequent_predicate_with_2hop_ancestor(expr) -> Optional[str]:
    """First consequent predicate with at least a 2-hop ancestor chain."""
    from siv.predicate_hierarchy import ancestor_chain
    consequent_preds = sorted(_collect_consequent_predicates(expr))
    for p in consequent_preds:
        chain = ancestor_chain(p)
        if len(chain) >= 3:  # predicate + ≥2 ancestors
            return p
    return None


def OW_de_quantify_to_c0(expr: "Expression") -> "Expression":
    """Replace the outermost quantifier with the fresh constant ``c_0``.

    The choice of ``c_0`` is fixed (does not depend on existing constants
    in the formula's signature) — predictability across the catalog matters
    more than domain overlap.

    Example: ``∀x.(D(x) → M(x))`` → ``D(c_0) → M(c_0)``

    Tier: overweak. Expected entailment: gold ⊨ cand. Witness axioms: none.
    """
    if not isinstance(expr, (AllExpression, ExistsExpression)):
        raise NotApplicable("no outer quantifier")
    bv_name = str(expr.variable)
    return _replace_variable_with_constant(expr.term, bv_name, "c_0")


def OW_de_quantify_to_c0_applies_to(expr: "Expression") -> bool:
    return isinstance(expr, (AllExpression, ExistsExpression))


def _replace_variable_with_constant(expr, var_name: str, const_name: str):
    """Recursively replace every occurrence of bound variable ``var_name``
    with a ConstantExpression named ``const_name``."""
    if isinstance(expr, IndividualVariableExpression) and str(expr.variable) == var_name:
        return read_expr(const_name)
    if isinstance(expr, ApplicationExpression):
        head, args = _uncurry(expr)
        new_args = [_replace_variable_with_constant(a, var_name, const_name) for a in args]
        return _curry(head, new_args)
    if isinstance(expr, AndExpression):
        return AndExpression(
            _replace_variable_with_constant(expr.first, var_name, const_name),
            _replace_variable_with_constant(expr.second, var_name, const_name),
        )
    if isinstance(expr, OrExpression):
        return OrExpression(
            _replace_variable_with_constant(expr.first, var_name, const_name),
            _replace_variable_with_constant(expr.second, var_name, const_name),
        )
    if isinstance(expr, ImpExpression):
        return ImpExpression(
            _replace_variable_with_constant(expr.first, var_name, const_name),
            _replace_variable_with_constant(expr.second, var_name, const_name),
        )
    if isinstance(expr, IffExpression):
        return IffExpression(
            _replace_variable_with_constant(expr.first, var_name, const_name),
            _replace_variable_with_constant(expr.second, var_name, const_name),
        )
    if isinstance(expr, NegatedExpression):
        return NegatedExpression(
            _replace_variable_with_constant(expr.term, var_name, const_name)
        )
    if isinstance(expr, AllExpression):
        if str(expr.variable) == var_name:
            # Shadowed; don't recurse
            return expr
        return AllExpression(
            expr.variable,
            _replace_variable_with_constant(expr.term, var_name, const_name),
        )
    if isinstance(expr, ExistsExpression):
        if str(expr.variable) == var_name:
            return expr
        return ExistsExpression(
            expr.variable,
            _replace_variable_with_constant(expr.term, var_name, const_name),
        )
    return expr


def OW_flip_outer_quantifier(expr: "Expression") -> "Expression":
    """Replace outermost ∀ with ∃, keeping the body intact.

    Example: ``∀x.(P(x) → Q(x))`` → ``∃x.(P(x) → Q(x))``

    Canonical Brunello LE-failure case.

    Tier: overweak. Expected entailment: gold ⊨ cand. Witness axioms: none.
    """
    if not isinstance(expr, AllExpression):
        raise NotApplicable("top-level operator is not ∀")
    return ExistsExpression(expr.variable, expr.term)


def OW_flip_outer_quantifier_applies_to(expr: "Expression") -> bool:
    return isinstance(expr, AllExpression)


def OW_weaken_to_existential(expr: "Expression") -> "Expression":
    """Convert a universal implication to an existential conjunction.

    Example: ``∀x.(D(x) → M(x))`` → ``∃x.(D(x) ∧ M(x))``

    Requires the witness axiom ``∃x.<antecedent>(x)`` for Vampire to confirm
    gold ⊨ cand (without it, gold could be vacuously true). Witness axiom
    is derived from the universal implication's antecedent by the generation
    pipeline; this AST operator just produces the transformed candidate.

    Tier: overweak. Expected entailment: gold ⊨ cand (under witness axiom).
    """
    if not isinstance(expr, AllExpression):
        raise NotApplicable("not a universal")
    body = expr.term
    if not isinstance(body, ImpExpression):
        raise NotApplicable("universal body is not an implication")
    new_body = AndExpression(body.first, body.second)
    return ExistsExpression(expr.variable, new_body)


def OW_weaken_to_existential_applies_to(expr: "Expression") -> bool:
    return isinstance(expr, AllExpression) and isinstance(expr.term, ImpExpression)


def OW_overrestrict_antecedent(expr: "Expression") -> "Expression":
    """Add ONE new atomic conjunct to the antecedent of a universal
    implication. Narrows the universal's range → weakens the claim.

    Example: ``∀x.(D(x) → M(x))`` → ``∀x.((D(x) ∧ Aux1(x)) → M(x))``

    Tier: overweak. Expected entailment: gold ⊨ cand. Witness axioms: none.
    """
    if not isinstance(expr, AllExpression) or not isinstance(expr.term, ImpExpression):
        raise NotApplicable("not a universal implication")
    body = expr.term
    aux = _synthesize_aux_predicate(expr)
    bv = expr.variable
    new_atom = read_expr(f"{aux}({bv})")
    new_ante = AndExpression(body.first, new_atom)
    return AllExpression(bv, ImpExpression(new_ante, body.second))


def OW_overrestrict_antecedent_applies_to(expr: "Expression") -> bool:
    return isinstance(expr, AllExpression) and isinstance(expr.term, ImpExpression)


# ── Catalog for downstream dispatch ─────────────────────────────────────────

SEVERITY_V1_OS_OPS = [
    OS_add_nucleus_conjunct,
    OS_strengthen_quantifier,
    OS_narrow_consequent,
    OS_strengthen_predicate,
    OS_drop_conjunctive_restrictor,
]
SEVERITY_V1_P_OPS = [
    P_drop_conjunct,
    P_drop_consequent_atom,
    P_weaken_predicate,
    P_drop_disjunctive_restrictor,
]
SEVERITY_V1_OW_OPS = [
    OW_drop_consequent_severely,
    OW_weaken_predicate_severely,
    OW_de_quantify_to_c0,
    OW_flip_outer_quantifier,
    OW_weaken_to_existential,
    OW_overrestrict_antecedent,
]
SEVERITY_V1_ALL_OPS = (
    SEVERITY_V1_OS_OPS + SEVERITY_V1_P_OPS + SEVERITY_V1_OW_OPS
)
SEVERITY_V1_TIER_MAP = {
    "overstrong": SEVERITY_V1_OS_OPS,
    "partial":    SEVERITY_V1_P_OPS,
    "overweak":   SEVERITY_V1_OW_OPS,
}
