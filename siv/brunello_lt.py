"""
Brunello-LT equivalence metric: Z3-based logical equivalence checking.

For each (candidate, gold) pair:
  1. Parse both FOL strings to NLTK expressions
  2. Convert to Z3 formulas
  3. Check equivalence: is (candidate ↔ gold) valid?

Two variants:
  - raw: no vocabulary alignment
  - aligned: symbol alignment via siv.aligner before checking

Note: Brunello et al. used manually-constructed per-story ontologies.
We auto-extract ontologies from gold FOL signatures for scalability.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from siv.fol_utils import normalize_fol_string, parse_fol, NLTK_AVAILABLE

if NLTK_AVAILABLE:
    from nltk.sem.logic import (
        Expression, ApplicationExpression, NegatedExpression,
        AllExpression, ExistsExpression,
        AndExpression, OrExpression, ImpExpression, IffExpression,
        EqualityExpression, IndividualVariableExpression, Variable,
    )

try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


# ── NLTK → Z3 conversion ──────────────────────────────────────────────────────

class _Z3Context:
    """Tracks Z3 sorts, functions, and variables during conversion."""

    def __init__(self):
        self.sort = z3.DeclareSort("Entity")
        self._predicates: Dict[str, z3.FuncDeclRef] = {}
        self._constants: Dict[str, z3.ExprRef] = {}
        self._variables: Dict[str, z3.ExprRef] = {}

    def get_predicate(self, name: str, arity: int) -> z3.FuncDeclRef:
        key = f"{name}/{arity}"
        if key not in self._predicates:
            self._predicates[key] = z3.Function(
                name, *([self.sort] * arity), z3.BoolSort()
            )
        return self._predicates[key]

    def get_constant(self, name: str) -> z3.ExprRef:
        if name not in self._constants:
            self._constants[name] = z3.Const(name, self.sort)
        return self._constants[name]

    def get_variable(self, name: str) -> z3.ExprRef:
        if name not in self._variables:
            self._variables[name] = z3.Const(name, self.sort)
        return self._variables[name]


def _nltk_to_z3(expr, ctx: _Z3Context) -> z3.ExprRef:
    """Recursively convert an NLTK Expression to a Z3 formula."""
    if isinstance(expr, AllExpression):
        var_name = str(expr.variable)
        v = ctx.get_variable(var_name)
        body = _nltk_to_z3(expr.term, ctx)
        return z3.ForAll([v], body)

    elif isinstance(expr, ExistsExpression):
        var_name = str(expr.variable)
        v = ctx.get_variable(var_name)
        body = _nltk_to_z3(expr.term, ctx)
        return z3.Exists([v], body)

    elif isinstance(expr, NegatedExpression):
        return z3.Not(_nltk_to_z3(expr.term, ctx))

    elif isinstance(expr, AndExpression):
        return z3.And(_nltk_to_z3(expr.first, ctx), _nltk_to_z3(expr.second, ctx))

    elif isinstance(expr, OrExpression):
        return z3.Or(_nltk_to_z3(expr.first, ctx), _nltk_to_z3(expr.second, ctx))

    elif isinstance(expr, ImpExpression):
        return z3.Implies(_nltk_to_z3(expr.first, ctx), _nltk_to_z3(expr.second, ctx))

    elif isinstance(expr, IffExpression):
        lhs = _nltk_to_z3(expr.first, ctx)
        rhs = _nltk_to_z3(expr.second, ctx)
        return lhs == rhs

    elif isinstance(expr, EqualityExpression):
        return _nltk_to_z3(expr.first, ctx) == _nltk_to_z3(expr.second, ctx)

    elif isinstance(expr, ApplicationExpression):
        # Uncurry: f(a)(b)(c) → pred(a, b, c)
        func = expr
        args = []
        while isinstance(func, ApplicationExpression):
            args.insert(0, func.argument)
            func = func.function
        pred_name = str(func).replace("'", "")
        if hasattr(func, "variable"):
            pred_name = func.variable.name
        z3_args = [_nltk_to_z3(a, ctx) for a in args]
        pred = ctx.get_predicate(pred_name, len(z3_args))
        return pred(*z3_args)

    elif isinstance(expr, IndividualVariableExpression):
        return ctx.get_variable(str(expr.variable))

    elif hasattr(expr, "variable"):
        # Constant or free variable
        name = str(expr.variable) if hasattr(expr, "variable") else str(expr)
        return ctx.get_constant(name)

    else:
        # Fallback: treat as constant
        return ctx.get_constant(str(expr))


# ── Public API ─────────────────────────────────────────────────────────────────

def brunello_lt_equivalence(
    candidate_fol: str,
    gold_fol: str,
    timeout: int = 10,
) -> Optional[float]:
    """Check logical equivalence via Z3, no vocabulary alignment.

    Returns 1.0 (equivalent), 0.0 (not equivalent), or None (parse/conversion error).
    """
    if not Z3_AVAILABLE:
        return None
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
        ctx = _Z3Context()
        cand_z3 = _nltk_to_z3(cand_expr, ctx)
        gold_z3 = _nltk_to_z3(gold_expr, ctx)

        # Check validity of (candidate ↔ gold) by checking unsatisfiability
        # of ¬(candidate ↔ gold)
        solver = z3.Solver()
        solver.set("timeout", timeout * 1000)  # Z3 timeout in milliseconds
        solver.add(z3.Not(cand_z3 == gold_z3))
        result = solver.check()

        if result == z3.unsat:
            return 1.0  # Equivalent
        elif result == z3.sat:
            return 0.0  # Not equivalent
        else:
            return None  # Unknown (timeout)

    except Exception:
        return None


def brunello_lt_equivalence_aligned(
    candidate_fol: str,
    gold_fol: str,
    timeout: int = 10,
) -> Optional[float]:
    """Check logical equivalence via Z3 with symbol alignment.

    Aligns candidate vocabulary to gold vocabulary before checking,
    giving Brunello-LT its best possible shot.
    """
    from siv.aligner import align_symbols, extract_symbols_from_fol

    cand_norm = normalize_fol_string(candidate_fol)
    gold_norm = normalize_fol_string(gold_fol)
    if not cand_norm or not gold_norm:
        return None

    gold_symbols = extract_symbols_from_fol(gold_norm)
    cand_symbols = extract_symbols_from_fol(cand_norm)
    alignment = align_symbols(gold_symbols, cand_symbols)

    # Build substitution: candidate name → gold name
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

    return brunello_lt_equivalence(aligned_cand, gold_norm, timeout=timeout)


def brunello_lt_batch(
    candidates: List[str],
    golds: List[str],
    timeout: int = 10,
    aligned: bool = False,
) -> Dict[str, object]:
    """Batch Brunello-LT over parallel premise lists.

    Returns {"mean": float, "per_premise": [float|None, ...]}.
    """
    if len(candidates) != len(golds):
        raise ValueError("candidates and golds must have the same length")

    fn = brunello_lt_equivalence_aligned if aligned else brunello_lt_equivalence
    per_premise = [fn(c, g, timeout=timeout) for c, g in zip(candidates, golds)]

    scored = [v for v in per_premise if v is not None]
    mean = sum(scored) / len(scored) if scored else 0.0

    return {"mean": mean, "per_premise": per_premise}
