"""
FOL Utilities: parsing, normalization, predicate extraction, and TPTP conversion.

Ported and extended from SIV_Evaluation_Framework (3).ipynb, Cells 9-10, 12.
All FOL strings use NLTK format:
  exists x.(P(x) & Q(x))
  all x.(P(x) -> Q(x))
  -P(x)   (negation)
"""
import re
import unicodedata
from typing import FrozenSet, Optional, Set, List

# ── NLTK logic imports ────────────────────────────────────────────────────────

NLTK_AVAILABLE = False
try:
    from nltk.sem.logic import (
        Expression, NegatedExpression,
        ApplicationExpression, BinaryExpression,
        Variable, AllExpression, ExistsExpression,
        AndExpression, OrExpression,
        ImpExpression, IffExpression, EqualityExpression,
        IndividualVariableExpression,
        is_indvar,
    )
    read_expr = Expression.fromstring
    NLTK_AVAILABLE = True
except ImportError:
    pass  # Callers that need NLTK will check NLTK_AVAILABLE


# ── XOR expansion ────────────────────────────────────────────────────────────

def _find_operand_boundary(s: str, pos: int, direction: str) -> int:
    """Find the boundary of a FOL operand adjacent to *pos*.

    *direction* is ``"left"`` (scan backwards) or ``"right"`` (scan forwards).
    Handles parenthesised sub-expressions and negated atoms.
    Returns the index of the first (left) or past-the-end (right) character
    of the operand.
    """
    if direction == "right":
        i = pos
        while i < len(s) and s[i] == ' ':
            i += 1
        if i >= len(s):
            return i
        # Handle leading negation(s)
        while i < len(s) and s[i] == '-':
            i += 1
            while i < len(s) and s[i] == ' ':
                i += 1
        if i >= len(s):
            return i
        if s[i] == '(':
            depth = 1
            i += 1
            while i < len(s) and depth > 0:
                if s[i] == '(':
                    depth += 1
                elif s[i] == ')':
                    depth -= 1
                i += 1
            return i
        else:
            # Atom: predicate(args) or bare identifier
            while i < len(s) and (s[i].isalnum() or s[i] in '_'):
                i += 1
            if i < len(s) and s[i] == '(':
                depth = 1
                i += 1
                while i < len(s) and depth > 0:
                    if s[i] == '(':
                        depth += 1
                    elif s[i] == ')':
                        depth -= 1
                    i += 1
            return i
    else:  # left
        i = pos - 1
        while i >= 0 and s[i] == ' ':
            i -= 1
        if i < 0:
            return 0
        if s[i] == ')':
            depth = 1
            i -= 1
            while i >= 0 and depth > 0:
                if s[i] == ')':
                    depth += 1
                elif s[i] == '(':
                    depth -= 1
                i -= 1
            # Walk back over predicate name before the '('
            while i >= 0 and (s[i].isalnum() or s[i] in '_'):
                i -= 1
            # Walk back over any leading negation(s)
            while i >= 0 and s[i] == '-':
                i -= 1
            while i >= 0 and s[i] == ' ':
                i -= 1
            return i + 1
        else:
            # Bare identifier (no parenthesized args)
            while i >= 0 and (s[i].isalnum() or s[i] in '_'):
                i -= 1
            # Walk back over leading negation(s)
            while i >= 0 and s[i] == '-':
                i -= 1
            while i >= 0 and s[i] == ' ':
                i -= 1
            return i + 1


def _expand_xor(fol: str) -> str:
    """Expand all ``__XOR__`` placeholders in *fol*.

    ``A __XOR__ B`` becomes ``((A & -(B)) | (-(A) & B))``.
    Processes from right to left so indices stay valid.
    """
    marker = "__XOR__"
    while marker in fol:
        idx = fol.rfind(marker)
        left_start = _find_operand_boundary(fol, idx, "left")
        right_end = _find_operand_boundary(fol, idx + len(marker), "right")

        a = fol[left_start:idx].strip()
        b = fol[idx + len(marker):right_end].strip()

        if not a or not b:
            # Can't determine operands; drop the marker to avoid infinite loop
            fol = fol[:idx] + fol[idx + len(marker):]
            continue

        expansion = f"(({a} & -({b})) | (-({a}) & {b}))"
        fol = fol[:left_start] + expansion + fol[right_end:]

    return fol


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize_fol_string(fol: str) -> str:
    """Convert Unicode symbols and common variants to NLTK ASCII format."""
    if not fol or not isinstance(fol, str):
        return ""

    # Step 1: Replace Unicode logic symbols BEFORE stripping non-ASCII.
    # Order matters: stripping non-ASCII first would silently delete these.
    for symbol, replacement in [
        # Quantifiers
        ("∀", "all "),  ("∃", "exists "),
        # Conjunction / disjunction
        ("∧", " & "),   ("∨", " | "),
        ("⋀", " & "),   ("⋁", " | "),   # n-ary variants (U+22C0, U+22C1)
        # Implication — FOLIO v2 uses ⇒ on many problems
        ("→", " -> "),  ("⇒", " -> "),  ("⟹", " -> "),
        # Biconditional
        ("↔", " <-> "), ("⟺", " <-> "), ("⇔", " <-> "),
        # Exclusive-or — expand A ⊕ B to ((A & -B) | (-A & B))
        # We first replace with a placeholder to avoid clashing with other
        # substitutions, then expand below.
        ("⊕", " __XOR__ "),
        # Negation
        ("¬", "-"),     ("~", "-"),
    ]:
        fol = fol.replace(symbol, replacement)

    # Expand XOR: A __XOR__ B → ((A & -(B)) | (-(A) & B))
    # We handle __XOR__ by locating each occurrence and extracting the
    # left and right operands using bracket-aware splitting.
    fol = _expand_xor(fol)

    # Step 2: Strip remaining non-ASCII (fancy quotes, curly arrows, etc.)
    fol = unicodedata.normalize("NFKD", fol).encode("ascii", "ignore").decode("ascii")

    # Step 3: ASCII keyword and whitespace normalisations
    for pattern, replacement in [
        (r"\bforall\b", "all"),
        (r"\bexist\b(?!s)", "exists"),  # "exist x" → "exists x" but keep "exists"
        (r"(?<=[A-Za-z0-9_])(all|exists)(?=\s)", r" \1"),  # ∀x∃y → "all x exists y"
        (r"\s+", " "),
    ]:
        fol = re.sub(pattern, replacement, fol)

    return fol.strip()


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_fol(fol_string: str) -> "Optional[Expression]":
    """
    Parse a FOL string into an NLTK Expression.
    Returns None if NLTK is unavailable or the string is syntactically invalid.
    """
    if not NLTK_AVAILABLE:
        return None
    try:
        normalized = normalize_fol_string(fol_string)
        if not normalized:
            return None
        return read_expr(normalized)
    except Exception:
        return None


def is_valid_fol(fol_string: str) -> bool:
    """Return True if *fol_string* parses as valid NLTK FOL."""
    return parse_fol(fol_string) is not None


def free_individual_variables(
    fol_string: str,
    declared_constants: FrozenSet[str] = frozenset(),
) -> Set[str]:
    """Return the set of free individual variable names in *fol_string*.

    Variables whose names appear in *declared_constants* are excluded —
    they are constants that happen to have variable-like names (e.g., 'j', 'c').

    Returns an empty set if NLTK is unavailable or the string fails to parse.
    """
    if not NLTK_AVAILABLE:
        return set()
    expr = parse_fol(fol_string)
    if expr is None:
        return set()
    return {
        str(v) for v in expr.free()
        if is_indvar(str(v)) and str(v) not in declared_constants
    }


# ── Predicate extraction ──────────────────────────────────────────────────────

def extract_predicates(fol_string: str) -> Set[str]:
    """
    Return the set of all predicate names that appear in *fol_string*.

    Primary path: walk the NLTK AST.
    Fallback (no NLTK / parse failure): regex over CamelCase identifiers
    followed by '('.
    """
    expr = parse_fol(fol_string)
    if expr is not None:
        return _extract_predicates_from_expr(expr)
    # Regex fallback
    matches = re.findall(r"([A-Z][A-Za-z0-9_]*)\s*\(", fol_string)
    keywords = {"All", "Exists", "Forall", "And", "Or", "Not", "Implies"}
    return {m for m in matches if m not in keywords}


def _extract_predicates_from_expr(expr) -> Set[str]:
    """Recursively walk an NLTK Expression and collect predicate names."""
    if not NLTK_AVAILABLE:
        return set()
    if isinstance(expr, ApplicationExpression):
        # Walk down the curried application chain to get the predicate head
        e = expr
        while isinstance(e, ApplicationExpression):
            e = e.function
        if hasattr(e, "variable"):
            return {e.variable.name}
        return set()
    elif isinstance(expr, BinaryExpression):
        return (
            _extract_predicates_from_expr(expr.first)
            | _extract_predicates_from_expr(expr.second)
        )
    elif isinstance(expr, NegatedExpression):
        return _extract_predicates_from_expr(expr.term)
    elif hasattr(expr, "term"):
        return _extract_predicates_from_expr(expr.term)
    return set()


# ── TPTP conversion ───────────────────────────────────────────────────────────

def convert_to_tptp(expr, _toplevel: bool = True) -> str:
    """
    Recursively convert an NLTK Expression to TPTP format for Vampire.

    TPTP conventions used here:
      - Variables are upper-case single letters (X, Y, …)
      - Predicates/constants are lower-case
      - Quantifiers: ?[X] : … (exists),  ![X] : … (all)

    Free variables are closed with universal quantifiers at the top level,
    following the standard convention that free variables in classical FOL
    are implicitly universally quantified.
    """
    if not NLTK_AVAILABLE:
        raise RuntimeError("NLTK is required for TPTP conversion.")

    # At the top level, close any free variables with universal quantifiers
    if _toplevel:
        free_vars = sorted(str(v) for v in expr.free())
        core = _convert_to_tptp_inner(expr)
        if free_vars:
            var_list = ", ".join(_tptp_var(v) for v in free_vars)
            return f"![{var_list}] : ({core})"
        return core

    return _convert_to_tptp_inner(expr)


def _convert_to_tptp_inner(expr) -> str:
    """Inner recursive TPTP conversion (does not close free variables)."""

    r = _convert_to_tptp_inner  # short alias for recursive calls

    if isinstance(expr, ExistsExpression):
        return f"?[{_tptp_var(str(expr.variable))}] : ({r(expr.term)})"
    elif isinstance(expr, AllExpression):
        return f"![{_tptp_var(str(expr.variable))}] : ({r(expr.term)})"
    elif isinstance(expr, NegatedExpression):
        return f"~({r(expr.term)})"
    elif isinstance(expr, AndExpression):
        return f"({r(expr.first)} & {r(expr.second)})"
    elif isinstance(expr, OrExpression):
        return f"({r(expr.first)} | {r(expr.second)})"
    elif isinstance(expr, ImpExpression):
        return f"({r(expr.first)} => {r(expr.second)})"
    elif isinstance(expr, IffExpression):
        return f"({r(expr.first)} <=> {r(expr.second)})"
    elif isinstance(expr, EqualityExpression):
        return f"({r(expr.first)} = {r(expr.second)})"
    elif isinstance(expr, ApplicationExpression):
        # Uncurry: f(a)(b)(c) → pred(a, b, c)
        func = expr.function
        args: List = [expr.argument]
        while isinstance(func, ApplicationExpression):
            args.insert(0, func.argument)
            func = func.function
        pred_name = _tptp_const(str(func).lower())
        args_str = ", ".join(r(a) for a in args)
        return f"{pred_name}({args_str})"
    elif isinstance(expr, IndividualVariableExpression):
        # NLTK treats lowercase u-z identifiers as bound-variable expressions;
        # TPTP requires variables to be uppercase identifiers.
        return _tptp_var(str(expr.variable))
    elif isinstance(expr, Variable):
        return _tptp_var(str(expr))
    else:
        return _tptp_const(str(expr).lower())


def _tptp_var(name: str) -> str:
    """Render an NLTK variable name as a valid TPTP variable identifier."""
    return name[0].upper() + name[1:] if name else name


def _tptp_const(name: str) -> str:
    """Render a constant or predicate name as a valid TPTP identifier.

    TPTP requires lowercase constants/functions/predicates to start with
    [a-z].  Identifiers that start with a digit (e.g. ``2008SummerOlympics``)
    are prefixed with ``c_`` to make them syntactically valid.
    """
    if not name:
        return name
    if name[0].isdigit():
        return f"c_{name}"
    return name


# ════════════════════════════════════════════════════════════════════════════
# severity_correlation_v1 — formula stratification
# ════════════════════════════════════════════════════════════════════════════

def formula_stratum(fol_string: str) -> Optional[int]:
    """Map an FOL string to one of 5 complexity strata for the
    severity-correlation experiment (paper Exp 1).

    Buckets (most specific check wins, in order below):

      1. propositional         — no quantifiers anywhere in the AST
      5. structurally rich     — multi-atom restrictor OR multi-atom
                                 consequent OR multiple top-level clauses
                                 (each containing a quantifier)
      4. nested quantifiers    — ≥1 quantifier inside another's scope
      3. restricted-simple     — single ∀.(R(x) → C(x)) with single-atom R
                                 and single-atom C
      2. single-quantifier     — default for any other single-quantifier
                                 form (∃x.P(x), ∀x.P(x), atomic-body cases)

    Check order matters: S5 dominates S4 (a multi-clause / multi-atom
    formula with nested quantifiers is bucketed S5 — structural richness
    dominates nesting). S3 is the most specific single-Q shape, so it's
    only assigned to formulas matching its exact pattern.

    Worked examples (the unit tests verify all of these):

      ``D(rex) & M(rex)``                      → 1
      ``∀x.P(x)``                              → 2
      ``∃x.D(x)``                              → 2
      ``∀x.(D(x) → M(x))``                     → 3
      ``∀x.((D(x) & Dom(x)) → M(x))``          → 5
      ``∀x.(D(x) → (M(x) & H(x)))``            → 5
      ``∀x.(D(x) → ∃y.O(x, y))``               → 4
      ``(∀x.P(x)) & (∀y.Q(y))``                → 5
      ``∀x.(D(x) → ∃y.(O(x, y) & P(x, y)))``   → 5  (S5 dominates S4)

    Returns ``None`` if the FOL string fails to parse.

    Known stratification imprecision:
    S2 catches single-quantifier formulas including some ∀-implication
    structures whose consequent is a NegatedExpression or OrExpression
    (e.g., ``∀x.(P(x) → ¬Q(x))`` or ``∀x.(P(x) → (Q(x) ∨ R(x)))``).
    These satisfy neither the strict S3 rule (which requires both
    restrictor and consequent to be ApplicationExpression) nor S5's
    "multi-atom conjunction in quantifier scope" rule, so they fall
    through to S2 as the default single-quantifier bucket. The OW
    operators that target ∀-implications legitimately apply to these
    formulas; the cell-targets in the YAML reflect operator applicability
    per cell rather than re-bucketing the formulas.
    See ``configs/severity_correlation_v1.yaml`` for the rationale and
    the affected cells.
    """
    expr = parse_fol(fol_string)
    if expr is None:
        return None

    if _count_quantifiers(expr) == 0:
        return 1
    if _is_structurally_rich(expr):
        return 5
    if _max_quantifier_nesting(expr) >= 2:
        return 4
    if _is_restricted_simple(expr):
        return 3
    return 2


def _count_quantifiers(expr) -> int:
    if not NLTK_AVAILABLE:
        return 0
    from nltk.sem.logic import (
        AllExpression, ExistsExpression, NegatedExpression,
        BinaryExpression, ApplicationExpression,
    )
    if isinstance(expr, (AllExpression, ExistsExpression)):
        return 1 + _count_quantifiers(expr.term)
    if isinstance(expr, BinaryExpression):
        return _count_quantifiers(expr.first) + _count_quantifiers(expr.second)
    if isinstance(expr, NegatedExpression):
        return _count_quantifiers(expr.term)
    return 0


def _max_quantifier_nesting(expr, current: int = 0) -> int:
    if not NLTK_AVAILABLE:
        return current
    from nltk.sem.logic import (
        AllExpression, ExistsExpression, NegatedExpression, BinaryExpression,
    )
    if isinstance(expr, (AllExpression, ExistsExpression)):
        return _max_quantifier_nesting(expr.term, current + 1)
    if isinstance(expr, BinaryExpression):
        return max(
            _max_quantifier_nesting(expr.first, current),
            _max_quantifier_nesting(expr.second, current),
        )
    if isinstance(expr, NegatedExpression):
        return _max_quantifier_nesting(expr.term, current)
    return current


def _flatten_and_local(expr) -> list:
    """Flatten nested AndExpressions into list of conjuncts."""
    from nltk.sem.logic import AndExpression
    if isinstance(expr, AndExpression):
        return _flatten_and_local(expr.first) + _flatten_and_local(expr.second)
    return [expr]


def _is_restricted_simple(expr) -> bool:
    """∀x.(P(x) → Q(x)) with single-atom restrictor and single-atom consequent."""
    if not NLTK_AVAILABLE:
        return False
    from nltk.sem.logic import (
        AllExpression, ImpExpression, ApplicationExpression,
    )
    if not isinstance(expr, AllExpression):
        return False
    body = expr.term
    if not isinstance(body, ImpExpression):
        return False
    if not isinstance(body.first, ApplicationExpression):
        return False
    if not isinstance(body.second, ApplicationExpression):
        return False
    if _count_quantifiers(body) > 0:
        return False
    return True


def _is_structurally_rich(expr) -> bool:
    """Multi-atom conjunction inside any quantifier scope OR multiple
    top-level quantified clauses."""
    if not NLTK_AVAILABLE:
        return False
    from nltk.sem.logic import AndExpression
    if isinstance(expr, AndExpression) and _has_multiple_quantified_clauses(expr):
        return True
    return _has_multi_atom_conjunction_under_quantifier(expr)


def _has_multiple_quantified_clauses(expr) -> bool:
    """Top-level And where ≥2 conjuncts each contain a quantifier."""
    conjuncts = _flatten_and_local(expr)
    if len(conjuncts) < 2:
        return False
    return sum(1 for c in conjuncts if _count_quantifiers(c) > 0) >= 2


def _has_multi_atom_conjunction_under_quantifier(expr, under: bool = False) -> bool:
    """Recursively detect whether any AndExpression with ≥2 atoms appears
    in the scope of a quantifier."""
    if not NLTK_AVAILABLE:
        return False
    from nltk.sem.logic import (
        AllExpression, ExistsExpression, NegatedExpression,
        BinaryExpression, AndExpression,
    )
    if isinstance(expr, (AllExpression, ExistsExpression)):
        return _has_multi_atom_conjunction_under_quantifier(expr.term, under=True)
    if under and isinstance(expr, AndExpression):
        if len(_flatten_and_local(expr)) >= 2:
            return True
    if isinstance(expr, BinaryExpression):
        return (_has_multi_atom_conjunction_under_quantifier(expr.first, under)
                or _has_multi_atom_conjunction_under_quantifier(expr.second, under))
    if isinstance(expr, NegatedExpression):
        return _has_multi_atom_conjunction_under_quantifier(expr.term, under)
    return False
