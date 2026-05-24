"""
Deterministic gold FOL parser.

Converts a FOLIO gold FOL string into a validated ``SentenceExtraction``
that satisfies ``validate_extraction()`` and is consumable by the compiler,
contrastive generator, and scorer without modification.

No LLM calls. Fully deterministic. Raises ``ParseError`` on any input
that cannot be faithfully represented in the schema.

Public API
----------
parse_gold_fol(fol_string, nl) -> SentenceExtraction
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from siv.fol_utils import normalize_fol_string, parse_fol, NLTK_AVAILABLE
from siv.schema import (
    AtomicFormula,
    Constant,
    Formula,
    InnerQuantification,
    PredicateDecl,
    SchemaViolation,
    SentenceExtraction,
    TripartiteQuantification,
    validate_extraction,
)

if NLTK_AVAILABLE:
    from nltk.sem.logic import (
        AllExpression,
        AndExpression,
        ApplicationExpression,
        BinaryExpression,
        ConstantExpression,
        EqualityExpression,
        ExistsExpression,
        Expression,
        IffExpression,
        ImpExpression,
        IndividualVariableExpression,
        NegatedExpression,
        OrExpression,
        is_indvar,
    )


class ParseError(Exception):
    """Raised when a gold FOL string cannot be converted to SentenceExtraction."""


# ════════════════════════════════════════════════════════════════════════════
# Public entry point
# ════════════════════════════════════════════════════════════════════════════


def parse_gold_fol(fol_string: str, nl: str = "") -> SentenceExtraction:
    """Parse a FOLIO gold FOL string into a validated SentenceExtraction.

    Parameters
    ----------
    fol_string : str
        Raw gold FOL from FOLIO dataset (Unicode or ASCII).
    nl : str
        The natural language sentence (passed through unchanged).

    Returns
    -------
    SentenceExtraction
        Validated extraction ready for compiler/contrastive_generator.

    Raises
    ------
    ParseError
        If the string cannot be parsed or converted.
    """
    if not NLTK_AVAILABLE:
        raise ParseError("NLTK is required for gold FOL parsing")

    # Phase A: NLTK parse
    normalized = normalize_fol_string(fol_string)
    if not normalized:
        raise ParseError("empty or invalid FOL string after normalization")

    expr = parse_fol(normalized)
    if expr is None:
        raise ParseError(f"NLTK parse failure: {normalized[:100]}")

    # Reject free individual variables (broken gold, Exp 3 overlap)
    free_indvars = {str(v) for v in expr.free() if is_indvar(str(v))}
    if free_indvars:
        raise ParseError(
            f"free individual variables: {sorted(free_indvars)} "
            f"— overlaps with known broken gold"
        )

    # Phase B: Symbol extraction
    pred_dict: Dict[str, int] = {}
    const_names: Set[str] = set()
    _extract_symbols(expr, pred_dict, const_names)

    # Phase C: Formula tree construction
    formula = _build_formula(expr, set())

    # Assemble SentenceExtraction
    predicates = [
        PredicateDecl(name=name, arity=arity, arg_types=["entity"] * arity)
        for name, arity in sorted(pred_dict.items())
    ]
    constants = [
        Constant(id=name, surface=name, type="entity")
        for name in sorted(const_names)
    ]

    extraction = SentenceExtraction(
        nl=nl,
        predicates=predicates,
        entities=[],
        constants=constants,
        formula=formula,
    )

    # Final validation
    try:
        validate_extraction(extraction)
    except SchemaViolation as e:
        raise ParseError(f"validation failure: {e}") from e

    return extraction


# ════════════════════════════════════════════════════════════════════════════
# Phase B: Symbol extraction
# ════════════════════════════════════════════════════════════════════════════


def _extract_symbols(
    expr: "Expression",
    pred_dict: Dict[str, int],
    const_names: Set[str],
) -> None:
    """Walk AST to collect predicates (name→arity) and constant names."""
    if isinstance(expr, ApplicationExpression):
        head, args = _uncurry(expr)
        pred_name = head.variable.name if hasattr(head, "variable") else str(head)
        arity = len(args)
        if pred_name in pred_dict:
            if pred_dict[pred_name] != arity:
                # Same predicate with different arities — use max
                pred_dict[pred_name] = max(pred_dict[pred_name], arity)
        else:
            pred_dict[pred_name] = arity
        for arg in args:
            if isinstance(arg, ConstantExpression):
                const_names.add(str(arg))
            elif isinstance(arg, ApplicationExpression):
                _extract_symbols(arg, pred_dict, const_names)

    elif isinstance(expr, EqualityExpression):
        pred_dict.setdefault("__eq__", 2)
        for side in (expr.first, expr.second):
            if isinstance(side, ConstantExpression):
                const_names.add(str(side))

    elif isinstance(expr, NegatedExpression):
        _extract_symbols(expr.term, pred_dict, const_names)

    elif isinstance(expr, (AllExpression, ExistsExpression)):
        _extract_symbols(expr.term, pred_dict, const_names)

    elif isinstance(expr, BinaryExpression):
        _extract_symbols(expr.first, pred_dict, const_names)
        _extract_symbols(expr.second, pred_dict, const_names)


# ════════════════════════════════════════════════════════════════════════════
# Phase C: Formula tree construction
# ════════════════════════════════════════════════════════════════════════════


def _build_formula(expr: "Expression", bound_vars: Set[str]) -> Formula:
    """Recursively convert an NLTK Expression into a Formula tree."""
    if isinstance(expr, AllExpression):
        return _handle_universal(expr, bound_vars)

    if isinstance(expr, ExistsExpression):
        return _handle_existential(expr, bound_vars)

    if isinstance(expr, NegatedExpression):
        inner = _build_formula(expr.term, bound_vars)
        return Formula(negation=inner)

    if isinstance(expr, AndExpression):
        return _handle_nary_connective("and", expr, bound_vars)

    if isinstance(expr, OrExpression):
        return _handle_nary_connective("or", expr, bound_vars)

    if isinstance(expr, ImpExpression):
        left = _build_formula(expr.first, bound_vars)
        right = _build_formula(expr.second, bound_vars)
        return Formula(connective="implies", operands=[left, right])

    if isinstance(expr, IffExpression):
        left = _build_formula(expr.first, bound_vars)
        right = _build_formula(expr.second, bound_vars)
        return Formula(connective="iff", operands=[left, right])

    if isinstance(expr, EqualityExpression):
        lhs = _arg_name(expr.first)
        rhs = _arg_name(expr.second)
        return Formula(atomic=AtomicFormula(pred="__eq__", args=[lhs, rhs]))

    if isinstance(expr, ApplicationExpression):
        head, args = _uncurry(expr)
        pred_name = head.variable.name if hasattr(head, "variable") else str(head)
        arg_names = [_arg_name(a) for a in args]
        return Formula(atomic=AtomicFormula(pred=pred_name, args=arg_names))

    raise ParseError(f"unsupported NLTK expression type: {type(expr).__name__}")


# ── Universal handling ─────────────────────────────────────────────────────


def _handle_universal(expr: "AllExpression", bound_vars: Set[str]) -> Formula:
    """Convert AllExpression to Formula with TripartiteQuantification."""
    var_name = str(expr.variable)
    body = expr.term
    new_bound = bound_vars | {var_name}

    # Case: body is implication → tripartite split
    if isinstance(body, ImpExpression):
        antecedent = body.first
        consequent = body.second

        restrictor_atoms, inner_quants, residual = _extract_restrictor(
            antecedent, var_name, new_bound
        )

        if restrictor_atoms and residual is None:
            # Clean tripartite: restrictor → atoms, consequent → nucleus
            nucleus_bound = new_bound | {iq.variable for iq in inner_quants}
            nucleus = _build_formula(consequent, nucleus_bound)
            return Formula(quantification=TripartiteQuantification(
                quantifier="universal",
                variable=var_name,
                var_type="entity",
                restrictor=restrictor_atoms,
                nucleus=nucleus,
                inner_quantifications=inner_quants,
            ))
        else:
            # Cannot cleanly extract restrictor — use empty restrictor + full body as nucleus
            nucleus = _build_formula(body, new_bound)
            if _is_degenerate(nucleus, var_name):
                raise ParseError(
                    f"degenerate universal pattern: all {var_name}.P({var_name})"
                )
            return Formula(quantification=TripartiteQuantification(
                quantifier="universal",
                variable=var_name,
                var_type="entity",
                restrictor=[],
                nucleus=nucleus,
                inner_quantifications=[],
            ))

    # Case: body is another quantifier (nested universal/existential)
    # or any other non-implication form
    nucleus = _build_formula(body, new_bound)
    if _is_degenerate(nucleus, var_name):
        raise ParseError(
            f"degenerate universal pattern: all {var_name}.P({var_name})"
        )
    return Formula(quantification=TripartiteQuantification(
        quantifier="universal",
        variable=var_name,
        var_type="entity",
        restrictor=[],
        nucleus=nucleus,
        inner_quantifications=[],
    ))


# ── Existential handling ───────────────────────────────────────────────────


def _handle_existential(expr: "ExistsExpression", bound_vars: Set[str]) -> Formula:
    """Convert ExistsExpression to Formula with TripartiteQuantification.

    Restrictor-selection rule: the restrictor is the *first* atomic conjunct
    of the body that is a unary, non-negated atom whose only argument is
    this existential's bound variable (the "type predicate" position). All
    other conjuncts — atomic or not, including unary atoms over outer
    bound vars and binary/ternary relations — go to the nucleus, joined
    with AND.

    This produces probes that pair the type predicate with each non-type
    constraint: ``∃y.(Wheel(y))``, ``∃y.(Wheel(y) ∧ HasWheel(x,y))``,
    ``∃y.(Wheel(y) ∧ Black(y))``, etc. — the §3 motivating decomposition.

    Falls back to empty restrictor when no type-predicate candidate exists
    (e.g. ``∃y.(Sees(x,y) ∧ Likes(x,y))``).
    """
    var_name = str(expr.variable)
    body = expr.term
    new_bound = bound_vars | {var_name}

    # Flatten body conjunction.
    conjuncts = _flatten_conjunction(body)

    # Classify every conjunct.
    type_predicates: List[AtomicFormula] = []
    other_atomic: List[AtomicFormula] = []
    nucleus_exprs: List["Expression"] = []
    for conj in conjuncts:
        atom = _expr_to_atomic(conj)
        if atom is None:
            nucleus_exprs.append(conj)
        elif (
            len(atom.args) == 1
            and atom.args[0] == var_name
            and not atom.negated
        ):
            type_predicates.append(atom)
        else:
            other_atomic.append(atom)

    # Pick at most one type predicate as restrictor; remaining type
    # predicates flow into the nucleus alongside other_atomic conjuncts.
    if type_predicates:
        restrictor_atoms = [type_predicates[0]]
        rest_atomic = type_predicates[1:] + other_atomic
    else:
        restrictor_atoms = []
        rest_atomic = other_atomic

    # Degenerate: ``∃x.P(x)`` — single type predicate, no other content.
    # Preserve the legacy behaviour of duplicating the atom in restrictor +
    # nucleus so downstream callers see a complete tree.
    if (
        restrictor_atoms
        and not rest_atomic
        and not nucleus_exprs
    ):
        atom = restrictor_atoms[0]
        return Formula(quantification=TripartiteQuantification(
            quantifier="existential",
            variable=var_name,
            var_type="entity",
            restrictor=[atom],
            nucleus=Formula(atomic=AtomicFormula(
                pred=atom.pred, args=list(atom.args), negated=atom.negated
            )),
            inner_quantifications=[],
        ))

    # Build nucleus from rest_atomic + nucleus_exprs.
    nucleus_pieces: List[Formula] = []
    nucleus_pieces.extend(Formula(atomic=a) for a in rest_atomic)
    nucleus_pieces.extend(_build_formula(e, new_bound) for e in nucleus_exprs)

    if not nucleus_pieces:
        # Degenerate fallback: only a non-type-predicate atom (e.g. binary).
        # Preserve legacy behaviour of putting it in the nucleus with empty
        # restrictor.
        if other_atomic:
            single = other_atomic[0]
            nucleus = Formula(atomic=single)
        elif restrictor_atoms:
            atom = restrictor_atoms[0]
            nucleus = Formula(atomic=atom)
            restrictor_atoms = [atom]  # keep restrictor too
        else:
            raise ParseError(
                f"empty existential body for variable {var_name}"
            )
    elif len(nucleus_pieces) == 1:
        nucleus = nucleus_pieces[0]
    else:
        nucleus = Formula(connective="and", operands=nucleus_pieces)

    if not restrictor_atoms and _is_degenerate(nucleus, var_name):
        raise ParseError(
            f"degenerate existential pattern: exists {var_name}.P({var_name})"
        )

    return Formula(quantification=TripartiteQuantification(
        quantifier="existential",
        variable=var_name,
        var_type="entity",
        restrictor=restrictor_atoms,
        nucleus=nucleus,
        inner_quantifications=[],
    ))


# ── Connective handling ────────────────────────────────────────────────────


def _handle_nary_connective(
    conn_type: str, expr: "Expression", bound_vars: Set[str]
) -> Formula:
    """Flatten and convert AND/OR to Formula with n-ary operands."""
    if conn_type == "and":
        leaves = _flatten_conjunction(expr)
    else:
        leaves = _flatten_disjunction(expr)

    operands = [_build_formula(leaf, bound_vars) for leaf in leaves]
    return Formula(connective=conn_type, operands=operands)


# ════════════════════════════════════════════════════════════════════════════
# Restrictor extraction
# ════════════════════════════════════════════════════════════════════════════


def _extract_restrictor(
    antecedent: "Expression",
    bound_var: str,
    bound_vars: Set[str],
) -> Tuple[List[AtomicFormula], List[InnerQuantification], Optional["Expression"]]:
    """Extract atomic restrictor from an antecedent expression.

    Returns (restrictor_atoms, inner_quantifications, residual).
    residual is None if the entire antecedent was consumed into restrictor.
    """
    conjuncts = _flatten_conjunction(antecedent)

    restrictor_atoms: List[AtomicFormula] = []
    inner_quants: List[InnerQuantification] = []
    residual_parts: List["Expression"] = []

    for conj in conjuncts:
        # Simple atom or negated atom
        atom = _expr_to_atomic(conj)
        if atom is not None:
            restrictor_atoms.append(atom)
            continue

        # Existential quantifier in antecedent → inner_quantification pattern
        if isinstance(conj, ExistsExpression):
            inner_var = str(conj.variable)
            inner_body_conjuncts = _flatten_conjunction(conj.term)
            all_atoms = True
            for ic in inner_body_conjuncts:
                a = _expr_to_atomic(ic)
                if a is not None:
                    restrictor_atoms.append(a)
                else:
                    all_atoms = False
                    residual_parts.append(ic)
            if all_atoms:
                inner_quants.append(InnerQuantification(
                    quantifier="existential",
                    variable=inner_var,
                    var_type="entity",
                ))
            else:
                # Cannot fully flatten — treat as residual
                residual_parts.append(conj)
                # Remove atoms we already added from this failed attempt
                # Actually, the atoms from the inner body are still valid
                # restrictor atoms if they mention the bound var.
                # Keep them and record the inner quant anyway.
                inner_quants.append(InnerQuantification(
                    quantifier="existential",
                    variable=inner_var,
                    var_type="entity",
                ))
            continue

        # Anything else is residual (disjunction, complex formula, etc.)
        residual_parts.append(conj)

    # Validate: restrictor must mention bound_var at least once
    if restrictor_atoms:
        mentions_bound = any(bound_var in a.args for a in restrictor_atoms)
        if not mentions_bound:
            # None of the atoms mention the bound var — cannot be a valid restrictor
            return [], [], antecedent

    residual = None if not residual_parts else antecedent
    return restrictor_atoms, inner_quants, residual


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════


def _flatten_conjunction(expr: "Expression") -> "List[Expression]":
    """Recursively flatten nested AndExpression into a flat list."""
    if isinstance(expr, AndExpression):
        return _flatten_conjunction(expr.first) + _flatten_conjunction(expr.second)
    return [expr]


def _flatten_disjunction(expr: "Expression") -> "List[Expression]":
    """Recursively flatten nested OrExpression into a flat list."""
    if isinstance(expr, OrExpression):
        return _flatten_disjunction(expr.first) + _flatten_disjunction(expr.second)
    return [expr]


def _uncurry(expr: "ApplicationExpression") -> "Tuple[Expression, List[Expression]]":
    """Walk curried application chain to get (predicate_head, [args])."""
    func = expr
    args: list = []
    while isinstance(func, ApplicationExpression):
        args.insert(0, func.argument)
        func = func.function
    return func, args


def _arg_name(expr: "Expression") -> str:
    """Extract the string name of a predicate argument."""
    if isinstance(expr, IndividualVariableExpression):
        return str(expr.variable)
    if isinstance(expr, ConstantExpression):
        return str(expr)
    # Fallback for other expression types
    return str(expr)


def _expr_to_atomic(expr: "Expression") -> Optional[AtomicFormula]:
    """Try converting an NLTK expression to an AtomicFormula.

    Returns None if the expression is not an atom (or negated atom).
    Handles NegatedExpression wrapping ApplicationExpression or EqualityExpression.
    """
    negated = False
    if isinstance(expr, NegatedExpression):
        negated = True
        expr = expr.term

    if isinstance(expr, ApplicationExpression):
        head, args = _uncurry(expr)
        pred_name = head.variable.name if hasattr(head, "variable") else str(head)
        arg_names = [_arg_name(a) for a in args]
        return AtomicFormula(pred=pred_name, args=arg_names, negated=negated)

    if isinstance(expr, EqualityExpression):
        lhs = _arg_name(expr.first)
        rhs = _arg_name(expr.second)
        return AtomicFormula(pred="__eq__", args=[lhs, rhs], negated=negated)

    return None


def _is_degenerate(nucleus: Formula, var_name: str) -> bool:
    """Check if empty-restrictor + this nucleus would trigger the forbidden pattern.

    The schema forbids: empty restrictor AND nucleus is a single atomic formula
    with exactly one argument that equals the bound variable.
    """
    return (
        nucleus.atomic is not None
        and len(nucleus.atomic.args) == 1
        and nucleus.atomic.args[0] == var_name
    )
