"""Tests for siv/schema.py."""
import json

import pytest
from pydantic import ValidationError

from siv.schema import (
    AtomicFormula,
    Constant,
    Entity,
    Formula,
    InnerQuantification,
    PredicateDecl,
    SchemaViolation,
    SentenceExtraction,
    TestSuite as _TestSuite,
    TripartiteQuantification,
    UnitTest,
    validate_extraction,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _preds(*triples):
    return [PredicateDecl(name=n, arity=ar, arg_types=ts) for (n, ar, ts) in triples]


def _ext(formula, preds=None, constants=None, entities=None, nl="x"):
    return SentenceExtraction(
        nl=nl,
        predicates=preds or [],
        constants=constants or [],
        entities=entities or [],
        formula=formula,
    )


def _atom(pred, *args, negated=False):
    return AtomicFormula(pred=pred, args=list(args), negated=negated)


def _atomic_f(*args, **kwargs):
    return Formula(atomic=_atom(*args, **kwargs))


# ── UnitTest invariants (C0) ────────────────────────────────────────────────

def test_unit_test_contrastive_requires_mutation_kind_positive():
    t = UnitTest(fol="P(a)", kind="contrastive", mutation_kind="negate_atom")
    assert t.mutation_kind == "negate_atom"


def test_unit_test_contrastive_without_mutation_kind_fails():
    with pytest.raises((SchemaViolation, ValidationError)):
        UnitTest(fol="P(a)", kind="contrastive", mutation_kind=None)


def test_unit_test_positive_with_mutation_kind_fails():
    with pytest.raises((SchemaViolation, ValidationError)):
        UnitTest(fol="P(a)", kind="positive", mutation_kind="negate_atom")


def test_unit_test_positive_without_mutation_kind_passes():
    t = UnitTest(fol="P(a)", kind="positive")
    assert t.mutation_kind is None


# ── PredicateDecl arity invariant ───────────────────────────────────────────

def test_predicate_decl_arg_types_must_match_arity():
    with pytest.raises((SchemaViolation, ValidationError)):
        PredicateDecl(name="P", arity=2, arg_types=["e"])


def test_predicate_decl_valid():
    p = PredicateDecl(name="P", arity=2, arg_types=["e", "e"])
    assert p.arity == 2


# ── Formula exclusivity (C1) ─────────────────────────────────────────────────

def _valid_scope():
    preds = _preds(("P", 1, ["e"]))
    constants = [Constant(id="a", surface="a", type="entity")]
    return preds, constants


def test_validate_rejects_empty_formula():
    preds, constants = _valid_scope()
    with pytest.raises(SchemaViolation):
        validate_extraction(_ext(Formula(), preds=preds, constants=constants))


def test_validate_rejects_multiple_cases():
    preds, constants = _valid_scope()
    f = Formula(atomic=_atom("P", "a"), connective="and", operands=[
        _atomic_f("P", "a"), _atomic_f("P", "a"),
    ])
    with pytest.raises(SchemaViolation):
        validate_extraction(_ext(f, preds=preds, constants=constants))


# ── Connective arity ────────────────────────────────────────────────────────

def test_validate_rejects_connective_without_operands():
    preds, constants = _valid_scope()
    f = Formula(connective="and")
    with pytest.raises(SchemaViolation):
        validate_extraction(_ext(f, preds=preds, constants=constants))


def test_validate_rejects_operands_without_connective():
    preds, constants = _valid_scope()
    f = Formula(operands=[_atomic_f("P", "a"), _atomic_f("P", "a")])
    with pytest.raises(SchemaViolation):
        validate_extraction(_ext(f, preds=preds, constants=constants))


def test_validate_rejects_implies_with_three_operands():
    preds, constants = _valid_scope()
    f = Formula(
        connective="implies",
        operands=[_atomic_f("P", "a"), _atomic_f("P", "a"), _atomic_f("P", "a")],
    )
    with pytest.raises(SchemaViolation):
        validate_extraction(_ext(f, preds=preds, constants=constants))


def test_validate_rejects_iff_with_one_operand():
    preds, constants = _valid_scope()
    f = Formula(connective="iff", operands=[_atomic_f("P", "a")])
    with pytest.raises(SchemaViolation):
        validate_extraction(_ext(f, preds=preds, constants=constants))


def test_validate_rejects_and_with_single_operand():
    preds, constants = _valid_scope()
    f = Formula(connective="and", operands=[_atomic_f("P", "a")])
    with pytest.raises(SchemaViolation):
        validate_extraction(_ext(f, preds=preds, constants=constants))


def test_validate_accepts_nary_and():
    preds, constants = _valid_scope()
    f = Formula(connective="and", operands=[
        _atomic_f("P", "a"), _atomic_f("P", "a"), _atomic_f("P", "a"),
    ])
    validate_extraction(_ext(f, preds=preds, constants=constants))


def test_validate_accepts_binary_implies():
    preds, constants = _valid_scope()
    f = Formula(connective="implies", operands=[
        _atomic_f("P", "a"), _atomic_f("P", "a"),
    ])
    validate_extraction(_ext(f, preds=preds, constants=constants))


# ── Predicate/arg resolution (C2) ────────────────────────────────────────────

def test_validate_rejects_undeclared_predicate():
    f = _atomic_f("Undeclared", "a")
    ext = _ext(
        f,
        preds=_preds(("P", 1, ["e"])),
        constants=[Constant(id="a", surface="a", type="entity")],
    )
    with pytest.raises(SchemaViolation):
        validate_extraction(ext)


def test_validate_rejects_arity_mismatch():
    f = _atomic_f("P", "a", "b")  # P declared unary
    ext = _ext(
        f,
        preds=_preds(("P", 1, ["e"])),
        constants=[
            Constant(id="a", surface="a", type="entity"),
            Constant(id="b", surface="b", type="entity"),
        ],
    )
    with pytest.raises(SchemaViolation):
        validate_extraction(ext)


def test_validate_rejects_undeclared_argument():
    f = _atomic_f("P", "ghost")
    ext = _ext(
        f,
        preds=_preds(("P", 1, ["e"])),
        constants=[Constant(id="a", surface="a", type="entity")],
    )
    with pytest.raises(SchemaViolation):
        validate_extraction(ext)


def test_validate_accepts_bound_variable():
    f = Formula(quantification=TripartiteQuantification(
        quantifier="universal", variable="x", var_type="entity",
        restrictor=[_atom("P", "x")],
        nucleus=_atomic_f("Q", "x"),
    ))
    ext = _ext(f, preds=_preds(("P", 1, ["e"]), ("Q", 1, ["e"])))
    validate_extraction(ext)


# ── Degenerate tripartite quantification ────────────────────────────────────

def test_validate_rejects_empty_restrictor_with_nucleus_single_atom_over_bound_var():
    f = Formula(quantification=TripartiteQuantification(
        quantifier="universal", variable="x", var_type="entity",
        restrictor=[],
        nucleus=_atomic_f("P", "x"),
    ))
    ext = _ext(f, preds=_preds(("P", 1, ["e"])))
    with pytest.raises(SchemaViolation):
        validate_extraction(ext)


def test_validate_rejects_restrictor_missing_bound_variable():
    """Bound-variable-in-restrictor invariant (C2, §7). A non-empty restrictor
    whose atoms never mention the quantification's own bound variable is a
    modeling error: the atoms belong at another scope."""
    # Construct premise-2-shaped extraction: inner existential on z whose
    # restrictor only mentions outer-x.
    predicates = _preds(
        ("PersonInClub", 1, ["entity"]),
        ("Chaperone", 2, ["entity", "entity"]),
        ("HighSchoolDance", 1, ["entity"]),
        ("Student", 1, ["entity"]),
        ("Attend", 2, ["entity", "entity"]),
        ("School", 1, ["entity"]),
    )
    constants = [Constant(id="theSchool", surface="the school", type="entity")]
    inner_z = TripartiteQuantification(
        quantifier="existential", variable="z", var_type="entity",
        restrictor=[
            _atom("Student", "x"),
            _atom("Attend", "x", "theSchool"),
            _atom("School", "theSchool"),
        ],
        nucleus=_atomic_f("Attend", "x", "z"),
    )
    outer = TripartiteQuantification(
        quantifier="universal", variable="x", var_type="entity",
        restrictor=[
            _atom("PersonInClub", "x"),
            _atom("Chaperone", "x", "y"),
            _atom("HighSchoolDance", "y"),
        ],
        nucleus=Formula(negation=Formula(quantification=inner_z)),
        inner_quantifications=[
            InnerQuantification(quantifier="existential", variable="y", var_type="entity"),
        ],
    )
    ext = _ext(
        Formula(quantification=outer),
        preds=predicates, constants=constants,
        nl="People in this club who chaperone high school dances are not students who attend the school.",
    )
    with pytest.raises(SchemaViolation) as exc:
        validate_extraction(ext)
    # Message must name the offending quantification's bound variable.
    assert "'z'" in str(exc.value) or "z" in str(exc.value)


def test_validate_accepts_legitimate_dependent_inner_quantification():
    """'Every student x owns a book y that x likes.' Inner restrictor
    mentions BOTH the inner bound variable (via Book(y), Likes(x,y))
    and the outer variable x (via Likes(x,y)). That's a legitimate
    dependent quantification and must validate."""
    predicates = _preds(
        ("Student", 1, ["entity"]),
        ("Book", 1, ["entity"]),
        ("Likes", 2, ["entity", "entity"]),
        ("Owns", 2, ["entity", "entity"]),
    )
    inner_y = TripartiteQuantification(
        quantifier="existential", variable="y", var_type="entity",
        restrictor=[
            _atom("Book", "y"),
            _atom("Likes", "x", "y"),
        ],
        nucleus=_atomic_f("Owns", "x", "y"),
    )
    outer = TripartiteQuantification(
        quantifier="universal", variable="x", var_type="entity",
        restrictor=[_atom("Student", "x")],
        nucleus=Formula(quantification=inner_y),
    )
    ext = _ext(Formula(quantification=outer), preds=predicates)
    validate_extraction(ext)  # no raise


def test_validate_accepts_empty_restrictor_with_nonatomic_nucleus():
    # Empty restrictor with a compound nucleus is not degenerate.
    f = Formula(quantification=TripartiteQuantification(
        quantifier="existential", variable="x", var_type="entity",
        restrictor=[],
        nucleus=Formula(connective="and", operands=[
            _atomic_f("P", "x"), _atomic_f("Q", "x"),
        ]),
    ))
    ext = _ext(f, preds=_preds(("P", 1, ["e"]), ("Q", 1, ["e"])))
    validate_extraction(ext)


# ── Exercise TestSuite container ────────────────────────────────────────────

def test_test_suite_holds_extraction_and_positives():
    preds, constants = _valid_scope()
    ext = _ext(_atomic_f("P", "a"), preds=preds, constants=constants)
    suite = _TestSuite(
        extraction=ext,
        positives=[UnitTest(fol="P(a)", kind="positive")],
        contrastives=[
            UnitTest(fol="-P(a)", kind="contrastive", mutation_kind="negate_atom"),
        ],
    )
    assert len(suite.positives) == 1
    assert len(suite.contrastives) == 1
