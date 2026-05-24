"""Tests for siv/compiler.py.

Exactly the nine Formula cases listed in the compiler spec. Each
case asserts that compile_canonical_fol emits the expected FOL (verified via
bidirectional Vampire entailment with a hand-crafted expected formula), and
that compile_sentence_test_suite produces an alpha-equivalent formula via its
structurally distinct Path B (also verified via Vampire).
"""
from __future__ import annotations

import pytest

from siv.compiler import compile_canonical_fol, compile_sentence_test_suite
from siv.fol_utils import free_individual_variables
from siv.schema import (
    AtomicFormula,
    Constant,
    Entity,
    Formula,
    InnerQuantification,
    PredicateDecl,
    SchemaViolation,
    SentenceExtraction,
    TripartiteQuantification,
)
from siv.vampire_interface import check_entailment, is_vampire_available


# ── Vampire helpers ─────────────────────────────────────────────────────────

vampire_required = pytest.mark.skipif(
    not is_vampire_available(),
    reason="Vampire not available; compiler soundness gate requires it.",
)


def _bidir(fol_a: str, fol_b: str) -> bool:
    fwd = check_entailment(fol_a, fol_b, timeout=10)
    bwd = check_entailment(fol_b, fol_a, timeout=10)
    return fwd is True and bwd is True


def _suite_conj(ext):
    """Return the conjunction of all positive FOLs in the TestSuite."""
    positives = compile_sentence_test_suite(ext).positives
    fols = [p.fol for p in positives]
    if len(fols) == 1:
        return fols[0]
    return "(" + " & ".join(f"({f})" for f in fols) + ")"


# ── Builder helpers ─────────────────────────────────────────────────────────

def _atom(p, *args, negated=False):
    return AtomicFormula(pred=p, args=list(args), negated=negated)


def _atomic_f(p, *args, negated=False):
    return Formula(atomic=_atom(p, *args, negated=negated))


def _preds(*triples):
    return [PredicateDecl(name=n, arity=a, arg_types=t) for (n, a, t) in triples]


def _const(id_, surface=None):
    return Constant(id=id_, surface=surface or id_, type="entity")


# ════════════════════════════════════════════════════════════════════════════
# Case 1 — Atomic
# ════════════════════════════════════════════════════════════════════════════

def _case1():
    return SentenceExtraction(
        nl="Miroslav Venhoda was a Czech choral conductor",
        predicates=_preds(("CzechChoralConductor", 1, ["entity"])),
        constants=[_const("miroslav", "Miroslav Venhoda")],
        formula=_atomic_f("CzechChoralConductor", "miroslav"),
    )


def test_case1_canonical_string():
    ext = _case1()
    assert compile_canonical_fol(ext) == "CzechChoralConductor(miroslav)"


@vampire_required
def test_case1_two_path_equivalence():
    ext = _case1()
    a = compile_canonical_fol(ext)
    b = _suite_conj(ext)
    assert _bidir(a, b)


# ════════════════════════════════════════════════════════════════════════════
# Case 2 — Quantification (employees-meetings)
# ════════════════════════════════════════════════════════════════════════════

def _case2():
    # "All employees who schedule meetings attend the company building"
    # Outer universal over x (employees); restrictor contains:
    #   Employee(x), Schedule(x, y), Meeting(y)
    # with an inner existential on y (meetings).
    # Nucleus: exists z. (CompanyBuilding(z) & Attend(x, z)).
    inner_z = TripartiteQuantification(
        quantifier="existential",
        variable="z",
        var_type="entity",
        restrictor=[_atom("CompanyBuilding", "z")],
        nucleus=_atomic_f("Attend", "x", "z"),
    )
    q = TripartiteQuantification(
        quantifier="universal",
        variable="x",
        var_type="entity",
        restrictor=[
            _atom("Employee", "x"),
            _atom("Schedule", "x", "y"),
            _atom("Meeting", "y"),
        ],
        nucleus=Formula(quantification=inner_z),
        inner_quantifications=[
            InnerQuantification(quantifier="existential", variable="y", var_type="entity"),
        ],
    )
    return SentenceExtraction(
        nl="All employees who schedule meetings attend the company building",
        predicates=_preds(
            ("Employee", 1, ["entity"]),
            ("Meeting", 1, ["entity"]),
            ("Schedule", 2, ["entity", "entity"]),
            ("CompanyBuilding", 1, ["entity"]),
            ("Attend", 2, ["entity", "entity"]),
        ),
        formula=Formula(quantification=q),
    )


CASE2_EXPECTED = (
    "all x.(exists y.(Employee(x) & Schedule(x, y) & Meeting(y)) "
    "-> exists z.(CompanyBuilding(z) & Attend(x, z)))"
)


@vampire_required
def test_case2_canonical_matches_spec():
    ext = _case2()
    a = compile_canonical_fol(ext)
    assert _bidir(a, CASE2_EXPECTED)


@vampire_required
def test_case2_two_path_equivalence():
    ext = _case2()
    a = compile_canonical_fol(ext)
    b = _suite_conj(ext)
    assert _bidir(a, b)


# ════════════════════════════════════════════════════════════════════════════
# Case 3 — Negation
# ════════════════════════════════════════════════════════════════════════════

def _case3():
    # Use the Formula.negation case over an atom so we exercise the negation
    # branch explicitly (rather than just AtomicFormula.negated=True).
    return SentenceExtraction(
        nl="Smith is not a Czech conductor",
        predicates=_preds(("CzechConductor", 1, ["entity"])),
        constants=[_const("smith", "Smith")],
        formula=Formula(negation=_atomic_f("CzechConductor", "smith")),
    )


def test_case3_canonical_string():
    ext = _case3()
    assert compile_canonical_fol(ext) == "-(CzechConductor(smith))"


@vampire_required
def test_case3_semantic_match():
    ext = _case3()
    a = compile_canonical_fol(ext)
    assert _bidir(a, "-CzechConductor(smith)")


@vampire_required
def test_case3_two_path_equivalence():
    ext = _case3()
    a = compile_canonical_fol(ext)
    b = _suite_conj(ext)
    assert _bidir(a, b)


# ════════════════════════════════════════════════════════════════════════════
# Case 4 — Connective-and
# ════════════════════════════════════════════════════════════════════════════

def _case4():
    return SentenceExtraction(
        nl="Alice is tall and Bob is short",
        predicates=_preds(
            ("Tall", 1, ["entity"]),
            ("Short", 1, ["entity"]),
        ),
        constants=[_const("alice"), _const("bob")],
        formula=Formula(connective="and", operands=[
            _atomic_f("Tall", "alice"),
            _atomic_f("Short", "bob"),
        ]),
    )


def test_case4_canonical_string():
    assert compile_canonical_fol(_case4()) == "(Tall(alice) & Short(bob))"


@vampire_required
def test_case4_two_path_equivalence():
    ext = _case4()
    a = compile_canonical_fol(ext)
    b = _suite_conj(ext)
    assert _bidir(a, b)


# ════════════════════════════════════════════════════════════════════════════
# Case 5 — Connective-or
# ════════════════════════════════════════════════════════════════════════════

def _case5():
    return SentenceExtraction(
        nl="The L-2021 monitor is either used in the library or has a type-c port",
        predicates=_preds(
            ("UsedIn", 2, ["entity", "entity"]),
            ("HasTypeC", 1, ["entity"]),
        ),
        constants=[_const("monitor", "L-2021 monitor"), _const("library")],
        formula=Formula(connective="or", operands=[
            _atomic_f("UsedIn", "monitor", "library"),
            _atomic_f("HasTypeC", "monitor"),
        ]),
    )


def test_case5_canonical_string():
    got = compile_canonical_fol(_case5())
    assert got == "(UsedIn(monitor, library) | HasTypeC(monitor))"


@vampire_required
def test_case5_two_path_equivalence():
    ext = _case5()
    a = compile_canonical_fol(ext)
    b = _suite_conj(ext)
    assert _bidir(a, b)


# ════════════════════════════════════════════════════════════════════════════
# Case 6 — Connective-implies
# ════════════════════════════════════════════════════════════════════════════
# NOTE: The compiler spec lists the expected FOL as `(Rains() -> Wet(ground))`
# but PredicateDecl.arity is Literal[1, 2], forbidding 0-ary predicates.
# We model "it rains" with a 1-ary predicate over a dummy constant
# "weather". Semantic equivalent at the sentence level.

def _case6():
    return SentenceExtraction(
        nl="If it rains, then the ground is wet",
        predicates=_preds(
            ("Rains", 1, ["entity"]),
            ("Wet", 1, ["entity"]),
        ),
        constants=[_const("weather"), _const("ground")],
        formula=Formula(connective="implies", operands=[
            _atomic_f("Rains", "weather"),
            _atomic_f("Wet", "ground"),
        ]),
    )


def test_case6_canonical_string():
    assert compile_canonical_fol(_case6()) == "(Rains(weather) -> Wet(ground))"


@vampire_required
def test_case6_two_path_equivalence():
    ext = _case6()
    a = compile_canonical_fol(ext)
    b = _suite_conj(ext)
    assert _bidir(a, b)


# ════════════════════════════════════════════════════════════════════════════
# Case 7 — Connective-iff
# ════════════════════════════════════════════════════════════════════════════

def _case7():
    return SentenceExtraction(
        nl="Archie can walk if and only if he has functional brainstems",
        predicates=_preds(
            ("CanWalk", 1, ["entity"]),
            ("HasFunctionalBrainstems", 1, ["entity"]),
        ),
        constants=[_const("archie")],
        formula=Formula(connective="iff", operands=[
            _atomic_f("CanWalk", "archie"),
            _atomic_f("HasFunctionalBrainstems", "archie"),
        ]),
    )


def test_case7_canonical_string():
    assert (
        compile_canonical_fol(_case7())
        == "(CanWalk(archie) <-> HasFunctionalBrainstems(archie))"
    )


@vampire_required
def test_case7_two_path_equivalence():
    ext = _case7()
    a = compile_canonical_fol(ext)
    b = _suite_conj(ext)
    assert _bidir(a, b)


# ════════════════════════════════════════════════════════════════════════════
# Case 8 — Nested case (quantifier with implies nucleus)
# ════════════════════════════════════════════════════════════════════════════

def _case8():
    # "If a legislator is found guilty [of theft], they will be suspended"
    #
    # Structure:
    #   all x. (Legislator(x) -> (exists y. (Theft(y) & FoundGuilty(x, y))
    #                             -> Suspended(x)))
    #
    # Outer universal over legislators; nucleus is a Formula(connective=implies)
    # whose antecedent is itself a quantification (existential over theft).
    inner_exists = TripartiteQuantification(
        quantifier="existential",
        variable="y",
        var_type="entity",
        restrictor=[_atom("Theft", "y")],
        nucleus=_atomic_f("FoundGuilty", "x", "y"),
    )
    nucleus = Formula(connective="implies", operands=[
        Formula(quantification=inner_exists),
        _atomic_f("Suspended", "x"),
    ])
    outer = TripartiteQuantification(
        quantifier="universal",
        variable="x",
        var_type="entity",
        restrictor=[_atom("Legislator", "x")],
        nucleus=nucleus,
    )
    return SentenceExtraction(
        nl="If a legislator is found guilty, they will be suspended",
        predicates=_preds(
            ("Legislator", 1, ["entity"]),
            ("Theft", 1, ["entity"]),
            ("FoundGuilty", 2, ["entity", "entity"]),
            ("Suspended", 1, ["entity"]),
        ),
        formula=Formula(quantification=outer),
    )


CASE8_EXPECTED = (
    "all x.(Legislator(x) -> "
    "(exists y.(Theft(y) & FoundGuilty(x, y)) -> Suspended(x)))"
)


@vampire_required
def test_case8_canonical_matches_spec():
    ext = _case8()
    a = compile_canonical_fol(ext)
    assert _bidir(a, CASE8_EXPECTED)


@vampire_required
def test_case8_two_path_equivalence():
    ext = _case8()
    a = compile_canonical_fol(ext)
    b = _suite_conj(ext)
    assert _bidir(a, b)


# ════════════════════════════════════════════════════════════════════════════
# Case 9 — Quantifier in connective
# ════════════════════════════════════════════════════════════════════════════

def _case9():
    # "If the forecast calls for rain, then all employees work from home"
    # Top-level connective=implies, atomic antecedent, quantification consequent.
    # We model "forecast calls for rain" as ForecastRain(weather) for the
    # same schema reason as case 6.
    consequent_q = TripartiteQuantification(
        quantifier="universal",
        variable="x",
        var_type="entity",
        restrictor=[_atom("Employee", "x")],
        nucleus=_atomic_f("WorkFromHome", "x"),
    )
    return SentenceExtraction(
        nl="If the forecast calls for rain, then all employees work from home",
        predicates=_preds(
            ("ForecastRain", 1, ["entity"]),
            ("Employee", 1, ["entity"]),
            ("WorkFromHome", 1, ["entity"]),
        ),
        constants=[_const("weather")],
        formula=Formula(connective="implies", operands=[
            _atomic_f("ForecastRain", "weather"),
            Formula(quantification=consequent_q),
        ]),
    )


CASE9_EXPECTED = (
    "(ForecastRain(weather) -> all x.(Employee(x) -> WorkFromHome(x)))"
)


def test_case9_canonical_string():
    assert compile_canonical_fol(_case9()) == CASE9_EXPECTED


@vampire_required
def test_case9_two_path_equivalence():
    ext = _case9()
    a = compile_canonical_fol(ext)
    b = _suite_conj(ext)
    assert _bidir(a, b)


# ════════════════════════════════════════════════════════════════════════════
# Pure-property tests: compiler does NOT consume nl (§6.4)
# ════════════════════════════════════════════════════════════════════════════

def test_compile_canonical_fol_independent_of_nl():
    ext = _case4()
    a = compile_canonical_fol(ext)
    ext2 = ext.model_copy(update={"nl": "totally different surface sentence"})
    assert compile_canonical_fol(ext2) == a


def test_compile_test_suite_independent_of_nl():
    ext = _case2()
    a = compile_sentence_test_suite(ext).positives[0].fol
    ext2 = ext.model_copy(update={"nl": "bogus"})
    assert compile_sentence_test_suite(ext2).positives[0].fol == a


# ════════════════════════════════════════════════════════════════════════════
# Sub-test count assertions
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("case_fn,expected", [
    (_case1, 1),  # atomic — canonical only.
    (_case2, 2),  # canonical + existential-restrictor sub-test (rule b).
    (_case3, 1),  # canonical only; negation of an atom.
    (_case4, 3),  # canonical + each AND-conjunct (rule a).
    (_case5, 1),  # canonical only; or operands not emittable.
    (_case6, 1),  # canonical only; implies operands blocked.
    (_case7, 1),  # canonical only; iff interiors blocked.
    (_case8, 1),  # canonical only; outer implies blocks descent.
    (_case9, 1),  # canonical only; implies operands blocked.
])
def test_subtest_count(case_fn, expected):
    suite = compile_sentence_test_suite(case_fn())
    assert len(suite.positives) == expected, (
        f"Expected {expected} positives; got {len(suite.positives)}: "
        f"{[p.fol for p in suite.positives]}"
    )


def test_case2_subtest_is_existential_restrictor_closure():
    """The case-2 sub-test is derived from rule (b): inner existential's
    restrictor atom CompanyBuilding(z) closed under the outer universal chain.
    """
    suite = compile_sentence_test_suite(_case2())
    assert len(suite.positives) == 2
    sub = suite.positives[1].fol
    assert "CompanyBuilding" in sub
    assert "Attend" not in sub  # inner nucleus dropped by conjunct-elim
    assert sub.startswith("all ")
    assert "exists " in sub  # outer universal's IQ + inner existential


def test_case4_subtests_are_individual_conjuncts():
    suite = compile_sentence_test_suite(_case4())
    fols = {p.fol for p in suite.positives}
    assert "(Tall(alice) & Short(bob))" in fols
    assert "Tall(alice)" in fols
    assert "Short(bob)" in fols


# ════════════════════════════════════════════════════════════════════════════
# Pre-work A — free-variable validator regression tests
# ════════════════════════════════════════════════════════════════════════════

def test_compile_canonical_rejects_free_variable():
    """Pre-work A: an entity used at top level without an enclosing quantifier
    should be caught by the post-compilation free-variable check.

    validate_extraction passes (entity 'x' is in global_scope), but the
    compiled FOL 'CopyrightViolation(x)' has 'x' free because it is an
    individual variable with no binding quantifier and not a declared constant.
    """
    ext = SentenceExtraction(
        nl="Something violates copyright",
        predicates=_preds(("CopyrightViolation", 1, ["entity"])),
        entities=[Entity(id="x", surface="something", type="entity")],
        formula=_atomic_f("CopyrightViolation", "x"),
    )
    with pytest.raises(SchemaViolation, match="free variable in canonical FOL"):
        compile_canonical_fol(ext)


def test_free_individual_variables_utility():
    """Pre-work A: direct unit tests for the free_individual_variables utility."""
    # Bound variable — empty set
    assert free_individual_variables("all x.(P(x))") == set()

    # Free variable, no declared constants
    assert free_individual_variables("P(x)") == {"x"}

    # x is a declared constant — should NOT be flagged
    assert free_individual_variables("P(x)", frozenset({"x"})) == set()

    # z is free, y is bound
    assert free_individual_variables("exists y.(P(y) & Q(z))") == {"z"}

    # All bound in nested quantifiers
    assert free_individual_variables("exists y.(exists z.(P(y,z)))") == set()

    # Parse failure returns empty set
    assert free_individual_variables("not valid fol )(((") == set()


# ════════════════════════════════════════════════════════════════════════════
# Pre-work B — suite-generator scope bug regression test
# ════════════════════════════════════════════════════════════════════════════

def _nested_existential_case():
    """exists y.(exists z.(PerformanceOf(y,z) & RenaissanceMusic(z))
                 & SpecializedIn(miroslav, y))"""
    q = TripartiteQuantification(
        quantifier="existential",
        variable="y",
        var_type="entity",
        restrictor=[
            _atom("PerformanceOf", "y", "z"),
            _atom("RenaissanceMusic", "z"),
        ],
        nucleus=_atomic_f("SpecializedIn", "miroslav", "y"),
        inner_quantifications=[
            InnerQuantification(quantifier="existential", variable="z", var_type="entity"),
        ],
    )
    return SentenceExtraction(
        nl="test nested existential",
        predicates=_preds(
            ("PerformanceOf", 2, ["entity", "entity"]),
            ("RenaissanceMusic", 1, ["entity"]),
            ("SpecializedIn", 2, ["entity", "entity"]),
        ),
        constants=[_const("miroslav")],
        formula=Formula(quantification=q),
    )


def test_nested_existential_probes_are_closed():
    """Pre-work B: probes from nested exists y.(exists z.(...)) must have
    all variables bound. Before the fix, the inner-quantification variable
    z (renamed to v1) was left free in restrictor-atom probes."""
    suite = compile_sentence_test_suite(_nested_existential_case())
    for p in suite.positives:
        fv = free_individual_variables(p.fol)
        assert not fv, f"Probe has free variables {fv}: {p.fol}"


def test_nested_existential_probe_has_correct_bindings():
    """Pre-work B: the PerformanceOf restrictor-atom probe must have both
    v0 and v1 bound by existentials — not just 'no free vars' but the
    right binding structure."""
    suite = compile_sentence_test_suite(_nested_existential_case())
    performance_probes = [p.fol for p in suite.positives if "PerformanceOf" in p.fol
                          and "SpecializedIn" not in p.fol]
    assert len(performance_probes) == 1, (
        f"Expected exactly 1 PerformanceOf restrictor probe, got {len(performance_probes)}"
    )
    probe = performance_probes[0]
    # Should be: exists v1.(exists v0.(PerformanceOf(v0,v1)))
    # Both variables bound by existentials
    assert probe.count("exists ") == 2, f"Expected 2 existential binders, got: {probe}"
    assert "PerformanceOf(v0,v1)" in probe or "PerformanceOf(v0, v1)" in probe, (
        f"Expected PerformanceOf(v0,v1) in probe: {probe}"
    )
