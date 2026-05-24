"""Tests for siv/stratum_classifier.py — gold FOL stratum classification."""
from __future__ import annotations

import pytest
from siv.stratum_classifier import classify_stratum_from_fol, STRATUM_LABELS


# ── Per-stratum tests ────────────────────────────────────────────────────────


def test_s1_atomic_ground():
    """Unary ground atom is S1."""
    assert classify_stratum_from_fol("Czech(miroslav)") == "S1_atomic"


def test_s1_atomic_unary():
    assert classify_stratum_from_fol("Happy(john)") == "S1_atomic"


def test_s1_atomic_conjunction():
    """Conjunction of ground atoms is still S1."""
    assert classify_stratum_from_fol("(Happy(john) & Tall(john))") == "S1_atomic"


def test_s2_universal_simple():
    assert classify_stratum_from_fol("all x.(Dog(x) -> Animal(x))") == "S2_universal_simple"


def test_s2_universal_simple_single_restrictor():
    """Universal with exactly one antecedent conjunct."""
    assert classify_stratum_from_fol("all x.(Mammal(x) -> HasSpine(x))") == "S2_universal_simple"


def test_s3_universal_multi_restrictor():
    assert classify_stratum_from_fol(
        "all x.((Cat(x) & Black(x)) -> Friendly(x))"
    ) == "S3_universal_multi_restrictor"


def test_s3_three_conjuncts():
    assert classify_stratum_from_fol(
        "all x.((A(x) & B(x) & C(x)) -> D(x))"
    ) == "S3_universal_multi_restrictor"


def test_s4_nested_quantifier():
    assert classify_stratum_from_fol(
        "all x.(Person(x) -> exists y.(Dog(y) & Owns(x, y)))"
    ) == "S4_nested_quantifier"


def test_s4_double_forall():
    assert classify_stratum_from_fol(
        "all x.(all y.((Parent(x, y) & Child(y, x)) -> Loves(x, y)))"
    ) == "S4_nested_quantifier"


def test_s5_relational():
    assert classify_stratum_from_fol("Taller(michael, john)") == "S5_relational"


def test_s5_relational_binary_with_constants():
    assert classify_stratum_from_fol("LocatedIn(paris, france)") == "S5_relational"


def test_s6_negation_nontrivial():
    """Negation wrapping a connective is non-trivial."""
    assert classify_stratum_from_fol(
        "-(Happy(john) & Tall(john))"
    ) == "S6_negation"


def test_s6_negation_wrapping_quantifier():
    assert classify_stratum_from_fol(
        "-(all x.(Dog(x) -> Animal(x)))"
    ) == "S6_negation"


def test_s6_trivial_negation_is_not_s6():
    """Negation of a single atom is trivial — should be S1, not S6."""
    assert classify_stratum_from_fol("-Happy(john)") == "S1_atomic"


def test_s7_existential():
    assert classify_stratum_from_fol("exists x.(Dog(x) & Flies(x))") == "S7_existential"


def test_s7_existential_not_under_forall():
    """Top-level existential."""
    assert classify_stratum_from_fol(
        "exists x.(Mammal(x) & Flies(x))"
    ) == "S7_existential"


def test_s8_other_implication_no_quantifier():
    """A bare implication with complex connectives but no quantifier: S8."""
    result = classify_stratum_from_fol("(Rains -> Wet(ground))")
    assert result == "S8_other"


# ── Tie-break tests ──────────────────────────────────────────────────────────


def test_tiebreak_s3_over_s6():
    """Multi-restrictor universal with negation in consequent: S3 wins."""
    result = classify_stratum_from_fol(
        "all x.((Dog(x) & Large(x)) -> -(Friendly(x) & Calm(x)))"
    )
    assert result == "S3_universal_multi_restrictor"


def test_tiebreak_s4_over_s2():
    """Nested quantifiers with simple restrictor: S4 wins over S2."""
    result = classify_stratum_from_fol(
        "all x.(Person(x) -> exists y.Loves(x, y))"
    )
    assert result == "S4_nested_quantifier"


def test_tiebreak_s6_over_s2():
    """Universal with non-trivial negation: S6 wins over S2."""
    result = classify_stratum_from_fol(
        "all x.(Dog(x) -> -(Fly(x) & Swim(x)))"
    )
    assert result == "S6_negation"


# ── Edge cases ───────────────────────────────────────────────────────────────


def test_parse_failure_returns_none():
    assert classify_stratum_from_fol("not valid fol!!!") is None


def test_empty_string_returns_none():
    assert classify_stratum_from_fol("") is None


def test_unicode_input():
    """Unicode FOL should be normalized and classified."""
    result = classify_stratum_from_fol("∀x (Dog(x) → Animal(x))")
    assert result == "S2_universal_simple"
