"""Tests for siv/nltk_perturbations.py — AST-level FOL perturbation operators."""
from __future__ import annotations

import random

import pytest

from siv.fol_utils import parse_fol, normalize_fol_string
from siv.nltk_perturbations import (
    NotApplicable,
    A_arity_decompose,
    A_compound_decompose,
    A_const_rename,
    A_const_to_unary,
    B_arg_swap,
    B_quantifier_swap,
    B_restrictor_add,
    B_restrictor_drop,
    B_scope_flip,
    C_entity_swap,
    C_negation_drop,
    C_predicate_substitute,
    D_random_predicates,
    select_perturbation,
)


def _parse(fol: str):
    return parse_fol(normalize_fol_string(fol))


def _roundtrip(expr) -> str:
    """Verify the expression round-trips and return its string form."""
    s = str(expr)
    reparsed = parse_fol(s)
    assert reparsed is not None, f"Round-trip failed for: {s}"
    return s


# ── Tier A ───────────────────────────────────────────────────────────────────


class TestTierA:
    def test_arity_decompose_binary_with_constant(self):
        expr = _parse("Loves(alice, bob)")
        result = A_arity_decompose(expr)
        s = _roundtrip(result)
        assert "LovesBob" in s or "LovesAlice" in s
        # Should be unary now
        assert s.count(",") < str(expr).count(",")

    def test_arity_decompose_not_applicable(self):
        expr = _parse("Happy(john)")
        with pytest.raises(NotApplicable):
            A_arity_decompose(expr)

    def test_const_to_unary(self):
        expr = _parse("Has(alice, fever)")
        result = A_const_to_unary(expr)
        s = _roundtrip(result)
        assert "HasFever" in s or "HasAlice" in s

    def test_compound_decompose(self):
        expr = _parse("all x.(ProfessionalTennisPlayer(x) -> Athlete(x))")
        result = A_compound_decompose(expr)
        s = _roundtrip(result)
        assert "Professional" in s
        assert "Player" in s or "Tennis" in s

    def test_compound_decompose_not_applicable_binary(self):
        """Binary predicates should not be decomposed."""
        expr = _parse("Taller(alice, bob)")
        with pytest.raises(NotApplicable):
            A_compound_decompose(expr)

    def test_const_rename(self):
        expr = _parse("Happy(john)")
        rng = random.Random(42)
        result = A_const_rename(expr, rng)
        s = _roundtrip(result)
        # Constant was renamed — not identical to original
        assert s != str(expr)
        assert "Happy" in s  # predicate unchanged

    def test_const_rename_deterministic(self):
        expr = _parse("Happy(john)")
        r1 = str(A_const_rename(expr, random.Random(42)))
        r2 = str(A_const_rename(expr, random.Random(42)))
        assert r1 == r2


# ── Tier B ───────────────────────────────────────────────────────────────────


class TestTierB:
    def test_arg_swap(self):
        expr = _parse("Loves(alice, bob)")
        result = B_arg_swap(expr)
        s = _roundtrip(result)
        assert "Loves(bob,alice)" in s

    def test_arg_swap_not_applicable_unary(self):
        expr = _parse("Happy(john)")
        with pytest.raises(NotApplicable):
            B_arg_swap(expr)

    def test_restrictor_drop(self):
        expr = _parse("all x.((Dog(x) & Large(x)) -> Scary(x))")
        result = B_restrictor_drop(expr)
        s = _roundtrip(result)
        assert "Large" not in s
        assert "Dog" in s
        assert "Scary" in s

    def test_restrictor_drop_not_applicable_single(self):
        expr = _parse("all x.(Dog(x) -> Animal(x))")
        with pytest.raises(NotApplicable):
            B_restrictor_drop(expr)

    def test_restrictor_add(self):
        expr = _parse("all x.(Dog(x) -> Animal(x))")
        result = B_restrictor_add(expr, story_predicates=["Fluffy", "Small"])
        s = _roundtrip(result)
        assert "Fluffy" in s or "Small" in s
        assert "&" in s  # New conjunct added

    def test_restrictor_add_not_applicable_no_extras(self):
        expr = _parse("all x.(Dog(x) -> Animal(x))")
        with pytest.raises(NotApplicable):
            B_restrictor_add(expr, story_predicates=["Dog", "Animal"])

    def test_scope_flip(self):
        expr = _parse("all x.(Person(x) -> exists y.(Dog(y) & Owns(x, y)))")
        result = B_scope_flip(expr)
        s = _roundtrip(result)
        assert s.startswith("exists")

    def test_scope_flip_not_applicable(self):
        expr = _parse("all x.(Dog(x) -> Animal(x))")
        with pytest.raises(NotApplicable):
            B_scope_flip(expr)

    def test_quantifier_swap(self):
        expr = _parse("all x.(Dog(x) -> Animal(x))")
        result = B_quantifier_swap(expr)
        s = _roundtrip(result)
        assert s.startswith("exists")

    def test_quantifier_swap_exists_to_forall(self):
        expr = _parse("exists x.(Dog(x) & Flies(x))")
        result = B_quantifier_swap(expr)
        s = _roundtrip(result)
        assert s.startswith("all")

    def test_quantifier_swap_not_applicable(self):
        expr = _parse("Happy(john)")
        with pytest.raises(NotApplicable):
            B_quantifier_swap(expr)


# ── Tier C ───────────────────────────────────────────────────────────────────


class TestTierC:
    def test_predicate_substitute(self):
        expr = _parse("Tall(john)")
        result = C_predicate_substitute(expr)
        s = _roundtrip(result)
        assert "Short" in s
        assert "Tall" not in s

    def test_predicate_substitute_not_applicable(self):
        expr = _parse("Xyz(john)")
        with pytest.raises(NotApplicable):
            C_predicate_substitute(expr)

    def test_negation_drop(self):
        expr = _parse("-Happy(john)")
        result = C_negation_drop(expr)
        s = _roundtrip(result)
        assert s == "Happy(john)"

    def test_negation_drop_nested(self):
        expr = _parse("all x.(Dog(x) -> -Fly(x))")
        result = C_negation_drop(expr)
        s = _roundtrip(result)
        # Negation removed: -Fly(x) became Fly(x), but -> still present
        assert "-Fly" not in s
        assert "Fly(x)" in s

    def test_negation_drop_not_applicable(self):
        expr = _parse("Happy(john)")
        with pytest.raises(NotApplicable):
            C_negation_drop(expr)

    def test_entity_swap(self):
        expr = _parse("Loves(alice, bob)")
        result = C_entity_swap(expr, story_constants=["carol", "dave"])
        s = _roundtrip(result)
        assert "carol" in s or "dave" in s

    def test_entity_swap_not_applicable(self):
        expr = _parse("all x.(Dog(x) -> Animal(x))")
        with pytest.raises(NotApplicable):
            C_entity_swap(expr, story_constants=[])


# ── Tier D ───────────────────────────────────────────────────────────────────


class TestTierD:
    def test_random_predicates(self):
        expr = _parse("all x.(Dog(x) -> Animal(x))")
        rng = random.Random(42)
        result = D_random_predicates(expr, rng)
        s = _roundtrip(result)
        assert "Dog" not in s
        assert "Animal" not in s

    def test_random_predicates_deterministic(self):
        expr = _parse("all x.(Dog(x) -> Animal(x))")
        r1 = str(D_random_predicates(expr, random.Random(42)))
        r2 = str(D_random_predicates(expr, random.Random(42)))
        assert r1 == r2


# ── Dispatch ─────────────────────────────────────────────────────────────────


class TestDispatch:
    def test_select_tier_b(self):
        expr = _parse("all x.(Dog(x) -> Animal(x))")
        rng = random.Random(42)
        result, op_name = select_perturbation("B", expr, rng)
        s = _roundtrip(result)
        assert s != str(expr)
        assert op_name.startswith("B_")

    def test_select_tier_d_always_works(self):
        expr = _parse("Happy(john)")
        rng = random.Random(42)
        result, op_name = select_perturbation("D", expr, rng)
        assert op_name == "D_random_predicates"
        _roundtrip(result)

    def test_select_with_exclusion(self):
        """Excluding the only applicable op should raise NotApplicable."""
        expr = _parse("Happy(john)")
        rng = random.Random(42)
        with pytest.raises(NotApplicable):
            select_perturbation("D", expr, rng, exclude_ops={"D_random_predicates"})

    def test_two_different_tier_b(self):
        """Generate two different Tier B perturbations via exclusion."""
        expr = _parse("all x.((Dog(x) & Large(x)) -> Animal(x))")
        rng = random.Random(42)
        _, op1 = select_perturbation("B", expr, rng, story_predicates=["Fast"])
        _, op2 = select_perturbation(
            "B", expr, rng, story_predicates=["Fast"], exclude_ops={op1}
        )
        assert op1 != op2
