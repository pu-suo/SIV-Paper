"""Tests for severity_correlation_v1 operators + stratification.

Contract tests for the 15 deterministic AST-transform operators and the
5-bucket stratification rule. Each operator has a transform test
verifying the AST rewrite produces the expected shape, plus an
applicability test (via the parametrized block at the bottom) verifying
the paired ``<NAME>_applies_to`` predicate correctly filters.

Catalog source of truth: ``configs/severity_correlation_v1.yaml``.
"""
from __future__ import annotations

import pytest

from siv.fol_utils import formula_stratum, parse_fol
from siv.nltk_perturbations import (
    # OS (5)
    OS_add_nucleus_conjunct, OS_add_nucleus_conjunct_applies_to,
    OS_strengthen_quantifier, OS_strengthen_quantifier_applies_to,
    OS_narrow_consequent, OS_narrow_consequent_applies_to,
    OS_strengthen_predicate, OS_strengthen_predicate_applies_to,
    OS_drop_conjunctive_restrictor, OS_drop_conjunctive_restrictor_applies_to,
    # P (4)
    P_drop_conjunct, P_drop_conjunct_applies_to,
    P_drop_consequent_atom, P_drop_consequent_atom_applies_to,
    P_weaken_predicate, P_weaken_predicate_applies_to,
    P_drop_disjunctive_restrictor, P_drop_disjunctive_restrictor_applies_to,
    # OW (6)
    OW_drop_consequent_severely, OW_drop_consequent_severely_applies_to,
    OW_weaken_predicate_severely, OW_weaken_predicate_severely_applies_to,
    OW_de_quantify_to_c0, OW_de_quantify_to_c0_applies_to,
    OW_flip_outer_quantifier, OW_flip_outer_quantifier_applies_to,
    OW_weaken_to_existential, OW_weaken_to_existential_applies_to,
    OW_overrestrict_antecedent, OW_overrestrict_antecedent_applies_to,
    # Catalog
    SEVERITY_V1_OS_OPS, SEVERITY_V1_P_OPS, SEVERITY_V1_OW_OPS,
    SEVERITY_V1_ALL_OPS, SEVERITY_V1_TIER_MAP,
)


# ════════════════════════════════════════════════════════════════════════════
# Catalog completeness (runs at freeze — these are not xfail'd)
# ════════════════════════════════════════════════════════════════════════════

def test_catalog_has_15_operators():
    """5 OS + 4 P + 6 OW = 15 operators total."""
    assert len(SEVERITY_V1_OS_OPS) == 5
    assert len(SEVERITY_V1_P_OPS) == 4
    assert len(SEVERITY_V1_OW_OPS) == 6
    assert len(SEVERITY_V1_ALL_OPS) == 15


def test_catalog_tier_map_is_complete():
    assert set(SEVERITY_V1_TIER_MAP.keys()) == {"overstrong", "partial", "overweak"}
    assert SEVERITY_V1_TIER_MAP["overstrong"] is SEVERITY_V1_OS_OPS
    assert SEVERITY_V1_TIER_MAP["partial"] is SEVERITY_V1_P_OPS
    assert SEVERITY_V1_TIER_MAP["overweak"] is SEVERITY_V1_OW_OPS


def test_catalog_operator_names_are_unique():
    names = [op.__name__ for op in SEVERITY_V1_ALL_OPS]
    assert len(names) == len(set(names))


def test_catalog_every_operator_has_applies_to():
    """Every operator must have a paired <NAME>_applies_to predicate."""
    import siv.nltk_perturbations as np_mod
    for op in SEVERITY_V1_ALL_OPS:
        predicate_name = f"{op.__name__}_applies_to"
        assert hasattr(np_mod, predicate_name), (
            f"missing applicability predicate: {predicate_name}"
        )


# ════════════════════════════════════════════════════════════════════════════
# Per-operator contract tests — TRANSFORM behavior
# ════════════════════════════════════════════════════════════════════════════

def test_OS_add_nucleus_conjunct_transforms_universal_implication():
    expr = parse_fol("all x.(D(x) -> M(x))")
    result = OS_add_nucleus_conjunct(expr)
    # Strengthening: result must be a ∀x.(D(x) → (M(x) ∧ <something>(x)))
    assert "all x." in str(result)
    assert "M(x)" in str(result)
    # New conjunct present
    assert "&" in str(result) or "and" in str(result).lower()


def test_OS_strengthen_quantifier_transforms_nested_existential():
    expr = parse_fol("all x.(H(x) -> exists y.(C(y) & Hold(x, y)))")
    result = OS_strengthen_quantifier(expr)
    # Inner ∃ becomes ∀
    assert "all y." in str(result)
    assert "exists y." not in str(result)


def test_OS_narrow_consequent_transforms_existential_conjunction():
    expr = parse_fol("exists x.(D(x) & M(x))")
    result = OS_narrow_consequent(expr)
    # Existential body gains an extra conjunct
    assert "exists x." in str(result)
    assert str(result) != str(expr)


def test_OS_strengthen_predicate_uses_hierarchy():
    # Gold has Animal in consequent; hierarchy has Mammal ⊏ Animal so the
    # transform should replace Animal with Mammal (or another subtype).
    expr = parse_fol("all x.(D(x) -> Animal(x))")
    result = OS_strengthen_predicate(expr)
    assert "Animal" not in str(result) or str(result) != str(expr)


def test_OS_drop_conjunctive_restrictor_transforms_universal():
    expr = parse_fol("all x.((D(x) & Dom(x)) -> M(x))")
    result = OS_drop_conjunctive_restrictor(expr)
    # One antecedent conjunct removed
    assert "all x." in str(result)
    s = str(result)
    # Either D(x) or Dom(x) survives, not both
    assert ("D(x)" in s) ^ ("Dom(x)" in s) or ("D(x)" in s and "Dom(x)" not in s) or ("Dom(x)" in s and "D(x)" not in s)


def test_P_drop_conjunct_transforms_multi_conjunct():
    expr = parse_fol("D(c) & M(c) & H(c)")
    result = P_drop_conjunct(expr)
    # One conjunct dropped — two remain
    s = str(result)
    n_atoms = sum(1 for pred in ("D(c)", "M(c)", "H(c)") if pred in s)
    assert n_atoms == 2


def test_P_drop_consequent_atom_transforms_implication():
    expr = parse_fol("all x.(R(x) -> (P(x) & Q(x)))")
    result = P_drop_consequent_atom(expr)
    s = str(result)
    # One of P(x), Q(x) survives in consequent; the other is dropped
    has_p = "P(x)" in s
    has_q = "Q(x)" in s
    assert has_p ^ has_q


def test_P_weaken_predicate_uses_hierarchy():
    expr = parse_fol("all x.(D(x) -> Mammal(x))")
    result = P_weaken_predicate(expr)
    # Mammal replaced with a 1-hop supertype (e.g., Animal)
    assert "Mammal" not in str(result)


def test_P_drop_disjunctive_restrictor_transforms_disjunctive_antecedent():
    expr = parse_fol("all x.((D(x) | C(x)) -> M(x))")
    result = P_drop_disjunctive_restrictor(expr)
    s = str(result)
    has_d = "D(x)" in s
    has_c = "C(x)" in s
    assert has_d ^ has_c


def test_OW_drop_consequent_severely_keeps_one_atom():
    expr = parse_fol("all x.(R(x) -> (P(x) & Q(x) & S(x)))")
    result = OW_drop_consequent_severely(expr)
    s = str(result)
    # Exactly one of P, Q, S survives
    n = sum(1 for p in ("P(x)", "Q(x)", "S(x)") if p in s)
    assert n == 1


def test_OW_weaken_predicate_severely_uses_2hop_hierarchy():
    # Labrador ⊏ Dog ⊏ Mammal ⊏ Animal — but we don't have Labrador in the
    # hierarchy directly. Use Dog → Animal (Dog ⊏ Mammal ⊏ Animal, 2 hops).
    expr = parse_fol("all x.(P(x) -> Dog(x))")
    result = OW_weaken_predicate_severely(expr)
    # Dog replaced with a ≥2-hop ancestor (Animal); not the 1-hop Mammal
    s = str(result)
    assert "Dog" not in s


def test_OW_de_quantify_to_c0_uses_fresh_constant():
    expr = parse_fol("all x.(D(x) -> M(x))")
    result = OW_de_quantify_to_c0(expr)
    s = str(result)
    # No outer quantifier; c_0 instantiated
    assert "all x." not in s
    assert "c_0" in s
    assert "D(c_0)" in s
    assert "M(c_0)" in s


def test_OW_flip_outer_quantifier_swaps_forall_to_exists():
    expr = parse_fol("all x.(P(x) -> Q(x))")
    result = OW_flip_outer_quantifier(expr)
    s = str(result)
    assert "exists x." in s
    assert "all x." not in s


def test_OW_weaken_to_existential_converts_universal_implication():
    expr = parse_fol("all x.(D(x) -> M(x))")
    result = OW_weaken_to_existential(expr)
    s = str(result)
    # Becomes ∃x.(D(x) ∧ M(x))
    assert "exists x." in s
    assert "&" in s or "and" in s.lower()
    assert "->" not in s


def test_OW_overrestrict_antecedent_adds_antecedent_conjunct():
    expr = parse_fol("all x.(D(x) -> M(x))")
    result = OW_overrestrict_antecedent(expr)
    s = str(result)
    # Antecedent gains a conjunct: ∀x.((D(x) ∧ <new>(x)) → M(x))
    assert "all x." in s
    assert "&" in s or "and" in s.lower()
    assert "D(x)" in s


# ════════════════════════════════════════════════════════════════════════════
# Per-operator contract tests — APPLICABILITY filtering
# ════════════════════════════════════════════════════════════════════════════
#
# Each operator's applicability predicate must correctly return True for an
# applicable formula and False for an inapplicable one.

_APPLICABLE_TESTS = [
    # (operator's applies_to fn, applicable formula, inapplicable formula)
    (OS_add_nucleus_conjunct_applies_to,        "all x.(D(x) -> M(x))",          None),
    (OS_strengthen_quantifier_applies_to,       "all x.(H(x) -> exists y.C(y))", "all x.P(x)"),
    (OS_narrow_consequent_applies_to,           "exists x.(D(x) & M(x))",        "all x.P(x)"),
    (OS_strengthen_predicate_applies_to,        "all x.(D(x) -> Animal(x))",     "all x.(P(x) -> Q(x))"),  # Q not in hierarchy
    (OS_drop_conjunctive_restrictor_applies_to, "all x.((D(x) & Dom(x)) -> M(x))", "all x.(D(x) -> M(x))"),
    (P_drop_conjunct_applies_to,                "D(c) & M(c) & H(c)",            "D(c)"),
    (P_drop_consequent_atom_applies_to,         "all x.(R(x) -> (P(x) & Q(x)))", "all x.(R(x) -> P(x))"),
    (P_weaken_predicate_applies_to,             "all x.(D(x) -> Mammal(x))",     "all x.(P(x) -> Q(x))"),
    (P_drop_disjunctive_restrictor_applies_to,  "all x.((D(x) | C(x)) -> M(x))", "all x.(D(x) -> M(x))"),
    (OW_drop_consequent_severely_applies_to,    "all x.(R(x) -> (P(x) & Q(x)))", "all x.(R(x) -> P(x))"),
    (OW_weaken_predicate_severely_applies_to,   "all x.(P(x) -> Dog(x))",        "all x.(P(x) -> Q(x))"),  # Q not in hierarchy
    (OW_de_quantify_to_c0_applies_to,           "all x.(D(x) -> M(x))",          "D(c) & M(c)"),
    (OW_flip_outer_quantifier_applies_to,       "all x.(P(x) -> Q(x))",          "exists x.P(x)"),
    (OW_weaken_to_existential_applies_to,       "all x.(D(x) -> M(x))",          "D(c) & M(c)"),
    (OW_overrestrict_antecedent_applies_to,     "all x.(D(x) -> M(x))",          "D(c) & M(c)"),
]


@pytest.mark.parametrize("applies_to,applicable_fol,inapplicable_fol", _APPLICABLE_TESTS)
def test_applicability_predicate_filters_correctly(
    applies_to, applicable_fol, inapplicable_fol,
):
    """Each <op>_applies_to(expr) returns True for applicable, False for not."""
    applicable_expr = parse_fol(applicable_fol)
    assert applies_to(applicable_expr) is True, (
        f"{applies_to.__name__} returned False on applicable formula: {applicable_fol}"
    )
    if inapplicable_fol is not None:
        inapplicable_expr = parse_fol(inapplicable_fol)
        assert applies_to(inapplicable_expr) is False, (
            f"{applies_to.__name__} returned True on inapplicable formula: {inapplicable_fol}"
        )


# ════════════════════════════════════════════════════════════════════════════
# Stratification tests
# ════════════════════════════════════════════════════════════════════════════

_STRATUM_CASES = [
    ("D(rex) & M(rex)",                           1, "propositional"),
    ("all x.P(x)",                                2, "single quantifier"),
    ("exists x.D(x)",                             2, "single existential"),
    ("all x.(D(x) -> M(x))",                      3, "restricted-simple ∀.(R→C)"),
    ("all x.((D(x) & Dom(x)) -> M(x))",           5, "multi-atom restrictor"),
    ("all x.(D(x) -> (M(x) & H(x)))",             5, "multi-atom consequent"),
    ("all x.(D(x) -> exists y.O(x, y))",          4, "nested quantifiers"),
    ("(all x.P(x)) & (all y.Q(y))",               5, "multiple top-level clauses"),
    ("all x.(D(x) -> exists y.(O(x, y) & P(x, y)))", 5, "S5 dominates S4"),
]


@pytest.mark.parametrize("fol,expected_stratum,label", _STRATUM_CASES)
def test_formula_stratum(fol, expected_stratum, label):
    assert formula_stratum(fol) == expected_stratum, (
        f"{label}: expected stratum {expected_stratum} for {fol}"
    )
