"""Conversion-correctness tests for the Smatch++ FOL baseline.

These tests verify that the FOL→Penman-triple conversion produces a
graph that Smatch++ aligns sensibly:

* Identical formulas score 1.0.
* Alpha-equivalent formulas (renamed bound variables) score ~1.0
  because bound variables share a generic ``"var"`` instance label.
* Quantifier flips, argument swaps, and unrelated formulas score
  strictly below 1.0 / strictly below the alpha-equivalent score.
* Unparseable inputs return ``None`` rather than raising.
"""
from __future__ import annotations

import pytest

from siv.smatchpp_baseline import fol_to_triples, smatchpp_score


def test_identical_formulas_score_one():
    s = smatchpp_score("all x.(P(x) -> Q(x))", "all x.(P(x) -> Q(x))")
    assert s == pytest.approx(1.0)


def test_alpha_equivalent_close_to_one():
    """Renaming bound variables should not change the structural graph."""
    s = smatchpp_score("all x.(P(x) -> Q(x))", "all y.(P(y) -> Q(y))")
    assert s is not None
    assert s > 0.95, f"alpha-equiv score {s} too low; expected > 0.95"


def test_quantifier_flip_below_one():
    """∀ vs ∃ should not score 1.0."""
    s = smatchpp_score("all x.(P(x) -> Q(x))", "exists x.(P(x) & Q(x))")
    assert s is not None
    assert s < 1.0


def test_argument_swap_below_one():
    """Swapping argument order changes the graph structure."""
    s = smatchpp_score("Loves(john, mary)", "Loves(mary, john)")
    assert s is not None
    assert s < 1.0


def test_unrelated_below_alpha():
    """Smatch++ rarely returns 0.0 (instance-label types anchor partial
    matches), so we assert that an unrelated pair scores strictly below
    a structurally identical alpha-equivalent pair."""
    s_alpha = smatchpp_score("all x.(P(x) -> Q(x))", "all y.(P(y) -> Q(y))")
    s_unrel = smatchpp_score(
        "all x.(P(x) -> Q(x))",
        "exists z.(R(z) & S(z, alice))",
    )
    assert s_alpha is not None and s_unrel is not None
    assert s_unrel < s_alpha
    # Sanity: should be substantially worse than alpha-equivalent.
    assert s_unrel < 0.7


def test_unparseable_returns_none():
    """Parser failure is logged and returns None, not 0.0 or an exception."""
    assert smatchpp_score("not a formula", "all x.P(x)") is None
    assert smatchpp_score("all x.P(x)", "((((") is None


def test_constant_swap_below_one():
    """Different constants should yield scores < 1.0."""
    s = smatchpp_score("Loves(john, mary)", "Loves(alice, bob)")
    assert s is not None
    assert s < 1.0


def test_predicate_change_below_one():
    """Different predicate names should not match."""
    s_same = smatchpp_score("P(x_const)", "P(x_const)")
    s_diff = smatchpp_score("P(x_const)", "Q(x_const)")
    assert s_same == pytest.approx(1.0)
    assert s_diff is not None
    assert s_diff < s_same


def test_triples_are_well_formed():
    """The triple list should be non-empty and rooted."""
    triples = fol_to_triples("all x.(P(x) -> Q(x))")
    assert len(triples) > 0
    roots = [t for t in triples if t[1] == ":root"]
    assert len(roots) == 1
    # All triples are 3-tuples of strings.
    for s, r, t in triples:
        assert isinstance(s, str)
        assert isinstance(r, str)
        assert isinstance(t, str)


def test_negation_changes_graph():
    s = smatchpp_score("P(x_const)", "-P(x_const)")
    assert s is not None
    assert s < 1.0


def test_connective_change_below_one():
    s_and = smatchpp_score("(P(alice) & Q(alice))", "(P(alice) & Q(alice))")
    s_or = smatchpp_score("(P(alice) & Q(alice))", "(P(alice) | Q(alice))")
    assert s_and == pytest.approx(1.0)
    assert s_or is not None
    assert s_or < s_and
