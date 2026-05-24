"""Unit tests for siv.propositional_le."""
from __future__ import annotations

import pytest

from siv.propositional_le import propositional_le, propositional_le_aligned


def test_identical_formulas_score_one():
    s = propositional_le("all x.(P(x) -> Q(x))", "all x.(P(x) -> Q(x))")
    assert s == pytest.approx(1.0)


def test_alpha_equivalent_score_one():
    """Bound-variable renaming is canonicalised, so alpha-equivalent
    formulas produce identical atom signatures and score 1.0."""
    s = propositional_le("all x.(P(x) -> Q(x))", "all y.(P(y) -> Q(y))")
    assert s == pytest.approx(1.0)


def test_p1138_quantifier_flip_returns_one():
    """The headline worked-example case: ∀x.HC(x)→∃y.(C(y)∧H(x,y)) vs
    the same body under ∃x. Propositional collapse erases the quantifier
    distinction, so LE = 1.0 — exactly the failure mode the paper cites."""
    gold = "all x.(HoldingCompany(x) -> exists y.(Company(y) & Holds(x, y)))"
    cand = "exists x.(HoldingCompany(x) -> exists y.(Company(y) & Holds(x, y)))"
    s = propositional_le(cand, gold)
    assert s == pytest.approx(1.0)


def test_argument_swap_below_one():
    """Loves(a,b) vs Loves(b,a) propositionalize to *distinct* atoms
    (arg tuples differ), so the formulas are 2-atom and agree on 2/4 rows.

    Concrete expected value: 0.5.
    """
    s = propositional_le("Loves(a, b)", "Loves(b, a)")
    assert s == pytest.approx(0.5)


def test_negation_returns_zero():
    """P(a) vs ¬P(a) — single shared atom, formulas disagree on every
    row of the 2-row truth table.

    NOTE: an earlier spec said LE=0.5 with description "disagree on
    every row". Those are inconsistent. The actual value under the
    standard LE definition (agreement_rows / 2^k) is 0.0; we use that.
    """
    s = propositional_le("P(a)", "-P(a)")
    assert s == pytest.approx(0.0)


def test_connective_flip_returns_half():
    """P(a)∧Q(a) vs P(a)∨Q(a) — two shared atoms, 4 rows; agree on
    (0,0) and (1,1) but disagree on (0,1) and (1,0).

    NOTE: an earlier spec said LE=0.75 with description "3/4 rows
    agree". Counting under the formula's truth table gives 2/4. We
    use the actual value 0.5.
    """
    s = propositional_le("P(a) & Q(a)", "P(a) | Q(a)")
    assert s == pytest.approx(0.5)


def test_unparseable_returns_none():
    s = propositional_le("not a formula", "all x.P(x)")
    assert s is None


def test_aligned_synonym_score_one():
    """When candidate predicate name (HasWheels) differs from gold's
    (HasWheel) but the rest of the formula is identical, the aligned
    variant should normalise the synonym and return 1.0."""
    cand = "all x.(Car(x) -> HasWheels(x))"
    gold = "all x.(Car(x) -> HasWheel(x))"

    # Sanity: unaligned variant should NOT score 1.0 (different atoms).
    s_raw = propositional_le(cand, gold)
    assert s_raw is not None
    assert s_raw < 1.0

    # Aligned variant should rescue (modulo alignment behaviour).
    s_aligned = propositional_le_aligned(cand, gold)
    assert s_aligned is not None
    assert s_aligned == pytest.approx(1.0)


def test_universal_vs_existential_atoms_same():
    """Quantifier-only difference with shared atom signatures returns 1.0."""
    s = propositional_le("all x.P(x)", "exists x.P(x)")
    assert s == pytest.approx(1.0)


def test_disjoint_predicate_sets():
    """P(a) and Q(b) share no atoms; 2 vars, 4 rows; agreement on rows
    where both atoms have the same truth value: (0,0)→F=F✓, (0,1)→F≠T,
    (1,0)→T≠F, (1,1)→T=T✓. 2/4 = 0.5."""
    s = propositional_le("P(a)", "Q(b)")
    assert s == pytest.approx(0.5)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
