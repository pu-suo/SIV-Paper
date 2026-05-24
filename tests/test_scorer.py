"""Tests for siv/scorer.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from siv.compiler import compile_canonical_fol, compile_sentence_test_suite
from siv.schema import (
    AtomicFormula,
    Constant,
    Formula,
    PredicateDecl,
    SentenceExtraction,
    TestSuite,
    UnitTest,
)
from siv.scorer import ScoreReport, score
from siv.vampire_interface import is_vampire_available


vampire_required = pytest.mark.skipif(
    not is_vampire_available(), reason="Vampire not available"
)


_EXAMPLES = json.loads(
    (Path(__file__).parent / "data" / "extraction_examples.json").read_text()
)


def _atomic_extraction():
    return SentenceExtraction(
        nl="Miroslav Venhoda was a Czech choral conductor.",
        predicates=[PredicateDecl(name="CzechChoralConductor", arity=1, arg_types=["entity"])],
        entities=[],
        constants=[Constant(id="miroslav", surface="Miroslav Venhoda", type="entity")],
        formula=Formula(atomic=AtomicFormula(pred="CzechChoralConductor", args=["miroslav"])),
    )


def _universal_extraction():
    return SentenceExtraction.model_validate(_EXAMPLES[6]["extraction"])  # No dog is a cat.


# ── Perfect candidate ──────────────────────────────────────────────────────

@vampire_required
def test_perfect_candidate_gives_f1_one_on_nonempty_contrastives():
    """Canonical FOL scored against its own test suite (non-empty
    contrastives) gives recall = 1.0, precision = 1.0, f1 = 1.0."""
    se = _universal_extraction()
    suite = compile_sentence_test_suite(se, timeout_s=5)
    assert len(suite.contrastives) > 0
    report = score(suite, compile_canonical_fol(se), timeout_s=5)
    assert report.recall == 1.0
    assert report.precision == 1.0
    assert report.f1 == 1.0


# ── Recall drop ─────────────────────────────────────────────────────────────

@vampire_required
def test_candidate_missing_positive_gives_recall_drop():
    """A candidate that fails to entail some positive fact reports the
    corresponding recall drop."""
    se = _universal_extraction()  # "No dog is a cat"
    suite = compile_sentence_test_suite(se, timeout_s=5)
    # Wrong candidate: entails the opposite of the canonical
    wrong_candidate = "all x.(Dog(x) -> Cat(x))"  # every dog IS a cat
    report = score(suite, wrong_candidate, timeout_s=5)
    assert report.recall < 1.0


# ── Precision drop ─────────────────────────────────────────────────────────

@vampire_required
def test_candidate_entailing_contrastive_gives_precision_drop():
    """A candidate that entails one of the suite's contrastives reports
    a precision drop."""
    se = _universal_extraction()
    suite = compile_sentence_test_suite(se, timeout_s=5)
    assert len(suite.contrastives) > 0
    # A candidate equal to one of the contrastives entails it.
    contrastive_fol = suite.contrastives[0].fol
    report = score(suite, contrastive_fol, timeout_s=5)
    assert report.precision is not None and report.precision < 1.0


# ── Empty-contrastives (recall-only regime) ────────────────────────────────

@vampire_required
def test_empty_contrastives_reports_recall_only():
    """A perfect candidate on an empty-contrastives test suite gives
    recall = 1.0, precision None, f1 None.

    Constructs the empty-contrastives suite manually rather than relying on
    a particular FOLIO example, since the relaxed gate may admit
    strictly-stronger contrastives for cases that were previously empty.
    """
    se = _atomic_extraction()
    canonical = compile_canonical_fol(se)
    suite = TestSuite(
        extraction=se,
        positives=[UnitTest(fol=canonical, kind="positive")],
        contrastives=[],
    )
    report = score(suite, canonical, timeout_s=5)
    assert report.recall == 1.0
    assert report.precision is None
    assert report.f1 is None


def test_empty_contrastives_per_test_results_includes_marker():
    """per_test_results must make the empty-contrastives situation explicit
    (so downstream consumers don't conflate with a perfect-precision case)."""
    se = _atomic_extraction()
    # Construct a test suite by hand with no contrastives.
    suite = TestSuite(
        extraction=se,
        positives=[UnitTest(fol="CzechChoralConductor(miroslav)", kind="positive")],
        contrastives=[],
    )
    if is_vampire_available():
        report = score(suite, "CzechChoralConductor(miroslav)", timeout_s=5)
    else:
        report = score(suite, "CzechChoralConductor(miroslav)", timeout_s=1)
    # Look for the no-contrastives marker.
    contrastive_entries = [e for e in report.per_test_results if e.kind == "contrastive"]
    assert len(contrastive_entries) == 1
    assert contrastive_entries[0].verdict == "no_contrastives"


# ── Score report shape ─────────────────────────────────────────────────────

@vampire_required
def test_score_report_fields_typed_correctly_nonempty():
    se = _universal_extraction()
    suite = compile_sentence_test_suite(se, timeout_s=5)
    report = score(suite, compile_canonical_fol(se), timeout_s=5)
    assert isinstance(report, ScoreReport)
    assert isinstance(report.recall, float)
    assert isinstance(report.precision, float)
    assert isinstance(report.f1, float)
    assert isinstance(report.positives_entailed, int)
    assert isinstance(report.contrastives_rejected, int)


# ── Canonical scores 1.0 on every base-corpus example ──────────────────────

@vampire_required
@pytest.mark.parametrize(
    "example", _EXAMPLES, ids=[ex["sentence"] for ex in _EXAMPLES],
)
def test_canonical_scores_perfectly_on_own_suite(example):
    """Gate clause: for every base-corpus example, the canonical FOL must score
    recall = 1.0 on its own test suite. If contrastives is non-empty, it
    must also score precision = 1.0."""
    se = SentenceExtraction.model_validate(example["extraction"])
    suite = compile_sentence_test_suite(se, timeout_s=5)
    report = score(suite, compile_canonical_fol(se), timeout_s=5)
    assert report.recall == 1.0, (example["sentence"], report)
    if report.contrastives_total > 0:
        assert report.precision == 1.0, (example["sentence"], report)
        assert report.f1 == 1.0
    else:
        assert report.precision is None
        assert report.f1 is None
