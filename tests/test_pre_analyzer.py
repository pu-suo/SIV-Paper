"""Tests for siv/pre_analyzer.py."""
import pytest

from siv.pre_analyzer import RequiredFeatures, compute_required_features


# ── Shape of the result ────────────────────────────────────────────────────

def test_required_features_has_only_two_bool_fields():
    rf = compute_required_features("Alice is tall.")
    assert isinstance(rf, RequiredFeatures)
    assert set(vars(rf).keys()) == {"requires_restrictor", "requires_negation"}
    assert isinstance(rf.requires_restrictor, bool)
    assert isinstance(rf.requires_negation, bool)


def test_required_features_is_frozen():
    rf = compute_required_features("Alice is tall.")
    with pytest.raises(Exception):
        rf.requires_restrictor = True  # type: ignore[misc]


# ── requires_restrictor ─────────────────────────────────────────────────────

def test_restrictor_positive_relcl_who():
    assert compute_required_features(
        "All employees who schedule meetings attend the company building."
    ).requires_restrictor is True


def test_restrictor_positive_relcl_that():
    assert compute_required_features(
        "Every student that takes a class passes."
    ).requires_restrictor is True


def test_restrictor_positive_regex_every_who():
    # Regex match even if spaCy's relcl detection missed it.
    assert compute_required_features(
        "Every person who runs is healthy."
    ).requires_restrictor is True


def test_restrictor_negative_simple_universal():
    assert compute_required_features(
        "All dogs are mammals."
    ).requires_restrictor is False


def test_restrictor_negative_plain_sentence():
    assert compute_required_features(
        "Alice is tall."
    ).requires_restrictor is False


# ── requires_negation ───────────────────────────────────────────────────────

def test_negation_positive_not_adverb():
    assert compute_required_features(
        "Alice is not tall."
    ).requires_negation is True


def test_negation_positive_no_lemma():
    assert compute_required_features(
        "No dog is a cat."
    ).requires_negation is True


def test_negation_positive_never_lemma():
    assert compute_required_features(
        "Students never pass."
    ).requires_negation is True


def test_negation_negative_plain_assertion():
    assert compute_required_features(
        "Alice is tall."
    ).requires_negation is False


def test_negation_negative_simple_universal():
    assert compute_required_features(
        "All dogs are mammals."
    ).requires_negation is False
