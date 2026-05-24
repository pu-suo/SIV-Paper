"""Tests for siv/aligner.py — cross-vocabulary alignment for soft SIV scoring."""
from __future__ import annotations

import pytest

from siv.aligner import (
    AlignmentResult,
    align_symbols,
    alignment_to_dict,
    extract_symbols_from_fol,
    rewrite_fol_strings,
    rewrite_test_suite,
)
from siv.schema import (
    AtomicFormula,
    Constant,
    Formula,
    PredicateDecl,
    SentenceExtraction,
    TestSuite,
    UnitTest,
)


# ── extract_symbols_from_fol ─────────────────────────────────────────────────


def test_extract_symbols_basic():
    """Parse a universally quantified formula: predicates with arities, no constants."""
    result = extract_symbols_from_fol("all x.(Dog(x) -> Animal(x))")
    assert result["predicates"] == {"Dog": 1, "Animal": 1}
    assert result["constants"] == set()


def test_extract_symbols_with_constants():
    """Parse a ground atom with two constants."""
    result = extract_symbols_from_fol("Likes(john, fido)")
    assert result["predicates"] == {"Likes": 2}
    assert "john" in result["constants"]
    assert "fido" in result["constants"]


def test_extract_symbols_binary_predicate_arity():
    """Binary predicate should have arity 2."""
    result = extract_symbols_from_fol("Owns(alice, book)")
    assert result["predicates"]["Owns"] == 2


def test_extract_symbols_mixed():
    """Formula with both quantified variables and constants."""
    result = extract_symbols_from_fol("all x.(Person(x) -> Likes(x, cake))")
    assert result["predicates"] == {"Person": 1, "Likes": 2}
    assert "cake" in result["constants"]
    # x is a bound variable, not a constant
    assert "x" not in result["constants"]


def test_extract_symbols_graceful_failure():
    """Unparseable FOL returns empty predicates and constants, no exception."""
    result = extract_symbols_from_fol("")
    assert result["predicates"] == {}
    assert result["constants"] == set()

    result2 = extract_symbols_from_fol("not valid fol at all !!!")
    assert isinstance(result2["predicates"], dict)
    assert isinstance(result2["constants"], set)


def test_extract_symbols_regex_fallback():
    """Regex fallback handles predicate extraction from non-NLTK strings."""
    # A string that might not parse but has clear predicate patterns
    result = extract_symbols_from_fol("HasTeeth(platypus)")
    assert "HasTeeth" in result["predicates"]
    assert result["predicates"]["HasTeeth"] == 1


# ── align_symbols ────────────────────────────────────────────────────────────


def test_synonym_alignment():
    """Semantically similar predicates with matching arity should align."""
    siv = {"predicates": {"HasTeeth": 1}, "constants": set()}
    cand = {"predicates": {"HasTooth": 1}, "constants": set()}
    result = align_symbols(siv, cand)
    assert result.predicate_map == {"HasTeeth": "HasTooth"}
    assert len(result.unaligned_siv_predicates) == 0
    assert len(result.unaligned_candidate_predicates) == 0


def test_arity_bucket_isolation():
    """Predicates with different arities must not align, even if names are identical."""
    siv = {"predicates": {"Have": 1, "Possess": 2}, "constants": set()}
    cand = {"predicates": {"Have": 2}, "constants": set()}
    # Use low threshold so the arity test isn't conflated with similarity
    result = align_symbols(siv, cand, threshold=0.3)
    # Have/1 from SIV cannot align to Have/2 from candidate (wrong arity bucket)
    assert "Have" not in result.predicate_map or result.predicate_map.get("Have") != "Have"
    # Possess/2 should align to Have/2 (both arity 2, only option in that bucket)
    assert result.predicate_map.get("Possess") == "Have"
    # Have/1 should be unaligned
    assert "Have/1" in result.unaligned_siv_predicates


def test_one_to_one_enforcement():
    """When two SIV predicates compete for one candidate, only one wins."""
    siv = {"predicates": {"Dog": 1, "Canine": 1}, "constants": set()}
    cand = {"predicates": {"Hound": 1}, "constants": set()}
    result = align_symbols(siv, cand)
    # At most one SIV predicate maps to Hound
    mapped_to_hound = [k for k, v in result.predicate_map.items() if v == "Hound"]
    assert len(mapped_to_hound) <= 1
    # The other should be unaligned
    if mapped_to_hound:
        other = "Dog" if mapped_to_hound[0] == "Canine" else "Canine"
        assert f"{other}/1" in result.unaligned_siv_predicates


def test_threshold_cutoff():
    """Semantically unrelated predicates should stay unaligned."""
    siv = {"predicates": {"Dog": 1}, "constants": set()}
    cand = {"predicates": {"PerformanceMetric": 1}, "constants": set()}
    result = align_symbols(siv, cand, threshold=0.6)
    assert "Dog" not in result.predicate_map
    assert "Dog/1" in result.unaligned_siv_predicates
    assert "PerformanceMetric/1" in result.unaligned_candidate_predicates


def test_determinism():
    """Two identical calls produce identical AlignmentResult."""
    siv = {
        "predicates": {"Dog": 1, "Cat": 1, "Animal": 1},
        "constants": {"fido", "whiskers"},
    }
    cand = {
        "predicates": {"Canine": 1, "Feline": 1, "Creature": 1},
        "constants": {"rex", "tom"},
    }
    r1 = align_symbols(siv, cand)
    r2 = align_symbols(siv, cand)
    assert r1.predicate_map == r2.predicate_map
    assert r1.constant_map == r2.constant_map
    assert r1.predicate_scores == r2.predicate_scores
    assert r1.constant_scores == r2.constant_scores


def test_identity_mappings_in_audit():
    """Exact-match symbols appear in maps with score ~1.0 but are skipped in rewrite."""
    siv = {"predicates": {"Dog": 1}, "constants": {"fido"}}
    cand = {"predicates": {"Dog": 1}, "constants": {"fido"}}
    result = align_symbols(siv, cand)
    # Identity mappings should be in the map
    assert result.predicate_map.get("Dog") == "Dog"
    assert result.constant_map.get("fido") == "fido"

    # Build a trivial test suite to verify rewrite is a no-op for identity maps
    extraction = SentenceExtraction(
        nl="Dogs are animals",
        predicates=[PredicateDecl(name="Dog", arity=1, arg_types=["entity"])],
        constants=[Constant(id="fido", surface="Fido", type="entity")],
        formula=Formula(atomic=AtomicFormula(pred="Dog", args=["fido"])),
    )
    suite = TestSuite(
        extraction=extraction,
        positives=[UnitTest(fol="Dog(fido)", kind="positive")],
    )
    rewritten = rewrite_test_suite(suite, result)
    assert rewritten.positives[0].fol == "Dog(fido)"  # unchanged


def test_constant_alignment():
    """Constants should align semantically."""
    siv = {"predicates": {}, "constants": {"platypus"}}
    cand = {"predicates": {}, "constants": {"platypus"}}
    result = align_symbols(siv, cand)
    assert result.constant_map.get("platypus") == "platypus"


# ── rewrite_test_suite ───────────────────────────────────────────────────────


def _make_simple_suite() -> TestSuite:
    """Build a minimal TestSuite for rewrite tests."""
    extraction = SentenceExtraction(
        nl="The dog has teeth",
        predicates=[
            PredicateDecl(name="HasTeeth", arity=1, arg_types=["entity"]),
            PredicateDecl(name="Dog", arity=1, arg_types=["entity"]),
        ],
        entities=[],
        constants=[Constant(id="fido", surface="Fido", type="animal")],
        formula=Formula(
            connective="and",
            operands=[
                Formula(atomic=AtomicFormula(pred="Dog", args=["fido"])),
                Formula(atomic=AtomicFormula(pred="HasTeeth", args=["fido"])),
            ],
        ),
    )
    return TestSuite(
        extraction=extraction,
        positives=[
            UnitTest(fol="(Dog(fido) & HasTeeth(fido))", kind="positive"),
            UnitTest(fol="Dog(fido)", kind="positive"),
            UnitTest(fol="HasTeeth(fido)", kind="positive"),
        ],
        contrastives=[
            UnitTest(
                fol="(-Dog(fido) & HasTeeth(fido))",
                kind="contrastive",
                mutation_kind="negate_atom",
            ),
        ],
    )


def test_rewrite_preserves_structure():
    """Rewritten suite has same count of positives/contrastives, same extraction."""
    suite = _make_simple_suite()
    alignment = AlignmentResult(
        predicate_map={"HasTeeth": "Have", "Dog": "Canine"},
        constant_map={"fido": "rex"},
        predicate_scores={"HasTeeth/1->Have/1": 0.72, "Dog/1->Canine/1": 0.85},
        constant_scores={"fido->rex": 0.65},
        unaligned_siv_predicates=[],
        unaligned_candidate_predicates=[],
        unaligned_siv_constants=[],
        unaligned_candidate_constants=[],
        threshold=0.6,
    )
    rewritten = rewrite_test_suite(suite, alignment)

    assert len(rewritten.positives) == len(suite.positives)
    assert len(rewritten.contrastives) == len(suite.contrastives)
    assert rewritten.extraction == suite.extraction  # unchanged

    # Check that FOL strings were rewritten
    assert rewritten.positives[0].fol == "(Canine(rex) & Have(rex))"
    assert rewritten.positives[1].fol == "Canine(rex)"
    assert rewritten.positives[2].fol == "Have(rex)"
    assert rewritten.contrastives[0].fol == "(-Canine(rex) & Have(rex))"

    # Kind and mutation_kind preserved
    assert rewritten.contrastives[0].kind == "contrastive"
    assert rewritten.contrastives[0].mutation_kind == "negate_atom"


def test_rewrite_empty_alignment_is_noop():
    """Empty alignment maps should produce unchanged FOL strings."""
    suite = _make_simple_suite()
    alignment = AlignmentResult(
        predicate_map={},
        constant_map={},
        predicate_scores={},
        constant_scores={},
        unaligned_siv_predicates=["HasTeeth/1", "Dog/1"],
        unaligned_candidate_predicates=[],
        unaligned_siv_constants=["fido"],
        unaligned_candidate_constants=[],
        threshold=0.6,
    )
    rewritten = rewrite_test_suite(suite, alignment)
    for orig, rewr in zip(suite.positives, rewritten.positives):
        assert orig.fol == rewr.fol


# ── witness axiom rewriting ──────────────────────────────────────────────────


def test_witness_axioms_rewritten():
    """Witness axioms should use aligned predicate names after rewriting."""
    axioms = ["exists x.Dog(x)", "exists x.HasTeeth(x)"]
    alignment = AlignmentResult(
        predicate_map={"Dog": "Canine", "HasTeeth": "Have"},
        constant_map={},
        predicate_scores={"Dog/1->Canine/1": 0.85, "HasTeeth/1->Have/1": 0.72},
        constant_scores={},
        unaligned_siv_predicates=[],
        unaligned_candidate_predicates=[],
        unaligned_siv_constants=[],
        unaligned_candidate_constants=[],
        threshold=0.6,
    )
    rewritten = rewrite_fol_strings(axioms, alignment)
    assert rewritten[0] == "exists x.Canine(x)"
    assert rewritten[1] == "exists x.Have(x)"


def test_witness_axioms_identity_noop():
    """Identity alignment should leave witness axioms unchanged."""
    axioms = ["exists x.Dog(x)"]
    alignment = AlignmentResult(
        predicate_map={"Dog": "Dog"},
        constant_map={},
        predicate_scores={},
        constant_scores={},
        unaligned_siv_predicates=[],
        unaligned_candidate_predicates=[],
        unaligned_siv_constants=[],
        unaligned_candidate_constants=[],
        threshold=0.6,
    )
    rewritten = rewrite_fol_strings(axioms, alignment)
    assert rewritten[0] == "exists x.Dog(x)"


# ── alignment_to_dict ────────────────────────────────────────────────────────


def test_alignment_to_dict_schema():
    """Serialized alignment has the expected JSON schema."""
    alignment = AlignmentResult(
        predicate_map={"Dog": "Canine"},
        constant_map={"fido": "rex"},
        predicate_scores={"Dog/1->Canine/1": 0.85},
        constant_scores={"fido->rex": 0.65},
        unaligned_siv_predicates=["Cat/1"],
        unaligned_candidate_predicates=["Feline/1"],
        unaligned_siv_constants=[],
        unaligned_candidate_constants=[],
        threshold=0.6,
    )
    d = alignment_to_dict(alignment)
    assert "predicate_map" in d
    assert "constant_map" in d
    assert d["predicate_map"]["Dog"]["candidate"] == "Canine"
    assert d["predicate_map"]["Dog"]["score"] == 0.85
    assert d["constant_map"]["fido"]["candidate"] == "rex"
    assert d["constant_map"]["fido"]["score"] == 0.65
    assert d["unaligned_siv_predicates"] == ["Cat/1"]
    assert d["unaligned_candidate_predicates"] == ["Feline/1"]
    assert d["threshold"] == 0.6
