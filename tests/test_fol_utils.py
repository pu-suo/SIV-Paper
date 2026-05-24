"""Tests for siv/fol_utils.py"""
import pytest
from siv.fol_utils import (
    normalize_fol_string,
    parse_fol,
    is_valid_fol,
    extract_predicates,
    convert_to_tptp,
    NLTK_AVAILABLE,
)


# ── normalize_fol_string ──────────────────────────────────────────────────────

def test_normalize_empty():
    assert normalize_fol_string("") == ""
    assert normalize_fol_string(None) == ""  # type: ignore

def test_normalize_forall():
    result = normalize_fol_string("forall x.P(x)")
    assert result == "all x.P(x)"

def test_normalize_exist_singular():
    result = normalize_fol_string("exist x.P(x)")
    assert "exists" in result

def test_normalize_whitespace():
    result = normalize_fol_string("all  x . P(x)")
    assert "  " not in result

def test_normalize_tilde_negation():
    result = normalize_fol_string("~P(x)")
    assert result == "-P(x)"

def test_normalize_preserves_nltk():
    fol = "all x.(Dog(x) -> Animal(x))"
    assert normalize_fol_string(fol) == fol


# ── is_valid_fol ──────────────────────────────────────────────────────────────

@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
def test_valid_universal():
    assert is_valid_fol("all x.(Dog(x) -> Animal(x))")

@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
def test_valid_existential():
    assert is_valid_fol("exists x.(Cat(x) & Happy(x))")

@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
def test_valid_ground():
    assert is_valid_fol("Star(sun)")

@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
def test_invalid_unmatched_paren():
    assert not is_valid_fol("exists x.Cat(x")

@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
def test_invalid_empty():
    assert not is_valid_fol("")


# ── extract_predicates ────────────────────────────────────────────────────────

@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
def test_extract_single():
    preds = extract_predicates("Tall(x)")
    assert "Tall" in preds

@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
def test_extract_multiple():
    preds = extract_predicates("exists x.(Car(x) & Crimson(x) & Running(x))")
    assert preds == {"Car", "Crimson", "Running"}

@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
def test_extract_universal():
    preds = extract_predicates("all x.(Dog(x) -> Animal(x))")
    assert "Dog" in preds
    assert "Animal" in preds

@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
def test_extract_negated():
    preds = extract_predicates("all x.(Bird(x) -> -Flies(x))")
    assert "Bird" in preds
    assert "Flies" in preds

def test_extract_regex_fallback():
    # If NLTK parse fails, regex should still find CamelCase predicates
    preds = extract_predicates("CrimsonCar(x) & MovesQuickly(x)")
    assert "CrimsonCar" in preds or "CrimsonCar" in preds


# ── convert_to_tptp ───────────────────────────────────────────────────────────

@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
def test_tptp_exists():
    expr = parse_fol("exists x.Cat(x)")
    tptp = convert_to_tptp(expr)
    assert "?[X]" in tptp
    assert "cat" in tptp.lower()

@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
def test_tptp_all():
    expr = parse_fol("all x.(Dog(x) -> Animal(x))")
    tptp = convert_to_tptp(expr)
    assert "![X]" in tptp
    assert "=>" in tptp

@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
def test_tptp_negation():
    expr = parse_fol("-Cat(x)")
    tptp = convert_to_tptp(expr)
    # Free variable x is universally closed: ![X] : (~(cat(X)))
    assert "~" in tptp
    assert "cat(X)" in tptp

@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
def test_tptp_conjunction():
    expr = parse_fol("Car(x) & Red(x)")
    tptp = convert_to_tptp(expr)
    assert "&" in tptp


# ── Regression: variable-casing in TPTP conversion ────────────────────────────

@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
def test_tptp_bound_variable_cased_consistently_in_body():
    """Quantifier binder and body references must agree on casing.

    TPTP requires variables to start with uppercase and constants with
    lowercase. The prior implementation uppercased the binder (e.g. `![X]`)
    but let body references fall through to `str(expr).lower()`, producing
    `![X] : legislator(x)` — a binder/body mismatch that makes Vampire treat
    `x` as a free constant and fail to prove alpha-equivalences.
    """
    expr = parse_fol("all x.(Dog(x) -> Animal(x))")
    tptp = convert_to_tptp(expr)
    # Binder is uppercase X; every X-argument in the body must also be X,
    # not x.
    assert "![X]" in tptp
    assert "dog(X)" in tptp
    assert "animal(X)" in tptp
    assert "dog(x)" not in tptp


@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
def test_vampire_check_axioms_flip_sat_to_unsat():
    """Witness axioms close the empty-domain escape hatch.

    `all x.(Dog(x) -> Mammal(x))` and `all x.(Dog(x) -> -Mammal(x))` are
    both satisfied by the empty-domain model where no dog exists, so without
    extra axioms Vampire reports `sat`. Adding `exists x.Dog(x)` forces a
    non-empty Dog extension and collapses the joint theory to `unsat`.
    """
    from siv.vampire_interface import is_vampire_available, vampire_check
    if not is_vampire_available():
        pytest.skip("Vampire not available")
    a = "all x.(Dog(x) -> Mammal(x))"
    b = "all x.(Dog(x) -> -Mammal(x))"
    assert vampire_check(a, b, check="unsat", timeout=5) == "sat"
    assert vampire_check(a, b, check="unsat", timeout=5,
                         axioms=["exists x.Dog(x)"]) == "unsat"


@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not installed")
def test_tptp_multichar_variable_name_uppercased():
    """Multi-character bound variables (e.g. `v0`, introduced by the
    Path-B alpha-renamer) must also render as TPTP variables. Previously the
    single-letter heuristic `len(name) == 1` failed `v0`, sending it through
    the lowercase fallback and breaking alpha-equivalence proofs entirely.
    """
    expr = parse_fol("all v0.(Dog(v0) -> Animal(v0))")
    tptp = convert_to_tptp(expr)
    assert "![V0]" in tptp
    assert "dog(V0)" in tptp
    assert "animal(V0)" in tptp
