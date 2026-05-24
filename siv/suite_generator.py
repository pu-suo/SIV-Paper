"""
Test-suite generator.

Takes a FOLIO gold FOL string and produces a validated TestSuite via a
fully deterministic pipeline:

  1. Parse gold FOL into a SentenceExtraction (tripartite-quantification
     tree) using siv.fol_parser.parse_gold_fol — no LLM, no network.
  2. Optionally round-trip-verify that re-compiling the extraction back
     to FOL is bidirectionally entailment-equivalent to the input
     (Vampire-checked).
  3. Compile positive sub-entailment probes and contrastive probes via
     siv.compiler.compile_sentence_test_suite.

Public API
----------
generate_test_suite(fol_string, nl, ...) -> SuiteResult
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from siv.compiler import compile_canonical_fol, compile_sentence_test_suite
from siv.fol_parser import ParseError, parse_gold_fol
from siv.fol_utils import normalize_fol_string
from siv.schema import TestSuite
from siv.vampire_interface import check_entailment, is_vampire_available


@dataclass
class SuiteResult:
    """Result of test-suite generation."""

    suite: Optional[TestSuite]
    error: Optional[str] = None
    round_trip_verified: bool = False
    num_positives: int = 0
    num_contrastives: int = 0


def generate_test_suite(
    fol_string: str,
    nl: str = "",
    verify_round_trip: bool = True,
    with_contrastives: bool = True,
    timeout_s: int = 5,
) -> SuiteResult:
    """Generate a deterministic test suite from a gold FOL annotation.

    Parameters
    ----------
    fol_string : str
        Raw gold FOL from FOLIO dataset.
    nl : str
        Natural language sentence (passed through to the extraction as
        a label; not used for any LLM call).
    verify_round_trip : bool
        If True, refuse to emit a suite when re-compiling the parsed
        extraction back to FOL is not bidirectionally equivalent to the
        input. Requires Vampire. If Vampire unavailable, skips
        verification.
    with_contrastives : bool
        If True, generate contrastive tests (requires Vampire).
    timeout_s : int
        Vampire timeout per check.

    Returns
    -------
    SuiteResult
        Contains the TestSuite (or None if generation failed), and
        diagnostic information.
    """
    # Step 1: Parse via deterministic gold parser
    try:
        extraction = parse_gold_fol(fol_string, nl=nl)
    except ParseError as e:
        return SuiteResult(suite=None, error=f"parse_error: {e}")

    # Step 2: Round-trip equivalence verification
    round_trip_ok = False
    if verify_round_trip and is_vampire_available():
        compiled = compile_canonical_fol(extraction)
        normalized_gold = normalize_fol_string(fol_string)

        fwd = check_entailment(compiled, normalized_gold, timeout=timeout_s)
        bwd = check_entailment(normalized_gold, compiled, timeout=timeout_s)

        if fwd is True and bwd is True:
            round_trip_ok = True
        else:
            return SuiteResult(
                suite=None,
                error=f"round_trip_failed: fwd={fwd}, bwd={bwd}",
                round_trip_verified=False,
            )
    elif not verify_round_trip:
        round_trip_ok = True  # Caller opted out
    # else: Vampire unavailable, skip verification but note it

    # Step 3: Generate test suite via existing downstream machinery
    suite = compile_sentence_test_suite(
        extraction,
        with_contrastives=with_contrastives,
        timeout_s=timeout_s,
    )

    return SuiteResult(
        suite=suite,
        round_trip_verified=round_trip_ok,
        num_positives=len(suite.positives),
        num_contrastives=len(suite.contrastives),
    )
