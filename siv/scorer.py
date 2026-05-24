"""
Scorer.

Runs every positive and contrastive in a ``TestSuite`` against a candidate
FOL via Vampire and emits the recall/precision/F1 triple.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from siv.contrastive_generator import derive_witness_axioms
from siv.schema import TestSuite
from siv.vampire_interface import vampire_check


@dataclass
class TestResult:
    """Per-probe result row.

    ``verdict`` is one of {"entailed", "not_entailed", "timeout", "unknown",
    "no_contrastives"}. ``probe_label`` is the (probe_kind, feature_target)
    pair set by the compiler / contrastive generator (may be ``None`` on
    legacy artifacts). ``mutation_kind`` is the contrastive operator name
    (always ``None`` for positives).
    """

    kind: str
    fol: str
    verdict: str
    probe_label: Optional[Tuple[str, Optional[str]]] = None
    mutation_kind: Optional[str] = None


@dataclass
class ScoreReport:
    """Score report / C8.

    ``precision`` and ``f1`` are ``None`` iff ``contrastives_total == 0`` —
    the test suite carried no contrastives, typically because the source
    sentence's structure does not admit provably-unsat mutations under the
    witness-axiom-augmented semantics (§6.5 structural coverage limits).
    """

    recall: float
    precision: Optional[float]
    f1: Optional[float]
    positives_entailed: int
    positives_total: int
    contrastives_rejected: int
    contrastives_total: int
    per_test_results: List[TestResult] = field(default_factory=list)
    # The "no_contrastives" marker (verdict on a synthesized empty-FOL row)
    # makes the structural-empty situation explicit for downstream
    # consumers.


def score(
    test_suite: TestSuite,
    candidate_fol: str,
    timeout_s: int = 5,
    witness_axioms_override: Optional[List[str]] = None,
) -> ScoreReport:
    """Compute recall/precision/F1 for ``candidate_fol`` against ``test_suite``.

    F1 = 2·recall·precision / (recall + precision). No coverage fraction.

    If ``witness_axioms_override`` is provided, use those axioms instead of
    deriving from the extraction.  Used by soft-mode scoring where witness
    axioms need to be rewritten to match the candidate's vocabulary.
    """
    positives_total = len(test_suite.positives)
    contrastives_total = len(test_suite.contrastives)
    if witness_axioms_override is not None:
        witnesses = witness_axioms_override
    else:
        witnesses = derive_witness_axioms(test_suite.extraction)

    positives_entailed = 0
    contrastives_rejected = 0
    per_test: List[TestResult] = []

    for t in test_suite.positives:
        verdict = _entails(candidate_fol, t.fol, timeout_s, witnesses)
        if verdict == "entailed":
            positives_entailed += 1
        per_test.append(TestResult(
            kind="positive",
            fol=t.fol,
            verdict=verdict,
            probe_label=t.probe_label,
            mutation_kind=t.mutation_kind,
        ))

    if contrastives_total == 0:
        # Recall-only regime: the test suite had no contrastives (typically
        # because the source is structurally weak per §6.5). Mark explicitly
        # so downstream consumers don't conflate this with a perfect-precision
        # result.
        per_test.append(TestResult(
            kind="contrastive",
            fol="",
            verdict="no_contrastives",
        ))
    else:
        for t in test_suite.contrastives:
            verdict = _entails(candidate_fol, t.fol, timeout_s, witnesses)
            # For contrastives we want the candidate NOT to entail them.
            # Conservative: treat anything other than a proved "entailed" as
            # "rejected" so a prover timeout doesn't spuriously drop precision.
            if verdict != "entailed":
                contrastives_rejected += 1
            per_test.append(TestResult(
                kind="contrastive",
                fol=t.fol,
                verdict=verdict,
                probe_label=t.probe_label,
                mutation_kind=t.mutation_kind,
            ))

    recall = positives_entailed / positives_total if positives_total else 0.0

    precision: Optional[float]
    f1: Optional[float]
    if contrastives_total == 0:
        precision = None
        f1 = None
    else:
        precision = contrastives_rejected / contrastives_total
        denom = recall + precision
        f1 = (2.0 * recall * precision / denom) if denom else 0.0

    return ScoreReport(
        recall=recall,
        precision=precision,
        f1=f1,
        positives_entailed=positives_entailed,
        positives_total=positives_total,
        contrastives_rejected=contrastives_rejected,
        contrastives_total=contrastives_total,
        per_test_results=per_test,
    )


def _entails(candidate: str, target: str, timeout_s: int, axioms: List[str]) -> str:
    verdict = vampire_check(
        candidate, target, check="entails", timeout=timeout_s, axioms=axioms,
    )
    if verdict == "unsat":
        return "entailed"
    if verdict == "sat":
        return "not_entailed"
    return verdict  # "timeout" or "unknown"
