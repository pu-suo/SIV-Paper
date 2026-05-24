"""
Soundness invariants.

Two CI-level checks that run on every compiled test suite and fail the build
on violation.

- ``check_entailment_monotonicity`` (C9a): the conjunction of a test suite's
  positives is bidirectionally equivalent to ``compile_canonical_fol(extraction)``.
  Runs WITHOUT witness axioms — pure compiler-path equivalence.
- ``check_contrastive_soundness`` (C9b): every contrastive in the test suite
  is admissible against gold, under the §6.5 witness axioms. Two relations
  are admissible:
    * ``incompatible`` — gold ∧ contrastive is unsat (mutually inconsistent).
    * ``strictly_stronger`` — gold does not entail contrastive (entails returns
      ``sat``).
  Legacy artifacts without a ``probe_relation`` are read as ``incompatible``.

Timeout and unknown are failures, not passes. There is no soundness bypass.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from siv.contrastive_generator import derive_witness_axioms
from siv.schema import SentenceExtraction, TestSuite
from siv.vampire_interface import vampire_check


def check_entailment_monotonicity(
    extraction: SentenceExtraction,
    test_suite: TestSuite,
    timeout_s: int = 10,
) -> Tuple[bool, Optional[str]]:
    """C9a. With ``P`` = conjunction of positives and ``Q`` = canonical FOL,
    Vampire-check both ``P ⊨ Q`` and ``Q ⊨ P`` without witness axioms.

    Returns ``(True, None)`` iff both directions proved.
    Returns ``(False, reason)`` on the first failure (sat / timeout / unknown).
    """
    from siv.compiler import compile_canonical_fol

    if not test_suite.positives:
        return False, "test suite has no positives"

    p_fol = _conjunction([t.fol for t in test_suite.positives])
    q_fol = compile_canonical_fol(extraction)

    v_pq = vampire_check(p_fol, q_fol, check="entails", timeout=timeout_s)
    if v_pq != "unsat":
        return False, f"P ⊨ Q failed: verdict={v_pq}"

    v_qp = vampire_check(q_fol, p_fol, check="entails", timeout=timeout_s)
    if v_qp != "unsat":
        return False, f"Q ⊨ P failed: verdict={v_qp}"

    return True, None


def check_contrastive_soundness(
    test_suite: TestSuite,
    timeout_s: int = 10,
) -> Tuple[bool, Optional[str]]:
    """C9b. For each contrastive ``C``, dispatch by ``probe_relation``:

    - ``incompatible`` (or legacy ``None``): assert gold ∧ C is unsat under
      §6.5 witness axioms (preserves prior C9b semantics).
    - ``strictly_stronger``: assert gold ⊭ C — i.e. ``vampire_check(gold, C,
      check="entails", axioms=witnesses)`` returns ``sat``. This catches the
      mis-tag where an equivalent or weaker mutant is incorrectly admitted as
      strictly stronger.

    Both checks are against the canonical (gold) FOL — they enforce the
    same gate the contrastive generator uses for admittance. Returns
    ``(True, None)`` iff every contrastive passes its dispatch.
    """
    from siv.compiler import compile_canonical_fol

    if not test_suite.contrastives:
        return True, None

    if not test_suite.positives:
        return False, "test suite has contrastives but no positives"

    from siv.contrastive_generator import swap_binary_args_witness_axioms

    gold_fol = compile_canonical_fol(test_suite.extraction)
    witnesses = derive_witness_axioms(test_suite.extraction)
    # swap_binary_args contrastives are admitted under augmented witnesses
    # (asymmetry/symmetry axioms from the frozen FOLIO table). The
    # soundness check must verify against the SAME regime that admitted
    # the contrastive — otherwise a swap admitted as "incompatible under
    # asymmetry" would falsely fail this check under default witnesses.
    swap_extra = swap_binary_args_witness_axioms(test_suite.extraction.formula)

    for i, c in enumerate(test_suite.contrastives):
        relation = c.probe_relation or "incompatible"
        op_witnesses = (
            witnesses + swap_extra
            if c.mutation_kind == "swap_binary_args"
            else witnesses
        )
        if relation == "incompatible":
            verdict = vampire_check(
                gold_fol, c.fol, check="unsat",
                timeout=timeout_s, axioms=op_witnesses,
            )
            if verdict != "unsat":
                return False, (
                    f"contrastive {i} ({c.mutation_kind}, incompatible) "
                    f"not unsat against gold: verdict={verdict}; fol={c.fol!r}"
                )
        elif relation == "strictly_stronger":
            verdict = vampire_check(
                gold_fol, c.fol, check="entails",
                timeout=timeout_s, axioms=op_witnesses,
            )
            if verdict != "sat":
                return False, (
                    f"contrastive {i} ({c.mutation_kind}, strictly_stronger) "
                    f"is entailed by gold (verdict={verdict}) — should not "
                    f"have been admitted; fol={c.fol!r}"
                )
        else:
            return False, (
                f"contrastive {i} ({c.mutation_kind}) has unknown "
                f"probe_relation={relation!r}"
            )

    return True, None


def _conjunction(fols: List[str]) -> str:
    if len(fols) == 1:
        return fols[0]
    return "(" + " & ".join(f"({f})" for f in fols) + ")"
