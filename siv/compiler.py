"""
Recursive Formula compiler.

Two entry points:
  - compile_canonical_fol(extraction) -> str
  - compile_sentence_test_suite(extraction) -> TestSuite

These are deliberately structurally distinct code paths: different traversal
order, different variable-naming, different string-assembly. A bug in one
must not silently propagate to the other. Their outputs must be
bidirectionally entailing (§8.1 soundness invariant).

Neither entry point reads `extraction.nl`; both are pure functions of the
extraction's structural content.

Output format: NLTK-compatible ASCII FOL.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from siv.schema import (
    AtomicFormula,
    Formula,
    SchemaViolation,
    SentenceExtraction,
    TestSuite,
    TripartiteQuantification,
    UnitTest,
    validate_extraction,
)


# ════════════════════════════════════════════════════════════════════════════
# Path A — compile_canonical_fol
#   Top-down recursion, original variable names, f-string assembly,
#   argument list joined with ", ".
# ════════════════════════════════════════════════════════════════════════════

def compile_canonical_fol(extraction: SentenceExtraction) -> str:
    """Emit the full canonical FOL for the extraction.

    Pure function of the extraction's structural content. Does not read `nl`.
    """
    validate_extraction(extraction)
    result = _a_formula(extraction.formula)
    # Post-compilation free-variable check (Pre-work A)
    from siv.fol_utils import free_individual_variables
    declared_constants = {c.id for c in extraction.constants}
    free_vars = free_individual_variables(result, frozenset(declared_constants))
    if free_vars:
        raise SchemaViolation(
            f"free variable in canonical FOL: {', '.join(sorted(free_vars))}"
        )
    return result


def _a_formula(f: Formula) -> str:
    if f.atomic is not None:
        return _a_atom(f.atomic)
    if f.quantification is not None:
        return _a_quant(f.quantification)
    if f.negation is not None:
        return f"-({_a_formula(f.negation)})"
    if f.connective is not None:
        return _a_connective(f.connective, f.operands or [])
    raise SchemaViolation("empty Formula encountered in compile_canonical_fol")


def _a_atom(a: AtomicFormula) -> str:
    if a.pred == "__eq__" and len(a.args) == 2:
        body = f"({a.args[0]} = {a.args[1]})"
        return f"-{body}" if a.negated else body
    body = f"{a.pred}({', '.join(a.args)})"
    return f"-{body}" if a.negated else body


def _a_connective(conn: str, operands: List[Formula]) -> str:
    parts = [_a_formula(op) for op in operands]
    if conn == "and":
        return "(" + " & ".join(parts) + ")"
    if conn == "or":
        return "(" + " | ".join(parts) + ")"
    if conn == "implies":
        return f"({parts[0]} -> {parts[1]})"
    if conn == "iff":
        return f"({parts[0]} <-> {parts[1]})"
    raise SchemaViolation(f"unknown connective {conn!r}")


def _a_quant(q: TripartiteQuantification) -> str:
    r_parts = [_a_atom(atom) for atom in q.restrictor]
    r: Optional[str]
    if len(r_parts) == 0:
        r = None
    elif len(r_parts) == 1:
        r = r_parts[0]
    else:
        r = "(" + " & ".join(r_parts) + ")"

    if r is not None:
        # Inner quantifications: declaration order, innermost-last. Apply in
        # reverse so the last declared wraps tightest around the restrictor.
        for iq in reversed(q.inner_quantifications):
            qk = "all" if iq.quantifier == "universal" else "exists"
            r = f"{qk} {iq.variable}.({r})"

    nuc = _a_formula(q.nucleus)
    outer = "all" if q.quantifier == "universal" else "exists"
    if r is None:
        return f"{outer} {q.variable}.({nuc})"
    if q.quantifier == "universal":
        return f"{outer} {q.variable}.({r} -> {nuc})"
    return f"{outer} {q.variable}.({r} & {nuc})"


# ════════════════════════════════════════════════════════════════════════════
# Path B — compile_sentence_test_suite
#   Alpha-renames every bound variable to v0, v1, v2, ... (structural
#   distinction from Path A), assembles output via str.join rather than
#   f-strings, and joins predicate arguments with "," (no space). Produces
#   alpha-equivalent output to Path A.
# ════════════════════════════════════════════════════════════════════════════

def compile_sentence_test_suite(
    extraction: SentenceExtraction,
    with_contrastives: bool = True,
    timeout_s: int = 5,
) -> TestSuite:
    """Emit a TestSuite for the extraction.

    Emits the full canonical FOL (via Path B) as the first positive, plus one
    positive `UnitTest` per sub-entailment test. Sub-tests are emitted only
    from positions where the canonical freestanding entails the closure.

    When ``with_contrastives=True`` (default), the test suite's ``contrastives``
    list is populated by ``generate_contrastives``; this requires
    Vampire. Pass ``with_contrastives=False`` for fast path-structure tests
    that only care about positives.
    """
    validate_extraction(extraction)
    counter = [0]
    canonical = _b_formula(extraction.formula, {}, counter)
    positives = [UnitTest(
        fol=canonical,
        kind="positive",
        mutation_kind=None,
        probe_label=("full_reference", None),
    )]

    sub_tests: List[Tuple[str, Tuple[str, Optional[str]]]] = []
    _subtest_walk(extraction.formula, [], sub_tests)

    seen = {canonical}
    for fol, label in sub_tests:
        if fol in seen:
            continue
        seen.add(fol)
        positives.append(UnitTest(
            fol=fol,
            kind="positive",
            mutation_kind=None,
            probe_label=label,
        ))

    contrastives: List[UnitTest] = []
    if with_contrastives:
        # Import here to avoid circular import (contrastive_generator imports
        # _a_formula from this module).
        from siv.contrastive_generator import generate_contrastives
        contrastives, _telemetry = generate_contrastives(extraction, timeout_s=timeout_s)

    return TestSuite(extraction=extraction, positives=positives, contrastives=contrastives)


# ── Sub-test walker (emission rules a/b/c) ────────────────────────────────────

def _atom_feature_target(a: AtomicFormula) -> str:
    """Predicate-vocabulary name for an atom — pred name, with equality
    rendered as ``"<arg1>=<arg2>"`` literal form so the argument identities
    are preserved in the label (matching contrastive_generator._atom_target)."""
    if a.pred == "__eq__" and len(a.args) == 2:
        return f"{a.args[0]}={a.args[1]}"
    return a.pred


def _and_operand_feature_target(op: Formula) -> Optional[str]:
    """Phase-1 spec feature_target extraction for rule (a)/(c) AND operands.

    Returns:
      - "<pred>" for an atom (or negated atom); equality atoms are rendered
        as "<arg1>=<arg2>".
      - "<var>:<restrictor_pred>" for a quantification with a unary
        restrictor over the bound variable; "<var>" alone otherwise.
      - None for compositionally complex operands (and/or/implies/iff
        roots, complex negations).
    """
    if op.atomic is not None:
        return _atom_feature_target(op.atomic)
    if op.negation is not None:
        inner = op.negation
        if inner.atomic is not None:
            return _atom_feature_target(inner.atomic)
        return None
    if op.quantification is not None:
        q = op.quantification
        for a in q.restrictor:
            if len(a.args) == 1 and a.args[0] == q.variable and not a.negated:
                return f"{q.variable}:{a.pred}"
        return q.variable
    return None


def _subtest_walk(
    node: Formula,
    enclosing: List[TripartiteQuantification],
    out: List[Tuple[str, Tuple[str, Optional[str]]]],
) -> None:
    """Walk Formula looking for emittable sub-tests.

    `enclosing` = list of enclosing TripartiteQuantifications, outermost first.
    Emissions append ``(fol, (probe_kind, feature_target))`` pairs to ``out``;
    the FOL is already closed over all free variables.

    Emission rules:
      (a) Direct operands of connective="and" (any depth of walk).
          probe_kind = "top_conjunct" when enclosing is empty,
          "nucleus_conjunct" when inside any quantification body.
      (b) Restrictor atoms of an existential TripartiteQuantification, emitted
          as `exists x.(atom)` wrapped by outer-quantifier chain.
          probe_kind = "restrictor_atom".
      (c) AND-conjuncts inside the nucleus of any quantification reached via
          walk (existential or universal); emitted with full-chain closure.
          Same dispatcher as (a), but enclosing is non-empty.

    Blocking: negation, or/implies/iff operands (except AND nested within),
    implies antecedent, iff operands, universal restrictors, atoms with
    negated=True leaves.
    """
    if node.atomic is not None:
        return
    if node.negation is not None:
        return
    if node.quantification is not None:
        q = node.quantification
        if q.quantifier == "existential":
            for atom in q.restrictor:
                _emit_rule_b(atom, q, enclosing, out)
        new_chain = enclosing + [q]
        _subtest_walk(q.nucleus, new_chain, out)
        return
    if node.connective == "and":
        for op in node.operands or []:
            _emit_rule_a(op, enclosing, out)
            _subtest_walk(op, enclosing, out)
        return
    # or / implies / iff: block all operands.
    return


def _emit_rule_a(
    op: Formula,
    enclosing: List[TripartiteQuantification],
    out: List[Tuple[str, Tuple[str, Optional[str]]]],
) -> None:
    """Emit a sub-test for an AND-conjunct operand."""
    counter = [0]
    rename = _build_rename(enclosing, counter)
    op_fol = _b_formula(op, rename, counter)
    closure = _wrap_chain(op_fol, enclosing, rename)
    # Defense-in-depth: skip probes with free variables (Pre-work B)
    from siv.fol_utils import free_individual_variables
    if free_individual_variables(closure):
        return
    probe_kind = "top_conjunct" if len(enclosing) == 0 else "nucleus_conjunct"
    feature_target = _and_operand_feature_target(op)
    out.append((closure, (probe_kind, feature_target)))


def _emit_rule_b(
    atom: AtomicFormula,
    inner_q: TripartiteQuantification,
    enclosing: List[TripartiteQuantification],
    out: List[Tuple[str, Tuple[str, Optional[str]]]],
) -> None:
    """Emit a sub-test for an existential restrictor atom: exists x.(atom)."""
    counter = [0]
    rename = _build_rename(enclosing + [inner_q], counter)
    atom_fol = _b_atom(atom, rename)
    # Only bind inner_q.variable if it actually appears in atom.args. Always
    # binding it produces vacuous probes like ``exists y.(Bear(x))`` when the
    # restrictor atom uses a different bound variable from the enclosing
    # chain — gibberish satisfies these because they amount to "Bear exists".
    inner_fol = atom_fol
    if inner_q.variable in atom.args:
        inner_fol = "".join(["exists ", rename[inner_q.variable], ".(", inner_fol, ")"])
    # Bind any inner_quantification variables that appear in the atom (Pre-work B fix)
    for iq in inner_q.inner_quantifications:
        if iq.variable in atom.args:
            inner_fol = "".join(["exists ", rename[iq.variable], ".(", inner_fol, ")"])
    closure = _wrap_chain(inner_fol, enclosing, rename)
    # Defense-in-depth: skip probes with free variables (Pre-work B)
    from siv.fol_utils import free_individual_variables
    if free_individual_variables(closure):
        return
    out.append((closure, ("restrictor_atom", _atom_feature_target(atom))))


def _build_rename(
    quants: List[TripartiteQuantification],
    counter: list,
) -> dict:
    """Build a Path-B-style rename map covering a chain of enclosing quantifiers."""
    rename: dict = {}
    for q in quants:
        rename[q.variable] = _b_fresh(counter)
        for iq in q.inner_quantifications:
            rename[iq.variable] = _b_fresh(counter)
    return rename


def _wrap_chain(
    inner_fol: str,
    enclosing: List[TripartiteQuantification],
    rename: dict,
) -> str:
    """Wrap `inner_fol` with enclosing quantifiers (innermost wraps first)."""
    import re
    for q in reversed(enclosing):
        if not q.restrictor:
            # Empty restrictor (Category 7): wrap with bare quantifier — but
            # only if the variable actually appears inside the body. Otherwise
            # the binder is vacuous and produces probes like
            # ``exists v0. exists v1. Bear(v0)`` that leak recall to gibberish
            # candidates. The variable name (v0/v1/...) is unique within this
            # closure, so a word-boundary check is sufficient.
            renamed = rename[q.variable]
            if not re.search(rf"\b{re.escape(renamed)}\b", inner_fol):
                continue
            outer = "all" if q.quantifier == "universal" else "exists"
            inner_fol = "".join([outer, " ", renamed, ".(", inner_fol, ")"])
        else:
            r_compiled = _compile_restrictor_for_wrap(q, rename)
            if q.quantifier == "universal":
                inner_fol = "".join([
                    "all ", rename[q.variable], ".(",
                    r_compiled, " -> ", inner_fol, ")",
                ])
            else:
                inner_fol = "".join([
                    "exists ", rename[q.variable], ".(",
                    r_compiled, " & ", inner_fol, ")",
                ])
    return inner_fol


def _compile_restrictor_for_wrap(
    q: TripartiteQuantification,
    rename: dict,
) -> str:
    """Compile q.restrictor (with inner_quantifications wrapping) for closure use."""
    r_parts = [_b_atom(a, rename) for a in q.restrictor]
    if len(r_parts) == 0:
        # Degenerate; treat as trivially true via a tautology is overkill here.
        # None of our nine cases hit this for sub-tests. Fall back to raising.
        raise SchemaViolation(
            "Cannot wrap sub-test: enclosing quantifier has empty restrictor"
        )
    if len(r_parts) == 1:
        r = r_parts[0]
    else:
        r = "".join(["(", " & ".join(r_parts), ")"])
    for iq in reversed(q.inner_quantifications):
        qk = "all" if iq.quantifier == "universal" else "exists"
        r = "".join([qk, " ", rename[iq.variable], ".(", r, ")"])
    return r


def _b_formula(f: Formula, rename: dict, counter: list) -> str:
    if f.atomic is not None:
        return _b_atom(f.atomic, rename)
    if f.quantification is not None:
        return _b_quant(f.quantification, rename, counter)
    if f.negation is not None:
        return "".join(["-(", _b_formula(f.negation, rename, counter), ")"])
    if f.connective is not None:
        return _b_connective(f.connective, f.operands or [], rename, counter)
    raise SchemaViolation("empty Formula encountered in compile_sentence_test_suite")


def _b_atom(a: AtomicFormula, rename: dict) -> str:
    args = [rename.get(x, x) for x in a.args]
    if a.pred == "__eq__" and len(args) == 2:
        body = "".join(["(", args[0], " = ", args[1], ")"])
        return "".join(["-", body]) if a.negated else body
    body = "".join([a.pred, "(", ",".join(args), ")"])
    return "".join(["-", body]) if a.negated else body


def _b_connective(
    conn: str,
    operands: List[Formula],
    rename: dict,
    counter: list,
) -> str:
    parts = [_b_formula(op, rename, counter) for op in operands]
    if conn == "and":
        return "".join(["(", " & ".join(parts), ")"])
    if conn == "or":
        return "".join(["(", " | ".join(parts), ")"])
    if conn == "implies":
        return "".join(["(", parts[0], " -> ", parts[1], ")"])
    if conn == "iff":
        return "".join(["(", parts[0], " <-> ", parts[1], ")"])
    raise SchemaViolation("unknown connective " + repr(conn))


def _b_quant(
    q: TripartiteQuantification,
    rename: dict,
    counter: list,
) -> str:
    new_rename = dict(rename)
    outer_name = _b_fresh(counter)
    new_rename[q.variable] = outer_name
    inner_binders = []
    for iq in q.inner_quantifications:
        nm = _b_fresh(counter)
        new_rename[iq.variable] = nm
        inner_binders.append((iq.quantifier, nm))

    r_parts = [_b_atom(a, new_rename) for a in q.restrictor]
    r: Optional[str]
    if len(r_parts) == 0:
        r = None
    elif len(r_parts) == 1:
        r = r_parts[0]
    else:
        r = "".join(["(", " & ".join(r_parts), ")"])

    if r is not None:
        for iq_quant, iq_name in reversed(inner_binders):
            qk = "all" if iq_quant == "universal" else "exists"
            r = "".join([qk, " ", iq_name, ".(", r, ")"])

    nuc = _b_formula(q.nucleus, new_rename, counter)
    outer = "all" if q.quantifier == "universal" else "exists"
    if r is None:
        return "".join([outer, " ", outer_name, ".(", nuc, ")"])
    if q.quantifier == "universal":
        return "".join([outer, " ", outer_name, ".(", r, " -> ", nuc, ")"])
    return "".join([outer, " ", outer_name, ".(", r, " & ", nuc, ")"])


def _b_fresh(counter: list) -> str:
    name = f"v{counter[0]}"
    counter[0] += 1
    return name

