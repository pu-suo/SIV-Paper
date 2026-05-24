"""
Smatch++ structural-graph baseline (Opitz, EACL 2023).

Converts a SIV ``Formula`` AST to Penman-style triples and scores a
candidate FOL string against a reference FOL string with Smatch++.
This is the FOL graph baseline used in Thatikonda et al. EMNLP 2025
and the closest off-the-shelf structural competitor to SIV.

The conversion follows the mapping below:

* Predicates become nodes with their name as the ``:instance`` label.
  Arguments become outgoing ``:argN`` edges.
* Bound variables become anonymous nodes labeled ``"var"`` so that
  alpha-equivalent formulas (``∀x.P(x)`` vs ``∀y.P(y)``) align.
* Constants become nodes labeled with their surface name so
  ``Loves(john)`` ≠ ``Loves(mary)``.
* Quantifiers become ``:instance = "universal"|"existential"`` nodes
  with ``:var``, ``:restrictor``, and ``:nucleus`` children.
* Connectives become nodes labeled with the connective name and
  ``:opN`` children.

Public API
----------
fol_to_triples(fol_string) -> list[tuple[str, str, str]]
smatchpp_score(candidate_fol, gold_fol) -> Optional[float]
"""
from __future__ import annotations

import logging
from typing import List, Optional, Set, Tuple

from siv.fol_parser import ParseError, parse_gold_fol
from siv.schema import (
    AtomicFormula,
    Formula,
    InnerQuantification,
    TripartiteQuantification,
)

logger = logging.getLogger(__name__)

Triple = Tuple[str, str, str]


# ════════════════════════════════════════════════════════════════════════════
# Smatch++ instance (lazy + cached)
# ════════════════════════════════════════════════════════════════════════════

_smatchpp_instance = None


def _get_smatchpp():
    """Construct a Smatch++ scorer with a triple-passthrough reader and
    a robust alignment solver. Cached at module scope.

    We use HillClimber (4 random inits) instead of the pure ILP solver
    because the underlying Cbc library has a known abort-on-pathological-
    input bug (CbcNauty SIGABRT) that crashes the whole Python process.
    HillClimber is the heuristic the smatchpp authors validate as
    near-ILP-optimal on AMR graphs; it cannot crash and runs faster.
    """
    global _smatchpp_instance
    if _smatchpp_instance is not None:
        return _smatchpp_instance

    from smatchpp import Smatchpp, interfaces, solvers

    class _TripleReader(interfaces.GraphReader):
        def _string2graph(self, input):
            return input  # already a list of triples

    _smatchpp_instance = Smatchpp(
        graph_reader=_TripleReader(),
        alignmentsolver=solvers.HillClimber(rand_inits=4),
    )
    return _smatchpp_instance


# ════════════════════════════════════════════════════════════════════════════
# FOL → Penman triples
# ════════════════════════════════════════════════════════════════════════════


class _Builder:
    """Walks a Formula AST and emits Penman-style triples."""

    def __init__(self):
        self.triples: List[Triple] = []
        self._counter = 0
        # variable name (e.g. "x") -> node id (e.g. "v3"), per active scope.
        # We use a stack of scopes; entering a quantifier pushes a new scope.
        self._var_scopes: List[dict] = [{}]

    # ── id minting ─────────────────────────────────────────────────────────
    def _fresh(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    # ── scope helpers ──────────────────────────────────────────────────────
    def _bind_var(self, name: str) -> str:
        """Bind ``name`` in the current scope and return its node id.
        Emits the ``:instance var`` triple once per binding."""
        node_id = self._fresh("v")
        self._var_scopes[-1][name] = node_id
        self.triples.append((node_id, ":instance", "var"))
        return node_id

    def _lookup_var(self, name: str) -> Optional[str]:
        """Return node id for ``name`` if bound in any active scope."""
        for scope in reversed(self._var_scopes):
            if name in scope:
                return scope[name]
        return None

    def _arg_node(self, name: str) -> str:
        """Resolve an atomic-arg name to a node id.

        If ``name`` matches an active bound variable, reuse its node so
        repeated uses of the same variable share a single node (which is
        what Smatch++ aligns on). Otherwise treat it as a constant and
        mint a fresh node labeled with the constant's surface name.
        """
        bound = self._lookup_var(name)
        if bound is not None:
            return bound
        node_id = self._fresh("c")
        # Lower-case to align with the Smatch++ generic standardizer
        # that lowercases all labels.
        self.triples.append((node_id, ":instance", name.lower()))
        return node_id

    # ── conversion ─────────────────────────────────────────────────────────
    def convert(self, formula: Formula) -> str:
        if formula.atomic is not None:
            return self._convert_atomic(formula.atomic)
        if formula.quantification is not None:
            return self._convert_quantification(formula.quantification)
        if formula.negation is not None:
            node_id = self._fresh("n")
            self.triples.append((node_id, ":instance", "not"))
            child_id = self.convert(formula.negation)
            self.triples.append((node_id, ":operand", child_id))
            return node_id
        if formula.connective is not None:
            return self._convert_connective(formula.connective, formula.operands or [])
        raise ValueError("Empty Formula has no active case")

    def _convert_atomic(self, atom: AtomicFormula) -> str:
        node_id = self._fresh("p")
        self.triples.append((node_id, ":instance", atom.pred.lower()))
        for i, arg in enumerate(atom.args):
            arg_id = self._arg_node(arg)
            self.triples.append((node_id, f":arg{i}", arg_id))
        if atom.negated:
            neg_id = self._fresh("n")
            self.triples.append((neg_id, ":instance", "not"))
            self.triples.append((neg_id, ":operand", node_id))
            return neg_id
        return node_id

    def _convert_connective(self, conn: str, operands: List[Formula]) -> str:
        node_id = self._fresh("k")
        self.triples.append((node_id, ":instance", conn))
        for i, operand in enumerate(operands):
            child_id = self.convert(operand)
            self.triples.append((node_id, f":op{i}", child_id))
        return node_id

    def _convert_quantification(self, q: TripartiteQuantification) -> str:
        node_id = self._fresh("q")
        self.triples.append((node_id, ":instance", q.quantifier))

        # Push a new scope and bind the outer variable.
        self._var_scopes.append({})
        outer_var_id = self._bind_var(q.variable)
        self.triples.append((node_id, ":var", outer_var_id))

        # Inner existentials (e.g. ∃y nested in the antecedent of a ∀x)
        # are bound in the same scope so atoms in the restrictor / nucleus
        # can reference them.
        for j, iq in enumerate(q.inner_quantifications):
            iv_id = self._fresh("iq")
            self.triples.append((iv_id, ":instance", iq.quantifier))
            inner_var_id = self._bind_var(iq.variable)
            self.triples.append((iv_id, ":var", inner_var_id))
            self.triples.append((node_id, f":inner_quant{j}", iv_id))

        # Restrictor: list of atoms. If empty, omit. If single, emit
        # directly. If multiple, wrap in a synthetic ``and`` node.
        if q.restrictor:
            if len(q.restrictor) == 1:
                r_id = self._convert_atomic(q.restrictor[0])
            else:
                r_id = self._fresh("k")
                self.triples.append((r_id, ":instance", "and"))
                for i, atom in enumerate(q.restrictor):
                    self.triples.append((r_id, f":op{i}", self._convert_atomic(atom)))
            self.triples.append((node_id, ":restrictor", r_id))

        # Nucleus
        nucleus_id = self.convert(q.nucleus)
        self.triples.append((node_id, ":nucleus", nucleus_id))

        # Pop the scope so outer references no longer see the bound vars.
        self._var_scopes.pop()
        return node_id


def fol_to_triples(fol_string: str) -> List[Triple]:
    """Parse a FOL string and convert to Penman-style triples.

    Raises ``ParseError`` if the FOL string cannot be parsed; the
    Smatch++ wrapper catches this and returns ``None``.
    """
    extraction = parse_gold_fol(fol_string)
    builder = _Builder()
    top_id = builder.convert(extraction.formula)
    builder.triples.append(("ROOT_OF_GRAPH", ":root", top_id))
    return builder.triples


# ════════════════════════════════════════════════════════════════════════════
# Public scorer
# ════════════════════════════════════════════════════════════════════════════


def smatchpp_score(
    candidate_fol: str,
    gold_fol: str,
) -> Optional[float]:
    """Compute Smatch++ F1 between two FOL strings, in [0, 1].

    Returns ``None`` if either FOL cannot be parsed or Smatch++ raises.
    """
    try:
        cand_triples = fol_to_triples(candidate_fol)
        gold_triples = fol_to_triples(gold_fol)
    except ParseError as e:
        logger.warning("Smatch++ FOL→graph conversion failed: %s", e)
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("Smatch++ FOL→graph conversion raised: %s", e)
        return None

    try:
        scorer = _get_smatchpp()
        result = scorer.score_pair(cand_triples, gold_triples)
        f1 = float(result["main"]["F1"]) / 100.0
    except Exception as e:  # noqa: BLE001
        logger.warning("Smatch++ scoring raised: %s", e)
        return None

    # Clamp to [0, 1] in case of rounding artifacts.
    if f1 < 0.0:
        f1 = 0.0
    elif f1 > 1.0:
        f1 = 1.0
    return f1
