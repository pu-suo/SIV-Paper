"""
Symbolic pre-analyzer.

Given a natural language sentence, compute two boolean flags used by the
extractor's tripwire enforcement:

- ``requires_restrictor``: the sentence structurally requires a
  ``TripartiteQuantification`` with a non-empty restrictor.
- ``requires_negation``: the sentence structurally requires some negation
  somewhere in the formula tree.

No LLM call. No network call. Deterministic.

**Forbidden**: no modal, temporal, proportional, collective, or ontological-
type detection; no fields on the result beyond the two bools.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import spacy
from spacy.language import Language

_NLP: Optional[Language] = None


def _nlp() -> Language:
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


@dataclass(frozen=True)
class RequiredFeatures:
    requires_restrictor: bool
    requires_negation: bool


_RESTRICTOR_REGEX = re.compile(
    r"^(all|every|each|no|any)\s+\w+\s+(who|that|which)\b",
    re.IGNORECASE,
)

_NEG_LEMMAS = {"no", "none", "never", "neither"}


def compute_required_features(sentence: str) -> RequiredFeatures:
    """Compute the two tripwire flags for ``sentence`` (§6.1)."""
    doc = _nlp()(sentence)

    requires_restrictor = _has_subject_relcl(doc) or bool(
        _RESTRICTOR_REGEX.match(sentence.strip())
    )
    requires_negation = _has_negation(doc)

    return RequiredFeatures(
        requires_restrictor=requires_restrictor,
        requires_negation=requires_negation,
    )


def _has_subject_relcl(doc) -> bool:
    """True if any subject NP is modified by a relative clause."""
    for tok in doc:
        if tok.dep_ in ("nsubj", "nsubjpass"):
            for child in tok.children:
                if child.dep_ == "relcl":
                    return True
    return False


def _has_negation(doc) -> bool:
    for tok in doc:
        if tok.lemma_.lower() in _NEG_LEMMAS:
            return True
        if tok.dep_ == "neg":
            return True
    return False
