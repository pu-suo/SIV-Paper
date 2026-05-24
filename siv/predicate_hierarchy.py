"""Predicate-hierarchy lookup for severity_correlation_v1 operators.

Loads the curated FOLIO predicate subsumption pairs from
``configs/severity_correlation_v1.yaml`` and exposes query functions:

  - parent_of(p)            : direct 1-hop supertype, or None
  - ancestor_chain(p)       : [p, parent, grandparent, ...] up to root
  - ancestor_at_distance(p, k): ancestor exactly k hops above p, or None
  - strict_subtype_of(p)    : a 1-hop strict subtype (alphabetically first)
  - is_in_hierarchy(p)      : True iff p appears as subtype or supertype

The hierarchy is a tree (each subtype has exactly one direct supertype).
Multi-hop chains are read off the chain directly. Used by:

  - OS_strengthen_predicate (supertype → subtype rewrite in consequent)
  - P_weaken_predicate (1-hop subtype → supertype rewrite)
  - OW_weaken_predicate_severely (≥2-hop subtype → ancestor rewrite)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml

_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "severity_correlation_v1.yaml"

_PARENT: Dict[str, str] = {}
_LOADED = False


def _load() -> None:
    """Lazy-load the hierarchy from the frozen YAML config."""
    global _LOADED
    if _LOADED:
        return
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    for pair in cfg.get("predicate_hierarchy", []):
        _PARENT[pair["subtype"]] = pair["supertype"]
    _LOADED = True


def parent_of(predicate: str) -> Optional[str]:
    """Return the 1-hop supertype, or None if predicate has no parent."""
    _load()
    return _PARENT.get(predicate)


def ancestor_chain(predicate: str) -> List[str]:
    """Return ``[predicate, parent, grandparent, ...]`` up to the root."""
    _load()
    chain = [predicate]
    cur = predicate
    while cur in _PARENT:
        cur = _PARENT[cur]
        chain.append(cur)
    return chain


def ancestor_at_distance(predicate: str, k: int) -> Optional[str]:
    """Return the ancestor exactly k hops above ``predicate``, or None.

    k=0 returns predicate itself, k=1 is parent, k=2 grandparent, etc.
    """
    chain = ancestor_chain(predicate)
    if 0 <= k < len(chain):
        return chain[k]
    return None


def strict_subtype_of(predicate: str) -> Optional[str]:
    """Return a strict 1-hop subtype of ``predicate``, alphabetically first.

    Used by OS_strengthen_predicate (rewrites supertype → subtype in
    consequent to strengthen).
    """
    _load()
    subs = sorted(s for s, parent in _PARENT.items() if parent == predicate)
    return subs[0] if subs else None


def is_in_hierarchy(predicate: str) -> bool:
    """True iff ``predicate`` appears as a subtype OR supertype in the table."""
    _load()
    return predicate in _PARENT or predicate in _PARENT.values()
