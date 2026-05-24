#!/usr/bin/env python3
"""Severity-correlation experiment (paper Exp 1) — generation pipeline.

Implements the deterministic AST-based candidate-generation pipeline
specified by ``configs/severity_correlation_v1.yaml``.

Pipeline steps (each writes an artifact in
``reports/experiments/severity_correlation/``):

  --step 1 : Load FOLIO golds, bucket by formula stratum (1-5).
             Output: golds_by_stratum.json
  --step 2 : For each (gold, applicable operator), apply deterministic
             AST transform.  Output: candidates_raw.jsonl
  --step 3 : Verify every candidate's tier via Vampire bidirectional
             entailment.  Derives witness axioms for the one operator
             that needs them (OW_weaken_to_existential).  Output:
             candidates_verified.jsonl (with kept=True/False per row)
  --step 4 : Sample retained candidates down to the YAML's cell targets
             (8 per strict ✓ cell, keep all in low_yield cells).
             Output: candidates.json (the final pool)
  --step 5 : Verbalize each candidate FOL → NL via a single deterministic
             LLM call per formula (cached).  Updates candidates.json in
             place with candidate_nl per row.

  --step deterministic : run steps 1-4 (no LLM call; everything except
                         the NL verbalization needed for BLEU/BERTScore)
  --step all           : run steps 1-5

Source of truth for the operator catalog, stratification, cell targets,
witness-axiom config, and analysis plan is
``configs/severity_correlation_v1.yaml``.  Operator implementations live
in ``siv/nltk_perturbations.py``; the stratification rule lives in
``siv/fol_utils.py:formula_stratum``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = _REPO_ROOT / "configs" / "severity_correlation_v1.yaml"
DATA_DIR = _REPO_ROOT / "reports" / "experiments" / "severity_correlation"
TEST_SUITES_PATH = _REPO_ROOT / "test_suites" / "test_suites.jsonl"

GOLDS_PATH = DATA_DIR / "golds_by_stratum.json"
CANDIDATES_RAW_PATH = DATA_DIR / "candidates_raw.jsonl"
CANDIDATES_VERIFIED_PATH = DATA_DIR / "candidates_verified.jsonl"
CANDIDATES_PATH = DATA_DIR / "candidates.json"
NL_CACHE_DIR = DATA_DIR / ".nl_cache"
META_PATH = DATA_DIR / "run_metadata.json"


# ── Config loader (cached) ───────────────────────────────────────────────────

_CONFIG: Optional[dict] = None


def load_config() -> dict:
    global _CONFIG
    if _CONFIG is None:
        with open(CONFIG_PATH) as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG


def _update_meta(step_key: str, payload: dict) -> None:
    """Append step results to run_metadata.json (preserves prior steps)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    meta = {}
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
    meta[step_key] = payload
    META_PATH.write_text(json.dumps(meta, indent=2))


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — Load FOLIO golds and bucket by stratum
# ═══════════════════════════════════════════════════════════════════════════

def step1_load_and_stratify(max_per_stratum: int = 50, seed: int = 42) -> None:
    """Read FOLIO test_suites.jsonl, compute stratum per premise, bucket,
    and (if any stratum exceeds max_per_stratum) sample down deterministically.
    """
    from siv.fol_utils import formula_stratum

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not TEST_SUITES_PATH.exists():
        logger.error("Missing test suites: %s", TEST_SUITES_PATH)
        sys.exit(1)

    buckets: Dict[int, list] = defaultdict(list)
    parse_fail = 0
    no_canonical = 0

    with open(TEST_SUITES_PATH) as f:
        for line in f:
            r = json.loads(line)
            canonical = r.get("canonical_fol") or r.get("gold_fol")
            if not canonical:
                no_canonical += 1
                continue
            stratum = formula_stratum(canonical)
            if stratum is None:
                parse_fail += 1
                continue
            buckets[stratum].append({
                "premise_id": r["premise_id"],
                "stratum": stratum,
                "canonical_fol": canonical,
                "gold_fol": r.get("gold_fol", canonical),
                "gold_nl": r.get("nl", ""),
                "source": "FOLIO",
            })

    # Sample down per stratum
    rng = random.Random(seed)
    sampled: Dict[int, list] = {}
    for s in range(1, 6):
        items = sorted(buckets.get(s, []), key=lambda x: x["premise_id"])
        if len(items) > max_per_stratum:
            sampled[s] = rng.sample(items, max_per_stratum)
            sampled[s].sort(key=lambda x: x["premise_id"])
        else:
            sampled[s] = items

    output = {str(s): sampled[s] for s in range(1, 6)}
    GOLDS_PATH.write_text(json.dumps(output, indent=2))

    counts = {s: len(sampled[s]) for s in range(1, 6)}
    folio_totals = {s: len(buckets.get(s, [])) for s in range(1, 6)}

    logger.info("Step 1 done. parse_fail=%d, no_canonical=%d", parse_fail, no_canonical)
    for s in range(1, 6):
        logger.info("  stratum %d: %d sampled (of %d available in FOLIO)",
                    s, counts[s], folio_totals[s])

    _update_meta("step1", {
        "max_per_stratum": max_per_stratum,
        "seed": seed,
        "folio_totals_by_stratum": folio_totals,
        "sampled_by_stratum": counts,
        "parse_fail": parse_fail,
        "no_canonical": no_canonical,
    })


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Apply operators (deterministic AST transforms)
# ═══════════════════════════════════════════════════════════════════════════

def step2_generate_candidates() -> None:
    """For each (gold, operator) pair where applies_to(gold) is True, apply
    the deterministic AST transform.  Write all generated candidates to
    candidates_raw.jsonl.  Failures (NotApplicable) are not recorded.
    """
    from siv.fol_utils import parse_fol
    from siv.nltk_perturbations import NotApplicable
    import siv.nltk_perturbations as np_mod

    if not GOLDS_PATH.exists():
        logger.error("Run step 1 first.")
        sys.exit(1)

    cfg = load_config()
    op_specs = cfg["operators"]
    op_names = [s["name"] for s in op_specs]
    op_tier = {s["name"]: s["tier"] for s in op_specs}

    golds_by_stratum = json.loads(GOLDS_PATH.read_text())

    records = []
    n_attempted = 0
    n_applies_true = 0
    n_generated = 0
    n_not_applicable = 0
    n_parse_fail = 0
    n_reparse_fail = 0
    per_op_generated: Counter = Counter()
    per_op_applies: Counter = Counter()

    for stratum_str, golds in golds_by_stratum.items():
        for gold in golds:
            expr = parse_fol(gold["canonical_fol"])
            if expr is None:
                n_parse_fail += 1
                continue

            for op_name in op_names:
                op_fn = getattr(np_mod, op_name)
                applies_to = getattr(np_mod, f"{op_name}_applies_to")
                n_attempted += 1
                if not applies_to(expr):
                    continue
                per_op_applies[op_name] += 1
                n_applies_true += 1

                try:
                    result = op_fn(expr)
                except NotApplicable:
                    n_not_applicable += 1
                    continue

                cand_fol = str(result)
                if parse_fol(cand_fol) is None:
                    n_reparse_fail += 1
                    continue

                records.append({
                    "premise_id": gold["premise_id"],
                    "stratum": gold["stratum"],
                    "operator": op_name,
                    "tier": op_tier[op_name],
                    "gold_fol": gold["gold_fol"],
                    "canonical_fol": gold["canonical_fol"],
                    "gold_nl": gold["gold_nl"],
                    "candidate_fol": cand_fol,
                    "source": gold["source"],
                })
                n_generated += 1
                per_op_generated[op_name] += 1

    with open(CANDIDATES_RAW_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    logger.info(
        "Step 2 done. attempted=%d, applies_to=%d, generated=%d, "
        "not_applicable=%d, parse_fail=%d, reparse_fail=%d",
        n_attempted, n_applies_true, n_generated,
        n_not_applicable, n_parse_fail, n_reparse_fail,
    )
    for op in op_names:
        logger.info("  %s: %d generated (%d applies_to)",
                    op, per_op_generated[op], per_op_applies[op])

    _update_meta("step2", {
        "n_attempted": n_attempted,
        "n_applies_to": n_applies_true,
        "n_generated": n_generated,
        "n_not_applicable": n_not_applicable,
        "n_parse_fail": n_parse_fail,
        "n_reparse_fail": n_reparse_fail,
        "per_op_generated": dict(per_op_generated),
        "per_op_applies": dict(per_op_applies),
    })


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 — Vampire verification (with witness axioms where needed)
# ═══════════════════════════════════════════════════════════════════════════

def step3_verify(timeout: int = 10) -> None:
    """For each raw candidate, run Vampire bidirectional entailment
    (cand⊨gold and gold⊨cand).  The operator's expected_entailment from
    the YAML determines whether the verdict is acceptable.  Candidates
    whose actual verdict differs from expected are marked kept=False.

    OW_weaken_to_existential uses a derived witness axiom
    (∃x.<antecedent>(x)) on the gold⊨cand check.
    """
    from siv.fol_utils import parse_fol
    from siv.vampire_interface import vampire_check, setup_vampire

    setup_vampire()

    if not CANDIDATES_RAW_PATH.exists():
        logger.error("Run step 2 first.")
        sys.exit(1)

    cfg = load_config()
    expected_by_op = {op["name"]: op["expected_entailment"] for op in cfg["operators"]}

    records = [
        json.loads(l) for l in CANDIDATES_RAW_PATH.read_text().splitlines() if l.strip()
    ]

    verified = []
    n_kept = 0
    by_reason: Counter = Counter()
    t0 = time.time()

    for i, rec in enumerate(records):
        gold = rec["canonical_fol"]
        cand = rec["candidate_fol"]
        op_name = rec["operator"]
        expected = expected_by_op[op_name]

        axioms = None
        if op_name == "OW_weaken_to_existential":
            witness = _derive_witness_for_weaken_to_existential(gold)
            if witness is not None:
                axioms = [witness]
        elif op_name in (
            "OS_strengthen_predicate",
            "P_weaken_predicate",
            "OW_weaken_predicate_severely",
        ):
            # Vampire treats predicate symbols as unrelated unless we
            # supply the subsumption axiom ∀x.(subtype(x) → supertype(x)).
            # The operator deterministically uses the hierarchy, so we
            # can recover the pair by diffing gold and cand predicates.
            axioms = _derive_subsumption_axioms(gold, cand)

        # Axioms (subsumption or witness) are passed to both directions.
        # They are safe to pass to forward (don't introduce spurious
        # entailment in the unintended direction) and necessary for
        # OS_strengthen_predicate's forward and the other predicate ops'
        # reverse checks.
        forward = vampire_check(cand, gold, "entails", timeout=timeout, axioms=axioms)
        reverse = vampire_check(gold, cand, "entails", timeout=timeout, axioms=axioms)

        actual = _classify_entailment(forward, reverse)
        kept = (actual == expected)

        rec_out = dict(rec)
        rec_out["forward_verdict"] = forward
        rec_out["reverse_verdict"] = reverse
        rec_out["actual_entailment"] = actual
        rec_out["expected_entailment"] = expected
        rec_out["witness_axioms"] = axioms
        rec_out["kept"] = kept
        if not kept:
            rec_out["drop_reason"] = f"actual={actual}, expected={expected}"
        verified.append(rec_out)

        if kept:
            n_kept += 1
        else:
            by_reason[f"actual={actual}, expected={expected}"] += 1

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            logger.info("  %d/%d  kept=%d  (%.1fs)", i + 1, len(records), n_kept, elapsed)

    with open(CANDIDATES_VERIFIED_PATH, "w") as f:
        for r in verified:
            f.write(json.dumps(r) + "\n")

    elapsed = time.time() - t0
    logger.info(
        "Step 3 done. verified=%d, kept=%d, dropped=%d, %.1fs",
        len(verified), n_kept, len(verified) - n_kept, elapsed,
    )
    for reason, n in by_reason.most_common():
        logger.info("  drop: %s — %d", reason, n)

    _update_meta("step3", {
        "n_verified": len(verified),
        "n_kept": n_kept,
        "n_dropped": len(verified) - n_kept,
        "wall_time_s": round(elapsed, 1),
        "drop_reasons": dict(by_reason),
        "vampire_timeout": timeout,
    })


def _classify_entailment(forward: str, reverse: str) -> str:
    """Map (forward, reverse) Vampire verdicts to a tier label.

    forward = vampire_check(cand, gold, "entails") — does cand ⊨ gold?
    reverse = vampire_check(gold, cand, "entails") — does gold ⊨ cand?

    "unsat" means Vampire proved the entailment; "sat" means it found a
    counter-model; "timeout"/"unknown" means we couldn't decide.
    """
    if forward not in ("unsat", "sat") or reverse not in ("unsat", "sat"):
        return "unresolved"
    if forward == "unsat" and reverse == "unsat":
        return "equivalent"
    if forward == "unsat" and reverse == "sat":
        return "cand_entails_gold"
    if forward == "sat" and reverse == "unsat":
        return "gold_entails_cand"
    return "incompatible"


def _derive_witness_for_weaken_to_existential(gold_fol: str) -> Optional[str]:
    """For gold = ∀x.(A(x) → C(x)), return witness ``∃x.A(x)``."""
    from siv.fol_utils import parse_fol
    from nltk.sem.logic import AllExpression, ImpExpression

    expr = parse_fol(gold_fol)
    if not isinstance(expr, AllExpression):
        return None
    if not isinstance(expr.term, ImpExpression):
        return None
    antecedent_str = str(expr.term.first)
    bv = str(expr.variable)
    return f"exists {bv}.{antecedent_str}"


def _derive_subsumption_axioms(gold_fol: str, cand_fol: str) -> Optional[List[str]]:
    """For predicate-hierarchy operators, derive ``∀x.(subtype(x) → supertype(x))``
    from the predicate-pair that differs between gold and cand.

    Vampire treats predicate symbols as unrelated unless we supply the
    subsumption axiom. The operator deterministically picks the pair from
    the curated hierarchy; we recover it here by diffing predicate names.
    Returns None if the diff is ambiguous (≠1 predicate added and ≠1
    removed) or the pair isn't in the hierarchy.
    """
    from siv.fol_utils import parse_fol
    from siv.nltk_perturbations import _find_predicates
    from siv.predicate_hierarchy import ancestor_chain

    gold_expr = parse_fol(gold_fol)
    cand_expr = parse_fol(cand_fol)
    if gold_expr is None or cand_expr is None:
        return None

    gold_preds = {name for name, _ in _find_predicates(gold_expr)}
    cand_preds = {name for name, _ in _find_predicates(cand_expr)}
    removed = gold_preds - cand_preds  # predicate in gold, dropped in cand
    added = cand_preds - gold_preds    # predicate in cand, new wrt gold
    if len(removed) != 1 or len(added) != 1:
        return None

    r = next(iter(removed))
    a = next(iter(added))
    # Which one is the subtype? Check both directions.
    if a in ancestor_chain(r):       # r ⊏ ... ⊏ a → r is subtype of a
        sub, sup = r, a
    elif r in ancestor_chain(a):     # a ⊏ ... ⊏ r → a is subtype of r
        sub, sup = a, r
    else:
        return None  # not in the same chain

    return [f"all x.({sub}(x) -> {sup}(x))"]


# ═══════════════════════════════════════════════════════════════════════════
# Step 4 — Sample to cell targets (yields the final candidate pool)
# ═══════════════════════════════════════════════════════════════════════════

def step4_sample(seed: int = 42) -> None:
    """Apply cell-count targets from the YAML.  For each (operator, stratum)
    cell:
      target = int (e.g., 8) → sample down to target if more retained
      target = "low_yield"   → keep all retained (no cap)
      target = null          → cell should be empty by construction
    """
    rng = random.Random(seed)

    if not CANDIDATES_VERIFIED_PATH.exists():
        logger.error("Run step 3 first.")
        sys.exit(1)

    cfg = load_config()
    cell_targets = cfg["cell_targets"]

    records = [
        json.loads(l) for l in CANDIDATES_VERIFIED_PATH.read_text().splitlines() if l.strip()
    ]
    kept = [r for r in records if r["kept"]]

    by_cell: Dict[tuple, list] = defaultdict(list)
    for r in kept:
        by_cell[(r["operator"], r["stratum"])].append(r)

    final: list = []
    cell_audit = []

    for op_name, op_targets in cell_targets.items():
        for s_key, target in op_targets.items():
            stratum = int(s_key[1:])  # "S1" → 1
            available = by_cell.get((op_name, stratum), [])
            n_avail = len(available)
            available_sorted = sorted(available, key=lambda x: x["premise_id"])

            if target is None:
                kept_n = 0
                if n_avail > 0:
                    logger.warning(
                        "Cell (%s, S%d) marked inapplicable but has %d candidates — dropping",
                        op_name, stratum, n_avail,
                    )
            elif target == "low_yield":
                kept_n = n_avail
                final.extend(available_sorted)
            elif isinstance(target, int):
                if n_avail > target:
                    sampled = rng.sample(available_sorted, target)
                    sampled.sort(key=lambda x: x["premise_id"])
                    final.extend(sampled)
                    kept_n = target
                else:
                    final.extend(available_sorted)
                    kept_n = n_avail
            else:
                logger.warning(
                    "Cell (%s, S%d) has unrecognised target %r — skipping",
                    op_name, stratum, target,
                )
                kept_n = 0

            cell_audit.append({
                "operator": op_name,
                "stratum": stratum,
                "target": target,
                "available_after_verification": n_avail,
                "kept": kept_n,
            })

    by_tier = Counter(r["tier"] for r in final)
    by_op = Counter(r["operator"] for r in final)
    by_stratum = Counter(r["stratum"] for r in final)

    output = {
        "design_id": cfg["design_id"],
        "generation_seed": seed,
        "n_candidates": len(final),
        "by_tier": dict(by_tier),
        "by_operator": dict(by_op),
        "by_stratum": dict(by_stratum),
        "cell_audit": cell_audit,
        "candidates": final,
    }
    CANDIDATES_PATH.write_text(json.dumps(output, indent=2))

    logger.info("Step 4 done. final candidates: %d", len(final))
    logger.info("  by tier:    %s", dict(by_tier))
    logger.info("  by stratum: %s", dict(by_stratum))

    _update_meta("step4", {
        "n_candidates": len(final),
        "by_tier": dict(by_tier),
        "by_operator": dict(by_op),
        "by_stratum": dict(by_stratum),
        "seed": seed,
    })


# ═══════════════════════════════════════════════════════════════════════════
# Step 5 — Verbalize candidate FOL → NL (LLM, deterministic, cached)
# ═══════════════════════════════════════════════════════════════════════════

def step5_verbalize_nl() -> None:
    """One LLM call per unique candidate FOL.  Deterministic prompt
    (temperature=0).  Each formula verbalized in isolation; the LLM does
    not see other candidates from the same gold."""
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env")

    if not CANDIDATES_PATH.exists():
        logger.error("Run step 4 first.")
        sys.exit(1)

    NL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    output = json.loads(CANDIDATES_PATH.read_text())
    candidates = output["candidates"]

    n_done = 0
    n_cached = 0
    n_fail = 0
    t0 = time.time()

    for i, c in enumerate(candidates):
        if c.get("candidate_nl") is not None:
            n_done += 1
            continue
        nl, hit = _verbalize_fol(c["candidate_fol"])
        if nl is None:
            n_fail += 1
        else:
            c["candidate_nl"] = nl
            if hit:
                n_cached += 1
            else:
                n_done += 1
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            logger.info(
                "  %d/%d  done=%d cached=%d fail=%d  (%.1fs)",
                i + 1, len(candidates), n_done, n_cached, n_fail, elapsed,
            )

    CANDIDATES_PATH.write_text(json.dumps(output, indent=2))
    elapsed = time.time() - t0
    logger.info("Step 5 done. done=%d cached=%d fail=%d, %.1fs",
                n_done, n_cached, n_fail, elapsed)

    _update_meta("step5", {
        "n_done": n_done,
        "n_cached": n_cached,
        "n_fail": n_fail,
        "wall_time_s": round(elapsed, 1),
    })


def _verbalize_fol(fol: str) -> tuple[Optional[str], bool]:
    """Return (nl, was_cached). Single deterministic LLM call when not cached."""
    key = hashlib.sha256(fol.encode()).hexdigest()
    cache_path = NL_CACHE_DIR / f"{key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())["nl"], True

    prompt = _build_nl_prompt(fol)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=200,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        nl = response.choices[0].message.content.strip()
    except Exception as e:  # noqa: BLE001
        logger.warning("LLM verbalization failed for %s: %s", fol[:80], e)
        return None, False

    cache_path.write_text(json.dumps({"fol": fol, "nl": nl}))
    return nl, False


def _build_nl_prompt(fol: str) -> str:
    return (
        "Verbalize the following first-order logic formula into a single "
        "natural-language sentence. Be faithful to the logic and concise; "
        "do not add information not present in the formula, and do not "
        "editorialize.\n"
        "\n"
        f"FOL: {fol}\n"
        "\n"
        "Natural language (one sentence):"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Step 6 — Score every candidate with the 6 metrics
# ═══════════════════════════════════════════════════════════════════════════

SCORED_PATH = DATA_DIR / "scored.json"


def step6_score(timeout: int = 5) -> None:
    """Score every candidate (and its gold) with all 6 metrics from the
    pre-registered analysis plan.

    Metrics (per configs/severity_correlation_v1.yaml):
      siv_soft_recall, siv_soft_f1, propositional_le_aligned,
      smatchpp, bleu, bertscore

    Writes scored.json with per-row score dicts.
    """
    from experiments.common import (
        load_test_suites,
        score_siv_soft,
        score_propositional_le_aligned,
        score_smatchpp,
        score_bleu, score_bertscore,
    )
    from siv.vampire_interface import setup_vampire

    setup_vampire()

    if not CANDIDATES_PATH.exists():
        logger.error("Run step 4 first.")
        sys.exit(1)

    suites = load_test_suites(TEST_SUITES_PATH)

    data = json.loads(CANDIDATES_PATH.read_text())
    candidates = data["candidates"]

    scored_rows = []
    n_done = 0
    n_fail = 0
    t0 = time.time()

    # Also score each premise's gold (canonical) as the reference floor
    # — included so per-tier means can show gold = 1.0 (or near).
    gold_keys_seen = set()

    for i, c in enumerate(candidates):
        pid = c["premise_id"]
        suite_row = suites.get(pid)
        if suite_row is None:
            n_fail += 1
            continue

        # Score the candidate
        cand_scores = _score_single(suite_row, c["candidate_fol"], c["gold_fol"], timeout)
        scored_rows.append({
            "premise_id": pid,
            "stratum": c["stratum"],
            "tier": c["tier"],
            "operator": c["operator"],
            "candidate_fol": c["candidate_fol"],
            "gold_fol": c["gold_fol"],
            "canonical_fol": c["canonical_fol"],
            "scores": cand_scores,
        })

        # Score gold once per premise
        if pid not in gold_keys_seen:
            gold_keys_seen.add(pid)
            gold_scores = _score_single(
                suite_row, c["canonical_fol"], c["gold_fol"], timeout,
            )
            scored_rows.append({
                "premise_id": pid,
                "stratum": c["stratum"],
                "tier": "gold",
                "operator": "(gold)",
                "candidate_fol": c["canonical_fol"],
                "gold_fol": c["gold_fol"],
                "canonical_fol": c["canonical_fol"],
                "scores": gold_scores,
            })

        n_done += 1
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            logger.info(
                "  %d/%d candidates  done=%d fail=%d  (%.1fs)",
                i + 1, len(candidates), n_done, n_fail, elapsed,
            )

    output = {
        "design_id": data["design_id"],
        "n_candidates": len(candidates),
        "n_scored_rows": len(scored_rows),
        "n_gold_rows": len(gold_keys_seen),
        "rows": scored_rows,
    }
    SCORED_PATH.write_text(json.dumps(output, indent=2))

    elapsed = time.time() - t0
    logger.info("Step 6 done. scored=%d rows (%d candidates + %d gold), %.1fs",
                len(scored_rows), len(candidates), len(gold_keys_seen), elapsed)

    _update_meta("step6", {
        "n_candidates_scored": n_done,
        "n_scoring_failed": n_fail,
        "n_gold_rows": len(gold_keys_seen),
        "n_total_rows": len(scored_rows),
        "wall_time_s": round(elapsed, 1),
        "metrics_scored": [
            "siv_soft_recall", "siv_soft_f1",
            "propositional_le_aligned", "smatchpp", "bleu", "bertscore",
        ],
    })


def _score_single(suite_row: dict, cand_fol: str, gold_fol: str, timeout: int) -> Dict[str, Optional[float]]:
    """Compute all 6 metrics for one (cand_fol, gold_fol) pair."""
    from experiments.common import (
        score_siv_soft, score_propositional_le_aligned, score_smatchpp,
        score_bleu, score_bertscore,
    )

    out: Dict[str, Optional[float]] = {}
    siv_rep = score_siv_soft(suite_row, cand_fol, timeout=timeout)
    if siv_rep is not None:
        out["siv_soft_recall"] = siv_rep.recall
        out["siv_soft_precision"] = siv_rep.precision
        out["siv_soft_f1"] = siv_rep.f1
    else:
        out["siv_soft_recall"] = None
        out["siv_soft_precision"] = None
        out["siv_soft_f1"] = None
    out["propositional_le_aligned"] = score_propositional_le_aligned(cand_fol, gold_fol, timeout=timeout)
    out["smatchpp"] = score_smatchpp(cand_fol, gold_fol)
    out["bleu"] = score_bleu(cand_fol, gold_fol)
    out["bertscore"] = score_bertscore(cand_fol, gold_fol)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Step 7 — Pre-registered analysis (aggregate ρ, per-tier means, AUC)
# ═══════════════════════════════════════════════════════════════════════════

ANALYSIS_DIR = DATA_DIR / "analysis"
GIB_FLOOR_PATH = DATA_DIR / "gib_floor.json"

METRICS_FOR_ANALYSIS = [
    "siv_soft_recall", "siv_soft_f1",
    "propositional_le_aligned", "smatchpp", "bleu", "bertscore",
]
# Severity rank: gold = 1 (reference floor, excluded from ρ),
# overstrong = 2 ≈ partial = 2, overweak = 3.
SEVERITY_RANK = {"gold": 1, "overstrong": 2, "partial": 2, "overweak": 3}


def step7_analyze() -> None:
    """Pre-registered analysis. The primacy hierarchy is:

    PRIMARY    — η² (variance explained by tier) + Cohen's d (REF-vs-OW
                 effect size), written to global_separation.json.
                 These measure the absolute-magnitude calibration of
                 severity that SIV is designed to provide.
    SECONDARY  — adjacent-tier AUCs (separation_aucs.json) and the
                 REF-vs-OW / REF-vs-perturbed AUCs already in
                 global_separation.json. Saturated by self-identity on
                 REF for SIV-soft-F1, SIV-soft-recall, Smatch++; see
                 §4 of results.md for the framing.
    TERTIARY   — aggregate Spearman ρ (rank_correlation.json). Retained
                 as a descriptive statistic. Punishes SIV's binary-leaning
                 scoring via within-premise ties; see §4.3 of results.md
                 for the discussion.

    All outputs go to reports/experiments/severity_correlation/analysis/.
    """
    import numpy as np
    from scipy import stats as scipy_stats

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    if not SCORED_PATH.exists():
        logger.error("Run step 6 first.")
        sys.exit(1)

    scored = json.loads(SCORED_PATH.read_text())
    rows = scored["rows"]

    # ── Per-tier means (descriptive evidence under the η² summary) ──────────
    per_tier = _compute_per_tier_means(rows)
    (ANALYSIS_DIR / "per_tier_means.json").write_text(json.dumps(per_tier, indent=2))
    logger.info("Wrote per_tier_means.json")

    # ── Severity-monotonicity check ─────────────────────────────────────────
    monotonic = _compute_monotonicity(per_tier)
    (ANALYSIS_DIR / "monotonicity.json").write_text(json.dumps(monotonic, indent=2))
    logger.info("Wrote monotonicity.json")

    # ── PRIMARY: η² + Cohen's d ─────────────────────────────────────────────
    global_sep = _compute_global_separation(rows)
    (ANALYSIS_DIR / "global_separation.json").write_text(json.dumps(global_sep, indent=2))
    logger.info("Wrote global_separation.json (PRIMARY)")
    for m in METRICS_FOR_ANALYSIS:
        cell = global_sep["metrics"][m]
        e3 = cell["eta_squared_3level"]["value"]
        d = cell["cohens_d_ref_vs_ow"]
        logger.info("  η²[%s] = %.4f  |  d_REF-vs-OW = %+.4f [%+.4f, %+.4f]",
                    m, e3, d["value"], d["ci_lo"], d["ci_hi"])

    # ── SECONDARY: aggregate AUC measures included in global_separation;
    #              the per-stratum decomposition + per-operator AUC below.
    per_stratum = _compute_per_stratum(rows)
    (ANALYSIS_DIR / "per_stratum.json").write_text(json.dumps(per_stratum, indent=2))
    logger.info("Wrote per_stratum.json")

    per_op_auc = _compute_per_operator_auc(rows)
    (ANALYSIS_DIR / "per_operator_auc.json").write_text(json.dumps(per_op_auc, indent=2))
    logger.info("Wrote per_operator_auc.json")

    # ── TERTIARY: aggregate Spearman ρ (retained, demoted) ──────────────────
    rho_results = _compute_aggregate_rho(rows)
    (ANALYSIS_DIR / "rank_correlation.json").write_text(json.dumps(rho_results, indent=2))
    logger.info("Wrote rank_correlation.json (TERTIARY)")
    for m in METRICS_FOR_ANALYSIS:
        v = rho_results.get(m, {})
        if v.get("mean_rho") is not None:
            logger.info("  ρ[%s] = %.4f  [%.4f, %.4f]  (n=%d)",
                        m, v["mean_rho"], v["ci_lo"], v["ci_hi"], v["n_premises"])

    _update_meta("step7", {
        "analysis_dir": str(ANALYSIS_DIR.relative_to(_REPO_ROOT)),
        "metrics": METRICS_FOR_ANALYSIS,
        "primary_headline_eta_squared_3level": {
            m: global_sep["metrics"][m]["eta_squared_3level"]["value"]
            for m in METRICS_FOR_ANALYSIS
        },
        "primary_headline_cohens_d_ref_vs_ow": {
            m: global_sep["metrics"][m]["cohens_d_ref_vs_ow"]["value"]
            for m in METRICS_FOR_ANALYSIS
        },
        "tertiary_aggregate_rho": {
            m: rho_results.get(m, {}).get("mean_rho")
            for m in METRICS_FOR_ANALYSIS
        },
    })


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY analysis: η² + Cohen's d
# ─────────────────────────────────────────────────────────────────────────────

def _compute_global_separation(rows: list, n_resamples: int = 1000,
                                seed: int = 42) -> dict:
    """Population-level discrimination statistics (primary analysis).

    For each metric:
      • η² (3-level OS/P/OW) and η² (4-level REF + OS/P/OW) — fraction of
        score variance explained by tier. Uses ALL candidates (not the
        last-write-wins per-(premise, tier) aggregation), so the 3-level
        denominator is the 267-candidate non-REF pool and the 4-level
        denominator is 372 (267 + 105 REF rows).
      • Cohen's d on REF vs OW, with 95% bootstrap CI over 1000 resamples
        of each tier independently. Uses per-(premise, tier) aggregation
        matching the official ρ code.
      • Auxiliary global ROC AUC for REF-vs-OW and REF-vs-perturbed (the
        same per-(premise, tier) aggregation), restricted to the 53
        ρ-eligible premises (≥2 non-gold tiers present).
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from collections import defaultdict

    # All-candidate groups for η² (one row per candidate)
    all_by_tier: dict = defaultdict(lambda: defaultdict(list))
    non_ref_by_tier: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m in METRICS_FOR_ANALYSIS:
            v = r["scores"].get(m)
            if v is None:
                continue
            all_by_tier[m][r["tier"]].append(float(v))
            if r["tier"] != "gold":
                non_ref_by_tier[m][r["tier"]].append(float(v))

    # Per-(premise, tier) cells for Cohen's d + AUC (last-write-wins)
    by_premise: dict = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        for m in METRICS_FOR_ANALYSIS:
            v = r["scores"].get(m)
            if v is not None:
                by_premise[r["premise_id"]][r["tier"]][m] = float(v)

    # 53-premise ρ-eligible set (premises with ≥2 non-gold tiers present)
    eligible_53 = sorted(
        pid for pid, by_tier in by_premise.items()
        if sum(1 for t in ("overstrong", "partial", "overweak") if t in by_tier) >= 2
    )

    def tier_pool(metric: str, tier: str, pids: list) -> "np.ndarray":
        out = []
        for pid in pids:
            if tier in by_premise[pid]:
                v = by_premise[pid][tier].get(metric)
                if v is not None:
                    out.append(v)
        return np.array(out, dtype=float)

    def eta_squared(groups: dict) -> Optional[float]:
        if not groups or all(len(g) == 0 for g in groups.values()):
            return None
        all_scores = np.concatenate([np.asarray(g, dtype=float) for g in groups.values()])
        grand_mean = all_scores.mean()
        ss_total = float(((all_scores - grand_mean) ** 2).sum())
        if ss_total == 0:
            return None
        ss_between = float(sum(
            len(g) * (np.mean(g) - grand_mean) ** 2
            for g in groups.values() if len(g) > 0
        ))
        return ss_between / ss_total

    def cohens_d(x: "np.ndarray", y: "np.ndarray") -> Optional[float]:
        n1, n2 = len(x), len(y)
        if n1 < 2 or n2 < 2:
            return None
        v1 = float(x.var(ddof=1))
        v2 = float(y.var(ddof=1))
        pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        if pooled <= 0:
            return None
        return float((x.mean() - y.mean()) / np.sqrt(pooled))

    def auc(neg: "np.ndarray", pos: "np.ndarray") -> Optional[float]:
        if len(neg) == 0 or len(pos) == 0:
            return None
        y_true = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
        y_score = -np.concatenate([neg, pos])
        if len(set(y_score.tolist())) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_score))

    def bootstrap(stat_fn, *arrays, n: int = n_resamples) -> tuple:
        rng = np.random.RandomState(seed)
        vals = []
        for _ in range(n):
            resamples = [a[rng.randint(0, len(a), size=len(a))] for a in arrays]
            v = stat_fn(*resamples)
            if v is not None and not np.isnan(v):
                vals.append(v)
        if not vals:
            return (None, None)
        return (float(np.percentile(vals, 2.5)),
                float(np.percentile(vals, 97.5)))

    out: dict = {}
    for m in METRICS_FOR_ANALYSIS:
        # η²
        groups3 = non_ref_by_tier[m]
        groups4 = all_by_tier[m]
        eta3 = eta_squared(groups3)
        eta4 = eta_squared(groups4)
        n_total_3 = sum(len(v) for v in groups3.values())
        n_total_4 = sum(len(v) for v in groups4.values())

        # Cohen's d on REF vs OW (per-(premise, tier) aggregation, 53 pool)
        ref = tier_pool(m, "gold", eligible_53)
        ow = tier_pool(m, "overweak", eligible_53)
        os_ = tier_pool(m, "overstrong", eligible_53)
        p_ = tier_pool(m, "partial", eligible_53)
        d = cohens_d(ref, ow)
        d_ci_lo, d_ci_hi = bootstrap(cohens_d, ref, ow)

        # AUCs (per-(premise, tier) aggregation, 53 pool)
        auc_ref_ow = auc(ref, ow)
        a1_lo, a1_hi = bootstrap(auc, ref, ow)
        perturbed = np.concatenate([os_, p_, ow])
        auc_ref_pert = auc(ref, perturbed)
        a2_lo, a2_hi = bootstrap(auc, ref, perturbed)

        out[m] = {
            "eta_squared_3level": {
                "value": eta3,
                "n_total": n_total_3,
                "levels": ["overstrong", "partial", "overweak"],
                "n_per_level": {t: len(non_ref_by_tier[m][t]) for t in ("overstrong","partial","overweak")},
            },
            "eta_squared_4level": {
                "value": eta4,
                "n_total": n_total_4,
                "levels": ["gold", "overstrong", "partial", "overweak"],
                "n_per_level": {t: len(all_by_tier[m][t]) for t in ("gold","overstrong","partial","overweak")},
            },
            "cohens_d_ref_vs_ow": {
                "value": d, "ci_lo": d_ci_lo, "ci_hi": d_ci_hi,
                "n_ref": int(len(ref)), "n_ow": int(len(ow)),
                "pool": "53-premise ρ-eligible (≥2 non-gold tiers)",
            },
            "auc_ref_vs_ow": {
                "value": auc_ref_ow, "ci_lo": a1_lo, "ci_hi": a1_hi,
                "n_ref": int(len(ref)), "n_ow": int(len(ow)),
                "pool": "53-premise ρ-eligible (≥2 non-gold tiers)",
            },
            "auc_ref_vs_perturbed": {
                "value": auc_ref_pert, "ci_lo": a2_lo, "ci_hi": a2_hi,
                "n_ref": int(len(ref)), "n_perturbed": int(len(perturbed)),
                "pool": "53-premise ρ-eligible (≥2 non-gold tiers)",
            },
        }

    return {
        "_primary_statistic": "eta_squared_3level",
        "_secondary_statistic": "cohens_d_ref_vs_ow",
        "_tertiary_statistic": "auc_ref_vs_ow + auc_ref_vs_perturbed (saturation-prone for self-identity metrics)",
        "_eligibility_filter_for_d_and_auc": "53 premises with ≥2 non-gold tiers",
        "_eta_squared_source": "All candidate rows in scored.json (n=267 perturbed; n=372 with REF)",
        "_bootstrap_resamples": n_resamples,
        "_seed": seed,
        "metrics": {m: out[m] for m in METRICS_FOR_ANALYSIS},
    }


def _compute_per_tier_means(rows: list) -> dict:
    """For each (tier, metric), compute mean + 95% bootstrap CI + n."""
    import numpy as np
    from collections import defaultdict

    grouped: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m in METRICS_FOR_ANALYSIS:
            v = r["scores"].get(m)
            if v is not None:
                grouped[r["tier"]][m].append(v)

    out: dict = {}
    rng = np.random.RandomState(42)
    for tier, metric_dict in grouped.items():
        out[tier] = {}
        for m in METRICS_FOR_ANALYSIS:
            vals = np.asarray(metric_dict.get(m, []), dtype=float)
            if len(vals) == 0:
                out[tier][m] = {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0}
                continue
            mean = float(vals.mean())
            boots = [vals[rng.randint(0, len(vals), size=len(vals))].mean()
                     for _ in range(1000)]
            ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
            out[tier][m] = {
                "mean": round(mean, 4),
                "ci_lo": round(float(ci_lo), 4),
                "ci_hi": round(float(ci_hi), 4),
                "n": len(vals),
            }
    return out


def _compute_monotonicity(per_tier: dict) -> dict:
    """For each metric, check whether per-tier means decrease across
    gold → overstrong → partial → overweak (lower = more wrong)."""
    tier_order = ["gold", "overstrong", "partial", "overweak"]
    out: dict = {}
    for m in METRICS_FOR_ANALYSIS:
        means = []
        for t in tier_order:
            v = per_tier.get(t, {}).get(m, {}).get("mean")
            means.append(v)
        out[m] = {
            "tier_order": tier_order,
            "means": means,
            "strictly_decreasing": _is_strictly_decreasing(means),
            "weakly_decreasing": _is_weakly_decreasing(means),
            "inversions": _find_inversions(tier_order, means),
        }
    return out


def _is_strictly_decreasing(seq: list) -> Optional[bool]:
    vals = [v for v in seq if v is not None]
    if len(vals) < 2:
        return None
    return all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))


def _is_weakly_decreasing(seq: list) -> Optional[bool]:
    vals = [v for v in seq if v is not None]
    if len(vals) < 2:
        return None
    return all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


def _find_inversions(labels: list, means: list) -> list:
    """Return [(label_a, label_b)] where mean_a < mean_b but a comes
    before b in the severity order (i.e., a should be ≥ b)."""
    out = []
    for i in range(len(labels) - 1):
        a, b = means[i], means[i + 1]
        if a is not None and b is not None and a < b:
            out.append([labels[i], labels[i + 1]])
    return out


def _compute_aggregate_rho(rows: list) -> dict:
    """Per-premise Spearman ρ across REF + OS + P + OW, averaged across
    premises with ≥2 distinct severity ranks present. REF (gold, rank 1)
    is included in the rank vector.

    Tertiary statistic. ρ punishes SIV's binary-leaning scoring through
    within-premise ties unrelated to metric quality (28% of SIV-soft-F1
    perturbed scores tie at exactly 1.0 with REF). η² + Cohen's d in
    _compute_global_separation are the primary statistics. ρ is retained
    as a descriptive statistic; see §4.3 of results.md for the discussion.
    """
    import numpy as np
    from scipy import stats as scipy_stats
    from collections import defaultdict

    # by_premise[pid][tier][metric] = score
    by_premise: dict = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        for m in METRICS_FOR_ANALYSIS:
            v = r["scores"].get(m)
            if v is not None:
                by_premise[r["premise_id"]][r["tier"]][m] = v

    rho_per_metric: dict = {m: [] for m in METRICS_FOR_ANALYSIS}
    n_used = 0
    for pid, by_tier in by_premise.items():
        present = [t for t in ("gold", "overstrong", "partial", "overweak") if t in by_tier]
        if len(present) < 2:
            continue
        gt_ranks = [SEVERITY_RANK[t] for t in present]
        # Need variation in GT ranks (REF=1, OS=P=2, OW=3; some pairs are tied at 2)
        if len(set(gt_ranks)) < 2:
            continue
        n_used += 1
        for m in METRICS_FOR_ANALYSIS:
            vals = [by_tier[t].get(m) for t in present]
            if any(v is None for v in vals):
                continue
            if len(set(vals)) < 2:
                rho_per_metric[m].append(0.0)
                continue
            rho, _ = scipy_stats.spearmanr(vals, [-r for r in gt_ranks])
            rho_per_metric[m].append(float(rho))

    rng = np.random.RandomState(42)
    out: dict = {}
    for m in METRICS_FOR_ANALYSIS:
        rhos = np.asarray(rho_per_metric[m], dtype=float)
        if len(rhos) == 0:
            out[m] = {"mean_rho": None, "ci_lo": None, "ci_hi": None, "n_premises": 0}
            continue
        mean_rho = float(rhos.mean())
        boots = [rhos[rng.randint(0, len(rhos), size=len(rhos))].mean()
                 for _ in range(1000)]
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
        out[m] = {
            "mean_rho": round(mean_rho, 4),
            "ci_lo": round(float(ci_lo), 4),
            "ci_hi": round(float(ci_hi), 4),
            "n_premises": len(rhos),
        }
    out["_n_premises_used_for_rho"] = n_used
    return out


def _compute_per_stratum(rows: list) -> dict:
    """Per-stratum decomposition: tier means and aggregate ρ within each stratum."""
    by_stratum: dict = {s: [] for s in range(1, 6)}
    for r in rows:
        by_stratum[r["stratum"]].append(r)
    out: dict = {}
    for s in range(1, 6):
        s_rows = by_stratum[s]
        out[str(s)] = {
            "n_rows": len(s_rows),
            "tier_means": _compute_per_tier_means(s_rows),
            "rank_correlation": _compute_aggregate_rho(s_rows),
        }
    return out


def _compute_per_operator_auc(rows: list) -> dict:
    """For each operator, AUC of (gold vs that operator's candidates) per metric."""
    import numpy as np
    from collections import defaultdict
    from experiments.common import auc_roc

    gold_scores: dict = defaultdict(list)
    op_scores: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m in METRICS_FOR_ANALYSIS:
            v = r["scores"].get(m)
            if v is None:
                continue
            if r["tier"] == "gold":
                gold_scores[m].append(v)
            else:
                op_scores[r["operator"]][m].append(v)

    out: dict = {}
    for op, by_metric in op_scores.items():
        out[op] = {}
        for m in METRICS_FOR_ANALYSIS:
            g = gold_scores.get(m, [])
            p = by_metric.get(m, [])
            if not g or not p:
                out[op][m] = None
                continue
            all_scores = np.array(g + p)
            labels = np.array([1] * len(g) + [0] * len(p))
            if len(set(all_scores.tolist())) < 2:
                out[op][m] = 0.5
                continue
            try:
                auc = float(auc_roc(all_scores, labels))
            except Exception:
                auc = None
            out[op][m] = round(auc, 4) if auc is not None else None
    return out


# ═══════════════════════════════════════════════════════════════════════════
# GIB floor — sanity check (separate from main analysis)
# ═══════════════════════════════════════════════════════════════════════════

def step_gib_floor(n_premise_pairs: int = 30, timeout: int = 5, seed: int = 42) -> None:
    """Compute the GIB metric floor: ~30 pairs of UNRELATED FOLIO premises
    scored as (gold, cand). Each pair shares no story so there's no logical
    relationship. All 6 metrics reported, not used in Spearman ρ.

    The seed selects which premises pair up; default 42.
    """
    from experiments.common import load_test_suites
    from siv.vampire_interface import setup_vampire

    setup_vampire()

    suites = load_test_suites(TEST_SUITES_PATH)
    rng = random.Random(seed)

    # Get premises with both canonical_fol and gold_fol
    pids = sorted([
        pid for pid, row in suites.items()
        if row.get("canonical_fol") and row.get("gold_fol")
    ])
    rng.shuffle(pids)

    pairs = []
    n_drawn = 0
    i = 0
    while n_drawn < n_premise_pairs and i + 1 < len(pids):
        pa, pb = pids[i], pids[i + 1]
        # Ensure different stories where possible
        sa = suites[pa].get("story_id")
        sb = suites[pb].get("story_id")
        if sa is not None and sb is not None and sa == sb:
            i += 1
            continue
        pairs.append((pa, pb))
        n_drawn += 1
        i += 2

    rows = []
    t0 = time.time()
    for pa, pb in pairs:
        gold_row = suites[pa]
        gold_fol = gold_row["gold_fol"]
        cand_fol = suites[pb]["canonical_fol"]
        scores = _score_single(gold_row, cand_fol, gold_fol, timeout)
        rows.append({
            "gold_premise_id": pa,
            "candidate_premise_id": pb,
            "gold_fol": gold_fol,
            "candidate_fol": cand_fol,
            "scores": scores,
        })

    # Per-metric means
    import numpy as np
    means: dict = {}
    for m in METRICS_FOR_ANALYSIS:
        vals = [r["scores"].get(m) for r in rows if r["scores"].get(m) is not None]
        if vals:
            arr = np.asarray(vals, dtype=float)
            means[m] = {
                "mean": round(float(arr.mean()), 4),
                "std": round(float(arr.std(ddof=1)), 4) if len(arr) > 1 else None,
                "min": round(float(arr.min()), 4),
                "max": round(float(arr.max()), 4),
                "n": len(arr),
            }
        else:
            means[m] = {"mean": None, "n": 0}

    output = {
        "n_pairs": len(rows),
        "seed": seed,
        "metric_means": means,
        "rows": rows,
    }
    GIB_FLOOR_PATH.write_text(json.dumps(output, indent=2))
    elapsed = time.time() - t0
    logger.info("GIB floor done. n=%d pairs, %.1fs", len(rows), elapsed)
    for m, stats in means.items():
        if stats.get("mean") is not None:
            logger.info("  %s mean=%.4f  (n=%d)", m, stats["mean"], stats["n"])

    _update_meta("gib_floor", {
        "n_pairs": len(rows),
        "seed": seed,
        "means": {m: stats.get("mean") for m, stats in means.items()},
    })


# ═══════════════════════════════════════════════════════════════════════════
# Plot — severity curve (mean metric score across tiers)
# ═══════════════════════════════════════════════════════════════════════════

# Tier order on x-axis: gold (reference) → overstrong → partial → overweak.
# OS and P share severity rank 2 but the spec puts OS before P, so we
# follow that order. OW is rank 3.
_TIER_ORDER = ["gold", "overstrong", "partial", "overweak"]
_TIER_LABEL = {
    "gold": "gold",
    "overstrong": "overstrong",
    "partial": "partial",
    "overweak": "overweak",
}

# Visual hero: SIV (thick, deep blue). Others muted.
_PLOT_STYLES = {
    "siv_soft_recall":         {"color": "#1f3a93", "linestyle": "-",  "linewidth": 2.5, "marker": "o", "markersize": 8},
    "siv_soft_f1":             {"color": "#1f3a93", "linestyle": "-",  "linewidth": 2.5, "marker": "o", "markersize": 8},
    "propositional_le_aligned": {"color": "#d35400", "linestyle": "-", "linewidth": 1.5, "marker": "^", "markersize": 7},
    "smatchpp":                {"color": "#16a085", "linestyle": "-",  "linewidth": 1.5, "marker": "D", "markersize": 6},
    "bleu":                    {"color": "#c0392b", "linestyle": "-",  "linewidth": 1.5, "marker": "v", "markersize": 6},
    "bertscore":               {"color": "#8e44ad", "linestyle": "-",  "linewidth": 1.5, "marker": "P", "markersize": 7},
}
_METRIC_DISPLAY = {
    "siv_soft_recall":         "SIV-soft (recall)",
    "siv_soft_f1":             "SIV",
    "propositional_le_aligned": "Propositional-LE",
    "smatchpp":                "Smatch++",
    "bleu":                    "BLEU",
    "bertscore":               "BERTScore",
}

# Metrics shown on the severity curve figure. Experiment 1 reports SIV as
# a single metric (F1); recall vs. F1 is only load-bearing in Experiment 2.
_PLOT_METRICS = [
    "siv_soft_f1",
    "propositional_le_aligned", "smatchpp", "bleu", "bertscore",
]


def step_plot_severity_curve() -> None:
    """Render the severity-curve line graph.

    Reads reports/experiments/severity_correlation/analysis/{per_tier_means,monotonicity}.json
    and writes severity_curve.{png,pdf} alongside. SIV (F1) is visually
    emphasized; metrics with inversions get a marker on the inverted point.
    Sized/styled to remain readable at 100% zoom in a single-column EMNLP
    paper (axes labels 14pt, ticks 12pt, legend 11pt; vector PDF).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not (ANALYSIS_DIR / "per_tier_means.json").exists():
        logger.error("Run step 7 first.")
        sys.exit(1)

    per_tier = json.loads((ANALYSIS_DIR / "per_tier_means.json").read_text())
    monotonic = json.loads((ANALYSIS_DIR / "monotonicity.json").read_text())

    fig, ax = plt.subplots(figsize=(7, 4))

    x_positions = list(range(len(_TIER_ORDER)))

    for m in _PLOT_METRICS:
        means, ci_los, ci_his = [], [], []
        for tier in _TIER_ORDER:
            row = per_tier.get(tier, {}).get(m, {})
            means.append(row.get("mean"))
            ci_los.append(row.get("ci_lo"))
            ci_his.append(row.get("ci_hi"))

        if any(v is None for v in means):
            continue

        style = _PLOT_STYLES[m]
        yerr_lo = [m_ - lo for m_, lo in zip(means, ci_los)]
        yerr_hi = [hi - m_ for m_, hi in zip(means, ci_his)]

        ax.plot(
            x_positions, means,
            label=_METRIC_DISPLAY[m],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=style["markersize"],
            markeredgecolor="white",
            markeredgewidth=0.5,
            zorder=3 if "siv_soft" in m else 2,
        )

        ax.errorbar(
            x_positions, means,
            yerr=[yerr_lo, yerr_hi],
            fmt="none",
            ecolor=style["color"],
            elinewidth=1.0,
            capsize=3,
            alpha=0.4,
            zorder=1,
        )

        # Highlight inversions: red X over the lower-of-the-pair point
        invs = monotonic.get(m, {}).get("inversions", [])
        for a, b in invs:
            if a in _TIER_ORDER and b in _TIER_ORDER:
                idx_b = _TIER_ORDER.index(b)
                ax.plot(
                    x_positions[idx_b], means[idx_b],
                    marker="x", markersize=14, color="red",
                    markeredgewidth=2.5, zorder=5,
                )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([_TIER_LABEL[t] for t in _TIER_ORDER], fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Mean metric score", fontsize=14)
    ax.set_xlabel("Severity tier (left = closer to gold)", fontsize=14)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax.legend(loc="lower left", fontsize=11, framealpha=0.95)

    # Annotations for the SIV line — the headline visual. The gold-tier label
    # (v=1.000) sits at the top of ylim; we let it extend just above the axis
    # spine (annotation_clip=False + savefig bbox_inches='tight' keeps it).
    siv_means = [per_tier.get(t, {}).get("siv_soft_f1", {}).get("mean")
                 for t in _TIER_ORDER]
    for i, v in enumerate(siv_means):
        if v is None:
            continue
        ax.annotate(
            f"{v:.3f}",
            xy=(i, v),
            xytext=(0, 6 if v >= 0.99 else 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            color=_PLOT_STYLES["siv_soft_f1"]["color"],
            fontweight="bold",
            annotation_clip=False,
        )

    plt.savefig(
        SEVERITY_PLOT_PATH_PNG, dpi=180, bbox_inches="tight", pad_inches=0.05
    )
    plt.savefig(
        SEVERITY_PLOT_PATH_PDF, bbox_inches="tight", pad_inches=0.05
    )
    plt.close()

    logger.info("Wrote %s", SEVERITY_PLOT_PATH_PNG)
    logger.info("Wrote %s", SEVERITY_PLOT_PATH_PDF)


# ═══════════════════════════════════════════════════════════════════════════
# Step 8 — Render results.md (paper-ready)
# ═══════════════════════════════════════════════════════════════════════════

RESULTS_PATH = DATA_DIR / "results.md"
SEVERITY_PLOT_PATH_PNG = DATA_DIR / "severity_curve.png"
SEVERITY_PLOT_PATH_PDF = DATA_DIR / "severity_curve.pdf"


def step8_render_results() -> None:
    """Render reports/experiments/severity_correlation/analysis/* into a
    paper-ready results.md alongside the pool + scoring artifacts."""
    if not SCORED_PATH.exists():
        logger.error("Run step 6 first.")
        sys.exit(1)
    if not (ANALYSIS_DIR / "per_tier_means.json").exists():
        logger.error("Run step 7 first.")
        sys.exit(1)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    scored = json.loads(SCORED_PATH.read_text())
    cand_data = json.loads(CANDIDATES_PATH.read_text())
    per_tier = json.loads((ANALYSIS_DIR / "per_tier_means.json").read_text())
    monotonic = json.loads((ANALYSIS_DIR / "monotonicity.json").read_text())
    rho = json.loads((ANALYSIS_DIR / "rank_correlation.json").read_text())
    per_stratum = json.loads((ANALYSIS_DIR / "per_stratum.json").read_text())
    per_op_auc = json.loads((ANALYSIS_DIR / "per_operator_auc.json").read_text())
    global_sep_path = ANALYSIS_DIR / "global_separation.json"
    global_sep = json.loads(global_sep_path.read_text()) if global_sep_path.exists() else None
    gib = json.loads(GIB_FLOOR_PATH.read_text()) if GIB_FLOOR_PATH.exists() else None

    lines = []
    lines.append("# severity_correlation — results\n")
    lines.append("Source of truth: [`configs/severity_correlation_v1.yaml`](../../../configs/severity_correlation_v1.yaml).\n")

    # ── Pool composition ────────────────────────────────────────────────────
    lines.append("\n## 1. Candidate pool\n")
    lines.append(f"Total candidates: **{cand_data['n_candidates']}** "
                 "(after Vampire verification + cell-target sampling)\n")
    lines.append("By tier: " + ", ".join(
        f"{t}={n}" for t, n in cand_data["by_tier"].items()) + "\n")
    lines.append("By stratum: " + ", ".join(
        f"S{s}={n}" for s, n in sorted(cand_data["by_stratum"].items())) + "\n")
    lines.append("\n### Per-(stratum × tier) breakdown\n")
    lines.append("| Stratum | Overstrong | Partial | Overweak | Total |")
    lines.append("|---|---:|---:|---:|---:|")
    from collections import Counter
    cells = Counter()
    for c in cand_data["candidates"]:
        cells[(c["stratum"], c["tier"])] += 1
    for s in range(1, 6):
        os_n = cells.get((s, "overstrong"), 0)
        p_n = cells.get((s, "partial"), 0)
        ow_n = cells.get((s, "overweak"), 0)
        lines.append(f"| S{s} | {os_n} | {p_n} | {ow_n} | {os_n + p_n + ow_n} |")

    # Verification stats
    step3_meta = meta.get("step3", {})
    if step3_meta:
        lines.append("")
        lines.append(
            f"Verification: {step3_meta.get('n_verified')} raw → "
            f"{step3_meta.get('n_kept')} retained "
            f"({100.0 * step3_meta.get('n_kept', 0) / max(step3_meta.get('n_verified', 1), 1):.1f}% retention)\n"
        )

    # ── Per-tier means (descriptive evidence under §4 η²) ───────────────────
    lines.append("\n## 2. Per-tier means (descriptive)\n")
    lines.append("Bootstrap 95% CI in parentheses. Descriptive evidence under §4 (η² + Cohen's d).\n")
    lines.append("| Tier | n | " + " | ".join(METRICS_FOR_ANALYSIS) + " |")
    lines.append("|---|---:|" + "|".join(["---:"] * len(METRICS_FOR_ANALYSIS)) + "|")
    for tier in ["gold", "overstrong", "partial", "overweak"]:
        row = per_tier.get(tier, {})
        n = max(
            (row.get(m, {}).get("n", 0) for m in METRICS_FOR_ANALYSIS),
            default=0,
        )
        cells_md = [f"**{tier}**", str(n)]
        for m in METRICS_FOR_ANALYSIS:
            s = row.get(m, {})
            if s.get("mean") is not None:
                cells_md.append(f"{s['mean']:.3f}")
            else:
                cells_md.append("—")
        lines.append("| " + " | ".join(cells_md) + " |")

    # ── Severity monotonicity ────────────────────────────────────────────────
    lines.append("\n## 3. Severity monotonicity check\n")
    lines.append(
        "Does mean(gold) > mean(OS) > mean(P) > mean(OW)? "
        "(weak = ≥, strict = >). Inversions listed where present.\n"
    )
    lines.append("| Metric | Strict ↓ | Weak ↓ | Inversions |")
    lines.append("|---|:---:|:---:|---|")
    for m in METRICS_FOR_ANALYSIS:
        d = monotonic.get(m, {})
        strict = d.get("strictly_decreasing")
        weak = d.get("weakly_decreasing")
        invs = d.get("inversions", [])
        invs_str = "; ".join(f"{a}<{b}" for a, b in invs) if invs else "—"
        strict_md = "✓" if strict else ("✗" if strict is False else "—")
        weak_md = "✓" if weak else ("✗" if weak is False else "—")
        lines.append(f"| {m} | {strict_md} | {weak_md} | {invs_str} |")

    # ── PRIMARY: η² + Cohen's d ────────────────────────────────────────────
    if global_sep is not None:
        lines.append("\n## 4. Magnitude of severity separation (primary)\n")
        lines.append(
            "The headline statistic is η² (one-way ANOVA, tier as factor) "
            "— the fraction of metric-score variance explained by severity "
            "tier — supplemented by Cohen's d on the REF-vs-OW contrast. "
            "η² and d reward absolute-magnitude calibration; rank-based "
            "statistics (AUC, ρ) are reported as secondary / tertiary "
            "below.\n"
        )
        lines.append("### 4.1 η² — variance explained by tier\n")
        lines.append("| Metric | η² (3-level, OS/P/OW) | η² (4-level, +REF) | n total (3-level) | n total (4-level) |")
        lines.append("|---|---:|---:|---:|---:|")
        for m in METRICS_FOR_ANALYSIS:
            cell = global_sep["metrics"][m]
            e3 = cell["eta_squared_3level"]
            e4 = cell["eta_squared_4level"]
            e3v = f"{e3['value']:.3f}" if e3['value'] is not None else "—"
            e4v = f"{e4['value']:.3f}" if e4['value'] is not None else "—"
            lines.append(f"| {m} | {e3v} | {e4v} | {e3['n_total']} | {e4['n_total']} |")
        lines.append("\nη² is the fraction of total score variance explained "
                     "by tier membership. Higher = severity tier is more "
                     "predictive of the score (better calibration).\n")

        lines.append("### 4.2 Cohen's d — REF-vs-OW effect size\n")
        lines.append("Bootstrap 95% CI over 1,000 resamples of each tier "
                     "(stratified). Aggregation matches the official ρ "
                     "extraction (one score per (premise, tier) cell), "
                     "restricted to the 53-premise ρ-eligible pool for "
                     "comparability.\n")
        lines.append("| Metric | Cohen's d | 95% CI | n_REF | n_OW |")
        lines.append("|---|---:|---|---:|---:|")
        for m in METRICS_FOR_ANALYSIS:
            cell = global_sep["metrics"][m]["cohens_d_ref_vs_ow"]
            v = cell["value"]
            if v is None:
                lines.append(f"| {m} | — | — | {cell['n_ref']} | {cell['n_ow']} |")
            else:
                lines.append(
                    f"| {m} | {v:+.3f} | [{cell['ci_lo']:+.3f}, "
                    f"{cell['ci_hi']:+.3f}] | {cell['n_ref']} | {cell['n_ow']} |"
                )

        lines.append("\n### 4.3 Global AUC — REF-vs-OW and REF-vs-perturbed (secondary)\n")
        lines.append(
            "AUC measures rank-separation; it saturates at 1.000 for any "
            "metric that returns its maximum score on REF by self-identity "
            "and strictly less on every perturbed candidate. Reported as "
            "secondary because (a) Smatch++ and SIV-soft-{recall,F1} all "
            "saturate at exactly 1.0 on REF by self-identity, making the "
            "AUC = 1.000 result largely a 'detected any change' statement, "
            "and (b) within-pool tie patterns differ across metrics in "
            "ways unrelated to severity calibration. See §4.1 (η²) and "
            "§4.2 (Cohen's d) for the substantive comparison.\n"
        )
        lines.append("| Metric | AUC REF-vs-OW | 95% CI | AUC REF-vs-perturbed | 95% CI |")
        lines.append("|---|---:|---|---:|---|")
        for m in METRICS_FOR_ANALYSIS:
            cell = global_sep["metrics"][m]
            a1 = cell["auc_ref_vs_ow"]
            a2 = cell["auc_ref_vs_perturbed"]
            a1v = f"{a1['value']:.3f}" if a1['value'] is not None else "—"
            a1ci = f"[{a1['ci_lo']:.3f}, {a1['ci_hi']:.3f}]" if a1['ci_lo'] is not None else "—"
            a2v = f"{a2['value']:.3f}" if a2['value'] is not None else "—"
            a2ci = f"[{a2['ci_lo']:.3f}, {a2['ci_hi']:.3f}]" if a2['ci_lo'] is not None else "—"
            lines.append(f"| {m} | {a1v} | {a1ci} | {a2v} | {a2ci} |")

    # ── Per-stratum decomposition ───────────────────────────────────────────
    lines.append("\n## 5. Per-stratum decomposition (secondary)\n")
    for s in range(1, 6):
        block = per_stratum.get(str(s), {})
        lines.append(f"\n### Stratum {s} (n={block.get('n_rows', 0)} rows)\n")
        # Per-tier means within stratum
        st_per_tier = block.get("tier_means", {})
        lines.append("| Tier | " + " | ".join(METRICS_FOR_ANALYSIS) + " |")
        lines.append("|---" + "|---:" * len(METRICS_FOR_ANALYSIS) + "|")
        for tier in ["gold", "overstrong", "partial", "overweak"]:
            r = st_per_tier.get(tier, {})
            cells_md = [tier]
            for m in METRICS_FOR_ANALYSIS:
                s_v = r.get(m, {})
                if s_v.get("mean") is not None:
                    cells_md.append(f"{s_v['mean']:.3f}")
                else:
                    cells_md.append("—")
            lines.append("| " + " | ".join(cells_md) + " |")
        # Within-stratum aggregate ρ
        st_rho = block.get("rank_correlation", {})
        n_pre = st_rho.get("_n_premises_used_for_rho", 0)
        lines.append(f"\nWithin-stratum ρ (n={n_pre} qualifying premises):  ")
        for m in METRICS_FOR_ANALYSIS:
            d = st_rho.get(m, {})
            mr = d.get("mean_rho")
            if mr is not None:
                lines.append(f"  - {m}: {mr:.4f}")

    # ── Per-operator AUC ────────────────────────────────────────────────────
    lines.append("\n## 6. Per-operator AUC (secondary)\n")
    lines.append(
        "AUC(gold vs operator's candidates) per metric. AUC=1.0 means "
        "the metric perfectly distinguishes gold from this operator's "
        "candidates; AUC=0.5 is chance.\n"
    )
    operators = sorted(per_op_auc.keys())
    lines.append("| Operator | " + " | ".join(METRICS_FOR_ANALYSIS) + " |")
    lines.append("|---" + "|---:" * len(METRICS_FOR_ANALYSIS) + "|")
    for op in operators:
        row = per_op_auc[op]
        cells_md = [op]
        for m in METRICS_FOR_ANALYSIS:
            v = row.get(m)
            cells_md.append(f"{v:.3f}" if v is not None else "—")
        lines.append("| " + " | ".join(cells_md) + " |")

    # ── TERTIARY: aggregate Spearman ρ ──────────────────────────────────────
    lines.append("\n## 7. Aggregate Spearman ρ on REF-OS-P-OW (tertiary)\n")
    lines.append(
        "ρ is retained as a descriptive statistic but is not the headline. "
        "Three reasons (see also §4.3 below): (a) REF anchoring at 1.0 "
        "saturates ρ for any continuous-scored metric; (b) OS and P tie "
        "at rank 2, so ρ does not validate the OS-vs-P ordering — that "
        "is the per-tier means table's job in §2; (c) SIV's "
        "binary-leaning scoring produces within-premise ties (28% of "
        "perturbed candidates tie at exactly 1.0 with REF for "
        "SIV-soft-F1) that Spearman penalises as ρ = 0, dragging "
        "SIV's aggregate ρ down for reasons unrelated to metric "
        "quality. η² and Cohen's d (§4) are the substantive comparison.\n"
    )
    n_used = rho.get("_n_premises_used_for_rho", 0)
    lines.append(f"Per-premise Spearman ρ over the four severity tiers "
                 f"(rank vector: REF=1, OS=P=2, OW=3), averaged across "
                 f"premises with ≥2 distinct ranks present (n={n_used} "
                 f"qualifying premises).\n")
    lines.append("| Metric | mean ρ | 95% CI | n premises |")
    lines.append("|---|---:|---|---:|")
    for m in METRICS_FOR_ANALYSIS:
        d = rho.get(m, {})
        mr = d.get("mean_rho")
        if mr is None:
            lines.append(f"| {m} | — | — | 0 |")
        else:
            lines.append(
                f"| {m} | {mr:.4f} | [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}] | "
                f"{d['n_premises']} |"
            )

    # ── GIB floor sanity ────────────────────────────────────────────────────
    if gib is not None:
        lines.append("\n## 8. GIB floor (sanity check, not in main analysis)\n")
        lines.append(
            f"{gib['n_pairs']} pairs of unrelated FOLIO premises scored as "
            f"(gold, cand). Should saturate near the metric's floor.\n"
        )
        lines.append("| Metric | mean | std | min | max | n |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for m in METRICS_FOR_ANALYSIS:
            s = gib["metric_means"].get(m, {})
            mean = s.get("mean")
            if mean is None:
                lines.append(f"| {m} | — | — | — | — | 0 |")
            else:
                std = f"{s.get('std', 0):.3f}" if s.get("std") is not None else "—"
                lines.append(
                    f"| {m} | {mean:.3f} | {std} | "
                    f"{s.get('min', 0):.3f} | {s.get('max', 0):.3f} | "
                    f"{s.get('n', 0)} |"
                )

    # ── Findings / honest disclosure ────────────────────────────────────────
    lines.append("\n## 9. Findings to disclose\n")
    lines.append(_render_honest_disclosure(meta, cand_data))

    RESULTS_PATH.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", RESULTS_PATH)


def _render_honest_disclosure(meta: dict, cand_data: dict) -> str:
    """Surface findings the spec asks to disclose explicitly."""
    lines = []

    # FOLIO availability per stratum
    step1 = meta.get("step1", {})
    if step1:
        folio = step1.get("folio_totals_by_stratum", {})
        sampled = step1.get("sampled_by_stratum", {})
        lines.append("\n**Stratum availability in FOLIO (FOLIO-vs-synthetic source ratio):**")
        for s in range(1, 6):
            f_n = folio.get(str(s), 0)
            sa_n = sampled.get(str(s), 0)
            lines.append(
                f"- S{s}: {sa_n} sampled of {f_n} FOLIO golds available "
                f"(synthetic augmentation: 0)"
            )
        # Highlight under-represented strata
        underrep = [str(s) for s in range(1, 6) if folio.get(str(s), 0) < 30]
        if underrep:
            lines.append(
                f"\nUnder-represented strata in FOLIO (< 30 golds): "
                f"{', '.join('S' + s for s in underrep)}. "
                "The design brief allows templated synthesis to fill; not "
                "yet implemented in v1."
            )

    # Empty/low-yield cells
    audit = cand_data.get("cell_audit", [])
    zero_target_cells = [c for c in audit
                         if isinstance(c.get("target"), int) and c.get("kept", 0) == 0]
    if zero_target_cells:
        lines.append(
            "\n**Cells with target=8 but 0 candidates retained** "
            "(applicability + Vampire verification dropped all candidates):"
        )
        for c in zero_target_cells:
            lines.append(f"- {c['operator']} × S{c['stratum']}: "
                         f"{c['available_after_verification']} available, 0 retained")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--step", required=True,
        choices=["1", "2", "3", "4", "5", "6", "7", "8", "gib", "plot", "deterministic", "all"],
        help=(
            "Which step. 'deterministic' = steps 1-4 (no LLM). "
            "'all' = steps 1-8 + gib floor."
        ),
    )
    parser.add_argument("--max-per-stratum", type=int, default=50)
    parser.add_argument("--vampire-timeout", type=int, default=10)
    parser.add_argument("--score-timeout", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.step in ("1", "deterministic", "all"):
        step1_load_and_stratify(max_per_stratum=args.max_per_stratum, seed=args.seed)
    if args.step in ("2", "deterministic", "all"):
        step2_generate_candidates()
    if args.step in ("3", "deterministic", "all"):
        step3_verify(timeout=args.vampire_timeout)
    if args.step in ("4", "deterministic", "all"):
        step4_sample(seed=args.seed)
    if args.step in ("5", "all"):
        step5_verbalize_nl()
    if args.step in ("6", "all"):
        step6_score(timeout=args.score_timeout)
    if args.step in ("7", "all"):
        step7_analyze()
    if args.step in ("gib", "all"):
        step_gib_floor(timeout=args.score_timeout, seed=args.seed)
    if args.step in ("plot", "all"):
        step_plot_severity_curve()
    if args.step in ("8", "all"):
        step8_render_results()


if __name__ == "__main__":
    main()
