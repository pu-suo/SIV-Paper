#!/usr/bin/env python3
"""Select 30 premises with surprising SIV scoring on the FOLIO reference.

These are candidates for hand-inspection of the reference annotation
itself (Appendix D — reference correction sensitivity).

Surprise signals (in priority order):
  1. Gold SIV-soft recall < 1.0  — the gold formula does NOT entail its
     own positive sub-entailment probes. Either the probes are
     malformed, the canonicalisation is wrong, OR the FOLIO reference
     itself encodes the wrong proposition.
  2. Gold SIV-strict recall < 1.0 — under no-alignment scoring; weaker
     evidence of an issue (alignment usually rescues these).
  3. Predicate-Jaccard between SIV-canonical and FOLIO gold < 0.8 — the
     two formulas use different vocabularies, suggesting the reference
     uses unexpected predicate names or extra/missing predicates.

We pull from premises that PASSED the Exp1 aligned-subset filter (368
total — these are the ones actually contributing to detection AUC and
to the rank-correlation pool used for ρ via Exp2). Among those, we
rank by surprise and take the top 30.

Output:
  reports/experiments/perturbation_detection/correction_candidates.md  (for hand inspection)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
EXP1_DIR = _REPO_ROOT / "reports" / "experiments" / "perturbation_detection"
TEST_SUITES_PATH = _REPO_ROOT / "test_suites" / "test_suites.jsonl"


def main():
    manifest = {}
    for line in (EXP1_DIR / "aligned_subset_manifest.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            manifest[r["premise_id"]] = r
    passing = {pid: r for pid, r in manifest.items() if r.get("passes")}

    gold_rows = {}
    for line in (EXP1_DIR / "scored_candidates.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["candidate_type"] == "gold":
                gold_rows[r["premise_id"]] = r

    suites = {}
    for line in TEST_SUITES_PATH.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            suites[r["premise_id"]] = r

    # Score each passing premise
    candidates = []
    for pid, mrow in passing.items():
        gr = gold_rows.get(pid)
        if gr is None:
            continue
        suite = suites.get(pid, {})
        s = gr["scores"]
        soft_recall = s.get("siv_soft_recall")
        strict_recall = s.get("siv_strict_recall")
        soft_precision = s.get("siv_soft_precision")
        soft_f1 = s.get("siv_soft_f1")
        jaccard = mrow.get("criteria", {}).get("jaccard")

        reasons = []
        score = 0.0
        if soft_recall is not None and soft_recall < 1.0:
            reasons.append(f"gold SIV-soft recall = {soft_recall:.2f}")
            score += 100 * (1.0 - soft_recall)
        if strict_recall is not None and strict_recall < 1.0:
            reasons.append(f"gold SIV-strict recall = {strict_recall:.2f}")
            score += 10 * (1.0 - strict_recall)
        if jaccard is not None and jaccard < 0.8:
            reasons.append(f"predicate Jaccard = {jaccard:.2f}")
            score += 5 * (1.0 - jaccard)
        if soft_f1 is not None and soft_f1 < 1.0:
            reasons.append(f"gold SIV-soft F1 = {soft_f1:.2f}")
            score += 1 * (1.0 - soft_f1)

        if not reasons:
            continue

        candidates.append({
            "premise_id": pid,
            "score": score,
            "reasons": reasons,
            "nl": suite.get("nl") or "(no NL stored)",
            "folio_gold": mrow.get("gold_fol", ""),
            "siv_canonical": suite.get("canonical_fol") or mrow.get("canonical_fol", ""),
            "siv_soft_recall": soft_recall,
            "siv_soft_precision": soft_precision,
            "siv_soft_f1": soft_f1,
            "siv_strict_recall": strict_recall,
            "jaccard": jaccard,
            "n_positives": len(suite.get("positives") or []),
            "n_contrastives": len(suite.get("contrastives") or []),
        })

    candidates.sort(key=lambda c: -c["score"])
    selected = candidates[:30]

    print(f"Total surprising premises (any signal):  {len(candidates)}")
    print(f"Selected for hand inspection:             {len(selected)}")

    # ── Markdown output ──
    lines = []
    lines.append("# Reference correction candidates — Appendix D (Exp 1)\n")
    lines.append(
        "30 premises from the Exp 1 aligned subset (n=368) that show "
        "surprising SIV scoring on the FOLIO reference itself. The most "
        "common signal is `gold SIV-soft recall < 1.0` — the gold "
        "formula fails to entail one or more of its own positive "
        "sub-entailment probes, which usually means either the FOLIO "
        "reference is mistranscribed or the SIV canonicalisation drifts "
        "from the intended semantics.\n"
    )
    lines.append("**Hand-inspection task**: for each row, decide whether "
                 "the FOLIO reference (column `FOLIO gold`) is wrong. If "
                 "yes, write a corrected formula in the `corrected_reference` "
                 "field of [reference_corrections.jsonl](reference_corrections.jsonl) "
                 "with one of these `correction_reason` values: "
                 "`mistranscribed_predicate`, `missing_quantifier`, "
                 "`malformed_nesting`, `wrong_connective`, "
                 "`scope_misplacement`, `other`.\n")
    lines.append("If the reference is correct and the surprise is "
                 "explained by a SIV-side issue (probe generation bug, "
                 "canonicalisation drift), leave that row out of "
                 "`reference_corrections.jsonl`. The recompute script "
                 "treats absent rows as 'no correction needed'.\n")

    lines.append(
        f"Selected {len(selected)} of {len(candidates)} surprising "
        f"premises in the aligned subset (n=368). Full ranked list "
        f"below; please fill in corrections for any premise where the "
        f"FOLIO reference is demonstrably wrong.\n"
    )

    lines.append(
        "| # | premise_id | gold soft recall | gold soft F1 | gold strict recall | Jaccard | n_pos | n_con | NL premise | FOLIO gold | SIV canonical | surprise reasons |"
    )
    lines.append(
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|---|"
    )
    for i, c in enumerate(selected, 1):
        soft_r = "—" if c["siv_soft_recall"] is None else f"{c['siv_soft_recall']:.2f}"
        soft_f1 = "—" if c["siv_soft_f1"] is None else f"{c['siv_soft_f1']:.2f}"
        strict_r = "—" if c["siv_strict_recall"] is None else f"{c['siv_strict_recall']:.2f}"
        jaccard = "—" if c["jaccard"] is None else f"{c['jaccard']:.2f}"
        nl = (c["nl"] or "").replace("|", "\\|").replace("\n", " ")
        gold = (c["folio_gold"] or "").replace("|", "\\|")
        siv_can = (c["siv_canonical"] or "").replace("|", "\\|")
        reasons = "; ".join(c["reasons"]).replace("|", "\\|")
        lines.append(
            f"| {i} | {c['premise_id']} | {soft_r} | {soft_f1} | "
            f"{strict_r} | {jaccard} | {c['n_positives']} | "
            f"{c['n_contrastives']} | {nl} | `{gold}` | `{siv_can}` | "
            f"{reasons} |"
        )

    lines.append("")
    lines.append("---\n")
    lines.append("## How corrections feed back into the analysis\n")
    lines.append(
        "After hand inspection, fill in "
        "`reports/experiments/perturbation_detection/reference_corrections.jsonl` "
        "(one JSON row per corrected premise, schema: "
        "`{premise_id, original_reference, corrected_reference, correction_reason}`). "
        "The recompute step (Block 11 follow-up) then loads the "
        "corrections, regenerates the affected gold positive probes via "
        "`siv.suite_generator.generate_test_suite`, "
        "re-scores the affected candidates in the Exp 2 ρ-pool, and "
        "produces:\n"
    )
    lines.append(
        "- `correction_sensitivity.json` with three Spearman ρ values: "
        "(a) on the corrected subset of 30 with corrected references, "
        "(b) on the full corpus with corrections applied where they "
        "exist, (c) on the full corpus uncorrected (the headline number "
        "for comparison). Each ρ comes with its 95% bootstrap CI.\n"
    )
    lines.append(
        "Expected behaviour: headline ρ moves by less than ±0.02. A "
        "larger move would be flagged as a paper-worthy finding.\n"
    )

    out_path = EXP1_DIR / "correction_candidates.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")

    # Also write a JSON sidecar so the recompute script doesn't have to
    # reparse the markdown.
    json_path = EXP1_DIR / "correction_candidates.json"
    json_path.write_text(json.dumps(selected, indent=2) + "\n")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
