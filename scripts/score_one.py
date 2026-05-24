"""Single-shot demo: gold FOL + candidate FOL → SIV score and per-test verdicts.

Usage:
    python scripts/score_one.py \
      --gold-fol      "∀x (Dog(x) → Mammal(x))" \
      --candidate-fol "∀x (Dog(x) → Mammal(x))"

    python scripts/score_one.py \
      --gold-file gold.txt --candidate-file cand.txt --json

Builds the test suite from the gold annotation (parser → compiler →
contrastive generator), runs the candidate against every positive and
contrastive via Vampire, and prints recall / precision / F1 along with
which individual tests passed or failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from siv.fol_utils import is_valid_fol, normalize_fol_string
from siv.suite_generator import generate_test_suite
from siv.scorer import score


def _read_arg(direct: str | None, file_path: str | None, label: str) -> str:
    if direct is not None:
        return direct
    if file_path is not None:
        return Path(file_path).read_text().strip()
    raise SystemExit(f"error: provide {label} via --{label}-fol or --{label}-file")


def _kind_pass(kind: str, verdict: str) -> bool:
    if kind == "positive":
        return verdict == "entailed"
    if kind == "contrastive":
        if verdict == "no_contrastives":
            return True  # informational only
        return verdict != "entailed"
    return False


def _format_label(probe_label) -> str:
    if probe_label is None:
        return "[—]"
    probe_kind, feature_target = probe_label
    if feature_target is None:
        return f"[{probe_kind}]"
    return f"[{probe_kind}:{feature_target}]"


def _print_text(report, gold_fol: str, candidate_fol: str, nl: str) -> None:
    print("=" * 72)
    print("SIV score")
    print("=" * 72)
    if nl:
        print(f"NL:        {nl}")
    print(f"Gold:      {gold_fol}")
    print(f"Candidate: {candidate_fol}")
    print()

    p = "—" if report.precision is None else f"{report.precision:.3f}"
    f = "—" if report.f1 is None else f"{report.f1:.3f}"
    print(f"recall    = {report.recall:.3f}    "
          f"({report.positives_entailed}/{report.positives_total} positives entailed)")
    print(f"precision = {p}    "
          f"({report.contrastives_rejected}/{report.contrastives_total} contrastives rejected)")
    print(f"F1        = {f}")
    if report.contrastives_total == 0:
        print("  note: no contrastives generated (recall-only regime).")
    print()

    failed_pos = [
        (i, r) for i, r in enumerate(report.per_test_results, 1)
        if r.kind == "positive" and not _kind_pass(r.kind, r.verdict)
    ]
    failed_con = [
        (i, r) for i, r in enumerate(report.per_test_results, 1)
        if r.kind == "contrastive" and not _kind_pass(r.kind, r.verdict)
    ]

    print("Per-test results:")
    for i, r in enumerate(report.per_test_results, 1):
        if r.kind == "contrastive" and r.verdict == "no_contrastives":
            print(f"  [{i:02d}] contrastive   (no contrastives produced)")
            continue
        ok = "PASS" if _kind_pass(r.kind, r.verdict) else "FAIL"
        label_str = _format_label(r.probe_label)
        print(f"  [{i:02d}] {ok:<4} {r.kind:<12} verdict={r.verdict:<14} "
              f"{label_str:<28} {r.fol}")

    print()
    print(f"Failed positives:    {len(failed_pos)}")
    if failed_pos:
        for _, r in failed_pos:
            print(f"  - {_format_label(r.probe_label)} ({r.verdict}) {r.fol}")
    print(f"Failed contrastives: {len(failed_con)}  "
          f"(candidate wrongly entails these — over-strong)")
    if failed_con:
        for _, r in failed_con:
            print(f"  - {_format_label(r.probe_label)} ({r.verdict}) {r.fol}")


def _print_json(report, gold_fol: str, candidate_fol: str, nl: str) -> None:
    out = {
        "nl": nl,
        "gold_fol": gold_fol,
        "candidate_fol": candidate_fol,
        "recall": report.recall,
        "precision": report.precision,
        "f1": report.f1,
        "positives_entailed": report.positives_entailed,
        "positives_total": report.positives_total,
        "contrastives_rejected": report.contrastives_rejected,
        "contrastives_total": report.contrastives_total,
        "per_test_results": [
            {
                "index": i,
                "kind": r.kind,
                "fol": r.fol,
                "verdict": r.verdict,
                "passed": _kind_pass(r.kind, r.verdict),
                "probe_label": list(r.probe_label) if r.probe_label else None,
                "mutation_kind": r.mutation_kind,
            }
            for i, r in enumerate(report.per_test_results, 1)
        ],
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold-fol", type=str)
    ap.add_argument("--gold-file", type=str)
    ap.add_argument("--candidate-fol", type=str)
    ap.add_argument("--candidate-file", type=str)
    ap.add_argument("--nl", type=str, default="")
    ap.add_argument("--no-round-trip", action="store_true",
                    help="Skip the Vampire round-trip equivalence check on the gold.")
    ap.add_argument("--timeout-s", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    gold_fol = _read_arg(args.gold_fol, args.gold_file, "gold")
    candidate_fol = _read_arg(args.candidate_fol, args.candidate_file, "candidate")

    cand_norm = normalize_fol_string(candidate_fol)
    if not is_valid_fol(cand_norm):
        msg = {"error": "candidate_parse_error", "candidate_fol": candidate_fol}
        if args.json:
            print(json.dumps(msg, indent=2, ensure_ascii=False))
        else:
            print(f"FAILED: candidate is not valid FOL: {candidate_fol}",
                  file=sys.stderr)
        return 2

    suite_result = generate_test_suite(
        fol_string=gold_fol,
        nl=args.nl,
        verify_round_trip=not args.no_round_trip,
        with_contrastives=True,
        timeout_s=args.timeout_s,
    )
    if suite_result.suite is None:
        msg = {"error": suite_result.error, "gold_fol": gold_fol}
        if args.json:
            print(json.dumps(msg, indent=2, ensure_ascii=False))
        else:
            print(f"FAILED to build suite from gold: {suite_result.error}",
                  file=sys.stderr)
        return 1

    report = score(suite_result.suite, cand_norm, timeout_s=args.timeout_s)

    if args.json:
        _print_json(report, gold_fol, candidate_fol, args.nl)
    else:
        _print_text(report, gold_fol, candidate_fol, args.nl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
