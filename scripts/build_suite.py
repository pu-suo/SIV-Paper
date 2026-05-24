"""Single-shot demo: gold FOL → SIV test suite.

Usage:
    python scripts/build_suite.py --gold-fol "∀x (Dog(x) → Mammal(x))"
    python scripts/build_suite.py --gold-fol "..." --nl "All dogs are mammals." --json
    python scripts/build_suite.py --gold-file path/to/gold.txt
    cat gold.txt | python scripts/build_suite.py --stdin

Pretty-prints the parsed extraction, canonical FOL, positives, and
contrastives Vampire produced from the supplied gold annotation.
Pair with `scripts/score_one.py` for the matching scoring CLI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from siv.compiler import compile_canonical_fol
from siv.suite_generator import generate_test_suite


def _read_gold(args: argparse.Namespace) -> str:
    if args.gold_fol is not None:
        return args.gold_fol
    if args.gold_file is not None:
        return Path(args.gold_file).read_text().strip()
    if args.stdin:
        return sys.stdin.read().strip()
    raise SystemExit(
        "error: provide gold FOL via --gold-fol, --gold-file, or --stdin"
    )


def _print_text(result, gold_fol: str, nl: str) -> None:
    suite = result.suite
    print("=" * 72)
    print("SIV test suite from gold FOL")
    print("=" * 72)
    if nl:
        print(f"NL:           {nl}")
    print(f"Gold FOL:     {gold_fol}")
    print(f"Round-trip:   {'verified' if result.round_trip_verified else 'skipped/failed'}")
    print()

    canonical = compile_canonical_fol(suite.extraction)
    print(f"Canonical FOL (compiler output):\n  {canonical}\n")

    print(f"Positives ({len(suite.positives)}):")
    if not suite.positives:
        print("  (none)")
    for i, t in enumerate(suite.positives, 1):
        print(f"  [{i:02d}] {t.fol}")
    print()

    print(f"Contrastives ({len(suite.contrastives)}):")
    if not suite.contrastives:
        print("  (none — structurally weak source)")
    for i, t in enumerate(suite.contrastives, 1):
        if t.mutation_kind:
            relation = t.probe_relation or "incompatible"
            tag = f"  ({t.mutation_kind}, {relation})"
        else:
            tag = ""
        print(f"  [{i:02d}]{tag} {t.fol}")


def _print_json(result, gold_fol: str, nl: str) -> None:
    suite = result.suite
    canonical = compile_canonical_fol(suite.extraction)
    out = {
        "nl": nl,
        "gold_fol": gold_fol,
        "round_trip_verified": result.round_trip_verified,
        "canonical_fol": canonical,
        "positives": [t.model_dump() for t in suite.positives],
        "contrastives": [t.model_dump() for t in suite.contrastives],
        "extraction": suite.extraction.model_dump(),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--gold-fol", type=str,
                     help="Gold FOL string (FOLIO syntax).")
    src.add_argument("--gold-file", type=str,
                     help="Path to a file containing the gold FOL string.")
    src.add_argument("--stdin", action="store_true",
                     help="Read the gold FOL string from stdin.")
    ap.add_argument("--nl", type=str, default="",
                    help="Optional NL sentence to attach to the extraction.")
    ap.add_argument("--no-contrastives", action="store_true",
                    help="Skip contrastive generation (positives only).")
    ap.add_argument("--no-round-trip", action="store_true",
                    help="Skip the Vampire round-trip equivalence check.")
    ap.add_argument("--timeout-s", type=int, default=5,
                    help="Vampire timeout per check (default 5s).")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON instead of pretty text.")
    args = ap.parse_args()

    gold_fol = _read_gold(args)

    result = generate_test_suite(
        fol_string=gold_fol,
        nl=args.nl,
        verify_round_trip=not args.no_round_trip,
        with_contrastives=not args.no_contrastives,
        timeout_s=args.timeout_s,
    )

    if result.suite is None:
        msg = {"error": result.error, "gold_fol": gold_fol}
        if args.json:
            print(json.dumps(msg, indent=2, ensure_ascii=False))
        else:
            print(f"FAILED: {result.error}", file=sys.stderr)
        return 1

    if args.json:
        _print_json(result, gold_fol, args.nl)
    else:
        _print_text(result, gold_fol, args.nl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
