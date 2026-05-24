# SIV — Structural Inference Verification

SIV is a graded, per-aspect diagnostic metric for natural-language to
First-Order-Logic translation faithfulness. It scores a candidate FOL
translation against a *test suite* of positive sub-entailments and
contrastive mutants, each verified by the Vampire theorem prover.

The test suites are derived **deterministically** from FOLIO gold FOL
annotations. The pipeline is LLM-free at every stage from
gold-FOL-in to test-suite-out.

```
FOLIO gold FOL ──► parse_gold_fol ──► SentenceExtraction
                                         │
                          ┌──────────────┴──────────────┐
                          ▼                             ▼
                       Compiler                Contrastive generator
                  (positive probes)         (6 ops, Vampire-filtered)
                          └──────────────┬──────────────┘
                                         ▼
                                     TestSuite
                                         │
                       candidate FOL ────┤
                                         ▼
                                      Scorer
                                  (recall, precision, F1)
```

## Repository layout

```
siv/                       core library (deterministic; no LLM call)
test_suites/               canonical FOLIO probe cache (shared input)
scripts/                   demo CLIs (build_suite, score_one, ...)
scripts/experiments/       experiment runners + shared scoring helpers
reports/experiments/       three paper experiments' outputs
tests/                     pytest unit / soundness tests
configs/                   experiment specs (YAML)
```

## The three paper experiments

Each experiment bundles its entire output set (pool, scoring,
analysis, results doc, figures) under `reports/experiments/<name>/`.

| Experiment | Question | Outputs |
| --- | --- | --- |
| **severity_correlation** (Exp 1) | Does the score correlate with structural error severity (η² across REF / OS / P / OW)? | [reports/experiments/severity_correlation/](reports/experiments/severity_correlation/) |
| **perturbation_detection** (Exp 2) | Does the score reliably drop on each structural perturbation class (per-class drop magnitude + within-pair detection rate)? | [reports/experiments/perturbation_detection/](reports/experiments/perturbation_detection/) |
| **diagnostic_trace** (Exp 3) | Can the per-probe failure trace identify which structural feature failed (perturbation-class recoverability macro-F1)? | [reports/experiments/diagnostic_trace/](reports/experiments/diagnostic_trace/) |

## Setup

```bash
bash scripts/setup.sh
# or:
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "from siv.vampire_interface import setup_vampire; setup_vampire('.')"
```

The Vampire 5.0.1 prover binary must live at `./vampire`. The
`setup_vampire` helper downloads the appropriate platform build from
the official Vampire release page
(<https://vprover.github.io/download.html>).

`OPENAI_API_KEY` is required only for Exp 1's NL-verbalization step
(step 5). Copy [`.env.example`](.env.example) to `.env` and fill in.

## Reproducing each experiment

The canonical test-suite cache is
[`test_suites/test_suites.jsonl`](test_suites/test_suites.jsonl) —
1,393 FOLIO premises with deterministically-derived probes. To
regenerate from FOLIO gold:

```bash
python scripts/regenerate_test_suites.py
```

### Exp 1 — severity_correlation

```bash
python scripts/experiments/run_severity_correlation.py --step all
# → reports/experiments/severity_correlation/{candidates.json,
#       scored.json, analysis/, results.md, severity_curve.{png,pdf}}
```

### Exp 2 — perturbation_detection

```bash
python scripts/experiments/run_perturbation_detection.py --step all
# → reports/experiments/perturbation_detection/{scored.jsonl,
#       drop_magnitude.json, detection_rate.json, results.md, ...}
# Prereq: Exp 1's golds_by_stratum.json (for pool disjointness).
```

### Exp 3 — diagnostic_trace

```bash
python scripts/experiments/run_diagnostic_trace.py --step all
# → reports/experiments/diagnostic_trace/{pd_rescore/,
#       trace_features.jsonl, metrics.json, predictions.jsonl,
#       contrastive_firing_matrix.json, results.md}
# Prereq: Exp 2's scored.jsonl (the rescore reads the PD pool).
```

### Auxiliary scripts

```bash
python scripts/experiments/run_reference_robustness.py --step all  # FOLIO reference-error robustness
python scripts/experiments/worked_example.py                       # P1138 ∀→∃ figure
python scripts/experiments/select_correction_candidates.py         # Appendix D candidate selector
```

## Tests

```bash
pytest tests/                                 # full suite
pytest tests/test_soundness_invariants.py     # C9a / C9b on the curated corpus
```

## License

See [`LICENSE`](LICENSE).
