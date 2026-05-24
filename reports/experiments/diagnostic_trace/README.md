# diagnostic_trace — paper Exp 3

Built on the asymmetry-axiom rescore of the PD pool (which lives here
under `pd_rescore/`).

**Pipeline** — single orchestrator at
[`scripts/experiments/run_diagnostic_trace.py`](../../../scripts/experiments/run_diagnostic_trace.py)
with `--step {rescore, extract, classify, all}`:
1. `--step rescore` → writes `pd_rescore/`.
2. `--step extract` → writes `trace_features.jsonl`.
3. `--step classify` → writes `metrics.json` + `predictions.jsonl` + `contrastive_firing_matrix.json`.

## Contents

| File | Source | Description |
|---|---|---|
| `results.md` | hand-written | Human-readable summary — headline macro-F1, per-class F1, arg_swap deep-dive, disclosed limits. |
| `predicted_patterns.md` | pre-registered | Per-class (failed_positives, fired_contrastives) signatures (frozen). |
| `trace_features.jsonl` | extract step | 1,865 trace-feature rows. **Gitignored**; regenerable. |
| `extraction.log` | extract step | Extractor run log. **Gitignored**. |
| `metrics.json` | classifier step | Macro-F1 + per-class precision/recall/F1 + CIs + score-only baseline. |
| `predictions.jsonl` | classifier step | Per-pair predicted class. |
| `contrastive_firing_matrix.json` | classifier step | Per-class × per-contrastive firing-rate matrix. |
| `arg_swap_breakdown.json`, `arg_swap_by_stratum.json` | one-off analysis | The arg_swap deep-dive cited in `results.md` §5. |
| `pd_rescore/` | rescore step | PD pool rescored under the asymmetry-axiom label table — Exp 3's input pool. |
