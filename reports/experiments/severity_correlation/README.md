# severity_correlation — paper Exp 1

**Design source of truth**: [`configs/severity_correlation_v1.yaml`](../../../configs/severity_correlation_v1.yaml).

**Pipeline**: [`scripts/experiments/run_severity_correlation.py`](../../../scripts/experiments/run_severity_correlation.py).

## Contents

| File | Source | Description |
|---|---|---|
| `results.md` | step 7 | Human-readable summary — Spearman ρ table, per-tier means, monotonicity check, per-stratum decomposition. |
| `severity_curve.{png,pdf}` | step plot | Severity-curve line graph (the paper figure). |
| `golds_by_stratum.json` | step 1 | 128 FOLIO golds bucketed by stratum (S1–S5). |
| `candidates_raw.jsonl` | step 2 | Every (gold, operator) AST application attempt. |
| `candidates_verified.jsonl` | step 3 | Vampire-confirmed tier per row. |
| `candidates.json` | step 4/5 | Final pool: 267 sampled + NL-verbalized candidates. |
| `scored.json` | step 6 | Full metric matrix per candidate. |
| `gib_floor.json` | step gib | Unrelated-formula floor (sanity only; not in Spearman ρ). |
| `analysis/` | step 7 | Pre-registered analysis outputs (per_tier_means, monotonicity, rank_correlation, per_stratum, per_operator_auc). |
| `run_metadata.json` | every step | Git commit + step bookkeeping. |
| `.nl_cache/` | step 5 | LLM verbalization cache (gitignored; regenerable). |

## Reproducibility

All operators are deterministic AST transforms. The only stochastic
input is FOLIO premise sampling within each stratum; that sample is
seeded (seed=42) and recorded in `candidates.json` metadata. NL
verbalization (for BLEU/BERTScore) uses a deterministic prompt with a
single per-formula LLM call (each formula verbalized in isolation;
outputs cached).
