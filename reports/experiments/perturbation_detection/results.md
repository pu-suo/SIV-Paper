# perturbation_detection — results

## 1. Question and answer

**Question:** Does SIV's score drop on structural perturbations of the gold FOL, reliably, across structural classes?

**Answer:** Yes, on all six classes. Within-pair detection rate for `SIV-strict-F1` is 0.978–1.000 across classes, with mean drops ranging from 0.169 (`restrictor_drop`) to 0.975 (`flip_outer_quantifier`) and 95% CIs excluding zero on every class. The architectural design — F1 over a contrastive arm — earns its keep on strictly-stronger classes, where SIV-recall alone is saturated by construction (§5).

**Scope.** This experiment verifies SIV's per-class detection behaviour. Baseline metrics (BLEU, BERTScore, Smatch++, LE-aligned, Brunello-LT-aligned) are included in the tables for context; this is not a metric comparison. Comparison-mode evaluation of SIV against baselines is the role of `severity_correlation` (RQ1) and is reported there. The reference pool used here is disjoint from `severity_correlation`'s 128-premise design pool by construction.

## 2. Pool composition

- Reference pool: **642 FOLIO premises**, disjoint from severity_correlation_v1's 128-premise pre-verification design pool by construction (subtracted from the structural-richness-filtered FOLIO base).
- Verified (reference, perturbed) pairs: **1,865** across 6 classes (51 raw candidates dropped by Vampire bidirectional entailment check; per-class drop reasons in `run_metadata.json`).

| Class | n (verified) | Expected entailment | Notes |
|---|---:|---|---|
| arg_swap | 494 | incompatible | argument-order axis |
| negation_drop | 132 | incompatible | polarity axis |
| restrictor_drop | 231 | cand ⊨ ref | strictly-stronger; load-bearing for the architectural-payoff demo (§5) |
| random_substitution | 638 | incompatible | lexical baseline class (predicate-name substitution) |
| flip_outer_quantifier | 355 | ref ⊨ cand | quantifier-scope axis (canonical LE-failure case) |
| strengthen_quantifier | 15 | cand ⊨ ref | strictly-stronger; **low-yield (n=15)** — secondary corroboration only |

## 3. Per-class drop magnitude

Mean of `score(reference) − score(perturbed)` over verified pairs in each class. Cell format: `mean ± std  [95% CI]`. Bootstrap 1,000 resamples, seed 42.

**Read order for SIV's validity claim:** SIV-strict-F1 row first — drops are nonzero with CIs excluding zero on every class, confirming SIV reliably responds to each structural class. SIV-strict-recall is included to show the recall-only saturation on strictly-stronger classes (`restrictor_drop`, `strengthen_quantifier`), which §5 unpacks.

| Metric | arg_swap | negation_drop | restrictor_drop | random_substitution | flip_outer_quantifier | strengthen_quantifier |
|---|---:|---:|---:|---:|---:|---:|
| BLEU | 0.607 ± 0.078  [0.601, 0.615] | 0.534 ± 0.102  [0.516, 0.551] | 0.564 ± 0.074  [0.554, 0.573] | 0.643 ± 0.090  [0.636, 0.650] | 0.474 ± 0.083  [0.466, 0.483] | 0.502 ± 0.063  [0.470, 0.532] |
| BERTScore | 0.169 ± 0.043  [0.166, 0.173] | 0.189 ± 0.066  [0.178, 0.201] | 0.200 ± 0.036  [0.195, 0.204] | 0.290 ± 0.043  [0.287, 0.294] | 0.150 ± 0.039  [0.147, 0.155] | 0.186 ± 0.030  [0.172, 0.201] |
| Smatch++ | 0.102 ± 0.040  [0.099, 0.106] | 0.089 ± 0.048  [0.081, 0.098] | 0.237 ± 0.084  [0.227, 0.248] | 0.162 ± 0.035  [0.159, 0.164] | 0.174 ± 0.071  [0.167, 0.181] | 0.067 ± 0.059  [0.040, 0.097] |
| LE-aligned | 0.146 ± 0.084  [0.139, 0.154] | 0.285 ± 0.180  [0.254, 0.318] | 0.121 ± 0.042  [0.115, 0.126] | 0.258 ± 0.120  [0.249, 0.267] | 0.000 ± 0.000  [0.000, 0.000] | 0.000 ± 0.000  [0.000, 0.000] |
| Brunello-LT-aligned | 1.000 ± 0.000  [1.000, 1.000] | 1.000 ± 0.000  [1.000, 1.000] | 1.000 ± 0.000  [1.000, 1.000] | 1.000 ± 0.000  [1.000, 1.000] | 1.000 ± 0.000  [1.000, 1.000] | 1.000 ± 0.000  [1.000, 1.000] |
| SIV-strict-recall | 0.818 ± 0.248  [0.796, 0.841] | 0.861 ± 0.226  [0.820, 0.899] | 0.000 ± 0.000  [0.000, 0.000] | 0.960 ± 0.130  [0.949, 0.970] | 0.984 ± 0.082  [0.975, 0.992] | 0.000 ± 0.000  [0.000, 0.000] |
| SIV-strict-F1 | 0.757 ± 0.314  [0.731, 0.784] | 0.820 ± 0.289  [0.770, 0.868] | 0.169 ± 0.046  [0.163, 0.175] | 0.942 ± 0.177  [0.928, 0.956] | 0.975 ± 0.125  [0.961, 0.986] | 0.325 ± 0.218  [0.232, 0.439] |
| SIV-soft-recall | 0.818 ± 0.248  [0.796, 0.841] | 0.861 ± 0.226  [0.820, 0.899] | 0.000 ± 0.000  [0.000, 0.000] | 0.960 ± 0.130  [0.949, 0.970] | 0.984 ± 0.082  [0.975, 0.992] | 0.000 ± 0.000  [0.000, 0.000] |
| SIV-soft-F1 | 0.757 ± 0.314  [0.731, 0.784] | 0.820 ± 0.289  [0.770, 0.868] | 0.169 ± 0.046  [0.163, 0.175] | 0.942 ± 0.177  [0.928, 0.956] | 0.975 ± 0.125  [0.961, 0.986] | 0.325 ± 0.218  [0.232, 0.439] |

Drop magnitudes vary across SIV's classes by design: `restrictor_drop` produces a smaller SIV-F1 drop (0.169) than `flip_outer_quantifier` (0.975) because the former is a strictly-stronger perturbation whose positive recall stays at 1.0 — only contrastives fire — while the latter breaks positives outright. This is calibration evidence: the score moves with the structural severity of the perturbation, not just its presence.

## 4. Per-class detection rate (within-pair)

Fraction of `(reference, perturbed)` pairs where `score(reference) > score(perturbed)` strictly. This is the paired companion to §3: drop magnitude says *how much* the score moves; detection rate says *how reliably*. Bootstrap 1,000 resamples, seed 42. Ties count as non-detections; per-class tie counts are reported in `detection_rate.json`.

| Metric | arg_swap | negation_drop | restrictor_drop | random_substitution | flip_outer_quantifier | strengthen_quantifier |
|---|---:|---:|---:|---:|---:|---:|
| BLEU | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| BERTScore | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| Smatch++ | 0.982 [0.970, 0.992] | 0.992 [0.977, 1.000] | 1.000 [1.000, 1.000] | 0.997 [0.992, 1.000] | 0.997 [0.992, 1.000] | 1.000 [1.000, 1.000] |
| LE-aligned | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Brunello-LT-aligned | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| SIV-strict-recall | 0.998 [0.994, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| SIV-strict-F1 | 0.998 [0.994, 1.000] | 0.992 [0.977, 1.000] | 0.996 [0.987, 1.000] | 0.998 [0.995, 1.000] | 0.997 [0.992, 1.000] | 1.000 [1.000, 1.000] |
| SIV-soft-recall | 0.998 [0.994, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| SIV-soft-F1 | 0.998 [0.994, 1.000] | 0.992 [0.977, 1.000] | 0.996 [0.987, 1.000] | 0.998 [0.995, 1.000] | 0.997 [0.992, 1.000] | 1.000 [1.000, 1.000] |

SIV-strict-F1 detection rate is ≥ 0.978 on every class; on strictly-stronger classes (`restrictor_drop`, `strengthen_quantifier`) the contrastive arm carries the signal while SIV-strict-recall saturates at detection rate ≈ 0 (every pair is a tie at 1.0). §5 unpacks that recall→F1 contrast.

## 5. OVERSTRONG sanity check — contrastive arm earning its keep

Strictly-stronger candidates pass all positive probes by construction (perturbed ⊨ reference), so SIV-recall is at the ceiling for both reference and perturbed and cannot distinguish them. Any SIV-F1 drop on these classes therefore comes **entirely from contrastive firings**. This is the direct test that SIV's contrastive arm is doing the architectural work.

Per-metric drops on the two strictly-stronger classes:

| Metric | restrictor_drop (n=231) | strengthen_quantifier (n=15) |
|---|---|---|
| BLEU | drop=0.564  (gold=1.000, pert=0.436) | drop=0.502  (gold=1.000, pert=0.498) |
| BERTScore | drop=0.200  (gold=1.000, pert=0.800) | drop=0.186  (gold=1.000, pert=0.814) |
| Smatch++ | drop=0.237  (gold=1.000, pert=0.763) | drop=0.067  (gold=1.000, pert=0.933) |
| LE-aligned | drop=0.121  (gold=1.000, pert=0.879) | drop=0.000  (gold=1.000, pert=1.000) |
| Brunello-LT-aligned | drop=1.000  (gold=1.000, pert=0.000) | drop=1.000  (gold=1.000, pert=0.000) |
| SIV-strict-recall | drop=0.000  (gold=1.000, pert=1.000) | drop=0.000  (gold=1.000, pert=1.000) |
| SIV-strict-F1 | drop=0.169  (gold=1.000, pert=0.831) | drop=0.325  (gold=1.000, pert=0.675) |
| SIV-soft-recall | drop=0.000  (gold=1.000, pert=1.000) | drop=0.000  (gold=1.000, pert=1.000) |
| SIV-soft-F1 | drop=0.169  (gold=1.000, pert=0.831) | drop=0.325  (gold=1.000, pert=0.675) |

### 5.1 Verdict

On `restrictor_drop` (n=231):
- `SIV-strict-recall` drop = 0.000 — saturated, as expected.
- `SIV-strict-F1` drop = 0.169 with 95% CI excluding zero — contrastives fire.

On `strengthen_quantifier` (n=15, low-yield):
- `SIV-strict-recall` drop = 0.000 — saturated, as expected.
- `SIV-strict-F1` drop = 0.325 — contrastives fire (wider CI from small n).

**SIV-F1 moves where SIV-recall provably cannot.** The contrastive arm is doing the architectural work, on both strictly-stronger classes.

## 6. `restrictor_drop` — the load-bearing class

`restrictor_drop` deletes one conjunct from the antecedent of a universal implication: `∀x.((A(x) ∧ B(x)) → C(x))` → `∀x.(A(x) → C(x))`. The perturbed formula is strictly stronger than the reference. By construction, every positive probe generated from the reference also passes under the perturbed formula, so any SIV-F1 drop is carried by the contrastive arm alone.

Per-metric behaviour on this class:

| Metric | Mean drop | 95% CI | Detection rate | 95% CI |
|---|---:|---|---:|---|
| BLEU | 0.564 | [0.554, 0.573] | 1.000 | [1.000, 1.000] |
| BERTScore | 0.200 | [0.195, 0.204] | 1.000 | [1.000, 1.000] |
| Smatch++ | 0.237 | [0.227, 0.248] | 1.000 | [1.000, 1.000] |
| LE-aligned | 0.121 | [0.115, 0.126] | 1.000 | [1.000, 1.000] |
| Brunello-LT-aligned | 1.000 | [1.000, 1.000] | 1.000 | [1.000, 1.000] |
| SIV-strict-recall | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] |
| SIV-strict-F1 | 0.169 | [0.163, 0.175] | 0.996 | [0.987, 1.000] |
| SIV-soft-recall | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] |
| SIV-soft-F1 | 0.169 | [0.163, 0.175] | 0.996 | [0.987, 1.000] |

Reading the row of interest for SIV's design claim: `SIV-strict-recall` detection rate is at the floor (every pair is a tie at 1.0) and `SIV-strict-F1` detection rate is at the ceiling (0.996). The F1 design recovers the entire detection signal on this class from the contrastive arm.

## 7. Provenance and sanity

- All 1,865 verified pairs received a Smatch++ score (1865/1865, 100%). No upstream filter is silently dropping Smatch++ rows.
- LE-aligned is the predicate-truth-table version (Yang et al. 2024, MALLS §4.3), shared with `severity_correlation`. Its detection rate of 0.000 on quantifier-flip classes (`flip_outer_quantifier`, `strengthen_quantifier`) — every pair ties at the metric ceiling — reproduces the canonical LE-failure case: stripping quantifiers before truth-table evaluation makes ∀x.φ and ∃x.φ indistinguishable. This is a sanity check that the implementation is faithful to its definition, not a finding against SIV.
- Brunello-LT-aligned is a binary 0/1 equivalence check (Z3); its drop = 1.000 on every non-equivalent class is structurally inevitable and informational rather than comparative.
- Reference-pool disjointness from `severity_correlation_v1` is enforced at step 1 of the pipeline by subtracting the 128 premise IDs in `reports/experiments/severity_correlation/golds_by_stratum.json`. The list of 642 reference premise IDs used here is in `reference_pool.jsonl`.
