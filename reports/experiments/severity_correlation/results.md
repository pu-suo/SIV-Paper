# severity_correlation — results

Source of truth: [`configs/severity_correlation_v1.yaml`](../../../configs/severity_correlation_v1.yaml).


## 1. Candidate pool

Total candidates: **267** (after Vampire verification + cell-target sampling)

By tier: overstrong=58, partial=25, overweak=184

By stratum: S1=38, S2=81, S3=47, S4=26, S5=75


### Per-(stratum × tier) breakdown

| Stratum | Overstrong | Partial | Overweak | Total |
|---|---:|---:|---:|---:|
| S1 | 8 | 15 | 15 | 38 |
| S2 | 9 | 1 | 71 | 81 |
| S3 | 10 | 4 | 33 | 47 |
| S4 | 8 | 1 | 17 | 26 |
| S5 | 23 | 4 | 48 | 75 |

Verification: 547 raw → 529 retained (96.7% retention)


## 2. Per-tier means (descriptive)

Bootstrap 95% CI in parentheses. Descriptive evidence under §4 (η² + Cohen's d).

| Tier | n | siv_soft_recall | siv_soft_f1 | propositional_le_aligned | smatchpp | bleu | bertscore |
|---|---:|---:|---:|---:|---:|---:|---:|
| **gold** | 105 | 1.000 | 1.000 | 1.000 | 1.000 | 0.485 | 0.872 |
| **overstrong** | 58 | 0.966 | 0.929 | 0.850 | 0.806 | 0.396 | 0.815 |
| **partial** | 25 | 0.324 | 0.442 | 0.796 | 0.629 | 0.315 | 0.729 |
| **overweak** | 184 | 0.035 | 0.052 | 0.713 | 0.740 | 0.397 | 0.822 |

## 3. Severity monotonicity check

Does mean(gold) > mean(OS) > mean(P) > mean(OW)? (weak = ≥, strict = >). Inversions listed where present.

| Metric | Strict ↓ | Weak ↓ | Inversions |
|---|:---:|:---:|---|
| siv_soft_recall | ✓ | ✓ | — |
| siv_soft_f1 | ✓ | ✓ | — |
| propositional_le_aligned | ✓ | ✓ | — |
| smatchpp | ✗ | ✗ | partial<overweak |
| bleu | ✗ | ✗ | partial<overweak |
| bertscore | ✗ | ✗ | partial<overweak |

## 4. Magnitude of severity separation (primary)

The headline statistic is η² (one-way ANOVA, tier as factor) — the fraction of metric-score variance explained by severity tier — supplemented by Cohen's d on the REF-vs-OW contrast. η² and d reward absolute-magnitude calibration; rank-based statistics (AUC, ρ) are reported as secondary / tertiary below.

### 4.1 η² — variance explained by tier

| Metric | η² (3-level, OS/P/OW) | η² (4-level, +REF) | n total (3-level) | n total (4-level) |
|---|---:|---:|---:|---:|
| siv_soft_recall | 0.875 | 0.936 | 267 | 372 |
| siv_soft_f1 | 0.802 | 0.896 | 267 | 372 |
| propositional_le_aligned | 0.075 | 0.337 | 267 | 372 |
| smatchpp | 0.072 | 0.434 | 267 | 372 |
| bleu | 0.027 | 0.113 | 267 | 372 |
| bertscore | 0.170 | 0.290 | 267 | 372 |

η² is the fraction of total score variance explained by tier membership. Higher = severity tier is more predictive of the score (better calibration).

### 4.2 Cohen's d — REF-vs-OW effect size

Bootstrap 95% CI over 1,000 resamples of each tier (stratified). Aggregation matches the official ρ extraction (one score per (premise, tier) cell), restricted to the 53-premise ρ-eligible pool for comparability.

| Metric | Cohen's d | 95% CI | n_REF | n_OW |
|---|---:|---|---:|---:|
| siv_soft_recall | +7.264 | [+6.143, +9.209] | 53 | 53 |
| siv_soft_f1 | +4.715 | [+3.957, +6.039] | 53 | 53 |
| propositional_le_aligned | +1.499 | [+1.287, +1.832] | 53 | 53 |
| smatchpp | +1.842 | [+1.587, +2.188] | 53 | 53 |
| bleu | +1.089 | [+0.723, +1.513] | 53 | 53 |
| bertscore | +1.346 | [+1.022, +1.706] | 53 | 53 |

### 4.3 Global AUC — REF-vs-OW and REF-vs-perturbed (secondary)

AUC measures rank-separation; it saturates at 1.000 for any metric that returns its maximum score on REF by self-identity and strictly less on every perturbed candidate. Reported as secondary because (a) Smatch++ and SIV-soft-{recall,F1} all saturate at exactly 1.0 on REF by self-identity, making the AUC = 1.000 result largely a 'detected any change' statement, and (b) within-pool tie patterns differ across metrics in ways unrelated to severity calibration. See §4.1 (η²) and §4.2 (Cohen's d) for the substantive comparison.

| Metric | AUC REF-vs-OW | 95% CI | AUC REF-vs-perturbed | 95% CI |
|---|---:|---|---:|---|
| siv_soft_recall | 1.000 | [1.000, 1.000] | 0.826 | [0.785, 0.868] |
| siv_soft_f1 | 1.000 | [1.000, 1.000] | 0.860 | [0.818, 0.897] |
| propositional_le_aligned | 0.934 | [0.887, 0.972] | 0.950 | [0.921, 0.975] |
| smatchpp | 1.000 | [1.000, 1.000] | 1.000 | [1.000, 1.000] |
| bleu | 0.768 | [0.672, 0.857] | 0.818 | [0.756, 0.879] |
| bertscore | 0.882 | [0.818, 0.938] | 0.900 | [0.847, 0.947] |

## 5. Per-stratum decomposition (secondary)


### Stratum 1 (n=54 rows)

| Tier | siv_soft_recall | siv_soft_f1 | propositional_le_aligned | smatchpp | bleu | bertscore |
|---|---:|---:|---:|---:|---:|---:|
| gold | 1.000 | 1.000 | 1.000 | 1.000 | 0.555 | 0.896 |
| overstrong | 1.000 | 1.000 | 0.852 | 0.807 | 0.381 | 0.832 |
| partial | 0.333 | 0.487 | 0.773 | 0.478 | 0.274 | 0.694 |
| overweak | 0.300 | 0.452 | 0.744 | 0.424 | 0.246 | 0.678 |

Within-stratum ρ (n=16 qualifying premises):  
  - siv_soft_recall: 0.7507
  - siv_soft_f1: 0.7507
  - propositional_le_aligned: 0.8476
  - smatchpp: 0.8757
  - bleu: 0.8476
  - bertscore: 0.8456

### Stratum 2 (n=110 rows)

| Tier | siv_soft_recall | siv_soft_f1 | propositional_le_aligned | smatchpp | bleu | bertscore |
|---|---:|---:|---:|---:|---:|---:|
| gold | 1.000 | 1.000 | 1.000 | 1.000 | 0.416 | 0.860 |
| overstrong | 1.000 | 1.000 | 0.854 | 0.826 | 0.352 | 0.827 |
| partial | 0.000 | 0.000 | 0.875 | 0.649 | 0.508 | 0.775 |
| overweak | 0.000 | 0.000 | 0.730 | 0.792 | 0.397 | 0.839 |

Within-stratum ρ (n=29 qualifying premises):  
  - siv_soft_recall: 0.9231
  - siv_soft_f1: 0.9231
  - propositional_le_aligned: 0.9619
  - smatchpp: 0.9550
  - bleu: 0.9091
  - bertscore: 0.8730

### Stratum 3 (n=71 rows)

| Tier | siv_soft_recall | siv_soft_f1 | propositional_le_aligned | smatchpp | bleu | bertscore |
|---|---:|---:|---:|---:|---:|---:|
| gold | 1.000 | 1.000 | 1.000 | 1.000 | 0.524 | 0.898 |
| overstrong | 0.800 | 0.800 | 0.900 | 0.764 | 0.351 | 0.824 |
| partial | 0.250 | 0.250 | 0.812 | 0.904 | 0.433 | 0.841 |
| overweak | 0.000 | 0.000 | 0.750 | 0.678 | 0.410 | 0.857 |

Within-stratum ρ (n=24 qualifying premises):  
  - siv_soft_recall: 0.7485
  - siv_soft_f1: 0.7485
  - propositional_le_aligned: 0.8777
  - smatchpp: 0.9659
  - bleu: 0.3764
  - bertscore: 0.8805

### Stratum 4 (n=34 rows)

| Tier | siv_soft_recall | siv_soft_f1 | propositional_le_aligned | smatchpp | bleu | bertscore |
|---|---:|---:|---:|---:|---:|---:|
| gold | 1.000 | 1.000 | 1.000 | 1.000 | 0.471 | 0.833 |
| overstrong | 1.000 | 1.000 | 0.656 | 0.827 | 0.383 | 0.789 |
| partial | 0.000 | 0.000 | 1.000 | 0.933 | 0.466 | 0.749 |
| overweak | 0.000 | 0.000 | 0.794 | 0.799 | 0.393 | 0.802 |

Within-stratum ρ (n=8 qualifying premises):  
  - siv_soft_recall: 0.7379
  - siv_soft_f1: 0.7379
  - propositional_le_aligned: 0.2333
  - smatchpp: 0.6083
  - bleu: -0.1416
  - bertscore: 0.4605

### Stratum 5 (n=103 rows)

| Tier | siv_soft_recall | siv_soft_f1 | propositional_le_aligned | smatchpp | bleu | bertscore |
|---|---:|---:|---:|---:|---:|---:|
| gold | 1.000 | 1.000 | 1.000 | 1.000 | 0.487 | 0.860 |
| overstrong | 1.000 | 0.908 | 0.894 | 0.809 | 0.441 | 0.810 |
| partial | 0.525 | 0.688 | 0.797 | 0.838 | 0.264 | 0.728 |
| overweak | 0.042 | 0.059 | 0.625 | 0.783 | 0.439 | 0.824 |

Within-stratum ρ (n=28 qualifying premises):  
  - siv_soft_recall: 0.7981
  - siv_soft_f1: 0.9078
  - propositional_le_aligned: 0.8052
  - smatchpp: 0.8136
  - bleu: -0.1625
  - bertscore: 0.4089

## 6. Per-operator AUC (secondary)

AUC(gold vs operator's candidates) per metric. AUC=1.0 means the metric perfectly distinguishes gold from this operator's candidates; AUC=0.5 is chance.

| Operator | siv_soft_recall | siv_soft_f1 | propositional_le_aligned | smatchpp | bleu | bertscore |
|---|---:|---:|---:|---:|---:|---:|
| OS_add_nucleus_conjunct | 0.500 | 0.500 | 1.000 | 1.000 | 0.846 | 0.866 |
| OS_drop_conjunctive_restrictor | 0.500 | 1.000 | 1.000 | 1.000 | 0.494 | 0.925 |
| OS_narrow_consequent | 0.500 | 0.500 | 1.000 | 1.000 | 0.712 | 0.874 |
| OS_strengthen_predicate | 0.833 | 0.833 | 0.500 | 1.000 | 0.756 | 0.732 |
| OS_strengthen_quantifier | 0.500 | 1.000 | 0.500 | 1.000 | 0.419 | 0.762 |
| OW_de_quantify_to_c0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.981 | 0.850 |
| OW_drop_consequent_severely | 1.000 | 1.000 | 1.000 | 1.000 | 0.975 | 0.994 |
| OW_flip_outer_quantifier | 1.000 | 1.000 | 0.500 | 1.000 | 0.386 | 0.681 |
| OW_overrestrict_antecedent | 1.000 | 1.000 | 1.000 | 1.000 | 0.849 | 0.818 |
| OW_weaken_predicate_severely | 1.000 | 1.000 | 1.000 | 1.000 | 0.543 | 0.930 |
| OW_weaken_to_existential | 1.000 | 1.000 | 1.000 | 1.000 | 0.362 | 0.733 |
| P_drop_conjunct | 1.000 | 1.000 | 1.000 | 1.000 | 0.939 | 0.969 |
| P_drop_disjunctive_restrictor | 1.000 | 1.000 | 1.000 | 1.000 | 0.352 | 0.962 |
| P_weaken_predicate | 0.917 | 0.917 | 0.833 | 1.000 | 0.667 | 0.860 |

## 7. Aggregate Spearman ρ on REF-OS-P-OW (tertiary)

ρ is retained as a secondary descriptive statistic but is not the headline. Three reasons (also detailed in §4.3 below): (a) REF anchoring at 1.0 saturates ρ for any continuous-scored metric; (b) OS and P tie at rank 2, so ρ does not validate the OS-vs-P ordering — that is the per-tier means table's job in §2; (c) SIV's binary-leaning scoring produces within-premise ties (28% of perturbed candidates tie at exactly 1.0 with REF for SIV-soft-F1) that Spearman penalises as ρ = 0, dragging SIV's aggregate ρ down for reasons unrelated to metric quality. η² and Cohen's d (§4) are the substantive comparison.

Per-premise Spearman ρ over the four severity tiers (rank vector: REF=1, OS=P=2, OW=3), averaged across premises with ≥2 distinct ranks present (n=105 qualifying premises).

| Metric | mean ρ | 95% CI | n premises |
|---|---:|---|---:|
| siv_soft_recall | 0.8095 | [0.7519, 0.8625] | 105 |
| siv_soft_f1 | 0.8387 | [0.7819, 0.8939] | 105 |
| propositional_le_aligned | 0.8280 | [0.7725, 0.8867] | 105 |
| smatchpp | 0.8813 | [0.8451, 0.9165] | 105 |
| bleu | 0.4121 | [0.2731, 0.5547] | 105 |
| bertscore | 0.7153 | [0.6215, 0.7954] | 105 |

## 8. GIB floor (sanity check, not in main analysis)

30 pairs of unrelated FOLIO premises scored as (gold, cand). Should saturate near the metric's floor.

| Metric | mean | std | min | max | n |
|---|---:|---:|---:|---:|---:|
| siv_soft_recall | 0.082 | 0.256 | 0.000 | 1.000 | 30 |
| siv_soft_f1 | 0.091 | 0.264 | 0.000 | 1.000 | 30 |
| propositional_le_aligned | 0.582 | 0.148 | 0.281 | 0.883 | 30 |
| smatchpp | 0.279 | 0.204 | 0.036 | 0.667 | 30 |
| bleu | 0.063 | 0.059 | 0.003 | 0.246 | 30 |
| bertscore | 0.579 | 0.077 | 0.433 | 0.701 | 30 |

## 9. Findings to disclose


**Stratum availability in FOLIO (FOLIO-vs-synthetic source ratio):**
- S1: 30 sampled of 549 FOLIO golds available (synthetic augmentation: 0)
- S2: 30 sampled of 107 FOLIO golds available (synthetic augmentation: 0)
- S3: 30 sampled of 256 FOLIO golds available (synthetic augmentation: 0)
- S4: 8 sampled of 8 FOLIO golds available (synthetic augmentation: 0)
- S5: 30 sampled of 473 FOLIO golds available (synthetic augmentation: 0)

Under-represented strata in FOLIO (< 30 golds): S4. The design brief allows templated synthesis to fill; not yet implemented in v1.

**Cells with target=8 but 0 candidates retained** (applicability + Vampire verification dropped all candidates):
- OS_strengthen_quantifier × S4: 0 available, 0 retained
- OS_narrow_consequent × S1: 0 available, 0 retained
- OS_strengthen_predicate × S1: 0 available, 0 retained
- OS_strengthen_predicate × S4: 0 available, 0 retained
- OS_strengthen_predicate × S5: 0 available, 0 retained
- P_drop_consequent_atom × S5: 0 available, 0 retained
- P_weaken_predicate × S2: 0 available, 0 retained
- P_weaken_predicate × S5: 0 available, 0 retained
- OW_weaken_predicate_severely × S2: 0 available, 0 retained
- OW_weaken_predicate_severely × S5: 0 available, 0 retained

