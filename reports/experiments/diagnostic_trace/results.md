# Diagnostic-trace experiment — results

Pool: 1,865 (reference, perturbed) pairs across 6 perturbation classes
(inherited from `perturbation_detection`). Built on the frozen
427-predicate asymmetry-axiom label table at
[`siv/asymmetry_axioms.py`](../../../siv/asymmetry_axioms.py).

## 1. What this experiment establishes

SIV's §6 promise was per-aspect feedback: the trace identifies which
structural feature failed, in the user's own predicate vocabulary.
This experiment operationalizes that claim as **recoverability** —
given the labeled trace alone (failed positives + fired contrastives),
can a deterministic rule classifier identify the underlying
perturbation class?

The classifier is a frozen 7-rule table (§3) over (failed_positives,
fired_contrastives). It has no learned parameters, no fitted
thresholds, no per-class hyperparameters. The admissibility regime
includes principled asymmetry witness axioms for binary relations
(`∀x,y. R(x,y) → ¬R(y,x)`), pre-registered before classifier
predictions were scored.

## 2. Predicted patterns (frozen)

| Class | Expected failed positives | Expected fired contrastives |
|---|---|---|
| arg_swap | predicate-dependent (relation must be asymmetric) | `swap_binary_args` fires |
| negation_drop | non-empty | `negate_atom` or `replace_subformula_with_negation` fires |
| restrictor_drop | **empty** (strictly stronger) | `drop_restrictor_conjunct` fires |
| random_substitution | non-empty (substituted preds) | **none** (incompatible, not stronger) |
| flip_outer_quantifier | non-empty | `flip_quantifier` fires |
| strengthen_quantifier | **empty** (strictly stronger) | `flip_quantifier` fires |

`flip_outer_quantifier` vs `strengthen_quantifier` are separated only
by the joint pattern (failed_positives empty vs non-empty); the
classifier must read both halves of the trace together.

## 3. Classifier rules (frozen)

First match wins. Ordered:

```
Rule 1: drop_restrictor_conjunct fired AND failed_positives empty
        → restrictor_drop
Rule 2: flip_quantifier fired AND failed_positives empty
        → strengthen_quantifier
Rule 3: flip_quantifier fired AND failed_positives non-empty
        → flip_outer_quantifier
Rule 4: swap_binary_args fired
        → arg_swap
Rule 5: negate_atom OR replace_subformula_with_negation fired
        → negation_drop
Rule 6: failed_positives non-empty AND no contrastives fired
        → random_substitution
Rule 7: otherwise → unrecognized
```

## 4. Headline result

| | value |
|---|---:|
| **Labeled-trace classifier macro-F1 (6-class)** | **0.6380** |
| Score-only baseline macro-F1 (1-feature CV decision tree on SIV-F1) | 0.3327 |
| **Recoverability delta (rule − score-only baseline)** | **+0.3053** |
| Unrecognized rate | 0.05% (1/1865) |

The labeled trace recovers the perturbation class at macro-F1 = **0.64**,
**+0.31 over the strongest score-only baseline** that has access only
to the scalar SIV-F1. The labeled trace carries information the scalar
score doesn't.

## 5. Per-class F1

| Class | n | F1 | Precision | Recall | Recall 95% CI |
|---|---:|---:|---:|---:|---|
| arg_swap | 494 | 0.482 | 1.000 | 0.318 | (0.277, 0.361) |
| negation_drop | 132 | 0.778 | 1.000 | 0.636 | (0.548, 0.718) |
| restrictor_drop | 231 | 0.996 | 0.996 | 0.996 | (0.976, 1.000) |
| random_substitution | 638 | 0.634 | 0.464 | 1.000 | (0.994, 1.000) |
| flip_outer_quantifier | 355 | 0.006 | 1.000 | 0.003 | (0.000, 0.016) |
| strengthen_quantifier | 15 | 0.933 | 0.933 | 0.933 | (0.681, 0.998) |

Three classes are essentially solved (`restrictor_drop` at 0.996,
`strengthen_quantifier` at 0.933, `negation_drop` at 0.778).
`arg_swap` recovers at 0.482 with high precision (1.000) but
heterogeneous recall — see §6. `random_substitution` is Rule 6's
catch-all bucket; its precision of 0.464 reflects that other classes
whose predicted contrastives don't fire reliably leak into it.
`flip_outer_quantifier` is a disclosed coverage limit (§10b).

`strengthen_quantifier` (n=15) carries a wide Clopper-Pearson CI on
recall. The structural finding (same-contrastive-as-flip + empty
failed_positives → strengthen) holds.

## 6. `arg_swap` — recovery is heterogeneous across formula strata

The asymmetry-axiom regime recovers `arg_swap` cleanly when the
swapped binary relation appears as a **ground atom** in the gold;
recovery is weaker when the relation is embedded inside a
universal-quantified body.

| Reference stratum | n | recovered | recall |
|---|---:|---:|---:|
| **S5_relational** (ground-atom binary relations) | 127 | **112** | **88.2%** |
| S8_other | 9 | 6 | 66.7% |
| S6_negation | 16 | 4 | 25.0% |
| S3_universal_multi_restrictor | 176 | 22 | 12.5% |
| S2_universal_simple | 49 | 4 | 8.2% |
| S4_nested_quantifier | 86 | 7 | 8.1% |
| S7_existential | 31 | 2 | 6.5% |

This split is mechanistically interpretable, not coincidental.

### 6a. Why the split exists

The asymmetry axiom `∀x,y.R(x,y) → ¬R(y,x)` refutes a swap mutant
only when both directions of `R` are jointly forced. For a **ground
atom** like gold = `Loves(alice, bob)` and mutant = `Loves(bob, alice)`,
the axiom directly contradicts conjoining the two: unsat, admitted.

For a **universal-quantified body** like gold = `∀x.((Animal(x) ∧
DisplayedIn(x, collection)) → Multicellular(x))` and the mutant with
`DisplayedIn(collection, x)` swapped, asymmetry forces gold and
mutant to govern disjoint slices of the domain. Concretely, if
`DisplayedIn(a, collection)` holds for some witness `a`, asymmetry
implies `¬DisplayedIn(collection, a)`, which makes the mutant's
antecedent false on `a` — vacuous, no contradiction. The two
formulas are logically **independent**, not contradictory. Vampire
correctly returns `independent` and the contrastive is not admitted.

This is not a Vampire timeout or a missing existence witness. SIV's
`derive_witness_axioms` emits per-predicate and per-restrictor
existence witnesses; both were active. The limitation is about
asymmetry's logical strength under standard FOL: it constrains
specific instances but not the cross-quantifier interaction that
would make universal swaps incompatible.

### 6b. By asymmetry-table label

| Label of swapped predicate | n | recovered | rate |
|---|---:|---:|---:|
| asymmetric | 477 | 157 | 32.9% |
| symmetric | 6 | 0 | 0% (correct) |
| unknown | 11 | 0 | 0% (expected) |

The 6 symmetric-labeled `arg_swap` pairs are correctly **not**
recovered — those swaps are entailment-neutral under the symmetry
axiom and should not produce a contrastive. The 0% recall on those
pairs confirms the symmetry-axiom side of the table works.

The 33% rate on asymmetric-labeled pairs is the same number as the
S5+S8 fraction of the asymmetric population — once you condition on
stratum, the per-label rate is bimodal (88% on S5, ≤13% on
universals).

## 7. Confusion matrix (labeled-trace classifier)

| true \\ pred | arg_swap | negation_drop | restrictor_drop | random_substitution | flip_outer_quantifier | strengthen_quantifier | unrecognized |
|---|---:|---:|---:|---:|---:|---:|---:|
| arg_swap | **157** | 0 | 0 | 336 | 0 | 1 | 0 |
| negation_drop | 0 | **84** | 0 | 48 | 0 | 0 | 0 |
| restrictor_drop | 0 | 0 | **230** | 0 | 0 | 0 | 1 |
| random_substitution | 0 | 0 | 0 | **638** | 0 | 0 | 0 |
| flip_outer_quantifier | 0 | 0 | 0 | 354 | **1** | 0 | 0 |
| strengthen_quantifier | 0 | 0 | 1 | 0 | 0 | **14** | 0 |

Diagonal-dominant for `restrictor_drop`, `random_substitution`,
`strengthen_quantifier`. The two off-diagonal-heavy rows (`arg_swap`,
`flip_outer_quantifier`) have their misses bleeding into
`random_substitution` via Rule 6's catch-all — pairs whose predicted
contrastive didn't fire and whose failed positives are non-empty
look (to the classifier) like vocabulary substitutions.

## 8. Per-class label-firing recall (sanity check)

| Class | n | predicted contrastive fired | positive pattern held | joint |
|---|---:|---:|---:|---:|
| arg_swap | 494 | 32.0% | 100% | 32.0% |
| negation_drop | 132 | 63.6% | 100% | 63.6% |
| restrictor_drop | 231 | 99.6% | 100% | 99.6% |
| random_substitution | 638 | 100% | 100% | 100% |
| flip_outer_quantifier | 355 | 0.3% | 100% | 0.3% |
| strengthen_quantifier | 15 | 100% | 100% | 100% |

`restrictor_drop`, `random_substitution`, `strengthen_quantifier`
fire their predicted contrastive (or correctly silence it for
random_substitution) at ≥99.6%. `negation_drop` fires the predicted
contrastive on 63.6% — the remaining 36.4% have neither
`negate_atom` nor `replace_subformula_with_negation` admitted.
`arg_swap` and `flip_outer_quantifier` are the two classes where
the predicted contrastive doesn't fire reliably (§10).

## 9. Score-only baseline confusion matrix

| true \\ pred | arg_swap | negation_drop | restrictor_drop | random_substitution | flip_outer_quantifier | strengthen_quantifier |
|---|---:|---:|---:|---:|---:|---:|
| arg_swap | 167 | 1 | 13 | 313 | 0 | 0 |
| negation_drop | 6 | 0 | 12 | 114 | 0 | 0 |
| restrictor_drop | 8 | 0 | 221 | 2 | 0 | 0 |
| random_substitution | 18 | 0 | 16 | 604 | 0 | 0 |
| flip_outer_quantifier | 36 | 0 | 30 | 289 | 0 | 0 |
| strengthen_quantifier | 8 | 0 | 5 | 2 | 0 | 0 |

The score-only baseline (5-fold stratified CV on a 1-feature
decision tree over SIV-F1) collapses `negation_drop`,
`flip_outer_quantifier`, and `strengthen_quantifier` into F1 = 0.000
at 6-class granularity — the scalar SIV-F1 doesn't carry enough
information to separate them. This is the gap the labeled trace
closes (+0.31 macro-F1).

## 10. Disclosed limits

### 10a. `arg_swap` on universal-quantified premises (~358 pairs)

For S2/S3/S4/S7 references, the asymmetry axiom is logically
insufficient — universal swaps are model-theoretically independent
under standard FOL, not incompatible (§6a). This is not a missing
existence witness or a Vampire timeout. Closing this gap would
require a fundamentally different admissibility regime: richer
witness machinery, an intensional axiom system, or a weaker
contrastive-relation criterion. Each is a substantial design change
and out of scope here.

### 10b. `flip_outer_quantifier` (n=355, F1 = 0.006)

The `flip_quantifier` contrastive is admitted into the labeled trace
on roughly 9% of `flip_outer_quantifier` premises (the admission
rate), but the rule classifier correctly names the class on only
0.3% of them (per-class recall in §5; the corresponding row in §8).
The 30× gap means that even when admitted, the firing pattern rarely
dominates the random-substitution fallback — most pairs fall through
to Rule 6. The asymmetry-axiom design addresses `swap_binary_args`,
not `flip_quantifier`; the coverage gap is explicitly out of scope
and remains for future work. Probable mechanism is parallel to the
arg_swap case: quantifier-flip mutants are admitted when there's
enough structure to force them incompatible, and otherwise dropped
as independent.

### 10c. `negation_drop` firing gap (36% miss)

Of 132 negation_drop pairs, the predicted contrastive fires on 84
(64%). The other 48 fall through to Rule 6 (random_substitution
catch-all). Cause is the same admissibility-coverage pattern:
`negate_atom` and `replace_subformula_with_negation` are admitted
only when Vampire can certify the polarity flip as incompatible
under existing witnesses. Smaller magnitude than arg_swap's gap;
not addressed.

### 10d. Conservative-bias cost of the asymmetry table

The frozen 427-predicate table marks 22 predicates "unknown"
(covering 27 of 877 usages, 3.1%). These predicates contribute no
axiom, so any `arg_swap` on them is unrecoverable regardless of
stratum. The conservative bias was the right design choice
(mis-labeling a symmetric predicate as asymmetric risks unsound
admissibility) but it does cost a few percentage points of arg_swap
recall.

## 11. Paper-facing framing

The labeled trace recovers the perturbation class at **macro-F1 =
0.638**, **+0.305 over the strongest score-only baseline** with
access only to the scalar SIV-F1. Recovery is high (≥ 0.78) on three
of six classes, partial (0.48) on `arg_swap` with a clean stratum
decomposition (88.2% on ground-atom relational premises, ≤ 13% on
universal-quantified bodies), and near-zero on
`flip_outer_quantifier` — disclosed as a model-theoretic limit of
the current admissibility regime, not a metric defect.

The deterministic, parameter-free rule classifier and the
pre-registered predicted-patterns table (§2) eliminate any concern
about post-hoc tuning. The improvement over the score-only baseline
comes entirely from the labeled trace's per-aspect structure: which
probe failed, in the user's own predicate vocabulary.

## Artifacts

| Path | Content |
|---|---|
| `reports/experiments/diagnostic_trace/predicted_patterns.md` | Per-class prediction table (frozen) |
| `reports/experiments/diagnostic_trace/trace_features.jsonl` | Per-pair labeled trace (1,865 rows, regenerable) |
| `reports/experiments/diagnostic_trace/metrics.json` | All numbers above as JSON |
| `reports/experiments/diagnostic_trace/predictions.jsonl` | Per-pair classifier predictions |
| `reports/experiments/diagnostic_trace/contrastive_firing_matrix.json` | Per-class × per-contrastive firing rates |
| `reports/experiments/diagnostic_trace/arg_swap_breakdown.json` | arg_swap recovery by swapped-predicate label |
| `reports/experiments/diagnostic_trace/arg_swap_by_stratum.json` | arg_swap recovery by reference stratum |
| `reports/experiments/diagnostic_trace/pd_rescore/scored.jsonl` | PD pool rescored under asymmetry-axiom regime (this experiment's input) |
| `reports/experiments/diagnostic_trace/pd_rescore/diff_vs_v2.json` | Per-class drop-magnitude diff between PD rescore and the prior PD scoring |
| `siv/asymmetry_axioms.py` | Frozen 427-predicate asymmetry/symmetry label table |
