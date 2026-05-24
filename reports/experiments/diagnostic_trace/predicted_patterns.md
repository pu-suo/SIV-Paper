# Diagnostic-trace experiment — predicted label-firing patterns

**FROZEN before classifier construction.** Recorded ahead of the
classifier construction so the recoverability classifier's rules can
only encode structural predictions made in advance, not patterns
discovered by inspecting classifier-failure cases.

Pool: the perturbation-detection 1,865-pair set.
Scoring: SIV **strict** mode (no vocabulary alignment), Vampire
timeout 5 s.

## Per-class predictions

| Class | Expected failed positives | Expected fired contrastives | Joint-pattern note |
|---|---|---|---|
| **arg_swap** | Positives over the swapped binary predicate may fail (predicate-dependent; the relation must be asymmetric under witness axioms). `full_reference` typically fails. | `swap_binary_args` fires on the swapped predicate. | Contrastive firing is the primary signal; positive-failure pattern is dependent on relation asymmetry. |
| **negation_drop** | Positives whose feature involves the dropped negation fail. `full_reference` typically fails. | `negate_atom` or `replace_subformula_with_negation` fires on the predicate whose polarity was changed. | Either contrastive name is admitted (they're alternative encodings of the same structural error). |
| **restrictor_drop** | **None.** The perturbed candidate is strictly stronger than the reference (the dropped restrictor conjunct makes the rule apply to a wider class), so it entails every positive sub-test by construction. | `drop_restrictor_conjunct` fires on the dropped restrictor predicate. | "No failed positives" is itself the distinguishing positive-side signal. |
| **random_substitution** | Positives whose feature_target is a substituted predicate name fail (the candidate's vocabulary doesn't include that predicate). `full_reference` typically fails. | **Mostly none.** The substituted formula is propositionally incompatible with the reference but not strictly stronger; under strict mode it doesn't entail any reference contrastive either. | Positives carry the signal; contrastives are silent. This is the only class where the diagnostic is positive-side. |
| **flip_outer_quantifier** | Positives requiring the universal scaffolding fail (`∀x.φ` → `∃x.φ` weakens the claim, so the perturbed candidate doesn't entail the universal-form positives). `full_reference` fails. | `flip_quantifier` fires on the flipped binder's variable (`x` or `<var>:<restrictor_pred>`). | Same contrastive as `strengthen_quantifier` — distinguished from it by the positive-failure pattern (failed_positives is non-empty here). |
| **strengthen_quantifier** | **None.** The strengthened quantifier (`∃y.ψ` → `∀y.ψ` in the consequent) yields a strictly-stronger formula that entails every reference positive. | `flip_quantifier` fires (other direction: the strengthening looks the same as a flip to the contrastive operator). | Same contrastive label as `flip_outer_quantifier`. Joint signal: same fired contrastive + empty failed_positives → strengthen_quantifier. |

## Joint-pattern requirement

Two pairs of classes share their contrastive-firing pattern:

- **flip_outer_quantifier vs strengthen_quantifier** — both fire
  `flip_quantifier`. Distinguished only by `failed_positives` empty
  vs non-empty.
- **restrictor_drop vs strengthen_quantifier** — both are
  strictly-stronger perturbations with no failed positives, but they
  fire different contrastives (`drop_restrictor_conjunct` vs
  `flip_quantifier`).

The classifier MUST use the joint signal — fired_contrastives AND
failed_positives together — to separate these. A contrastive-only
classifier collapses the first pair.

## What this prediction commits to

If a class's predicted-firing pattern holds on a candidate, that
candidate is recoverable. If the prediction does not hold, the
candidate either falls through to "unrecognized" or is misclassified
into another class. Either way, that is a result — not a tuning
opportunity. The classifier rules are derived exclusively from this table.

## What this prediction does NOT commit to

- A specific F1 threshold to clear.
- Behavior on candidates whose perturbation class falls outside the
  six listed. The classifier is closed-world over these six.
- Behavior under SIV soft mode (vocabulary alignment). Predictions
  are calibrated to strict mode, in particular the
  random_substitution positive-failure prediction.
