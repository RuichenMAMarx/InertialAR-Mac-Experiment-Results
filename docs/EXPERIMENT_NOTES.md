# Experiment notes

## What the local work establishes

The Mac adaptation supports packed variable-length causal attention without
FlashAttention, preserves isolation between molecules, and propagates finite
gradients. The tested geometry path is stable for the asymmetric fixtures under
atom permutation, rotation, and translation.

The official processed QM9 tensor loads successfully. Its dataset wrapper
reports 130,754 molecules and 641 functional-group condition classes. Real
samples flow through the original collator, both Transformer stages, the token
head, and the EDM coordinate loss.

Checkpoint save/reload, autoregressive token generation, coordinate sampling,
NPZ conversion, and the RDKit/EDM evaluator have also been exercised locally.

## Probe design

Representations are mean-pooled per molecule from packed token ranges. Six
locations are retained:

- `r_input`: token and geometry input before the Transformer.
- `r_context_init`: Transformer input after positional embedding.
- `r_context`: final first-stage autoregressive state.
- `r_type_context`: state directly consumed by the atom-type head.
- `r_coord_condition`: second-stage state supplied to the coordinate EDM net.
- `r_molecule`: mean-pooled coordinate-condition representation.

The original condition ID is replaced with a constant null embedding. This is
required because QM9 condition IDs are generated from the functional-group
pattern used as the probe target. The tiny checkpoint had no CFG row, so the
constant row was initialized as the mean learned condition embedding.

Binary probes exclude `Carbon_Atom_Count` and `Total_Count`, use a seeded
192/64 split, standardize features on the training portion, and solve a ridge
linear model. Accuracy is insufficient for rare labels; AUC and positive counts
must be considered together.

## Limitations

- The tiny model has 327,609 parameters and received only eight updates.
- Only 256 molecules were cached for the first probe.
- Labels with very few positives have unstable or undefined AUC values.
- The result does not demonstrate causal control or useful steering vectors.
- The official 7.17 GB QM9 checkpoint was not used locally.
- Paper-level stability, validity, uniqueness, and hit-rate comparisons remain
  pending on a Linux NVIDIA GPU.

## Next scientific experiment

Repeat representation extraction with the official checkpoint on at least
10,000 molecules. Use fixed train/validation/test indices, repeated seeds,
simple molecular descriptor baselines, and probe calibration. Derive steering
directions only for labels with adequate support, then evaluate interventions
at `alpha = -2, -1, 0, 1, 2` using target hit rate, validity, atom stability,
molecule stability, uniqueness, and response monotonicity.

