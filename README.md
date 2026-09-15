# InertialAR Mac Experiment Results

This repository records a bounded reproduction of InertialAR on an Apple
Silicon Mac, followed by real-QM9 representation extraction and preliminary
linear probing.

The local machine could run CPU smoke tests but could not expose MPS to the
active PyTorch process. The released 7.17 GB QM9 checkpoint was therefore not
loaded locally. Results involving the tiny model validate implementation and
experiment plumbing; they are not reproductions of the paper's molecular
generation metrics.

## Environment

- macOS 26.3.1, Apple Silicon arm64
- Python 3.9.6
- PyTorch 2.3.1
- NumPy 1.26.4
- SciPy 1.13.1
- pandas 2.2.3
- RDKit 2025.09.2
- torch-geometric 2.6.1
- OGB 1.2.1
- Device used: CPU (`mps_built=True`, `mps_available=False`)

## Completed stages

| Stage | Purpose | Outcome |
|---|---|---|
| 01 | Mac attention and device adaptation | Dense causal attention fallback passed packed-sequence isolation and gradient checks |
| 02 | Geometry mechanisms | Canonical ordering, inertial-frame invariance, GeoRoPE, and Nyström checks passed |
| 03 | Synthetic tiny training | Full forward/backward and optimizer step passed |
| 04 | Tiny checkpoint generation | Checkpoint reload, autoregressive generation, EDM sampling, and evaluator plumbing passed |
| 05 | Real QM9 loader | Loaded 130,754 processed molecules and packed a real variable-length batch |
| 06 | Real QM9 tiny optimization | Eight CPU optimizer updates completed; checkpoint saved and strictly reloaded |
| 07 | Representation cache and probe | Six 64-dimensional representations extracted for 256 real molecules with condition-label leakage removed |

## Main results

Geometry validation reached a maximum canonical-coordinate discrepancy of
`1.34e-6` after rotation and translation. The inertial frame determinant was
`1.0`, and all Nyström factors were finite.

The real-QM9 smoke batch contained four molecules and 26 packed tokens. A full
forward/backward update took about 4.45 seconds on CPU with approximately
1.34 GB peak RSS.

The Stage 07 probe used a constant null condition. This matters because the
dataset condition ID is itself constructed from functional-group labels; using
the original condition would directly leak the target. Representative held-out
AUC values from `r_molecule` were:

| Functional group | AUC |
|---|---:|
| Alcohol hydroxyl | 0.940 |
| Amide | 0.885 |
| Nitrile | 0.834 |
| Ketone | 0.781 |
| Ether | 0.772 |

These probe values are pipeline checks from a 32-dimensional model trained for
only eight updates. They do not establish scientifically useful latent
directions. A valid steering study requires the official checkpoint or a
substantially trained model, larger splits, repeated seeds, descriptor
baselines, and intervention-time generation metrics.

## Repository contents

- `results/experiment_summary.json`: consolidated machine-readable results.
- `results/latent_probe_summary.csv`: selected leakage-controlled probe results.
- `docs/EXPERIMENT_NOTES.md`: methods, interpretation, and limitations.
- `docs/CLOUD_GPU_RUNBOOK.md`: recommended official-checkpoint workflow.
- `provenance/SHA256SUMS.txt`: hashes for locally used data and generated cache.

Large datasets, model checkpoints, virtual environments, paper PDFs, and the
full reproduction pack are deliberately excluded.

## Upstream

The experiments are based on the public InertialAR repository:
https://github.com/HaoruiLi46/InertialAR

