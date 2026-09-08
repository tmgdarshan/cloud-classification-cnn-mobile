# Cross-Source Ground-Based Cloud Classification with ResNets

Benchmark code for classifying ground-based cloud images across two sensor types
that see the sky very differently: CCSN (regional camera views) and GCD
(wide-angle whole-sky views).

## Background

This started as a learning project: build, train, and validate a CNN for cloud
classification from scratch in a modern framework, and practice reproducible
research. Working with the public datasets surfaced a
concrete problem, byte-identical duplicate images crossing the official
train/test splits, and the project shifted into a small methods study:
audit the leakage, build leakage-free grouped splits, harmonize the two datasets'
label schemes, and measure what joint training and architecture depth actually
buy once the comparison is fair.

## What the project does

- **CCSN** (Zhang et al., 2018): regional cloud photos, 2,337 images in the
  five-class benchmark.
- **GCD** (Liu et al., 2022): whole-sky camera images, 14,306 images in the
  five-class benchmark.

The two datasets are combined into five shared classes (Cumulus, Altocumulus,
Cirrus, Stratocumulus, Cumulonimbus), with three deliberate steps:

1. **Duplicate removal and grouped splitting.** Public releases contain
   byte-identical repeat exposures that straddle the official train/test split
   (156 clusters in GCD; 3 label-conflicting pairs in CCSN, which are dropped).
   `StratifiedGroupKFold` (seed 42) keeps each duplicate cluster in one split and
   defines an 80% development pool and an untouched 20% test set.
2. **Shared five-class labels.** Fine WMO genera and coarse whole-sky categories
   are mapped into a common label space; clear sky, contrails, and mixed scenes
   are excluded.
3. **Balanced batch sampling.** GCD is ~86% of the training data, so joint
   training samples both sources equally per batch on average.

Three architectures are compared (ResNet-18/34/50), each trained with three seeds
{42, 43, 44}. Numbers in the paper are the mean ± standard deviation across seeds.

## Main results

- **Architecture depth barely matters.** Joint models reach ~74% source-balanced
  accuracy for all three (74.1 / 73.7 / 74.2%), within one seed standard
  deviation of each other. ResNet-18 is the practical choice: 53% fewer
  parameters, 56% fewer FLOPs.
- **CCSN in-domain is hard: ~57–62% under every training configuration.** The
  468-image regional test component stays difficult regardless of data mixture
  or training budget.
- **The apparent joint-training gain on CCSN is mostly a training-budget
  effect.** Once the CCSN-only baseline gets the same number of optimizer steps
  as the joint model, the ResNet-18 advantage disappears (−0.2 ± 1.1 pp);
  ResNet-34 keeps a modest +2.4 ± 1.0 pp.
- **Cross-source transfer is lossy both ways** (~35–41% vs ~90% in-domain), with
  CCSN→GCD consistently a few points above GCD→CCSN.

Full tables, confusion matrices, and the 36-run matrix live in the manuscript and
in `artifacts/`, which are kept locally and not tracked in this repository.

## Layout

```
config/     TOML config (composition model; see config/README.md)
docs/       PROJECT_GUIDE, DECISIONS, CHANGELOG, KNOWN_ISSUES, architecture notes
metadata/   Dataset inventories and the canonical split manifests (metadata/splits/)
scripts/    Manifest builder, dataset/config inspection CLIs, figure regeneration
src/        run_harmonized.py (benchmark runner), tune_resnet_family.py (tuning),
            evaluation.py, plus the split/manifest/config infrastructure
tests/      Unit and integration tests

report/ and artifacts/ (manuscript, PDF, figures, multi-seed results, tuning
logs, predictions) are produced locally by the code above and are gitignored.
```

## Running it

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest                                     # 127 tests

# hyperparameter tuning on the development pool
python src/tune_resnet_family.py --model all --epochs 5

# a benchmark run (single-source baselines, transfer, joint)
python src/run_harmonized.py --model resnet18 --config config/training/tuned_resnet18.toml --epochs 15

# regenerate the report figures
python scripts/plot_comparative_convergence.py
python scripts/regenerate_confusion_matrices.py
```

Datasets are not in Git. Point `CLOUD_DATA_ROOT` at a directory containing the
`CCSN/` and `GCD/` image folders. Benchmark runs need a CUDA GPU.

## References

- J. Zhang et al., "CloudNet: Ground-based cloud classification with deep
  convolutional neural networks," *Geophys. Res. Lett.*, 2018.
- S. Liu et al., "Ground-based remote sensing cloud classification via context
  graph attention network," *IEEE TGRS*, 2022.
- K. He et al., "Deep residual learning for image recognition," *CVPR*, 2016.
