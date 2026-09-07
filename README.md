# Cross-Source Ground-Based Cloud Classification with Deep Residual Networks

This repository provides an empirical benchmarking framework for ground-based cloud classification across heterogeneous optical sensors using deep residual convolutional neural networks (ResNet-18, ResNet-34, and ResNet-50).

---

## 1. Overview

Ground-based meteorological cloud classification faces severe domain shift between different sensor geometries:
- **CCSN (Cirrus Cumulus Stratus Nimbus)**: Ground-based cloud image dataset captured with digital cameras focusing on regional cloud features and textures (Zhang et al., 2018; 2,537 images in 11-class view; 2,337 in 5-class harmonized benchmark).
- **GCD (Ground-based Cloud Dataset)**: Ground-based all-sky cloud observations providing a wide field-of-view whole-sky perspective, collected across nine Chinese provinces (Liu et al., 2022; 14,306 cloud images in 5-class harmonized benchmark).

To evaluate cross-source transfer and joint training with empirical rigor, the framework:
1. **Mitigates Exact-Byte Duplicate Leakage**: Audits duplicate captures using SHA-256 file-byte hashing and purges 3 conflicting duplicate pairs (6 images) in CCSN. Binds exact duplicate clusters into atomic units via `StratifiedGroupKFold` (Seed 42), establishing an **80% development pool** (with internal validation for checkpoint selection) and an immutable **20% final test holdout**.
2. **Defines a Five-Class Compatibility Taxonomy**: Maps fine telephoto genera and operational whole-sky categories into five shared cloud classes (Cumulus, Altocumulus, Cirrus, Stratocumulus, Cumulonimbus), excising non-cloud scenes (clear sky) and anthropogenic or heterogeneous categories (contrails, mixed clouds).
3. **Applies Physically Conservative Augmentation**: Eliminates vertical flipping, as ground-based cloud images possess a meaningful vertical orientation (condensation bases at the bottom, buoyant vertical growth upward).
4. **Counteracts Sample Volume Imbalance**: Enforces **source-balanced batch sampling** (equal contribution in expectation between CCSN and GCD) to prevent gradient starvation on the smaller CCSN dataset (85.96% GCD vs. 14.04% CCSN).
5. **Establishes ResNet Baselines**: Uses **ResNet-18** as the headline operational model and **ResNet-34 / ResNet-50** as capacity sensitivity checks, optimized with a selected baseline configuration (differential learning rates, AdamW, cosine annealing).

For complete scientific specifications, see the [Official Research Protocol](docs/OFFICIAL_PROTOCOL.md) and [Methodological Decision Records](docs/DECISIONS.md).

---

## 2. Repository Layout

```text
├── artifacts/              # Model checkpoints, metric summaries, and confusion matrices
├── config/                 # Configuration files
│   ├── datasets/           # Dataset profiles (ccsn, gcd, gcd_5class, gcd_6class, harmonized_5bin)
│   └── training/           # Selected training configurations (tuned_resnet18/34/50.toml)
├── docs/                   # Documentation and project records
│   ├── PROJECT_GUIDE.md    # Orientation, standards, source-of-truth map, history
│   ├── OFFICIAL_PROTOCOL.md# Authoritative research protocol and methodology
│   ├── DECISIONS.md        # Methodological decision records (D-007 to D-012)
│   ├── KNOWN_ISSUES.md     # Observed limitations and audit history
│   └── audits/             # Expert review memos and audit trails
├── metadata/               # Inventories and split definitions
│   └── splits/             # Canonical group-aware JSON manifests (single source of truth)
├── report/                 # Academic publication manuscript (LaTeX and compiled PDF)
├── scripts/                # Utility scripts
│   ├── build_canonical_manifests.py       # Deterministic manifest builder
│   ├── discover_dataset.py                # Dataset inventory CLI
│   ├── show_config.py                     # Modular config inspection CLI
│   ├── plot_comparative_convergence.py    # ResNet-family convergence plot
│   └── regenerate_confusion_matrices.py   # 3-seed confusion matrices + saved predictions
├── src/                    # Production codebase
│   ├── run_harmonized.py                  # Primary benchmark runner (baselines, transfer, joint)
│   ├── tune_resnet_family.py              # Hyperparameter exploration & convergence engine
│   ├── evaluation.py                      # Classification metrics & bootstrap CIs
│   ├── experiment_registry.py             # Provenance registry
│   ├── training_state.py                  # CPU state-dict snapshot utilities
│   └── dataset_*/split_*/config_loader.py # Dataset, split, and config infrastructure
└── tests/                  # Automated unit and integration test suite
```

---

## 3. Quickstart & Reproducibility

### Environment Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Run Automated Tests

```powershell
pytest tests/ -v
```

### Hyperparameter Exploration (Development Pool)

```powershell
python src/tune_resnet_family.py --model all --epochs 5
```

### Official Benchmark Runs (15 Epochs)

The primary execution engine (`src/run_harmonized.py`) evaluates single-source baselines, cross-source transfer, and the joint CCSN+GCD model:

```powershell
# ResNet-18 Headline Benchmark
python src/run_harmonized.py --model resnet18 --config config/training/tuned_resnet18.toml --epochs 15

# ResNet-34 Capacity Check
python src/run_harmonized.py --model resnet34 --config config/training/tuned_resnet34.toml --epochs 15

# ResNet-50 Capacity Check
python src/run_harmonized.py --model resnet50 --config config/training/tuned_resnet50.toml --epochs 15
```

### Plot Convergence Trajectories

```powershell
python scripts/plot_comparative_convergence.py
```

---

## 4. Benchmark Results Summary

Evaluated on the **canonical harmonized final test partition** ($N=3,330$ images: CCSN test component $n=468$, GCD test component $n=2,862$) with 95% conditional image-level percentile bootstrap confidence intervals ($B=1,000$):

| Architecture | Parameters (FLOPs) | CCSN In-Domain ($n=468$) | CCSN $\to$ GCD Transfer ($n=2,862$) | GCD In-Domain ($n=2,862$) | GCD $\to$ CCSN Transfer ($n=468$) | Joint on CCSN ($n=468$) | Joint on GCD ($n=2,862$) | **Source-Balanced Avg (Headline)** | Pooled Test ($n=3,330$) |
|---|---|---|---|---|---|---|---|---|---|
| **ResNet-18** | 11.3M (1.82G) | 56.84% | 57.86% | 89.34% | 37.18% | **60.90%** | 88.26% | **74.58%** | 84.41% |
| **ResNet-34** | 21.4M (3.66G) | 57.91% | 30.68% | 89.06% | 34.83% | **60.68%** | 88.36% | **74.52%** | 84.47% |
| **ResNet-50** | 24.0M (4.12G) | 59.19% | 43.99% | 89.45% | 33.76% | **60.26%** | 89.06% | **74.66%** | **85.02%** |

### Key Findings & Framing
- **Headline Source-Balanced Accuracy**: The headline metric is the joint model evaluated separately on the CCSN and GCD test components and averaged across sources:
  - **ResNet-18 joint model**: **74.58%** source-balanced accuracy (Balanced Acc: 73.77%, Macro-F1: 74.03%).
  - **ResNet-50 joint model**: **74.66%** source-balanced accuracy (Balanced Acc: 73.44%, Macro-F1: 73.92%).
- **ResNet-18 as Practical Selected Baseline**: Because the gap between ResNet-18 and ResNet-50 is only **0.08 percentage points** (with ResNet-18 achieving higher balanced accuracy: 73.77% vs. 73.44%), ResNet-18 is framed as the practical selected baseline for resource-constrained edge deployment due to its 53% parameter reduction (11.3M vs. 24.0M) and 56% FLOP reduction (1.82 vs. 4.12 GFLOPs), while explicitly avoiding a formal claim of statistical equivalence.
- **Partition Terminology & Fairness**:
  - **CCSN Test Component ($n=468$)**: Harmonized regional camera component of the final test partition.
  - **GCD Test Component ($n=2,862$)**: Harmonized whole-sky camera component of the final test partition.
  - **Pooled Test Partition ($n=3,330$)**: Combined test partition. While valid (84.41% on ResNet-18 vs. 85.02% on ResNet-50), it is GCD-heavy (85.95% GCD), making source-balanced averaging the fairer headline metric to prevent GCD sample dominance.
  - **Internal Validation Partition ($n=2,663$)**: Development-only partition used for checkpoint selection, strictly partitioned from final test evidence.
- **Joint Training Improvement**: Joint training with source-balanced batch sampling improved accuracy on the sparse CCSN test component by **+4.06 percentage points** on ResNet-18 (56.84% to 60.90%).
- **Diagnostic Transfer Asymmetry**: Regional camera models transfer moderately to whole-sky views (57.86% on ResNet-18), whereas whole-sky models transfer poorly to regional views (37.18%), driven by an extreme collapse in Cumulonimbus recall ($\Delta = -56.7$ pp, 75.9% down to 19.2%).

*Detailed per-class metrics, confusion matrices, and bootstrap confidence intervals are archived in `artifacts/harmonized_results/`.*
