# CCSN 11-Class Genus Hyperparameter Tuning and Production Evaluation

## 1. Overview & Protocol Isolation

This document records the empirical results, winning hyperparameters, cryptographic checksums, and verification steps for **ResNet-18** trained on the 11-class CCSN genus taxonomy (`metadata/splits/ccsn_11class_canonical.json`).

### Strict Data Partitioning & Isolation
- **Dataset**: CCSN 11-class canonical holdout split (`metadata/splits/ccsn_11class_canonical.json`).
- **Partitions**:
  - **Train**: 1,622 images (strictly isolated).
  - **Validation**: 407 images (used for model selection driven by minimum validation loss).
  - **Test**: 508 images (strictly isolated; zero access during hyperparameter tuning or multi-seed training; evaluated only on frozen checkpoints).
- **Taxonomy (11 Genera)**:
  `Ac` (Altocumulus), `As` (Altostratus), `Cb` (Cumulonimbus), `Cc` (Cirrocumulus), `Ci` (Cirrus), `Cs` (Cirrostratus), `Ct` (Contraila), `Cu` (Cumulus), `Ns` (Nimbostratus), `Sc` (Stratocumulus), `St` (Stratus).

> [!NOTE]
> **Note**: These re-audited metrics supersede an earlier contaminated-structure run.

---

## 2. Hyperparameter Tuning Exploration (10-Trial Sweep)

A 10-trial sweep was executed on `metadata/splits/ccsn_11class_canonical.json` across 60 epochs per trial on Seed 42.

- **Selection Metric**: Minimum validation cross-entropy loss (unsmoothed NLL tiebreaker).
- **Winner**: `trial_07_low_weight_decay` (`AdamW + Low Weight Decay (1e-3)`).
- **Winning TOML**: [`config/training/tuned_resnet18_ccsn11.toml`](../config/training/tuned_resnet18_ccsn11.toml)
- **Tuning Trajectory JSON**: [`artifacts/tuning/tuning_results_resnet18_ccsn11.json`](../artifacts/tuning/tuning_results_resnet18_ccsn11.json)

### Sweep Results Table

| Trial ID | Configuration Name | Optimizer | LR Backbone | LR Head | Weight Decay | Label Smoothing | Best Val Loss | Val Top-1 (%) | Val Top-3 (%) | Val Macro-F1 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `trial_07_low_weight_decay` | AdamW + Low Weight Decay (1e-3) **(Winner)** | AdamW | 5e-05 | 5e-04 | 1e-03 | 0.10 | **1.7078** | **48.89%** | **80.34%** | **46.61%** |
| `trial_10_optimal_differential` | Fine Differential LR (4e-5 / 4e-4, LS 0.05) | AdamW | 4e-05 | 4e-04 | 1e-02 | 0.05 | 1.6219 | 48.40% | 80.59% | 46.91% |
| `trial_05_sgd_conservative` | SGD with Momentum (Low LR) | SGD | 1e-04 | 1e-03 | 1e-04 | 0.10 | 1.7459 | 46.93% | 77.89% | 45.09% |
| `trial_02_conservative_adamw` | Conservative AdamW (Low LR) | AdamW | 2e-05 | 2e-04 | 1e-02 | 0.10 | 1.7317 | 44.96% | 79.61% | 42.75% |
| `trial_09_heavy_dropout` | AdamW + Heavy Dropout (0.4/0.5) | AdamW | 5e-05 | 5e-04 | 1e-02 | 0.10 | 1.7455 | 47.17% | 80.84% | 45.12% |
| `trial_06_high_weight_decay` | AdamW + High Weight Decay (5e-2) | AdamW | 5e-05 | 5e-04 | 5e-02 | 0.10 | 1.7338 | 43.73% | 79.61% | 40.54% |
| `trial_08_no_label_smoothing` | AdamW + No Label Smoothing (0.0) | AdamW | 5e-05 | 5e-04 | 1e-02 | 0.00 | 1.5531 | 44.72% | 81.08% | 41.09% |
| `trial_04_sgd_momentum` | SGD with Momentum (Std LR) | SGD | 5e-04 | 5e-03 | 1e-04 | 0.10 | 1.7654 | 46.44% | 78.87% | 45.22% |
| `trial_01_baseline_adamw` | Baseline AdamW (Std LRs) | AdamW | 5e-05 | 5e-04 | 1e-02 | 0.10 | 1.7507 | 44.96% | 79.61% | 41.32% |
| `trial_03_aggressive_adamw` | Aggressive AdamW (High LR) | AdamW | 1e-04 | 1e-03 | 1e-02 | 0.10 | 1.7824 | 45.21% | 79.61% | 43.56% |

---

## 3. Final Multi-Seed Test Performance (Held-out Test Partition, N=508)

Using the winning configuration from Trial 07, ResNet-18 was trained across three independent seeds (`42, 43, 44`) for 60 epochs each. Each model restored the minimum validation loss state dict.

### Performance Summary Table

| Metric | 3-Seed Mean +/- SD | Seed 42 | Seed 43 | Seed 44 |
| :--- | :---: | :---: | :---: | :---: |
| **Top-1 Accuracy** | **49.34% +/- 2.53%** | 46.46% | 51.18% | 50.39% |
| **Top-3 Accuracy** | **78.48% +/- 0.69%** | 77.76% | 79.13% | 78.54% |
| **Macro-F1** | **45.82% +/- 3.08%** | 42.27% | 47.72% | 47.47% |

---

## 4. Per-Genus Sensitivity / Recall Breakdown

| Genus Code | Cloud Genus Name | Test Count | Recall (3-Seed Mean +/- SD) | Seed 42 | Seed 43 | Seed 44 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `Ac` | Altocumulus | 43 | **27.13% +/- 3.55%** | 27.9% | 23.3% | 30.2% |
| `As` | Altostratus | 37 | **22.52% +/- 1.56%** | 21.6% | 24.3% | 21.6% |
| `Cb` | Cumulonimbus | 49 | **79.59% +/- 7.07%** | 71.4% | 83.7% | 83.7% |
| `Cc` | Cirrocumulus | 54 | **48.15% +/- 6.68%** | 42.6% | 46.3% | 55.6% |
| `Ci` | Cirrus | 28 | **55.95% +/- 10.31%** | 50.0% | 67.9% | 50.0% |
| `Cs` | Cirrostratus | 57 | **32.75% +/- 11.94%** | 19.3% | 36.8% | 42.1% |
| `Ct` | Contraila | 40 | **93.33% +/- 1.44%** | 92.5% | 95.0% | 92.5% |
| `Cu` | Cumulus | 37 | **45.95% +/- 7.15%** | 51.4% | 48.6% | 37.8% |
| `Ns` | Nimbostratus | 55 | **54.55% +/- 7.27%** | 47.3% | 61.8% | 54.5% |
| `Sc` | Stratocumulus | 68 | **65.20% +/- 7.25%** | 73.5% | 61.8% | 60.3% |
| `St` | Stratus | 40 | **6.67% +/- 3.82%** | 2.5% | 7.5% | 10.0% |

---

## 5. Cryptographic Checksums & Artifact Inventory

All generated artifacts are tracked under version control for reproducible evaluation:

| Relative Path | Size | Description | SHA-256 Checksum |
| :--- | :---: | :--- | :--- |
| `artifacts/ccsn_genus/resnet18_ccsn11_seed42.pth` | 45.3 MB | Frozen ResNet-18 weights (Seed 42) | `946813edbc94fa5500415fc4f30f17b293a3ba3e804fd6f73ce143346f157555` |
| `artifacts/ccsn_genus/resnet18_ccsn11_seed43.pth` | 45.3 MB | Frozen ResNet-18 weights (Seed 43) | `f03e34e8586dd601a874050f31222c915eed5c4d518de10e122811a4c53f25e0` |
| `artifacts/ccsn_genus/resnet18_ccsn11_seed44.pth` | 45.3 MB | Frozen ResNet-18 weights (Seed 44) | `969306efa3b3d6fa6c5e352788b5a64cb9aeb8db61860ba2a2511ba8731c01e5` |
| `artifacts/ccsn_genus/predictions_seed42.npz` | 35.6 KB | Raw test logits & predictions (Seed 42) | `0839c54d81c39e60000e3d2a68a38508f59864a366e78c785e90d9fa92d42506` |
| `artifacts/ccsn_genus/predictions_seed43.npz` | 35.6 KB | Raw test logits & predictions (Seed 43) | `24655b388cab2beb691c73288e762bd85f809d3bafbe8a9e91f28af6bb31187a` |
| `artifacts/ccsn_genus/predictions_seed44.npz` | 35.7 KB | Raw test logits & predictions (Seed 44) | `619328c67182010c5d20c15cfd89c4b883d9e852c6990b34a8d24056581c308d` |
| `artifacts/ccsn_genus/cm_ccsn11_resnet18.png` | 337 KB | 11x11 Confusion Matrix Figure | `bfcd3af3ea8c0400330326eac3e9252d650a79d7d1d821e47f1cef39d6dab88e` |
| `artifacts/ccsn_genus/ccsn11_evaluation_summary.json` | 26.9 KB | Machine-readable metrics & matrices | `8ad4919b40591a49fa09cb6429362eea603bab53fcbcb3eb94f49498016a30e0` |
| `artifacts/ccsn_genus/README.md` | 5.9 KB | Benchmark summary report | `504b98e1e60bfa085d3e53288bfd455f0d3c226d1a51caf8807155e4b8bc86d9` |

> [!TIP]
> **Checksum Regeneration Note**: To verify or regenerate the full SHA-256 inventory across all artifacts without drift, run:
> `python -c "import hashlib, pathlib; [print(f'{f.name}: {hashlib.sha256(f.read_bytes()).hexdigest()}') for f in sorted(pathlib.Path('artifacts/ccsn_genus').iterdir()) if f.is_file()]"`

---

## 6. Exact Reproduction Commands

```powershell
# 1. Activate Python environment
.\.venv\Scripts\activate

# 2. Run 10-trial hyperparameter sweep (Seed 42, 60 epochs per trial)
python src/tune_resnet_family.py --model resnet18 --taxonomy ccsn11 --seed 42 --epochs 60

# 3. Train final models across 3 seeds (strictly isolated from test partition)
python scripts/train_ccsn_genus.py --config config/training/tuned_resnet18_ccsn11.toml --seeds 42 43 44

# 4. Evaluate frozen checkpoints on held-out test split
python scripts/eval_ccsn_genus.py --config config/training/tuned_resnet18_ccsn11.toml --seeds 42 43 44

# 5. Alternatively, run unified orchestrator
python scripts/run_ccsn_genus_evaluation.py --skip-train
```
