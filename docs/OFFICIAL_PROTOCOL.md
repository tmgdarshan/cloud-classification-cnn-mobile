# Official Research Protocol: Cross-Source Cloud Classification

This document is the authoritative, evidence-bound specification for the cross-source cloud classification pipeline. All source code, configurations, and documentation reference this protocol.

---

## 1. Datasets and Optical Modalities

The study evaluates two ground-based cloud image sources:
1. **CCSN (Cirrus Cumulus Stratus Nimbus)**: Ground-based cloud image dataset captured with digital cameras focusing on regional cloud features and textures (Zhang et al., 2018; 2,537 verified images in the 11-class pool; 2,337 images in the 5-class harmonized benchmark after contrail exclusion).
2. **GCD (Ground-based Cloud Dataset)**: Ground-based all-sky cloud observations providing a wide field-of-view whole-sky perspective, collected across nine provinces in China (Liu et al., 2022; 14,306 cloud images in the 5-class harmonized benchmark after excluding clear-sky and mixed classes).

Source configurations and archival provenance are defined in [`config/datasets/ccsn.toml`](../config/datasets/ccsn.toml) and [`config/datasets/gcd.toml`](../config/datasets/gcd.toml).

---

## 2. Exclusions and Data Sanitization

To ensure empirical validity and address label ambiguity:
- **Conflicting Duplicate Pairs Purged**: In CCSN, 3 conflicting duplicate pairs (6 images) with identical SHA-256 file-byte hashes but contradictory labels (`Ac` vs. `As` twice, `Cc` vs. `Cs` once) were excised, leaving 2,537 verified images in the 11-class pool.
- **Exclusions for the Shared Classification Task**:
  - **CCSN `Ct` (contrails, 200 images)**: Anthropogenic, human-generated cloud category excised to focus on natural cloud discrimination, leaving 2,337 CCSN images in the five-class compatibility space.
  - **GCD `7_mixed` (955 images)**: Heterogeneous mixed cloud scenes excised because they combine multiple cloud classes without a single dominant meteorological category.
  - **GCD `4_clearsky` (3,739 images)**: Cloud-free images (allowing $\le 10\%$ cloudiness per GCD authors) excised to focus strictly on cloud-type discrimination and align with CCSN (which contains zero clear-sky images), leaving 14,306 GCD images in the five-class compatibility space.

---

## 3. Five-Class Compatibility Taxonomy

The shared classification space is an operational **five-class compatibility taxonomy**, not a mathematically complete or genus-preserving WMO taxonomy. It groups fine telephoto genera and operational whole-sky categories into five shared cloud bins:

| Harmonized Class | CCSN Genera Included | GCD Categories Included | Physical Cloud Morphology |
| :--- | :--- | :--- | :--- |
| **Cumulus** | `Cu` | `1_cumulus` | Detached, dense convective clouds with sharp outlines and flat bases. |
| **Altocumulus** | `Ac`, `Cc` | `2_altocumulus` | Mid-tropospheric rolls and high-level ice ripples showing regular patches. |
| **Cirrus** | `Ci`, `Cs` | `3_cirrus` | High-altitude detached fibrous ice filaments and veil-like sheets. |
| **Stratocumulus** | `Sc`, `St`, `As` | `5_stratocumulus` | Continuous low stratiform decks and dense non-precipitating sheets. |
| **Cumulonimbus** | `Cb`, `Ns` | `6_cumulonimbus` | Deep convective towers and thick precipitation-bearing cloud layers. |

*Note: Grouping coarse operational categories into compatibility bins involves intentional meteorological compromises necessitated by GCD's coarse whole-sky label scheme. In particular, mapping Nimbostratus (`Ns`) to Cumulonimbus and Altostratus (`As`) to Stratocumulus are operational compatibility choices rather than clean WMO genus equivalences. While $Ns$ and $Cb$ share dark precipitation-bearing bases, their convective dynamics differ fundamentally; likewise, $As$ is a mid-tropospheric layer whereas $Sc$ is a low stratiform deck.*

**Authoritative Machine Definition**: [`metadata/splits/harmonized_5bin_canonical.json`](../metadata/splits/harmonized_5bin_canonical.json).  
**Human Decision Record**: Decision D-007 in [`docs/DECISIONS.md`](DECISIONS.md).

---

## 4. Development and Test Split Protocol

To mitigate cross-split duplicate leakage, all partitions treat exact-byte SHA-256 duplicate clusters as indivisible atomic units (note that exact-byte grouping does not identify near-duplicate exposures with sensor noise or varying lighting):

- **Partitioning Method**: `StratifiedGroupKFold` (Canonical Seed 42).
- **Canonical Harmonized Test Partition (20%, $N=3,330$ images)**: Isolated once into an immutable evaluation partition, reserved exclusively for final benchmark evaluation without test-set peeking. Decomposes into:
  - **Harmonized CCSN Test Component ($n=468$)**: Unseen test partition of regional camera imagery.
  - **Harmonized GCD Test Component ($n=2,862$)**: Unseen test partition of whole-sky camera imagery.
  - **Pooled Test Partition ($n=3,330$)**: Sample-concatenated evaluation partition (GCD-heavy, 85.95% GCD).
- **Development Pool (80%, $N=13,313$ images)**: Used exclusively for training and checkpoint selection:
  - **Parameter Training (64%, $N=10,650$)**: Backpropagation weight optimization (1,495 CCSN, 9,155 GCD).
  - **Internal Development Validation Partition (16%, $N=2,663$)**: Internal fold-0 partition used for minimum-validation-loss checkpoint selection and hyperparameter assessment; does not constitute final evidence.

### Methodological Notes on Sampling & Training Budgets
1. **Validation Loss Weighting**: While joint training samples each source with approximately 50% probability in expectation, the validation split is sample-averaged over 374 CCSN images and 2,289 GCD images ($L_{\text{val}} \approx 0.1404 L_{\text{CCSN}} + 0.8596 L_{\text{GCD}}$). Checkpoint selection is therefore inherently weighted towards GCD validation loss.
2. **Training Exposure & Update Budget**: Joint training executes 167 batches per epoch (batch size 64) with 10,650 draws with replacement per epoch (~5,325 CCSN draws in expectation, averaging 3.56 exposures per CCSN training image, compared to 1 draw per image across 24 batches in CCSN-only training). The observed CCSN test component accuracy gain (+4.06 percentage points on ResNet-18) represents the complete joint recipe (expanded update budget, source-balanced draws, and cross-source features) rather than isolated feature transfer alone.

---

## 5. Model Architecture and Selected Baseline Configuration

- **Headline Model**: **ResNet-18** (11,309,637 parameters, ~11.3M; 1.82 GFLOPs) provides the practical operational baseline, achieving **74.58% source-balanced accuracy** on the final test partition, within 0.08 percentage points of ResNet-50 (**74.66%**) while requiring 53% fewer parameters. ResNet-18 is framed as the practical selected baseline for resource-constrained deployment without claiming formal statistical equivalence.
- **Capacity Sensitivity Checks**: **ResNet-34** (21,417,797 parameters, ~21.4M; 3.66 GFLOPs) and **ResNet-50** (24,034,373 parameters, ~24.0M; 4.12 GFLOPs) evaluate whether additional depth aids cross-source transfer.
- **Classifier Head**: Small regularized classification head replacing the ImageNet 1,000-class layer:
  `Dropout(0.3) -> Linear(d_in, 256) -> BatchNorm1d -> GELU -> Dropout(0.2) -> Linear(256, 5)`.  
  *(Both dropout layers are situated within the replacement classification head).*
- **Selected Optimization Baseline** (evaluated on the development pool):
  - Optimizer: AdamW with decoupled weight decay ($\lambda = 1\times 10^{-2}$).
  - Differential Learning Rates: $\eta_{\text{backbone}} = 5\times 10^{-5}$, $\eta_{\text{head}} = 5\times 10^{-4}$.
  - Learning Rate Schedule: Cosine Annealing to $\eta_{\text{min}} = 10^{-6}$ over 15 epochs.
  - Label Smoothing: $\alpha = 0.0$ (hard cross-entropy loss).
  - Batch Size: 64.
  - Checkpoint Rule: Fixed 15-epoch budget restoring the minimum validation-loss snapshot.
- **Physically Conservative Augmentation**:
  - Horizontal flip ($p = 0.5$).
  - Bounded rotation ($\pm 15^\circ$).
  - Random resized crop (scale $0.8$ to $1.0$).
  - Mild photometric jitter ($\pm 10\%$).
  - **Vertical Flip Strictly Prohibited**: Ground-based cloud imagery possesses meaningful vertical orientation (condensation bases at bottom, buoyant tops upward).
- **Source-Balanced Batch Sampling**: In the joint CCSN+GCD model, `WeightedRandomSampler` with inverse source-size weights enforces equal contribution in expectation between CCSN and GCD during training, preventing GCD volume dominance (85.96% GCD vs. 14.04% CCSN).

---

## 6. Evaluation Metrics and Statistical Uncertainty

- **Primary Headline Metric**: **Source-Balanced Average** (unweighted arithmetic mean of CCSN test component and GCD test component accuracies), preventing the 6:1 GCD sample preponderance from dominating conclusions. Top-1 Accuracy, Balanced Accuracy (Macro Recall), and Macro-averaged F1-score are reported.
- **Evaluation Partitions**:
  1. *In-Domain Baselines*: CCSN Model on CCSN Test Component ($n=468$); GCD Model on GCD Test Component ($n=2,862$).
  2. *Cross-Source Transfer*: CCSN Model on GCD Test Component; GCD Model on CCSN Test Component.
  3. *Joint CCSN+GCD Model*: Evaluated on CCSN Test Component ($n=468$), GCD Test Component ($n=2,862$), and Pooled Test Partition ($n=3,330$, secondary).
- **Statistical Uncertainty**: 95% non-parametric percentile bootstrap confidence intervals ($B = 1,000$) computed conditionally at the image level on the final test partition. (Note: image-level bootstrap intervals do not capture training-seed or session-level variance).
- **Diagnostic Transfer Asymmetry**: CCSN models transfer moderately to GCD (57.86% on ResNet-18) whereas GCD models transfer poorly to CCSN (37.18%), driven by an extreme collapse in Cumulonimbus recall ($\Delta = -56.7$ pp, 75.9% down to 19.2%). This asymmetry reflects multiple confounding domain shifts between the two datasets (including optical characteristics, field of view, spatial resolution, and class label distributions), none of which are isolated by cross-source evaluation alone.

---

## 7. Execution Commands

All production benchmarks run via the unified production engine:

```powershell
# Activate environment
.venv\Scripts\activate

# ResNet-18 Headline Benchmark
python src/run_harmonized.py --model resnet18 --config config/training/tuned_resnet18.toml --epochs 15

# ResNet-34 Capacity Check
python src/run_harmonized.py --model resnet34 --config config/training/tuned_resnet34.toml --epochs 15

# ResNet-50 Capacity Check
python src/run_harmonized.py --model resnet50 --config config/training/tuned_resnet50.toml --epochs 15

# Comparative Convergence Plotting
python scripts/plot_comparative_convergence.py
```
