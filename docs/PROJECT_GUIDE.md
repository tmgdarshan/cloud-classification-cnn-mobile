# Project Guide

Orientation for anyone (human or tool) working in this repository. Read this
first, then the file(s) relevant to the task. The repository is the single
source of truth; this guide points at where that truth lives.

---

## 1. Purpose and scope

`cloud-classification-cnn-mobile` supports **MSc thesis research in Atmospheric
Science**: classifying ground-based cloud images with the ResNet family
(ResNet-18/34/50, ImageNet-pretrained) across two heterogeneous sensors.

**In scope / demonstrated:** exact-byte duplicate auditing and group-aware
splitting; a five-class cross-source compatibility taxonomy; source-specific
in-domain baselines; cross-source transfer in both directions; a joint
CCSN+GCD model with source-balanced sampling; bootstrap uncertainty and
confusion matrices.

**Out of scope / future work:** mobile deployment, distillation, INT8
quantization, on-device latency, and SLURM/HPC execution. The repository name
anticipates this; none of it is a current deliverable.

This repository holds **source code, documentation, configuration, and
metadata only.** Datasets and trained model weights live outside Git.

Preferred public terms: *five-class compatibility taxonomy*, *cross-source
transfer*, *joint CCSN+GCD model*, *development / final-test split*, *selected
baseline configuration*.

---

## 2. Development standards

- **Repository is the source of truth.** Governance, decisions, and project
  state live in these documents, not in anyone's memory or a chat log.
- **The researcher is the final authority** on all scientific matters:
  methodology, experiment design, dataset selection, interpretation,
  conclusions. Anything touching scientific behaviour — data splits,
  normalization, class definitions, metrics, architecture, hyperparameters —
  is explained and approved before it changes. **Preserve existing scientific
  behaviour unless explicitly asked to change it.**
- **Config-driven, not edit-in-place.** Hyperparameters, paths, and device
  settings come from config files, never from editing constants in code.
- **Reproducibility over convenience.** Single recorded seed (Python, NumPy,
  PyTorch/CUDA); deterministic options for reported runs; a run records enough
  to regenerate it (resolved config, git commit, dataset checksums, library
  versions, seed).
- **No data or artifacts in Git.** No datasets (raw, processed, sampled,
  zipped), no checkpoints (`*.pth`/`*.pt`/`*.ckpt`), no metrics tables, logs,
  or figures. `.gitignore` is authoritative; never `git add -f` these.
- **Resolve the data location from config or `CLOUD_DATA_ROOT`** — never from a
  repo-local `data/` folder, `__file__`, or the working directory.
- **Offline-first.** The training path must not require network downloads;
  load pretrained weights from a local cache.
- **Small, reviewable changes.** Prefer the minimal diff. Explain changes that
  alter observable behaviour (outputs, defaults, preprocessing, config
  semantics) before making them.
- **Single source of truth in code** for normalization stats, class maps, and
  split logic — defined once, imported, never duplicated.
- **No hardcoded filesystem paths**, no import side effects (importing a module
  must not read data, download weights, or start training).
- **Commit or push only when asked.** Work on feature branches; never rewrite
  shared history without coordination.

---

## 3. Where the truth and implementation live

| Path | Role / when to read |
|---|---|
| `metadata/splits/*_canonical.json` | Canonical split assignments, classes, sample records, counts — machine truth for every current benchmark |
| `docs/OFFICIAL_PROTOCOL.md` | Authoritative, evidence-bound research protocol and methodology |
| `docs/DECISIONS.md` | Accepted scientific and methodological decisions (D-007…D-012) |
| `docs/KNOWN_ISSUES.md` | Accepted limitations and audit records (KI-001) |
| `README.md`, `docs/project_dashboard.md` | Public overview and current status |
| `docs/project_specification.md` | Approved scientific/functional requirements (researcher-maintained) |
| `src/run_harmonized.py` | Primary benchmark runner: training, checkpoint selection, transfer, joint evaluation |
| `src/tune_resnet_family.py` | Ten-trial family tuning and convergence outputs; development data only |
| `src/evaluation.py` | Metrics, confusion matrices, image-level bootstrap intervals |
| `src/training_state.py` | Detached, cloned CPU state-dict snapshots |
| `src/experiment_registry.py` | Experiment registry JSON, checkpoint digests, summary verification |
| `config/training/tuned_resnet{18,34,50}.toml` | Selected flat runner configurations |
| `config/datasets/harmonized_5bin.toml` | Pointer profile to the canonical harmonized manifest |
| `metadata/{ccsn,gcd}_inventory.json` | Observed local dataset counts, resolutions, exact-byte duplicates |
| `src/dataset_discovery.py`, `scripts/discover_dataset.py` | Inventory creation and SHA-256 duplicate discovery |
| `src/config_loader.py`, `scripts/show_config.py` | Modular config inspection/validation (separate from the flat runner config) |
| `src/dataset_index.py`, `dataset_loading.py`, `split_protocol.py`, `split_generator.py`, `split_manifest.py` | Generic tested dataset/split infrastructure; the runner loads canonical JSON directly |
| `scripts/build_canonical_manifests.py` | Deterministic manifest builder — writes canonical splits; do not casually run |
| `scripts/plot_comparative_convergence.py`, `plot_transfer_asymmetry.py` | Plot saved convergence / transfer artifacts |
| `artifacts/harmonized_results/harmonized_summary_resnet*.json` | Saved canonical benchmark aggregates |
| `report/cloud_classification_resnet.tex` + `.pdf` | Academic manuscript and compiled output |
| `docs/audits/` | Independent review memos |
| `tests/unit/`, `tests/integration/` | Contract tests plus synthetic dataset-discovery coverage |

Raw images are at `CCSN/CCSN_v2/` and `GCD/` under `CLOUD_DATA_ROOT` if set,
otherwise the repository root. Raw GCD has `train`/`test` folders; those
assignments are superseded by the canonical manifests. No directory of
copied/merged images is required.

**Do not** regenerate manifests, change partitions/taxonomy, or remove
validation without explicit approval for a new protocol.

---

## 4. Scientific facts to preserve

**CCSN** (*Cirrus Cumulus Stratus Nimbus*). Local raw count 2,543. Three
cross-class exact-duplicate pairs (six images: Ac/As twice, Cc/Cs once) with
identical file-byte SHA-256 are excluded, leaving **2,537** in the 11-class
view. Excluding 200 contrails (`Ct`, an anthropogenic category) leaves
**2,337** harmonized CCSN images.

**GCD** (*Ground-based Cloud Dataset*, TJNU). Local raw count 19,000. Exclude
`7_mixed` (955) for the **18,045**-image six-class view, then `4_clearsky`
(3,739, non-cloud) for the **14,306**-image five-class view. Historical audit:
159 exact-duplicate groups in raw GCD, 156 crossing the raw train/test
folders, concentrated in clear sky. Exact-byte duplication does not establish
capture time, camera identity, or scene independence.

| Compatibility class (canonical order) | CCSN labels | GCD label |
|---|---|---|
| cumulus | Cu | 1_cumulus |
| altocumulus | Ac, Cc | 2_altocumulus |
| cirrus | Ci, Cs | 3_cirrus |
| stratocumulus | Sc, St, As | 5_stratocumulus |
| cumulonimbus | Cb, Ns | 6_cumulonimbus |

This is an operational compatibility mapping, **not** a genus-preserving WMO
taxonomy.

**Split framing:** ~80% development / ~20% final test; validation is internal
to development. Nominal shares are approximate because exact-duplicate groups
are indivisible. The protocol is a **grouped stratified holdout** (`grouped_stratified_holdout`
v1.0): class-stratified, exact-duplicate clusters kept atomic, one fixed
partition at seed 42 (`StratifiedGroupKFold` derives it). Internal fold 0 is
validation, folds 1–4 are training, the test fold field is -1. **No
cross-validation experiment is run** — the fold fields exist only to carve out
fold-0 as the validation set.

**Identifiers vs prose:** the keys `harmonized_5bin`,
`harmonized_5bin_canonical.json`, and `config/datasets/harmonized_5bin.toml`
are frozen names; in prose the label space is always the *five-class
compatibility taxonomy*.

| Canonical view | Parameter training | Internal validation | Final test | Total |
|---|---:|---:|---:|---:|
| CCSN 11-class | 1,622 | 407 | 508 | 2,537 |
| GCD 6-class | 11,548 | 2,888 | 3,609 | 18,045 |
| GCD 5-class | 9,155 | 2,289 | 2,862 | 14,306 |
| Harmonized five-class | 10,650 | 2,663 | 3,330 | 16,643 |
| — harmonized CCSN component | 1,495 | 374 | 468 | 2,337 |
| — harmonized GCD component | 9,155 | 2,289 | 2,862 | 14,306 |

Harmonized development has 13,313 images. Do not substitute the standalone
CCSN 508-image test set for the harmonized CCSN 468-image test set.

---

## 5. Training, selection, evaluation

- 224-pixel inputs, ImageNet normalization. Head:
  `Dropout(0.3) -> Linear(d_in, 256) -> BatchNorm1d -> GELU -> Dropout(0.2) -> Linear(256, 5)`.
  Both dropout layers sit inside the replacement head; the backbone is a
  standard pretrained ResNet.
- All three selected TOMLs: AdamW, backbone LR `5e-5`, head LR `5e-4`, weight
  decay `0.01`, label smoothing `0.0`, batch 64, 15 epochs, seed 42, cosine
  schedule to `1e-6`.
- Augmentation: random resized crop scale 0.8–1.0, horizontal flip 0.5,
  rotation ±15°, mild colour jitter. **No vertical flipping** (ground-based
  cloud images have a meaningful vertical orientation).
- Joint training uses `WeightedRandomSampler` with inverse source-size
  weights: equal source contribution **in expectation**, not exactly half of
  each batch. Joint validation is sample-weighted over concatenated sources,
  so checkpoint selection is inherently weighted toward GCD validation loss
  (≈ 0.14·L_CCSN + 0.86·L_GCD).
- The runner selects the **minimum validation-loss** checkpoint over a fixed
  15-epoch budget (not early stopping). The tuner selects on common unsmoothed
  validation cross-entropy, breaking ties with validation macro-F1.
- Reported metrics: accuracy, balanced accuracy (macro recall), macro-F1;
  per-class metrics and confusion matrices saved. CIs: 1,000 image-level
  percentile bootstrap resamples (`seed=42`), conditional on the saved
  model/test set — they do not quantify training-seed or scene-level
  uncertainty. The **source-balanced average** is the arithmetic mean of the
  two source-specific metrics, distinct from pooled-test accuracy.

### Saved benchmark reference (accuracy %, existing artifacts — not reruns)

| Model | CCSN in-domain | CCSN→GCD | GCD in-domain | GCD→CCSN | Joint/CCSN | Joint/GCD | Joint source avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet-18 | 56.84 | 57.86 | 89.34 | 37.18 | 60.90 | 88.26 | 74.58 |
| ResNet-34 | 57.91 | 30.68 | 89.06 | 34.83 | 60.68 | 88.36 | 74.52 |
| ResNet-50 | 59.19 | 43.99 | 89.45 | 33.76 | 60.26 | 89.06 | 74.66 |

A 0.08-point gap between 74.58 and 74.66 is **percentage points**, not
evidence of statistical equivalence. ResNet-18 also has the higher joint
balanced accuracy (73.77 vs 73.44).

---

## 6. Project history

Plain history of the work, newest first. Detailed rationale is in
`docs/DECISIONS.md`; milestones in `docs/CHANGELOG.md`.

- **Phase 4 — Group-aware manifests, harmonized benchmark suite.** Remediated
  GCD train/test duplicate leakage (KI-001) in canonical manifests; built the
  four deterministic `metadata/splits/*_canonical.json`; approved the
  five-class compatibility taxonomy (D-007) and the group-aware split (D-008);
  prohibited vertical flips (D-010); consolidated the pipeline on
  `src/run_harmonized.py` + `src/tune_resnet_family.py` and removed the
  earlier stage-engine / SG-HCV prototypes (D-012); added the experiment
  registry and the academic manuscript.
- **Phase 3.5 — Scientific dataset approval.** Froze CCSN and GCD scientific
  specifications (`taxonomy`, `class_map`, `annotation_source`,
  `independent_sampling_unit`, `[provenance]`); moved both configs to
  `approval_status = "approved"`. Recorded that CCSN (11 WMO genera +
  contrail) and GCD (coarsened 7-class scheme) taxonomies are **not**
  interchangeable.
- **Phase 3 — Dataset discovery, validation, inventory.** Read-only discovery
  tooling and deterministic inventories; cross-split duplicate detection;
  recorded observed dataset facts; logged GCD leakage as KI-001.
- **Phase 2 / 2.5 — Configuration domain model.** Separate concepts
  (environment, dataset, model, training, evaluation, experiment); experiments
  compose the others by reference. The dataset owns `num_classes`.
- **Phase 1 — Repository foundation.** Structure, `.gitignore`, documentation
  scaffolding, Python 3.12 environment.

Datasets, splits, inventories, and approved specifications are not modified
without an explicit new decision record.

---

## 7. Commands and environment

PowerShell at the repository root; `.venv` uses Python 3.12, PyTorch
2.6.0+cu124, torchvision 0.21.0+cu124.

```powershell
# Routine verification
.venv/Scripts/python.exe -m pytest tests/ -q

# Benchmark (training + final-test evaluation)
.venv/Scripts/python.exe src/run_harmonized.py --model resnet18 --config config/training/tuned_resnet18.toml --epochs 15

# Tuning sweep (development data only)
.venv/Scripts/python.exe src/tune_resnet_family.py --model all --epochs 5

# Plot saved convergence artifacts
.venv/Scripts/python.exe scripts/plot_comparative_convergence.py
```
