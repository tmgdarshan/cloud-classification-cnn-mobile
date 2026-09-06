# Project Dashboard

Current project status and operational overview.

---

## 1. Phase Status

| Phase | Description | Status | Notes |
| :--- | :--- | :---: | :--- |
| **Phase 1** | Repository foundation & environment setup | Complete | Python 3.12 venv, PyTorch, Torchvision |
| **Phase 2** | Configuration domain model | Complete | TOML-based modular configuration |
| **Phase 3** | Dataset discovery, validation & inventory | Complete | Verified resolutions & inventories |
| **Phase 3.5** | Scientific dataset approval & provenance | Complete | Approved taxonomies & sampling units |
| **Phase 4** | Group-aware manifests, harmonized benchmark suite | Implementation complete | Single runner (`run_harmonized.py`), five-class taxonomy. Pending: commit of the working tree; run/provenance follow-ups in `docs/audits/`. |
| **Phase 5** | Edge deployment & mobile optimization | Planned | Model distillation & quantization |

---

## 2. Approved Datasets and Manifests

All active training and evaluation pipelines consume frozen canonical manifests generated with atomic exact-byte duplicate isolation:

| Dataset / View | Total Images | Classes | Partition Protocol | Taxonomy Type | Approval Status |
| :--- | :---: | :---: | :--- | :--- | :---: |
| **CCSN 11-Class** | 2,537 | 11 | StratifiedGroupKFold (80/20 Dev/Test) | 10 WMO genera + contrail (3 conflicting pairs / 6 images purged) | Approved |
| **GCD 6-Class** | 18,045 | 6 | StratifiedGroupKFold (80/20 Dev/Test) | 6 operational sky conditions (excl. mixed) | Approved |
| **GCD 5-Class** | 14,306 | 5 | StratifiedGroupKFold (80/20 Dev/Test) | 5 cloud-only operational classes (excl. clearsky) | Approved |
| **Harmonized 5-Class** | 16,643 | 5 | StratifiedGroupKFold (80/20 Dev/Test) | Five-class compatibility taxonomy | Approved |

*Notes*:
- Observational redundancy and cross-split duplicate contamination in raw benchmark releases (Audit Record KI-001) are remediated for exact-byte duplicates in canonical manifests via atomic SHA-256 duplicate clustering and `StratifiedGroupKFold` (Decision D-008).
- The five-class compatibility taxonomy is established as an operational cross-source compatibility scheme, not a strict WMO genus taxonomy (Decision D-007).
- For complete methodological specifications, see [`docs/OFFICIAL_PROTOCOL.md`](OFFICIAL_PROTOCOL.md) and [`docs/DECISIONS.md`](DECISIONS.md).

---

## 3. Production Pipeline Status

The production workflow is consolidated around the following core components:

- **Single Source of Truth for Partitions**: Canonical manifests in [`metadata/splits/*_canonical.json`](../metadata/splits/).
- **Official Benchmark Runner**: [`src/run_harmonized.py`](../src/run_harmonized.py) (executes single-source baselines, cross-source transfer, and the joint CCSN+GCD model with source-balanced batch sampling and bootstrap confidence intervals).
- **Hyperparameter Exploration Engine**: [`src/tune_resnet_family.py`](../src/tune_resnet_family.py) (systematic optimization across learning rates and regularizers on the development pool).
- **Evaluation metrics**: [`src/evaluation.py`](../src/evaluation.py) (confusion matrices, balanced accuracy, macro-F1, bootstrap confidence intervals).
- **Snapshot Safety Utilities**: [`src/training_state.py`](../src/training_state.py) (CPU-isolated state dict snapshotting).
- **Legacy code**: earlier stage-engine and SG-HCV prototypes were removed once the pipeline was consolidated (D-012); recover from git history if needed.

---

## 4. Project History & Orientation

Phase-by-phase history and the source-of-truth map are in
[`docs/PROJECT_GUIDE.md`](PROJECT_GUIDE.md). Methodological rationale is in
[`docs/DECISIONS.md`](DECISIONS.md); milestones in
[`docs/CHANGELOG.md`](CHANGELOG.md).
