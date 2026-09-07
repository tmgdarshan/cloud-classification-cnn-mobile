# Changelog

Significant approved project milestones. Newest first.

## Documentation & naming pass (2026-09-06)
- Removed the "AI working cluster / Phase Handoff" governance layer
  (`docs/PROJECT_STANDARD.md`, `docs/phase_handoffs/`, `PROJECT_CONTEXT.tmp.md`);
  consolidated the surviving human-facing content into `docs/PROJECT_GUIDE.md`.
- Standardized project vocabulary:
  one split-protocol name `grouped_stratified_holdout` v1.0; "joint" not
  "merged"; "cross-source transfer" not "zero-shot"; "cross-source" not
  "cross-sensor"; "five-class" in prose; `src/atmospheric_evaluation.py` ->
  `src/evaluation.py`. Manifest sample assignments, seeds, and saved results
  unchanged.
- Synchronized the directory READMEs, `docs/DECISIONS.md` ordering, and this
  changelog with the current tree.

## Phase 4 — Group-Aware Manifests & Harmonized Benchmark Suite
- Built four deterministic canonical split manifests under `metadata/splits/`
  (`ccsn_11class`, `gcd_6class`, `gcd_5class`, `harmonized_5bin`) remediating
  GCD train/test duplicate leakage (KI-001) via a grouped stratified holdout
  (`StratifiedGroupKFold`, seed 42) that keeps exact-duplicate clusters atomic.
- Approved decisions D-007 (five-class compatibility taxonomy), D-008
  (group-aware repartitioning), D-009 (test-holdout excluded from all
  selection), D-010 (no vertical flips), D-011 (validated tuned baseline),
  D-012 (pipeline consolidation).
- Consolidated the pipeline on `src/run_harmonized.py` (single-source
  baselines, cross-source transfer, joint model) and `src/tune_resnet_family.py`
  (ten-trial tuning). The earlier stage-engine and SG-HCV prototype scripts
  were removed once the pipeline was consolidated.
- Added `src/experiment_registry.py` / `artifacts/experiment_registry.json`,
  `src/training_state.py` (CPU state-dict snapshots), `src/evaluation.py`
  (metrics + bootstrap CIs), and the academic manuscript under `report/`.

## Phase 3.5 — Scientific Dataset Approval
- Approved and froze the CCSN and GCD scientific specifications: `taxonomy`,
  `class_map`, `annotation_source`, `independent_sampling_unit`, and `[provenance]`
  (primary publication + dataset repository). Both configs moved to
  `approval_status = "approved"`.
- Established authoritative provenance: CCSN = Zhang et al. (2018), *CloudNet*,
  GRL 45:8665–8672; GCD = the TJNU Ground-based Cloud Dataset, Liu et al. (2022),
  IEEE TGRS 60, art. 5602711 (DOI 10.1109/TGRS.2021.3063255).
- Recorded that CCSN (11 WMO genera + contrail) and GCD (7-class scheme grouping
  multiple genera + clear sky + mixed) use **non-interchangeable** taxonomies.
- Approvals only: no datasets, splits, inventories, or resolutions were modified;
  KI-001 was not remediated; the joint dataset remains deferred (delivered in Phase 4).

## Phase 3 — Dataset Discovery, Validation & Inventory
- Added read-only discovery tooling (`src/dataset_discovery.py`,
  `scripts/discover_dataset.py`) and deterministic inventories
  (`metadata/ccsn_inventory.json`, `metadata/gcd_inventory.json`).
- Added cross-split duplicate detection (data-leakage integrity check).
- Recorded observed/established dataset facts (`relative_data_path`,
  `dataset_version`, `official_split`, `num_classes`) into the dataset configs.
- Recorded GCD train/test duplicate leakage as KNOWN_ISSUES KI-001.

## Phase 2.5 — Configuration domain-model refactor
- Separated configuration into environment / dataset / model / training /
  evaluation / experiment concepts; experiments compose them by reference.

## Phase 2 — Configuration system
- Added TOML configuration profiles and a loading/composition layer
  (`src/config_loader.py`).

## Phase 1 — Repository foundation
- Established repository structure, `.gitignore`, and governance/documentation
  scaffolding.
