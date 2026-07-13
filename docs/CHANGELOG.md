# Changelog

Significant approved project milestones. Newest first.

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
  KI-001 was not remediated; the merged dataset remains deferred.

## Governance — AI Working Cluster Policy
- Formalized **AI working clusters** and the required, frozen **Phase Handoff**
  artifact. Amended `docs/PROJECT_STANDARD.md` (AI Working Cluster Policy + Phase
  Handoff sections) and added `docs/phase_handoffs/` (`README.md`, `_TEMPLATE.md`).
- Added pointers in `CLAUDE.md` and `docs/project_dashboard.md`.
- Reconstructed `docs/phase_handoffs/phase_03.md` retroactively as the inbound
  bridge for Phase 3.5.
- Made a **Next Cluster Prompt** a required, frozen section of every handoff — a
  copy-pasteable prompt that points a fresh session at the repository (never a
  summary of past discussions).
- Defined a Phase Handoff as a frozen, implementation-independent description of
  the **repository state** at phase completion; added mandatory **Handoff
  Version** and **Repository Baseline** fields; versioned the handoff format as
  Handoff Specification v1.0.

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
