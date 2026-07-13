# Changelog

Significant approved project milestones. Newest first.

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
