# Project Dashboard

Current project status and progress overview.

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Repository foundation | Complete |
| 2 | Configuration system | Complete |
| 2.5 | Configuration domain-model refactor | Complete |
| 3 | Dataset Discovery, Validation & Inventory | Complete |
| 3.5 | Scientific Dataset Approval | Next (approvals only) |
| 4 | Dataset Loading Layer | Planned |
| 5 | Dataset harmonization / merged dataset | Planned |

## Datasets (discovered, independent)

| Dataset | Images | Class folders | Split | Notes |
|---------|--------|---------------|-------|-------|
| CCSN | 2,543 | 11 | none | mixed native resolutions (400×400 + 256×256) |
| GCD | 19,000 | 7 | official train/test | 512×512; train/test duplicate leakage (KI-001) |

Inventories are the authoritative record of observed data: see `metadata/`.
The merged dataset is intentionally not yet inventoried or configured.

## Notes
- Work is held in the working tree pending Independent Audit and Researcher
  Approval; nothing is committed yet.
