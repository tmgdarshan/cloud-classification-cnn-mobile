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

## Handoffs
- Latest: `docs/phase_handoffs/phase_03_gov.md` (governance micro-cluster G-001).
- Also: `docs/phase_handoffs/phase_03.md` (Phase 3, reconstructed retroactively).
- Phase Handoffs are frozen per-phase bridges between AI working clusters
  (see *AI Working Cluster Policy* in `docs/PROJECT_STANDARD.md`); format is
  Handoff Specification v1.0.

## Notes
- Phases 1–3 and governance micro-cluster G-001 are committed; working tree is
  clean. Phase 3.5 (Scientific Dataset Approval) is the next cluster.
