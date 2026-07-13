# Project Dashboard

Current project status and progress overview.

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Repository foundation | Complete |
| 2 | Configuration system | Complete |
| 2.5 | Configuration domain-model refactor | Complete |
| 3 | Dataset Discovery, Validation & Inventory | Complete |
| 3.5 | Scientific Dataset Approval | Complete |
| 4 | Dataset Loading Layer | Planned |
| 5 | Dataset harmonization / merged dataset | Planned |

## Datasets (independent, specifications approved in Phase 3.5)

| Dataset | Images | Classes | Split | Taxonomy | approval_status |
|---------|--------|---------|-------|----------|-----------------|
| CCSN | 2,543 | 11 | none | 10 WMO genera + contrail | approved |
| GCD | 19,000 | 7 | official train/test | 7-class sky-condition (grouped genera) | approved |

Notes: CCSN has mixed native resolutions (400×400 + 256×256); GCD is 512×512 with
train/test duplicate leakage (KI-001). GCD = TJNU Ground-based Cloud Dataset; its
7 classes group multiple WMO genera, so CCSN and GCD taxonomies are **not**
interchangeable. Inventories are the authoritative record of observed data: see
`metadata/`. The merged dataset is intentionally not yet inventoried or configured.

## Handoffs
- Latest: `docs/phase_handoffs/phase_03_5.md` (Phase 3.5, Scientific Dataset Approval).
- Also: `docs/phase_handoffs/phase_03_gov.md` (governance micro-cluster G-001);
  `docs/phase_handoffs/phase_03.md` (Phase 3, reconstructed retroactively).
- Phase Handoffs are frozen per-phase bridges between AI working clusters
  (see *AI Working Cluster Policy* in `docs/PROJECT_STANDARD.md`); format is
  Handoff Specification v1.0.

## Notes
- Phases 1–3, governance micro-cluster G-001, and Phase 3.5 (Scientific Dataset
  Approval) are complete. CCSN and GCD carry approved, frozen scientific
  specifications (`approval_status = "approved"`). Phase 4 (Dataset Loading Layer)
  has not begun.
