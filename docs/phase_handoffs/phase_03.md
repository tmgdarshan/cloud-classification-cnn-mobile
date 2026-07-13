# Phase Handoff — Phase 3 (Dataset Discovery, Validation & Inventory)

**Handoff Version:** 1.0

> Reconstructed retroactively when the AI Working Cluster Policy was adopted.
> Compiled from `docs/CHANGELOG.md`, `docs/DECISIONS.md`, `docs/KNOWN_ISSUES.md`,
> the dataset configs, and the inventories under `metadata/`. Frozen.

**Repository Baseline:** Working Tree (pre-commit) — committed together with the
G-001 governance changes.

**Cluster:** Phase 3 (reconstructed). Work suited to: Opus / medium effort.

## Phase completed
Phase 3 — Dataset Discovery, Validation & Inventory.

## Objectives achieved
- Added read-only discovery tooling (`src/dataset_discovery.py`,
  `scripts/discover_dataset.py`).
- Produced deterministic, machine-readable inventories
  (`metadata/ccsn_inventory.json`, `metadata/gcd_inventory.json`).
- Implemented cross-split duplicate detection (a data-leakage integrity check).
- Recorded observed/established dataset facts into the dataset configs.

## Repository changes
- New: `src/dataset_discovery.py`, `scripts/discover_dataset.py`,
  `metadata/ccsn_inventory.json`, `metadata/gcd_inventory.json`,
  `metadata/README.md`.
- Updated: `config/datasets/ccsn.toml`, `config/datasets/gcd.toml`
  (established facts only); `docs/project_dashboard.md`, `docs/DECISIONS.md`,
  `docs/CHANGELOG.md`, `docs/KNOWN_ISSUES.md`.

## Scientific decisions approved
- **Dataset metadata policy:** configs store only established facts after
  discovery — `relative_data_path`, `dataset_version`, `official_split`,
  `num_classes`. `taxonomy`, class mapping, `annotation_source`, and
  `independent_sampling_unit` are deferred to Scientific Dataset Approval and are
  never inferred from directory names.
- **Inventories are authoritative for observed resolutions;** no
  `expected_resolution` is stored. CCSN is a mix of 400x400 and 256x256.
- **512x512 standardization is preprocessing policy,** not a dataset property.
- **GCD train/test duplicate leakage is documented, not fixed** (KI-001).
- **Merged dataset is deferred** until its harmonization methodology is designed
  and approved.

## Deferred decisions
- Per-dataset `taxonomy`, `class_map`, `annotation_source`,
  `independent_sampling_unit` -> Phase 3.5 (Scientific Dataset Approval).
- 512x512 preprocessing configuration -> a later training/preprocessing phase.
- KI-001 remediation strategy -> a separate scientific decision.
- Merged-dataset harmonization and inventory -> Phase 5.

## Known issues
- **KI-001 — GCD train/test duplicate leakage:** 156 of 159 exact-byte duplicate
  groups span the official `train` and `test` splits (concentrated in
  `4_clearsky`). Open; datasets are not modified.

## Documentation synchronization status
- Complete: `docs/project_dashboard.md`, `docs/DECISIONS.md`,
  `docs/CHANGELOG.md`, `docs/KNOWN_ISSUES.md`, `metadata/README.md`.

## Current repository status
- Phase 3 implementation and documentation complete.
- Dataset configs are `approval_status = "draft"`.
- Phase 3 work is held in the working tree, uncommitted, pending Independent
  Audit and Researcher Approval.

## Recommended next phase
- Phase 3.5 — Scientific Dataset Approval (approvals only): review inventories;
  approve taxonomy, class maps, and the remaining deferred metadata; freeze the
  dataset specifications. Do not enter Phase 4 without explicit researcher
  approval.

## Suggested Claude model and effort

| Phase / Task | Claude Model | Effort | Reason |
|---|---|---|---|
| Phase 3.5 — Scientific Dataset Approval | Opus | Medium | Scientific-methodology and governance work; low code volume, high correctness/authority sensitivity. |

## Next Cluster Prompt

> Required. Copy-paste verbatim into a fresh Claude Code session to open the
> Phase 3.5 cluster. Recorded here as the bridge; used once, then superseded.

```text
We are continuing the Cloud Classification CNN MSc research software project in a
fresh AI working cluster. Do not rely on any previous conversation or memory. The
repository is the authoritative source of truth; reconstruct all context from the
repository and the latest Phase Handoff.

Read, in order: CLAUDE.md, docs/PROJECT_STANDARD.md, docs/project_dashboard.md,
docs/DECISIONS.md, docs/KNOWN_ISSUES.md, docs/phase_handoffs/phase_03.md, then
inspect the dataset configs (config/datasets/) and the inventories (metadata/).

Current repository status: Phases 1, 2, 2.5, and 3 are complete. Deterministic
dataset inventories exist under metadata/. The CCSN and GCD dataset configs hold
only established facts and are approval_status = "draft". GCD train/test duplicate
leakage is documented as KI-001 (not fixed). The merged dataset is deferred.

Next approved objective — Phase 3.5, Scientific Dataset Approval (approvals only):
review the dataset inventories; approve each dataset's taxonomy, class_map,
annotation_source, and independent_sampling_unit; and freeze the dataset
specifications. Present per-dataset decision sheets (observed folder tokens plus
reference mappings) for the researcher to approve or override. Never infer
taxonomy automatically; never modify datasets.

Out of scope: Phase 4 and beyond, any dataset modification, KI-001 remediation,
the merged dataset, and any preprocessing/training code. Do not change scientific
behavior.

Wait for researcher approval before any implementation.

Read CLAUDE.md, docs/PROJECT_STANDARD.md, the latest Phase Handoff, and the
repository before proposing any work. Do not rely on previous conversations.
```
