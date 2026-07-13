# Phase Handoff — Governance micro-cluster G-001 (AI Working Cluster Policy)

**Handoff Version:** 1.0

> Frozen once this cluster is complete. Corrections go in the next handoff.

**Repository Baseline:** Working Tree (pre-commit) — this handoff is committed as
part of the G-001 commit.

**Cluster:** Governance micro-cluster G-001. Opus / medium effort.

## Phase completed
Governance micro-cluster **G-001** — formalization of the AI Working Cluster
Policy, the Phase Handoff artifact, the Next Cluster Prompt requirement, and
Handoff Specification v1.0.

## Objectives achieved
- Formalized the **AI Working Cluster Policy** in `docs/PROJECT_STANDARD.md`.
- Established the **Phase Handoff** as a required, frozen artifact; created
  `docs/phase_handoffs/` with `README.md` and `_TEMPLATE.md`.
- Made a **Next Cluster Prompt** the required final section of every handoff.
- Defined a handoff as a frozen, implementation-independent description of the
  **repository state**; added mandatory **Handoff Version** and **Repository
  Baseline** fields; versioned the format as **Handoff Specification v1.0**.
- Reconstructed `phase_03.md` retroactively as the inbound bridge for Phase 3.5.

## Repository changes
- Amended: `docs/PROJECT_STANDARD.md` (AI Working Cluster Policy, Phase Handoff,
  Next Cluster Prompt, handoff versioning).
- New: `docs/phase_handoffs/README.md`, `_TEMPLATE.md`, `phase_03.md`,
  `phase_03_gov.md`.
- Pointers: `CLAUDE.md` (§3, §9), `docs/project_dashboard.md` (Handoffs section),
  `docs/CHANGELOG.md`.

## Scientific decisions approved
- None. This cluster changed **development governance only**. No scientific
  behavior, datasets, configuration `approval_status`, normalization, splits,
  metrics, or metadata were touched.

## Deferred decisions
- None introduced by this cluster. Phase 3's deferred items remain open:
  per-dataset `taxonomy`, `class_map`, `annotation_source`, and
  `independent_sampling_unit` -> Phase 3.5.

## Known issues
- **KI-001 — GCD train/test duplicate leakage:** unchanged (open, documented, not
  fixed).

## Documentation synchronization status
- Complete: `docs/PROJECT_STANDARD.md`, `CLAUDE.md`, `docs/project_dashboard.md`,
  `docs/CHANGELOG.md`, and the new `docs/phase_handoffs/` files.
- `docs/project_specification.md` reviewed — unchanged (no functional/scientific
  requirement affected).
- Root `README.md` reviewed — **not** updated here; its stale content
  (mobile-images / 224×224) is a separate, pre-existing issue tracked for a later
  documentation cleanup phase.

## Current repository status
- Governance amendment complete; committed as the G-001 commit on approval.
- Dataset configs remain `approval_status = "draft"`.
- No scientific state changed. Ready to open the Phase 3.5 cluster.

## Recommended next phase
- **Phase 3.5 — Scientific Dataset Approval** (approvals only): review the dataset
  inventories; approve each dataset's `taxonomy`, `class_map`, `annotation_source`,
  and `independent_sampling_unit`; and freeze the dataset specifications. Do not
  enter Phase 4 without explicit researcher approval.

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
docs/DECISIONS.md, docs/KNOWN_ISSUES.md, docs/phase_handoffs/phase_03_gov.md (the
latest handoff) and docs/phase_handoffs/phase_03.md, then inspect the dataset
configs (config/datasets/) and the inventories (metadata/).

Current repository status: Phases 1, 2, 2.5, and 3 are complete, and governance
micro-cluster G-001 (AI Working Cluster Policy + Phase Handoff format v1.0) is
complete. Deterministic dataset inventories exist under metadata/. The CCSN and
GCD dataset configs hold only established facts and are approval_status = "draft".
GCD train/test duplicate leakage is documented as KI-001 (not fixed). The merged
dataset is deferred.

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
