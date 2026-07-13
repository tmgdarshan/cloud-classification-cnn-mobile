# Phase Handoff — Phase 3.5 (Scientific Dataset Approval)

**Handoff Version:** 1.0

> Frozen once this phase is complete. Corrections go in the next handoff.

**Repository Baseline:** Working Tree (pre-commit) — this handoff is committed
together with the Phase 3.5 changes it closes.

**Cluster:** Phase 3.5 — Scientific Dataset Approval. Opus / medium effort;
approvals-only, high correctness/authority sensitivity, low code volume.

## Phase completed
Phase 3.5 — Scientific Dataset Approval. Reviewed the deterministic inventories,
established authoritative provenance, and froze the researcher-approved scientific
specifications for the CCSN and GCD datasets.

## Objectives achieved
- Approved and froze CCSN and GCD specifications: `taxonomy`, `class_map`,
  `annotation_source`, `independent_sampling_unit`, and `[provenance]`.
- Moved both dataset configs to `approval_status = "approved"`.
- Established authoritative, verified provenance (primary publication + dataset
  repository) for both datasets, including the GCD publication DOI.
- Recorded that CCSN and GCD use non-interchangeable taxonomies.

## Repository changes
- Updated: `config/datasets/ccsn.toml`, `config/datasets/gcd.toml` (approved
  scientific fields + `[provenance]` + `approval_status = "approved"`).
- Updated: `config/datasets/README.md` (documents the approved keys and
  `[provenance]`); `docs/DECISIONS.md` (Phase 3.5 decisions);
  `docs/project_dashboard.md` (status, datasets table, notes, handoffs);
  `docs/CHANGELOG.md` (Phase 3.5 milestone).
- New: `docs/phase_handoffs/phase_03_5.md` (this handoff).
- No source code, datasets, splits, inventories, or resolutions changed.

## Scientific decisions approved
- **CCSN** — taxonomy "11-class cloud taxonomy (10 WMO genera + contrail)";
  `class_map` maps the 11 verbatim folder tokens to WMO genera, with **`Ct` =
  Contrail** (the one non-WMO, human-made category). `annotation_source` avoids
  the "expert-labeled" claim (unsupported by the authors' own materials).
  Provenance: Zhang et al. (2018), *CloudNet*, GRL 45:8665–8672
  (repo: github.com/upuil/CCSN-Database).
- **GCD** — the TJNU Ground-based Cloud Dataset; taxonomy "7-class operational
  sky-condition taxonomy". `class_map` records the **coarsened WMO scheme**: each
  numbered class groups multiple genera (e.g. `2_altocumulus` = altocumulus and
  cirrocumulus), plus non-cloud `4_clearsky` (cloudiness ≤ 10%) and `7_mixed`.
  Not interchangeable with CCSN's genera taxonomy — this protects the deferred
  merge. Provenance: Liu et al. (2022), IEEE TGRS 60, art. 5602711
  (DOI 10.1109/TGRS.2021.3063255; repo: github.com/shuangliutjnu/TJNU-Ground-based-Cloud-Dataset).
- **`independent_sampling_unit = "image"`** for both, matching what the datasets
  actually provide (no camera/session/sequence metadata). For GCD this does not
  imply statistical independence; KI-001 documents duplicate-related dependence.

## Deferred decisions
- Merged-dataset harmonization and inventory → Phase 5 (unchanged).
- KI-001 remediation strategy → a separate scientific decision (unchanged).
- 512×512 preprocessing configuration → a later training/preprocessing phase.

## Known issues
- **KI-001 — GCD train/test duplicate leakage:** unchanged (open, documented, not
  fixed). Datasets were not modified in this phase.

## Documentation synchronization status
- Complete: `config/datasets/{ccsn,gcd}.toml`, `config/datasets/README.md`,
  `docs/DECISIONS.md`, `docs/CHANGELOG.md`, `docs/project_dashboard.md`, and this
  handoff.
- `docs/project_specification.md` reviewed — no functional/scientific requirement
  affected.
- Root `README.md` reviewed — not updated; its stale content is a separate,
  pre-existing issue tracked for a later documentation cleanup phase.

## Current repository status
- Phase 3.5 implementation and documentation complete; committed on approval.
- CCSN and GCD dataset configs are `approval_status = "approved"` with frozen
  scientific specifications and provenance. `merged_v1` remains `draft` (deferred).
- Config loader accepts the new keys without schema change; no run behavior
  changed (all experiments reference `merged_v1`, still draft).
- No scientific behavior beyond dataset-specification approval was changed.

## Recommended next phase
- **Phase 4 — Dataset Loading Layer:** implement config-driven, offline-first
  dataset loading that consumes the approved dataset specifications (class_map,
  splits) — no scientific-behavior changes without explicit researcher approval.

## Suggested Claude model and effort

| Phase / Task | Claude Model | Effort | Reason |
|---|---|---|---|
| Phase 4 — Dataset Loading Layer | Opus | Medium | Code-bearing phase touching the data path; correctness and reproducibility sensitive. |

## Next Cluster Prompt

> Required. Copy-paste verbatim into a fresh Claude Code session to start the next
> cluster. Recorded (frozen) here as the bridge that launched the next cluster;
> used once, then superseded by the next phase's handoff. It points a fresh
> session at the repository — it never summarizes past discussions.

```text
We are continuing the Cloud Classification CNN MSc research software project in a
fresh AI working cluster. Do not rely on any previous conversation or memory. The
repository is the authoritative source of truth; reconstruct all context from the
repository and the latest Phase Handoff.

Read, in order: CLAUDE.md, docs/PROJECT_STANDARD.md, docs/project_dashboard.md,
docs/DECISIONS.md, docs/KNOWN_ISSUES.md, docs/phase_handoffs/phase_03_5.md (the
latest handoff), then inspect the dataset configs (config/datasets/), the
inventories (metadata/), and src/config_loader.py.

Current repository status: Phases 1, 2, 2.5, 3 are complete; governance
micro-cluster G-001 is complete; and Phase 3.5 (Scientific Dataset Approval) is
complete. CCSN and GCD carry approved, frozen scientific specifications
(taxonomy, class_map, annotation_source, independent_sampling_unit, provenance)
and are approval_status = "approved". The merged dataset remains deferred and
draft. GCD train/test duplicate leakage is documented as KI-001 (not fixed).

Next approved objective — Phase 4, Dataset Loading Layer: design and implement
config-driven, offline-first dataset loading that consumes the approved dataset
specifications (class_map and official splits) with deterministic behavior. Plan
and explain before implementing; do not change scientific behavior (splits,
normalization, class definitions) without explicit researcher approval.

Out of scope: KI-001 remediation, the merged dataset, preprocessing/normalization
policy changes, training, and any modification of datasets or inventories.

Wait for researcher approval before any implementation.

Read CLAUDE.md, docs/PROJECT_STANDARD.md, the latest Phase Handoff, and the
repository before proposing any work. Do not rely on previous conversations.
```
