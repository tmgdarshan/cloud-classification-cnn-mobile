# docs/phase_handoffs/

Frozen **Phase Handoffs** — the bridges between AI working clusters (see the
*AI Working Cluster Policy* in `../PROJECT_STANDARD.md`).

Each phase closes by writing a handoff here. A handoff is a required artifact and
is **frozen** once its phase is complete: it is a permanent historical record.
Corrections belong in the next handoff, not in edits to a frozen one.

## Naming

- `phase_<NN>.md` — a numbered phase, e.g. `phase_03.md`, `phase_04.md`.
- `phase_<NN>_<label>.md` — a sub-phase or micro-cluster, e.g. `phase_03_5.md`
  (Phase 3.5) or `phase_03_gov.md` (a governance micro-cluster).

Use `_TEMPLATE.md` as the starting structure so every handoff is consistent and
auditable.

## Reading order for a new cluster

A new AI working cluster starts by reading `../../CLAUDE.md`,
`../PROJECT_STANDARD.md`, `../project_dashboard.md`, `../DECISIONS.md`,
`../KNOWN_ISSUES.md`, the **latest** handoff in this directory, and the
repository itself — then proposes work. It must not rely on memory of previous
conversations.
