# Project Standard

Permanent development governance standards for this repository.

## Phase Workflow

Work proceeds in numbered phases. Each phase follows this sequence:

    Plan -> Approve -> Implement -> Independent Audit
        -> Documentation Synchronization -> Researcher Approval -> Commit

- **Plan / Approve** — the plan is proposed and the researcher approves it before
  any implementation.
- **Implement** — small, reviewable changes; no scientific behaviour is changed
  in an infrastructure phase.
- **Independent Audit** — an external review (performed by the researcher).
- **Documentation Synchronization** — required before the phase is considered
  finished (see below).
- **Researcher Approval -> Commit** — the repository is committed only after the
  researcher approves both implementation and documentation.

## Documentation Synchronization Review

At the completion of every phase, before the phase is considered finished, review
the repository documentation and update only what is necessary so that it
accurately reflects the current repository state. Where applicable this includes:
`README.md`, `docs/project_dashboard.md`, `docs/CHANGELOG.md`,
`docs/KNOWN_ISSUES.md`, `docs/DECISIONS.md`, `docs/project_specification.md`, and
any directory `README.md` (e.g. `config/`, `metadata/`, `datasets/`, `scripts/`,
`tests/`).

For each document: update only affected sections; preserve historical decisions;
do not rewrite unrelated content; keep documentation synchronized with the
implementation. If no update is required, explicitly state that the document was
reviewed and no change was necessary. A Documentation Synchronization Report
(table) is provided at the end of each phase.

**A phase is not complete until both the implementation and the documentation are
synchronized.**

## Source of Truth

The repository is the single source of truth. Governance, decisions, and project
state live in the repository (these documents), never in assistant memory. The
researcher is the final authority on all scientific decisions (see `CLAUDE.md`).
