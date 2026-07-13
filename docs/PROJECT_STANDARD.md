# Project Standard

Permanent development governance standards for this repository.

## Phase Workflow

Work proceeds in numbered phases. Each phase follows this sequence:

    Plan -> Approve -> Implement -> Independent Audit
        -> Documentation Synchronization -> Researcher Approval -> Commit
        -> Phase Handoff

- **Plan / Approve** — the plan is proposed and the researcher approves it before
  any implementation.
- **Implement** — small, reviewable changes; no scientific behaviour is changed
  in an infrastructure phase.
- **Independent Audit** — an external review (performed by the researcher).
- **Documentation Synchronization** — required before the phase is considered
  finished (see below).
- **Researcher Approval -> Commit** — the repository is committed only after the
  researcher approves both implementation and documentation.
- **Phase Handoff** — a required, frozen artifact that closes the cluster and
  bridges to the next (see *AI Working Cluster Policy* and *Phase Handoff* below).

## AI Working Cluster Policy

Each major phase — or a well-defined sub-phase — is executed within its own **AI
working cluster**: a self-contained unit of work with a clear beginning and end.
Clusters are connected through the **repository**, never through conversation
history. An AI conversation is a temporary execution environment; the repository
is the permanent knowledge base.

A cluster runs:

    Repository State -> Phase Objective -> Plan -> Researcher Approval
        -> Implementation -> Independent Audit -> Documentation Synchronization
        -> Researcher Approval -> Commit -> Phase Handoff -> (new cluster)

**No implementation occurs after the Phase Handoff.** The next cluster begins by
reading `CLAUDE.md`, `docs/PROJECT_STANDARD.md`, `docs/project_dashboard.md`,
`docs/DECISIONS.md`, `docs/KNOWN_ISSUES.md`, the latest Phase Handoff under
`docs/phase_handoffs/`, and the repository itself — and only then proposes work.
It must not rely on memory of previous conversations.

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

## Phase Handoff

A Phase Handoff is a **frozen, implementation-independent description of the
repository state at the completion of a phase** — it documents the repository, not
the AI conversation.

At the completion of every phase, before the cluster ends, Claude writes a Phase
Handoff to `docs/phase_handoffs/phase_<NN>.md` (sub-phases use a suffix, e.g.
`phase_03_5.md`). The Phase Handoff is a **required artifact**: a phase is not
complete until its handoff is written, approved, and committed with the phase's
other changes.

Each handoff is **frozen** once its phase is complete — it is a historical
record. Corrections are made in the next handoff, never by rewriting a frozen
one.

Every handoff records, concisely:

- Handoff Version — the handoff-specification version it follows (current: v1.0)
- Repository Baseline — the exact state the handoff describes: a commit hash, or
  `Working Tree (pre-commit)` when the handoff is committed together with the
  phase it closes
- Phase completed
- Objectives achieved
- Repository changes
- Scientific decisions approved
- Deferred decisions
- Known issues
- Documentation synchronization status
- Current repository status
- Recommended next phase
- Suggested Claude model and effort

The handoff is a commit message for humans and future AI sessions: it carries
only what is needed to continue, so the next cluster starts with focused context
instead of thousands of tokens of historical discussion.

The handoff format is itself **versioned** (Handoff Specification v1.0). When the
format evolves, the specification version is incremented (v1.1, v2.0, …) rather
than rewriting old handoffs; each handoff stays valid because it records the
version it follows.

### Next Cluster Prompt (required)

Every handoff ends with a **Next Cluster Prompt**: a ready-to-use prompt that
initializes the next AI working cluster, copy-pasteable into a fresh Claude Code
session without modification.

The prompt must:

- Assume no previous conversation exists.
- Instruct Claude to reconstruct context from the repository and the latest Phase
  Handoff — never from a summary of old discussions.
- State the current repository status.
- State the next approved objective.
- Explicitly state what is out of scope.
- Instruct Claude to wait for researcher approval before any implementation.
- End with: *"Read `CLAUDE.md`, `docs/PROJECT_STANDARD.md`, the latest Phase
  Handoff, and the repository before proposing any work. Do not rely on previous
  conversations."*

The Next Cluster Prompt is an **operational convenience**: it is recorded (frozen)
inside the handoff as the historical bridge that launched the next cluster, but it
is *used once* to start that cluster and is superseded when the next phase writes
its own handoff. The repository — not the prompt — remains the authoritative
source of truth; the prompt only points a fresh session at it and must never
substitute a discussion summary for reconstruction from the repository.

## Source of Truth

The repository is the single source of truth. Governance, decisions, and project
state live in the repository (these documents), never in assistant memory. The
researcher is the final authority on all scientific decisions (see `CLAUDE.md`).
