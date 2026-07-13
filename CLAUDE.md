# CLAUDE.md

Guidance for Claude Code when working in this repository.

## 1. Project Overview

`cloud-classification-cnn-mobile` supports **MSc thesis research in Atmospheric
Science**: training CNNs (ResNet family) to classify cloud types from images
(CCSN, GCD, and a derived merged dataset).

This repository holds **source code, documentation, configuration, and metadata
only**. Datasets and trained models live **outside** Git. Work must run on a
**local workstation** now and on **DKRZ Levante** (SLURM HPC) later, from the
same codebase.

## 2. Scientific Authority

The human researcher is the **final authority** on all scientific matters:
methodology, experiment design, dataset selection, interpretation of results,
and thesis conclusions.

Claude may **recommend and explain**, but must **not make scientific decisions
autonomously**. When a change would affect scientific behavior — data splits,
normalization statistics, class definitions, metrics, model architecture, or
hyperparameters — **stop, explain the impact, and ask for a decision.**

**Preserve existing scientific behavior unless explicitly asked to change it.**

## 3. Repository Governance

When present, consult the following documents before making significant changes:

- `PROJECT_STANDARD.md` — permanent development governance
- `project_specification.md` — approved scientific and functional requirements
- `DECISIONS.md` — accepted scientific and methodological decisions
- `project_dashboard.md` — current project status
- `KNOWN_ISSUES.md` — accepted limitations and deferred work
- `CHANGELOG.md` — significant approved project milestones
- `docs/phase_handoffs/` — frozen Phase Handoffs bridging AI working clusters;
  read the latest before proposing work

If these documents conflict, **stop and ask the researcher for clarification.**

## 4. Dataset Policy

- **Never commit datasets to Git** — not raw, processed, sampled, or zipped.
  This includes CCSN, GCD, processed variants, and any merged dataset.
- Datasets exist **only** on the local machine and on Levante. The repository
  stores only their *contract*: expected layout, class lists, checksums, source
  citations, and preparation scripts.
- **Resolve the data location from configuration or an environment variable**,
  never from a repository-local `data/` folder, `__file__`, or the working
  directory.
- If a task seems to need data inside the repo, **stop and ask** — reference it
  via config instead.

## 5. Reproducibility Requirements

- **Prioritize reproducibility over convenience.** A result is only "done" when
  it can be regenerated on another machine.
- **Config-driven, not edit-in-place.** Hyperparameters, paths, and device
  settings come from config files — not from editing constants in the code.
- **Determinism.** Set a single random seed (Python, NumPy, PyTorch/CUDA) and
  record it. Keep non-deterministic options (e.g. `cudnn.benchmark`) off for
  reported runs.
- **Traceability.** Every run should record enough to reproduce it: the resolved
  config, git commit, dataset version/checksums, library versions, and seed.
- **Regenerable, not committed.** Weights, metrics, and figures are outputs of
  `code + config + data`, not checked-in artifacts.
- **Honest provenance.** Output names must match what produced them (e.g. a
  ResNet18 run must not be saved with a ResNet34 name).

## 6. Engineering Rules

- **Small, reviewable changes.** Prefer the minimal diff that solves the task.
- **Explain major changes before implementing them**, especially anything
  touching training, data, or evaluation.
- **Do not silently change observable behaviour.** If a refactor alters outputs,
  defaults, preprocessing, configuration semantics, or user-facing behaviour,
  explain the change before implementing it.
- **No hardcoded filesystem paths** (`/home/...`, `C:\...`, `<repo>/data`).
  Use configuration-based paths only.
- **Modular design, no import side effects.** Importing a module must not read
  data, download weights, or start training. Put such work behind functions.
- **Single source of truth** for normalization stats, class maps, and split
  logic — defined once, imported, never duplicated.
- **Prefer the simplest design that satisfies the approved requirements.** Avoid
  introducing abstractions before they have a demonstrated need.
- **Clear documentation** for public functions and any non-obvious scientific
  choice.

## 7. Environment and Portability

- The **same code** runs locally and on Levante; only **configuration** differs.
- Select environment and data/output roots via config or environment variables
  (e.g. an environment name plus a data-root variable) — never bake
  machine-specific values into source.
- **Levante assumptions:** jobs run under **SLURM**; compute nodes have **no
  outbound internet**, so pre-stage datasets and any pretrained weights; use
  scratch storage for fast run I/O and shared storage for datasets.
- **Offline-first:** the training path must not require network downloads. Load
  pretrained weights from a local cache.
- Device, worker count, and AMP settings come from config with a safe CPU
  fallback — not from hardcoded literals.

## 8. Artifact Policy

- **Generated artifacts are not source code.** Model checkpoints (`*.pth`,
  `*.pt`, `*.ckpt`), metrics tables, logs, and figures must **never be
  committed**.
- Artifacts are written to a **configurable run/output directory** (local
  `runs/`, Levante scratch), which is git-ignored — never into `src/`.
- Share final artifacts via a Release or Levante storage, not via Git history.
- `.gitignore` is authoritative for exclusions (datasets, checkpoints, zips,
  run outputs, IDE files, caches). Never `git add -f` these.

## 9. Expected Claude Workflow

1. **Understand first.** Each working cluster begins by reading this file,
   `docs/PROJECT_STANDARD.md`, the dashboard, `docs/DECISIONS.md`,
   `docs/KNOWN_ISSUES.md`, the latest Phase Handoff (`docs/phase_handoffs/`), and
   the relevant code — before acting. Each phase closes with a Phase Handoff.
2. **Plan and explain.** For non-trivial work, state the goal, affected files,
   scientific/reproducibility impact, and verification approach — then wait for
   confirmation on anything that touches scientific behavior.
3. **Implement in small steps** with clear, reviewable diffs.
4. **Keep data and artifacts out of Git**; keep paths configuration-based.
5. **Verify.** Exercise the changed path (a short run or test) and report what
   you actually observed — including failures or skipped steps.
6. **Git discipline.** Work on feature branches; **commit or push only when
   asked.** Never rewrite shared history without explicit human coordination.
7. **Ask when unsure.** Surface uncertainty rather than guessing on scientific
   or data-governance questions.
