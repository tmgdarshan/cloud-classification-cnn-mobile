# config/

Configuration is a **domain model**, not a bag of settings. Each directory is one
independent concept, and an experiment composes the others by reference.

## Concepts

| Directory | Answers | Examples |
|-----------|---------|----------|
| `environments/` | *Where* am I running? | `local`, `levante` |
| `datasets/` | *Which* scientific data? | `merged_v1`, `ccsn`, `gcd` |
| `models/` | *Which* neural network? | `resnet18`, `resnet34`, `resnet50` |
| `training/` | *How* is it trained? | `baseline`, `smoke` |
| `evaluation/` | *How* is it measured? | `standard` |
| `experiments/` | *What* scientific question? | `baseline`, `architecture_comparison`, `smoke` |

Files are TOML, read with the standard library (`tomllib`, Python 3.11+).

## Composition

A **run** = `environment × dataset × model × training × evaluation` + experiment
metadata. Resolving an experiment expands its models list into one run each:

    python scripts/show_config.py --environment local --experiment architecture_comparison

`${VAR}` references in environment paths (e.g. `${CLOUD_DATA_ROOT}`) are expanded
at load time and fail loudly if unset. Generated dataset inventories produced in
Phase 3 live in the top-level `metadata/` directory, not here.

> The loader (`src/config_loader.py`) performs **configuration loading and
> composition only**. It is not wired into training, which is unchanged.
