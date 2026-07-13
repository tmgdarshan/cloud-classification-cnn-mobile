# config/environments/

Environment-specific profiles, so the same code runs unchanged in different
places by selecting a profile.

## Profiles

- `local.toml` — local workstation.
- `levante.toml` — DKRZ Levante (SLURM HPC; no compute-node internet).

## Keys

| Key | Meaning |
|-----|---------|
| `name` | Profile name (required). |
| `data_root` | Dataset root (required). Datasets are never in Git; typically `${CLOUD_DATA_ROOT}`. |
| `artifacts_root` | Where run outputs are written (git-ignored locally; scratch on Levante). |
| `device` | `auto` / `cuda` / `cpu` (informational until wired into training). |
| `num_workers`, `pin_memory` | DataLoader settings (informational until wired). |

`${VAR}` references are expanded from the environment at load time and fail
loudly if unset. These profiles differ only in machine-specific settings, never
in scientific behaviour.
