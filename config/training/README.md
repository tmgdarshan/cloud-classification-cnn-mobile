# config/training/

Training protocols answer one question: **how is the model trained?**

## Profiles

- `baseline.toml` — transcribed from the current script (draft).
- `smoke.toml` — fast pipeline-validation protocol (non-scientific).

## Keys

| Key | Meaning |
|-----|---------|
| `name` | Profile name (required). |
| `approval_status` | `draft` / `validation` / `approved`. |
| `optimizer`, `learning_rate`, `weight_decay` | Optimisation settings. |
| `batch_size`, `num_epochs` | Training loop settings. |
| `augmentation` | Augmentation policy (kept here until an augmentation *study* needs its own axis). |
| `seed` | Random seed for reproducibility. |

These are hyperparameters, not scientific questions. Multiple experiments may
share one protocol; an experiment references a protocol by name.
