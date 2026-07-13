# config/experiments/

Experiment profiles answer one question: **what scientific question am I asking?**

An experiment composes the other concepts by reference — it never re-states their
contents, and its filename is **never** tied to an architecture.

## Profiles

- `baseline.toml` — Baseline Classification (single model).
- `architecture_comparison.toml` — one question, several models.
- `smoke.toml` — pipeline validation (non-scientific).

## Keys

| Key | Meaning |
|-----|---------|
| `name` | Human-readable experiment name (required). |
| `objective` | The scientific question / purpose. |
| `approval_status` | `draft` (unapproved) · `validation` (non-scientific) · `approved`. |
| `dataset` | Dataset reference (`config/datasets/<key>.toml`). |
| `model` **or** `models` | One model, or a list — both accepted; a scalar is normalised to a one-element list. |
| `training_protocol` | Reference into `config/training/`. |
| `evaluation_protocol` | Reference into `config/evaluation/`. |

## Composition
Resolving an experiment expands the models list into **one run per model**
(`architecture_comparison` → 3 runs; `baseline` → 1). A run's `approval_status`
is derived from its scientific components: `validation` if any is validation,
`approved` only if all are approved, otherwise `draft`.
