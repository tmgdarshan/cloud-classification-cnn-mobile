# config/experiments/

Experiment profiles answer one question: **what scientific question am I asking?**

An experiment composes the other concepts by reference — it never re-states their
contents, and its filename is **never** tied to an architecture.

## Profiles

| File | `approval_status` | Notes |
|---|---|---|
| `harmonized_5bin_resnet18.toml` | `approved` | ResNet-18 on the five-class compatibility taxonomy across CCSN, GCD, and the joint pool. |
| `stage1_ccsn_baseline.toml` | `approved` | CCSN 11-class baseline (ResNet-18/34/50). |
| `stage1_gcd_6class_baseline.toml` | `approved` | GCD 6-class baseline (ResNet-18/34/50). |
| `architecture_comparison.toml` | `superseded` | Early multi-model comparison; see `src/run_harmonized.py` + `config/training/tuned_resnet*.toml`. |
| `baseline.toml` | `superseded` | Early single-model baseline on the joint pool. |
| `smoke.toml` | `validation` | Pipeline validation; non-scientific. |

> The production benchmarks run through `src/run_harmonized.py` with a flat
> `config/training/tuned_resnet*.toml`, not through experiment composition. The
> modular experiment profiles above are kept for config inspection and the
> earlier staged plan.

## Keys

| Key | Meaning |
|-----|---------|
| `name`, `objective` / `description` | Human-readable name; the scientific question / purpose. |
| `approval_status` | `draft` · `validation` · `approved` · `superseded`. |
| `environment` | Reference into `config/environments/`. |
| `dataset` | Reference into `config/datasets/`. |
| `model` **or** `models` | One model, or a list; a scalar normalises to a one-element list. |
| `training_protocol` | Reference into `config/training/`. |
| `evaluation_protocol` | Reference into `config/evaluation/`. |
| `epochs`, `batch_size`, `seed` | Optional per-experiment overrides. |

## Composition

Resolving an experiment expands the models list into **one run per model**. A
run's `approval_status` derives from its scientific components: `validation` if
any is validation, `approved` only if all are approved, otherwise `draft`.
