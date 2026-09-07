# config/

Configuration is split by concept. Each directory answers one question, and an
experiment composes the others by reference. Files are TOML, read with the
standard library (`tomllib`, Python 3.11+). `src/config_loader.py` loads and
composes these; it is not wired into training.

| Directory | Question | Files |
|---|---|---|
| `environments/` | Where does it run? | `local`, `levante` (DKRZ SLURM HPC) |
| `datasets/` | Which data? | `ccsn`, `gcd`, `gcd_5class`, `gcd_6class`, `harmonized_5bin`, `merged_v1` (superseded) |
| `models/` | Which network? | `resnet18`, `resnet34`, `resnet50` |
| `training/` | How is it trained? | modular: `baseline` (draft), `smoke`; flat: `tuned_resnet{18,34,50}*.toml` |
| `evaluation/` | How is it measured? | `standard` |
| `experiments/` | Which question? | `harmonized_5bin_resnet18`, `architecture_comparison` (superseded), `baseline` (superseded), `smoke` |

The production benchmarks do **not** go through experiment composition. They run
`src/run_harmonized.py --config config/training/tuned_resnet{18,34,50}*.toml`,
which reads a flat key/value file directly. The modular profiles are kept for
config-inspection tests and the earlier staged plan.

## Two kinds of training config

**Modular protocols** (`baseline.toml`, `smoke.toml`) carry
`optimizer`, `learning_rate`, `weight_decay`, `batch_size`, `num_epochs`,
`augmentation`, `seed`, and are referenced by name from an experiment.

**Flat tuned configs** (`tuned_resnet*.toml`) are read directly by the harmonized
runner. The joint-pool configs (`tuned_resnet{18,34,50}.toml`) all carry the same
recipe: AdamW, `lr_backbone = 5e-5`, `lr_head = 5e-4`, `weight_decay = 0.01`,
`label_smoothing = 0.0`, cosine annealing to `eta_min = 1e-6`, `batch_size = 64`,
`epochs = 15`, `seed = 42`, plus an `[augmentation]` block (no vertical flip;
horizontal flip 0.5; rotation ±15°; colour jitter; resized-crop scale 0.8–1.0).
The single-source variants:

| File | Pool | Epochs | Tuning trial |
|---|---|---|---|
| `tuned_resnet18_gcd.toml`, `_resnet34_gcd`, `_resnet50_gcd` | GCD only | 15 | Trial 08 |
| `tuned_resnet18_ccsn.toml`, `_resnet34_ccsn`, `_resnet50_ccsn` | CCSN only | 90 (budget-matched) | Trial 03 / 06 / 06 |
| `tuned_resnet18_ccsn_15ep.toml`, `_resnet34_ccsn_15ep`, `_resnet50_ccsn_15ep` | CCSN only | 15 (control) | Trial 03 / 06 / 06 |

## Dataset profile keys

`name` / `key`, `approval_status` (`draft` / `validation` / `approved` /
`superseded`), `sources`, `dataset_version`, `relative_data_path` (under
`CLOUD_DATA_ROOT`), `official_split`, `num_classes` (owned by the dataset, not the
model), `excluded_class_folders`, `taxonomy`, `class_map` (verbatim folder token
→ approved class name), `annotation_source`, `independent_sampling_unit`,
`manifest_path` (harmonized only), `[provenance]`. Observed image resolutions are
**not** stored here — the inventory under `metadata/` is authoritative.

CCSN and GCD are independent: CCSN uses 11 WMO genera plus contrail; GCD uses a
coarsened 7-class scheme. `harmonized_5bin.toml` points at
`metadata/splits/harmonized_5bin_canonical.json`, the machine-authoritative
five-class mapping (Decision D-007).

## Model profile keys

`name`, `architecture` (torchvision identifier), `pretrained`, `weights`
(torchvision enum). `num_classes` is not here — the head is sized from the
dataset at build time, so one model file works across datasets with different
class counts. Adding an architecture is one new file referenced from an
experiment.

## Environment / evaluation keys

Environment: `name`, `data_root` (required; datasets are never in Git),
`artifacts_root`, `device`, `num_workers`, `pin_memory`. `${VAR}` references
expand at load time and fail loudly if unset. Profiles differ only in
machine-specific settings.

Evaluation: `name`, `approval_status`, `primary_metric`, `secondary_metrics`,
`produce_confusion_matrix`, `produce_classification_report`.

## Experiment keys and composition

`name`, `objective` / `description`, `approval_status`, `environment`, `dataset`,
`model` or `models`, `training_protocol`, `evaluation_protocol`, and optional
`epochs` / `batch_size` / `seed` overrides. Resolving an experiment expands the
models list into one run per model; a run is `approved` only if every referenced
component is approved.
