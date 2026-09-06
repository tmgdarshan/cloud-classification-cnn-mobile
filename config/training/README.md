# config/training/

Training profiles answer one question: **how is the model trained?**

Two families live here:

## 1. Modular protocols (composed via `src/config_loader.py`)

| File | `approval_status` | Notes |
|---|---|---|
| `baseline.toml` | `draft` | Early Adam / lr 1e-3 protocol transcribed from an exploratory script; not researcher-approved. |
| `smoke.toml` | `validation` | Fast pipeline check (1 epoch, batch 4); non-scientific. |

Keys: `name`, `approval_status`, `optimizer`, `learning_rate`, `weight_decay`,
`batch_size`, `num_epochs`, `augmentation`, `seed`. An experiment references one
of these by name.

## 2. Selected flat configs (consumed directly by `src/run_harmonized.py --config`)

| File | `approval_status` | Model |
|---|---|---|
| `tuned_resnet18.toml` | `approved` | ResNet-18 (headline baseline) |
| `tuned_resnet34.toml` | `approved` | ResNet-34 (capacity check) |
| `tuned_resnet50.toml` | `approved` | ResNet-50 (capacity check) |

These are **not** modular protocols — the harmonized runner reads them as a flat
key/value file. All three carry the same selected recipe (D-011): AdamW,
`lr_backbone = 5e-5`, `lr_head = 5e-4`, `weight_decay = 0.01`,
`label_smoothing = 0.0`, `scheduler = cosine_annealing`, `eta_min = 1e-6`,
`batch_size = 64`, `epochs = 15`, `seed = 42`, plus an `[augmentation]` block
(no vertical flip; horizontal flip 0.5; rotation ±15°; colour jitter;
resized-crop scale 0.8–1.0) and a `[validation_performance]` block recording the
tuning trial that selected the recipe.
