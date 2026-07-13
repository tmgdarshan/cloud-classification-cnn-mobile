# config/models/

Model profiles answer one question: **which neural network am I using?**

Each file is an architecture identity and nothing more. Filenames may name an
architecture here (that is their explicit purpose) and are stable forever.

## Keys

| Key | Meaning |
|-----|---------|
| `name` | Profile name (required). |
| `architecture` | torchvision model identifier (e.g. `resnet34`). |
| `pretrained` | Whether to initialise from pretrained weights. |
| `weights` | torchvision weights enum (e.g. `IMAGENET1K_V1`); pre-stage on Levante. |

`num_classes` is **not** here — it is a property of the dataset (the classifier
head is sized from the dataset at build time), so a model file works unchanged
across datasets with different class counts.

## Adding an architecture
Drop in one file (e.g. `efficientnet_b0.toml`, `convnext_tiny.toml`) and
reference it from an experiment. No other structure changes. Only the models an
experiment actually references are kept here, so unused architectures are not
added speculatively.
