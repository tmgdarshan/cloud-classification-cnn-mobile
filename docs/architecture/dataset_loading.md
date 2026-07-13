# Dataset Loading Architecture

Describes the Dataset Loading Layer introduced in Phase 4. Unlike a Phase
Handoff, this document is **not frozen** -- it is a living reference to the
current design and is expected to be updated as later phases extend the
loading layer (e.g. preprocessing/augmentation, training). See
`docs/phase_handoffs/phase_04.md` for the frozen record of what Phase 4
actually shipped.

## Purpose

Load the approved CCSN and GCD dataset specifications
(`config/datasets/{ccsn,gcd}.toml`) into indexed, labeled samples that a
future training phase can consume -- without deciding anything scientific
itself (splits, class definitions, normalization) and without depending on
torch/torchvision beyond the minimum needed to expose a PyTorch `Dataset`.

## Two layers, one direction of dependency

```
config/datasets/*.toml (approved, frozen)
            |
            v
  src/dataset_index.py   (Layer A -- pure, stdlib only)
            |
            v
  src/dataset_loading.py (Layer B -- torch + Pillow adapter)
            |
            v
  (future) preprocessing / augmentation / training
```

Layer B depends on Layer A; Layer A never depends on Layer B. This means the
class-folder contract (which folders exist, which label each maps to,
whether images are present) can be fully tested without torch, torchvision,
or Pillow ever being imported.

## Layer A -- `src/dataset_index.py`

**Framework independence is the core invariant of this layer.** It imports
only `dataclasses`, `pathlib`, and `typing` -- never `torch`, `torchvision`,
or `PIL`, even transitively. It represents repository *structure* only: it
never opens an image file, never preprocesses, never augments, never
batches, and never builds a `DataLoader`.

### Public API

- `DatasetLoadingError(RuntimeError)` -- the single exception type raised for
  every contract violation in both Layer A and Layer B.
- `DatasetIndex` (frozen dataclass) -- `dataset_key`, `split`,
  `classes`, `class_to_idx`, `class_display_names`, `samples`,
  `num_classes`. Enforces `num_classes == len(classes) ==
  len(class_to_idx)` in `__post_init__`.
- `available_splits(dataset_cfg, dataset_root) -> list[str]` -- returns `[]`
  when the dataset's `official_split` is `false`; otherwise reports which of
  the canonical `train`/`test` directories are actually present on disk.
  Never invents a split.
- `build_index(dataset_cfg, dataset_root, *, split=None) -> DatasetIndex` --
  validates the on-disk class folders exactly match the approved
  `class_map`, assigns labels in `class_map` key order, and enumerates
  images per class into a deterministic, sorted sample list.

### Invariants

- **Class ordering comes only from `class_map`.** `class_to_idx` is derived
  by enumerating `class_map`'s keys in file order -- never from
  `sorted(os.listdir(...))` or any other filesystem-derived order. This is
  the single most important scientific-integrity guarantee in this layer:
  it is what lets `config/datasets/*.toml` remain the authoritative source
  of class definitions (Phase 3.5, `docs/DECISIONS.md`).
- **The on-disk structure must match the approved contract exactly.** Any
  missing approved class folder, any unexpected extra folder, any approved
  class folder with zero images, or any `num_classes` / `class_map` length
  mismatch raises `DatasetLoadingError` immediately. Nothing is
  auto-repaired or silently tolerated.
- **Splits are observed, never invented.** `official_split = false` (CCSN)
  always yields a single unsplit collection; requesting a split for such a
  dataset is an error. `official_split = true` (GCD) requires an explicit
  `split` argument -- `split=None` is treated as ambiguous, not "give me
  everything."
- **Determinism.** `samples` is sorted by relative POSIX path. Identical
  on-disk contents always produce an identical `DatasetIndex`, on any OS.
  There is no randomness and no seed anywhere in this layer.

## Layer B -- `src/dataset_loading.py`

A thin PyTorch adapter. Pillow is the only new dependency it introduces
beyond Layer A.

### Public API

- `CloudImageDataset(dataset_cfg, dataset_root, *, split=None,
  transform=None)` -- a `torch.utils.data.Dataset` subclass. The
  constructor calls `dataset_index.build_index(...)` once and stores the
  resulting `DatasetIndex`; all metadata properties (`dataset_key`,
  `split`, `classes`, `class_to_idx`, `class_display_names`, `num_classes`)
  are direct passthroughs -- Layer B never recomputes or duplicates that
  information.
- `CloudImageDataset.from_manifest_bundle(bundle, dataset_cfg, dataset_root,
  inventory, protocol, *, subset, cv_seed=None, fold=None, transform=None)`
  (Phase 4, M4) -- builds a dataset restricted to one subset (`"test"`,
  `"dev_pool"`, `"cv_fold"`, `"cv_train"`, `"cv_val"`) of a split-manifest
  bundle produced by `split_generator` (M3). Validates the bundle against the
  *current* config/inventory/protocol by calling
  `split_manifest.validate_manifest_bundle` and `validate_protocol_match`
  directly, then filters the same `DatasetIndex` machinery used by the plain
  constructor -- sample order and label assignment are untouched. See
  `docs/architecture/split_protocol.md` for the manifest side of this.
- `__len__()` -- number of samples in the index.
- `__getitem__(i)` -- lazily opens the image file with Pillow,
  `.convert("RGB")`, applies `transform` if supplied (exactly once), and
  returns `(image_or_transformed, label)`.

### Invariants

- **No preprocessing, no augmentation, no batching here.** If `transform`
  is `None`, the RGB `PIL.Image` is returned completely unchanged -- no
  resize, no normalization, no tensor conversion. Those are the explicit
  responsibility of a later phase, injected via `transform`.
- **Images are decoded lazily and never cached.** Each `__getitem__` call
  re-opens and re-decodes the file; nothing is memoized across calls. This
  keeps memory bounded regardless of dataset size and keeps behavior
  identical whether an index is touched once or a thousand times.
- **RGB conversion is unconditional.** Every image -- regardless of its
  on-disk mode -- is converted to RGB before being handed to `transform` (or
  returned). This matches the datasets' observed reality (both CCSN and GCD
  are all-RGB JPEG per their inventories) while making the contract explicit
  for any future edge case.
- **Decode failures are informative, not silent.** A corrupt or unreadable
  image raises `DatasetLoadingError` naming the dataset key, split, and
  relative path -- it does not skip the sample or substitute a placeholder.
- **No `DataLoader`, no shuffling, no seeding.** Batching and shuffling are
  out of scope for Phase 4 by explicit decision; they belong to the future
  training phase, which is expected to wrap a `CloudImageDataset` in its own
  seeded `DataLoader`.

## Known extension points

- **Preprocessing / augmentation** plugs in entirely through the
  `transform` argument of `CloudImageDataset`. Neither layer needs to change
  to support it.
- **Batching, shuffling, and seeding** are deliberately left to a future
  training phase, which will wrap a `CloudImageDataset` in a seeded
  `torch.utils.data.DataLoader`.
- **The merged dataset (`merged_v1`)** remains `draft` and out of scope;
  once its harmonization methodology and `class_map` are approved, it can be
  loaded through the same two layers unchanged.
- ~~Wiring a validated manifest into `CloudImageDataset`~~ -- done in Phase 4,
  M4 (`from_manifest_bundle`, above).

## Known limitation carried through, not fixed

**KI-001 (GCD train/test duplicate leakage)** is not addressed by either
layer. `build_index` and `CloudImageDataset` load GCD's official `train` and
`test` splits exactly as they exist on disk, duplicates included. See
`docs/KNOWN_ISSUES.md` -- remediation (de-duplication, split regeneration, or
acceptance) remains a separate, deferred scientific decision.
