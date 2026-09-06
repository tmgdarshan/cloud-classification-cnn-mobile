# Split Protocol Architecture

Describes the split-generation infrastructure introduced in Phase 4 (M3).
Like `docs/architecture/dataset_loading.md`, this is a living reference. See
`docs/DECISIONS.md` for the approved scientific methodology and
`docs/PROJECT_GUIDE.md` for how it relates to the canonical manifests the
production runner actually consumes.

> Note: the production benchmark manifests under `metadata/splits/` are built
> by `scripts/build_canonical_manifests.py`, not by this generator. This
> module remains the tested, dataset-agnostic infrastructure for the
> `grouped_stratified_holdout` protocol; see `docs/DECISIONS.md`.

## Purpose

Some datasets (currently CCSN) ship with no official train/test split.
Inventing one is a scientific decision, not an engineering default -- so this
infrastructure exists to turn an *approved, explicit* split protocol into a
deterministic, version-controlled artifact (a "manifest") that a future
loading/training phase can consume, without the generator or validator ever
having to know *which* dataset it's working with.

## Three components, one direction of dependency

```
config/datasets/<key>.toml      metadata/<key>_inventory.json      SplitProtocol
 (approved dataset config)         (authoritative inventory)        (spec object)
              \                          |                          /
               \                         |                         /
                v                        v                        v
                    src/split_generator.py  (Generator)
                              |
                              v  in-memory manifest bundle {test, dev_pool, cv}
                              |
                    src/split_manifest.py   (Validator)
                              |
                              v
              (future) CloudImageDataset manifest-consumption (not yet built)
```

The Generator depends on `dataset_index.build_index` (Layer A) to know which
samples and classes exist, and on `split_protocol.SplitProtocol` for the
numeric parameters of the split. The Validator depends on `split_protocol`
for its shared inventory-fingerprint helper and on a `DatasetIndex` to check
sample-level integrity. Neither component depends on `dataset_loading.py`;
wiring a validated manifest into `CloudImageDataset` is explicitly deferred to
a later milestone.

## `src/split_protocol.py` -- the specification object

**All protocol constants live here, never inline in generator logic.**
`SplitProtocol` is a frozen dataclass (`name`, `version`, `test_fraction`,
`canonical_seed`, `num_folds`, `variance_seeds`) with basic bounds validation
in `__post_init__`. `GROUPED_STRATIFIED_HOLDOUT_V1` is the one protocol
instance (`docs/DECISIONS.md`, Phase 4 M3): 20% permanent test at seed 42.
The `num_folds` / `variance_seeds` fields exist for the generator
infrastructure's optional dev-pool fold partitions; **the shipped benchmark
manifests use a single fixed partition — no cross-validation is run** (see
`scripts/build_canonical_manifests.py`).

This module also owns `compute_inventory_fingerprint(inventory)` -- a pure,
sort-key-independent sha256 fingerprint of an inventory dict. Both the
generator (to stamp a manifest's provenance) and the validator (to
independently recompute and check it) import this single function, so the
notion of "which inventory produced this manifest" has one definition.

## `src/split_generator.py` -- dataset-agnostic generation

**`generate_split_manifests(dataset_cfg, dataset_root, inventory, protocol,
*, cross_class_resolution=None)`** is driven entirely by its four inputs.
No dataset key or name is ever compared against a literal string anywhere in
this module -- `tests/unit/test_split_generator.py` enforces this directly
by asserting the substrings `"ccsn"` / `"gcd"` do not appear in the module's
source at all.

### Invariants

- **Protocol precondition, not a special case.** The generator refuses to
  run against a dataset whose config has `official_split = true` -- this is
  a property of the *protocol* (it fills a gap for datasets with no
  official split), not a rule about any particular dataset.
- **Duplicate groups are read from the inventory, never recomputed.** The
  generator trusts `metadata/<key>_inventory.json` as the authoritative,
  already-computed record of duplicates (`dataset_discovery`, Phase 3) and
  cross-checks that every referenced path still exists in the current
  `DatasetIndex`, raising loudly on drift instead of silently rescanning.
- **Cross-class duplicate groups halt by default.** A duplicate group whose
  members span more than one class folder is a label-integrity anomaly, not
  routine duplication. Unless the caller supplies a `cross_class_resolution`
  covering every such group, generation raises `SplitGenerationError` and
  produces no manifest at all. The only supported resolution action is
  `"exclude_group"` -- the generator never encodes a semantic judgment about
  which of two conflicting labels is correct; any real resolution decision
  belongs to the researcher.
- **Duplicate-group atomicity.** Every within-class duplicate group is
  collapsed into one atomic unit before any partitioning happens: a unit is
  always placed entirely in test or entirely in dev, and (if in dev) entirely
  in one CV fold. This makes the approved test fraction a *target*, not a
  guarantee, once atomicity is enforced -- documented, not silently
  approximated.
- **Determinism via local RNGs only.** Every shuffle uses a fresh
  `random.Random(seed)` instance, never the global `random` module. The
  holdout split uses `protocol.canonical_seed`; each CV fold partition uses
  its own `variance_seed`, applied independently per class so that per-class
  stratification order never leaks between classes or between seeds.
- **Provenance ties a manifest to its exact origin.** Every manifest records
  `dataset_key`, `protocol_name`/`version`, `inventory_fingerprint`,
  `class_map`, `num_samples_total`, `cross_class_resolution`, and
  `excluded_samples` -- enough to detect, later, whether the dataset,
  inventory, or approved class definitions have drifted since generation.

## `src/split_manifest.py` -- validation only

**`validate_manifest_bundle(bundle, index, inventory)`** checks a generated
bundle's schema, sample-level integrity, and provenance consistency. It
makes no scientific decisions and never repairs a manifest -- a bundle either
passes whole or fails with a specific explanation. Checks include: required
keys and manifest types; every provenance field matches the *current*
`DatasetIndex` and inventory (re-deriving the fingerprint rather than trusting
the stored one); test and dev_pool are disjoint and together exactly
partition the non-excluded samples; every duplicate group stays within one
partition; and every CV fold assignment exactly covers the dev pool with
in-range fold indices and undivided duplicate groups.

## `src/split_manifest.py` -- consumption-time helpers (Phase 4, M4)

Two additional functions support consuming a manifest bundle, without
touching `validate_manifest_bundle` itself:

- **`validate_protocol_match(bundle, protocol)`** checks the bundle's
  provenance `protocol_name`/`protocol_version` against a given
  `SplitProtocol`, guarding against silently loading a manifest generated
  under a different (e.g. superseded) protocol version.
- **`select_subset(bundle, subset, *, cv_seed=None, fold=None)`** resolves
  which sample paths belong to one of five subsets -- `"test"`, `"dev_pool"`,
  `"cv_fold"`, `"cv_train"`, `"cv_val"` (`"cv_fold"` and `"cv_val"` return the
  same samples; `"cv_train"` is their complement within the dev pool for that
  `(cv_seed, fold)`). This stays framework-independent (no torch/PIL) so
  subset interpretation is testable without ever loading an image.

`CloudImageDataset.from_manifest_bundle` (`src/dataset_loading.py`, Phase 4
M4) is the sole consumer of these: it builds a fresh `DatasetIndex`, calls
`validate_manifest_bundle` and `validate_protocol_match` directly against the
*current* dataset config, inventory, and protocol (live re-validation at the
point of use, not a trust-once-and-forget check), then restricts its sample
list via `select_subset` -- sample order and label assignment come entirely
from the existing `DatasetIndex`, unchanged. The loader never generates,
modifies, or re-derives a split; it only consumes the result of M3's
generator and validator.

## Known extension points

- **Actually writing manifests to `metadata/splits/<dataset_key>/`** on real
  data (paralleling `scripts/discover_dataset.py`'s pattern of a pure
  function plus a thin CLI writer) has not been built yet; `to_json()` in
  `split_generator.py` provides the deterministic serialization a future
  script would use.
- **Resolving CCSN's 3 real cross-class duplicate groups** is a separate,
  still-open researcher decision (`docs/DECISIONS.md`, Phase 4 M3) -- the
  infrastructure supports recording a resolution, but none has been supplied
  yet, so generating a manifest from the real CCSN data will legitimately
  halt today.
