# metadata/

Generated **metadata artifacts** — reproducible dataset inventories produced by
Phase 3 (Dataset Discovery, Validation & Inventory): `ccsn_inventory.json` and
`gcd_inventory.json`.

These are not datasets and not configuration. They are a machine-readable record
of *what the datasets actually contain* (structure, splits, counts, observed
dimensions/formats, corrupt and duplicate/cross-split findings), derived from the
data under `CLOUD_DATA_ROOT`.

Unlike model checkpoints, these artifacts are small, deterministic, and **tracked
in Git** so the documented inventory travels with the repository and can be
diffed against future scans to detect dataset changes.

> The merged dataset is intentionally not inventoried yet (deferred until its
> harmonization methodology is approved).
