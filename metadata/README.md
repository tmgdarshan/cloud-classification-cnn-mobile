# metadata/

Generated **metadata artifacts** — small, deterministic, and tracked in Git so
the documented record travels with the repository and can be diffed against
future scans.

## Inventories (Phase 3)

`ccsn_inventory.json`, `gcd_inventory.json` — a machine-readable record of *what
the datasets actually contain*: structure, splits, per-class counts, observed
dimensions/formats, corrupt files, and exact-byte duplicate / cross-split
findings — derived from the data under `CLOUD_DATA_ROOT` by
`src/dataset_discovery.py`.

## Canonical split manifests (Phase 4)

`splits/*_canonical.json` — the **single source of truth** for every current
benchmark partition. See [`splits/README.md`](splits/README.md).

> The joint (harmonized) dataset is inventoried indirectly: its manifest is
> derived from the approved CCSN and GCD source labels, not from a separate
> raw scan.
