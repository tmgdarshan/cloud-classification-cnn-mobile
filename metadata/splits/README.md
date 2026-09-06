# metadata/splits/

The **canonical split manifests** — the single source of truth for every
current benchmark partition. Built once by
`scripts/build_canonical_manifests.py`; consumed (never modified) by
`src/run_harmonized.py` and `src/tune_resnet_family.py`.

| File | View | Total | Classes |
|---|---|---:|---:|
| `ccsn_11class_canonical.json` | CCSN, 11-class | 2,537 | 11 |
| `gcd_6class_canonical.json` | GCD, drop `7_mixed` | 18,045 | 6 |
| `gcd_5class_canonical.json` | GCD, drop `4_clearsky` + `7_mixed` | 14,306 | 5 |
| `harmonized_5bin_canonical.json` | CCSN + GCD, five-class compatibility taxonomy | 16,643 | 5 |

## Protocol

`grouped_stratified_holdout` v1.0 (D-008): a class-stratified split at
**seed 42** into a fixed **train (64%) / validation (16%) / test holdout (20%)**
partition, with exact-byte SHA-256 duplicate clusters bound to atomic
`group_id`s so a cluster never straddles partitions. `StratifiedGroupKFold`
derives the split; internal fold 0 is the validation set. **No cross-validation
experiment is run.**

## Manifest fields

`dataset_key`, `taxonomy_name`, `protocol`, `seed`, `classes`, `class_to_idx`,
`num_classes`, per-view exclusion/drop lists, `mappings` (harmonized only),
`summary` (counts), `samples` (per-image `path` / `class` / `label` /
`group_id` / `split` / `fold`), and `sha256` — the SHA-256 of the `samples`
list, so partition integrity can be re-verified independently.

## Other files

`ccsn/`, `gcd_6class/` — `split_manifest_bundle.json` outputs from the
dataset-agnostic generator infrastructure (`src/split_generator.py`); not
consumed by the production runner.
