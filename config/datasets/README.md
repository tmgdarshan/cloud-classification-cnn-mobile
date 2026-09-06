# config/datasets/

Dataset profiles answer one question: **which scientific data am I analysing?**

## Profiles

| File | `approval_status` | Notes |
|---|---|---|
| `ccsn.toml` | `approved` | CCSN source dataset, 11-class. Scientific spec frozen (Phase 3.5). |
| `gcd.toml` | `approved` | GCD source dataset, 7-class. Scientific spec frozen (Phase 3.5). |
| `gcd_5class.toml` | `approved` | GCD cloud-only view (drops `4_clearsky`, `7_mixed`). |
| `gcd_6class.toml` | `approved` | GCD view dropping `7_mixed` only. |
| `harmonized_5bin.toml` | `approved` | Pointer profile to the canonical harmonized manifest; five-class compatibility taxonomy (D-007). |
| `merged_v1.toml` | `superseded` | Early exploratory merge. Superseded by `harmonized_5bin`; retained for config-inspection tests. |

CCSN and GCD are **independent** — do not assume they share a layout or a
taxonomy (CCSN: 11 WMO genera + contrail; GCD: a coarsened 7-class scheme
grouping multiple genera + clear sky + mixed).

## Keys

| Key | Meaning |
|-----|---------|
| `name`, `key` | Human-readable name; short stable identifier used by references. |
| `approval_status` | `draft` / `validation` / `approved` / `superseded`. |
| `sources` | Source dataset(s) this is built from. |
| `dataset_version` | Version of this dataset definition. |
| `relative_data_path` | Location under `CLOUD_DATA_ROOT` (confirmed by discovery). |
| `official_split` | Whether the dataset ships a predefined train/test split (observed). |
| `num_classes` | Number of classes — **owned by the dataset**, not the model (observed). |
| `excluded_class_folders` | Class folders dropped for a derived view (e.g. `gcd_5class`). |
| `taxonomy` | Human-readable name of the class scheme (researcher-approved). |
| `class_map` | Table mapping verbatim folder tokens → approved class names. |
| `annotation_source` | How/by whom the dataset was labeled (researcher-approved). |
| `independent_sampling_unit` | Unit that must not be split across train/test. |
| `manifest_path` | (harmonized) canonical manifest this profile points at. |
| `[provenance]` | Frozen `primary_publication` + `dataset_repository`. |

**Written after discovery** (observations): `relative_data_path`,
`dataset_version`, `official_split`, `num_classes`.
**Approved in Phase 3.5** (researcher-approved, never inferred from directory
names): `taxonomy`, `class_map`, `annotation_source`,
`independent_sampling_unit`, `provenance`.

> Observed image **resolutions** are not stored here — the inventory under
> `metadata/` is authoritative (CCSN is a mix of 400×400 and 256×256). The
> dataset owns `num_classes`; models stay class-agnostic.
