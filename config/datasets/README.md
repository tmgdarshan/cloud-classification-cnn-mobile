# config/datasets/

Dataset profiles answer one question: **which scientific data am I analysing?**

## Profiles

- `merged_v1.toml` — merged CCSN + GCD (v1); referenced by the current experiments.
  Still `draft` (its harmonization methodology is deferred).
- `ccsn.toml`, `gcd.toml` — **independent** source datasets, deliberately kept
  separate: do not assume they share a layout or a taxonomy. Their scientific
  specifications are **approved and frozen** (Phase 3.5); `approval_status =
  "approved"`.

## Keys

| Key | Meaning |
|-----|---------|
| `name` | Human-readable dataset name (required). |
| `key` | Short stable identifier used by references. |
| `approval_status` | `draft` / `validation` / `approved`. |
| `sources` | Source dataset(s) this is built from. |
| `dataset_version` | Version of this dataset definition. |
| `relative_data_path` | Location under `CLOUD_DATA_ROOT` (confirmed by discovery). |
| `official_split` | Whether the dataset ships a predefined train/test split (observed). |
| `num_classes` | Number of classes — **owned by the dataset**, not the model (observed). |
| `taxonomy` | Human-readable name of the class scheme (researcher-approved). |
| `class_map` | Table mapping verbatim folder tokens → approved class names. |
| `annotation_source` | How/by whom the dataset was labeled (researcher-approved). |
| `independent_sampling_unit` | Unit that must not be split across train/test. |
| `[provenance]` | Frozen `primary_publication` + `dataset_repository`. |

**Written after discovery** (established facts / observations): `relative_data_path`,
`dataset_version`, `official_split`, `num_classes`.

**Approved in the Scientific Dataset Approval step** (Phase 3.5 — researcher-approved,
never inferred from directory names): `taxonomy`, `class_map` (`Ci -> Cirrus`, …),
`annotation_source`, `independent_sampling_unit`, and frozen `provenance`. CCSN and
GCD carry these fields; note their taxonomies differ (CCSN: 11 WMO genera + contrail;
GCD: a coarsened 7-class scheme grouping multiple genera + clear sky + mixed).

> Observed image **resolutions** are not stored here — the inventory under
> `metadata/` is the authoritative source (e.g. CCSN is a mix of 400×400 and
> 256×256). The dataset owns `num_classes`; models stay class-agnostic.
