# config/datasets/

Dataset profiles answer one question: **which scientific data am I analysing?**

## Profiles

- `merged_v1.toml` — merged CCSN + GCD (v1); referenced by the current experiments.
- `ccsn.toml`, `gcd.toml` — **independent** source datasets. They are deliberately
  kept separate and are Phase-3 discovery stubs: do not assume they share a layout.

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

**Written after discovery** (established facts / observations): `relative_data_path`,
`dataset_version`, `official_split`, `num_classes`.

**Deferred to the Scientific Dataset Approval step** (researcher-approved, never
inferred from directory names): `taxonomy`, the class mapping (`Ci -> Cirrus`, …),
`annotation_source`, and `independent_sampling_unit`.

> Observed image **resolutions** are not stored here — the inventory under
> `metadata/` is the authoritative source (e.g. CCSN is a mix of 400×400 and
> 256×256). The dataset owns `num_classes`; models stay class-agnostic.
