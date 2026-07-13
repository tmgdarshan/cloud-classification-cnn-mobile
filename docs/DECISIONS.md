# Decisions

Record of accepted scientific and methodological decisions. Newest first.

## Phase 3.5 — Scientific Dataset Approval
- **CCSN and GCD specifications are approved and frozen.** Both configs move to
  `approval_status = "approved"`. The previously deferred fields — `taxonomy`,
  `class_map`, `annotation_source`, `independent_sampling_unit` — are now
  researcher-approved and recorded in `config/datasets/{ccsn,gcd}.toml`, together
  with frozen `provenance` (primary publication + dataset repository).
- **CCSN taxonomy:** "11-class cloud taxonomy (10 WMO genera + contrail)".
  `class_map` maps the 11 verbatim folder tokens to WMO genera; **`Ct` = Contrail**
  is the one non-WMO, human-made category (not cirrostratus, which is `Cs`).
  Provenance: Zhang et al. (2018), *CloudNet*, Geophysical Research Letters 45,
  8665–8672. `annotation_source` deliberately avoids "expert-labeled": the
  authors' own materials state the categories follow the WMO genera-based
  classification recommendation but do not describe an expert-labeling process.
- **GCD is the TJNU Ground-based Cloud Dataset.** Taxonomy: "7-class operational
  sky-condition taxonomy". **GCD is a coarsened WMO scheme** — each numbered
  class groups multiple WMO genera (e.g. `2_altocumulus` = *altocumulus and
  cirrocumulus*; `3_cirrus` = *cirrus and cirrostratus*; `5_stratocumulus` =
  *stratocumulus, stratus and altostratus*; `6_cumulonimbus` = *cumulonimbus and
  nimbostratus*), plus a non-cloud `4_clearsky` (cloudiness ≤ 10%) and a
  `7_mixed` category. It is therefore **not interchangeable** with CCSN's
  11-class genera taxonomy — this distinction protects the deferred merged-dataset
  work. Provenance: Liu et al. (2022), *Ground-based Remote Sensing Cloud
  Classification via Context Graph Attention Network*, IEEE TGRS 60, art.
  5602711 (DOI 10.1109/TGRS.2021.3063255).
- **`independent_sampling_unit = "image"` for both datasets.** The datasets
  provide no higher-level grouping metadata (camera, capture session, time
  sequence), so the image is the unit the data actually supports; a scene-level
  unit was deliberately **not** invented. For GCD this declaration does **not**
  imply statistical independence between images — KI-001 documents observed
  duplicate-related dependence. Declaring the unit is not KI-001 remediation.
- **No datasets, splits, inventories, or resolutions were modified.** This phase
  approved specifications only.

## Phase 3 — Dataset Discovery & Inventory
- **Dataset metadata policy.** Dataset configs record only established facts
  after discovery: `relative_data_path`, `dataset_version`, `official_split`,
  `num_classes`. `taxonomy`, class mapping, `annotation_source`, and
  `independent_sampling_unit` are deferred to the Scientific Dataset Approval
  step — researcher-approved, never inferred from directory names.
- **Inventory is authoritative for observed resolutions**, not the dataset
  config. No `expected_resolution` is stored; CCSN has multiple native
  resolutions, recorded accurately in `metadata/ccsn_inventory.json`.
- **512×512 standardization is a preprocessing policy**, not a dataset property.
  It will live in the preprocessing/training configuration in a later phase.
- **GCD train/test leakage is documented, not fixed** (KI-001). The dataset and
  its splits are not modified; remediation is a separate scientific decision.
- **Merged dataset is deferred.** It becomes an official dataset only after the
  harmonization methodology is designed and approved.

## Phase 2 / 2.5 — Configuration domain model
- **Configuration is a domain model** of independent concepts (environment,
  dataset, model, training, evaluation, experiment). Experiments compose the
  others by reference and are never named after a single architecture.
- **The dataset owns `num_classes`;** models are class-agnostic.
