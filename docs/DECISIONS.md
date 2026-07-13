# Decisions

Record of accepted scientific and methodological decisions. Newest first.

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
