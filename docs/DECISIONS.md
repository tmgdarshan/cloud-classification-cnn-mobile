# Decisions

Record of accepted scientific and methodological decisions. Newest first.

## Phase 4 (M3) — Split Protocol for Datasets Without an Official Split
- **Approved protocol name/version:** `stratified_holdout_cv` v1.0. Applies to
  any dataset whose config has `official_split = false` (currently CCSN only;
  GCD already has an official split and this protocol explicitly refuses to
  run against it).
- **Permanent holdout:** a stratified 20% test partition (`test_fraction =
  0.2`), generated once with **canonical seed 42**, write-once and version
  controlled — never regenerated for individual experiments.
- **Development pool:** the remaining ~80%. Stratified **5-fold**
  cross-validation is drawn from the development pool only, independently for
  each of five **published variance seeds**: `{7, 21, 42, 84, 168}`. Variance
  seeds affect only the fold partition of the development pool; they never
  regenerate the permanent test holdout. Nested CV is reserved for explicit,
  separately researcher-approved comparisons and is not part of this
  protocol.
- **Duplicate policy:** within-class duplicate groups (from the dataset's
  authoritative inventory) are treated as a single atomic unit and always
  placed together in the same partition/fold — a duplicate group is never
  split across test/dev or across folds. **Cross-class duplicate groups are
  treated as a label-integrity anomaly, not an ordinary duplicate.** By
  default, detecting any cross-class duplicate group **halts split
  generation entirely — no manifest is produced** — until an explicit,
  researcher-approved resolution is supplied. The only supported resolution
  action is excluding the group from the assignable pool (`exclude_group`);
  the protocol deliberately never encodes a semantic interpretation of which
  label in a cross-class group is "correct." Any resolution used is recorded
  in the manifest's provenance.
- **Manifests are first-class, version-controlled research artifacts**,
  equivalent in status to an official dataset train/test split: deterministic,
  immutable once approved, and consumed (not created or modified) by the
  loader. Each manifest's provenance records the dataset key, the protocol
  name/version, a content fingerprint of the exact inventory it was generated
  from, the approved `class_map`, and any cross-class resolution applied —
  tying every manifest to the precise dataset config + inventory + protocol
  that produced it.
- **CCSN today:** applying this protocol surfaces 3 real cross-class
  duplicate groups already present in `metadata/ccsn_inventory.json`
  (`Ac/Ac-N186.jpg`↔`As/As-N139.jpg`, `Ac/Ac-N202.jpg`↔`As/As-N175.jpg`,
  `Cc/Cc-N179.jpg`↔`Cs/Cs-N244.jpg`). Per the policy above, generating a real
  CCSN manifest **halts** until the researcher supplies an explicit
  resolution for these three groups — this is expected, correct behavior,
  not a defect.
- **No datasets, splits, inventories, or approved specifications were
  modified.** This decision approves a split-generation methodology and its
  infrastructure only; it does not itself resolve the 3 cross-class groups,
  run the generator against real data, or write any manifest to disk.

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
