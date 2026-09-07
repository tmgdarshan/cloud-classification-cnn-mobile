# Decisions

Record of accepted scientific and methodological decisions. Newest first.

## Phase 4 — Methodological Rigor, Group-Aware Splitting & Cross-Source Harmonization
- **D-007: Five-Class Compatibility Taxonomy.**
  Ground-based cloud imagery from CCSN (single-station, high-resolution camera,
  fine-grained genera) and GCD (whole-sky imager, operational sky-condition categories)
  are mapped into a five-class compatibility taxonomy:
  1. `cumulus` <- CCSN `Cu`, GCD `1_cumulus`
  2. `altocumulus` <- CCSN `Ac`, `Cc`, GCD `2_altocumulus`
  3. `cirrus` <- CCSN `Ci`, `Cs`, GCD `3_cirrus`
  4. `stratocumulus` <- CCSN `Sc`, `St`, `As`, GCD `5_stratocumulus`
  5. `cumulonimbus` <- CCSN `Cb`, `Ns`, GCD `6_cumulonimbus`
  CCSN `Ct` (contrail), GCD `4_clearsky`, and GCD `7_mixed` are excluded.
  *Scientific Caveat*: This scheme is formally approved as a *cross-source compatibility
  taxonomy* for domain transfer and joint training analysis, not as a genus-preserving
  WMO taxonomy.
- **D-008: Group-Aware Repartitioning to Remediate Exact-Byte Redundancy & Cross-Split Contamination (Audit Record KI-001).**
  Official GCD train/test splits contain 156 exact-byte duplicate groups spanning partitions (KI-001).
  To ensure empirical validity, the official split is replaced in canonical manifests (`metadata/splits/gcd_6class_canonical.json`,
  `gcd_5class_canonical.json`, `harmonized_5bin_canonical.json`) with a deterministic grouped stratified
  holdout (`grouped_stratified_holdout` v1.0; `StratifiedGroupKFold`, seed 42) into an 80% development pool
  (internally 64% parameter training / 16% validation for model and checkpoint selection) and an immutable
  20% permanent test holdout. Exact-byte SHA-256 duplicate clusters share atomic group IDs and never
  straddle partitions (exact-byte grouping does not identify near-duplicate scenes). Full protocol details are in the paper, `report/cloud_classification_resnet.tex`.
- **D-009: Strict Evaluation Discipline & No Test-Set Model Selection.**
  The permanent test holdout is excluded from every selection decision and used only for
  final benchmark reporting. Within development data: the production runner
  (`src/run_harmonized.py`) restores the **minimum validation-loss** checkpoint over a fixed
  epoch budget; the tuner (`src/tune_resnet_family.py`) ranks trials by common **unsmoothed
  validation cross-entropy**, breaking ties with **validation macro-F1**.
- **D-010: Physically Conservative Data Augmentation.**
  `RandomVerticalFlip` is strictly prohibited in scientific baseline runs to preserve
  the vertical image orientation of ground-based cloud imagery (condensation base vs.
  convective top; ground-to-zenith camera framing).
- **D-011: Validated Tuned Baseline Designation.**
  The tuned ResNet family configuration (differential learning rates: backbone 5e-5,
  head 5e-4; AdamW; weight decay 1e-2; label smoothing 0.0; cosine annealing) is
  formally designated as a *validated tuned baseline* for comparative benchmarks,
  not claimed as the global "best" or optimal hyperparameters. It was selected for
  stable convergence across all three architectures rather than peak per-model
  validation accuracy (on the ResNet-18 sweep several other trials score higher).
  Formal search-space characterization and multi-trial ablation are planned for Phase 5.
- **D-012: Official Production Pipeline Consolidation.**
  The research benchmarking pipeline is consolidated around `src/run_harmonized.py`
  as the single canonical execution engine (supporting single-source in-domain baselines,
  cross-source transfer, and harmonized joint training with source-balanced
  sampling) and `src/tune_resnet_family.py` for hyperparameter optimization dynamics.
  Earlier exploratory prototype scripts (stage-1/2/3 engines, the SG-HCV
  engine) have been removed; see git history if needed.
  Use the term **joint** (joint model / joint training), not "merged".

## Phase 4 (M3) — Split-Generation Infrastructure for Datasets Without an Official Split
- **Approved protocol name/version:** `grouped_stratified_holdout` v1.0. Applies to
  any dataset whose config has `official_split = false` (currently CCSN only;
  GCD already has an official split and this protocol explicitly refuses to
  run against it).
- **Permanent holdout:** a class-stratified 20% test partition (`test_fraction =
  0.2`), generated once with **canonical seed 42**, write-once and version
  controlled — never regenerated for individual experiments.
- **Development pool:** the remaining ~80%.

  > **Status note (2026-09-06):** the shipped canonical manifests
  > (`metadata/splits/*_canonical.json`, built by
  > `scripts/build_canonical_manifests.py`) use the single grouped stratified
  > holdout above with **one fixed development/validation partition at seed 42**
  > (internal fold 0 = validation). The 5-fold cross-validation with published
  > variance seeds `{7, 21, 42, 84, 168}` described below is **reserved
  > generator infrastructure — it was never exercised for the benchmarks** and
  > is not part of the shipped protocol. `src/split_protocol.py` keeps the
  > `num_folds` / `variance_seeds` fields for that infrastructure only.

  (Reserved design: stratified 5-fold cross-validation drawn from the
  development pool only, independently for each variance seed; variance seeds
  affect only the dev-pool fold partition and never regenerate the permanent
  test holdout. Nested CV is reserved for explicit, separately researcher-approved
  comparisons.)
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
  from, the approved `class_map`, and any cross-class resolution applied.
- **CCSN cross-class resolution (applied).** Applying this protocol surfaces 3 real
  cross-class duplicate groups already present in `metadata/ccsn_inventory.json`
  (`Ac/Ac-N186.jpg`↔`As/As-N139.jpg`, `Ac/Ac-N202.jpg`↔`As/As-N175.jpg`,
  `Cc/Cc-N179.jpg`↔`Cs/Cs-N244.jpg`). These six images are **excluded** from all
  canonical manifests (`exclude_group`; see `excluded_samples` in
  `ccsn_11class_canonical.json`), leaving 2,537 in the CCSN 11-class view. The
  exclusion is the researcher-approved resolution recorded under D-007 / Phase 3.5;
  the "halt until resolution" behavior above is the default, and the resolution
  has been supplied.

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
  11-class genera taxonomy — this distinction protects the deferred joint-dataset
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
- **Joint dataset is deferred.** It becomes an official dataset only after the
  harmonization methodology is designed and approved (delivered in Phase 4).

## Phase 2 / 2.5 — Configuration domain model
- **Configuration is a domain model** of independent concepts (environment,
  dataset, model, training, evaluation, experiment). Experiments compose the
  others by reference and are never named after a single architecture.
- **The dataset owns `num_classes`;** models are class-agnostic.
