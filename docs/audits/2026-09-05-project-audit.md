# Project audit - 5 September 2026

**Assessment: useful research results and verified exact-duplicate isolation, but important scientific, execution, provenance, and reporting defects remain. Phase 4 should not yet be treated as a fully reproducible, publication-ready baseline.**

This is an audit of the existing working tree against its documentation, rather than approval of new scientific decisions. No training code, datasets, split assignments, checkpoints, or existing research documents were changed. The audit adds this report and diagnostic evidence under `artifacts/audit_2026-09-05/`.

The starting Git HEAD was `ea529c6e`. Many Phase 4 files and all four canonical manifests were untracked; several older source files were already deleted in the working tree. These pre-existing changes were preserved. Current code inspection cannot establish exactly which earlier version generated every saved result.

**Coverage and verification**

Reviewed the README walkthrough, governance documents, decision log, dashboard, issues, architecture notes, phase handoffs, configuration profiles, source modules, scripts, tests, inventories, split manifests, experiment registry, saved metrics, checkpoints, notebook, LaTeX manuscript, and all 12 rendered PDF pages. Selected deleted legacy training and merging code was inspected through `git show HEAD:...`. No separately named walkthrough file or SQL/SQLite database was found; the project's data store is image directories plus JSON metadata and experiment records. The notebook contains one empty code cell.

| Check | Observed result |
|---|---|
| Existing test suite | `111 passed in 8.01s` with `.venv/Scripts/python.exe -m pytest tests/ -q` |
| Python syntax | All 35 Python files under `src`, `scripts`, and `tests` parsed |
| TOML syntax | Three files fail parsing due to a UTF-8 BOM; see finding 9 |
| Raw data | All 21,543 images fully decoded and SHA-256 hashed; zero decode failures |
| Inventories | Counts, total bytes, dimensions, and exact duplicate groups match both stored inventories |
| Four canonical manifests | All referenced images exist; labels/fold conventions checked; no repeated source/path rows; sample hashes match |
| Duplicate isolation | No actual image SHA-256 crosses partitions in any canonical manifest; no declared duplicate group crosses folds |
| Cross-source duplicates | Zero exact-byte hashes shared between CCSN and GCD |
| Checkpoints | All 13 `.pth` files loaded with `weights_only=True`; tensor values finite |
| Registry | Four records; all four checkpoint hashes match; three manifest hashes match; one record has no manifest |
| Metrics | All 31 atmospheric metrics JSON files have internally consistent accuracy, balanced accuracy, and macro-F1 aggregates within rounding tolerance |
| Manuscript production table | All 63 checked point estimates match saved summaries; 34 of their CI cells differ beyond one-decimal rounding tolerance |
| PDF | All 12 pages rendered with installed Ghostscript and visually inspected; tables visibly clipped on pages 8, 9, and 11 |

The initial sandbox could not launch the virtual environment's base interpreter. Tests and diagnostics succeeded after approved access outside the sandbox. This was an access limitation, not a broken Python installation. Recorded runtime: Python 3.12.10, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124; remaining versions are in the evidence JSON.

No models were retrained and no new inference was run on permanent test images. Hash and aggregate consistency do not prove that a checkpoint reproduces a saved prediction table. Per-image predictions, original run commands, and complete historical runtime records are not available for that stronger verification. Near-duplicate, session, temporal, or camera-level independence was not established by exact hashing.

**The actual data**

| Dataset | Raw images | Resolutions | Exact duplicate groups | Raw cross-split groups |
|---|---:|---|---:|---:|
| CCSN | 2,543 | 2,332 at 400x400; 211 at 256x256 | 17 | Not applicable |
| GCD | 19,000 | All 512x512 | 159 | 156 |

CCSN contains three cross-class duplicate pairs, comprising six images: Ac/As twice and Cc/Cs once. All six are excluded from the canonical manifests. The other 14 CCSN duplicate groups remain atomic. GCD has no exact cross-class duplicate group in the current files.

| Canonical manifest | Classes | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|---:|
| CCSN 11-class | 11 | 1,622 | 407 | 508 | 2,537 |
| GCD 6-class | 6 | 11,548 | 2,888 | 3,609 | 18,045 |
| GCD 5-class | 5 | 9,155 | 2,289 | 2,862 | 14,306 |
| Harmonized 5-bin | 5 | 10,650 | 2,663 | 3,330 | 16,643 |

The harmonized pool contains 2,337 CCSN and 14,306 GCD images. Its test pool is 468 CCSN plus 2,862 GCD images. GCD clear sky comprises 3,739/19,000 = 19.68% of the raw dataset; GCD mixed has 955 images; CCSN contrails have 200 images.

**Procedure-by-procedure assessment**

| Step used so far | Assessment |
|---|---|
| Repository foundation and governance | Clear policy exists, but Phase 4 working-tree state and documentation no longer agree |
| Dataset discovery | Actual exact-byte duplicate results independently verified; no perceptual detection implemented |
| Dataset approval and taxonomy | Five-bin mappings agree with the recorded compatibility decision; sensor claims and parts of the manuscript provenance do not |
| Generic index, loader, split generator, validator | Useful tested infrastructure with explicit invariants; newer production runners largely bypass it |
| Canonical split construction | Within-manifest exact-group isolation verified; immutability, provenance, and cross-manifest holdout consistency incomplete |
| SG-HCV execution | Current runner broken; selection and seed behavior differ from documented methodology |
| Earlier Stage 1/2/3 scripts | Fixture/prototype paths remain active; they are not equivalent to canonical real-data experiments |
| Hyperparameter tuning | Ten trials per architecture recorded, but comparison metric changes with label smoothing; seeds not established by the tuning code |
| Harmonized training | Validation-based checkpointing and separate test datasets present; CPU snapshot and silent image failures need repair |
| Source balancing | Training source probability is balanced in expectation; validation remains sample-weighted |
| Evaluation | Aggregate arithmetic consistent; CI reporting, dependence assumptions, and claims of significance need correction |
| Experiment records | Saved hashes help, but records are incomplete, mutable, and can mix incompatible runs |
| Walkthrough and manuscript | Important factual/methodological discrepancies and broken reproduction steps |
| Mobile/HPC readiness | Planning exists; measured mobile inference, quantization evidence, and an offline HPC execution path are not demonstrated |

**Findings, in correction priority order**

**1. Critical - The manuscript claims methods and dataset properties that the evidence does not support.**

`report/cloud_classification_resnet.tex:96` describes dHash plus SSIM thresholds and near-duplicate connected components. `src/dataset_discovery.py:48` and its scan loop implement file SHA-256 only. The 156 GCD leakage groups are exact-byte groups. No implementation or result archive for the described perceptual audit was found. The manuscript also says six conflicting pairs, but the evidence contains three pairs/six images, with different class examples.

At manuscript lines 67-113, raw GCD is described as 10,000 images, contrails as 40, mixed clouds as 439, and clear sky as 37.39%. These conflict with the verified counts above. CCSN's acronym is also incorrectly expanded as “Nematocumulus” in README.

The GCD authors describe 19,000 images, collected across nine Chinese provinces in 2019-2020, with 10,000 training and 9,000 test images. Their repository does not establish the manuscript's single Wuxi fisheye-station description. Clear sky allows cloudiness up to 10%, rather than necessarily containing zero clouds. The manuscript's GCD bibliography entry also differs from the dataset's prescribed citation. [Primary GCD source](https://github.com/shuangliutjnu/TJNU-Ground-based-Cloud-Dataset).

CCSN's primary citation is *CloudNet*, Geophysical Research Letters, DOI 10.1029/2018GL077787, not the manuscript's residual-learning/IEEE GRSL entry. Its public description calls the dataset Cirrus Cumulus Stratus Nimbus. The public release description specifies 256x256 images, whereas the local inventory has mixed resolutions: document this local-release provenance discrepancy rather than overriding the observed data. [CCSN authors' repository](https://github.com/upuil/CCSN-Database).

The CloudNet paper describes labeling informed by meteorological experts' experience; the older handoff's blanket statement that the authors' own materials do not support expert involvement is too strong. Correct that in a new record rather than rewriting the frozen handoff. [CloudNet primary paper](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2018GL077787).

**Correction:** reconcile manuscript methods, counts, citations, and acquisition descriptions with evidence before publication. Treat camera-distortion explanations as hypotheses unless acquisition metadata and controlled experiments establish them. Do not present an unimplemented near-duplicate audit as completed.

**2. High - Hyperparameter trials are ranked using different validation objectives.**

`src/tune_resnet_family.py:290` constructs cross entropy with each trial's label smoothing; line 343 uses the same criterion for validation; line 583 ranks all trials by that validation loss. Varying smoothing changes the objective, so these loss values are not directly comparable as a common selection score. A local reproduction with identical logits/labels gave loss 0.0707 at smoothing 0, 0.3907 at 0.1, and 0.5507 at 0.15. This matches the documented change in target distributions under label smoothing. [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html).

The recorded sweep selects the no-smoothing trial for all three architectures. That is not proof that removing smoothing improves discrimination. Training loss curves across differently smoothed trials have the same comparability limitation.

**Correction:** use one researcher-approved validation scoring rule for every trial, independent of its training loss; for example, unsmoothed validation NLL or a fixed macro-F1 criterion. Reassess selection before calling these configurations optimal. The reported test point estimates need not be numerically wrong for this selection claim to be unsupported.

**3. High - The canonical SG-HCV CLI cannot execute its advertised real-data path.**

At `scripts/run_sghcv_experiments.py:87`, the loader returns four values, making the six-value return on line 88 unreachable. Main performs both a four-value unpack and a six-value unpack at lines 120-121. The exact loader function was executed in isolation and reproduced `not enough values to unpack (expected 6, got 4)`. Fixture mode also passes `splits` and `folds` without defining them. `src/train_sghcv.py` itself has no executable main entry point, despite the dashboard naming it as a runner.

**Correction:** use one consistent six-field contract, initialize fixture split/fold arguments, and test both CLI branches before launching training. This is an execution failure separate from the validity of the manifests.

**4. High - SG-HCV model selection differs from the documented procedure.**

`src/train_sghcv.py:282` selects by validation accuracy, not minimum validation loss. It chooses the highest-accuracy model across different folds for final testing, rather than a documented final refit or ensemble. OOF predictions come from best-accuracy epochs, but the fold table at line 296 records final-epoch metrics. These summarize different checkpoints. Selecting an epoch using the same fold that supplies its OOF score also makes that score a model-selection estimate, not an untouched outer evaluation.

Only seed 42 folds are used by this engine; the approved variance seeds `{7,21,42,84,168}` are not executed. No training RNG seeding occurs in this runner. Its registry guesses `ccsn_canonical.json`, while the actual file is `ccsn_11class_canonical.json`.

**Correction:** explicitly settle the final-model protocol, selection metric, repeated-seed requirement, and meaning of OOF results; then align artifacts and documentation with that procedure.

**5. High - “Permanent holdout” membership changes across canonical manifests.**

`scripts/build_canonical_manifests.py` independently repartitions each filtered dataset and each harmonized source. Identical seed values do not preserve membership when sample sets, class labels, ordering, or group identifiers change.

| Baseline manifest compared with harmonized | Shared images | Changed split | Baseline test now in harmonized train | Harmonized test already in baseline train |
|---|---:|---:|---:|---:|
| CCSN 11-class | 2,337 | 1,201 | 285 | 293 |
| GCD 5-class | 14,306 | 7,543 | 1,852 | 1,851 |
| GCD 6-class | 14,306 | 7,521 | 1,816 | 1,802 |

These are path-based measurements from the actual manifests. This does **not** establish leakage within the harmonized README table: its in-domain and merged models consume the same harmonized manifest. It does mean that a checkpoint pretrained on another manifest cannot automatically be treated as unseen on the harmonized test set, and that cross-protocol comparisons are not paired evaluations on identical images.

**Correction:** either approve and preserve a shared source-level holdout before deriving taxonomies, or explicitly version these as separate protocols and prohibit incompatible checkpoint reuse. Do not regenerate existing holdouts silently.

**6. High - Missing or corrupt images become labeled black training/test samples.**

`src/run_harmonized.py:100` and the analogous preloader in `src/tune_resnet_family.py` catch all image exceptions and substitute zeros while retaining the original class. A missing-image reproduction confirmed the loader reports success and returns a black image with its original label.

No current raw image failed this audit, so this is a confirmed code defect, not evidence that the present results used corrupt images.

**Correction:** fail with path/source context before training or evaluation; require an explicitly recorded policy for any permitted exclusions.

**7. High - Best-checkpoint snapshots are not safe on CPU.**

`src/run_harmonized.py:291`, `src/train_sghcv.py:287`, and the transfer engine snapshot state using `{k: v.cpu() ...}`. For CPU tensors this can share storage with the live model. Subsequent updates then change the purported best state. A tensor reproduction confirmed the aliasing.

**Correction:** make detached clones/deep copies. This limits the advertised CPU fallback; it does not establish that existing GPU runs have this defect.

**8. High - Recorded hyperparameters and run identity are insufficient for reproducibility.**

Tuning writes `seed = 42` into generated TOML but never seeds its training RNGs. Disabling `cudnn.benchmark` is not equivalent to seeding. The harmonized runner seeds once at process start, so `--experiment merged` and `--experiment all` reach merged model initialization after different RNG consumption even with the same seed.

The harmonized summary is automatically loaded and partially updated (`src/run_harmonized.py:444`), which allows results from different invocations to coexist in one apparent comparison. Checkpoint reuse checks existence rather than matching seed, split fingerprint, preprocessing, taxonomy, and hyperparameters. Re-running or reusing a checkpoint evaluates the test set again; the merged branch also runs combined inference after separate source evaluations. Repeated forward passes alone are not adaptive test leakage, but “evaluated strictly once” is not literally enforced.

The registry omits Git revision/dirty state, dependency versions, full resolved configuration, pretrained-weight identity, and per-image predictions. It registers merged runs but not the corresponding in-domain/transfer results. Its four records include `sghcv_ccsn_resnet18` with no manifest; the associated metric file has only 44 test images, not the canonical 508. This record is not evidence of a full canonical CCSN evaluation.

**Correction:** identify each run uniquely, persist its full inputs and predictions, reset/record seeds per experimental condition, verify checkpoint provenance before reuse, and label reduced/fixture experiments explicitly.

**9. High - Configuration and quickstart paths do not form a working unified workflow.**

Three TOML files contain a BOM and fail `tomllib`: `config/datasets/gcd_6class.toml`, `config/experiments/stage1_ccsn_baseline.toml`, and `config/experiments/stage1_gcd_6class_baseline.toml`.

`config/experiments/harmonized_5bin_resnet18.toml` uses `training`/`evaluation` rather than the loader's `training_protocol`/`evaluation_protocol`. Reproduction stops at the missing `training_protocol` key. It also references a nonexistent `config/datasets/harmonized_5bin.toml`.

`gcd_5class.toml` declares no official split, but its raw folder has `train` and `test`, so the generic index rejects it. The six-class configuration also requires filtering the raw seven-class layout; the generic index requires an exact class-folder match. With `CLOUD_DATA_ROOT` at the repository root, the CCSN profile resolves `CCSN_v2` whereas active runners use `CCSN/CCSN_v2`.

The README asks for `pip install -r requirements.txt`, but that file and a dependency lock/package manifest are absent. The harmonized runner silently ignores a missing config, catches parse errors and continues, lets TOML override explicit CLI epochs/batch/seed values, and does not consume several declared augmentation/scheduler fields.

**Correction:** make one explicit config contract resolve every production run, validate all shipped profiles, fail on invalid configs, document CLI precedence, and provide reproducible dependency installation.

**10. High - Strict source-only zero-shot selection is not demonstrated.**

The tuning workflow selects hyperparameters using the concatenation of CCSN and GCD validation sets. The README then uses those same configurations for single-source training and labels cross-dataset evaluation zero-shot. Model fitting is source-only, but selection has used labeled target-domain validation examples.

**Correction:** distinguish target-test-unseen transfer from target-data-unseen/source-only selection. For a strict source-only claim, approve source-only hyperparameter selection. No target test-set selection was found in the tuning code.

**11. High - Manuscript confidence intervals and deployment numbers need correction.**

The production table contains 34 discrepant CI cells out of 63 inspected metric cells. All corresponding point estimates agree with saved JSON. Example: ResNet-18 CCSN balanced accuracy is 55.55 with saved CI `[50.25,60.60]`, but the manuscript gives `[51.5,59.8]`; saved macro-F1 CI is `[49.72,60.00]`, but the manuscript gives `[50.8,59.5]`. The full mismatch list is in `details.json`.

Measured parameter counts of the implemented five-class head are 11,309,637 (ResNet-18), 21,417,797 (ResNet-34), and 24,034,373 (ResNet-50), rather than the reported 11.20M, 21.31M, and 23.55M. Saved checkpoint byte sizes are recorded in the evidence.

No matching inference-latency benchmark procedure or results were found for the manuscript's GPU/embedded-CPU timings, nor evidence for its claimed sub-0.5% INT8 accuracy degradation. Training throughput is not inference latency. The code has no horizon-mask transform despite a manuscript claim about successful horizon masking.

**Correction:** generate tables from saved records, label unmeasured deployment quantities as future work, and measure the actual implemented architectures before making deployment claims.

**12. Medium - Confidence intervals do not cover all the uncertainty claimed.**

`src/atmospheric_evaluation.py:55` bootstraps individual images. Current test partitions retain duplicate repetitions: 3 extra identical files in CCSN 11-class, 32 in GCD 6-class, 1 in GCD 5-class, and 4 in harmonized. This sampling does not respect known dependence, and cannot capture unknown session/camera dependence or training-seed variability. The magnitude of any CI distortion was not estimated here.

The reported 4.06, 2.77, and 1.07 percentage-point CCSN gains are observed differences, not established statistical significance. A CI for each individual model is not a paired CI for their difference. There is no CI for the source-balanced average. In addition, report generation omits explicit `labels` for `classification_report`, so a subset missing an entire class can fail; the bootstrap's macro-class set can vary when classes disappear in resampling.

**Correction:** state that current intervals are image-bootstrap, conditional on the fitted model and available test set; approve group-aware/paired analysis and repeated training seeds as appropriate. Save per-image predictions so this analysis is possible.

**13. Medium - Balancing and physics claims exceed what the implementation guarantees.**

Weighted sampling makes each source probability 0.5 in expectation; it does not make every batch exactly half CCSN. [PyTorch sampling documentation](https://docs.pytorch.org/docs/2.14/data.html#torch.utils.data.WeightedRandomSampler). The merged validation loss is computed over concatenated images without source weighting, so selection remains dominated by GCD even when training uses balanced sampling. This can be intentional, but the objective should be explicit.

The no-vertical-flip policy is implemented in active transforms. Its fisheye explanation is geometrically incorrect as written: reflecting a centered circular image preserves radius and therefore the center-versus-horizon relationship; it does not invert zenith angle. The suitability of flips/crops depends on acquisition geometry. This audit does not recommend changing the approved augmentations automatically.

**Correction:** document expected rather than exact batch balance, state the validation weighting objective, and correct the augmentation rationale without inventing unverified sensor properties.

**14. Medium - Multiple prototype and split-generation paths remain misleading or broken.**

`scripts/run_stage1_benchmarks.py` always generates class-colored synthetic fixtures. Stage 2 and Stage 3 also request fixtures, then mishandle the `(samples, groups)` tuple now returned by `create_fixture_if_needed`. Their older engines use ordinary `StratifiedKFold`, not canonical duplicate groups. Stage 1 evaluates the last fold's model despite a “best fold” comment.

`scripts/generate_splits.py` imports the real generator but never calls it: it writes descriptive JSON with no actual sample/fold assignments. Its “saved manifests” message is therefore misleading. The working canonical construction is a different script.

Legacy `HEAD:src/train_multiple_datasets.py` evaluates the test loader each epoch and saves these results as validation metrics; it also contains vertical flipping and `cudnn.benchmark=True`. Legacy `HEAD:src/merger.py` retains the leaky GCD official split and randomly partitions CCSN without duplicate groups. Those files are already deleted locally, but the handoff's claimed `src/legacy/` archive does not exist.

**Correction:** identify one supported real-data execution path, clearly label fixtures/prototypes, retire incompatible entry points, and distinguish legacy metrics from canonical benchmarks. Do not infer that fixture metrics produced the README table merely because fixture scripts exist.

**15. Medium - Canonical manifests are neither write-once nor bound to full input provenance.**

The builder overwrites canonical files with mode `w`. It reads duplicate groups from stored inventories but scans current `*.jpg` paths without validating inventory freshness. Hardcoded lists/exclusions duplicate configuration. Manifest `sha256` covers sample records, not input image bytes, inventory fingerprint, complete taxonomy metadata, or generator version. Production loaders do not invoke the generic validator or independently recompute this hash.

The current files pass the independent audit, but a future changed image/inventory could silently defeat these assumptions.

**Correction:** version and validate manifest provenance, fail on input drift, and enforce an explicit approved replacement procedure for frozen splits.

**16. Medium - Registry writes can erase history and misstate verification.**

`src/experiment_registry.py:78` catches any registry-read failure and starts from an empty list; a corrupted temporary registry was silently replaced in the reproduction. The fixed experiment ID overwrites prior runs, and the final write is neither atomic nor locked. `get_registry_summary_df` labels a checkpoint verified merely when a saved hash is nonempty, without rehashing it at display time.

All four current checkpoint hashes actually matched this audit, so current mismatch is not alleged.

**Correction:** fail visibly on malformed records, retain distinct run IDs, use atomic/serialized writes, and distinguish a recorded digest from a verified current file.

**17. Medium - Data exclusion, portability, and phase completion claims need synchronization.**

The raw `CCSN/` and `GCD/` folders are visible as untracked files because `.gitignore` excludes `data/` but not these actual paths. They are not currently committed; a broad add risks including them. New training runners hardcode repository-relative dataset locations, device/workers, and ImageNet normalization instead of consuming the environment model. ImageNet constructors can download weights when a cache is absent; no required local-weight path/offline preflight or SLURM workflow is implemented.

The dashboard/handoff call Phase 4 complete, but many Phase 4 files are untracked, old archive paths are missing, the changelog ends at Phase 3.5, several directory READMEs are stale, and `project_specification.md` is still a TODO. Decision D-011 records smoothing 0.1 while current tuned configs use 0.0; tuning automatically writes `approval_status="approved"`. An automated winner is not evidence of researcher approval. The earlier decision still says cross-class resolution is pending, while later handoff claims exclusion was approved; reconcile this provenance in a new decision record.

**Correction:** protect actual dataset paths from accidental staging, make data/output/weight roots configurable, document offline execution, and synchronize current status with verified implementation and approvals. Preserve frozen historical handoffs.

**18. Medium - The compiled report is visibly clipped and its source does not reproduce cleanly.**

Pages 8 and 9 clip rightmost results columns; page 11 clips the computational-profile table heading. The existing LaTeX log records large overfull boxes. Current source references two nonexistent long-form confusion-matrix filenames before the existing shortened filenames at lines 356-363. The delivered PDF can therefore exist while the current `.tex` source does not rebuild identically.

**Correction:** fix figure references, fit tables to the page, regenerate, and visually verify the full output. These are observed defects in the existing PDF; this audit did not edit or re-export the research report.

**Recommended correction sequence**

1. Preserve current evidence and mark manuscript conclusions/results as provisional. Correct unsupported method/acquisition claims and discrepant CIs before sharing the report as finished research.
2. Repair the reproducible execution blockers: SG-HCV contracts, TOML parsing/schema, missing dependency specification, fail-fast image loading, CPU snapshots, and explicit CLI precedence. Add targeted tests for these real failure modes.
3. Obtain a recorded methodological decision on the common tuning metric, source-only transfer selection, final CV model procedure, shared versus separate holdouts, and repeated-seed/uncertainty protocol. These choices affect scientific behavior.
4. Freeze run/manifest provenance and save per-image predictions. Re-run only experiments affected by approved scientific or implementation corrections, under a declared test-use policy; do not silently repurpose old test results.
5. Rebuild manuscript tables and figures from verified artifacts; synchronize the dashboard, decisions, issues, walkthrough, and new handoff. Then proceed to mobile/HPC benchmarking with actual measurements.

**Audit evidence and reproduction**

- `artifacts/audit_2026-09-05/evidence.json`: runtime, full raw-data checks, canonical partition checks, cross-manifest counts, registry hash verification, syntax checks, and tuning summaries.
- `artifacts/audit_2026-09-05/details.json`: diagnostic reproductions, all checkpoint hashes/shapes/sizes, parameter counts, metric consistency checks, missing figure paths, and all 34 production-table CI discrepancies.
- `artifacts/audit_2026-09-05/check_audit.py`: read-only raw-data/metadata audit, writing its own evidence only.
- `artifacts/audit_2026-09-05/test_details.py`: bounded diagnostic reproductions. Its successful assertion of an observed defect does not mean that defect has been fixed.
- `artifacts/audit_2026-09-05/report-page-01.png` through `report-page-12.png`: rendered pages used for visual review.

```powershell
.venv/Scripts/python.exe -m pytest tests/ -q
.venv/Scripts/python.exe artifacts/audit_2026-09-05/check_audit.py
.venv/Scripts/python.exe -m pytest artifacts/audit_2026-09-05/test_details.py -q -s
```

The diagnostics remain under the existing ignored artifacts directory. This report is a new, uncommitted audit document; no commit or push was performed.
