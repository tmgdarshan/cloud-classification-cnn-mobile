# Complete Repository Audit - 7 September 2026

Verdict: **CONDITIONAL PASS**.

The repository is now strong enough to support the current manuscript as an
empirical cross-source cloud-classification benchmark, provided the remaining
provenance and terminology issues are handled before treating it as a locked
research archive. The core data boundary is healthy: canonical manifests are
present, exact-byte duplicate groups stay atomic, 20,582 referenced images hash
cleanly, and the saved manuscript numbers match benchmark JSONs. The codebase
also passes the current automated suite.

The main risks are no longer ordinary coding bugs. They are research-software
risks: old checkpoints lack provenance sidecars, saved tuning artifacts predate
the current unsmoothed-loss selection metadata, public-facing "zero-shot"
terminology remains in a few places, and repeated/partial result artifacts can
confuse future readers unless the official path is kept very explicit.

## Verification Matrix

| Check | Result | Evidence |
|---|---:|---|
| Full test suite | Pass, 127/127 | `.venv/Scripts/python.exe -m pytest -q` |
| Python syntax compile | Pass | `.venv/Scripts/python.exe -m compileall -q src scripts tests` |
| Canonical image references | Pass | 20,582 image files hashed, none missing |
| Split leakage | Pass | 0 group leakage and 0 byte-hash leakage across canonical train/val/test |
| Table/result parity | Pass | 189 master-table values checked, 0 mismatches |
| Current PDF render | Pass | Ghostscript processed pages 1 through 12 |
| LaTeX references | Pass | No unresolved references |
| LaTeX boxes | Minor warning | one 8.08 pt overfull abstract line, one underfull Table 2 line |
| Notebooks | No active notebooks | old `notebooks/data_expolration.ipynb` is deleted in the working tree |

Note: `artifacts/final_sweep_2026-09-06/audit.py` reports `rendered_pages: 13`
because an old `page-13.png` remains in the audit output directory. The actual
current PDF render and LaTeX log both report 12 pages.

## Executive Summary

Overall code quality: good research-code quality. The active code is small,
mostly modular, and substantially cleaner than the deleted legacy scripts.
There is still duplication between the generic dataset/split modules and the
direct canonical-manifest runner, but the official execution path is now clear.

ML quality: solid baseline engineering. ImageNet-pretrained ResNet-18/34/50,
physically conservative augmentation, differential learning rates, AdamW,
validation-loss checkpoint selection, source-balanced joint sampling, and
balanced/macro metrics are appropriate for this thesis-scale benchmark.

Scientific reliability: defensible as a cross-source compatibility benchmark,
not as a strict WMO-genus classifier. The exact-byte leakage control, clear
split terminology, and operational taxonomy framing are the strongest parts.
The remaining scientific discipline is to avoid causal overclaiming around
domain shift and to avoid "zero-shot" wording.

Production readiness: not production-ready for deployment, and the docs already
say mobile/edge work is future scope. It is close to research-reproducible, but
needs checkpoint sidecars for existing weights, environment/commit capture, and
cleaner artifact provenance before archival freeze.

## Critical Issues

### 1. High - Existing trained checkpoints have no provenance sidecars

Location: `artifacts/harmonized_results/*.pth`; sidecar expectation in
`src/run_harmonized.py:384-429`.

Explanation: the runner now supports metadata sidecars and fails closed when
`--reuse-checkpoints` is used without metadata unless
`--allow-unverified-checkpoints` is passed. That is the right behavior. However,
the current checkpoint directory contains no `*.meta.json` files, so the saved
weights cannot be verified against their original seed, manifest, optimizer,
epoch count, learning rates, label smoothing, scheduler, or augmentation config.

Recommended fix: create sidecars for the existing official checkpoints from
trusted run logs, or rerun the official benchmarks once to regenerate
checkpoints with metadata. Include checkpoint SHA-256, manifest SHA-256,
training config, validation metrics, git commit, package versions, and device.

### 2. High - Saved tuning artifacts are older than current selection metadata

Location: current selector in `src/tune_resnet_family.py:374-400` and
`src/tune_resnet_family.py:615-640`; old artifacts in
`artifacts/tuning/tuning_results_resnet*.json`.

Explanation: current code ranks by `best_val_unsmoothed_loss` and records
`selection_metric`, but the saved JSONs only contain `val_loss`, `val_acc`, and
`val_macro_f1`. The manuscript now says the tuning table reflects five-epoch
screening runs scored on unsmoothed CE, but the historical files themselves do
not contain the unsmoothed-loss evidence.

Recommended fix: rerun tuning with the current code or label the old JSONs as
historical screening artifacts. If rerunning is too expensive, add a provenance
note saying Trial 08 has zero label smoothing, so its stored `val_loss` equals
unsmoothed CE for the selected baseline, but other smoothed trials cannot be
retrospectively re-ranked by unsmoothed CE from the current JSON alone.

### 3. High - "Zero-shot" terminology remains in outward-facing text/artifacts

Location: `report/cloud_classification_resnet.tex:262`,
`artifacts/harmonized_results/harmonized_summary_resnet18.json`,
deprecated aliases in `src/run_harmonized.py:669-672`.

Explanation: the experiments are supervised single-source fine-tuning followed
by off-source evaluation. This is direct cross-source transfer, not zero-shot
learning in the usual ML sense. The code aliases are acceptable for backward
compatibility, but manuscript captions and artifact dataset names should not use
the term publicly.

Recommended fix: rename manuscript caption wording to "direct cross-source
transfer". Keep deprecated JSON keys only as internal compatibility aliases and
avoid regenerating figures or captions with "zero-shot".

### 4. Medium - Harmonized manifest construction is source-separate, not one pooled split

Location: `scripts/build_canonical_manifests.py`, where CCSN and GCD
harmonized samples are partitioned separately before concatenation.

Explanation: this is reasonable for a source-balanced benchmark because each
source gets its own 64/16/20 allocation. It is not the same as running one
global `StratifiedGroupKFold` over the fully pooled harmonized dataset.

Recommended fix: keep the design, but describe it precisely as source-stratified
canonical partitioning. Avoid implying that a single pooled SGKF produced the
harmonized split unless that becomes true.

### 5. Medium - Joint checkpoint selection remains sample-weighted toward GCD

Location: validation loop in `src/run_harmonized.py:301-321`; disclosure in
`report/cloud_classification_resnet.tex:200-204`.

Explanation: joint training uses source-balanced sampling, but validation loss
is a sample-average over 374 CCSN and 2,289 GCD validation images. This means
checkpoint selection is dominated by GCD validation loss even though the
headline metric is source-balanced.

Recommended fix: current disclosure is acceptable. For a stricter future
benchmark, select checkpoints by the unweighted mean of CCSN validation loss and
GCD validation loss.

### 6. Medium - Result summaries retain legacy naming aliases

Location: `src/run_harmonized.py:672`, `src/run_harmonized.py:907`, older
artifact names in `artifacts/figures/`.

Explanation: the runner writes canonical keys plus deprecated aliases
(`zeroshot_*`, `merged_joint_model`). This avoids breaking old scripts, but it
can confuse later readers about what result is authoritative.

Recommended fix: keep aliases internal only, add a `schema_version` and
`canonical_keys` section to summary JSON, and have plotting scripts prefer only
canonical `cross_source_*` and `joint_model` keys.

### 7. Medium - Unknown scheduler values silently fall back to cosine

Location: `src/run_harmonized.py:253-261`.

Explanation: if a config typo sets `scheduler = "cosine"` instead of
`"cosine_annealing"`, the runner only prints a notice and proceeds with cosine.
For official research runs, invalid config should fail loudly.

Recommended fix: raise `ValueError` for unknown scheduler names. Keep permissive
fallback only in an exploratory mode if needed.

### 8. Medium - The test suite verifies contracts, not full experiment replay

Location: `tests/unit/test_canonical_manifests.py`,
`tests/unit/test_canonical_pipeline_contracts.py`.

Explanation: tests cover manifest schema, split disjointness, report-call
contracts, fail-fast image loading, and registry helpers. They do not replay
checkpoint inference, compare per-image predictions, test calibration, or run
small end-to-end training with checkpoint reuse metadata.

Recommended fix: add a tiny synthetic end-to-end smoke test for
`run_harmonized.py` using a mocked/lightweight model path, plus per-image
prediction CSVs for official benchmarks.

## ML & Research Review

Dataset construction is the strongest part of the project. CCSN and GCD are
kept as separate sources, raw official GCD splits are superseded by canonical
manifests, and exact-byte duplicate clusters are kept atomic. The audit verified
zero split leakage by `group_id` and by actual SHA-256 file hash. The project is
honest that exact-byte hashing does not catch near-duplicates.

The five-class taxonomy is scientifically usable because it is framed as an
operational compatibility taxonomy. It is not a clean WMO taxonomy. The
manuscript now explicitly calls out the two fragile mappings: `Ns -> C5` and
`As -> C4`. That is exactly the kind of caveat an atmospheric reader needs.

Class imbalance is handled at two levels: source imbalance is handled through
inverse-volume `WeightedRandomSampler`, while metric imbalance is handled
through balanced accuracy and macro-F1. Class-level imbalance inside each source
is not directly reweighted, which is acceptable for the current design but
should be named as a limitation.

The CNN design is appropriate: ImageNet-pretrained ResNet backbones with a small
regularized head, dropout inside the head, BatchNorm1d, GELU, and hard CE after
tuning selected label smoothing 0.0. The architecture is a credible baseline
family rather than a custom atmospheric model.

Training methodology is mostly sound: AdamW, differential learning rates, cosine
annealing, AMP, CPU-cloned best-state snapshots, and fixed validation-loss
selection. The biggest research limitation is that one seed per architecture
does not measure training-seed variance. Bootstrap CIs over fixed predictions
do not replace repeated training runs.

Evaluation methodology is well chosen for the headline: source-balanced average
prevents the 2,862-image GCD component from dominating the 468-image CCSN
component. The manuscript correctly avoids formal equivalence claims for the
0.08 pp ResNet-18/ResNet-50 margin. Calibration, robustness under perturbation,
and out-of-distribution detection are not covered.

Domain-shift discussion is plausible. Keep it observational. Cross-source
performance differences combine camera optics, field of view, geography,
weather distribution, class priors, resolution, and annotation conventions. The
work does not isolate causal visual mechanisms.

## Software Engineering Review

Repository architecture is clear: `src/` for active code, `scripts/` for
utilities, `config/` for profiles, `metadata/splits/` for canonical manifests,
`artifacts/` for outputs, `docs/` for protocol/audit records, and `report/` for
the paper. Legacy scripts are deleted, which removes a large source of
methodological drift.

Separation of concerns is much improved. `dataset_discovery.py` observes data,
`dataset_index.py` builds deterministic class-folder indices, `split_generator`
and `split_manifest` handle generic splitting, `run_harmonized.py` executes the
official benchmark, and `evaluation.py` reports metrics.

The main design debt is that the official runner bypasses much of the generic
config-loader/dataset-loader stack and reads the canonical JSON directly. This
is simple and less fragile for the thesis result, but it means there are two
parallel pathways: generic infrastructure and final benchmark runner.

Error handling is generally good. Image decode failures now raise, config parse
errors raise, missing manifests fail, and checkpoint reuse fails closed unless
overridden. Remaining weaker spots are silent scheduler fallback and metadata
parse failure returning `None`.

Documentation is unusually strong for a thesis codebase: the official protocol,
decision records, known issues, dashboard, and audits tell a coherent story.
`docs/project_specification.md` is still a TODO, and older audit docs retain
stale warnings that have since been fixed. Those are not execution problems, but
they can confuse a reviewer.

Version-control hygiene is the least polished area. The working tree contains a
large number of modified, deleted, and untracked files. That is normal during a
cleanup sprint, but before final handoff it should be committed in coherent
chunks or archived deliberately.

## Performance Optimization Opportunities

The main memory bottleneck is full preloading of all train/val/test images into
RAM tensors in `src/run_harmonized.py:539-548` and
`src/tune_resnet_family.py:560-569`. At 224x224 uint8 this is manageable for the
current dataset, but it will not scale comfortably to larger datasets or higher
resolution experiments.

DataLoader performance is conservative: `num_workers=0` and `pin_memory=False`
in the training loaders. This is stable on Windows but leaves GPU utilization on
the table. For Linux/HPC runs, test `num_workers=4-8`, `pin_memory=True`, and
`persistent_workers=True`.

Pretrained weight loading relies on torchvision weight enums. If weights are not
already cached, this can trigger network behavior outside the repository's
offline-first standard. Record the weight enum and support a local cache/path for
official reruns.

The evaluation script writes JSON and confusion figures per condition, but not
per-image predictions. Per-image prediction CSV/Parquet would make future
audits, paired tests, calibration, and error slicing much easier.

## Refactoring Roadmap

Immediate fixes:

1. Rename public "zero-shot" text to "direct cross-source transfer".
2. Generate or reconstruct `*.meta.json` sidecars for all official checkpoints.
3. Make unknown scheduler/config values fail loudly in official mode.
4. Fix `docs/project_specification.md` or remove it from the source-of-truth map
   until it contains real requirements.
5. Clean stale audit render files before each audit run.

Short-term improvements:

1. Add per-image prediction exports for every official condition.
2. Add a synthetic end-to-end runner test covering checkpoint metadata and
   summary schema.
3. Add `schema_version` to harmonized summary JSON files and retire public use
   of deprecated aliases.
4. Record git commit, dependency versions, device, and torchvision weight enum
   in experiment registry and checkpoint sidecars.
5. Add optional source-balanced validation loss for joint checkpoint selection.

Long-term improvements:

1. Run repeated-seed experiments for ResNet-18 and ResNet-50 to quantify
   training variance around the 0.08 pp margin.
2. Add calibration metrics such as ECE, Brier score, and reliability diagrams.
3. Add robustness tests for resolution, crop sensitivity, color/illumination
   shift, and near-duplicate scene leakage.
4. Add deployment experiments: INT8 quantization, latency, memory profiling, and
   edge-device validation.
5. Consider a domain-adaptation or multi-branch sensor-aware baseline once the
   ResNet benchmark is frozen.

## Repository Scorecard

| Area | Score | Rationale |
|---|---:|---|
| Architecture | 8 | Clear active paths, legacy removed; some parallel infrastructure remains |
| Data Pipeline | 9 | Canonical manifests, leakage checks, fail-fast loading |
| CNN Implementation | 8 | Sound ResNet baselines and head; no custom atmospheric model |
| Training Pipeline | 7 | Good optimizer/scheduler/checkpointing; one-seed and validation weighting limits |
| Evaluation | 8 | Balanced metrics and bootstrap CIs; no calibration or paired architecture CI |
| Reproducibility | 7 | Seeds/configs/registry exist; old checkpoints and tuning artifacts lack full lineage |
| Documentation | 8 | Strong protocol/audits; one TODO doc and stale audit records remain |
| Maintainability | 7 | Small modules and tests; some duplication and artifact naming drift |
| Scientific Rigor | 8 | Good caveats and leakage control; domain-shift causality remains observational |
| Production Readiness | 5 | Research-ready, not deployment-ready; no edge validation yet |

## Top-20 Improvements

1. Create `*.meta.json` sidecars for every official checkpoint.
2. Export per-image predictions for every reported condition.
3. Replace public "zero-shot" wording with "direct cross-source transfer".
4. Rerun or clearly mark tuning artifacts to match current unsmoothed-loss
   selection metadata.
5. Record git commit, package versions, device, and torchvision weight enum.
6. Make unknown scheduler/config values raise for official runs.
7. Add a source-balanced validation-loss option for joint checkpoint selection.
8. Add repeated-seed runs for the ResNet-18/50 headline comparison.
9. Add calibration metrics and reliability diagrams.
10. Add robustness checks for illumination, crop, resolution, and camera field
    of view.
11. Add near-duplicate/perceptual duplicate analysis beyond exact SHA-256.
12. Add a synthetic end-to-end runner smoke test.
13. Add summary JSON schema versions and canonical-key declarations.
14. Retire or isolate deprecated `zeroshot_*` and `merged_joint_model` aliases.
15. Clean stale audit render outputs before every audit render.
16. Complete or remove `docs/project_specification.md`.
17. Commit the current cleanup in coherent version-control chunks.
18. Add offline pretrained-weight cache configuration.
19. Add optional higher-throughput DataLoader settings for Linux/HPC.
20. Add deployment profiling once the research benchmark is frozen.
