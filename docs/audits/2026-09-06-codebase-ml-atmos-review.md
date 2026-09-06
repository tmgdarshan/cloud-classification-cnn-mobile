# Codebase, ML, and Atmospheric-Science Review - 6 September 2026

Scope: read-only review of the current codebase, saved artifacts, canonical
manifests, report TeX, and rendered PDF. No production code, manuscript text,
result JSON, checkpoint, or configuration file was changed. Current PDF source:
`report/cloud_classification_resnet.pdf`.

## Verification Snapshot

- Full test suite: `.venv/Scripts/python.exe -m pytest -q` -> 127 passed.
- Python syntax compilation: `src`, `scripts`, and `tests` compile cleanly.
- Current PDF render: Ghostscript processed pages 1 through 12.
- Current LaTeX log: output is 12 pages; no unresolved references. It reports
  one overfull box in the abstract line and one underfull box in Table 2.
- Current final-sweep audit: 20,582 referenced image files hashed; no missing
  images; no group leakage or actual byte-hash leakage across train/val/test in
  the canonical manifests.
- Appendix production table values match saved benchmark JSONs for all checked
  component, pooled, and source-average table values.

## High-Priority Findings

### 1. Existing checkpoints still lack metadata sidecars

`src/run_harmonized.py` now has helpers to save and validate checkpoint metadata
(`get_checkpoint_meta_path`, `load_checkpoint_metadata`,
`save_checkpoint_metadata`, and `validate_checkpoint_provenance`). However,
`artifacts/harmonized_results/` currently contains checkpoint `.pth` files and
summary JSON files, but no `*.meta.json` sidecars.

Effect: `--reuse-checkpoints` cannot actually verify the already-existing
weights against the requested seed, manifest, optimizer, epoch count, learning
rates, or label smoothing. The validator returns immediately when metadata is
missing. For manuscript-grade provenance, regenerate sidecar metadata from the
original run logs if available, or make reuse fail/require an explicit override
when metadata is absent.

### 2. Checkpoint reuse still warns instead of protecting the run

At `src/run_harmonized.py:380-388`, provenance mismatches are printed as
warnings. At `src/run_harmonized.py:695-705`, the joint checkpoint is then loaded
anyway. This is convenient for local work, but weak for a final benchmark.

Effect: a future run can reuse weights with different requested settings and
still produce fresh summaries. The registry now tries to prefer checkpoint
metadata when present (`src/run_harmonized.py:807-814`), which is good, but
missing or partial metadata remains risky. For official runs, mismatched or
missing training provenance should fail unless a deliberate `--allow-provenance-
mismatch` style flag is passed.

### 3. Saved tuning artifacts predate the current unsmoothed-loss selection code

The current tuner records `val_unsmoothed_loss`, ranks by
`best_val_unsmoothed_loss`, and stores `selection_metric` metadata
(`src/tune_resnet_family.py:374-400`, `615-640`). The existing saved tuning JSON
files under `artifacts/tuning/` still contain only `train_loss`, `train_acc`,
`val_loss`, `val_acc`, and `val_macro_f1`, with no unsmoothed validation-loss
history or selection metadata.

Effect: the manuscript is now careful enough to say the table reflects five-
epoch screening runs, but the old JSON files cannot prove that the current
unsmoothed-loss selector produced the selected trials. This does not overturn
Trial 08, because Trial 08 uses zero label smoothing, but it is a provenance gap.
Best fix: rerun or reconstruct tuning with the current code, or explicitly mark
the saved tuning artifacts as historical screening artifacts.

### 4. "Zero-shot" remains in public-facing result artifacts and figure caption

The current runtime keys include `cross_source_*` and keep `zeroshot_*` as
deprecated aliases (`src/run_harmonized.py:595-598`, `669-672`). That code
choice is reasonable for backwards compatibility. The manuscript still uses
"Direct Zero-Shot Cross-Modal Evaluation" in the transfer-asymmetry caption
(`report/cloud_classification_resnet.tex:262`), and the ResNet-18 summary JSON
still contains visible `"Zero-Shot CCSN to GCD"` / `"Zero-Shot GCD to CCSN"`
dataset names.

Effect: an ML reviewer may object because these are supervised single-source
fine-tuned classifiers evaluated on the other source, not zero-shot learning in
the usual modern sense. Use "direct cross-source transfer" or "off-source
transfer" in outward-facing text.

## Medium-Priority Findings

### 5. The manifest-building script partitions each source separately for the harmonized benchmark

In `scripts/build_canonical_manifests.py`, the harmonized CCSN samples and GCD
samples are each passed separately through `partition_samples_stratified_group`
before concatenation. This preserves source-specific 64/16/20 allocations and
is consistent with the source-balanced evaluation design.

Effect: the split is not one global StratifiedGroupKFold over the fully pooled
harmonized dataset. That is not necessarily wrong, and may be preferable for
source balance, but the manuscript should avoid implying that a single pooled
SGKF generated the harmonized split unless that is intentionally true.

### 6. Validation selection is GCD-weighted even when training is source-balanced

The manuscript already discloses this correctly. Joint training uses inverse
source-size weighted sampling, but validation loss is sample-averaged over the
combined validation set. With 374 CCSN validation images and 2,289 GCD validation
images, checkpoint selection is dominated by GCD validation loss.

Effect: the source-balanced headline is evaluated fairly across sources, but
checkpoint selection is not source-balanced. This is acceptable if disclosed;
for a stricter future benchmark, compute validation selection as the average of
CCSN and GCD validation losses.

### 7. Joint training has a larger CCSN exposure budget than CCSN-only training

The manuscript also discloses this correctly. With replacement sampling gives
roughly 5,325 CCSN draws per joint epoch, compared with one pass over 1,495 CCSN
images per CCSN-only epoch. The +4.06 pp CCSN gain is therefore a gain from the
whole joint recipe, not isolated evidence that GCD features alone improve CCSN.

Effect: keep using "joint recipe" or "joint training with source-balanced
sampling" language. Avoid causal shorthand such as "transfer improves CCSN."

### 8. The atmospheric taxonomy is defensible as a compatibility taxonomy, but two bins need care

The manuscript correctly avoids claiming a genus-preserving WMO taxonomy.
Scientifically, the delicate mappings are `Ns -> Cumulonimbus` and
`As -> Stratocumulus`. These pair precipitating/deep-layer or mid-level sheet
clouds with operational classes that are not one-to-one physical equivalents.

Effect: keep calling the classes compatibility bins. Avoid saying they are
meteorologically exact categories. The current table mostly handles this well.

### 9. The current PDF is 12 pages and structurally coherent, with two wording nits

The PDF now contains the intended main figures: convergence, transfer asymmetry,
and the CCSN confusion comparison. Table 3 is readable. The appendix tables are
dense but usable.

Remaining wording issues:
- `report/cloud_classification_resnet.tex:42`: "avoid prior distribution
  collapse" is a little causal for an exclusion rationale.
- `report/cloud_classification_resnet.tex:314`: "dark non-precipitating rain
  cloud bases" is internally inconsistent; rain-cloud wording suggests
  precipitation.

## Lower-Priority Engineering Notes

### 10. Config fields are partly documentary in the production runner

`src/run_harmonized.py` reads selected flat TOML keys: architecture, epochs,
batch size, seed, learning rates, weight decay, label smoothing, dropout, and
optimizer. The tuned TOML files also include scheduler and augmentation fields,
but the runner hardcodes the cosine scheduler and transforms in code.

Effect: current values appear aligned, but a future config change to augmentation
or scheduler fields would not necessarily affect execution. Either consume those
fields or validate that they match the hardcoded official protocol.

### 11. Evaluation bootstrap is image-level and conditional on fixed weights

`src/evaluation.py` bootstraps images with a fixed seed and fixed predictions.
That supports the manuscript's "sampling variability conditional on the test
partition" language. It does not capture training-seed variance, model-selection
variance, or source-average uncertainty.

Effect: the current caption is appropriate. Do not upgrade it to stronger
statistical claims without paired or repeated-seed analysis.

### 12. Audit artifacts need cleanup to prevent stale-count confusion

The refreshed audit reports a 12-page Ghostscript render, but the folder still
contains an old `page-13.png` from the previous PDF. The audit evidence therefore
has a stale rendered-page count in one field.

Effect: harmless to the manuscript, but future audit scripts should clear their
own render prefix before writing fresh page PNGs.

## Overall Assessment

From an atmospheric-research supervision view, the work is now coherent if it is
presented as an empirical cross-source compatibility benchmark rather than a
general WMO cloud-physics classifier. The data curation story, exact-duplicate
control, compatibility taxonomy, and source-balanced headline are defensible.
The main scientific discipline needed is restrained causal language around
domain shift, transfer asymmetry, and the CCSN gain.

From an ML engineering view, the core pipeline is in much better shape than the
legacy project: tests pass, canonical manifests are clean, image loading is
fail-fast, split leakage checks pass, and the headline numbers reproduce from
saved results. The remaining risks are provenance and reproducibility rather
than obvious coding breakage: existing checkpoints lack metadata sidecars, old
tuning JSONs predate current selection metadata, and official checkpoint reuse
should probably fail closed.
