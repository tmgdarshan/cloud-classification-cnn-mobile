# Final manuscript and repository sweep - 6 September 2026

Review of the current working tree and delivered PDF. No manuscript, production
code, configuration, checkpoint, or benchmark-result edits were made. Audit
evidence and page renders are in `artifacts/final_sweep_2026-09-06/`.

## Verified

- `python -m pytest tests -q`: **127 passed** in 5.53 seconds.
- All 32 Python files under `src`, `scripts`, and `tests` parse successfully.
- All **189 numbers** in the seven component/pooled rows per architecture of
  Appendix Table 5 (point estimates and CI endpoints) match saved JSON exactly.
- Source-average accuracies reproduce **74.58%, 74.52%, 74.66%**. The
  ResNet-18/50 difference remains **0.08 percentage points**.
- The 30 tuning accuracy cells agree at displayed precision, allowing the
  half-up display of ResNet-18 Trial 02: stored 83.25%, displayed 83.3%.
- Hashed **20,582 distinct referenced image files**. None missing; no actual
  SHA-256 hash crosses train/validation/test in any of the four canonical
  manifests, including across sources in the harmonized manifest. Label indices
  agree with class names. Split counts agree with Table 1.
- All **13** pages of the existing PDF were rendered and visually inspected.
  No clipping or overlap was observed. Table 3 is readable. The current log has
  no overfull boxes or unresolved references; it has one underfull box in Table 2
  and an epstopdf shell-escape warning that did not prevent output.
- The report and artifact copies of the convergence figure have identical hashes.

## Findings requiring attention

### 1. PDF/structure differs from the stated final version

`report/cloud_classification_resnet.log:700` records **13 pages**, confirmed by
rendering the actual PDF. Page 12 contains only reference [6], leaving almost the
entire page blank; the appendix starts on page 13. Pages 6 and 8 also have large
blank areas before forced figures. The `[H]` placement and appendix `\clearpage`
at TeX line 369 contribute to this layout.

The TeX includes the convergence plot, class-distribution plot, and two confusion
panels. It **does not include the transfer-asymmetry plot**, even though
`report/figures/transfer_asymmetry_recall_resnet18.png` exists. The class-distribution
plot remains in the main paper at TeX line 265. This does not match the requested
final figure inventory. Restore the intended figure inclusion and review page
breaks before treating this PDF as the final 12-page version.

### 2. Confusion-matrix explanation names the wrong destination class

At `report/cloud_classification_resnet.tex:318`, the text says Altocumulus recall
improves with reduced confusion into Stratocumulus. Figure 3 on PDF page 10
shows Altocumulus-to-Stratocumulus **increasing from 5/97 (5.2%) to 8/97 (8.2%)**.
The large reduction is Altocumulus-to-**Cirrus**, from **25/97 (25.8%) to
13/97 (13.4%)**. Replace the destination class in the explanation.

### 3. Transfer explanation still overstates causality and the recall comparison

TeX lines 278 and 281 say prior disparity and optics "directly explain" the
asymmetry and that models "heavily rely" on hemispherical geometry. No feature
ablation or attribution experiment is presented to establish these mechanisms;
the following paragraph itself calls them hypotheses. Use that qualification
consistently.

The 75.9% versus 19.2% Cumulonimbus comparison is **between opposite transfer
directions**, involving different trained models and different test sources.
It is not the same model's in-domain-to-out-of-domain recall drop. Name both
directions explicitly instead of describing a within-model collapse.

### 4. Tuning reproducibility and historical evidence do not match current code

`src/tune_resnet_family.py:502` onward does not seed Torch or NumPy, yet
`save_optimal_toml` writes `seed = 42` at line 480. That is a production default,
not evidence that the tuning run used seed 42. Add explicit tuning seeds and
record them in the trial output; keep production defaults separately identified.

The saved tuning JSONs contain five epochs per trial and no
`best_val_unsmoothed_loss`, `val_unsmoothed_loss`, or selection-rule metadata.
The current tuner ranks by unsmoothed loss at line 600 and records those fields.
Consequently, these historical artifacts do not demonstrate that the current
selection rule produced the selected trials. Describe the historical baseline
selection honestly, preserve those artifacts, and avoid implying they were
regenerated with the current tuner. The selected Trial 08 has zero label
smoothing, so its displayed validation-loss curve can still represent
unsmoothed CE. State explicitly that Figure 1 shows **five-epoch Trial 08
screening runs**, not the 15-epoch production trajectories.

### 5. Reused checkpoints can acquire incorrect run metadata

`src/run_harmonized.py:610` loads existing joint weights based on filename alone.
There is no check of the training configuration, seed, or original manifest
against the requested run. Lines 694 onward then register the current CLI/config
values as hyperparameters for those old weights. For example, reuse with a new
`--seed` or `--epochs` can mislabel the checkpoint's training provenance.
Store checkpoint metadata and validate reuse against it; record evaluation
settings separately from original training settings.

The runner seeds only once at line 413, so a fresh `--experiment joint` run and
the joint phase of a fresh `--experiment all` run consume different random
streams despite the same seed argument. If these commands are intended to
reproduce the same joint run, seed each training condition explicitly.

### 6. Root-level pytest collects an obsolete diagnostic and fails

`python -m pytest -q` produced **1 failed, 127 passed**. The failure is
`artifacts/audit_2026-09-05/test_details.py:37`, which expects a missing image to
load silently. The current loader correctly raises. The documented `tests/`
command is green. Limit pytest discovery to `tests/` or rename/archive the old
diagnostic. It also writes to historical audit evidence at line 35 before failing,
so it should not be collected as a routine test.

## Smaller discrepancies and presentation improvements

- **Precision:** source averages are calculated from already rounded metrics
  (`src/run_harmonized.py:664`). Recovering integer true-positive counts from
  saved per-class recalls/supports gives ResNet-34 source-balanced recall
  **73.86140864%, rounding to 73.86%**, versus the stored/published 73.87%.
  Retain full precision until final formatting. The headline accuracy comparison
  is unaffected.
- **Memory units:** TeX line 335's 43.1 and 91.7 are approximately **MiB of FP32
  parameter storage**, not decimal MB or measured inference memory. Decimal
  parameter sizes are approximately 45.2 and 96.1 MB; activations/runtime overhead
  are additional.
- **Tuning comparison:** the claim "1.5 to 2.9" pp at TeX line 219 does not
  describe a consistent comparison. Trial 08 minus standard SGD Trial 04 in
  the displayed table gives approximately **1.5, 2.0, 1.2 pp**. Specify the
  compared trials and derive the range from those rows.
- **Appendix legibility:** both appendix tables are heavily reduced to fit page
  13. Their type and the convergence legends are small at normal reading scale.
  Use reclaimed whitespace or separate appendix pages to improve readability.
- **Headline uncertainty:** Table 5 provides component/pooled CIs, but no CI for
  the source-balanced headline or paired architecture difference. Clarify this
  scope in Table 3's caption; the existing refusal to claim equivalence is sound.
- **Configuration contract:** the production runner reads selected scalar TOML
  fields but hardcodes augmentations and the cosine schedule/eta_min. Changing
  those TOML entries does not change execution. Consume or explicitly validate
  these fields before future experiments.

## Scope limits

This sweep checks saved-result consistency, current code, tests, image hashes,
and PDF presentation. It does not retrain models, replay checkpoint inference,
or independently validate the cited publications. Saved aggregate agreement
alone cannot establish that a particular checkpoint reproduces every reported
prediction; per-image predictions and complete run lineage remain useful
provenance work. Existing uncommitted work was preserved.
