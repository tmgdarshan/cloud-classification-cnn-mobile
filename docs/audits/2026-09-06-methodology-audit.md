# Methodology audit — 6 September 2026

An assessment of whether the scientific procedures in this project are sound and
well executed. Read-only: no retraining, no manifest or result changes. Scope:
data curation, the split protocol, the taxonomy, training, model selection,
tuning, evaluation, and the transfer/joint analyses.

**Overall verdict:** the methodology is sound and, for an MSc-scale project,
unusually well disclosed. Every material weakness below is already acknowledged
somewhere in `docs/` or the manuscript. Nothing here invalidates the headline
results; the items in §9 would tighten the claims.

---

## 1. Duplicate audit & leakage control — **Sound**

- Exact-byte SHA-256 grouping is the right first-line control, correctly
  implemented: verified that **no `group_id` straddles train/val/test or CV
  folds** in any of the four canonical manifests.
- Cross-class conflicting pairs in CCSN (6 images) are excluded rather than
  silently relabelled — the conservative choice, and the resolution is recorded.
- **Caveat (disclosed):** exact-byte grouping does not catch near-duplicate
  bursts (same scene, sensor noise / lighting drift). GCD in particular is an
  automated-schedule camera; residual perceptual duplication almost certainly
  remains, which can still inflate the 89% GCD in-domain number. Stated in
  `KNOWN_ISSUES.md` §1 and the manuscript.
- **Minor (disclosed):** a few exact duplicates sit entirely *within* the test
  partition (CCSN 3, GCD 6-class 32, GCD 5-class 1, harmonized 4 extra copies).
  The image-level bootstrap counts them as independent draws, marginally
  narrowing CIs. Negligible for harmonized (4 / 3,330); larger relatively for
  GCD 6-class. Worth a one-line note wherever the GCD 6-class CIs are used.

## 2. Exclusions (contrails, clear sky, mixed) — **Sound**

- All three exclusions are defensible for a *cloud-type* classification task and
  the counts are exact (200 `Ct`, 955 `7_mixed`, 3,739 `4_clearsky`).
- The `4_clearsky` argument is particularly strong: CCSN contains zero clear-sky
  images, so keeping it would hand the model a pure dataset-identity shortcut.
- These are documented as task-scoping decisions, not data manipulation.

## 3. Five-class compatibility taxonomy — **Sound (with correct framing)**

- The mapping is not an arbitrary genus merge: each harmonized class corresponds
  **one-to-one to a GCD operational category**, with CCSN genera folded in to
  match GCD's own coarsening (GCD `2_altocumulus` already means *altocumulus +
  cirrocumulus*; `5_stratocumulus` already means *stratocumulus + stratus +
  altostratus*; etc.). This is the most defensible way to build a shared label
  space from these two datasets and the manuscript should keep foregrounding it.
- Correctly framed as an *operational compatibility taxonomy*, not a
  genus-preserving WMO scheme (D-007, and the 2026-09-05 review endorses the
  phrasing).
- **Inherited caveat:** folding altostratus (a mid-level layer cloud) into a
  "stratocumulus" bin is physically loose — but that is GCD's coarsening, not a
  choice made here, and it is disclosed.

## 4. Split protocol (`grouped_stratified_holdout` v1.0) — **Sound**

- Class-stratified, duplicate-group-aware, single fixed 64/16/20 partition at
  seed 42. Appropriate for the sample sizes; a permanent holdout that never
  informs selection is the correct discipline (D-009).
- **Caveat:** it is a *single* holdout, so the reported numbers carry
  partition-choice variance that is not quantified. Acceptable for a thesis
  baseline; a k-fold or multi-seed repartition would let you put an interval on
  it. The generator infrastructure for this exists but was deliberately not run.

## 5. Training procedure — **Sound with caveats**

- ImageNet-pretrained ResNet, custom regularised head, differential learning
  rates (backbone 5e-5 / head 5e-4), AdamW with decoupled weight decay, cosine
  schedule, fixed 15-epoch budget with minimum-val-loss checkpoint restore —
  all standard and correctly implemented.
- Physically conservative augmentation (no vertical flip) is well motivated;
  keep the *simple* rationale ("ground-based images have a meaningful vertical
  orientation"), not the stronger physical claims the 2026-09-05 review flagged.
- **Caveat 1 — class imbalance:** the harmonized set is ~3.7:1
  (cumulonimbus:cumulus). Training uses no class weighting, no class-balanced
  sampler, and `label_smoothing = 0`. This is *mitigated* by reporting balanced
  accuracy and macro-F1 alongside accuracy, but the model is still fit to the
  imbalanced prior. A weighted loss or class-balanced sampler would be a
  reasonable ablation.
- **Caveat 2 — joint sampler balances sources, not classes:** the joint model
  still sees GCD's cumulonimbus-heavy class distribution; "source-balanced" is
  not "class-balanced".
- **Caveat 3 — input resolution:** GCD (512²) is downsampled ~5× to 224²; CCSN
  (400²/256²) much less. Fine-texture classes (cirrus, altocumulus) may be
  disadvantaged on GCD. Not discussed in the manuscript; worth a sentence.

## 6. Model / checkpoint selection — **Sound**

- Selection is strictly on validation data: runner restores the minimum
  validation-loss checkpoint; the test holdout is touched once. Verified in code
  (`run_harmonized.py` compares `val_loss` only).
- **Disclosed subtlety:** for the joint model, validation loss is sample-averaged
  over 374 CCSN + 2,289 GCD images, so checkpoint selection is ~86% weighted to
  GCD. Correctly documented in `OFFICIAL_PROTOCOL.md` §4.

## 7. Hyperparameter tuning — **Weak point (disclosed as D-011)**

- Ten trials, screened at **5 epochs**, with the selected trial's `best_epoch`
  landing on epoch 5 (the last) for all three architectures — i.e. the sweep
  ranks configurations **before convergence**, then the winner is retrained for
  15. Rankings at epoch 5 need not hold at epoch 15.
- Trial 08 (the selected recipe) is **6th of 10** on the ResNet-18 sweep by
  validation accuracy; it wins only for ResNet-34. The rationale — "stable
  convergence across the whole family" rather than peak per-model score — is
  reasonable but currently under-argued. D-011 hedges appropriately ("not the
  global best"); the manuscript should state the selection criterion explicitly
  (stability + cross-family consistency, screened at 5 epochs).
- The tuner's selection metric (unsmoothed val CE, macro-F1 tiebreak) is sound;
  the issue is the 5-epoch screen, not the metric.

## 8. Evaluation & uncertainty — **Sound**

- Accuracy + balanced accuracy + macro-F1, per-class tables, confusion matrices,
  and 1,000-sample image-level percentile bootstrap CIs (`seed = 42`). Method
  correctly described as **conditional on the fitted model and this test set**.
- Headline = source-balanced average, the right choice given the 6:1 GCD:CCSN
  test ratio; pooled accuracy is reported as secondary.
- Correctly **avoids** a statistical-equivalence claim for ResNet-18 vs
  ResNet-50 (0.08 pp).
- **Not done (and not claimed):** a paired bootstrap on the *difference* between
  architectures, and any estimate of training-seed variance (single seed → the
  CIs look more authoritative than the experiment supports). Both are disclosed
  as limitations rather than glossed.

## 9. Transfer & joint analyses — **Sound, n = 1**

- Cross-source transfer: one model per direction, one seed. The asymmetry
  (CCSN→GCD 57.9% vs GCD→CCSN 37.2% on ResNet-18) is large enough to be real,
  but the mechanism is correctly stated as an **observational hypothesis** —
  "cross-source accuracy comparisons aggregate multiple confounding domain
  shifts" (manuscript §Transfer).
- Joint-training gain (+4.06 pp on the CCSN component) is correctly attributed
  to the **whole recipe** (expanded update budget + source-balanced draws +
  cross-source features), not isolated feature transfer — the update-budget
  disparity (3.56 CCSN draws/image vs 1) is quantified in `OFFICIAL_PROTOCOL.md`.

---

## What would strengthen the work (priority order)

1. **Multi-seed production runs** (≥3 seeds) for at least ResNet-18 — turns the
   single-point numbers into intervals that include training variance. Biggest
   credibility gain for the least conceptual change.
2. **State the tuning-selection criterion explicitly** in the manuscript, and
   ideally re-screen the top ~3 trials at the full 15 epochs before committing.
3. **One class-imbalance ablation** (weighted CE or class-balanced sampler) to
   show the headline is not an artifact of fitting the prior.
4. **Paired bootstrap** on ResNet-18 − ResNet-50 (and joint − CCSN-only) so the
   comparisons rest on the difference distribution, not two overlapping CIs.
5. **A sentence on input resolution** (GCD 512→224) in the limitations.
6. **Near-duplicate (perceptual) audit** of GCD, or at minimum an explicit
   statement of expected residual leakage magnitude on the 89% GCD number.
7. Longer term: an **external test set** (e.g. a third whole-sky dataset) is the
   only real check on cross-station generalization — already deferred to Phase 5.

None of items 1–7 are blockers for reporting the current results *as a
duplicate-controlled baseline with single-partition, single-seed point
estimates* — which is exactly how the documents frame them.
