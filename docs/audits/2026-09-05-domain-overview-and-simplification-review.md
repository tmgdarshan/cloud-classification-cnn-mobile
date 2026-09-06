# Domain overview and simplification review - 2026-09-05

Role assumed for this review: atmospheric researcher specializing in cloud physics, with applied ML/CNN experience. This is a read-only expert review of the current project shape after the mechanical fixes. It does not retrain models, regenerate manifests, edit source code, or change scientific results.

## Overall assessment

The project has a valid scientific core:

- two real ground-based cloud image datasets, CCSN and GCD;
- a documented duplicate/leakage problem;
- canonical duplicate-aware manifests;
- a reasonable five-class compatibility task;
- ResNet transfer-learning baselines;
- cross-source transfer and joint-training experiments;
- bootstrap uncertainty estimates and confusion matrices.

The main issue is not lack of rigor. The project has too many parallel explanations of the same rigor. README, dashboard, decisions, configs, code comments, manuscript, and legacy scripts repeat the taxonomy, split protocol, augmentation rationale, and "official" workflow with slightly different wording. That makes the work look more complicated than it is, and it creates drift.

The clean thesis story should be:

> We study cross-source ground-based cloud classification using CCSN and GCD. Because the raw data contain duplicate/repeated images, we evaluate only on canonical duplicate-group-aware manifests. CCSN and GCD labels are mapped into a five-class compatibility taxonomy for shared experiments. ResNet backbones pretrained on ImageNet are fine-tuned under source-specific and joint-training conditions. Validation data are used for model selection; the final test holdout is used for reporting.

That story keeps the rigor and removes much of the machinery from the reader's path.

## Review tracks

### 1. Atmospheric and cloud-physics framing

The five-class target is defensible if it is framed as a compatibility taxonomy, not a physically complete WMO taxonomy. The mapping in `metadata/splits/harmonized_5bin_canonical.json` and `scripts/build_canonical_manifests.py` groups CCSN fine labels and GCD operational labels into cumulus, altocumulus, cirrus, stratocumulus, and cumulonimbus.

This is a practical ML target. It is not a full atmospheric taxonomy, because several GCD classes already merge multiple WMO genera. For example, GCD "altocumulus" includes altocumulus and cirrocumulus; GCD "stratocumulus" includes stratocumulus, stratus, and altostratus; GCD "cumulonimbus" includes cumulonimbus and nimbostratus. The manuscript and README should avoid saying the mapping is "mathematically grounded" or genus-preserving. "Cross-dataset compatibility taxonomy" is the right phrase.

The current cloud-physics language is sometimes too strong. Claims about fisheye optics causing particular transfer failures, peripheral distortion, microphysical texture loss, or horizon artifacts should be presented as plausible interpretations unless you run targeted evidence, such as source-stratified error analysis by image region, saliency/occlusion checks, or camera geometry metadata. The model sees RGB texture and geometry; it does not observe thermodynamic phase, cloud-base height, or microphysics directly.

Vertical flipping should stay prohibited for thesis baseline runs. The simpler rationale is enough: ground-based images have a meaningful vertical image orientation, so vertical flips can create views outside the acquisition distribution. Avoid saying vertical flips invert zenith radius in centered fisheye images, because a vertical reflection preserves distance from the center. Horizontal flip and small rotations are reasonable approximations only if framed as image-level invariances, not as atmospheric laws.

Suggested simplification:

- Replace heavy phrases like "atmospheric physics-preserving augmentations" with "physically conservative augmentations."
- Replace "mathematically grounded 5-bin taxonomy" with "approved five-class compatibility taxonomy."
- Keep clear sky, mixed cloud, and contrails out of the harmonized cloud-only task, but state that this narrows the task rather than "eliminating distribution collapse."

### 2. Naming and scientific terminology

The naming should be calmer and more literal.

Current terms that should be simplified:

- "Harmonized" and "merged" are used together. Use "joint" for the training condition and "harmonized" for the label space.
- "Zero-shot" is defensible only in the narrow sense that the fitted weights did not train on the target-source images. Because tuning used the harmonized validation pool, use "cross-source transfer" in the main text, and explain the exact selection protocol.
- "Permanent test holdout" is useful, but it should be attached to one protocol. Because holdout membership changes across some manifests, do not imply a universal permanent holdout shared by every taxonomy.
- "5-bin" and "5-class" both appear. Choose one public term. I recommend "five-class compatibility taxonomy" in prose and `harmonized_5bin` only as a file key.
- CCSN should be expanded as Cirrus Cumulus Stratus Nimbus, not Nematocumulus.

Recommended naming scheme:

| Concept | Use this | Avoid this |
|---|---|---|
| Shared label space | five-class compatibility taxonomy | mathematically grounded WMO harmonization |
| Joint model | joint CCSN+GCD model | merged domain-generalized engine |
| Transfer evaluation | CCSN-to-GCD transfer, GCD-to-CCSN transfer | strict zero-shot unless selection is source-only |
| Split protocol | 80% development / 20% test; validation inside development | primary 64/16/20 story in prose |
| Canonical data | canonical manifest | database, magical permanent benchmark |

### 3. Data and split protocol

The train/validation/test split is methodologically correct for ML work. The problem is presentation. A thesis reader expects train/test, and then accepts validation once it is explained as model selection data.

The best explanation is:

- The final evaluation uses a held-out 20% test set.
- The remaining 80% is the development pool.
- Inside the development pool, one fold is used for validation/checkpoint selection and the rest for fitting weights.

That is scientifically clearer than leading with 64/16/20. It also answers the user's concern: the project is still a train/test study at the top level, with validation used inside training discipline.

Do not regenerate the manifests just to simplify language. The current canonical manifests are already the evidence-bearing artifacts. Changing them would change the experiment.

The cross-manifest issue from the previous audit still matters: holdout membership changes between baseline and harmonized manifests. This does not automatically invalidate the harmonized table, but it means the paper should not compare across manifest protocols as if every model shared one universal final test set.

Suggested simplification:

- Keep `metadata/splits/*_canonical.json` as the single source of truth.
- Stop repeating mapping tables in README, comments, dashboard, and manuscript except where needed for reader understanding.
- Present split logic as "development/test" first; only expose train/validation details in the methods subsection.
- Treat `scripts/build_canonical_manifests.py` as a reproducibility script, not something users casually rerun.

### 4. ML/CNN methodology

ResNet-18, ResNet-34, and ResNet-50 are good baseline choices. They are familiar, reviewer-friendly, and suitable for testing whether more depth helps transfer. The current results suggest the larger models do not clearly dominate source-balanced performance. That is a useful conclusion: ResNet-18 can be the primary operational baseline, with ResNet-34/50 as capacity sensitivity checks.

The custom head in `src/run_harmonized.py` uses dropout, a 256-unit layer, batch normalization, GELU, and another dropout before the classifier. That is acceptable, but it should not be oversold. The thesis can simply say "the ImageNet classifier head was replaced with a small regularized five-class head." The exact architecture can go in a table.

Differential learning rates are reasonable: smaller updates for pretrained convolutional features and larger updates for the new classifier head. AdamW and cosine annealing are also reasonable baseline choices.

The tuning workflow is now improved because `src/tune_resnet_family.py` records and ranks `val_unsmoothed_loss` as a common selection score. The remaining scientific issue is that the saved tuned configs are marked `approval_status = "approved"` automatically. It would be cleaner to separate "algorithmically selected" from "researcher-approved." A config can be selected by a sweep without claiming human methodological approval.

Pretrained ImageNet weights should be recorded as part of run provenance. Right now model constructors load torchvision ImageNet weights directly. For reproducibility and HPC/offline execution, the project should record the exact weight enum and either pre-cache weights or allow a local weight path.

Suggested simplification:

- Make ResNet-18 the headline model; keep ResNet-34/50 as sensitivity analysis.
- Do not call the tuning result "optimal." Call it "selected baseline configuration."
- Describe the head once, in one methods table.
- Use a small helper for model construction shared by tuning and production if code is simplified later.

### 5. Evaluation and statistics

The metric set is reasonable: top-1 accuracy, balanced accuracy, macro-F1, class-level precision/recall/F1, and confusion matrices. Balanced accuracy and macro-F1 are especially important because the classes are imbalanced.

The bootstrap CIs in `src/atmospheric_evaluation.py` are useful but should be described precisely: they are image-level percentile bootstrap intervals conditional on the fitted model and current test set. They do not capture training-seed variability, unknown temporal dependence, camera/session dependence, or paired uncertainty in model differences.

The source-balanced average is a useful reporting metric for cross-source fairness, but it is not a new physical metric. It is the arithmetic mean of CCSN and GCD source-specific scores. Report it plainly.

For claims like "joint training improves CCSN by +4.06 percentage points," the rigorous version is:

> In this run, joint training improved CCSN holdout accuracy by 4.06 percentage points relative to the CCSN-only baseline.

To claim statistical significance, save per-image predictions and run paired bootstrap or McNemar-style comparison on the same test images.

Suggested simplification:

- Keep one main table with accuracy, balanced accuracy, macro-F1.
- Move per-class precision/recall/F1 to appendix or artifacts.
- Phrase CIs as conditional image-bootstrap CIs.
- Avoid significance language unless paired tests are added.

### 6. Execution and software architecture

The current official code path is mostly clear:

- `src/run_harmonized.py` for production experiments;
- `src/tune_resnet_family.py` for tuning;
- `src/atmospheric_evaluation.py` for reports;
- `metadata/splits/*_canonical.json` for partitions;
- `config/training/tuned_resnet*.toml` for selected hyperparameters.

The complication comes from older architecture layers and legacy paths still sitting near the official workflow. The generic config loader, dataset index, manifest loaders, split generator, and legacy engines are useful historical infrastructure, but the thesis runner bypasses much of it.

This is not necessarily bad for research code. It becomes bad when the docs imply every layer is part of the active workflow.

Suggested simplification:

- Declare one "official thesis pipeline" in README.
- Move all prototype claims to `legacy` documentation.
- Keep config files only if the runner consumes them.
- Avoid creating separate dataset configs for derived harmonized views unless they are just pointers.
- Do not call the image collection a database unless a real database exists.

### 7. Configuration and provenance

The simplified `config/datasets/harmonized_5bin.toml` is now the right idea: it points to the canonical manifest and source datasets instead of repeating the mapping. That avoids drift.

The remaining issue is that mappings still appear in multiple places:

- decisions;
- manifest;
- manifest builder;
- source comments;
- manuscript;
- README.

Some repetition is normal, but only one place should be authoritative. I recommend:

- authoritative machine source: `metadata/splits/harmonized_5bin_canonical.json`;
- human decision source: `docs/DECISIONS.md`;
- builder source: `scripts/build_canonical_manifests.py`;
- README/manuscript: short explanation only.

The registry is useful but incomplete. It currently records the joint merged runs better than the single-source and transfer conditions. For future reruns, registry entries should include full resolved config, git status, torchvision weight enum, manifest hash, checkpoint hash, environment, and per-image prediction file.

Suggested simplification:

- Keep TOML configs small.
- Generate method tables from JSON/TOML instead of typing values repeatedly.
- Treat the registry as provenance, not the source of truth for scientific claims until it stores complete run inputs and predictions.

### 8. Report/LaTeX readiness

The LaTeX should be redone after the project language is simplified. Do not patch it sentence by sentence yet; too many sections need conceptual tightening.

The future LaTeX should:

- correct CCSN/GCD dataset facts and citations;
- remove unimplemented dHash/SSIM and horizon-mask claims;
- replace strict causal transfer explanations with evidence-bounded interpretations;
- present the split protocol as 80% development / 20% final test;
- explain validation as model selection within development;
- rename zero-shot to cross-source transfer unless strict source-only hyperparameter selection is rerun;
- generate CI values directly from saved JSON;
- fix clipped tables and figure references.

### 9. What can be simplified without losing rigor

High-confidence simplifications:

1. Make `src/run_harmonized.py` the only active production runner.
2. Keep SG-HCV as legacy/supplementary unless you need it for a separate validation study.
3. Present the split as development/test, not as a complex three-way hierarchy.
4. Make `harmonized_5bin.toml` a pointer profile only.
5. Remove repeated mapping prose from README and dashboard.
6. Rename "zero-shot" to "cross-source transfer" in public-facing text.
7. Use "selected baseline" instead of "optimal configuration."
8. Make ResNet-18 the primary model and ResNet-34/50 secondary capacity checks.
9. Keep atmospheric augmentation rationale short and conservative.
10. Put detailed per-class tables and CI diagnostics in appendix/artifacts.

Simplifications that need researcher approval because they change scientific behavior:

1. Regenerating train/validation/test or train/test manifests.
2. Changing the five-class mapping.
3. Removing validation data entirely.
4. Changing augmentation policy.
5. Rerunning tuning with source-only selection.
6. Rerunning experiments with new seeds or paired statistical testing.

## Recommended next move before LaTeX

Create a small "official protocol" document that has only:

1. datasets used;
2. exclusions;
3. five-class compatibility mapping;
4. split rule as 80% development / 20% test;
5. validation/checkpoint rule;
6. models and selected hyperparameters;
7. metrics and uncertainty definition;
8. exact commands.

Then make README, dashboard, and LaTeX point to that protocol instead of restating everything. This will make the project easier to defend because reviewers will see one coherent method instead of many overlapping explanations.

