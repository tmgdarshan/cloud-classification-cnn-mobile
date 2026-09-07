# Independent Audit - ResNet-18 Multi-Seed Harmonized Benchmark

Date: 2026-09-07

Scope: branch `feature/independent-pool-tuning`, compared to `origin/main` at `15e6df57b5b8eb2b5c54cfe6714d9381978312e9`. Evidence bundle: `artifacts/audit_2026-09-07_independent/evidence.json`.

## Checklist Verdicts

| Item | Verdict | Evidence |
|---|---|---|
| A1 code hunks semantics-preserving | FAIL | `git diff -U0 origin/main -- src/run_harmonized.py src/tune_resnet_family.py` shows throughput/logging hunks, selection-logging hunks, and semantic tuning hunks. Semantic examples: `src/tune_resnet_family.py:581-642` adds `--pool` selective tuning data; `src/tune_resnet_family.py:495-542` changes emitted config schema/epochs by pool. Forbidden model/loss/transform core stayed textually unchanged at `src/run_harmonized.py:158-170`, `src/run_harmonized.py:175-204`, `src/run_harmonized.py:216-222`, `src/run_harmonized.py:312`, `src/run_harmonized.py:323`. |
| A2 cuDNN flags | PASS | Current: `src/run_harmonized.py:58`, `src/tune_resnet_family.py:41` set `torch.backends.cudnn.benchmark = False`. `git show origin/main:... | Select-String "torch.backends.cudnn"` also returns only `benchmark = False`; no `cudnn.deterministic` assignment in either revision. |
| A3 symmetric DataLoader change | FAIL | Production train loaders use seeded generator/workers/persistent workers at `src/run_harmonized.py:257-299`; validation loader remains `num_workers=0`, `pin_memory=False` at `src/run_harmonized.py:294-300`; test eval also remains `num_workers=0`, `pin_memory=False` at `src/run_harmonized.py:452-457`. |
| A4 deterministic rerun seed 42 twice | FAIL | Same default seed-42 rerun command failed twice before epoch 1 with `RuntimeError: bad allocation` at `src/run_harmonized.py:350`; no second completed accuracy could be compared. Command printed RTX 4070/CUDA before failure. |
| B1 test split read/use | FAIL | `run_harmonized.py` preloads test tensors before training at `src/run_harmonized.py:692` and `src/run_harmonized.py:707`. I found no path using them for checkpoint selection, but the literal "never loaded during training" condition is violated. |
| B2 tuner skips test | PASS | `src/tune_resnet_family.py:585-589` reads manifest samples and `continue`s when `split == "test"`. |
| B3 checkpoint selection uses validation only | PASS | Validation loop is `val_loader` only at `src/run_harmonized.py:367-387`; min-loss and max-F1 snapshots are selected at `src/run_harmonized.py:392-405`. |
| B4 per-pool tuning isolation | PASS | CCSN pool appends only CCSN train/val at `src/tune_resnet_family.py:592-596`; GCD pool appends only GCD train/val at `src/tune_resnet_family.py:597-601`; audit counts match CCSN `1495/374`, GCD `9155/2289`. |
| C1 manifest counts | PASS | Audit recomputation: `ccsn train/val/test = 1495/374/468`, `gcd = 9155/2289/2862`, total `10650/2663/3330`; matches `metadata/splits/harmonized_5bin_canonical.json:44-57`. |
| C2 class index order | PASS | Manifest `class_to_idx` at `metadata/splits/harmonized_5bin_canonical.json:12-18` is `cumulus:0, altocumulus:1, cirrus:2, stratocumulus:3, cumulonimbus:4`; audit found `label_mismatch_count=0` for both sources. |
| C3 duplicate/hash split leakage | PASS | Audit hashed all manifest-referenced local images: `missing_file_count=0`, `group_leak_count=0`, `image_hashes_in_multiple_splits=0`. |
| D1 budget arithmetic | PASS | DataLoader does not set `drop_last`, so default is `False`. Audit: `ceil(1495/64)=24`, `ceil(9155/64)=144`; CCSN90 `24*90=2160`, GCD15 `144*15=2160`. |
| D2 config epoch/identity checks | FAIL | Epochs pass: `tuned_resnet18_ccsn.toml:19` is `90`, `tuned_resnet18_ccsn_15ep.toml:19` is `15`. But the two CCSN configs also differ in `name` and budget-note metadata, and `git diff --no-index` shows `tuned_resnet18_joint.toml` differs from approved `tuned_resnet18.toml` in comment/name, so not byte-identical. |
| D3 configs match winning trials | PASS | Audit parsed TOML plus tuning JSON: CCSN config matches `trial_03_aggressive_adamw`; GCD and joint match `trial_08_no_label_smoothing`; see `evidence.json:config_checks.config_matches_winner = true` for all three. |
| E1 per-run metric recomputation | CANNOT-VERIFY | No saved predictions or machine-readable confusion matrices found under `artifacts/harmonized_thorough`; `src/evaluation.py:97-184` computes `cm` but only saves PNG plus rounded JSON summary. I could only verify internal consistency from rounded per-class support/recall/F1. |
| E2 three-seed means/std | PASS | Recomputed from run summary JSONs in `evidence.json:three_seed_metrics`. Agent's master statistics table was not found in repo, so direct comparison to that table is not possible. |
| E3 paired contrasts/stats | CANNOT-VERIFY | I recomputed paired differences and paired sample stds in `evidence.json:contrasts`; no agent master contrast table was present to confirm whether it used paired std. |
| E4 bootstrap CI method | PASS | `src/evaluation.py:27-82`: percentile bootstrap, default `n_bootstraps=1000`, fixed `RandomState(seed)`, same `boot_idx` indexes `y_true` and `y_pred`. Called with `seed=42` at `src/evaluation.py:108`. |
| E5 CCSN-15 anchor vs approved | FAIL | Approved single-run CCSN-15 accuracy is `56.84%`; recomputed current 3-seed min-loss mean is `58.9033%`, sample std `0.9617`; difference `2.0633pp` is slightly greater than `2*std=1.9234pp`. |
| F1 F1 checkpoint selection | PASS | Max-F1 selection uses `if cur_v_f1 > best_val_f1` at `src/run_harmonized.py:400-405`; sidecars name criterion `max_val_macro_f1`; seed43 CCSN90 loss/F1 checkpoint hashes differ: `B6B150...` vs `D94BF3...`. |
| F2 both checkpoints evaluated separately | PASS | Loss eval paths at `src/run_harmonized.py:823-840` and `1145-1173`; F1 eval paths at `src/run_harmonized.py:843-860` and `1186-1214`; results stored separately under `criterion_loss` and `criterion_f1` at `src/run_harmonized.py:866-885`, `1230-1257`. |
| F3 CCSN best epochs erratic | PASS | Audit table: CCSN15 loss/F1 epochs `7/7`, `12/10`, `4/10`; CCSN90 loss/F1 epochs `36/36`, `42/64`, `51/36`. Erratic enough to support the diagnostic, though I did not see a ~77 epoch in these ResNet-18 artifacts. |
| G1 fresh checkpoints | PASS | `evidence.json:checkpoint_reused_flags` contains only `False`. |
| G2 CUDA used / CPU aborts | PASS | Artifacts say RTX 4070 in `artifacts/harmonized_thorough/compute_environment.json`; code aborts on CPU at `src/run_harmonized.py:541-542`, `src/tune_resnet_family.py:551-552`; rerun output printed `GPU=NVIDIA GeForce RTX 4070`. |
| G3 approved artifacts/configs not modified | PASS | `git diff --name-status origin/main -- artifacts/harmonized_results config/training/tuned_resnet18.toml config/training/tuned_resnet34.toml config/training/tuned_resnet50.toml` returned no paths. Mtimestamps show approved outputs from 2026-09-05/06, not this audit run. |
| G4 commits/pushes | PASS | `git rev-parse HEAD origin/main` both return `15e6df57b5b8eb2b5c54cfe6714d9381978312e9`; `git log origin/main..HEAD` returned empty. No local commit exists to push from this branch. |
| H1 independent seed-43 rerun | FAIL | Blocked by same runtime failure seen in A4: the default CCSN-15 run fails at first forward pass with `RuntimeError: bad allocation`; no seed-43 accuracy could be generated under the claimed path. |

## Recomputed ResNet-18 Contrasts

All values are paired by seed `[42,43,44]`; dispersion is sample std of paired differences (`ddof=1`). "Detectable" is the requested descriptive two-sigma screen: `abs(mean) > 2*paired_std`.

### Min Validation Loss

| Contrast | Accuracy pp mean/std/verdict | Balanced acc pp mean/std/verdict | Macro-F1 pp mean/std/verdict |
|---|---:|---:|---:|
| Budget: CCSN90 - CCSN15 | `+2.2767 / 1.2532 / not_detectable` | `+2.0767 / 0.6652 / detectable` | `+2.2300 / 1.4551 / not_detectable` |
| Joint: Joint_on_CCSN - CCSN90 | `-0.2100 / 1.0700 / not_detectable` | `+0.7300 / 0.4419 / not_detectable` | `+0.5133 / 0.4244 / not_detectable` |
| OldClaim: Joint_on_CCSN - CCSN15 | `+2.0667 / 0.8065 / detectable` | `+2.8067 / 0.7067 / detectable` | `+2.7433 / 1.0339 / detectable` |

### Max Validation Macro-F1

| Contrast | Accuracy pp mean/std/verdict | Balanced acc pp mean/std/verdict | Macro-F1 pp mean/std/verdict |
|---|---:|---:|---:|
| Budget: CCSN90 - CCSN15 | `+1.2067 / 0.6178 / not_detectable` | `-0.4033 / 1.7302 / not_detectable` | `-0.2967 / 1.0347 / not_detectable` |
| Joint: Joint_on_CCSN - CCSN90 | `+1.4967 / 0.8550 / not_detectable` | `+3.8433 / 0.4155 / detectable` | `+3.6067 / 0.3089 / detectable` |
| OldClaim: Joint_on_CCSN - CCSN15 | `+2.7033 / 1.0565 / detectable` | `+3.4400 / 1.5907 / detectable` | `+3.3100 / 0.8487 / detectable` |

## Material Findings

1. The code changes are not purely throughput/logging: `tune_resnet_family.py` now changes the tuning data pool and generated configs by pool. That invalidates a blanket "throughput/logging only" claim.
2. The production runner preloads test tensors before training. I found no checkpoint-selection/test-label leakage path, but it violates the stricter audit requirement that test samples never be loaded during training.
3. The independent rerun path fails with `RuntimeError: bad allocation` under the default claimed DataLoader settings, so the reported runs are not independently reproducible in this environment as-is.
4. The budget-matched conclusion changes the story: with min-loss selection, Joint_on_CCSN minus CCSN90 is not detectable for accuracy, balanced accuracy, or macro-F1. The old Joint minus CCSN15 contrast remains detectable, but that is confounded by training budget.
5. The saved artifacts omit per-image predictions and machine-readable confusion matrices, so exact metric recomputation from raw predictions is impossible from current artifacts.
6. The CCSN-15 three-seed mean fails the requested sanity anchor against the approved single-run 56.84% by a small margin.

## Nits

- `git diff --check origin/main -- src/run_harmonized.py src/tune_resnet_family.py` reports `src/run_harmonized.py:1400: new blank line at EOF`.
- `tuned_resnet18_joint.toml` is hyperparameter-equivalent to the approved `tuned_resnet18.toml`, but not byte-identical due comment/name.
- `full_pipeline.log` contains UTF-16-looking NUL-separated text and records a later ResNet-34 pipeline failure from another run, not a clean completed master log.

## Final Judgment

The ResNet-18 multi-seed result is useful but not yet trustworthy enough to rewrite Section 6 as a strong joint-training win. The data integrity and validation-only checkpoint selection look solid, but the audit finds non-logging semantic tuning changes, asymmetric loader changes, missing raw prediction artifacts, a failed independent rerun, and a budget-matched contrast that is not detectable under min-validation-loss selection. The defensible rewrite is narrower: joint training improves over the original 15-epoch CCSN-only baseline, but after matching CCSN-only update budget, the min-loss evidence does not support a robust additional joint-training gain on the CCSN holdout.
