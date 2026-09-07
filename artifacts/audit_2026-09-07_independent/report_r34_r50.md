# ResNet-34/50 Harmonized Matrix Independent Audit

- Branch: `feature/independent-pool-tuning`; HEAD `15e6df57b5b8eb2b5c54cfe6714d9381978312e9`; `origin/main` `15e6df57b5b8eb2b5c54cfe6714d9381978312e9`; requested baseline `15e6df57`.
- Compute environment: CUDA available = `True`, device = `NVIDIA GeForce RTX 4070`, Torch CUDA = `12.4`.
- Inventory: 17/24 requested ResNet-34/50 runs have summaries and both checkpoint+meta pairs. Missing/incomplete: resnet50 ccsn90 seed44, resnet50 gcd15 seed42, resnet50 gcd15 seed43, resnet50 gcd15 seed44, resnet50 joint15 seed42, resnet50 joint15 seed43, resnet50 joint15 seed44.

## Run Inventory
| Arch | Arm | Seed Status | Reuse Flags loss/f1 |
| --- | --- | --- | --- |
| resnet34 | ccsn15 | 42:ok, 43:ok, 44:ok | 42:False/False, 43:False/False, 44:False/False |
| resnet34 | ccsn90 | 42:ok, 43:ok, 44:ok | 42:False/False, 43:False/False, 44:False/False |
| resnet34 | gcd15 | 42:ok, 43:ok, 44:ok | 42:False/False, 43:False/False, 44:False/False |
| resnet34 | joint15 | 42:ok, 43:ok, 44:ok | 42:False/False, 43:False/False, 44:False/False |
| resnet50 | ccsn15 | 42:ok, 43:ok, 44:ok | 42:False/False, 43:False/False, 44:False/False |
| resnet50 | ccsn90 | 42:ok, 43:ok, 44:missing | 42:False/False, 43:False/False |
| resnet50 | gcd15 | 42:missing, 43:missing, 44:missing | n/a |
| resnet50 | joint15 | 42:missing, 43:missing, 44:missing | n/a |

## Config Identity
| Arch | joint cfg exists | joint vs approved diffs | CCSN90 ep | CCSN15 ep | joint winner | CCSN winner | GCD winner |
| --- | --- | --- | --- | --- | --- | --- | --- |
| resnet34 | False | "missing_joint_specific_config" | 90 | 15 | True | True | True |
| resnet50 | False | "missing_joint_specific_config" | 90 | 15 | True | True | True |

## Metric Internal Consistency
All recomputable per-class blocks matched top-line accuracy, balanced accuracy, and macro-F1 within the 0.02 pp rounding tolerance. Exceptions: none. Source-balanced rows are arithmetic source averages and have no per-class rows to recompute.

## Three-Seed Means
### resnet34
| Criterion | Arm | Target | Acc | Bal Acc | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| loss | ccsn15 | ccsn | 58.05 +/- 1.78 | 55.74 +/- 0.65 | 55.35 +/- 0.88 |
| loss | ccsn15 | gcd | 40.67 +/- 0.99 | 49.74 +/- 2.52 | 45.46 +/- 1.91 |
| loss | ccsn90 | ccsn | 57.48 +/- 1.62 | 55.12 +/- 0.70 | 54.79 +/- 0.50 |
| loss | ccsn90 | gcd | 38.87 +/- 1.96 | 48.05 +/- 2.57 | 43.63 +/- 2.04 |
| loss | gcd15 | gcd | 89.33 +/- 0.47 | 91.08 +/- 0.23 | 90.97 +/- 0.29 |
| loss | gcd15 | ccsn | 36.61 +/- 1.17 | 44.47 +/- 1.29 | 35.77 +/- 1.34 |
| loss | joint15 | ccsn | 59.90 +/- 2.48 | 58.12 +/- 1.29 | 57.88 +/- 2.14 |
| loss | joint15 | gcd | 87.59 +/- 2.11 | 88.70 +/- 3.12 | 88.89 +/- 2.94 |
| loss | joint15 | joint | 83.69 +/- 2.12 | 84.23 +/- 2.56 | 84.20 +/- 2.79 |
| loss | joint15 | source_balanced | 73.74 +/- 2.20 | 73.41 +/- 2.15 | 73.39 +/- 2.48 |

| Criterion | Arm | Target | Acc | Bal Acc | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| f1 | ccsn15 | ccsn | 57.91 +/- 1.06 | 55.86 +/- 0.72 | 55.53 +/- 1.10 |
| f1 | ccsn15 | gcd | 37.80 +/- 4.06 | 46.79 +/- 4.42 | 42.06 +/- 4.37 |
| f1 | ccsn90 | ccsn | 58.26 +/- 1.71 | 53.56 +/- 0.90 | 53.46 +/- 0.62 |
| f1 | ccsn90 | gcd | 41.03 +/- 2.79 | 47.99 +/- 0.45 | 43.89 +/- 2.07 |
| f1 | gcd15 | gcd | 89.19 +/- 0.53 | 91.16 +/- 0.75 | 90.95 +/- 0.69 |
| f1 | gcd15 | ccsn | 36.25 +/- 1.01 | 44.92 +/- 1.70 | 35.19 +/- 0.92 |
| f1 | joint15 | ccsn | 60.97 +/- 0.65 | 58.31 +/- 0.91 | 58.59 +/- 0.67 |
| f1 | joint15 | gcd | 88.89 +/- 0.24 | 90.49 +/- 0.34 | 90.67 +/- 0.26 |
| f1 | joint15 | joint | 84.96 +/- 0.13 | 85.61 +/- 0.22 | 85.84 +/- 0.15 |
| f1 | joint15 | source_balanced | 74.93 +/- 0.21 | 74.40 +/- 0.43 | 74.63 +/- 0.34 |

### resnet50
| Criterion | Arm | Target | Acc | Bal Acc | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| loss | ccsn15 | ccsn | 59.12 +/- 1.37 | 57.70 +/- 1.77 | 57.24 +/- 1.43 |
| loss | ccsn15 | gcd | 41.13 +/- 3.18 | 48.68 +/- 0.90 | 43.13 +/- 0.52 |
| loss | ccsn90 | ccsn | incomplete | incomplete | incomplete |
| loss | ccsn90 | gcd | incomplete | incomplete | incomplete |
| loss | gcd15 | gcd | incomplete | incomplete | incomplete |
| loss | gcd15 | ccsn | incomplete | incomplete | incomplete |
| loss | joint15 | ccsn | incomplete | incomplete | incomplete |
| loss | joint15 | gcd | incomplete | incomplete | incomplete |
| loss | joint15 | joint | incomplete | incomplete | incomplete |
| loss | joint15 | source_balanced | incomplete | incomplete | incomplete |

| Criterion | Arm | Target | Acc | Bal Acc | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| f1 | ccsn15 | ccsn | 61.04 +/- 1.01 | 59.30 +/- 1.15 | 59.10 +/- 1.31 |
| f1 | ccsn15 | gcd | 41.00 +/- 2.15 | 46.61 +/- 2.67 | 40.64 +/- 2.72 |
| f1 | ccsn90 | ccsn | incomplete | incomplete | incomplete |
| f1 | ccsn90 | gcd | incomplete | incomplete | incomplete |
| f1 | gcd15 | gcd | incomplete | incomplete | incomplete |
| f1 | gcd15 | ccsn | incomplete | incomplete | incomplete |
| f1 | joint15 | ccsn | incomplete | incomplete | incomplete |
| f1 | joint15 | gcd | incomplete | incomplete | incomplete |
| f1 | joint15 | joint | incomplete | incomplete | incomplete |
| f1 | joint15 | source_balanced | incomplete | incomplete | incomplete |

## Paired Contrasts
### resnet34
| Criterion | Contrast | Acc diff | Bal Acc diff | Macro-F1 diff |
| --- | --- | --- | --- | --- |
| loss | Budget_CCSN90_minus_CCSN15 | -0.57 +/- 0.32 (not detectable) | -0.61 +/- 0.54 (not detectable) | -0.56 +/- 0.42 (not detectable) |
| loss | Joint_JointOnCCSN_minus_CCSN90 | 2.42 +/- 0.99 (detectable) | 3.00 +/- 1.74 (not detectable) | 3.09 +/- 1.65 (not detectable) |
| loss | OldClaim_JointOnCCSN_minus_CCSN15 | 1.85 +/- 1.10 (not detectable) | 2.38 +/- 1.26 (not detectable) | 2.53 +/- 1.26 (detectable) |
| loss | Transfer_CCSN15onGCD_minus_GCD15onCCSN | 4.06 +/- 2.08 (not detectable) | 5.28 +/- 3.03 (not detectable) | 9.69 +/- 2.91 (detectable) |
| loss | Transfer_GCD15onCCSN_minus_CCSN15onGCD | -4.06 +/- 2.08 (not detectable) | -5.28 +/- 3.03 (not detectable) | -9.69 +/- 2.91 (detectable) |

| Criterion | Contrast | Acc diff | Bal Acc diff | Macro-F1 diff |
| --- | --- | --- | --- | --- |
| f1 | Budget_CCSN90_minus_CCSN15 | 0.35 +/- 1.43 (not detectable) | -2.29 +/- 0.53 (detectable) | -2.07 +/- 0.63 (detectable) |
| f1 | Joint_JointOnCCSN_minus_CCSN90 | 2.71 +/- 1.57 (not detectable) | 4.75 +/- 1.64 (detectable) | 5.13 +/- 0.88 (detectable) |
| f1 | OldClaim_JointOnCCSN_minus_CCSN15 | 3.06 +/- 0.44 (detectable) | 2.46 +/- 1.20 (detectable) | 3.06 +/- 0.91 (detectable) |
| f1 | Transfer_CCSN15onGCD_minus_GCD15onCCSN | 1.54 +/- 3.26 (not detectable) | 1.86 +/- 2.73 (not detectable) | 6.87 +/- 3.77 (not detectable) |
| f1 | Transfer_GCD15onCCSN_minus_CCSN15onGCD | -1.54 +/- 3.26 (not detectable) | -1.86 +/- 2.73 (not detectable) | -6.87 +/- 3.77 (not detectable) |

### resnet50
| Criterion | Contrast | Acc diff | Bal Acc diff | Macro-F1 diff |
| --- | --- | --- | --- | --- |
| loss | Budget_CCSN90_minus_CCSN15 | incomplete | incomplete | incomplete |
| loss | Joint_JointOnCCSN_minus_CCSN90 | incomplete | incomplete | incomplete |
| loss | OldClaim_JointOnCCSN_minus_CCSN15 | incomplete | incomplete | incomplete |
| loss | Transfer_CCSN15onGCD_minus_GCD15onCCSN | incomplete | incomplete | incomplete |
| loss | Transfer_GCD15onCCSN_minus_CCSN15onGCD | incomplete | incomplete | incomplete |

| Criterion | Contrast | Acc diff | Bal Acc diff | Macro-F1 diff |
| --- | --- | --- | --- | --- |
| f1 | Budget_CCSN90_minus_CCSN15 | incomplete | incomplete | incomplete |
| f1 | Joint_JointOnCCSN_minus_CCSN90 | incomplete | incomplete | incomplete |
| f1 | OldClaim_JointOnCCSN_minus_CCSN15 | incomplete | incomplete | incomplete |
| f1 | Transfer_CCSN15onGCD_minus_GCD15onCCSN | incomplete | incomplete | incomplete |
| f1 | Transfer_GCD15onCCSN_minus_CCSN15onGCD | incomplete | incomplete | incomplete |

## Transfer Per-Class Recall
### resnet34
| Criterion | Class | CCSN15 on GCD | GCD15 on CCSN | Diff |
| --- | --- | --- | --- | --- |
| loss | cumulus | 46.99 +/- 11.32 | 75.00 +/- 2.78 | -28.01 +/- 8.54 (detectable) |
| loss | altocumulus | 76.95 +/- 3.23 | 59.11 +/- 4.65 | 17.84 +/- 7.63 (detectable) |
| loss | cirrus | 57.13 +/- 8.14 | 53.33 +/- 5.80 | 3.80 +/- 10.30 (not detectable) |
| loss | stratocumulus | 47.99 +/- 11.71 | 12.78 +/- 1.05 | 35.20 +/- 10.85 (detectable) |
| loss | cumulonimbus | 19.66 +/- 3.73 | 22.12 +/- 2.55 | -2.46 +/- 5.22 (not detectable) |

| Criterion | Class | CCSN15 on GCD | GCD15 on CCSN | Diff |
| --- | --- | --- | --- | --- |
| f1 | cumulus | 43.50 +/- 6.41 | 80.55 +/- 7.35 | -37.06 +/- 1.14 (detectable) |
| f1 | altocumulus | 75.93 +/- 3.11 | 59.79 +/- 4.12 | 16.14 +/- 3.14 (detectable) |
| f1 | cirrus | 62.12 +/- 9.71 | 52.94 +/- 3.11 | 9.18 +/- 6.90 (not detectable) |
| f1 | stratocumulus | 27.24 +/- 13.60 | 12.10 +/- 0.79 | 15.15 +/- 13.51 (not detectable) |
| f1 | cumulonimbus | 25.15 +/- 4.17 | 19.23 +/- 2.54 | 5.92 +/- 6.52 (not detectable) |

### resnet50
- criterion_loss: incomplete_seed_set
- criterion_f1: incomplete_seed_set
## Cross-Architecture Joint Arm
### loss / ccsn overall accuracy
| Arch | Mean +/- sample std |
| --- | --- |
| resnet18 | 60.97 +/- 0.54 |
| resnet34 | 59.90 +/- 2.48 |
| resnet50 | incomplete |
| Pair | Mean diff | Pooled std | Exceeds 2x pooled std |
| --- | --- | --- | --- |
| resnet18_vs_resnet34 | 1.07 | 1.79 | False |
| resnet18_vs_resnet50 | incomplete | incomplete | incomplete |
| resnet34_vs_resnet50 | incomplete | incomplete | incomplete |

### loss / source_balanced overall accuracy
| Arch | Mean +/- sample std |
| --- | --- |
| resnet18 | 74.09 +/- 1.14 |
| resnet34 | 73.74 +/- 2.20 |
| resnet50 | incomplete |
| Pair | Mean diff | Pooled std | Exceeds 2x pooled std |
| --- | --- | --- | --- |
| resnet18_vs_resnet34 | 0.35 | 1.75 | False |
| resnet18_vs_resnet50 | incomplete | incomplete | incomplete |
| resnet34_vs_resnet50 | incomplete | incomplete | incomplete |

### f1 / ccsn overall accuracy
| Arch | Mean +/- sample std |
| --- | --- |
| resnet18 | 62.32 +/- 1.42 |
| resnet34 | 60.97 +/- 0.65 |
| resnet50 | incomplete |
| Pair | Mean diff | Pooled std | Exceeds 2x pooled std |
| --- | --- | --- | --- |
| resnet18_vs_resnet34 | 1.35 | 1.11 | False |
| resnet18_vs_resnet50 | incomplete | incomplete | incomplete |
| resnet34_vs_resnet50 | incomplete | incomplete | incomplete |

### f1 / source_balanced overall accuracy
| Arch | Mean +/- sample std |
| --- | --- |
| resnet18 | 75.54 +/- 0.58 |
| resnet34 | 74.93 +/- 0.21 |
| resnet50 | incomplete |
| Pair | Mean diff | Pooled std | Exceeds 2x pooled std |
| --- | --- | --- | --- |
| resnet18_vs_resnet34 | 0.61 | 0.44 | False |
| resnet18_vs_resnet50 | incomplete | incomplete | incomplete |
| resnet34_vs_resnet50 | incomplete | incomplete | incomplete |

## Reproduction Attempt
- Attempted: `True`; command: `uv run python src/run_harmonized.py --experiment ccsn --model resnet34 --seed 43 --config config/training/tuned_resnet34_ccsn_15ep.toml --output-dir artifacts/audit_2026-09-07_independent/repro_r34_ccsn15_seed43`.
- Completed: `False`; log: `D:\cloud-classification-cnn-mobile\artifacts\audit_2026-09-07_independent\reproduction_r34_ccsn15_seed43.log`.
- Failure tail: `        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ /   File "d:\cloud-classification-cnn-mobile\.venv\Lib\site-packages\torch\utils\data\_utils\collate.py", line 270, in collate_tensor_fn /     storage = elem._typed_storage()._new_shared(numel, device=elem.device) /               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ /   File "d:\cloud-classification-cnn-mobile\.venv\Lib\site-packages\torch\storage.py", line 1198, in _new_shared /     untyped_storage = torch.UntypedStorage._new_shared( /                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ /   File "d:\cloud-classification-cnn-mobile\.venv\Lib\site-packages\torch\storage.py", line 413, in _new_shared /     return cls._new_using_filename_cpu(size) /            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ / RuntimeError: Couldn't open shared file mapping: <torch_7792_1371697923_98>, error code: <1455> / `

## E1/B1 Status
No per-image prediction files or machine-readable confusion matrices were found for ResNet-34/50 (`count=0`). Confusion matrices are PNG figures only (`count=32`). Test tensor preload remains in `src/run_harmonized.py:679-707`; evaluation loaders are built at `src/run_harmonized.py:452-457`.

## Final Judgment
For completed ResNet-34, CCSN in-domain accuracy sits in the claimed band (57.48-59.90% under min-val-loss for CCSN15/CCSN90/Joint-on-CCSN). The min-val-loss joint-vs-CCSN90 CCSN contrast is 2.42 +/- 0.99 pp, which is detectable for overall accuracy by the |mean| > 2x paired-std screen, while the corresponding balanced-accuracy and macro-F1 contrasts are not. Transfer asymmetry is large and partly detectable for ResNet-34 (CCSN15-on-GCD minus GCD15-on-CCSN accuracy 4.06 +/- 2.08 pp). Architecture comparisons cannot be completed for ResNet-50 because the joint ResNet-50 runs are missing; ResNet-18 vs ResNet-34 joint-arm differences are within seed noise where computable, so the old 0.08 pp ResNet-50 margin is not validated by this partial matrix.
