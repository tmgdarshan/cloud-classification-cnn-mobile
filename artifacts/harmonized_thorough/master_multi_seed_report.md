# PUBLICATION-GRADE MASTER MULTI-SEED REBUILD REPORT

Benchmark: Harmonized 5-Class Cloud Classification across ResNet-18, ResNet-34, ResNet-50
Seeds: [42, 43, 44] | Checkpoint Selection: Min Val Loss (Canonical) vs Max Val Macro-F1 (Diagnostic)

## 1. Master Multi-Seed Results Table: Min Validation Loss (Canonical)

| Arch | Condition / Arm | Evaluated On | 3-Seed Top-1 Acc | Balanced Acc | Macro-F1 | Approved Baseline (Seed 42) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **RESNET18** | 1. CCSN-15ep (Control) | CCSN Holdout | 58.90% ± 0.96% | 54.93% ± 0.47% | 55.28% ± 1.02% | 56.84% |
| | 2. CCSN-90ep (Budget-Matched) | CCSN Holdout | 61.18% ± 0.54% | 57.00% ± 0.25% | 57.51% ± 0.43% | — |
| | 3. GCD-15ep | GCD Holdout | 89.88% ± 0.20% | 92.10% ± 0.29% | 91.74% ± 0.13% | 89.34% |
| | 4. Joint-15ep (Approved) | CCSN Holdout | 60.97% ± 0.54% | 57.73% ± 0.31% | 58.02% ± 0.03% | —% |
| | 4. Joint-15ep (Approved) | GCD Holdout | 87.22% ± 1.77% | 88.35% ± 1.87% | 88.85% ± 1.81% | —% |
| | 4. Joint-15ep (Approved) | Combined Holdout | 83.53% ± 1.59% | — | — | —% |
| | 4. Joint-15ep (Approved) | Source-Balanced Avg | 74.09% ± 1.14% | — | — | —% |
| **RESNET34** | 1. CCSN-15ep (Control) | CCSN Holdout | 58.05% ± 1.78% | 55.74% ± 0.65% | 55.35% ± 0.88% | 57.91% |
| | 2. CCSN-90ep (Budget-Matched) | CCSN Holdout | 57.48% ± 1.62% | 55.12% ± 0.70% | 54.79% ± 0.50% | — |
| | 3. GCD-15ep | GCD Holdout | 89.33% ± 0.47% | 91.08% ± 0.23% | 90.97% ± 0.29% | 89.06% |
| | 4. Joint-15ep (Approved) | CCSN Holdout | 59.90% ± 2.48% | 58.12% ± 1.29% | 57.88% ± 2.14% | —% |
| | 4. Joint-15ep (Approved) | GCD Holdout | 87.59% ± 2.11% | 88.70% ± 3.12% | 88.89% ± 2.94% | —% |
| | 4. Joint-15ep (Approved) | Combined Holdout | 83.69% ± 2.12% | — | — | —% |
| | 4. Joint-15ep (Approved) | Source-Balanced Avg | 73.74% ± 2.20% | — | — | —% |
| **RESNET50** | 1. CCSN-15ep (Control) | CCSN Holdout | 59.12% ± 1.37% | 57.70% ± 1.77% | 57.24% ± 1.43% | 59.19% |
| | 2. CCSN-90ep (Budget-Matched) | CCSN Holdout | 58.55% ± 3.03% | 54.81% ± 2.70% | 54.77% ± 2.48% | — |
| | 3. GCD-15ep | GCD Holdout | 89.49% ± 0.40% | 91.45% ± 0.38% | 91.37% ± 0.13% | 89.45% |
| | 4. Joint-15ep (Approved) | CCSN Holdout | 61.54% ± 2.94% | 59.03% ± 1.01% | 58.89% ± 1.72% | —% |
| | 4. Joint-15ep (Approved) | GCD Holdout | 86.90% ± 1.04% | 88.15% ± 1.37% | 88.72% ± 1.11% | —% |
| | 4. Joint-15ep (Approved) | Combined Holdout | 83.33% ± 1.29% | — | — | —% |
| | 4. Joint-15ep (Approved) | Source-Balanced Avg | 74.22% ± 1.97% | — | — | —% |

## 1. Master Multi-Seed Results Table: Max Validation Macro-F1 (Diagnostic)

| Arch | Condition / Arm | Evaluated On | 3-Seed Top-1 Acc | Balanced Acc | Macro-F1 | Approved Baseline (Seed 42) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **RESNET18** | 1. CCSN-15ep (Control) | CCSN Holdout | 59.62% ± 0.37% | 56.24% ± 0.88% | 56.50% ± 0.57% | 56.84% |
| | 2. CCSN-90ep (Budget-Matched) | CCSN Holdout | 60.82% ± 0.86% | 55.84% ± 1.65% | 56.21% ± 1.47% | — |
| | 3. GCD-15ep | GCD Holdout | 89.68% ± 0.41% | 91.68% ± 0.66% | 91.53% ± 0.31% | 89.34% |
| | 4. Joint-15ep (Approved) | CCSN Holdout | 62.32% ± 1.42% | 59.68% ± 1.73% | 59.81% ± 1.36% | —% |
| | 4. Joint-15ep (Approved) | GCD Holdout | 88.76% ± 0.30% | 90.29% ± 0.25% | 90.47% ± 0.23% | —% |
| | 4. Joint-15ep (Approved) | Combined Holdout | 85.05% ± 0.11% | — | — | —% |
| | 4. Joint-15ep (Approved) | Source-Balanced Avg | 75.54% ± 0.58% | — | — | —% |
| **RESNET34** | 1. CCSN-15ep (Control) | CCSN Holdout | 57.91% ± 1.07% | 55.86% ± 0.72% | 55.53% ± 1.10% | 57.91% |
| | 2. CCSN-90ep (Budget-Matched) | CCSN Holdout | 58.26% ± 1.71% | 53.56% ± 0.90% | 53.46% ± 0.62% | — |
| | 3. GCD-15ep | GCD Holdout | 89.19% ± 0.53% | 91.16% ± 0.75% | 90.95% ± 0.69% | 89.06% |
| | 4. Joint-15ep (Approved) | CCSN Holdout | 60.97% ± 0.65% | 58.31% ± 0.91% | 58.59% ± 0.67% | —% |
| | 4. Joint-15ep (Approved) | GCD Holdout | 88.89% ± 0.24% | 90.49% ± 0.34% | 90.67% ± 0.26% | —% |
| | 4. Joint-15ep (Approved) | Combined Holdout | 84.96% ± 0.13% | — | — | —% |
| | 4. Joint-15ep (Approved) | Source-Balanced Avg | 74.93% ± 0.22% | — | — | —% |
| **RESNET50** | 1. CCSN-15ep (Control) | CCSN Holdout | 61.04% ± 1.01% | 59.30% ± 1.15% | 59.10% ± 1.31% | 59.19% |
| | 2. CCSN-90ep (Budget-Matched) | CCSN Holdout | 60.47% ± 2.89% | 57.88% ± 1.30% | 57.73% ± 2.06% | — |
| | 3. GCD-15ep | GCD Holdout | 89.73% ± 0.68% | 91.71% ± 0.67% | 91.61% ± 0.48% | 89.45% |
| | 4. Joint-15ep (Approved) | CCSN Holdout | 61.90% ± 1.63% | 57.43% ± 1.03% | 57.59% ± 1.06% | —% |
| | 4. Joint-15ep (Approved) | GCD Holdout | 89.66% ± 0.28% | 91.35% ± 0.26% | 91.42% ± 0.32% | —% |
| | 4. Joint-15ep (Approved) | Combined Holdout | 85.76% ± 0.47% | — | — | —% |
| | 4. Joint-15ep (Approved) | Source-Balanced Avg | 75.78% ± 0.96% | — | — | —% |

## 2. Key Contrasts & Hypothesis Tests (Criterion: Min Val Loss)

| Arch | Contrast / Hypothesis | Paired Diffs [s42, s43, s44] (pp) | Mean Diff ± Std (pp) | Verdict (|Mean| > Std) |
| :--- | :--- | :---: | :---: | :---: |
| **RESNET18** | **A. Budget Effect** (CCSN90 - CCSN15) | [0.85, 3.2, 2.78] | +2.28 ± 1.25 | **DETECTABLE** |
| | **B. Joint Effect vs Budget-Matched** (Joint - CCSN90) | [0.86, -0.21, -1.28] | -0.21 ± 1.07 | NOT DETECTABLE |
| | **C. Old Joint Claim** (Joint - CCSN15) | [1.71, 2.99, 1.5] | +2.07 ± 0.81 | **DETECTABLE** |
| **RESNET34** | **A. Budget Effect** (CCSN90 - CCSN15) | [-0.22, -0.64, -0.85] | -0.57 ± 0.32 | **DETECTABLE** |
| | **B. Joint Effect vs Budget-Matched** (Joint - CCSN90) | [3.0, 1.28, 2.99] | +2.42 ± 0.99 | **DETECTABLE** |
| | **C. Old Joint Claim** (Joint - CCSN15) | [2.78, 0.64, 2.14] | +1.85 ± 1.10 | **DETECTABLE** |
| **RESNET50** | **A. Budget Effect** (CCSN90 - CCSN15) | [-1.92, -3.2, 3.42] | -0.57 ± 3.51 | NOT DETECTABLE |
| | **B. Joint Effect vs Budget-Matched** (Joint - CCSN90) | [2.77, 7.26, -1.07] | +2.99 ± 4.17 | NOT DETECTABLE |
| | **C. Old Joint Claim** (Joint - CCSN15) | [0.85, 4.06, 2.35] | +2.42 ± 1.61 | **DETECTABLE** |

## 3. Best-Epoch Selection Stability (Criterion: Min Val Loss)

| Arch | Condition / Arm | Seed 42 Epoch | Seed 43 Epoch | Seed 44 Epoch | Mean ± Std Epoch | Max Budget |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **RESNET18** | ccsn15 | 7 | 12 | 4 | 7.7 ± 4.0 | 15 |
| **RESNET18** | ccsn90 | 36 | 42 | 51 | 43.0 ± 7.5 | 90 |
| **RESNET18** | gcd15 | 15 | 14 | 13 | 14.0 ± 1.0 | 15 |
| **RESNET18** | joint | 15 | 6 | 4 | 8.3 ± 5.9 | 15 |
| **RESNET34** | ccsn15 | 7 | 3 | 4 | 4.7 ± 2.1 | 15 |
| **RESNET34** | ccsn90 | 7 | 3 | 4 | 4.7 ± 2.1 | 90 |
| **RESNET34** | gcd15 | 15 | 9 | 12 | 12.0 ± 3.0 | 15 |
| **RESNET34** | joint | 15 | 3 | 11 | 9.7 ± 6.1 | 15 |
| **RESNET50** | ccsn15 | 3 | 5 | 4 | 4.0 ± 1.0 | 15 |
| **RESNET50** | ccsn90 | 2 | 5 | 4 | 3.7 ± 1.5 | 90 |
| **RESNET50** | gcd15 | 10 | 9 | 9 | 9.3 ± 0.6 | 15 |
| **RESNET50** | joint | 2 | 7 | 4 | 4.3 ± 2.5 | 15 |

## 4. Appendix: Per-Seed Raw Test Metrics (Criterion: Min Val Loss)

| Arch | Arm | Seed | Evaluated On | Top-1 Acc (%) | Balanced Acc (%) | Macro-F1 (%) | Selected Epoch |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| RESNET18 | ccsn15 | 42 | CCSN Holdout | 59.83 | 55.46 | 56.46 | 7 |
| RESNET18 | ccsn15 | 43 | CCSN Holdout | 57.91 | 54.77 | 54.69 | 12 |
| RESNET18 | ccsn15 | 44 | CCSN Holdout | 58.97 | 54.55 | 54.69 | 4 |
| RESNET18 | ccsn90 | 42 | CCSN Holdout | 60.68 | 56.77 | 57.01 | 36 |
| RESNET18 | ccsn90 | 43 | CCSN Holdout | 61.11 | 57.27 | 57.78 | 42 |
| RESNET18 | ccsn90 | 44 | CCSN Holdout | 61.75 | 56.97 | 57.74 | 51 |
| RESNET18 | gcd15 | 42 | GCD Holdout | 89.69 | 91.77 | 91.59 | 15 |
| RESNET18 | gcd15 | 43 | GCD Holdout | 90.08 | 92.28 | 91.80 | 14 |
| RESNET18 | gcd15 | 44 | GCD Holdout | 89.87 | 92.25 | 91.83 | 13 |
| RESNET18 | joint | 42 | CCSN Holdout | 61.54 | 57.59 | 58.01 | 15 |
| RESNET18 | joint | 42 | GCD Holdout | 88.75 | 90.06 | 90.45 | 15 |
| RESNET18 | joint | 43 | CCSN Holdout | 60.90 | 57.52 | 58.00 | 6 |
| RESNET18 | joint | 43 | GCD Holdout | 87.63 | 88.64 | 89.21 | 6 |
| RESNET18 | joint | 44 | CCSN Holdout | 60.47 | 58.09 | 58.06 | 4 |
| RESNET18 | joint | 44 | GCD Holdout | 85.29 | 86.35 | 86.88 | 4 |
| RESNET34 | ccsn15 | 42 | CCSN Holdout | 57.48 | 56.48 | 55.49 | 7 |
| RESNET34 | ccsn15 | 43 | CCSN Holdout | 56.62 | 55.26 | 54.40 | 3 |
| RESNET34 | ccsn15 | 44 | CCSN Holdout | 60.04 | 55.47 | 56.15 | 4 |
| RESNET34 | ccsn90 | 42 | CCSN Holdout | 57.26 | 55.76 | 55.01 | 7 |
| RESNET34 | ccsn90 | 43 | CCSN Holdout | 55.98 | 55.23 | 54.22 | 3 |
| RESNET34 | ccsn90 | 44 | CCSN Holdout | 59.19 | 54.38 | 55.14 | 4 |
| RESNET34 | gcd15 | 42 | GCD Holdout | 89.13 | 90.97 | 91.08 | 15 |
| RESNET34 | gcd15 | 43 | GCD Holdout | 89.87 | 91.35 | 91.20 | 9 |
| RESNET34 | gcd15 | 44 | GCD Holdout | 88.99 | 90.93 | 90.64 | 12 |
| RESNET34 | joint | 42 | CCSN Holdout | 60.26 | 58.34 | 58.40 | 15 |
| RESNET34 | joint | 42 | GCD Holdout | 89.17 | 90.86 | 90.97 | 15 |
| RESNET34 | joint | 43 | CCSN Holdout | 57.26 | 56.73 | 55.53 | 3 |
| RESNET34 | joint | 43 | GCD Holdout | 85.19 | 85.13 | 85.53 | 3 |
| RESNET34 | joint | 44 | CCSN Holdout | 62.18 | 59.29 | 59.71 | 11 |
| RESNET34 | joint | 44 | GCD Holdout | 88.40 | 90.12 | 90.18 | 11 |
| RESNET50 | ccsn15 | 42 | CCSN Holdout | 58.12 | 56.21 | 55.82 | 3 |
| RESNET50 | ccsn15 | 43 | CCSN Holdout | 60.68 | 59.66 | 58.67 | 5 |
| RESNET50 | ccsn15 | 44 | CCSN Holdout | 58.55 | 57.24 | 57.24 | 4 |
| RESNET50 | ccsn90 | 42 | CCSN Holdout | 56.20 | 51.69 | 52.00 | 2 |
| RESNET50 | ccsn90 | 43 | CCSN Holdout | 57.48 | 56.45 | 55.54 | 5 |
| RESNET50 | ccsn90 | 44 | CCSN Holdout | 61.97 | 56.29 | 56.77 | 4 |
| RESNET50 | gcd15 | 42 | GCD Holdout | 89.76 | 91.76 | 91.47 | 10 |
| RESNET50 | gcd15 | 43 | GCD Holdout | 89.03 | 91.02 | 91.22 | 9 |
| RESNET50 | gcd15 | 44 | GCD Holdout | 89.69 | 91.57 | 91.41 | 9 |
| RESNET50 | joint | 42 | CCSN Holdout | 58.97 | 58.99 | 57.64 | 2 |
| RESNET50 | joint | 42 | GCD Holdout | 86.20 | 87.49 | 87.99 | 2 |
| RESNET50 | joint | 43 | CCSN Holdout | 64.74 | 60.06 | 60.85 | 7 |
| RESNET50 | joint | 43 | GCD Holdout | 88.09 | 89.73 | 90.00 | 7 |
| RESNET50 | joint | 44 | CCSN Holdout | 60.90 | 58.04 | 58.17 | 4 |
| RESNET50 | joint | 44 | GCD Holdout | 86.41 | 87.23 | 88.17 | 4 |