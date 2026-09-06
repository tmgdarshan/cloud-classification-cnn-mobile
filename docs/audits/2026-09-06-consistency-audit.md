# Repository consistency audit — 6 September 2026

Full walkthrough of tracked files, the uncommitted Phase 4 working tree, the
canonical manifests, saved result artifacts, the manuscript, and the test suite.
Companion to the 2026-09-05 project audit; this one focuses on
**documentation ↔ code ↔ data consistency** and records what was fixed in the
2026-09-06 documentation pass.

## Verified sound (no change needed)

- **Leakage:** all four canonical manifests are clean — zero `group_id` straddles
  train/val/test, zero straddles CV folds. Duplicate-cluster atomicity holds.
- **Count reconciliation is exact:** CCSN 2,543 raw − 6 cross-class dupes = 2,537;
  − 200 `Ct` = 2,337. GCD 19,000 − 955 `7_mixed` = 18,045; − 3,739 `4_clearsky`
  = 14,306. Harmonized = 2,337 + 14,306 = 16,643. Every split sums correctly.
- **Harmonized mapping:** 0 forbidden source classes leaked in; 0 mapping
  mismatches; harmonized-GCD per-class == `gcd_5class` per-class.
- **Manuscript tables:** every point estimate and CI cell checked in Table 4
  (appendix master) and all 30 cells of Table 3 (tuning) match the saved
  `artifacts/**/*.json` exactly.
- **Bootstrap:** B = 1000, percentile method, `seed = 42`, image-level — matches
  the documented method.
- **Code ↔ protocol:** runner head, 224 px, cosine→1e-6, `WeightedRandomSampler`,
  minimum-val-loss checkpoint, defaults 15 / 64 — all match `OFFICIAL_PROTOCOL.md`.

## Fixed in the 2026-09-06 pass

| Finding | Resolution |
|---|---|
| AI-working-cluster governance layer (`CLAUDE.md`, `PROJECT_STANDARD.md`, `phase_handoffs/`, `PROJECT_CONTEXT.tmp.md`) | Removed; consolidated into `docs/PROJECT_GUIDE.md` |
| Six different split-protocol name strings | One: `grouped_stratified_holdout` v1.0 (`docs/audits/2026-09-06-naming-audit.md`) |
| "merged" / "zero-shot" / "cross-sensor" / "5-bin" / "atmospheric evaluation" drift | Standardized vocabulary; `atmospheric_evaluation.py` → `evaluation.py` |
| `DECISIONS.md` — "newest first" broken (two "Phase 4" blocks, substantive one last) | Reordered; Phase 4 (D-007–D-012) on top, Phase 4 (M3) below it |
| `DECISIONS.md` D-009 — "governed by validation loss **and** macro-F1" | Corrected: runner = min val-loss only; macro-F1 is the tuner's tiebreak |
| `DECISIONS.md` M3 — 5-fold CV / variance seeds `{7,21,42,84,168}` presented as *the* protocol (never run) | Marked reserved generator infrastructure; not the shipped protocol |
| `DECISIONS.md` M3 — "halts until researcher supplies CCSN cross-class resolution" vs. the 6 images already excluded | Added: the exclusion **is** the approved resolution (D-007 / Phase 3.5) |
| `README.md` — "DECISIONS.md (D-001 to D-012)" | Only D-007–D-012 exist; corrected |
| `CHANGELOG.md` — no Phase 3.5-freeze / Phase 4 entries | Added |
| `phase_04.md` handoff referenced non-existent files (`src/train_sghcv.py`, `src/tune_resnet18.py`, `scripts/run_experiments.py`), "109 tests" | Handoff removed; history folded into `PROJECT_GUIDE.md` / `CHANGELOG.md` |
| `config/datasets/README.md`, `config/training/README.md`, `config/experiments/README.md`, `metadata/README.md` stale | Rewritten against the current tree; `metadata/splits/README.md` added |
| `docs/architecture/*.md` linked deleted handoffs; "(once written)" | Re-pointed; noted the canonical manifests come from `build_canonical_manifests.py` |
| `.gitignore` — `PROJECT_CONTEXT.tmp.md` and `tmp/` not actually ignored | Fixed |
| `scripts/plot_transfer_asymmetry.py` — hardcoded `D:\cloud-classification-cnn-mobile` | Derived from `__file__` |
| `dashboard` — Phase 4 "Complete" while the whole working tree is uncommitted | "Implementation complete; pending commit / provenance follow-ups" |

## Open — not addressed here

1. **The entire Phase 4 working tree is uncommitted** (runners, manifests,
   configs, report, tests). GitHub does not reflect the documented state. A
   review-and-commit pass is the next step.
2. **`config/experiments/harmonized_5bin_resnet18.toml`** sets `epochs = 10`,
   `batch_size = 128` — contradicts the 15 / 64 protocol; appears unused by the
   runner. `stage1_*` experiment configs are `approved` but reference the draft
   `baseline` training protocol.
3. **`artifacts/experiment_registry.json`** holds a stale 1-epoch SG-HCV smoke
   record (empty manifest provenance) alongside the real ones.
4. **Run/provenance** (carried from 2026-09-05): saved artifacts lack full
   runtime/git/config lineage and per-image predictions; hash consistency does
   not prove a checkpoint reproduces a saved prediction table.
5. `docs/project_specification.md` is still a stub.
6. `notebooks/data_expolration.ipynb` — misspelt filename, one empty cell.
