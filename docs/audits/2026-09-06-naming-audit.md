# Naming audit — 6 September 2026

**Scope:** identifier and terminology consistency across docs, code, config, and
the manuscript. No scientific content, split assignments, seeds, or saved
results are changed by this audit — only names and wording.

This follows the 2026-09-05 simplification review, which found the project's
main weakness is not rigor but that *"README, dashboard, decisions, configs,
code comments, manuscript, and legacy scripts repeat the taxonomy, split
protocol, augmentation rationale, and 'official' workflow with slightly
different wording."* This audit enumerates that drift and fixes a single
vocabulary.

---

## 1. Split-protocol name — six strings for one protocol

The production benchmarks use **one** partitioning method: a class-stratified,
exact-duplicate-group-aware split into a fixed train / validation / test
partition at seed 42 (`StratifiedGroupKFold` used to derive it; **no
cross-validation experiment is run**). It currently appears under six names:

| String | Locations |
|---|---|
| `stratified_group_holdout_3way_v1.0` | `metadata/splits/*_canonical.json`, `scripts/build_canonical_manifests.py`, `docs/architecture/split_protocol.md` |
| `harmonized_3way_train_val_test` | `artifacts/experiment_registry.json` (production records), `src/run_harmonized.py:691` |
| `true_stratified_group_holdout_5fold_cv` | `artifacts/experiment_registry.json` (stale SG-HCV record), `artifacts/sghcv_results/`, `src/legacy/train_sghcv.py`, `src/experiment_registry.py` docstring |
| `stratified_holdout_cv` / `_v1.0` | `src/split_protocol.py` (`STRATIFIED_HOLDOUT_CV_V1`), `docs/DECISIONS.md` "Phase 4 (M3)", `metadata/splits/ccsn/split_manifest_bundle.json`, `scripts/legacy/generate_splits.py`, unit tests |
| `stratified_5fold_cv_v1.0` | `metadata/splits/gcd_6class/split_manifest_bundle.json`, `scripts/legacy/generate_splits.py` |
| `3way_train_val_test_leak_free` | `tests/unit/test_experiment_registry.py` fixture |

Additionally, `docs/DECISIONS.md` "Phase 4 (M3)" and `docs/architecture/split_protocol.md`
describe a **5-fold CV with variance seeds `{7,21,42,84,168}`** that was never
run against real data — the canonical manifests are single-seed, fold-0 =
validation. `src/split_protocol.py` still hard-codes that unused CV design.

**Recommendation:**
- Canonical name: **`grouped_stratified_holdout`** (version `1.0`). Each word
  earns its place: *grouped* = exact-duplicate clusters stay atomic; *stratified*
  = class proportions preserved; *holdout* = one fixed test partition, never
  regenerated. Drop `cv` / `3way` / `5fold` — no CV is performed.
- Use it verbatim in: the manifest `protocol` field (builder), the registry
  `protocol` field (`run_harmonized.py`), `STRATIFIED_HOLDOUT_CV_V1` →
  `GROUPED_STRATIFIED_HOLDOUT_V1`, `DECISIONS.md`, `docs/architecture/split_protocol.md`, tests.
- `docs/DECISIONS.md` "Phase 4 (M3)": mark the 5-fold-CV / multi-seed variant
  **superseded** — record that the shipped protocol is the single-seed grouped
  holdout above. Trim the unused `num_folds` / `variance_seeds` design from
  `src/split_protocol.py` (or document it as reserved, not approved).
- SG-HCV legacy code was removed; no SG-HCV strings remain in tracked source.

> Note: a data-split protocol is an ML construct, so an ML-accurate name is
> correct here — the fix is *one* clear name, not an atmospheric-science name.

---

## 2. "merged" vs "joint"

Decision **D-012** already directs "joint" and archives "merged". Not executed —
"merged" persists in:

- result keys: `merged_joint_model` (primary; `joint_model` exists only as an alias)
- registry ids: `harmonized_5bin_resnet{18,34,50}_merged`
- saved labels: `"Merged Joint Holdout"`, `"Merged Model on CCSN"`, `"Merged Model on GCD"`
- checkpoint files: `artifacts/harmonized_results/harmonized_merged_resnet*.pth`
- figure files: `confusion_matrix_merged_*`, `report/figures/cm_merged_on_ccsn_resnet18.png` (**cited by the manuscript**)
- config: `config/datasets/merged_v1.toml` (already `superseded`)

**Recommendation:** canonical term **`joint`** (the joint CCSN+GCD model /
joint training). In `run_harmonized.py` make `joint_model` the primary result
key and `merged_joint_model` the deprecated alias; rename the manuscript figure
to `cm_joint_on_ccsn_resnet18.png` and update `\includegraphics`. Leave the
already-written gitignored `artifacts/` JSON/checkpoints as historical outputs
(regeneration will emit the new names); `merged_v1.toml` keeps its name as a
frozen superseded identifier but its prose says "joint".

---

## 3. "zero-shot" vs "cross-source transfer"

`run_harmonized.py` already writes `cross_source_ccsn_to_gcd` as an alias, but
`zeroshot_ccsn_to_gcd` is the primary key and the term used in every saved
label and figure name (`Zero-Shot CCSN to GCD`, `zero_shot_*`). "Zero-shot" is
also inaccurate — this is cross-domain evaluation of a fully supervised
classifier, not zero-shot learning.

**Recommendation:** canonical term **`cross-source transfer`** /
`cross_source_{ccsn_to_gcd,gcd_to_ccsn}`. Make it the primary result key;
`zeroshot_*` becomes the deprecated alias. Prose and manuscript already mostly
say "cross-source transfer" — finish it.

---

## 4. Study-level term: "cross-sensor" vs "cross-source" vs "cross-dataset"

| Term | Where |
|---|---|
| "Cross-Sensor" | `README.md` H1, manuscript `\title` |
| "Cross-Source" | `docs/OFFICIAL_PROTOCOL.md` (authoritative), dashboard, most prose |
| "cross-dataset" | `docs/DECISIONS.md` D-007, `config/datasets/harmonized_5bin.toml` |

**Recommendation:** **`cross-source`** for the study and the transfer direction
(matches the authoritative protocol doc); **`cross-dataset compatibility
taxonomy`** stays as the taxonomy's proper name (D-007, and the 2026-09-05
review endorses that exact phrase). Update the README H1 and manuscript title
from "Cross-Sensor" → "Cross-Source".

---

## 5. "5-bin" vs "5-class" vs "five-class"

`harmonized_5bin`, "5-Bin", "5-bin compatibility space", "5-class harmonized
benchmark", "Five-Class Compatibility Taxonomy" all coexist.

**Recommendation:** prose is always **"five-class compatibility taxonomy"** /
"harmonized five-class" (spell the number, drop "bin"). Keep the *identifiers*
`harmonized_5bin` / `harmonized_5bin_canonical.json` / `harmonized_5bin.toml`
**as-is** — they are frozen keys referenced by the manifest `dataset_key`
field; renaming them touches machine truth for no scientific gain. Note the
identifier/prose distinction once in `PROJECT_GUIDE.md`.

---

## 6. "atmospheric" / "meteorological" decoration on generic ML machinery

`src/atmospheric_evaluation.py` computes standard classification metrics —
accuracy, balanced accuracy, macro-F1, confusion matrix, image-level bootstrap
CI. Nothing in the computation is atmospheric. The label appears as:

- `src/atmospheric_evaluation.py`, `generate_atmospheric_report()`
- "atmospheric classification reports", "Per-Class Meteorological Table" (comments)
- dashboard "Meteorological Evaluation", README "Meteorological metrics"
- manuscript §"Meteorological Error Analysis", §"Atmospheric Orientation Invariant", "atmospheric physics-preserving augmentations"
- `src/run_harmonized.py`, `src/tune_resnet_family.py` docstrings: "Physical atmospheric augmentations"

The 2026-09-05 review already asked for this: *"Replace heavy phrases like
'atmospheric physics-preserving augmentations' with 'physically conservative
augmentations.'"*

**Recommendation:**
- `src/atmospheric_evaluation.py` → **`src/classification_metrics.py`**;
  `generate_atmospheric_report` → `generate_metrics_report`. Update the 3
  importers (`run_harmonized.py`, `tune_resnet_family.py` if used,
  `src/legacy/train_sghcv.py`), the saved-file prefix `atmospheric_metrics_` →
  `classification_metrics_`, docs, and the README tree.
- Augmentation wording everywhere → **"physically conservative augmentations"**.
- Keep genuine domain language where it *is* about clouds: the vertical-flip
  rationale, the domain-shift framing, the error analysis of confusions —
  reword the manuscript section titles to "Error Analysis and CCSN
  Generalization Gain" and "Vertical-Orientation Invariant" (drop
  "Meteorological"/"Atmospheric" as ornament, keep the physics in the body).

---

## 7. Opaque acronym: SG-HCV / `sghcv`

Expanded nowhere; the legacy scripts that used it (`src/legacy/`, `scripts/legacy/`)
were removed in the 2026-09-06 cleanup. Only `artifacts/sghcv_results/` and one
stale `experiment_registry.json` record remain (both gitignored; the record is
removed in the registry cleanup).

---

## 8. Application boundary

| Layer | Action |
|---|---|
| `docs/*`, `README.md`, `report/*.tex` (+ recompile), tracked `report/figures/` | Full standardization to the vocabulary above |
| `src/*.py`, `scripts/*.py` (non-legacy), `config/*` | Rename modules/functions/keys; keep old result keys as deprecated aliases; update tests |
| `metadata/splits/*_canonical.json` `protocol` field | Rewrite the string only (mechanical; no fold/seed/sample change) via a re-run of the builder or a targeted string edit, then re-verify hashes |
| `artifacts/**` (gitignored historical outputs) | Left as-is; regeneration emits the new names |
| `src/legacy/`, `scripts/legacy/` | Removed in the 2026-09-06 cleanup (D-012) |

## 9. Canonical vocabulary (summary)

| Concept | Canonical | Deprecated / variant |
|---|---|---|
| Split protocol | `grouped_stratified_holdout` v1.0 | `stratified_holdout_cv`, `stratified_group_holdout_3way`, `harmonized_3way_train_val_test`, `stratified_5fold_cv`, `3way_train_val_test_leak_free` |
| Combined model / training | joint model, joint training | merged, merged_joint, merged_model |
| Off-source evaluation | cross-source transfer | zero-shot, zeroshot |
| Study / transfer framing | cross-source | cross-sensor, cross-dataset (except the taxonomy name) |
| Shared label space | five-class compatibility taxonomy (prose); `harmonized_5bin` (identifier) | 5-bin, 5-class, five-bin |
| Metrics module | `src/evaluation.py` / `generate_evaluation_report` | `atmospheric_evaluation.py` / `generate_atmospheric_report` |
| Augmentation policy | physically conservative augmentations | atmospheric physics-preserving augmentations |

---

## Outcome (applied 2026-09-06)

- All rows of §9 applied across tracked `docs/`, `src/`, `scripts/`, `config/`,
  `tests/`, `report/` (+ manuscript recompiled, 12 pp, 0 errors). 127 tests pass.
- Metrics module named **`src/evaluation.py`** (not `classification_metrics.py`);
  function `generate_evaluation_report`; saved-file prefix `evaluation_metrics_`.
- Manifest `protocol` / `taxonomy_name` strings edited in place; `sha256`
  (hash of `samples` only) verified **unchanged** on all four manifests.
- `zeroshot_*` / `merged_joint_model` remain as **labeled deprecated aliases**
  in `run_harmonized.py`; `cross_source_*` / `joint_model` are primary.
- Gitignored `artifacts/**` historical outputs left as-is (not mass-renamed);
  consuming scripts accept both key spellings.
- `src/legacy/` + `scripts/legacy/` removed entirely (D-012); the numbered
  `--experiment 1/2/3` and `merged` CLI aliases dropped.
