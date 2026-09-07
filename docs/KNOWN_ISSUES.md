# Known Issues & Methodological Limitations

Accepted limitations and methodological audit records.

---

## KI-001: Exact-Byte Redundancy and Cross-Split Contamination in Benchmark Releases

- **Status**: Remediated for canonical benchmarks via group-aware manifest generation; raw official archives on disk remain uncurated to preserve archival provenance.
- **Observational Context**: In public releases of ground-based sky datasets, repeated exposures can produce identical image files. In the public GCD release, an audit identified 156 exact-byte SHA-256 duplicate clusters spanning the official `train` and `test` directories. In CCSN, 3 conflicting duplicate pairs (6 images) with identical SHA-256 file-byte hashes received contradictory genus labels.
- **Empirical Impact & Risk**: Evaluating models on uncurated splits containing duplicate files across partitions introduces a risk of test-set memorization, potentially inflating reported generalization metrics.
- **Resolution (Decisions D-007, D-008)**: Canonical benchmark evaluation strictly consumes frozen group-aware manifests (`metadata/splits/gcd_6class_canonical.json`, `gcd_5class_canonical.json`, `harmonized_5bin_canonical.json`, and `ccsn_11class_canonical.json`). Exact-byte duplicate clusters are bound under atomic `group_id` identifiers and partitioned via `StratifiedGroupKFold` (Seed 42), ensuring exact duplicates never cross partitions. The 3 conflicting pairs in CCSN were purged.
- **Verification**: Enforced by `tests/unit/test_canonical_manifests.py` and `tests/unit/test_canonical_pipeline_contracts.py`.

---

## Limitations and Future Work

1. **Exact-Byte vs. Scene-Level Duplication**: SHA-256 grouping isolates exact-byte duplicates. It does not identify near-duplicate scenes across different cameras, sessions, or timestamps where pixels have minor sensor noise.
2. **Transfer Confounders**: Performance differences between CCSN $\to$ GCD and GCD $\to$ CCSN reflect multiple combined domain shifts (camera optics, field of view, lighting, image resolution, class label definitions, and sample sizes).
3. **Training budget in joint training**: joint training gives each CCSN image ~3.6 optimizer exposures per epoch versus 1 in a 15-epoch CCSN-only run. The multi-seed analysis (paper, Section 6) matches this budget explicitly with a 90-epoch CCSN-only arm; under that comparison the joint gain on the CCSN component is not consistent across architectures.
4. **Seed variance on CCSN**: the 468-image CCSN test component has an across-seed standard deviation of 1-3 pp, comparable to the differences between training configurations. Single-run CCSN comparisons are unreliable.
5. **Mobile deployment**: quantization and on-device latency benchmarks are future work; no on-device measurements have been performed.
