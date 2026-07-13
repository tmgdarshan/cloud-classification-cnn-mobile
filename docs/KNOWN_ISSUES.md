# Known Issues

Accepted limitations and deferred work. Each entry is documented rather than
fixed, pending a separate decision.

## KI-001 — GCD train/test duplicate leakage

- **Status:** Open — documented, not yet resolved.
- **Observed (Phase 3 discovery):** 156 of 159 exact-byte duplicate groups in
  GCD span both the official `train` and `test` splits (see
  `metadata/gcd_inventory.json`). Identical images appear in both splits,
  concentrated in `4_clearsky`.
- **Impact:** Training and evaluating on GCD's official split as-is risks data
  leakage, which would inflate reported evaluation metrics.
- **Decision:** Deferred. The dataset and its splits are **not** modified.
  How to handle this — de-duplicate, regenerate splits, or accept as a
  documented limitation — is a separate scientific decision.

## TODO

- Further issues to be recorded as they are discovered or accepted.
