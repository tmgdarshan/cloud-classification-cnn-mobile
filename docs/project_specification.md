# Project Specification

This project provides a compact, reproducible benchmark for cross-source
ground-based cloud classification using CCSN and GCD imagery.

## Required Workflow

1. Use the frozen canonical manifests in `metadata/splits/` for all reported
   train, validation, and test partitions.
2. Train and evaluate ResNet-18, ResNet-34, and ResNet-50 through
   `src/run_harmonized.py`.
3. Treat the source-balanced joint CCSN+GCD result as the headline metric.
4. Keep the manuscript focused on dataset curation, taxonomy harmonization,
   source-balanced evaluation, compact ResNet comparison, and observed transfer
   asymmetry.

## Out of Scope

- Mobile deployment, quantization, and edge latency measurement.
- Formal statistical equivalence testing between architectures.
- Causal attribution of transfer failures to individual visual mechanisms.
- A genus-preserving WMO taxonomy across CCSN and GCD.
