# scripts/

Operational and utility entry-point scripts.

- `build_canonical_manifests.py` - rebuild the frozen manifest JSON files from
  the approved inventories and taxonomy decisions.
- `discover_dataset.py` - read-only dataset discovery and inventory generation.
- `plot_comparative_convergence.py` - regenerate the compact ResNet-family
  convergence figure used in the manuscript.
- `regenerate_confusion_matrices.py` - re-evaluate the ResNet-18 minimum-val-loss
  checkpoints to rebuild the confusion-matrix figures and save raw predictions.
- `show_config.py` - print resolved config profiles for inspection.

The active workflow is intentionally small: build or verify manifests, tune or
run the harmonized benchmark, then regenerate only the figures referenced by the
paper.
