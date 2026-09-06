"""Split protocol specification (Phase 4, M3).

A :class:`SplitProtocol` is a pure specification object: the numeric
parameters that define a grouped, class-stratified holdout split for a
dataset with no official split. It carries no splitting logic and touches
no data -- ``split_generator.py`` consumes a ``SplitProtocol`` instance to
know *how* to split, while remaining completely agnostic to *which*
dataset it is splitting.

The shipped production benchmarks use a single fixed train / validation /
test partition at ``canonical_seed`` (see ``scripts/build_canonical_manifests.py``
and ``docs/DECISIONS.md``). The ``num_folds`` / ``variance_seeds`` fields
support the dataset-agnostic generator infrastructure but are not part of
the shipped benchmark protocol -- no cross-validation experiment is run.

This module also provides :func:`compute_inventory_fingerprint`, a small
pure helper shared by the generator and the validator so a manifest's
recorded identity of "the inventory that produced it" can be independently
recomputed and checked, rather than trusted blindly.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = ["SplitProtocol", "GROUPED_STRATIFIED_HOLDOUT_V1", "compute_inventory_fingerprint"]


@dataclass(frozen=True)
class SplitProtocol:
    """A named, versioned split-protocol specification.

    The permanent holdout and development-pool membership are fixed by
    ``canonical_seed`` alone. ``variance_seeds`` / ``num_folds`` are
    generator-infrastructure parameters for optional fold partitions of the
    development pool; they are never regenerated for individual experiments
    and are not exercised by the shipped benchmarks.
    """

    name: str
    version: str
    test_fraction: float
    canonical_seed: int
    num_folds: int
    variance_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not (0.0 < self.test_fraction < 1.0):
            raise ValueError(f"test_fraction must be in (0, 1), got {self.test_fraction}")
        if self.num_folds < 2:
            raise ValueError(f"num_folds must be >= 2, got {self.num_folds}")
        if not self.variance_seeds:
            raise ValueError("variance_seeds must be non-empty")


# The protocol approved for datasets with official_split = false (see
# docs/DECISIONS.md for the researcher-approved record of this methodology).
# Plain language: a class-stratified holdout that keeps exact-duplicate
# image groups together in one partition.
GROUPED_STRATIFIED_HOLDOUT_V1 = SplitProtocol(
    name="grouped_stratified_holdout",
    version="1.0",
    test_fraction=0.2,
    canonical_seed=42,
    num_folds=5,
    variance_seeds=(7, 21, 42, 84, 168),
)


def compute_inventory_fingerprint(inventory: Mapping[str, Any]) -> str:
    """A deterministic content fingerprint of a dataset inventory.

    Ties a generated manifest to the *exact* inventory that produced it,
    independent of key ordering or incidental formatting differences.
    """
    canonical = json.dumps(inventory, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
