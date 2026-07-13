"""Split generator (Phase 4, M3).

Generates deterministic split manifests for a dataset that has no official
train/test split, driven entirely by three inputs: a dataset's approved
configuration, its authoritative inventory, and a :class:`SplitProtocol`.
No dataset is ever named or special-cased in this module's logic -- it is
data, not code, that makes a run "about" a particular dataset.

This module represents split *policy* only. It never reads image pixels,
never preprocesses, never augments, and never touches ``dataset_loading``.
It reuses ``dataset_index.build_index`` (Layer A) to know which samples and
classes exist, and the dataset's own inventory (``metadata/*_inventory.json``,
already computed and version-controlled) to know about duplicate groups --
it never recomputes hashes or rescans the filesystem for duplicates itself.

Determinism: every random choice goes through a fresh ``random.Random(seed)``
instance -- never the global ``random`` module -- so identical inputs always
produce an identical manifest bundle.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping

from dataset_index import DatasetIndex, build_index
from split_protocol import SplitProtocol, compute_inventory_fingerprint

__all__ = ["SplitGenerationError", "generate_split_manifests", "to_json"]


class SplitGenerationError(RuntimeError):
    """Raised when split generation cannot proceed or halts by policy."""


# --------------------------------------------------------------------------- #
# Duplicate-group / cross-class helpers
# --------------------------------------------------------------------------- #
def _class_of(rel_path: str) -> str:
    return rel_path.split("/", 1)[0]


def _group_id(group: list[str]) -> str:
    return "|".join(sorted(group))


def _is_cross_class(group: list[str]) -> bool:
    return len({_class_of(p) for p in group}) > 1


def _resolve_cross_class_groups(
    all_groups: list[list[str]],
    cross_class_resolution: Mapping[str, str],
    *,
    dataset_key: str,
) -> set[str]:
    """Validate cross-class duplicate groups against the supplied resolution.

    Returns the set of sample paths to exclude from splitting. Raises
    :class:`SplitGenerationError` if any cross-class group is unresolved, if
    the resolution references an unknown group, or if it specifies an
    unsupported action. Cross-class duplicate groups are a label-integrity
    anomaly, not an ordinary duplicate -- this function never interprets
    which label is "correct"; the only supported resolution is exclusion.
    """
    cross_class = {_group_id(g): g for g in all_groups if _is_cross_class(g)}

    unresolved = sorted(set(cross_class) - set(cross_class_resolution))
    if unresolved:
        raise SplitGenerationError(
            f"Dataset '{dataset_key}' has {len(unresolved)} unresolved cross-class "
            f"duplicate group(s); split generation halted. Group(s): {unresolved}. "
            "Cross-class duplicates are a label-integrity anomaly and require an "
            "explicit researcher-approved cross_class_resolution before generation "
            "can proceed."
        )

    unknown_refs = sorted(set(cross_class_resolution) - set(cross_class))
    if unknown_refs:
        raise SplitGenerationError(
            f"cross_class_resolution references group id(s) not found among "
            f"dataset '{dataset_key}''s cross-class duplicate groups: {unknown_refs}."
        )

    invalid_actions = {
        gid: action for gid, action in cross_class_resolution.items() if action != "exclude_group"
    }
    if invalid_actions:
        raise SplitGenerationError(
            f"Unsupported cross_class_resolution action(s): {invalid_actions}. "
            "Only 'exclude_group' is supported -- this generator never encodes a "
            "semantic interpretation of contradictory labels."
        )

    return {p for gid, group in cross_class.items() for p in group}


# --------------------------------------------------------------------------- #
# Atomic units (duplicate-group-aware) and stratified assignment
# --------------------------------------------------------------------------- #
def _atomic_units(base_samples: set[str], within_class_groups: list[list[str]]) -> list[tuple[str, ...]]:
    """Collapse samples into atomic units: a duplicate group is one unit."""
    grouped: set[str] = set()
    units: list[tuple[str, ...]] = []
    for group in within_class_groups:
        members = [p for p in group if p in base_samples]
        if not members:
            continue
        classes = {_class_of(p) for p in members}
        if len(classes) > 1:
            raise SplitGenerationError(
                f"Internal invariant violated: group {members} was treated as "
                "within-class but spans multiple classes."
            )
        units.append(tuple(sorted(members)))
        grouped.update(members)

    singles = base_samples - grouped
    units.extend((p,) for p in sorted(singles))
    return units


def _group_units_by_class(units: list[tuple[str, ...]]) -> dict[str, list[tuple[str, ...]]]:
    by_class: dict[str, list[tuple[str, ...]]] = {}
    for unit in units:
        by_class.setdefault(_class_of(unit[0]), []).append(unit)
    return by_class


def _stratified_units_split(
    units_by_class: Mapping[str, list[tuple[str, ...]]],
    test_fraction: float,
    seed: int,
) -> tuple[dict[str, list[tuple[str, ...]]], dict[str, list[tuple[str, ...]]]]:
    """Split each class's atomic units into test/dev, approximately stratified.

    Duplicate-group atomicity means the target test fraction is a target,
    not a guarantee: a unit is never split across test and dev.
    """
    test_units_by_class: dict[str, list[tuple[str, ...]]] = {}
    dev_units_by_class: dict[str, list[tuple[str, ...]]] = {}
    for cls in sorted(units_by_class):
        units = units_by_class[cls]
        total = sum(len(u) for u in units)
        target_test = round(test_fraction * total)

        rng = random.Random(seed)
        shuffled = list(units)
        rng.shuffle(shuffled)

        test_units: list[tuple[str, ...]] = []
        dev_units: list[tuple[str, ...]] = []
        accumulated = 0
        for unit in shuffled:
            if accumulated < target_test:
                test_units.append(unit)
                accumulated += len(unit)
            else:
                dev_units.append(unit)

        test_units_by_class[cls] = test_units
        dev_units_by_class[cls] = dev_units
    return test_units_by_class, dev_units_by_class


def _stratified_cv_fold_assignment(
    dev_units_by_class: Mapping[str, list[tuple[str, ...]]],
    num_folds: int,
    seed: int,
) -> dict[str, int]:
    """Assign every dev-pool sample a fold index, per-class round-robin."""
    fold_assignment: dict[str, int] = {}
    for cls in sorted(dev_units_by_class):
        units = dev_units_by_class[cls]
        rng = random.Random(seed)
        shuffled = list(units)
        rng.shuffle(shuffled)
        for i, unit in enumerate(shuffled):
            fold = i % num_folds
            for path in unit:
                fold_assignment[path] = fold
    return fold_assignment


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def _build_provenance(
    protocol: SplitProtocol,
    inventory: Mapping[str, Any],
    index: DatasetIndex,
    resolution: Mapping[str, str],
    excluded_samples: set[str],
) -> dict[str, Any]:
    return {
        "dataset_key": index.dataset_key,
        "protocol_name": protocol.name,
        "protocol_version": protocol.version,
        "inventory_fingerprint": compute_inventory_fingerprint(inventory),
        "inventory_schema_version": inventory.get("schema_version"),
        "class_map": dict(index.class_display_names),
        "num_samples_total": len(index.samples),
        "cross_class_resolution": dict(sorted(resolution.items())),
        "excluded_samples": sorted(excluded_samples),
    }


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def generate_split_manifests(
    dataset_cfg: Mapping[str, Any],
    dataset_root: Path | str,
    inventory: Mapping[str, Any],
    protocol: SplitProtocol,
    *,
    cross_class_resolution: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Generate a deterministic split-manifest bundle for one dataset.

    Returns ``{"test": {...}, "dev_pool": {...}, "cv": {seed: {...}, ...}}``,
    all plain JSON-serializable dicts. Raises :class:`SplitGenerationError`
    if the dataset already has an official split, if the inventory does not
    match the dataset config, or if unresolved cross-class duplicate groups
    are found.
    """
    key = dataset_cfg.get("key", dataset_cfg.get("name", "<unknown>"))

    if dataset_cfg.get("official_split", False):
        raise SplitGenerationError(
            f"Dataset '{key}' already has an official split (official_split = "
            "true); this protocol generates splits only for datasets without one."
        )
    if inventory.get("dataset_key") != key:
        raise SplitGenerationError(
            f"Inventory dataset_key '{inventory.get('dataset_key')}' does not "
            f"match dataset config key '{key}'."
        )

    index = build_index(dataset_cfg, dataset_root)
    all_sample_paths = {p for p, _label in index.samples}

    all_groups = [list(g) for g in inventory.get("duplicates", {}).get("groups", [])]
    referenced_paths = {p for g in all_groups for p in g}
    drifted = sorted(referenced_paths - all_sample_paths)
    if drifted:
        raise SplitGenerationError(
            f"Dataset '{key}' inventory references sample path(s) not present in "
            f"the current on-disk dataset (inventory/dataset drift): {drifted[:5]}..."
        )

    resolution = dict(cross_class_resolution or {})
    excluded_samples = _resolve_cross_class_groups(all_groups, resolution, dataset_key=key)
    within_class_groups = [g for g in all_groups if not _is_cross_class(g)]

    base_samples = all_sample_paths - excluded_samples
    units = _atomic_units(base_samples, within_class_groups)
    units_by_class = _group_units_by_class(units)

    test_units_by_class, dev_units_by_class = _stratified_units_split(
        units_by_class, protocol.test_fraction, protocol.canonical_seed
    )
    test_samples = sorted(p for units_ in test_units_by_class.values() for u in units_ for p in u)
    dev_samples = sorted(p for units_ in dev_units_by_class.values() for u in units_ for p in u)

    provenance = _build_provenance(protocol, inventory, index, resolution, excluded_samples)

    test_manifest = {
        "manifest_type": "permanent_test",
        "test_fraction_target": protocol.test_fraction,
        "seed": protocol.canonical_seed,
        "samples": test_samples,
        "provenance": provenance,
    }
    dev_pool_manifest = {
        "manifest_type": "development_pool",
        "seed": protocol.canonical_seed,
        "samples": dev_samples,
        "provenance": provenance,
    }

    cv_manifests: dict[int, dict[str, Any]] = {}
    for cv_seed in protocol.variance_seeds:
        fold_assignment = _stratified_cv_fold_assignment(dev_units_by_class, protocol.num_folds, cv_seed)
        cv_manifests[cv_seed] = {
            "manifest_type": "cv_folds",
            "cv_seed": cv_seed,
            "num_folds": protocol.num_folds,
            "dev_pool_seed": protocol.canonical_seed,
            "fold_assignment": dict(sorted(fold_assignment.items())),
            "provenance": provenance,
        }

    return {"test": test_manifest, "dev_pool": dev_pool_manifest, "cv": cv_manifests}


def to_json(manifest: Mapping[str, Any]) -> str:
    """Serialise a single manifest deterministically (sorted keys, trailing newline)."""
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"
