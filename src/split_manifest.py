"""Manifest validator (Phase 4, M3).

Validates a generated split-manifest bundle for schema correctness,
sample-level integrity against a :class:`~dataset_index.DatasetIndex`, and
provenance consistency with the inventory that supposedly produced it. This
module makes no scientific decisions -- it only checks that a manifest is
internally consistent and matches the repository state it claims to
describe. It never creates or modifies a manifest; that is
``split_generator``'s job.
"""
from __future__ import annotations

from typing import Any, Mapping

from dataset_index import DatasetIndex, DatasetLoadingError
from split_protocol import SplitProtocol, compute_inventory_fingerprint

__all__ = [
    "ManifestValidationError",
    "validate_manifest_bundle",
    "validate_protocol_match",
    "select_subset",
]

_SUPPORTED_SUBSETS = ("test", "dev_pool", "cv_fold", "cv_train", "cv_val")


class ManifestValidationError(RuntimeError):
    """Raised when a split manifest fails schema, integrity, or provenance checks."""


def _require_keys(d: Mapping[str, Any], keys: tuple[str, ...], context: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ManifestValidationError(f"{context} is missing required key(s): {missing}")


def _validate_provenance(
    provenance: Mapping[str, Any],
    index: DatasetIndex,
    inventory: Mapping[str, Any],
    *,
    context: str,
) -> None:
    _require_keys(
        provenance,
        (
            "dataset_key", "protocol_name", "protocol_version", "inventory_fingerprint",
            "class_map", "num_samples_total", "cross_class_resolution", "excluded_samples",
        ),
        f"{context} provenance",
    )
    if provenance["dataset_key"] != index.dataset_key:
        raise ManifestValidationError(
            f"{context} provenance dataset_key '{provenance['dataset_key']}' does not "
            f"match the current DatasetIndex '{index.dataset_key}'."
        )
    expected_fingerprint = compute_inventory_fingerprint(inventory)
    if provenance["inventory_fingerprint"] != expected_fingerprint:
        raise ManifestValidationError(
            f"{context} provenance inventory_fingerprint does not match the "
            "supplied inventory -- this manifest was generated from a different "
            "inventory than the one being validated against."
        )
    if provenance["class_map"] != index.class_display_names:
        raise ManifestValidationError(
            f"{context} provenance class_map does not match the current dataset's "
            "approved class_map."
        )


def _validate_samples_subset(samples: list[str], index: DatasetIndex, context: str) -> None:
    known = {p for p, _label in index.samples}
    unknown = sorted(set(samples) - known)
    if unknown:
        raise ManifestValidationError(f"{context} references unknown sample path(s): {unknown[:5]}")
    if len(samples) != len(set(samples)):
        raise ManifestValidationError(f"{context} contains duplicate sample path entries.")


def _duplicate_groups_excluding(inventory: Mapping[str, Any], excluded: set[str]) -> list[list[str]]:
    groups = []
    for group in inventory.get("duplicates", {}).get("groups", []):
        members = [p for p in group if p not in excluded]
        if len(members) > 1:
            groups.append(members)
    return groups


def validate_manifest_bundle(
    bundle: Mapping[str, Any],
    index: DatasetIndex,
    inventory: Mapping[str, Any],
) -> None:
    """Validate a full ``{"test", "dev_pool", "cv"}`` manifest bundle.

    Raises :class:`ManifestValidationError` on any schema, integrity, or
    provenance inconsistency. A bundle either passes whole or fails with an
    explanation -- nothing is repaired.
    """
    _require_keys(bundle, ("test", "dev_pool", "cv"), "manifest bundle")
    test_m, dev_m, cv_bundle = bundle["test"], bundle["dev_pool"], bundle["cv"]

    _require_keys(test_m, ("manifest_type", "samples", "provenance"), "test manifest")
    _require_keys(dev_m, ("manifest_type", "samples", "provenance"), "dev_pool manifest")
    if test_m["manifest_type"] != "permanent_test":
        raise ManifestValidationError("test manifest has an unexpected manifest_type.")
    if dev_m["manifest_type"] != "development_pool":
        raise ManifestValidationError("dev_pool manifest has an unexpected manifest_type.")

    _validate_provenance(test_m["provenance"], index, inventory, context="test manifest")
    _validate_provenance(dev_m["provenance"], index, inventory, context="dev_pool manifest")

    excluded = set(test_m["provenance"]["excluded_samples"])
    _validate_samples_subset(test_m["samples"], index, "test manifest")
    _validate_samples_subset(dev_m["samples"], index, "dev_pool manifest")

    test_set, dev_set = set(test_m["samples"]), set(dev_m["samples"])
    overlap = test_set & dev_set
    if overlap:
        raise ManifestValidationError(f"test and dev_pool overlap on {len(overlap)} sample(s).")

    all_known = {p for p, _label in index.samples}
    expected_base = all_known - excluded
    actual_base = test_set | dev_set
    if actual_base != expected_base:
        missing = sorted(expected_base - actual_base)
        extra = sorted(actual_base - expected_base)
        raise ManifestValidationError(
            "test + dev_pool does not exactly partition the non-excluded samples. "
            f"Missing: {missing[:5]}. Extra: {extra[:5]}."
        )

    duplicate_groups = _duplicate_groups_excluding(inventory, excluded)
    for group in duplicate_groups:
        in_test = [p in test_set for p in group]
        if any(in_test) and not all(in_test):
            raise ManifestValidationError(
                f"Duplicate group {group} is split across test and dev_pool "
                "(duplicate-group atomicity violated)."
            )

    for cv_seed, cv_m in cv_bundle.items():
        context = f"cv manifest (seed={cv_seed})"
        _require_keys(
            cv_m, ("manifest_type", "num_folds", "fold_assignment", "provenance"), context
        )
        if cv_m["manifest_type"] != "cv_folds":
            raise ManifestValidationError(f"{context} has an unexpected manifest_type.")
        _validate_provenance(cv_m["provenance"], index, inventory, context=context)

        fold_assignment = cv_m["fold_assignment"]
        if set(fold_assignment) != dev_set:
            missing = sorted(dev_set - set(fold_assignment))
            extra = sorted(set(fold_assignment) - dev_set)
            raise ManifestValidationError(
                f"{context} fold_assignment does not exactly cover dev_pool. "
                f"Missing: {missing[:5]}. Extra: {extra[:5]}."
            )

        num_folds = cv_m["num_folds"]
        bad_folds = {p: f for p, f in fold_assignment.items() if not (0 <= f < num_folds)}
        if bad_folds:
            raise ManifestValidationError(
                f"{context} has out-of-range fold indices: {list(bad_folds.items())[:5]}."
            )

        for group in duplicate_groups:
            group_in_dev = [p for p in group if p in dev_set]
            if len(group_in_dev) > 1:
                folds = {fold_assignment[p] for p in group_in_dev}
                if len(folds) > 1:
                    raise ManifestValidationError(
                        f"{context}: duplicate group {group_in_dev} spans multiple "
                        "folds (duplicate-group atomicity violated)."
                    )


# --------------------------------------------------------------------------- #
# Consumption-time helpers (Phase 4, M4)
#
# These do not re-implement schema/integrity validation -- validate_manifest_
# bundle above remains the single source of truth for that. They add the two
# checks a consumer needs beyond it: that the bundle was produced under the
# expected protocol, and which sample paths belong to a requested subset.
# --------------------------------------------------------------------------- #
def validate_protocol_match(bundle: Mapping[str, Any], protocol: SplitProtocol) -> None:
    """Verify a manifest bundle was generated under the given protocol.

    Consumption-time protection against silently loading a manifest that was
    generated under a different (e.g. superseded) split protocol version.
    """
    provenance = bundle.get("test", {}).get("provenance", {})
    _require_keys(provenance, ("protocol_name", "protocol_version"), "manifest provenance")
    if provenance["protocol_name"] != protocol.name or provenance["protocol_version"] != protocol.version:
        raise ManifestValidationError(
            f"Manifest was generated under protocol '{provenance['protocol_name']}' "
            f"v{provenance['protocol_version']}, but '{protocol.name}' v{protocol.version} "
            "was expected."
        )


def select_subset(
    bundle: Mapping[str, Any],
    subset: str,
    *,
    cv_seed: int | None = None,
    fold: int | None = None,
) -> list[str]:
    """Return the sorted sample paths belonging to one subset of a manifest bundle.

    Supported subsets: ``"test"``, ``"dev_pool"``, ``"cv_fold"`` (all dev-pool
    samples assigned to ``fold`` under ``cv_seed``), ``"cv_train"`` (all
    dev-pool samples *not* assigned to ``fold`` under ``cv_seed``), and
    ``"cv_val"`` (an alias of ``"cv_fold"`` -- the same samples, named for the
    train/val framing a caller uses them in). This function only reads an
    already-validated bundle; it makes no scientific decisions and never
    mutates the design.
    """
    if subset not in _SUPPORTED_SUBSETS:
        raise DatasetLoadingError(
            f"Unsupported subset '{subset}'. Supported subsets: {_SUPPORTED_SUBSETS}."
        )

    if subset == "test":
        return sorted(bundle["test"]["samples"])
    if subset == "dev_pool":
        return sorted(bundle["dev_pool"]["samples"])

    if cv_seed is None or fold is None:
        raise DatasetLoadingError(
            f"Subset '{subset}' requires both cv_seed and fold to be specified."
        )
    cv_bundle = bundle["cv"]
    if cv_seed not in cv_bundle:
        raise DatasetLoadingError(
            f"cv_seed {cv_seed} is not part of this manifest bundle. Available "
            f"seeds: {sorted(cv_bundle)}."
        )
    cv_manifest = cv_bundle[cv_seed]
    num_folds = cv_manifest["num_folds"]
    if not (0 <= fold < num_folds):
        raise DatasetLoadingError(
            f"fold {fold} is out of range for cv_seed {cv_seed} (num_folds={num_folds})."
        )

    fold_assignment = cv_manifest["fold_assignment"]
    if subset in ("cv_fold", "cv_val"):
        return sorted(p for p, f in fold_assignment.items() if f == fold)
    return sorted(p for p, f in fold_assignment.items() if f != fold)  # cv_train
