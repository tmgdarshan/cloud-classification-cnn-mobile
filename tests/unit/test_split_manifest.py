"""Unit tests for the split manifest validator (Phase 4, M3)."""
import copy
from pathlib import Path

import pytest

import split_generator as sg
import split_manifest as sm
import split_protocol as sp
from dataset_index import build_index


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")


def _protocol() -> sp.SplitProtocol:
    return sp.SplitProtocol(
        name="stratified_holdout_cv", version="1.0", test_fraction=0.2,
        canonical_seed=42, num_folds=5, variance_seeds=(7, 21, 42, 84, 168),
    )


def _cfg() -> dict:
    return {
        "key": "synthetic_a",
        "official_split": False,
        "num_classes": 3,
        "class_map": {"alpha": "Alpha", "beta": "Beta", "gamma": "Gamma"},
    }


def _inventory() -> dict:
    return {
        "dataset_key": "synthetic_a",
        "schema_version": 2,
        "duplicates": {
            "groups": [
                ["alpha/a1.jpg", "alpha/a2.jpg"],
                ["beta/b1.jpg", "gamma/g1.jpg"],
            ]
        },
    }


def _build_fixture(root: Path) -> None:
    for name in ("a1", "a2", "a3", "a4"):
        _touch(root / "alpha" / f"{name}.jpg")
    for name in ("b1", "b2", "b3"):
        _touch(root / "beta" / f"{name}.jpg")
    for name in ("g1", "g2", "g3"):
        _touch(root / "gamma" / f"{name}.jpg")


RESOLUTION = {"beta/b1.jpg|gamma/g1.jpg": "exclude_group"}


def _make_bundle_and_index(tmp_path):
    _build_fixture(tmp_path)
    cfg, inv = _cfg(), _inventory()
    bundle = sg.generate_split_manifests(cfg, tmp_path, inv, _protocol(), cross_class_resolution=RESOLUTION)
    index = build_index(cfg, tmp_path)
    return bundle, index, inv


# --- happy path ---------------------------------------------------------------- #
def test_valid_bundle_passes_validation(tmp_path):
    bundle, index, inv = _make_bundle_and_index(tmp_path)
    sm.validate_manifest_bundle(bundle, index, inv)  # must not raise


# --- schema ---------------------------------------------------------------------- #
def test_missing_top_level_key_rejected(tmp_path):
    bundle, index, inv = _make_bundle_and_index(tmp_path)
    del bundle["cv"]
    with pytest.raises(sm.ManifestValidationError, match="missing required key"):
        sm.validate_manifest_bundle(bundle, index, inv)


def test_wrong_manifest_type_rejected(tmp_path):
    bundle, index, inv = _make_bundle_and_index(tmp_path)
    bundle = copy.deepcopy(bundle)
    bundle["test"]["manifest_type"] = "something_else"
    with pytest.raises(sm.ManifestValidationError, match="manifest_type"):
        sm.validate_manifest_bundle(bundle, index, inv)


# --- provenance -------------------------------------------------------------------- #
def test_provenance_dataset_key_mismatch_rejected(tmp_path):
    bundle, index, inv = _make_bundle_and_index(tmp_path)
    bundle = copy.deepcopy(bundle)
    bundle["test"]["provenance"]["dataset_key"] = "wrong_key"
    with pytest.raises(sm.ManifestValidationError, match="dataset_key"):
        sm.validate_manifest_bundle(bundle, index, inv)


def test_inventory_fingerprint_mismatch_rejected(tmp_path):
    bundle, index, inv = _make_bundle_and_index(tmp_path)
    different_inventory = copy.deepcopy(inv)
    different_inventory["duplicates"]["groups"] = []
    with pytest.raises(sm.ManifestValidationError, match="inventory_fingerprint"):
        sm.validate_manifest_bundle(bundle, index, different_inventory)


def test_provenance_missing_field_rejected(tmp_path):
    bundle, index, inv = _make_bundle_and_index(tmp_path)
    bundle = copy.deepcopy(bundle)
    del bundle["test"]["provenance"]["class_map"]
    with pytest.raises(sm.ManifestValidationError, match="missing required key"):
        sm.validate_manifest_bundle(bundle, index, inv)


# --- partition integrity ------------------------------------------------------------ #
def test_overlap_between_test_and_dev_rejected(tmp_path):
    bundle, index, inv = _make_bundle_and_index(tmp_path)
    bundle = copy.deepcopy(bundle)
    leaked_sample = bundle["dev_pool"]["samples"][0]
    bundle["test"]["samples"].append(leaked_sample)
    with pytest.raises(sm.ManifestValidationError, match="overlap"):
        sm.validate_manifest_bundle(bundle, index, inv)


def test_incomplete_partition_rejected(tmp_path):
    bundle, index, inv = _make_bundle_and_index(tmp_path)
    bundle = copy.deepcopy(bundle)
    bundle["dev_pool"]["samples"].pop()
    with pytest.raises(sm.ManifestValidationError, match="does not exactly partition"):
        sm.validate_manifest_bundle(bundle, index, inv)


def test_duplicate_sample_entry_rejected(tmp_path):
    bundle, index, inv = _make_bundle_and_index(tmp_path)
    bundle = copy.deepcopy(bundle)
    bundle["test"]["samples"].append(bundle["test"]["samples"][0])
    with pytest.raises(sm.ManifestValidationError, match="duplicate sample path"):
        sm.validate_manifest_bundle(bundle, index, inv)


def test_unknown_sample_path_rejected(tmp_path):
    bundle, index, inv = _make_bundle_and_index(tmp_path)
    bundle = copy.deepcopy(bundle)
    bundle["test"]["samples"].append("alpha/does_not_exist.jpg")
    with pytest.raises(sm.ManifestValidationError, match="unknown sample path"):
        sm.validate_manifest_bundle(bundle, index, inv)


# --- duplicate-group atomicity -------------------------------------------------------- #
def test_duplicate_group_split_across_test_and_dev_rejected(tmp_path):
    bundle, index, inv = _make_bundle_and_index(tmp_path)
    bundle = copy.deepcopy(bundle)
    test_set, dev_set = set(bundle["test"]["samples"]), set(bundle["dev_pool"]["samples"])
    if "alpha/a1.jpg" in dev_set:
        dev_set.discard("alpha/a1.jpg")
        test_set.add("alpha/a1.jpg")
    else:
        test_set.discard("alpha/a1.jpg")
        dev_set.add("alpha/a1.jpg")
    bundle["test"]["samples"] = sorted(test_set)
    bundle["dev_pool"]["samples"] = sorted(dev_set)
    # keep fold_assignment covering exactly dev_set for this test's purpose is not needed;
    # the test/dev atomicity check runs before cv checks.
    with pytest.raises(sm.ManifestValidationError, match="atomicity"):
        sm.validate_manifest_bundle(bundle, index, inv)


# --- cv fold integrity ----------------------------------------------------------------- #
def test_cv_fold_assignment_incomplete_rejected(tmp_path):
    bundle, index, inv = _make_bundle_and_index(tmp_path)
    bundle = copy.deepcopy(bundle)
    first_seed = next(iter(bundle["cv"]))
    fa = bundle["cv"][first_seed]["fold_assignment"]
    fa.pop(next(iter(fa)))
    with pytest.raises(sm.ManifestValidationError, match="does not exactly cover"):
        sm.validate_manifest_bundle(bundle, index, inv)


def test_cv_out_of_range_fold_index_rejected(tmp_path):
    bundle, index, inv = _make_bundle_and_index(tmp_path)
    bundle = copy.deepcopy(bundle)
    first_seed = next(iter(bundle["cv"]))
    fa = bundle["cv"][first_seed]["fold_assignment"]
    some_key = next(iter(fa))
    fa[some_key] = 999
    with pytest.raises(sm.ManifestValidationError, match="out-of-range"):
        sm.validate_manifest_bundle(bundle, index, inv)


def test_cv_duplicate_group_split_across_folds_rejected(tmp_path):
    bundle, index, inv = _make_bundle_and_index(tmp_path)
    bundle = copy.deepcopy(bundle)
    dev_set = set(bundle["dev_pool"]["samples"])
    if not {"alpha/a1.jpg", "alpha/a2.jpg"} <= dev_set:
        pytest.skip("alpha duplicate group landed in test for this seed; nothing to assert")
    first_seed = next(iter(bundle["cv"]))
    fa = bundle["cv"][first_seed]["fold_assignment"]
    fa["alpha/a1.jpg"] = 0
    fa["alpha/a2.jpg"] = 1
    with pytest.raises(sm.ManifestValidationError, match="atomicity"):
        sm.validate_manifest_bundle(bundle, index, inv)


# --- validate_protocol_match (Phase 4, M4) ------------------------------------- #
def test_validate_protocol_match_accepts_matching_protocol(tmp_path):
    bundle, _index, _inv = _make_bundle_and_index(tmp_path)
    sm.validate_protocol_match(bundle, _protocol())  # must not raise


def test_validate_protocol_match_rejects_mismatched_name(tmp_path):
    bundle, _index, _inv = _make_bundle_and_index(tmp_path)
    other = sp.SplitProtocol(
        name="some_other_protocol", version="1.0", test_fraction=0.2,
        canonical_seed=42, num_folds=5, variance_seeds=(42,),
    )
    with pytest.raises(sm.ManifestValidationError, match="expected"):
        sm.validate_protocol_match(bundle, other)


def test_validate_protocol_match_rejects_mismatched_version(tmp_path):
    bundle, _index, _inv = _make_bundle_and_index(tmp_path)
    other = sp.SplitProtocol(
        name="stratified_holdout_cv", version="2.0", test_fraction=0.2,
        canonical_seed=42, num_folds=5, variance_seeds=(42,),
    )
    with pytest.raises(sm.ManifestValidationError, match="expected"):
        sm.validate_protocol_match(bundle, other)


# --- select_subset (Phase 4, M4) ------------------------------------------------ #
def test_select_subset_test_and_dev_pool(tmp_path):
    bundle, _index, _inv = _make_bundle_and_index(tmp_path)
    assert sm.select_subset(bundle, "test") == sorted(bundle["test"]["samples"])
    assert sm.select_subset(bundle, "dev_pool") == sorted(bundle["dev_pool"]["samples"])


def test_select_subset_cv_fold_and_cv_val_are_identical(tmp_path):
    bundle, _index, _inv = _make_bundle_and_index(tmp_path)
    seed = next(iter(bundle["cv"]))
    fold_result = sm.select_subset(bundle, "cv_fold", cv_seed=seed, fold=0)
    val_result = sm.select_subset(bundle, "cv_val", cv_seed=seed, fold=0)
    assert fold_result == val_result


def test_select_subset_cv_train_is_complement_of_cv_fold(tmp_path):
    bundle, _index, _inv = _make_bundle_and_index(tmp_path)
    seed = next(iter(bundle["cv"]))
    fold_result = set(sm.select_subset(bundle, "cv_fold", cv_seed=seed, fold=0))
    train_result = set(sm.select_subset(bundle, "cv_train", cv_seed=seed, fold=0))
    dev_set = set(bundle["dev_pool"]["samples"])
    assert fold_result | train_result == dev_set
    assert fold_result.isdisjoint(train_result)


def test_select_subset_unsupported_subset_rejected(tmp_path):
    bundle, _index, _inv = _make_bundle_and_index(tmp_path)
    from dataset_index import DatasetLoadingError
    with pytest.raises(DatasetLoadingError, match="Unsupported subset"):
        sm.select_subset(bundle, "bogus_subset")


def test_select_subset_cv_requires_seed_and_fold(tmp_path):
    bundle, _index, _inv = _make_bundle_and_index(tmp_path)
    from dataset_index import DatasetLoadingError
    with pytest.raises(DatasetLoadingError, match="requires both cv_seed and fold"):
        sm.select_subset(bundle, "cv_train")


def test_select_subset_unknown_cv_seed_rejected(tmp_path):
    bundle, _index, _inv = _make_bundle_and_index(tmp_path)
    from dataset_index import DatasetLoadingError
    with pytest.raises(DatasetLoadingError, match="not part of this manifest"):
        sm.select_subset(bundle, "cv_fold", cv_seed=99999, fold=0)


def test_select_subset_out_of_range_fold_rejected(tmp_path):
    bundle, _index, _inv = _make_bundle_and_index(tmp_path)
    from dataset_index import DatasetLoadingError
    seed = next(iter(bundle["cv"]))
    with pytest.raises(DatasetLoadingError, match="out of range"):
        sm.select_subset(bundle, "cv_fold", cv_seed=seed, fold=999)
