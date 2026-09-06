"""Unit tests for the split protocol specification (Phase 4, M3)."""
import pytest

import split_protocol as sp


def test_default_protocol_values():
    proto = sp.GROUPED_STRATIFIED_HOLDOUT_V1
    assert proto.name == "grouped_stratified_holdout"
    assert proto.version == "1.0"
    assert proto.test_fraction == 0.2
    assert proto.canonical_seed == 42
    assert proto.num_folds == 5
    assert proto.variance_seeds == (7, 21, 42, 84, 168)


@pytest.mark.parametrize("bad_fraction", [0.0, 1.0, -0.1, 1.5])
def test_invalid_test_fraction_rejected(bad_fraction):
    with pytest.raises(ValueError):
        sp.SplitProtocol(
            name="x", version="1.0", test_fraction=bad_fraction,
            canonical_seed=42, num_folds=5, variance_seeds=(42,),
        )


def test_invalid_num_folds_rejected():
    with pytest.raises(ValueError):
        sp.SplitProtocol(
            name="x", version="1.0", test_fraction=0.2,
            canonical_seed=42, num_folds=1, variance_seeds=(42,),
        )


def test_empty_variance_seeds_rejected():
    with pytest.raises(ValueError):
        sp.SplitProtocol(
            name="x", version="1.0", test_fraction=0.2,
            canonical_seed=42, num_folds=5, variance_seeds=(),
        )


def test_inventory_fingerprint_is_deterministic_regardless_of_key_order():
    inv_a = {"dataset_key": "x", "schema_version": 2, "duplicates": {"groups": [["a", "b"]]}}
    inv_b = {"schema_version": 2, "duplicates": {"groups": [["a", "b"]]}, "dataset_key": "x"}
    assert sp.compute_inventory_fingerprint(inv_a) == sp.compute_inventory_fingerprint(inv_b)


def test_inventory_fingerprint_changes_with_content():
    inv_a = {"dataset_key": "x", "duplicates": {"groups": [["a", "b"]]}}
    inv_b = {"dataset_key": "x", "duplicates": {"groups": [["a", "c"]]}}
    assert sp.compute_inventory_fingerprint(inv_a) != sp.compute_inventory_fingerprint(inv_b)
