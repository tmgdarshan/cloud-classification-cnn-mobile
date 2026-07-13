"""Unit tests for the dataset-agnostic split generator (Phase 4, M3).

Uses small synthetic fixtures with deliberately non-CCSN/GCD names to prove
the generator is driven entirely by dataset_cfg + inventory + protocol, and
never special-cases a particular dataset.
"""
import inspect
from pathlib import Path

import pytest

import split_generator as sg
import split_protocol as sp


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder -- this layer never reads pixel content")


def _protocol(**overrides) -> sp.SplitProtocol:
    defaults = dict(
        name="stratified_holdout_cv", version="1.0", test_fraction=0.2,
        canonical_seed=42, num_folds=5, variance_seeds=(7, 21, 42, 84, 168),
    )
    defaults.update(overrides)
    return sp.SplitProtocol(**defaults)


def _small_cfg() -> dict:
    return {
        "key": "synthetic_a",
        "official_split": False,
        "num_classes": 3,
        "class_map": {"alpha": "Alpha", "beta": "Beta", "gamma": "Gamma"},
    }


def _small_inventory() -> dict:
    return {
        "dataset_key": "synthetic_a",
        "schema_version": 2,
        "duplicates": {
            "groups": [
                ["alpha/a1.jpg", "alpha/a2.jpg"],       # within-class
                ["beta/b1.jpg", "gamma/g1.jpg"],        # cross-class
            ]
        },
    }


def _build_small_fixture(root: Path) -> None:
    for name in ("a1", "a2", "a3", "a4"):
        _touch(root / "alpha" / f"{name}.jpg")
    for name in ("b1", "b2", "b3"):
        _touch(root / "beta" / f"{name}.jpg")
    for name in ("g1", "g2", "g3"):
        _touch(root / "gamma" / f"{name}.jpg")


CROSS_CLASS_GROUP_ID = "beta/b1.jpg|gamma/g1.jpg"


# --- halting on unresolved cross-class duplicates --------------------------- #
def test_generate_halts_on_unresolved_cross_class_duplicates(tmp_path):
    _build_small_fixture(tmp_path)
    with pytest.raises(sg.SplitGenerationError, match="cross-class"):
        sg.generate_split_manifests(_small_cfg(), tmp_path, _small_inventory(), _protocol())


def test_unsupported_resolution_action_rejected(tmp_path):
    _build_small_fixture(tmp_path)
    with pytest.raises(sg.SplitGenerationError, match="Unsupported"):
        sg.generate_split_manifests(
            _small_cfg(), tmp_path, _small_inventory(), _protocol(),
            cross_class_resolution={CROSS_CLASS_GROUP_ID: "acknowledge_and_keep"},
        )


def test_unknown_group_id_in_resolution_rejected(tmp_path):
    _build_small_fixture(tmp_path)
    with pytest.raises(sg.SplitGenerationError, match="not found"):
        sg.generate_split_manifests(
            _small_cfg(), tmp_path, _small_inventory(), _protocol(),
            cross_class_resolution={
                CROSS_CLASS_GROUP_ID: "exclude_group",
                "bogus.jpg|other.jpg": "exclude_group",
            },
        )


def test_generate_succeeds_with_exclude_group_resolution(tmp_path):
    _build_small_fixture(tmp_path)
    bundle = sg.generate_split_manifests(
        _small_cfg(), tmp_path, _small_inventory(), _protocol(),
        cross_class_resolution={CROSS_CLASS_GROUP_ID: "exclude_group"},
    )
    excluded = set(bundle["test"]["provenance"]["excluded_samples"])
    assert excluded == {"beta/b1.jpg", "gamma/g1.jpg"}
    all_split_samples = set(bundle["test"]["samples"]) | set(bundle["dev_pool"]["samples"])
    assert excluded.isdisjoint(all_split_samples)


# --- structural preconditions ------------------------------------------------ #
def test_official_split_dataset_rejected(tmp_path):
    _build_small_fixture(tmp_path)
    cfg = _small_cfg()
    cfg["official_split"] = True
    with pytest.raises(sg.SplitGenerationError, match="official split"):
        sg.generate_split_manifests(cfg, tmp_path, _small_inventory(), _protocol())


def test_dataset_key_mismatch_rejected(tmp_path):
    _build_small_fixture(tmp_path)
    inv = _small_inventory()
    inv["dataset_key"] = "some_other_key"
    with pytest.raises(sg.SplitGenerationError, match="does not match"):
        sg.generate_split_manifests(_small_cfg(), tmp_path, inv, _protocol())


def test_inventory_dataset_drift_rejected(tmp_path):
    _build_small_fixture(tmp_path)
    inv = _small_inventory()
    inv["duplicates"]["groups"].append(["alpha/does_not_exist.jpg", "alpha/a1.jpg"])
    with pytest.raises(sg.SplitGenerationError, match="drift"):
        sg.generate_split_manifests(
            _small_cfg(), tmp_path, inv, _protocol(),
            cross_class_resolution={CROSS_CLASS_GROUP_ID: "exclude_group"},
        )


# --- duplicate-group atomicity ------------------------------------------------ #
def test_within_class_duplicate_group_stays_together(tmp_path):
    _build_small_fixture(tmp_path)
    bundle = sg.generate_split_manifests(
        _small_cfg(), tmp_path, _small_inventory(), _protocol(),
        cross_class_resolution={CROSS_CLASS_GROUP_ID: "exclude_group"},
    )
    test_set = set(bundle["test"]["samples"])
    dev_set = set(bundle["dev_pool"]["samples"])
    a1_in_test, a2_in_test = "alpha/a1.jpg" in test_set, "alpha/a2.jpg" in test_set
    a1_in_dev, a2_in_dev = "alpha/a1.jpg" in dev_set, "alpha/a2.jpg" in dev_set
    assert a1_in_test == a2_in_test
    assert a1_in_dev == a2_in_dev


def test_within_class_duplicate_group_shares_fold_when_in_dev(tmp_path):
    _build_small_fixture(tmp_path)
    bundle = sg.generate_split_manifests(
        _small_cfg(), tmp_path, _small_inventory(), _protocol(),
        cross_class_resolution={CROSS_CLASS_GROUP_ID: "exclude_group"},
    )
    dev_set = set(bundle["dev_pool"]["samples"])
    if "alpha/a1.jpg" in dev_set:
        for cv_seed, cv_m in bundle["cv"].items():
            assert cv_m["fold_assignment"]["alpha/a1.jpg"] == cv_m["fold_assignment"]["alpha/a2.jpg"]


# --- provenance --------------------------------------------------------------- #
def test_provenance_contains_expected_fields(tmp_path):
    _build_small_fixture(tmp_path)
    inv = _small_inventory()
    bundle = sg.generate_split_manifests(
        _small_cfg(), tmp_path, inv, _protocol(),
        cross_class_resolution={CROSS_CLASS_GROUP_ID: "exclude_group"},
    )
    prov = bundle["test"]["provenance"]
    assert prov["dataset_key"] == "synthetic_a"
    assert prov["protocol_name"] == "stratified_holdout_cv"
    assert prov["protocol_version"] == "1.0"
    assert prov["inventory_fingerprint"] == sp.compute_inventory_fingerprint(inv)
    assert prov["class_map"] == {"alpha": "Alpha", "beta": "Beta", "gamma": "Gamma"}
    assert prov["num_samples_total"] == 10
    assert prov["cross_class_resolution"] == {CROSS_CLASS_GROUP_ID: "exclude_group"}


def test_dev_pool_and_test_provenance_are_identical(tmp_path):
    _build_small_fixture(tmp_path)
    bundle = sg.generate_split_manifests(
        _small_cfg(), tmp_path, _small_inventory(), _protocol(),
        cross_class_resolution={CROSS_CLASS_GROUP_ID: "exclude_group"},
    )
    assert bundle["test"]["provenance"] == bundle["dev_pool"]["provenance"]


# --- CV structure -------------------------------------------------------------- #
def test_cv_manifests_cover_all_variance_seeds(tmp_path):
    _build_small_fixture(tmp_path)
    bundle = sg.generate_split_manifests(
        _small_cfg(), tmp_path, _small_inventory(), _protocol(),
        cross_class_resolution={CROSS_CLASS_GROUP_ID: "exclude_group"},
    )
    assert set(bundle["cv"].keys()) == {7, 21, 42, 84, 168}
    dev_set = set(bundle["dev_pool"]["samples"])
    for cv_m in bundle["cv"].values():
        assert set(cv_m["fold_assignment"].keys()) == dev_set
        assert all(0 <= f < 5 for f in cv_m["fold_assignment"].values())


# --- exact-fraction respected on a duplicate-free, larger fixture ------------- #
def _build_large_fixture(root: Path) -> None:
    for i in range(50):
        _touch(root / "alpha" / f"a{i:03d}.jpg")
    for i in range(50):
        _touch(root / "beta" / f"b{i:03d}.jpg")


def _large_cfg() -> dict:
    return {
        "key": "synthetic_large",
        "official_split": False,
        "num_classes": 2,
        "class_map": {"alpha": "Alpha", "beta": "Beta"},
    }


def _large_inventory() -> dict:
    return {"dataset_key": "synthetic_large", "schema_version": 2, "duplicates": {"groups": []}}


def test_test_fraction_is_exact_with_no_duplicate_groups(tmp_path):
    _build_large_fixture(tmp_path)
    bundle = sg.generate_split_manifests(_large_cfg(), tmp_path, _large_inventory(), _protocol())
    test_samples = bundle["test"]["samples"]
    alpha_test = [p for p in test_samples if p.startswith("alpha/")]
    beta_test = [p for p in test_samples if p.startswith("beta/")]
    assert len(alpha_test) == 10   # round(0.2 * 50)
    assert len(beta_test) == 10


def test_fold_assignment_differs_across_variance_seeds(tmp_path):
    _build_large_fixture(tmp_path)
    bundle = sg.generate_split_manifests(_large_cfg(), tmp_path, _large_inventory(), _protocol())
    fa_7 = bundle["cv"][7]["fold_assignment"]
    fa_21 = bundle["cv"][21]["fold_assignment"]
    assert fa_7 != fa_21


# --- determinism --------------------------------------------------------------- #
def test_full_bundle_generation_is_deterministic(tmp_path):
    _build_small_fixture(tmp_path)
    inv = _small_inventory()
    resolution = {CROSS_CLASS_GROUP_ID: "exclude_group"}
    bundle_1 = sg.generate_split_manifests(_small_cfg(), tmp_path, inv, _protocol(), cross_class_resolution=resolution)
    bundle_2 = sg.generate_split_manifests(_small_cfg(), tmp_path, inv, _protocol(), cross_class_resolution=resolution)
    assert bundle_1 == bundle_2


# --- architectural guard: no dataset special-casing --------------------------- #
def test_no_ccsn_or_gcd_special_casing_in_source():
    src_generator = inspect.getsource(sg)
    src_protocol = inspect.getsource(sp)
    for forbidden in ("ccsn", "gcd"):
        assert forbidden not in src_generator.lower(), f"'{forbidden}' found in split_generator.py"
        assert forbidden not in src_protocol.lower(), f"'{forbidden}' found in split_protocol.py"
