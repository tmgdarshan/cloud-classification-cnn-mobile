"""Unit tests for the dataset indexing layer (Phase 4, Layer A).

Uses small synthetic fixtures shaped like CCSN (no official split) and GCD
(official train/test split) -- no real dataset, no torch, no PIL required.
Placeholder files stand in for images; this layer never reads pixels.
"""
from pathlib import Path

import pytest

import dataset_index as di


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real image, just a placeholder file")


def _ccsn_like_cfg() -> dict:
    return {
        "key": "ccsn_fixture",
        "official_split": False,
        "num_classes": 3,
        "class_map": {"Ac": "Altocumulus", "Cu": "Cumulus", "St": "Stratus"},
    }


def _gcd_like_cfg() -> dict:
    return {
        "key": "gcd_fixture",
        "official_split": True,
        "num_classes": 2,
        "class_map": {"1_cumulus": "Cumulus", "4_clearsky": "Clear sky"},
    }


def _build_ccsn_like_fixture(root: Path) -> None:
    _touch(root / "Ac" / "Ac-001.jpg")
    _touch(root / "Ac" / "Ac-002.jpg")
    _touch(root / "Cu" / "Cu-001.jpg")
    _touch(root / "St" / "St-001.jpg")
    _touch(root / "St" / "St-002.jpg")
    _touch(root / "St" / "St-003.jpg")


def _build_gcd_like_fixture(root: Path) -> None:
    for split in ("train", "test"):
        _touch(root / split / "1_cumulus" / f"{split}_c_001.jpg")
        _touch(root / split / "1_cumulus" / f"{split}_c_002.jpg")
        _touch(root / split / "4_clearsky" / f"{split}_s_001.jpg")


# --- CCSN-like: no official split ------------------------------------------ #
def test_ccsn_like_available_splits_is_empty(tmp_path):
    _build_ccsn_like_fixture(tmp_path)
    assert di.available_splits(_ccsn_like_cfg(), tmp_path) == []


def test_ccsn_like_indexing_builds_one_unsplit_collection(tmp_path):
    _build_ccsn_like_fixture(tmp_path)
    idx = di.build_index(_ccsn_like_cfg(), tmp_path)

    assert idx.split is None
    assert idx.classes == ["Ac", "Cu", "St"]                 # class_map order
    assert idx.class_to_idx == {"Ac": 0, "Cu": 1, "St": 2}
    assert idx.class_display_names == {
        "Ac": "Altocumulus", "Cu": "Cumulus", "St": "Stratus",
    }
    assert idx.num_classes == 3
    assert len(idx.samples) == 6

    labels = [label for _, label in idx.samples]
    assert labels.count(idx.class_to_idx["St"]) == 3
    assert labels.count(idx.class_to_idx["Ac"]) == 2
    assert labels.count(idx.class_to_idx["Cu"]) == 1


def test_ccsn_like_rejects_split_request(tmp_path):
    _build_ccsn_like_fixture(tmp_path)
    with pytest.raises(di.DatasetLoadingError):
        di.build_index(_ccsn_like_cfg(), tmp_path, split="train")


# --- GCD-like: official split ----------------------------------------------- #
def test_gcd_like_available_splits(tmp_path):
    _build_gcd_like_fixture(tmp_path)
    assert di.available_splits(_gcd_like_cfg(), tmp_path) == ["test", "train"]


def test_gcd_like_indexing_train_and_test(tmp_path):
    _build_gcd_like_fixture(tmp_path)
    train_idx = di.build_index(_gcd_like_cfg(), tmp_path, split="train")
    test_idx = di.build_index(_gcd_like_cfg(), tmp_path, split="test")

    assert train_idx.split == "train"
    assert test_idx.split == "test"
    assert train_idx.classes == ["1_cumulus", "4_clearsky"]  # class_map order
    assert train_idx.class_to_idx == {"1_cumulus": 0, "4_clearsky": 1}
    assert len(train_idx.samples) == 3   # 2 cumulus + 1 clearsky
    assert len(test_idx.samples) == 3

    # samples are relative to the split dir, not prefixed with the split name
    all_paths = [p for p, _ in train_idx.samples + test_idx.samples]
    assert all(not p.startswith("train/") and not p.startswith("test/") for p in all_paths)


def test_gcd_like_requires_explicit_split(tmp_path):
    _build_gcd_like_fixture(tmp_path)
    with pytest.raises(di.DatasetLoadingError):
        di.build_index(_gcd_like_cfg(), tmp_path)   # split=None is ambiguous


def test_gcd_like_invalid_split_name(tmp_path):
    _build_gcd_like_fixture(tmp_path)
    with pytest.raises(di.DatasetLoadingError):
        di.build_index(_gcd_like_cfg(), tmp_path, split="val")


# --- class_map / contract validation ---------------------------------------- #
def test_missing_approved_class_folder_fails_loud(tmp_path):
    _touch(tmp_path / "Ac" / "Ac-001.jpg")
    _touch(tmp_path / "Cu" / "Cu-001.jpg")
    # "St" folder is entirely missing
    with pytest.raises(di.DatasetLoadingError):
        di.build_index(_ccsn_like_cfg(), tmp_path)


def test_unexpected_extra_class_folder_fails_loud(tmp_path):
    _build_ccsn_like_fixture(tmp_path)
    _touch(tmp_path / "Ns" / "Ns-001.jpg")   # not in class_map
    with pytest.raises(di.DatasetLoadingError):
        di.build_index(_ccsn_like_cfg(), tmp_path)


def test_declared_excluded_class_folder_is_ignored(tmp_path):
    _build_gcd_like_fixture(tmp_path)
    _touch(tmp_path / "train" / "7_mixed" / "mixed_001.jpg")
    cfg = _gcd_like_cfg()
    cfg["excluded_class_folders"] = ["7_mixed"]

    idx = di.build_index(cfg, tmp_path, split="train")

    assert len(idx.samples) == 3
    assert all(not rel.startswith("7_mixed/") for rel, _ in idx.samples)


def test_empty_approved_class_fails_loud(tmp_path):
    _touch(tmp_path / "Ac" / "Ac-001.jpg")
    _touch(tmp_path / "Cu" / "Cu-001.jpg")
    (tmp_path / "St").mkdir()   # approved class folder exists but is empty
    with pytest.raises(di.DatasetLoadingError):
        di.build_index(_ccsn_like_cfg(), tmp_path)


def test_num_classes_mismatch_fails_loud(tmp_path):
    _build_ccsn_like_fixture(tmp_path)
    cfg = _ccsn_like_cfg()
    cfg["num_classes"] = 99   # disagrees with class_map's 3 entries
    with pytest.raises(di.DatasetLoadingError):
        di.build_index(cfg, tmp_path)


def test_missing_class_map_fails_loud(tmp_path):
    _build_ccsn_like_fixture(tmp_path)
    cfg = {"key": "no_map", "official_split": False}
    with pytest.raises(di.DatasetLoadingError):
        di.build_index(cfg, tmp_path)


# --- determinism ------------------------------------------------------------- #
def test_index_is_deterministic(tmp_path):
    _build_ccsn_like_fixture(tmp_path)
    cfg = _ccsn_like_cfg()
    first = di.build_index(cfg, tmp_path)
    second = di.build_index(cfg, tmp_path)
    assert first == second


def test_samples_sorted_by_relative_path(tmp_path):
    _build_ccsn_like_fixture(tmp_path)
    idx = di.build_index(_ccsn_like_cfg(), tmp_path)
    paths = [p for p, _ in idx.samples]
    assert paths == sorted(paths)


# --- invariant ---------------------------------------------------------------- #
def test_dataset_index_invariant_enforced():
    with pytest.raises(di.DatasetLoadingError):
        di.DatasetIndex(
            dataset_key="broken",
            split=None,
            classes=["a", "b"],
            class_to_idx={"a": 0, "b": 1},
            class_display_names={"a": "A", "b": "B"},
            samples=[],
            num_classes=99,   # inconsistent with classes/class_to_idx
        )
