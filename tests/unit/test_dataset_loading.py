"""Unit tests for the dataset loading layer (Phase 4, Layer B).

Uses tiny synthetic on-disk fixtures shaped like CCSN (no official split) and
GCD (official train/test split), with real (tiny) JPEG images -- no real
dataset required.
"""
from pathlib import Path

import pytest
from PIL import Image
from torch.utils.data import Dataset

import dataset_loading as dl
from dataset_index import DatasetLoadingError


def _make_image(path: Path, size=(4, 4), color=(10, 20, 30), mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color).save(path, "JPEG")


def _ccsn_like_cfg() -> dict:
    return {
        "key": "ccsn_fixture",
        "official_split": False,
        "num_classes": 2,
        "class_map": {"Ac": "Altocumulus", "Cu": "Cumulus"},
    }


def _gcd_like_cfg() -> dict:
    return {
        "key": "gcd_fixture",
        "official_split": True,
        "num_classes": 2,
        "class_map": {"1_cumulus": "Cumulus", "4_clearsky": "Clear sky"},
    }


def _build_ccsn_like_fixture(root: Path) -> None:
    _make_image(root / "Ac" / "Ac-001.jpg", color=(10, 20, 30))
    _make_image(root / "Ac" / "Ac-002.jpg", color=(40, 50, 60))
    _make_image(root / "Cu" / "Cu-001.jpg", color=(70, 80, 90))


def _build_gcd_like_fixture(root: Path) -> None:
    for split in ("train", "test"):
        _make_image(root / split / "1_cumulus" / f"{split}_c_001.jpg", color=(10, 20, 30))
        _make_image(root / split / "4_clearsky" / f"{split}_s_001.jpg", color=(200, 210, 220))


# --- construction / metadata passthrough ------------------------------------ #
def test_construction_builds_index_and_exposes_metadata(tmp_path):
    _build_ccsn_like_fixture(tmp_path)
    ds = dl.CloudImageDataset(_ccsn_like_cfg(), tmp_path)

    assert isinstance(ds, Dataset)
    assert ds.dataset_key == "ccsn_fixture"
    assert ds.split is None
    assert ds.classes == ["Ac", "Cu"]
    assert ds.class_to_idx == {"Ac": 0, "Cu": 1}
    assert ds.class_display_names == {"Ac": "Altocumulus", "Cu": "Cumulus"}
    assert ds.num_classes == 2


def test_len_matches_index_samples(tmp_path):
    _build_ccsn_like_fixture(tmp_path)
    ds = dl.CloudImageDataset(_ccsn_like_cfg(), tmp_path)
    assert len(ds) == 3
    assert len(ds) == len(ds.index.samples)


# --- __getitem__ / RGB conversion / transform ------------------------------- #
def test_getitem_returns_rgb_pil_image_and_int_label_when_no_transform(tmp_path):
    _build_ccsn_like_fixture(tmp_path)
    ds = dl.CloudImageDataset(_ccsn_like_cfg(), tmp_path)
    image, label = ds[0]
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert isinstance(label, int)


def test_getitem_labels_match_index_order(tmp_path):
    _build_ccsn_like_fixture(tmp_path)
    ds = dl.CloudImageDataset(_ccsn_like_cfg(), tmp_path)
    for i, (_rel_path, expected_label) in enumerate(ds.index.samples):
        _, label = ds[i]
        assert label == expected_label


def test_grayscale_image_is_converted_to_rgb(tmp_path):
    _make_image(tmp_path / "Ac" / "Ac-001.jpg", size=(4, 4), color=128, mode="L")
    _make_image(tmp_path / "Cu" / "Cu-001.jpg", size=(4, 4), color=64, mode="L")
    ds = dl.CloudImageDataset(_ccsn_like_cfg(), tmp_path)
    image, _ = ds[0]
    assert image.mode == "RGB"


def test_transform_none_returns_pil_image_unchanged(tmp_path):
    _build_ccsn_like_fixture(tmp_path)
    ds = dl.CloudImageDataset(_ccsn_like_cfg(), tmp_path, transform=None)
    image, _ = ds[0]
    assert isinstance(image, Image.Image)


def test_transform_is_applied_exactly_once(tmp_path):
    _build_ccsn_like_fixture(tmp_path)
    calls = []

    def fake_transform(img):
        calls.append(img)
        return "transformed"

    ds = dl.CloudImageDataset(_ccsn_like_cfg(), tmp_path, transform=fake_transform)
    result, label = ds[0]

    assert result == "transformed"
    assert len(calls) == 1
    assert isinstance(calls[0], Image.Image)
    assert isinstance(label, int)


# --- CCSN behaviour: no official split --------------------------------------- #
def test_ccsn_like_exposes_single_dataset_with_no_split(tmp_path):
    _build_ccsn_like_fixture(tmp_path)
    ds = dl.CloudImageDataset(_ccsn_like_cfg(), tmp_path)
    assert ds.split is None


def test_ccsn_like_rejects_split_request(tmp_path):
    _build_ccsn_like_fixture(tmp_path)
    with pytest.raises(DatasetLoadingError):
        dl.CloudImageDataset(_ccsn_like_cfg(), tmp_path, split="train")


# --- GCD behaviour: official train/test -------------------------------------- #
def test_gcd_like_train_and_test_splits(tmp_path):
    _build_gcd_like_fixture(tmp_path)
    train_ds = dl.CloudImageDataset(_gcd_like_cfg(), tmp_path, split="train")
    test_ds = dl.CloudImageDataset(_gcd_like_cfg(), tmp_path, split="test")

    assert train_ds.split == "train"
    assert test_ds.split == "test"
    assert len(train_ds) == 2
    assert len(test_ds) == 2

    image, label = train_ds[0]
    assert isinstance(image, Image.Image)
    assert isinstance(label, int)


def test_gcd_like_requires_explicit_split(tmp_path):
    _build_gcd_like_fixture(tmp_path)
    with pytest.raises(DatasetLoadingError):
        dl.CloudImageDataset(_gcd_like_cfg(), tmp_path)


# --- corrupted image handling ------------------------------------------------- #
def test_corrupt_image_raises_informative_dataset_loading_error(tmp_path):
    _make_image(tmp_path / "Ac" / "Ac-001.jpg", color=(10, 20, 30))
    (tmp_path / "Cu").mkdir(parents=True)
    (tmp_path / "Cu" / "Cu-001.jpg").write_bytes(b"not really a jpeg")

    ds = dl.CloudImageDataset(_ccsn_like_cfg(), tmp_path)
    idx = next(i for i, (rel, _label) in enumerate(ds.index.samples) if rel.startswith("Cu/"))

    with pytest.raises(DatasetLoadingError) as excinfo:
        ds[idx]

    message = str(excinfo.value)
    assert "ccsn_fixture" in message
    assert "Cu/Cu-001.jpg" in message


# --- determinism / no caching -------------------------------------------------- #
def test_deterministic_indexing_inherited_from_layer_a(tmp_path):
    _build_ccsn_like_fixture(tmp_path)
    ds1 = dl.CloudImageDataset(_ccsn_like_cfg(), tmp_path)
    ds2 = dl.CloudImageDataset(_ccsn_like_cfg(), tmp_path)

    assert ds1.index.samples == ds2.index.samples
    assert [ds1[i][1] for i in range(len(ds1))] == [ds2[i][1] for i in range(len(ds2))]


def test_images_are_not_cached_between_accesses(tmp_path):
    _build_ccsn_like_fixture(tmp_path)
    ds = dl.CloudImageDataset(_ccsn_like_cfg(), tmp_path)
    image_first, _ = ds[0]
    image_second, _ = ds[0]
    assert image_first is not image_second  # freshly decoded each time
