"""Integration tests for dataset discovery against a synthetic fixture.

Builds a tiny on-disk dataset (valid images, a duplicate, a corrupt file, a
non-image file, and a train/test split) and checks the inventory. No real
dataset is required.
"""
from pathlib import Path

from PIL import Image

import dataset_discovery as dd


def _make_image(path: Path, size=(8, 8), color=(120, 130, 140)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, "JPEG")


def _build_fixture(root: Path) -> None:
    # train split, two class folders (distinct colours -> distinct bytes/hashes)
    _make_image(root / "train" / "Ci" / "a.jpg", color=(10, 20, 30))
    _make_image(root / "train" / "Ci" / "b.jpg", color=(40, 50, 60))
    _make_image(root / "train" / "Cu" / "c.jpg", color=(70, 80, 90))
    # test split
    _make_image(root / "test" / "Ci" / "d.jpg", color=(100, 110, 120))
    # an exact duplicate of a.jpg within the train split (same bytes)
    (root / "train" / "Ci" / "dup.jpg").write_bytes((root / "train" / "Ci" / "a.jpg").read_bytes())
    # an exact copy of a.jpg in the TEST split -> cross-split (leakage) duplicate
    (root / "test" / "Ci" / "leak.jpg").write_bytes((root / "train" / "Ci" / "a.jpg").read_bytes())
    # a corrupt "image"
    (root / "train" / "Cu" / "broken.jpg").write_bytes(b"not really a jpeg")
    # a non-image file
    (root / "train" / "notes.txt").write_text("ignore me")


def test_discovery_reports_structure_and_anomalies(tmp_path):
    _build_fixture(tmp_path)
    inv = dd.discover_dataset(tmp_path, key="fixture", label="Fixture")

    assert inv["structure"]["detected_splits"] == ["test", "train"]
    # class folder names are reported verbatim, not interpreted
    assert set(inv["structure"]["class_folders"]["train"]) == {"Ci", "Cu"}
    assert inv["structure"]["class_folders"]["train"]["Ci"] == 3  # a, b, dup
    assert inv["structure"]["class_folders"]["test"]["Ci"] == 2   # d, leak

    assert inv["summary"]["num_corrupt_images"] == 1
    assert inv["corrupt_images"][0]["path"].endswith("broken.jpg")
    # a.jpg == dup.jpg == leak.jpg -> one group, and it spans train & test
    assert inv["summary"]["num_duplicate_groups"] == 1
    assert inv["summary"]["num_cross_split_duplicate_groups"] == 1
    assert inv["duplicates"]["cross_split_groups"][0]["splits"] == ["test", "train"]
    assert inv["summary"]["num_non_image_files"] == 1   # notes.txt

    assert inv["images"]["dimensions"] == {"8x8": 6}    # 6 valid images


def test_inventory_is_deterministic(tmp_path):
    _build_fixture(tmp_path)
    first = dd.to_json(dd.discover_dataset(tmp_path, key="fixture"))
    second = dd.to_json(dd.discover_dataset(tmp_path, key="fixture"))
    assert first == second  # pure function of the data, no timestamps
