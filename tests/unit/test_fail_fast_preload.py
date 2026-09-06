from pathlib import Path

import pytest
from PIL import Image

import run_harmonized
import tune_resnet_family


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (20, 30, 40)).save(path, "JPEG")


def test_harmonized_preload_rejects_missing_image(tmp_path):
    valid = tmp_path / "valid.jpg"
    missing = tmp_path / "missing.jpg"
    _make_image(valid)

    with pytest.raises(RuntimeError) as excinfo:
        run_harmonized.preload_images_parallel_preallocated(
            [(valid, 0), (missing, 1)],
            target_size=(4, 4),
            max_workers=1,
        )

    assert str(missing) in str(excinfo.value)


def test_tuning_preload_rejects_corrupt_image(tmp_path):
    valid = tmp_path / "valid.jpg"
    corrupt = tmp_path / "corrupt.jpg"
    _make_image(valid)
    corrupt.write_bytes(b"not a jpeg")

    with pytest.raises(RuntimeError) as excinfo:
        tune_resnet_family.preload_images_parallel(
            [(valid, 0), (corrupt, 1)],
            target_size=(4, 4),
            max_workers=1,
        )

    assert str(corrupt) in str(excinfo.value)
