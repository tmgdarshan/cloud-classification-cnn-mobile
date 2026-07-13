"""Dataset discovery, validation & inventory (Phase 3).

Observational only. This module READS a dataset directory and reports what it
actually contains. It:

  * never modifies data (no resize / normalize / augment / preprocess),
  * never interprets folder names as scientific classes (folder names are
    reported verbatim; the Discover -> Report -> Ask workflow leaves the
    class mapping to the researcher),
  * never computes image-content statistics (no brightness / entropy /
    histograms / texture),
  * treats each dataset independently (no assumption that two datasets share a
    directory layout or a split structure).

The output is a deterministic inventory (a pure function of the data on disk,
with no timestamps) suitable for tracking in ``metadata/`` and diffing against
future scans to detect dataset changes.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

INVENTORY_SCHEMA_VERSION = 2

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")

# Directory names commonly used for dataset splits. Presence is *detected and
# reported*, never assumed.
KNOWN_SPLIT_NAMES = ("train", "training", "test", "testing", "val", "valid", "validation")

_UNINTERPRETED_NOTE = (
    "Folder names are reported verbatim and are NOT interpreted as scientific "
    "classes. Class mapping is approved by the researcher after this report."
)


class DiscoveryError(RuntimeError):
    """Raised when a dataset cannot be discovered (e.g. path does not exist)."""


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _inspect_image(path: Path) -> dict[str, Any]:
    """Read an image's format/mode/size and detect corruption (read-only)."""
    try:  # verification pass — detects truncated / corrupt files
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:  # noqa: BLE001 - any failure means "corrupt/unreadable"
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:  # metadata pass — verify() invalidates the object, so reopen
        with Image.open(path) as image:
            return {
                "ok": True,
                "format": image.format,
                "mode": image.mode,
                "width": image.width,
                "height": image.height,
            }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _split_and_class(parts: tuple[str, ...], detected_splits: list[str]) -> tuple[str, str]:
    """Attribute a file (given its path parts relative to the scan root) to a
    split and a top-level class folder, without interpreting either name."""
    if detected_splits and parts and parts[0] in detected_splits:
        split, rest = parts[0], parts[1:]
    else:
        split, rest = "", parts
    # ``rest`` still includes the filename; a length >= 2 means at least one dir.
    klass = rest[0] if len(rest) >= 2 else ""
    return split, klass


def _split_of(rel_posix: str, detected_splits: list[str]) -> str:
    """Return the split a relative path belongs to, or '' if not under a split."""
    first = rel_posix.split("/", 1)[0]
    return first if first in detected_splits else ""


def discover_dataset(scan_path: Path | str, *, key: str, label: str | None = None) -> dict[str, Any]:
    """Scan *scan_path* and return a deterministic inventory dictionary."""
    scan_path = Path(scan_path)
    if not scan_path.is_dir():
        raise DiscoveryError(f"Dataset path is not a directory: {scan_path}")

    top_dirs = sorted(p.name for p in scan_path.iterdir() if p.is_dir())
    top_entries = sorted(p.name for p in scan_path.iterdir())
    detected_splits = [d for d in top_dirs if d.lower() in KNOWN_SPLIT_NAMES]

    formats: Counter = Counter()
    modes: Counter = Counter()
    dimensions: Counter = Counter()
    file_sizes: list[int] = []
    filename_lengths: list[int] = []
    example_filenames: list[str] = []
    corrupt: list[dict[str, str]] = []
    hash_to_paths: dict[str, list[str]] = defaultdict(list)
    class_counts: dict[tuple[str, str], int] = defaultdict(int)
    num_files = num_images = num_non_image = 0

    for root, dirs, files in os.walk(scan_path):
        dirs.sort()
        files.sort()
        root_path = Path(root)
        for fname in files:
            fpath = root_path / fname
            rel = fpath.relative_to(scan_path)
            num_files += 1
            file_sizes.append(fpath.stat().st_size)
            filename_lengths.append(len(fname))
            if len(example_filenames) < 5:
                example_filenames.append(rel.as_posix())

            if fpath.suffix.lower() in IMAGE_EXTENSIONS:
                num_images += 1
                split, klass = _split_and_class(rel.parts, detected_splits)
                class_counts[(split, klass)] += 1
                info = _inspect_image(fpath)
                if info["ok"]:
                    formats[info["format"]] += 1
                    modes[info["mode"]] += 1
                    dimensions[f"{info['width']}x{info['height']}"] += 1
                else:
                    corrupt.append({"path": rel.as_posix(), "error": info["error"]})
                hash_to_paths[_sha256(fpath)].append(rel.as_posix())
            else:
                num_non_image += 1

    class_folders, ungrouped = _assemble_class_folders(class_counts, detected_splits)
    duplicate_groups = sorted(
        (sorted(paths) for paths in hash_to_paths.values() if len(paths) > 1)
    )
    # Scientific-integrity check: a duplicate group that spans two or more named
    # splits (e.g. train and test) is a data-leakage risk.
    cross_split_groups = []
    for group in duplicate_groups:
        named = sorted({_split_of(p, detected_splits) for p in group} - {""})
        if len(named) >= 2:
            cross_split_groups.append({"paths": group, "splits": named})

    return {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "dataset_key": key,
        "dataset_label": label or key,
        "scanned_path": scan_path.as_posix(),
        "note": _UNINTERPRETED_NOTE,
        "summary": {
            "num_files": num_files,
            "num_images": num_images,
            "num_non_image_files": num_non_image,
            "num_corrupt_images": len(corrupt),
            "num_duplicate_groups": len(duplicate_groups),
            "num_duplicate_extra_files": sum(len(g) - 1 for g in duplicate_groups),
            "num_cross_split_duplicate_groups": len(cross_split_groups),
            "total_bytes": sum(file_sizes),
        },
        "structure": {
            "top_level_entries": top_entries,
            "detected_splits": detected_splits,
            "class_folders": class_folders,
            "images_without_class_folder": ungrouped,
        },
        "images": {
            "formats": dict(sorted(formats.items())),
            "modes": dict(sorted(modes.items())),
            "dimensions": dict(sorted(dimensions.items())),
        },
        "file_sizes_bytes": _number_stats(file_sizes),
        "filenames": {
            "min_length": min(filename_lengths) if filename_lengths else 0,
            "max_length": max(filename_lengths) if filename_lengths else 0,
            "examples": example_filenames,
        },
        "corrupt_images": sorted(corrupt, key=lambda item: item["path"]),
        "duplicates": {
            "cross_split_applicable": bool(detected_splits),
            "num_groups": len(duplicate_groups),
            "num_cross_split_groups": len(cross_split_groups),
            "groups": duplicate_groups,
            "cross_split_groups": cross_split_groups,
        },
    }


def _assemble_class_folders(
    class_counts: dict[tuple[str, str], int],
    detected_splits: list[str],
) -> tuple[dict[str, Any], int]:
    """Group image counts by (split, class) into a readable, uninterpreted map."""
    ungrouped = 0
    if detected_splits:
        grouped: dict[str, dict[str, int]] = {}
        for (split, klass), count in class_counts.items():
            if klass:
                grouped.setdefault(split, {})[klass] = count
            else:
                ungrouped += count
        return {s: dict(sorted(v.items())) for s, v in sorted(grouped.items())}, ungrouped

    flat: dict[str, int] = {}
    for (_split, klass), count in class_counts.items():
        if klass:
            flat[klass] = count
        else:
            ungrouped += count
    return dict(sorted(flat.items())), ungrouped


def _number_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 1),
        "median": int(statistics.median(values)),
    }


def to_json(inventory: dict[str, Any]) -> str:
    """Serialise an inventory deterministically (sorted keys, trailing newline)."""
    return json.dumps(inventory, indent=2, sort_keys=True) + "\n"
