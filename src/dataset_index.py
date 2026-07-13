"""Dataset indexing layer (Phase 4, Layer A).

Reads the on-disk class-folder structure of an approved dataset and produces a
deterministic, labeled sample index. This module represents repository
*structure* only: it never reads image pixels, never preprocesses or
augments, never creates DataLoaders, and never imports training code.

Class labels come exclusively from the approved ``class_map`` in the
dataset's configuration (see ``config/datasets/README.md`` and
``docs/DECISIONS.md``, Phase 3.5) -- class ordering is never derived from the
filesystem. This module has no import side effects and depends on the
standard library only (no torch, torchvision, or PIL).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

# Mirrors dataset_discovery.IMAGE_EXTENSIONS, duplicated here (rather than
# imported) so this module never pulls in PIL even transitively.
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")

# Canonical split directory names this layer recognises. Presence is
# *observed*, never assumed: a dataset with official_split = false yields no
# splits regardless of what directories happen to exist on disk.
_SPLIT_NAMES = ("train", "test")


class DatasetLoadingError(RuntimeError):
    """Raised when a dataset's on-disk structure violates its approved config."""


@dataclass(frozen=True)
class DatasetIndex:
    """A deterministic, labeled sample index for one dataset (and split, if any).

    ``samples`` paths are relative POSIX paths rooted at the resolved class
    directory (the split directory, or the dataset root for datasets with no
    official split), e.g. ``"Ac/Ac-N001.jpg"``.
    """

    dataset_key: str
    split: str | None
    classes: list[str]
    class_to_idx: dict[str, int]
    class_display_names: dict[str, str]
    samples: list[tuple[str, int]]
    num_classes: int

    def __post_init__(self) -> None:
        if not (self.num_classes == len(self.classes) == len(self.class_to_idx)):
            raise DatasetLoadingError(
                "Internal invariant violated: num_classes == len(classes) == "
                f"len(class_to_idx) does not hold for dataset '{self.dataset_key}' "
                f"(num_classes={self.num_classes}, classes={len(self.classes)}, "
                f"class_to_idx={len(self.class_to_idx)})."
            )


def _class_map(dataset_cfg: Mapping[str, Any]) -> dict[str, str]:
    class_map = dataset_cfg.get("class_map")
    if not class_map:
        key = dataset_cfg.get("key", dataset_cfg.get("name", "<unknown>"))
        raise DatasetLoadingError(
            f"Dataset '{key}' has no approved 'class_map' (pending Phase 3.5 "
            "scientific approval)."
        )
    return dict(class_map)


def available_splits(dataset_cfg: Mapping[str, Any], dataset_root: Path | str) -> list[str]:
    """Return the split directories actually present under *dataset_root*.

    Returns an empty list when the dataset's approved config declares
    ``official_split = false`` (e.g. CCSN) -- a split is never invented even
    if directories happen to exist. When ``official_split = true`` (e.g.
    GCD), this reports which of the canonical split directories (``train``,
    ``test``) are actually present, sorted.
    """
    if not dataset_cfg.get("official_split", False):
        return []
    root = Path(dataset_root)
    if not root.is_dir():
        raise DatasetLoadingError(f"Dataset root is not a directory: {root}")
    present = {p.name for p in root.iterdir() if p.is_dir()}
    return sorted(name for name in _SPLIT_NAMES if name in present)


def _resolve_class_dir(
    dataset_cfg: Mapping[str, Any],
    dataset_root: Path,
    split: str | None,
) -> Path:
    official_split = bool(dataset_cfg.get("official_split", False))
    key = dataset_cfg.get("key", dataset_cfg.get("name", "<unknown>"))

    if official_split:
        splits = available_splits(dataset_cfg, dataset_root)
        if split is None:
            raise DatasetLoadingError(
                f"Dataset '{key}' has an official split; specify one of "
                f"{splits} explicitly (split=None is ambiguous)."
            )
        if split not in splits:
            raise DatasetLoadingError(
                f"Invalid split '{split}' for dataset '{key}'. Available "
                f"splits: {splits or '(none found on disk)'}."
            )
        return dataset_root / split

    if split is not None:
        raise DatasetLoadingError(
            f"Dataset '{key}' has no official split (official_split = false); "
            f"got split='{split}'. Inventing a split is out of scope -- load "
            "it as a single unsplit collection with split=None."
        )
    return dataset_root


def _enumerate_class_images(class_dir: Path) -> list[str]:
    """Return image filenames (relative POSIX, sorted) found under *class_dir*."""
    found: list[str] = []
    for path in class_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS:
            found.append(path.relative_to(class_dir).as_posix())
    return sorted(found)


def build_index(
    dataset_cfg: Mapping[str, Any],
    dataset_root: Path | str,
    *,
    split: str | None = None,
) -> DatasetIndex:
    """Build a deterministic :class:`DatasetIndex` for an approved dataset.

    Validates that the on-disk class folders exactly match the approved
    ``class_map`` (no missing, no extra, none empty), and that
    ``num_classes`` (when present in the config) agrees with ``class_map``.
    Raises :class:`DatasetLoadingError` on any contract violation; never
    repairs or silently tolerates a mismatch.
    """
    dataset_root = Path(dataset_root)
    key = dataset_cfg.get("key", dataset_cfg.get("name", "<unknown>"))
    class_map = _class_map(dataset_cfg)
    classes = list(class_map.keys())
    class_to_idx = {token: idx for idx, token in enumerate(classes)}

    declared_num_classes = dataset_cfg.get("num_classes")
    if declared_num_classes is not None and declared_num_classes != len(classes):
        raise DatasetLoadingError(
            f"Dataset '{key}' config mismatch: num_classes={declared_num_classes} "
            f"but class_map has {len(classes)} entries."
        )

    class_dir = _resolve_class_dir(dataset_cfg, dataset_root, split)
    if not class_dir.is_dir():
        raise DatasetLoadingError(f"Dataset directory not found: {class_dir}")

    observed = {p.name for p in class_dir.iterdir() if p.is_dir()}
    expected = set(classes)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        split_note = f" split={split!r}" if split else ""
        raise DatasetLoadingError(
            f"Dataset '{key}'{split_note} class folders do not match the "
            f"approved class_map. Missing: {missing or 'none'}. Unexpected: "
            f"{extra or 'none'}."
        )

    samples: list[tuple[str, int]] = []
    for token in classes:
        images = _enumerate_class_images(class_dir / token)
        if not images:
            raise DatasetLoadingError(
                f"Approved class '{token}' for dataset '{key}' has no images "
                f"under {class_dir / token}."
            )
        label = class_to_idx[token]
        samples.extend((f"{token}/{rel}", label) for rel in images)

    samples.sort(key=lambda item: item[0])

    return DatasetIndex(
        dataset_key=str(key),
        split=split,
        classes=classes,
        class_to_idx=class_to_idx,
        class_display_names=dict(class_map),
        samples=samples,
        num_classes=len(classes),
    )
