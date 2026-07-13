"""Dataset loading layer (Phase 4, Layer B).

Wraps a :class:`dataset_index.DatasetIndex` in a standard PyTorch ``Dataset``
that lazily decodes images. This layer is intentionally thin: it does not
discover datasets, derive labels, invent splits, preprocess, augment, batch,
shuffle, or read training configuration -- those belong to later phases.
Pillow is the only new dependency this module introduces.

:meth:`CloudImageDataset.from_manifest_bundle` (Phase 4, M4) additionally
consumes a split-manifest bundle produced by ``split_generator`` (M3). It
never generates, modifies, or re-derives a split itself -- it calls
``split_manifest.validate_manifest_bundle`` and ``validate_protocol_match``
directly against the *current* dataset config, inventory, and protocol, then
restricts its own sample list to the requested subset. Methodology lives in
``split_generator``/``split_manifest``; this module only consumes it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from PIL import Image
from torch.utils.data import Dataset

from dataset_index import DatasetIndex, DatasetLoadingError, build_index
from split_manifest import select_subset, validate_manifest_bundle, validate_protocol_match
from split_protocol import SplitProtocol

__all__ = ["CloudImageDataset"]


class CloudImageDataset(Dataset):
    """A PyTorch ``Dataset`` over one dataset's (and split's) approved images.

    Wraps a :class:`~dataset_index.DatasetIndex` built via
    :func:`dataset_index.build_index`. Every image is decoded lazily in
    :meth:`__getitem__`, converted to RGB, and never cached. An optional
    ``transform`` is applied to the decoded image before it is returned; if
    ``transform`` is ``None`` the RGB :class:`PIL.Image.Image` is returned
    unchanged (no resizing, normalization, or tensor conversion happens here).
    """

    def __init__(
        self,
        dataset_cfg: Mapping[str, Any],
        dataset_root: Path | str,
        *,
        split: str | None = None,
        transform: Callable[[Image.Image], Any] | None = None,
        _prebuilt_index: DatasetIndex | None = None,
    ) -> None:
        # _prebuilt_index is internal-only: it lets from_manifest_bundle hand
        # in an already-filtered index without a second build_index scan.
        if _prebuilt_index is not None:
            self._index: DatasetIndex = _prebuilt_index
        else:
            self._index = build_index(dataset_cfg, dataset_root, split=split)
        # Mirrors the single-line split/no-split join dataset_index resolves
        # internally; the class-folder contract validation stays exclusively
        # in build_index, this is only path arithmetic.
        root = Path(dataset_root)
        self._class_dir = root / split if split is not None else root
        self._transform = transform

    # --- manifest-based construction (Phase 4, M4) --------------------------- #
    @classmethod
    def from_manifest_bundle(
        cls,
        bundle: Mapping[str, Any],
        dataset_cfg: Mapping[str, Any],
        dataset_root: Path | str,
        inventory: Mapping[str, Any],
        protocol: SplitProtocol,
        *,
        subset: str,
        cv_seed: int | None = None,
        fold: int | None = None,
        transform: Callable[[Image.Image], Any] | None = None,
    ) -> "CloudImageDataset":
        """Build a dataset restricted to one subset of a split-manifest bundle.

        ``subset`` is one of ``"test"``, ``"dev_pool"``, ``"cv_fold"``,
        ``"cv_train"``, or ``"cv_val"`` (the last two require ``cv_seed`` and
        ``fold``). The bundle is validated against the *current* dataset
        config, inventory, and protocol at the moment of consumption --
        calling ``split_manifest.validate_manifest_bundle`` and
        ``validate_protocol_match`` directly, never re-implementing their
        logic. Sample order and label assignment come entirely from the
        dataset's own :class:`~dataset_index.DatasetIndex`, unchanged.
        """
        key = dataset_cfg.get("key", dataset_cfg.get("name", "<unknown>"))
        if dataset_cfg.get("official_split", False):
            raise DatasetLoadingError(
                f"Dataset '{key}' has an official split; manifest-based subsets "
                "apply only to datasets with official_split = false."
            )

        base_index = build_index(dataset_cfg, dataset_root)
        validate_manifest_bundle(bundle, base_index, inventory)
        validate_protocol_match(bundle, protocol)

        selected = set(select_subset(bundle, subset, cv_seed=cv_seed, fold=fold))
        filtered_samples = [(p, label) for p, label in base_index.samples if p in selected]

        subset_index = DatasetIndex(
            dataset_key=base_index.dataset_key,
            split=base_index.split,
            classes=base_index.classes,
            class_to_idx=base_index.class_to_idx,
            class_display_names=base_index.class_display_names,
            samples=filtered_samples,
            num_classes=base_index.num_classes,
        )
        return cls(dataset_cfg, dataset_root, transform=transform, _prebuilt_index=subset_index)

    # --- read-only passthrough of DatasetIndex metadata --------------------- #
    @property
    def index(self) -> DatasetIndex:
        """The underlying :class:`DatasetIndex` (read-only access)."""
        return self._index

    @property
    def dataset_key(self) -> str:
        return self._index.dataset_key

    @property
    def split(self) -> str | None:
        return self._index.split

    @property
    def classes(self) -> list[str]:
        return self._index.classes

    @property
    def class_to_idx(self) -> dict[str, int]:
        return self._index.class_to_idx

    @property
    def class_display_names(self) -> dict[str, str]:
        return self._index.class_display_names

    @property
    def num_classes(self) -> int:
        return self._index.num_classes

    # --- PyTorch Dataset interface ------------------------------------------ #
    def __len__(self) -> int:
        return len(self._index.samples)

    def __getitem__(self, i: int) -> tuple[Any, int]:
        rel_path, label = self._index.samples[i]
        image_path = self._class_dir / rel_path
        try:
            with Image.open(image_path) as raw:
                image = raw.convert("RGB")
        except Exception as exc:  # noqa: BLE001 - any failure means "undecodable"
            raise DatasetLoadingError(
                f"Failed to decode image for dataset '{self._index.dataset_key}' "
                f"(split={self._index.split!r}) at relative path '{rel_path}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if self._transform is not None:
            image = self._transform(image)
        return image, label
