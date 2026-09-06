# -*- coding: utf-8 -*-
"""
Unit tests for canonical manifest integrity and duplicate-group isolation.
"""
import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR = REPO_ROOT / "metadata" / "splits"


@pytest.mark.parametrize("manifest_name", [
    "ccsn_11class_canonical.json",
    "gcd_6class_canonical.json",
    "gcd_5class_canonical.json",
    "harmonized_5bin_canonical.json",
])
def test_manifest_integrity_and_no_leakage(manifest_name):
    manifest_path = SPLITS_DIR / manifest_name
    assert manifest_path.exists(), f"Manifest file missing: {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "samples" in data
    assert "summary" in data
    assert len(data["samples"]) > 0

    samples = data["samples"]
    train_paths = {s["path"] for s in samples if s["split"] == "train"}
    val_paths = {s["path"] for s in samples if s["split"] == "val"}
    test_paths = {s["path"] for s in samples if s["split"] == "test"}

    # 1. Check strict partition disjointness
    assert len(train_paths & val_paths) == 0, f"Train and Val overlap in {manifest_name}!"
    assert len(train_paths & test_paths) == 0, f"Train and Test overlap in {manifest_name}!"
    assert len(val_paths & test_paths) == 0, f"Val and Test overlap in {manifest_name}!"
    assert len(train_paths | val_paths | test_paths) == len(samples), f"Total sample count mismatch in {manifest_name}!"

    # 2. Check group-aware isolation (no duplicate group straddles partitions)
    group_partitions = {}
    for s in samples:
        gid = s["group_id"]
        split = s["split"]
        group_partitions.setdefault(gid, set()).add(split)

    for gid, parts in group_partitions.items():
        assert len(parts) == 1, f"Group {gid} leaked across partitions {parts} in {manifest_name}!"

