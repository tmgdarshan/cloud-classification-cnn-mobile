# -*- coding: utf-8 -*-
"""
Unit tests verifying taxonomy extension contracts in tune_resnet_family.py.
"""
from pathlib import Path
import sys
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tune_resnet_family import load_taxonomy_manifest_and_samples, METADATA_SPLITS, CCSN_DIR


def test_tune_resnet_family_taxonomy_ccsn11():
    """Verify that --taxonomy ccsn11 loads the canonical 11-class manifest with num_classes==11."""
    manifest, tr_samples, va_samples, num_classes = load_taxonomy_manifest_and_samples(
        taxonomy="ccsn11",
        pool="ccsn",
        metadata_dir=METADATA_SPLITS,
        ccsn_dir=CCSN_DIR,
    )

    assert num_classes == 11, f"Expected num_classes == 11 for ccsn11 taxonomy, got {num_classes}"
    assert manifest["dataset_key"] == "ccsn"
    assert len(manifest["classes"]) == 11

    # Verify train and val sample counts match canonical specification (1622 train / 407 val)
    assert len(tr_samples) == 1622, f"Expected 1622 train samples, got {len(tr_samples)}"
    assert len(va_samples) == 407, f"Expected 407 val samples, got {len(va_samples)}"

    # Verify zero test peeking: ensure no test samples are present
    test_paths = {s["path"] for s in manifest["samples"] if s["split"] == "test"}
    assert len(test_paths) == 508, f"Expected 508 test samples in manifest, got {len(test_paths)}"
    tr_paths_set = {str(p.relative_to(CCSN_DIR)).replace("\\", "/") for p, _ in tr_samples}
    va_paths_set = {str(p.relative_to(CCSN_DIR)).replace("\\", "/") for p, _ in va_samples}
    assert tr_paths_set.isdisjoint(test_paths), "Test samples leaked into train split!"
    assert va_paths_set.isdisjoint(test_paths), "Test samples leaked into val split!"

    # Verify all samples point to valid files under CCSN_DIR
    for p, lbl in tr_samples[:10]:
        assert p.is_relative_to(CCSN_DIR)
        assert 0 <= lbl <= 10


def test_tune_resnet_family_taxonomy_harmonized5_backward_compatibility():
    """Verify that default harmonized5 taxonomy maintains num_classes == 5."""
    manifest, tr_samples, va_samples, num_classes = load_taxonomy_manifest_and_samples(
        taxonomy="harmonized5",
        pool="joint",
        metadata_dir=METADATA_SPLITS,
    )

    assert num_classes == 5, f"Expected num_classes == 5 for harmonized5 taxonomy, got {num_classes}"
    assert len(manifest["classes"]) == 5
    assert len(tr_samples) > 0
    assert len(va_samples) > 0
