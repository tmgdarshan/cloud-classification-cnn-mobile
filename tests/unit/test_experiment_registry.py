# -*- coding: utf-8 -*-
"""
Unit tests for Canonical Experiment Registry.
"""
import json
from pathlib import Path
import pytest
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiment_registry import (
    RegistryError,
    compute_file_sha256,
    get_registry_summary_df,
    load_registry,
    register_experiment,
)


def test_experiment_registry_lifecycle(tmp_path):
    temp_registry = tmp_path / "test_registry.json"
    temp_manifest = tmp_path / "test_manifest.json"
    with open(temp_manifest, "w", encoding="utf-8") as f:
        json.dump({"classes": ["Cu", "Ci"]}, f)

    record = register_experiment(
        experiment_id="test_run_001",
        dataset_key="test_dataset",
        taxonomy_name="Test Cloud Taxonomy",
        manifest_path=temp_manifest,
        protocol="grouped_stratified_holdout_v1.0",
        model_architecture="resnet18",
        hyperparameters={"epochs": 1, "batch_size": 32},
        metrics={
            "test_accuracy": 92.5,
            "test_balanced_accuracy": 91.0,
            "test_macro_f1": 90.5,
            "bootstrap_95ci": {
                "accuracy": {"ci_lower": 90.0, "ci_upper": 95.0},
            },
        },
        checkpoint_path=None,
        figure_path="artifacts/figures/test_cm.png",
        notes="Automated test entry",
        registry_file=temp_registry,
    )

    assert record["experiment_id"] == "test_run_001"
    assert record["manifest_sha256"] == compute_file_sha256(temp_manifest)
    assert temp_registry.exists()

    loaded = load_registry(temp_registry)
    assert len(loaded) == 1
    assert loaded[0]["experiment_id"] == "test_run_001"

    df = get_registry_summary_df(temp_registry)
    assert len(df) == 1
    assert "92.5% [90.0%, 95.0%]" in df["Test Acc (95% CI)"].values[0]


def test_corrupt_registry_fails_loud(tmp_path):
    temp_registry = tmp_path / "broken_registry.json"
    temp_registry.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(RegistryError):
        load_registry(temp_registry)


def test_register_does_not_overwrite_corrupt_registry(tmp_path):
    temp_registry = tmp_path / "broken_registry.json"
    temp_manifest = tmp_path / "test_manifest.json"
    temp_manifest.write_text("{}", encoding="utf-8")
    original = "{not valid json"
    temp_registry.write_text(original, encoding="utf-8")

    with pytest.raises(RegistryError):
        register_experiment(
            experiment_id="test_run_001",
            dataset_key="test_dataset",
            taxonomy_name="Test Cloud Taxonomy",
            manifest_path=temp_manifest,
            protocol="protocol",
            model_architecture="resnet18",
            hyperparameters={},
            metrics={},
            registry_file=temp_registry,
        )

    assert temp_registry.read_text(encoding="utf-8") == original
