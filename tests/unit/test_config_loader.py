"""Unit tests for the configuration loader and composition (loading only)."""
from pathlib import Path

import pytest

import config_loader as cl

ENV = {"CLOUD_DATA_ROOT": "/data/clouds", "SCRATCH": "/scratch/u"}


# --- per-concept loaders --------------------------------------------------- #
def test_load_model_is_architecture_identity():
    model = cl.load_model("resnet34")
    assert model["architecture"] == "resnet34"
    assert "num_classes" not in model  # class count belongs to the dataset


def test_dataset_owns_num_classes():
    ds = cl.load_dataset("merged_v1")
    assert ds["num_classes"] == 6
    assert ds["relative_data_path"] == "merged_dataset"


def test_training_protocol_holds_hyperparameters():
    tp = cl.load_training_protocol("baseline")
    assert tp["num_epochs"] == 10 and tp["augmentation"] == "none"


def test_evaluation_protocol_schema():
    ev = cl.load_evaluation_protocol("standard")
    assert ev["primary_metric"] == "accuracy"


def test_local_environment_expands_data_root():
    env = cl.load_environment("local", environ=ENV)
    assert env["data_root"] == "/data/clouds"


def test_missing_env_var_fails_loud():
    with pytest.raises(cl.ConfigError):
        cl.load_environment("local", environ={})


# --- composition ----------------------------------------------------------- #
def test_baseline_resolves_to_one_run():
    runs = cl.resolve_experiment("local", "baseline", environ=ENV)
    assert len(runs) == 1
    run = runs[0]
    assert run.model["name"] == "resnet18"
    assert run.dataset["key"] == "merged_v1"
    # dataset_path uses OS-native separators (Windows local / Linux Levante).
    assert run.dataset_path == str(Path("/data/clouds") / "merged_dataset")


def test_architecture_comparison_expands_to_multiple_runs():
    runs = cl.resolve_experiment("local", "architecture_comparison", environ=ENV)
    assert [r.model["name"] for r in runs] == ["resnet18", "resnet34", "resnet50"]
    # Same dataset and protocols across the comparison.
    assert {r.dataset["key"] for r in runs} == {"merged_v1"}
    assert {r.training["name"] for r in runs} == {"baseline"}


def test_harmonized_5bin_experiment_resolves():
    run = cl.resolve_experiment("local", "harmonized_5bin_resnet18", environ=ENV)[0]

    assert run.dataset["key"] == "harmonized_5bin"
    assert run.model["name"] == "resnet18"
    assert run.training["name"] == "tuned_resnet18"


def test_scalar_model_is_normalised_to_list():
    assert cl._normalize_models({"name": "x", "model": "resnet18"}) == ["resnet18"]
    assert cl._normalize_models({"name": "x", "models": ["a", "b"]}) == ["a", "b"]


def test_experiment_without_model_fails_loud():
    with pytest.raises(cl.ConfigError):
        cl._normalize_models({"name": "x"})


# --- approval_status propagation ------------------------------------------ #
def test_draft_components_yield_draft_run():
    run = cl.resolve_experiment("local", "baseline", environ=ENV)[0]
    assert run.approval_status == "draft"


def test_validation_experiment_yields_validation_run():
    run = cl.resolve_experiment("local", "smoke", environ=ENV)[0]
    assert run.approval_status == "validation"


def test_derive_status_precedence():
    approved = {"approval_status": "approved"}
    draft = {"approval_status": "draft"}
    validation = {"approval_status": "validation"}
    assert cl._derive_run_status(approved, approved, approved, approved) == "approved"
    assert cl._derive_run_status(approved, draft, approved, approved) == "draft"
    assert cl._derive_run_status(approved, validation, draft, approved) == "validation"


def test_unknown_reference_fails_loud():
    with pytest.raises(cl.ConfigError):
        cl.load_model("does_not_exist")


def test_read_toml_accepts_utf8_bom(tmp_path):
    cfg_dir = tmp_path / "config"
    model_dir = cfg_dir / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "bom.toml").write_bytes(b"\xef\xbb\xbfname = \"bom\"\narchitecture = \"resnet18\"\n")

    model = cl.load_model("bom", config_dir=cfg_dir)

    assert model["name"] == "bom"


def test_all_shipped_toml_files_parse():
    for path in cl.DEFAULT_CONFIG_DIR.rglob("*.toml"):
        assert cl._read_toml(path), f"Failed to parse {path}"
