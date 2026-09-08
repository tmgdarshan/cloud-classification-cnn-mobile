# -*- coding: utf-8 -*-
"""
Unit tests for canonical manifest contracts and production pipeline data structures.
"""
import ast
import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR = REPO_ROOT / "metadata" / "splits"


@pytest.mark.parametrize("manifest_key,expected_classes", [
    ("ccsn_11class_canonical.json", 11),
    ("gcd_6class_canonical.json", 6),
    ("gcd_5class_canonical.json", 5),
    ("harmonized_5bin_canonical.json", 5),
])
def test_canonical_manifest_structure_and_contract(manifest_key: str, expected_classes: int):
    manifest_path = SPLITS_DIR / manifest_key
    assert manifest_path.exists(), f"Manifest missing: {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Core metadata contracts
    assert "dataset_key" in manifest
    assert "taxonomy_name" in manifest
    assert "classes" in manifest
    assert len(manifest["classes"]) == expected_classes
    assert "class_to_idx" in manifest
    assert len(manifest["class_to_idx"]) == expected_classes

    # Samples contract
    samples = manifest["samples"]
    assert len(samples) > 0

    splits_found = set()
    for s in samples:
        assert "path" in s
        assert "label" in s
        assert "group_id" in s
        assert "split" in s
        splits_found.add(s["split"])

    # Verify 3-way partition contract
    assert splits_found == {"train", "val", "test"}


def test_harmonized_canonical_manifest_mappings():
    manifest_path = SPLITS_DIR / "harmonized_5bin_canonical.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "mappings" in manifest
    mappings = manifest["mappings"]
    assert "ccsn_to_harmonized" in mappings
    assert "gcd_to_harmonized" in mappings

    # Check key meteorological genera mappings
    ccsn_map = mappings["ccsn_to_harmonized"]
    assert ccsn_map["Cu"] == "cumulus"
    assert ccsn_map["Cb"] == "cumulonimbus"
    assert ccsn_map["Ci"] == "cirrus"
    assert ccsn_map["Ac"] == "altocumulus"
    assert ccsn_map["Sc"] == "stratocumulus"

    gcd_map = mappings["gcd_to_harmonized"]
    assert gcd_map["1_cumulus"] == "cumulus"
    assert gcd_map["6_cumulonimbus"] == "cumulonimbus"


def test_atmospheric_report_call_contract(tmp_path):
    import numpy as np
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from evaluation import generate_evaluation_report

    classes = ["cumulus", "altocumulus", "cirrus", "stratocumulus", "cumulonimbus"]
    y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4, 1, 1, 2, 3, 3])

    # Must execute with exact caller signature from run_harmonized.py without TypeError
    rep = generate_evaluation_report(
        y_true,
        y_pred,
        classes,
        dataset_name="Test Dataset (Five-Class, RESNET18)",
        model_name="resnet18",
        output_dir=tmp_path / "figures",
        title_suffix="Test Holdout",
    )

    assert "metrics_summary" in rep
    assert "overall_accuracy" in rep["metrics_summary"]
    assert "balanced_accuracy" in rep["metrics_summary"]
    assert "macro_f1" in rep["metrics_summary"]
    assert "bootstrap_95ci" in rep["metrics_summary"]



def test_run_harmonized_report_call_signatures():
    """Verify all calls to generate_evaluation_report in run_harmonized.py adhere to the (y_true, y_pred, classes) contract."""
    runner_path = REPO_ROOT / "src" / "run_harmonized.py"
    tree = ast.parse(runner_path.read_text(encoding="utf-8"), filename=str(runner_path))

    report_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name == "generate_evaluation_report":
                report_calls.append(node)

    assert len(report_calls) == 7, f"Expected 7 report call sites, found {len(report_calls)}"
    for call in report_calls:
        # Must have at most 3 positional arguments: y_true, y_pred, class_names
        assert len(call.args) <= 3, f"Call at line {call.lineno} passes too many positional args: {len(call.args)}"
        kw_names = {kw.arg for kw in call.keywords}
        assert "dataset_name" in kw_names, f"Call at line {call.lineno} missing keyword arg 'dataset_name'"
        assert "model_name" in kw_names, f"Call at line {call.lineno} missing keyword arg 'model_name'"
