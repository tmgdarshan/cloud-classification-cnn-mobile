# -*- coding: utf-8 -*-
"""
Single Canonical Experiment Registry for Rigorous Scientific Provenance.

Tracks all official benchmark experiments with:
- Unique run ID and timestamp (ISO 8601)
- Dataset key and formal taxonomy name
- Canonical manifest SHA256 and sample counts
- Split seed and protocol (e.g., grouped_stratified_holdout_v1.0)
- Model architecture and hyperparameter configuration
- Model checkpoint path and checkpoint SHA256
- Comprehensive metrics: Loss, Accuracy, Balanced Accuracy, Macro-F1, Weighted-F1, Bootstrap 95% CIs
- Publication figure path
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_FILE = REPO_ROOT / "artifacts" / "experiment_registry.json"


class RegistryError(RuntimeError):
    """Raised when the experiment registry cannot be read or written safely."""


def compute_file_sha256(path: Path | str) -> str:
    """Computes SHA256 checksum of a file."""
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def register_experiment(
    experiment_id: str,
    dataset_key: str,
    taxonomy_name: str,
    manifest_path: Path | str,
    protocol: str,
    model_architecture: str,
    hyperparameters: dict[str, Any],
    metrics: dict[str, Any],
    checkpoint_path: Path | str | None = None,
    figure_path: Path | str | None = None,
    notes: str = "",
    registry_file: Path = REGISTRY_FILE,
) -> dict[str, Any]:
    """Registers an experiment run into the canonical JSON registry."""
    registry_file.parent.mkdir(parents=True, exist_ok=True)

    manifest_p = Path(manifest_path) if manifest_path else None
    manifest_hash = compute_file_sha256(manifest_p) if manifest_p and manifest_p.exists() else ""

    checkpoint_p = Path(checkpoint_path) if checkpoint_path else None
    checkpoint_hash = compute_file_sha256(checkpoint_p) if checkpoint_p and checkpoint_p.exists() else ""

    record = {
        "experiment_id": experiment_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_key": dataset_key,
        "taxonomy_name": taxonomy_name,
        "manifest_path": str(manifest_p) if manifest_p else "",
        "manifest_sha256": manifest_hash,
        "protocol": protocol,
        "model_architecture": model_architecture,
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "checkpoint_path": str(checkpoint_p) if checkpoint_p else "",
        "checkpoint_sha256": checkpoint_hash,
        "figure_path": str(figure_path) if figure_path else "",
        "notes": notes,
    }

    records = load_registry(registry_file)

    # Update if existing experiment_id or append
    existing_idx = next((i for i, r in enumerate(records) if r.get("experiment_id") == experiment_id), None)
    if existing_idx is not None:
        records[existing_idx] = record
    else:
        records.append(record)

    tmp_file = registry_file.with_suffix(registry_file.suffix + ".tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        f.write("\n")
    os.replace(tmp_file, registry_file)

    print(f"[+] Successfully registered experiment '{experiment_id}' in {registry_file}")
    return record


def load_registry(registry_file: Path = REGISTRY_FILE) -> list[dict[str, Any]]:
    """Loads all registered experiments."""
    if not registry_file.exists():
        return []
    try:
        with open(registry_file, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception as exc:
        raise RegistryError(f"Failed to load experiment registry {registry_file}: {exc}") from exc
    if not isinstance(records, list):
        raise RegistryError(f"Experiment registry {registry_file} must contain a JSON list.")
    return records


def checkpoint_status(record: dict[str, Any]) -> str:
    checkpoint_path = record.get("checkpoint_path")
    expected_hash = record.get("checkpoint_sha256")
    if not checkpoint_path:
        return "None"
    p = Path(checkpoint_path)
    if not p.exists():
        return "Missing"
    actual_hash = compute_file_sha256(p)
    if expected_hash and actual_hash == expected_hash:
        return "Verified"
    if expected_hash:
        return "Mismatch"
    return "Unverified"


def get_registry_summary_df(registry_file: Path = REGISTRY_FILE) -> pd.DataFrame:
    """Generates a concise summary DataFrame of registered experiments."""
    records = load_registry(registry_file)
    if not records:
        return pd.DataFrame()

    rows = []
    for r in records:
        m = r.get("metrics", {})
        boot = m.get("bootstrap_95ci", {})
        acc_ci = boot.get("accuracy", {})
        test_acc = m.get('test_holdout_accuracy', m.get('test_accuracy', m.get('overall_accuracy', 'N/A')))
        acc_str = f"{test_acc}%" if test_acc != 'N/A' else 'N/A'
        if "ci_lower" in acc_ci and "ci_upper" in acc_ci:
            acc_str += f" [{acc_ci['ci_lower']}%, {acc_ci['ci_upper']}%]"

        bal_acc = m.get('test_holdout_balanced_accuracy', m.get('test_balanced_accuracy', m.get('balanced_accuracy', 'N/A')))
        macro_f1 = m.get('test_holdout_macro_f1', m.get('test_macro_f1', m.get('macro_f1', 'N/A')))

        rows.append({
            "Experiment ID": r.get("experiment_id"),
            "Taxonomy": r.get("taxonomy_name"),
            "Model": r.get("model_architecture"),
            "Protocol": r.get("protocol"),
            "Test Acc (95% CI)": acc_str,
            "Balanced Acc": f"{bal_acc}%" if bal_acc != 'N/A' else 'N/A',
            "Macro-F1": f"{macro_f1}%" if macro_f1 != 'N/A' else 'N/A',
            "Checkpoint": checkpoint_status(r),
            "Timestamp": r.get("timestamp_utc", "")[:19].replace("T", " "),
        })

    return pd.DataFrame(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Canonical Experiment Registry Manager")
    parser.add_argument("--list", action="store_true", help="List all registered experiments")
    args = parser.parse_args()

    df = get_registry_summary_df()
    if df.empty:
        print("[*] Experiment registry is currently empty.")
    else:
        print("\n" + "=" * 105)
        print("CANONICAL EXPERIMENT REGISTRY")
        print("=" * 105)
        print(df.to_string(index=False))
        print("=" * 105)


if __name__ == "__main__":
    main()
