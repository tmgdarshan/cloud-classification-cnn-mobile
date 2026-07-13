"""Print the resolved run(s) for a given environment and experiment.

Usage:
    python scripts/show_config.py --environment local --experiment architecture_comparison

An experiment expands into one run per model. The environment may also be
provided via the CLOUD_ENV variable. This script only *loads and prints*
configuration; it does not run training or touch data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make the config loader importable without installing a package (pre-packaging).
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config_loader import ConfigError, resolve_experiment  # noqa: E402


def _run_to_dict(run) -> dict:
    return {
        "environment_name": run.environment_name,
        "experiment_name": run.experiment_name,
        "approval_status": run.approval_status,
        "environment": dict(run.environment),
        "experiment": dict(run.experiment),
        "dataset": dict(run.dataset),
        "model": dict(run.model),
        "training": dict(run.training),
        "evaluation": dict(run.evaluation),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show resolved project run(s).")
    parser.add_argument(
        "--environment", "-e",
        default=os.environ.get("CLOUD_ENV", "local"),
        help="Environment profile name (default: $CLOUD_ENV or 'local').",
    )
    parser.add_argument(
        "--experiment", "-x",
        required=True,
        help="Experiment profile name (e.g. 'baseline', 'architecture_comparison').",
    )
    args = parser.parse_args(argv)

    try:
        runs = resolve_experiment(args.environment, args.experiment)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    exp_name = runs[0].experiment["name"]
    print(f"Experiment '{exp_name}' -> {len(runs)} run(s) [{runs[0].approval_status}]:")
    for run in runs:
        print(f"  - model={run.model['name']:<12} dataset={run.dataset['key']:<10} "
              f"training={run.training['name']:<10} evaluation={run.evaluation['name']}")
    print()
    print(json.dumps([_run_to_dict(r) for r in runs], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
