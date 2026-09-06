"""Configuration loading, validation, and inspection infrastructure for the project.

This module provides configuration-inspection and schema-validation utilities for
the TOML configuration domain model (environments, datasets, models, training protocols,
evaluation protocols, and experiment composition).

The production training and evaluation pipeline is consolidated around `src/run_harmonized.py`
and frozen canonical manifests (`metadata/splits/*_canonical.json`). This module serves
as configuration-inspection infrastructure, verifying config consistency and composing
exploratory parameter profiles without runtime side effects.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:  # Python >= 3.11 ships tomllib in the standard library.
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - fallback for Python 3.10
    try:
        import tomli as _toml  # type: ignore[no-redef]
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Reading TOML configuration requires Python 3.11+ (tomllib) or the "
            "'tomli' package on Python 3.10. Install with: pip install tomli"
        ) from exc


# Repository root = two levels up from this file (src/config_loader.py -> root).
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_DIR = _REPO_ROOT / "config"

_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

# approval_status values, in order of precedence when deriving a run's status.
_STATUS_VALIDATION = "validation"
_STATUS_DRAFT = "draft"
_STATUS_APPROVED = "approved"


class ConfigError(RuntimeError):
    """Raised when a configuration cannot be found, parsed, or resolved."""


@dataclass(frozen=True)
class RunConfig:
    """A single resolved run: one environment, dataset, model, and pair of protocols.

    The inner mappings are kept as plain dictionaries so each concept's schema can
    evolve without changing this class.
    """

    environment_name: str
    experiment_name: str
    environment: Mapping[str, Any]
    experiment: Mapping[str, Any]
    dataset: Mapping[str, Any]
    model: Mapping[str, Any]
    training: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    approval_status: str

    @property
    def data_root(self) -> str:
        return self.environment["data_root"]

    @property
    def artifacts_root(self) -> str:
        return self.environment.get("artifacts_root", "artifacts")

    @property
    def dataset_path(self) -> str:
        """Absolute dataset location (``data_root`` joined with ``relative_data_path``).

        Raises :class:`ConfigError` if the dataset has no ``relative_data_path``
        yet (source-dataset stubs are populated during Phase 3 discovery).
        """
        relative = self.dataset.get("relative_data_path")
        if not relative:
            raise ConfigError(
                f"Dataset '{self.dataset.get('name')}' has no 'relative_data_path' "
                "(pending Phase 3 discovery)."
            )
        return str(Path(self.data_root) / relative)


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        return _toml.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:  # tomllib raises TOMLDecodeError; wrap for a clear message.
        raise ConfigError(f"Failed to parse configuration file {path}: {exc}") from exc


def _expand_env_vars(value: Any, environ: Mapping[str, str]) -> Any:
    """Recursively expand ``${VAR}`` references in strings, failing loudly if unset."""
    if isinstance(value, str):
        def _replace(match: "re.Match[str]") -> str:
            name = match.group(1)
            if name not in environ:
                raise ConfigError(
                    f"Environment variable '${{{name}}}' referenced in configuration is not set."
                )
            return environ[name]

        return _ENV_VAR_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {key: _expand_env_vars(val, environ) for key, val in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item, environ) for item in value]
    return value


def _load_profile(
    subdir: str,
    name: str,
    config_dir: Path | str,
    *,
    required: tuple[str, ...] = ("name",),
) -> dict[str, Any]:
    data = _read_toml(Path(config_dir) / subdir / f"{name}.toml")
    for key in required:
        if key not in data:
            raise ConfigError(
                f"{subdir}/{name}.toml is missing required key '{key}'."
            )
    return data


# --------------------------------------------------------------------------- #
# Per-concept loaders
# --------------------------------------------------------------------------- #
def load_environment(
    name: str,
    *,
    config_dir: Path | str = DEFAULT_CONFIG_DIR,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Load and resolve an environment profile (expands ``${VAR}`` paths)."""
    environ = os.environ if environ is None else environ
    data = _load_profile("environments", name, config_dir, required=("name", "data_root"))
    return _expand_env_vars(data, environ)


def load_dataset(name: str, *, config_dir: Path | str = DEFAULT_CONFIG_DIR) -> dict[str, Any]:
    return _load_profile("datasets", name, config_dir)


def load_model(name: str, *, config_dir: Path | str = DEFAULT_CONFIG_DIR) -> dict[str, Any]:
    return _load_profile("models", name, config_dir)


def load_training_protocol(name: str, *, config_dir: Path | str = DEFAULT_CONFIG_DIR) -> dict[str, Any]:
    return _load_profile("training", name, config_dir)


def load_evaluation_protocol(name: str, *, config_dir: Path | str = DEFAULT_CONFIG_DIR) -> dict[str, Any]:
    return _load_profile("evaluation", name, config_dir)


def load_experiment(name: str, *, config_dir: Path | str = DEFAULT_CONFIG_DIR) -> dict[str, Any]:
    return _load_profile(
        "experiments",
        name,
        config_dir,
        required=("name", "dataset", "training_protocol", "evaluation_protocol"),
    )


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #
def _normalize_models(experiment: Mapping[str, Any]) -> list[str]:
    """Return the experiment's models as a list, accepting scalar ``model`` sugar."""
    if "models" in experiment:
        models = experiment["models"]
        if isinstance(models, str):
            models = [models]
        if not isinstance(models, list) or not models:
            raise ConfigError(
                f"Experiment '{experiment.get('name')}' has an invalid 'models' value "
                "(expected a non-empty list or a string)."
            )
        return models
    if "model" in experiment:
        return [experiment["model"]]
    raise ConfigError(
        f"Experiment '{experiment.get('name')}' must define 'model' or 'models'."
    )


def _derive_run_status(*components: Mapping[str, Any]) -> str:
    """Derive a run's approval_status from its scientific components.

    Precedence: ``validation`` if any component is validation; ``approved`` only
    if all are approved; otherwise ``draft`` (the fail-safe default).
    """
    statuses = [c.get("approval_status", _STATUS_DRAFT) for c in components]
    if _STATUS_VALIDATION in statuses:
        return _STATUS_VALIDATION
    if all(status == _STATUS_APPROVED for status in statuses):
        return _STATUS_APPROVED
    return _STATUS_DRAFT


def resolve_experiment(
    environment: str,
    experiment: str,
    *,
    config_dir: Path | str = DEFAULT_CONFIG_DIR,
    environ: Mapping[str, str] | None = None,
) -> list[RunConfig]:
    """Resolve an environment + experiment into one :class:`RunConfig` per model."""
    env = load_environment(environment, config_dir=config_dir, environ=environ)
    exp = load_experiment(experiment, config_dir=config_dir)

    dataset = load_dataset(exp["dataset"], config_dir=config_dir)
    training = load_training_protocol(exp["training_protocol"], config_dir=config_dir)
    evaluation = load_evaluation_protocol(exp["evaluation_protocol"], config_dir=config_dir)
    status = _derive_run_status(exp, dataset, training, evaluation)

    runs: list[RunConfig] = []
    for model_name in _normalize_models(exp):
        model = load_model(model_name, config_dir=config_dir)
        runs.append(
            RunConfig(
                environment_name=environment,
                experiment_name=experiment,
                environment=env,
                experiment=exp,
                dataset=dataset,
                model=model,
                training=training,
                evaluation=evaluation,
                approval_status=status,
            )
        )
    return runs
