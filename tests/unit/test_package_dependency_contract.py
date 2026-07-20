"""Release metadata contracts for the self-contained GraphOS distribution."""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _metadata() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def _epistemic_requirements(metadata: dict) -> list[str]:
    project = metadata["project"]
    values = list(project["dependencies"])
    for dependencies in project["optional-dependencies"].values():
        values.extend(dependencies)
    return [value for value in values if value.startswith("epistemic-graph")]


def test_every_epistemic_graph_dependency_requests_full_numeric_runtime() -> None:
    requirements = _epistemic_requirements(_metadata())

    assert requirements
    assert set(requirements) == {
        "epistemic-graph[full]>=2.23.1,<3.0.0"
    }


def test_flat_requirements_requests_full_numeric_runtime() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    epistemic = [line for line in requirements if line.startswith("epistemic-graph")]

    assert epistemic == ["epistemic-graph[full]>=2.23.1,<3.0.0"]


def test_agent_runtimes_use_only_the_native_optimization_runtime() -> None:
    optional = _metadata()["project"]["optional-dependencies"]
    for extra in ("agent-runtime", "agent-headless"):
        assert "dspy-ai>=3.2.1" not in optional[extra]
        assert not any(value.startswith("litellm") for value in optional[extra])


def test_provider_agent_runtime_excludes_duplicate_optimizer_stack() -> None:
    optional = _metadata()["project"]["optional-dependencies"]

    assert "dspy-ai>=3.2.1" not in optional["agent-runtime"]
    assert not any(value.startswith("litellm") for value in optional["agent-runtime"])
    assert not any(
        value.startswith("agent-utilities[agent") for value in optional["logfire"]
    )
    assert "agent" not in optional


def test_release_metadata_excludes_runtime_generated_package_state() -> None:
    metadata = _metadata()
    excluded = set(
        metadata["tool"]["setuptools"]["exclude-package-data"]["*"]
    )
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()

    assert ".agent_data/**" in excluded
    assert "**/.agent_data/**" in excluded
    assert "prune agent_utilities/.agent_data" in manifest


def test_documented_config_example_is_current_agent_config() -> None:
    from agent_utilities.core.config import (
        AgentConfig,
        _validate_agent_config_without_settings,
    )

    example = json.loads(
        (ROOT / "docs/examples/config.json").read_text(encoding="utf-8")
    )
    configured = {
        key: value for key, value in example.items() if not key.startswith("_comment")
    }
    aliases = {
        field.alias for field in AgentConfig.model_fields.values() if field.alias
    }
    unknown = set(configured) - set(AgentConfig.model_fields) - aliases

    assert unknown == set()
    validated = _validate_agent_config_without_settings(configured)
    assert validated.a2a_broker == "epistemic_graph"
    assert validated.a2a_storage == "epistemic_graph"
    assert validated.tool_guard_mode == "strict"


if __name__ == "__main__":  # pragma: no cover - convenient release smoke check
    raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
