#!/usr/bin/python
"""Tests for Conductor per-step model_id routing (b5-07).

CONCEPT:ORCH-1.27
"""

from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.graph.executor import pick_specialist_model
from agent_utilities.models import ModelDefinition, ModelRegistry
from agent_utilities.models.sdd import Task

pytestmark = pytest.mark.concept("ORCH-1.27")


def _registry() -> ModelRegistry:
    return ModelRegistry(
        models=[
            ModelDefinition(
                id="cheap-local",
                name="Cheap",
                provider="openai",
                model_id="llama-3.2-3b-instruct",
                base_url="http://localhost:1234/v1",
                tier="light",
                is_default=True,
            ),
            ModelDefinition(
                id="premium-cloud",
                name="Premium",
                provider="openai",
                model_id="gpt-4o",
                api_key_env="OPENAI_API_KEY",
                tier="heavy",
            ),
        ]
    )


def _deps(requested=None):
    deps = MagicMock()
    deps.model_registry = _registry()
    deps.requested_model_id = requested
    deps.agent_model = MagicMock(name="default_model")
    return deps


# --- Task schema ------------------------------------------------------------


def test_task_has_model_id_field():
    assert Task().model_id is None
    assert Task(model_id="premium-cloud").model_id == "premium-cloud"


# --- pick_specialist_model precedence --------------------------------------


def test_step_model_id_takes_precedence():
    deps = _deps(requested="cheap-local")
    with patch(
        "agent_utilities.core.model_factory.create_model", return_value="BUILT"
    ) as cm:
        out = pick_specialist_model(
            deps, "python_programmer", step_model_id="premium-cloud"
        )
    assert out == "BUILT"
    # built the Conductor-assigned model, not the per-turn requested one
    assert cm.call_args.kwargs["model_id"] == "gpt-4o"


def test_unknown_step_model_id_falls_back_to_requested():
    deps = _deps(requested="premium-cloud")
    with patch(
        "agent_utilities.core.model_factory.create_model", return_value="BUILT"
    ) as cm:
        out = pick_specialist_model(deps, "node", step_model_id="does-not-exist")
    assert out == "BUILT"
    # fell through to the requested-model override
    assert cm.call_args.kwargs["model_id"] == "gpt-4o"


def test_no_step_model_id_unchanged_behavior():
    deps = _deps(requested="cheap-local")
    with patch(
        "agent_utilities.core.model_factory.create_model", return_value="BUILT"
    ) as cm:
        out = pick_specialist_model(deps, "node")  # no step_model_id
    assert out == "BUILT"
    assert (
        cm.call_args.kwargs["model_id"] == "llama-3.2-3b-instruct"
    )  # requested override


def test_no_registry_returns_default():
    deps = MagicMock()
    deps.model_registry = None
    deps.agent_model = "DEFAULT"
    assert (
        pick_specialist_model(deps, "node", step_model_id="premium-cloud") == "DEFAULT"
    )
