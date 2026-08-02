"""Regression tests for D-CDX-41: graph-os must not accept/advertise an
"economy" delegation ``model_class`` and then fail 17s deep into
orchestration setup when no economy-tier model is actually configured.

Root cause: ``Orchestrator.execute_dynamic_workflow`` validated only that
``model_class`` was a KNOWN name (``"economy" in {"economy", "standard"}``)
early, but only checked whether a model was actually CONFIGURED for that
class deep inside a ``try`` block — after ``WorkflowStore.load_workflow``,
``GovernedDynamicWorkflow.from_graph_plan``, and
``workflow.build_upstream_capability`` had already run. A cost-conscious
caller explicitly requesting the cheaper class paid for the entire
orchestration setup before discovering "configured economy model class is
unavailable" — a cost-routing/configuration gap masquerading as an
orchestration failure.

The fix moves the availability check immediately after the syntactic
``model_class`` validation (before any workflow/capability setup), and adds
``agent_utilities.orchestration.agent_runner.available_model_classes()`` so
a caller/preflight can check availability BEFORE preview/execution instead
of only discovering it from a deep failure.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest


def _configure_chat_models(monkeypatch: pytest.MonkeyPatch, levels: list[str]) -> None:
    from agent_utilities.core.config import ChatModelConfig
    from agent_utilities.core.config import config as agent_config

    monkeypatch.setattr(
        agent_config,
        "chat_models",
        [
            ChatModelConfig(
                id=f"test-{level}-model-{i}",
                provider="openai",
                intelligence_level=level,
            )
            for i, level in enumerate(levels)
        ],
    )


def test_available_model_classes_reports_standard_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.orchestration.agent_runner import available_model_classes

    _configure_chat_models(monkeypatch, ["normal"])

    result = available_model_classes()
    assert result == {"economy": False, "standard": True}


def test_available_model_classes_reports_both_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.orchestration.agent_runner import available_model_classes

    _configure_chat_models(monkeypatch, ["light", "normal"])

    result = available_model_classes()
    assert result == {"economy": True, "standard": True}


def test_available_model_classes_reports_neither_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_utilities.orchestration.agent_runner import available_model_classes

    _configure_chat_models(monkeypatch, [])

    result = available_model_classes()
    assert result == {"economy": False, "standard": False}


@pytest.mark.asyncio
async def test_unavailable_model_class_fails_before_workflow_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact reported bug: model_class="economy" is a syntactically
    valid name but no economy-tier model is configured. Must fail with the
    documented RuntimeError, and — the actual regression — must do so
    WITHOUT ever calling WorkflowStore.load_workflow (proving the check now
    runs before the expensive orchestration setup, not 17s into it)."""
    from agent_utilities.knowledge_graph import workflow_store
    from agent_utilities.orchestration.manager import Orchestrator

    _configure_chat_models(monkeypatch, ["normal"])  # economy NOT configured

    engine: Any = SimpleNamespace()
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.engine = engine

    # ``load_workflow`` is invoked via ``asyncio.to_thread`` (it is a SYNC
    # method), so the spy must be a plain sync callable, not an AsyncMock.
    from unittest.mock import Mock

    load_workflow_spy = Mock()
    monkeypatch.setattr(workflow_store.WorkflowStore, "load_workflow", load_workflow_spy)

    with pytest.raises(RuntimeError, match="configured economy model class is unavailable"):
        await orchestrator.execute_dynamic_workflow(
            "some-workflow",
            model_class="economy",
        )

    load_workflow_spy.assert_not_called()


@pytest.mark.asyncio
async def test_available_model_class_still_reaches_workflow_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model_class that IS configured must still proceed into workflow
    resolution as before — the fail-fast check must not become a false
    positive that blocks legitimate standard-tier requests."""
    from agent_utilities.knowledge_graph import workflow_store
    from agent_utilities.orchestration.manager import Orchestrator

    _configure_chat_models(monkeypatch, ["normal"])

    engine: Any = SimpleNamespace()
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.engine = engine

    # ``load_workflow`` is invoked via ``asyncio.to_thread`` (it is a SYNC
    # method), so the spy must be a plain sync callable, not an AsyncMock.
    from unittest.mock import Mock

    load_workflow_spy = Mock(return_value=None)
    monkeypatch.setattr(workflow_store.WorkflowStore, "load_workflow", load_workflow_spy)

    with pytest.raises(ValueError, match="not found in KG or catalog"):
        await orchestrator.execute_dynamic_workflow(
            "some-workflow",
            model_class="standard",
        )

    load_workflow_spy.assert_called_once()


@pytest.mark.asyncio
async def test_preresolved_orchestrator_model_bypasses_the_availability_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the caller supplies its own ``orchestrator_model``, model_class
    availability is never consulted (existing bypass semantics, preserved by
    the fix) — matches ``test_manager_fallback_is_only_for_upstream_unavailability``
    in test_governed_dynamic_workflow.py, which relies on exactly this."""
    from agent_utilities.knowledge_graph import workflow_store
    from agent_utilities.models.graph import GraphPlan
    from agent_utilities.models.sdd import Task
    from agent_utilities.orchestration.manager import Orchestrator

    _configure_chat_models(monkeypatch, [])  # NOTHING configured

    engine: Any = SimpleNamespace()
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.engine = engine

    monkeypatch.setattr(
        workflow_store.WorkflowStore,
        "load_workflow",
        lambda _self, _name: GraphPlan(steps=[Task(id="reviewer", description="review")]),
    )

    from unittest.mock import MagicMock

    fake_model = MagicMock()

    from agent_utilities.capabilities import governed_dynamic_workflow as gdw_module

    monkeypatch.setattr(
        gdw_module.GovernedDynamicWorkflow,
        "build_upstream_capability",
        lambda self, _orchestrator: None,
    )
    monkeypatch.setattr(
        gdw_module.GovernedDynamicWorkflow,
        "execute",
        AsyncMock(
            side_effect=RuntimeError("stop here — this test only checks the preflight bypass")
        ),
    )

    with pytest.raises(RuntimeError, match="stop here"):
        await orchestrator.execute_dynamic_workflow(
            "review",
            model_class="economy",
            orchestrator_model=fake_model,
        )
