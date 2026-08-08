"""D-CDX-70: RegistryPipeline must not retarget a verified GraphSession.

``IntelligencePipeline`` (aliased ``RegistryPipeline``) intentionally targets its
OWN shared ``graph_name`` (default ``"__commons__"``,
CONCEPT:AU-KG.query.vendor-agnostic-traversal) — a graph that legitimately
differs from whatever graph a CALLER's ambient verified ``GraphSession`` happens
to be scoped to (e.g. a live delegation's tenant graph). Before this fix, running
the pipeline under such a mismatched ambient session raised
``PermissionError("A graph-scoped view cannot retarget the verified
GraphSession")`` straight out of ``graph_compute.GraphComputeEngine._send`` and
aborted the whole registry scan — a real production failure observed during a
live post-rollout ServiceNow delegation, not a test-fixture artifact (that
variant was D-OTR-2's).

This module exercises the SAME session machinery production code uses
(``current_session``/``use_session``/``GraphSession.with_graph``) so the guard
being satisfied here is the real one, while stubbing out the heavy pipeline
internals (the engine client and ``PipelineRunner``) that would otherwise
require a running epistemic-graph engine.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

import agent_utilities.knowledge_graph.pipeline as pipeline_mod
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    current_session,
    use_session,
)
from agent_utilities.knowledge_graph.pipeline import IntelligencePipeline
from agent_utilities.models.knowledge_graph import PipelineConfig
from agent_utilities.security.brain_context import ActorContext, ActorType


def _delegation_session(graph: str) -> GraphSession:
    """A verified session scoped to a live delegation's tenant graph — NOT commons."""
    return GraphSession(
        actor=ActorContext(
            actor_id="servicenow-delegation",
            actor_type=ActorType.AUTOMATED_SERVICE,
            tenant_id="tenant-a",
            authenticated=True,
        ),
        tenant="tenant-a",
        scopes=frozenset({"kg:read", "kg:write"}),
        graph=graph,
    )


def _checking_runner_class(expected_graph: str) -> type:
    """A fake ``PipelineRunner`` that enforces the SAME contract ``_send`` does.

    Standing in for the real engine round trip: a fixed-graph client used
    under a mismatched ambient session raises exactly the production
    PermissionError. Records the graph it actually observed so the test can
    assert the retarget happened, not just that no exception was raised.
    """

    class _CheckingRunner:
        observed_graph: str | None = None

        def __init__(self, phases: list[Any]) -> None:
            del phases

        async def run(self, ctx: Any) -> dict[str, Any]:
            del ctx
            session = current_session()
            _CheckingRunner.observed_graph = session.graph if session else None
            if session is None or session.graph != expected_graph:
                raise PermissionError(
                    "A graph-scoped view cannot retarget the verified GraphSession"
                )
            return {}

    return _CheckingRunner


def _stub_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the real engine acquisition with a lightweight double.

    ``IntelligencePipeline.__init__`` calls ``GraphComputeEngine.get_or_create``
    directly (module-level import, not through ``current_session``), and
    ``PipelineContext.graph`` is a pydantic field typed ``GraphComputeEngine``
    with ``arbitrary_types_allowed`` — a plain stand-in object would fail
    validation, so the double must satisfy ``isinstance``, hence ``spec=``.
    """
    engine = MagicMock(spec=GraphComputeEngine)
    engine.node_ids.return_value = []
    engine.number_of_edges.return_value = 0
    monkeypatch.setattr(GraphComputeEngine, "get_or_create", lambda **_kw: engine)


@pytest.mark.asyncio
async def test_registry_pipeline_retargets_a_mismatched_ambient_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pipeline transitions into ITS OWN graph instead of crashing."""
    _stub_engine(monkeypatch)
    checking_runner = _checking_runner_class("__commons__")
    monkeypatch.setattr(pipeline_mod, "PipelineRunner", checking_runner)

    delegation_session = _delegation_session("tenant-a-delegation")
    pipeline = IntelligencePipeline(
        PipelineConfig(workspace_path="/tmp/does-not-matter"),
        backend=None,
        graph_name="__commons__",
    )

    outer_session = current_session()
    with use_session(delegation_session):
        metadata = await pipeline.run()

    assert metadata is not None
    # The runner observed the pipeline's OWN graph, not the caller's — proving
    # the transition actually happened rather than the guard being bypassed.
    assert checking_runner.observed_graph == "__commons__"
    # Whatever session was ambient before this test's own `use_session` block
    # must come back unchanged — not left retargeted at "__commons__".
    assert current_session() is outer_session


@pytest.mark.asyncio
async def test_registry_pipeline_restores_the_callers_session_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A phase failure inside the pipeline must not leak the retargeted session."""
    _stub_engine(monkeypatch)

    class _ExplodingRunner:
        def __init__(self, phases: list[Any]) -> None:
            del phases

        async def run(self, ctx: Any) -> dict[str, Any]:
            del ctx
            raise RuntimeError("synthetic phase failure")

    monkeypatch.setattr(pipeline_mod, "PipelineRunner", _ExplodingRunner)

    delegation_session = _delegation_session("tenant-a-delegation")
    pipeline = IntelligencePipeline(
        PipelineConfig(workspace_path="/tmp/does-not-matter"),
        backend=None,
        graph_name="__commons__",
    )

    outer_session = current_session()
    with use_session(delegation_session):
        with pytest.raises(RuntimeError, match="synthetic phase failure"):
            await pipeline.run()
        # Restored to the caller's OWN session inside the `with use_session`
        # block too — the retarget must not survive the raised exception.
        assert current_session() is delegation_session

    assert current_session() is outer_session


@pytest.mark.asyncio
async def test_registry_pipeline_is_a_noop_retarget_when_already_on_its_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No session swap is needed (or performed) when the ambient graph already matches."""
    _stub_engine(monkeypatch)
    checking_runner = _checking_runner_class("__commons__")
    monkeypatch.setattr(pipeline_mod, "PipelineRunner", checking_runner)

    same_graph_session = _delegation_session("__commons__")
    pipeline = IntelligencePipeline(
        PipelineConfig(workspace_path="/tmp/does-not-matter"),
        backend=None,
        graph_name="__commons__",
    )

    with use_session(same_graph_session):
        await pipeline.run()
        # Identity, not just equality: proves no `with_graph` copy was swapped in.
        assert current_session() is same_graph_session

    assert checking_runner.observed_graph == "__commons__"
