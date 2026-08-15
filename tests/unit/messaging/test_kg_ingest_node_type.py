"""Regression tests for the messaging KG-ingestion retired `type` node property.

The engine retired the `type` node PROPERTY in favor of `node_type`
(`EpistemicGraphBackend.add_node` / `GraphComputeEngine.add_node` both raise the
one canonical ``retired_node_type_property_error()`` on a stray `type` key).
Messaging KG
ingestion (`agent_utilities.messaging.kg_ingest.ingest_message_to_kg`) calls
`engine.store_memory(...)` -- the shared node-builder/helper defined in
`agent_utilities.knowledge_graph.core.engine_memory.MemoryMixin` -- which built
its node dict from a `MemoryNode`, a `RegistryNode` subclass whose own Pydantic
`type` field (`RegistryNodeType.MEMORY`) was leaking straight through
`IntelligenceGraphEngine._serialize_node()` into the property dict handed to
`_upsert_node`/`add_node`, causing every live message to fail KG ingestion.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.messaging import kg_ingest
from agent_utilities.messaging.models import EventType, InboundEvent, Message
from agent_utilities.models.knowledge_graph import retired_node_type_property_error


class _RecordingNativeBackend:
    """A minimal native-typed backend recording exactly what ``_upsert_node``'s
    typed path hands it -- mirrors ``_PolicyRecordingNative`` in
    ``tests/unit/knowledge_graph/test_native_typed_ingestion.py``, the
    established fixture for this exact seam (real ``EpistemicGraphBackend``/
    ``GraphComputeEngine`` both raise on a stray ``type`` key here).
    """

    typed_mutation_support = "native"
    cypher_support = "native"

    def __init__(self) -> None:
        self.nodes: list[tuple[str, dict[str, Any]]] = []

    def add_node(self, node_id: str, **properties: Any) -> None:
        if "type" in properties:
            # Mirrors EpistemicGraphBackend.add_node / GraphComputeEngine.add_node's
            # own fail-closed guard -- a regression here must fail this test the
            # same way it fails against the real engine, not just get recorded.
            raise retired_node_type_property_error()
        self.nodes.append((node_id, dict(properties)))


def _bare_engine(backend: Any) -> IntelligenceGraphEngine:
    """A bare ``IntelligenceGraphEngine``-shaped instance (bypasses ``__init__``,
    mirroring ``tests/unit/knowledge_graph/test_native_typed_ingestion.py``'s own
    ``_engine()`` helper) with just enough state for ``store_memory``'s inline
    (``_local=True``) path: the given (native-typed) backend and a no-op embedder.
    """
    engine = IntelligenceGraphEngine.__new__(IntelligenceGraphEngine)
    engine.backend = backend
    engine.hybrid_retriever = type("_NoEmbedder", (), {"embed_model": None})()
    return engine


def test_store_memory_emits_node_type_not_the_retired_type_property() -> None:
    """``store_memory`` -- the shared helper messaging's ``kg_ingest.py`` calls via
    ``engine.store_memory(...)`` -- must reach the native engine backend with a
    property dict using ``node_type``, never the retired ``type`` key, for a
    ``MemoryNode`` (whose own Pydantic ``type`` field is ``RegistryNodeType.MEMORY``).
    Exercises the REAL ``_upsert_node``/``_serialize_node`` (not a stub) so this
    proves the actual production write path, not just an intermediate contract.
    """
    backend = _RecordingNativeBackend()
    engine = _bare_engine(backend)

    memory_id = engine.store_memory(
        content="hello from messaging",
        memory_type="episodic",
        name="Chat: someone on slack",
        tags=["platform:slack", "channel:general", "messaging", "conversation"],
        trust_score=0.7,
        agent_id="messaging_router",
        _local=True,
    )

    assert len(backend.nodes) == 1
    node_id, props = backend.nodes[0]
    assert node_id == memory_id
    assert "type" not in props, f"retired 'type' property leaked through: {props}"
    # RegistryNode.to_graph_properties() writes node_type as the enum's own
    # .value; RegistryNodeType.MEMORY = "memory" (every RegistryNodeType member
    # is lowercase snake_case, not Title-Case) -- assert the real projected
    # value rather than a capitalized literal.
    assert props["node_type"] == "memory"


@pytest.mark.asyncio
async def test_ingest_message_to_kg_delegates_to_store_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ingest_message_to_kg`` (the actual messaging entry point from the bug
    report) forwards to ``engine.store_memory`` -- confirming the call chain
    the property-dict regression above exercises is the one messaging really
    uses, end to end from an inbound event.
    """
    calls: list[dict[str, Any]] = []

    class _StubEngine:
        def store_memory(self, **kwargs: Any) -> str:
            calls.append(kwargs)
            return "mem:stub"

    event = InboundEvent(
        event_type=EventType.MESSAGE,
        platform="slack",
        channel_id="C1",
        user_id="u1",
        user_name="alice",
        message=Message(id="m1", author_id="u1", channel_id="C1", content="hi there"),
    )

    memory_id = await kg_ingest.ingest_message_to_kg(
        event, knowledge_engine=_StubEngine()
    )

    assert memory_id == "mem:stub"
    assert len(calls) == 1
    assert calls[0]["memory_type"] == "episodic"
    assert "hi there" in calls[0]["content"]
