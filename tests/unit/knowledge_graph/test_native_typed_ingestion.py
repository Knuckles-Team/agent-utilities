"""Lossless native mutation routing used by GraphOS capability ingestion."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent_utilities.knowledge_graph.backends.brain_guarded_backend import (
    BrainGuardedBackend,
)
from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)
from agent_utilities.knowledge_graph.backends.fanout_backend import FanOutBackend
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor


class _PolicyRecordingNative:
    """A governance-wrapper-shaped backend that records its public seam."""

    typed_mutation_support = "native"
    cypher_support = "native"

    def __init__(self) -> None:
        self.nodes: list[tuple[str, dict[str, Any]]] = []
        self.edges: list[tuple[str, str, dict[str, Any]]] = []
        self.cypher_calls = 0

    def add_node(self, node_id: str, **properties: Any) -> None:
        self.nodes.append((node_id, dict(properties)))

    def add_edge(self, source_id: str, target_id: str, /, **properties: Any) -> None:
        self.edges.append((source_id, target_id, dict(properties)))

    def execute(self, _query: str, _params: dict[str, Any]) -> list[dict[str, Any]]:
        self.cypher_calls += 1
        return []


class _UnmarkedNative(_PolicyRecordingNative):
    typed_mutation_support = ""


def _engine(backend: Any) -> IntelligenceGraphEngine:
    engine = IntelligenceGraphEngine.__new__(IntelligenceGraphEngine)
    engine.backend = backend
    return engine


def test_native_capability_ingestion_uses_outer_typed_policy_seam() -> None:
    backend = _PolicyRecordingNative()
    engine = _engine(backend)

    engine._upsert_node(
        "Skill",
        "skill:synthetic",
        {
            "id": "skill:synthetic",
            "name": "synthetic",
            "synonyms": ["synthetic", "example"],
            "metadata": {"runnable": True},
        },
    )
    engine._upsert_edge(
        "skill:synthetic",
        "resource:skill:synthetic",
        "BINDS_RUNNABLE",
        {"evidence": ["installed", "verified"]},
    )

    assert backend.cypher_calls == 0
    assert backend.nodes == [
        (
            "skill:synthetic",
            {
                "id": "skill:synthetic",
                "name": "synthetic",
                "synonyms": ["synthetic", "example"],
                "metadata": {"runnable": True},
                "node_type": "Skill",
            },
        )
    ]
    assert backend.edges == [
        (
            "skill:synthetic",
            "resource:skill:synthetic",
            {
                "evidence": ["installed", "verified"],
                "relationship": "BINDS_RUNNABLE",
            },
        )
    ]


@pytest.mark.parametrize("operation", ["node", "edge"])
def test_unmarked_native_wrapper_fails_closed(operation: str) -> None:
    backend = _UnmarkedNative()
    engine = _engine(backend)

    with pytest.raises(RuntimeError, match="did not declare lossless typed mutations"):
        if operation == "node":
            engine._upsert_node("Skill", "skill:x", {"id": "skill:x"})
        else:
            engine._upsert_edge("skill:x", "resource:x", "BINDS_RUNNABLE", {})

    assert backend.nodes == []
    assert backend.edges == []
    assert backend.cypher_calls == 0


def test_epistemic_graph_explicitly_declares_typed_mutations() -> None:
    backend = EpistemicGraphBackend.__new__(EpistemicGraphBackend)
    assert backend.typed_mutation_support == "native"


def test_graph_compute_declares_and_executes_the_typed_node_contract() -> None:
    writes: list[tuple[str, dict[str, Any]]] = []
    backend = GraphComputeEngine.__new__(GraphComputeEngine)
    backend._client = SimpleNamespace(
        nodes=SimpleNamespace(
            add=lambda node_id, properties: writes.append((node_id, properties))
        )
    )
    engine = _engine(backend)

    engine._upsert_node(
        "Skill",
        "skill:synthetic",
        {"name": "synthetic", "metadata": {"runnable": True}},
    )

    assert backend.typed_mutation_support == "native"
    assert backend.cypher_support == "native"
    guard = BrainGuardedBackend.__new__(BrainGuardedBackend)
    guard._inner = backend
    assert guard.typed_mutation_support == "native"
    assert guard.cypher_support == "native"
    assert writes == [
        (
            "skill:synthetic",
            {
                "id": "skill:synthetic",
                "name": "synthetic",
                "metadata": {"runnable": True},
                "node_type": "Skill",
            },
        )
    ]


def test_governed_typed_edge_keeps_source_as_a_property() -> None:
    backend = _PolicyRecordingNative()
    guard = BrainGuardedBackend.__new__(BrainGuardedBackend)
    guard._inner = backend
    engine = _engine(guard)
    actor = ActorContext(
        actor_id="subject:opaque:synthetic",
        actor_type=ActorType.SYSTEM,
        roles=(),
        tenant_id="tenant:opaque:synthetic",
        authenticated=True,
    )

    with use_actor(actor):
        engine._upsert_edge(
            "skill:synthetic",
            "resource:skill:synthetic",
            "BINDS_RUNNABLE",
            {"source": "system"},
        )

    assert backend.edges[0][0:2] == (
        "skill:synthetic",
        "resource:skill:synthetic",
    )
    assert backend.edges[0][2]["source"] == "system"
    assert backend.edges[0][2]["relationship"] == "BINDS_RUNNABLE"


def test_epistemic_graph_typed_node_is_an_atomic_field_upsert() -> None:
    graph = SimpleNamespace()
    operations: list[list[dict[str, Any]]] = []
    graph.batch_update = lambda batch: operations.append(batch)
    backend = EpistemicGraphBackend.__new__(EpistemicGraphBackend)
    backend._graph = graph

    backend.add_node(
        "skill:synthetic",
        node_type="Skill",
        synonyms=["synthetic", "example"],
        metadata={"runnable": True},
    )

    assert operations == [
        [
            {
                "op": "upsert_node",
                "id": "skill:synthetic",
                "properties": {
                    "id": "skill:synthetic",
                    "node_type": "Skill",
                    "synonyms": ["synthetic", "example"],
                    "metadata": {"runnable": True},
                },
            }
        ]
    ]


def test_epistemic_graph_typed_edge_is_an_idempotent_upsert() -> None:
    graph = SimpleNamespace()
    operations: list[list[dict[str, Any]]] = []
    graph.batch_update = lambda batch: operations.append(batch)
    backend = EpistemicGraphBackend.__new__(EpistemicGraphBackend)
    backend._graph = graph

    backend.add_edge(
        "skill:synthetic",
        "resource:skill:synthetic",
        relationship="BINDS_RUNNABLE",
        evidence=["installed", "verified"],
    )

    assert operations == [
        [
            {
                "op": "upsert_edge",
                "source": "skill:synthetic",
                "target": "resource:skill:synthetic",
                "properties": {
                    "relationship": "BINDS_RUNNABLE",
                    "evidence": ["installed", "verified"],
                },
            }
        ]
    ]


def test_fanout_keeps_typed_node_and_edge_mutations_structured() -> None:
    authority = _PolicyRecordingNative()
    fanout = FanOutBackend.__new__(FanOutBackend)
    fanout._authority = authority
    fanout._authority_writes = 0
    queued: list[tuple[str, dict[str, Any]]] = []
    fanout._enqueue = lambda operation, payload: queued.append((operation, payload))

    fanout.add_node(
        "skill:synthetic",
        node_type="Skill",
        synonyms=["synthetic", "example"],
    )
    fanout.add_edge(
        "skill:synthetic",
        "resource:skill:synthetic",
        relationship="BINDS_RUNNABLE",
        evidence=["installed", "verified"],
    )

    assert fanout.typed_mutation_support == "native"
    assert fanout._authority_writes == 2
    assert [operation for operation, _payload in queued] == [
        "upsert_node",
        "upsert_edge",
    ]
    assert queued[0][1]["properties"]["synonyms"] == ["synthetic", "example"]
    assert queued[1][1]["props"]["evidence"] == ["installed", "verified"]

    writer = SimpleNamespace()
    replayed: list[tuple[str, str, dict[str, Any]]] = []
    writer._upsert_node = lambda label, node_id, properties: replayed.append(
        (label, node_id, properties)
    )
    fanout._node_writer = lambda _backend: writer
    fanout._apply(
        object(),
        SimpleNamespace(op=queued[0][0], payload=queued[0][1]),
    )
    assert replayed[0][0:2] == ("Skill", "skill:synthetic")
    assert replayed[0][2]["synonyms"] == ["synthetic", "example"]
