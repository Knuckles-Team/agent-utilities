from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from agent_utilities.knowledge_graph.backends.base import GraphBackend
from agent_utilities.knowledge_graph.core.owl_bridge import OWLBridge


class _RecordingBackend(GraphBackend):
    """Conformant test backend that records only the write surface under test."""

    typed_mutation_support: str = ""
    cypher_support: str = "full"
    read_only: bool = False

    def __init__(self) -> None:
        self.cypher_calls: list[tuple[str, dict[str, Any] | None]] = []

    def execute(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        include_epistemic: bool = False,
    ) -> list[dict[str, Any]]:
        self.cypher_calls.append((query, params))
        return []

    def execute_batch(
        self, query: str, batch: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return []

    def create_schema(self) -> None:
        pass

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        pass

    def semantic_search(
        self, query_embedding: list[float], n_results: int = 5
    ) -> list[dict[str, Any]]:
        return []

    def prune(self, criteria: dict[str, Any]) -> None:
        pass

    def close(self) -> None:
        pass


class _NativeBackend(_RecordingBackend):
    typed_mutation_support = "native"
    cypher_support = "native"
    edges: list[tuple[str, str, dict[str, Any]]]

    def add_edge(self, source: str, target: str, /, **properties: Any) -> None:
        self.edges.append((source, target, dict(properties)))


class _MissingNativeEdgeBackend(_RecordingBackend):
    typed_mutation_support = "native"
    cypher_support = "native"


class _NonCallableNativeEdgeBackend(_RecordingBackend):
    typed_mutation_support = "native"
    cypher_support = "native"
    add_edge: None = None


def test_backfeed_uses_typed_edge_for_native_backend() -> None:
    """Native backfeed must avoid the unsupported comma-pattern edge write."""
    backend = _NativeBackend()
    backend.edges = []
    bridge = OWLBridge(
        graph=SimpleNamespace(nodes=()), owl_backend=None, backend=backend
    )
    engine = SimpleNamespace(backend=object())

    assert (
        bridge._backfeed_one_inference(
            engine,
            {
                "subject": "source:1",
                "predicate": "dependsOn",
                "object": "target:1",
                "inference_type": "transitive_closure",
            },
        )
        is True
    )

    assert backend.edges == [
        (
            "source:1",
            "target:1",
            {
                "inferred": True,
                "inferred_from": "owl_reasoner",
                "inference_type": "transitive_closure",
                "relationship": "DEPENDSON",
            },
        )
    ]
    assert backend.cypher_calls == []


def test_backfeed_operational_failure_is_warned_and_skipped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Operational adapter failures stay best-effort but remain diagnosable."""
    bridge = OWLBridge(
        graph=SimpleNamespace(nodes=()), owl_backend=None, backend=object()
    )

    def fail_write(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("backend unavailable")

    monkeypatch.setattr(bridge, "_write_backfed_edge", fail_write)
    with caplog.at_level(logging.WARNING):
        assert (
            bridge._backfeed_one_inference(
                SimpleNamespace(backend=object()),
                {
                    "subject": "source:1",
                    "predicate": "dependsOn",
                    "object": "target:1",
                },
            )
            is False
        )

    assert "Inferred-edge backfeed failed" in caplog.text
    assert "backend unavailable" in caplog.text


@pytest.mark.parametrize("cypher_support", ["full", "subset"])
def test_backfeed_uses_documented_cypher_backend(
    cypher_support: str,
) -> None:
    """Only the documented rich backend tiers may receive the fallback write."""
    backend = _RecordingBackend()
    backend.cypher_support = cypher_support
    bridge = OWLBridge(
        graph=SimpleNamespace(nodes=()), owl_backend=None, backend=backend
    )

    assert (
        bridge._backfeed_one_inference(
            SimpleNamespace(backend=object()),
            {
                "subject": "source:1",
                "predicate": "dependsOn",
                "object": "target:1",
            },
        )
        is True
    )

    assert len(backend.cypher_calls) == 1
    query, params = backend.cypher_calls[0]
    assert "MERGE (s)-[r:DEPENDSON]->(t)" in query
    assert params == {"sid": "source:1", "tid": "target:1"}


def test_backfeed_requires_an_explicit_backend_on_normal_path() -> None:
    """A missing backend must fail before capability lookup or writes."""
    bridge = OWLBridge(graph=SimpleNamespace(nodes=()), owl_backend=None)

    with pytest.raises(
        RuntimeError, match="graph backend required for inferred-edge backfeed"
    ):
        bridge._backfeed_one_inference(
            SimpleNamespace(backend=object()),
            {
                "subject": "source:1",
                "predicate": "dependsOn",
                "object": "target:1",
            },
        )


@pytest.mark.parametrize(
    "backend",
    [_MissingNativeEdgeBackend(), _NonCallableNativeEdgeBackend()],
    ids=["missing", "non-callable"],
)
def test_native_backfeed_requires_callable_typed_edge_writer(
    backend: _RecordingBackend,
) -> None:
    """Native authority without add_edge must fail closed, never use execute."""
    bridge = OWLBridge(
        graph=SimpleNamespace(nodes=()), owl_backend=None, backend=backend
    )

    with pytest.raises(
        RuntimeError,
        match="native graph authority does not expose typed edge mutations",
    ):
        bridge._backfeed_one_inference(
            SimpleNamespace(backend=object()),
            {
                "subject": "source:1",
                "predicate": "dependsOn",
                "object": "target:1",
            },
        )
    assert backend.cypher_calls == []


@pytest.mark.parametrize(
    ("cypher_support", "read_only", "message"),
    [
        ("none", False, "graph backend lacks lossless inferred-edge mutations"),
        ("unknown", False, "graph backend lacks lossless inferred-edge mutations"),
        (
            "full",
            True,
            "read-only graph backend cannot persist inferred-edge backfeed",
        ),
    ],
)
def test_backfeed_rejects_unsupported_or_read_only_backend(
    cypher_support: str, read_only: bool, message: str
) -> None:
    """Unsupported and read-only authorities must never receive an edge write."""
    backend = _RecordingBackend()
    backend.cypher_support = cypher_support
    backend.read_only = read_only
    bridge = OWLBridge(
        graph=SimpleNamespace(nodes=()), owl_backend=None, backend=backend
    )

    with pytest.raises(RuntimeError, match=message):
        bridge._backfeed_one_inference(
            SimpleNamespace(backend=object()),
            {
                "subject": "source:1",
                "predicate": "dependsOn",
                "object": "target:1",
            },
        )
    assert backend.cypher_calls == []
