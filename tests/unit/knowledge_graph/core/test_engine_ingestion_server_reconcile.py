"""IngestionMixin.ingest_mcp_server reconciles :Server identity through the
engine-native RegisterServer RPC (CONCEPT:EG-KG.sharding.server-registry, W2.5) -- the SAME
mechanism a self-registering fleet server's own heartbeat uses
(``mcp/server_factory.py``) -- rather than being the sole raw-Cypher writer.
Falls back to the legacy ``MERGE`` only when the engine-native registry
client is unreachable through this backend.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from agent_utilities.knowledge_graph.core.engine_ingestion import (
    _SERVER_RECONCILE_TTL_SECS,
    IngestionMixin,
)


class _FakeServerRegistry:
    def __init__(self, *, raises: bool = False) -> None:
        self.calls: list[tuple[str, str, dict[str, Any] | None, int]] = []
        self._raises = raises

    def register(
        self,
        name: str,
        url: str,
        *,
        resources: dict[str, Any] | None = None,
        ttl_secs: int = 300,
    ) -> bool:
        if self._raises:
            raise RuntimeError("engine unreachable")
        self.calls.append((name, url, resources, ttl_secs))
        return True


class _FakeClient:
    def __init__(self, server_registry: Any) -> None:
        self.server_registry = server_registry


class _FakeGraph:
    def __init__(self, client: Any) -> None:
        self.client = client


class _FakeBackend:
    """Records Cypher ``execute`` calls (the legacy fallback path) and, when
    constructed with a registry, exposes the ``.graph.client.server_registry``
    chain ``_reconcile_server_registration`` walks (the primary path)."""

    def __init__(self, *, server_registry: Any = None) -> None:
        self.executed: list[tuple[str, dict[str, Any]]] = []
        self.graph = (
            _FakeGraph(_FakeClient(server_registry)) if server_registry else None
        )

    def execute(self, query: str, params: dict[str, Any]) -> list[Any]:
        self.executed.append((query, params))
        return []


class _Engine(IngestionMixin):
    """Minimal harness carrying only what ``ingest_mcp_server`` touches."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.hybrid_retriever = MagicMock(embed_model=None)
        self.graph = MagicMock()

    @staticmethod
    def _serialize_node(node: Any, label: str) -> dict[str, Any]:  # noqa: ARG004
        return node.model_dump()

    def _upsert_node(self, label: str, node_id: str, data: dict[str, Any]) -> None:
        pass


def _merges(backend: _FakeBackend) -> list[tuple[str, dict[str, Any]]]:
    return [(q, p) for q, p in backend.executed if "MERGE (s:Server" in q]


def test_ingest_mcp_server_reconciles_via_register_server_rpc() -> None:
    registry = _FakeServerRegistry()
    backend = _FakeBackend(server_registry=registry)
    engine = _Engine(backend)

    engine.ingest_mcp_server("graph-os", "mcp-ref://deadbeef", tools=[])

    assert len(registry.calls) == 1
    name, url, resources, ttl_secs = registry.calls[0]
    assert name == "graph-os"
    assert url  # a bounded reference, never asserted verbatim (opaque)
    assert isinstance(resources, dict)
    assert ttl_secs == _SERVER_RECONCILE_TTL_SECS
    # The RPC path succeeded -- the legacy Cypher MERGE must NOT also run.
    assert _merges(backend) == []


def test_ingest_mcp_server_falls_back_to_cypher_merge_without_engine_client() -> None:
    backend = _FakeBackend(server_registry=None)
    engine = _Engine(backend)

    engine.ingest_mcp_server("legacy-server", "mcp-ref://cafef00d", tools=[])

    merges = _merges(backend)
    assert len(merges) == 1
    _, params = merges[0]
    assert params["id"] == "srv:legacy-server"
    assert params["name"] == "legacy-server"


def test_ingest_mcp_server_falls_back_to_cypher_merge_on_rpc_failure() -> None:
    registry = _FakeServerRegistry(raises=True)
    backend = _FakeBackend(server_registry=registry)
    engine = _Engine(backend)

    engine.ingest_mcp_server("flaky-server", "mcp-ref://0badf00d", tools=[])

    assert len(registry.calls) == 0  # the attempt was made, but raised
    assert len(_merges(backend)) == 1


def test_ingest_mcp_server_does_nothing_without_a_backend() -> None:
    engine = _Engine(backend=None)
    # Must not raise even though there is no backend to reconcile through.
    engine.ingest_mcp_server("no-backend-server", "mcp-ref://0000", tools=[])
