"""Shared native connector-ingestion primitive — fakes coverage.

Exercises ``ingest_entities`` / ``ingest_documents`` with a fake engine client (no
engine required). CONCEPT:AU-KG.ingest.enterprise-source-extractor.
"""

from __future__ import annotations

from typing import Any

import msgpack
import pytest

from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.knowledge_graph.memory.native_ingest import (
    NativeIngestError,
    enrich_pending_documents,
    ingest_documents,
    ingest_entities,
)
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor


@pytest.fixture(autouse=True)
def _governed_session():
    actor = ActorContext(
        actor_id="subject:opaque:synthetic",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=(),
        tenant_id="tenant:opaque:synthetic",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        scopes=frozenset({"kg:write"}),
        graph="graph:opaque:synthetic",
        policy_version="policy:opaque:synthetic",
        audience="epistemic-graph",
    )
    with use_actor(actor), use_session(session):
        yield


class _FakeNodes:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, Any]] = {}

    def properties(self, node_id: str) -> dict[str, Any] | None:
        return self.values.get(node_id)

    def list(self) -> list[tuple[str, dict[str, Any]]]:
        return list(self.values.items())


class _FakeChanges:
    def __init__(self, nodes: _FakeNodes) -> None:
        self.nodes = nodes
        self.edges: list[tuple[str, str, dict[str, Any]]] = []
        self.applied: list[dict[str, Any]] = []
        self.records: dict[str, dict[str, Any]] = {}
        self.versions: dict[str, dict[str, Any]] = {}

    def get(self, envelope_id: str) -> dict[str, Any] | None:
        return self.records.get(envelope_id)

    def content_version(self, object_id: str) -> dict[str, Any] | None:
        return self.versions.get(object_id)

    def cursor(self, _source: str, _partition: str = "") -> None:
        return None

    def apply(self, envelope: dict[str, Any]) -> dict[str, Any]:
        self.applied.append(envelope)
        mutation = envelope["mutation"]
        for operation in mutation["operations"]:
            method = operation["method"]
            params = method["params"]
            properties = msgpack.unpackb(params["properties_msgpack"], raw=False)
            if method["method"] == "AddNode":
                self.nodes.values[params["node_id"]] = properties
            elif method["method"] == "AddEdge":
                self.edges.append(
                    (params["source_id"], params["target_id"], properties)
                )
        version = envelope["content_version"]
        self.versions[version["object_id"]] = version
        self.records[envelope["envelope_id"]] = envelope
        return {
            "batch_id": mutation["batch_id"],
            "replayed": False,
            "projection_pending": False,
        }


class _FakeRdf:
    def validate_shacl(self, _shapes: str, _data_graph: str) -> dict[str, Any]:
        return {"conforms": True, "results": []}


class _FakeClient:
    def __init__(self) -> None:
        self.nodes = _FakeNodes()
        self.changes = _FakeChanges(self.nodes)
        self.rdf = _FakeRdf()

    @staticmethod
    def supports(operation: str) -> bool:
        return operation == "ApplyChangeEnvelope"


class _RawTxnOnlyClient:
    def __init__(self) -> None:
        self.txn = object()


def test_ingest_entities_writes_typed_nodes_and_stamps_provenance():
    c = _FakeClient()
    res = ingest_entities(
        [{"id": "erpnext:so:1", "node_type": "SalesOrder", "total": 10}],
        [
            {
                "source": "erpnext:so:1",
                "target": "erpnext:cust:1",
                "relationship": "orderedBy",
            }
        ],
        source="erpnext-agent",
        domain="erpnext",
        client=c,
    )
    assert res == {"nodes": 1, "edges": 1}
    n = c.nodes.values["erpnext:so:1"]
    assert n["node_type"] == "SalesOrder"
    assert n["source"] == "erpnext-agent" and n["domain"] == "erpnext"
    assert c.changes.edges[0] == (
        "erpnext:so:1",
        "erpnext:cust:1",
        {"relationship": "orderedBy"},
    )


def test_ingest_documents_writes_document_nodes():
    c = _FakeClient()
    res = ingest_documents(
        [{"id": "sn:kb:1", "text": "reset vpn", "title": "VPN", "source_uri": "u"}],
        source="servicenow-api",
        domain="servicenow",
        client=c,
    )
    assert res == {"nodes": 1, "edges": 0}
    d = c.nodes.values["sn:kb:1"]
    assert d["node_type"] == "Document"
    assert d["text"] == "reset vpn"
    assert d["title"] == "VPN"
    assert d["source"] == "servicenow-api"


def test_ingest_documents_commits_canonical_relationships_with_documents():
    client = _FakeClient()
    result = ingest_documents(
        [{"id": "mm:post:1", "text": "hello"}],
        [
            {
                "source": "mm:post:1",
                "target": "mm:channel:1",
                "relationship": "postedInChannel",
            }
        ],
        source="mattermost-mcp",
        domain="mattermost",
        client=client,
    )

    assert result == {"nodes": 1, "edges": 1}
    assert len(client.changes.applied) == 1
    assert client.changes.edges == [
        ("mm:post:1", "mm:channel:1", {"relationship": "postedInChannel"})
    ]


def test_ingest_documents_rejects_retired_edge_alias_before_transaction():
    client = _FakeClient()
    with pytest.raises(NativeIngestError, match="canonical relationship"):
        ingest_documents(
            [{"id": "doc:1", "text": "hello"}],
            [{"source": "doc:1", "target": "topic:1", "type": "about"}],
            source="source",
            domain="domain",
            client=client,
        )
    assert client.changes.applied == []


def test_documents_without_text_are_skipped():
    c = _FakeClient()
    with pytest.raises(NativeIngestError, match="at least one document"):
        ingest_documents([{"id": "x"}], source="s", domain="d", client=c)


def test_engine_failure_is_not_acknowledged(monkeypatch):
    # Simulate no reachable engine so the test is deterministic regardless of
    # whether a live engine happens to be reachable from the test host.
    import agent_utilities.knowledge_graph.memory.native_ingest as ni

    def unavailable():
        raise NativeIngestError("unavailable")

    monkeypatch.setattr(ni, "native_authority", unavailable)
    with pytest.raises(NativeIngestError, match="unavailable"):
        ingest_entities([{"id": "a", "node_type": "T"}], source="s", domain="d")
    with pytest.raises(NativeIngestError, match="unavailable"):
        ingest_documents([{"id": "a", "text": "t"}], source="s", domain="d")


def test_empty_is_rejected():
    with pytest.raises(NativeIngestError, match="at least one entity"):
        ingest_entities([], source="s", domain="d", client=_FakeClient())


def test_retired_structural_aliases_are_rejected_before_transaction():
    client = _FakeClient()
    with pytest.raises(NativeIngestError, match="canonical node_type"):
        ingest_entities(
            [{"id": "a", "type": "RetiredAlias"}],
            source="s",
            domain="d",
            client=client,
        )
    with pytest.raises(NativeIngestError, match="canonical relationship"):
        ingest_entities(
            [{"id": "a", "node_type": "Thing"}],
            [{"source": "a", "target": "b", "type": "RetiredAlias"}],
            source="s",
            domain="d",
            client=client,
        )
    assert client.changes.applied == []


def test_raw_transaction_only_client_is_rejected() -> None:
    with pytest.raises(NativeIngestError, match="support ChangeEnvelope"):
        ingest_entities(
            [{"id": "a", "node_type": "Thing"}],
            source="source",
            domain="domain",
            client=_RawTxnOnlyClient(),
        )


def test_ingest_documents_stamps_provenance_and_needs_enrichment_flag():
    """CONCEPT:AU-KG.enrichment.topic-classification-topology — every connector-written Document is
    flagged for the hub-side catch-up sweep and carries full provenance."""
    c = _FakeClient()
    res = ingest_documents(
        [{"id": "searxng:result:1", "text": "some search result body", "title": "R1"}],
        source="searxng-mcp",
        domain="searxng",
        client=c,
    )
    assert res == {"nodes": 1, "edges": 0}
    d = c.nodes.values["searxng:result:1"]
    assert d["needs_enrichment"] is True
    assert "created_at" not in d
    assert "fetched_at" not in d
    assert d["backend"] == "native_ingest"
    assert d["content_hash"]


def test_identical_document_redelivery_applies_one_change_envelope() -> None:
    client = _FakeClient()
    document = {"id": "source:document:opaque", "text": "stable content"}

    first = ingest_documents(
        [document], source="source", domain="domain", client=client
    )
    second = ingest_documents(
        [document], source="source", domain="domain", client=client
    )

    assert first == second == {"nodes": 1, "edges": 0}
    assert len(client.changes.applied) == 1


def test_source_document_timestamps_are_preserved() -> None:
    client = _FakeClient()
    ingest_documents(
        [
            {
                "id": "source:document:timestamped",
                "text": "stable content",
                "created_at": "2026-01-02T03:04:05Z",
                "fetched_at": "2026-01-02T03:05:06Z",
            }
        ],
        source="source",
        domain="domain",
        client=client,
    )

    node = client.nodes.values["source:document:timestamped"]
    assert node["created_at"] == "2026-01-02T03:04:05Z"
    assert node["fetched_at"] == "2026-01-02T03:05:06Z"


class _FakeBackend:
    def __init__(self) -> None:
        self.add_node_calls: list[tuple[str, dict[str, Any]]] = []

    def add_node(self, node_id: str, **props: Any) -> None:
        self.add_node_calls.append((node_id, props))


class _FakeEngineForSweep:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.backend = _FakeBackend()

    def query_cypher(self, _q: str) -> list[dict[str, Any]]:
        return self._rows


@pytest.mark.asyncio
async def test_enrich_pending_documents_processes_and_clears_flag(monkeypatch):
    rows = [
        {"d": {"id": "doc:a", "text": "hello world", "title": "A"}},
        {"d": {"id": "doc:b", "text": "goodbye world", "title": "B"}},
    ]
    engine = _FakeEngineForSweep(rows)

    processed: list[str] = []
    processed_metadata: list[dict[str, Any]] = []
    enriched_calls: list[str] = []

    class _FakeProcessor:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            pass

        def process(self, _text: str, *, document_id: str, **_kw: Any) -> None:
            processed.append(document_id)
            processed_metadata.append(dict(_kw.get("metadata") or {}))

    class _FakeIngestionEngine:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            pass

        async def _enrich_text(
            self, source_id: str, _text: str, _source_type: str, _title: str = ""
        ) -> dict[str, int]:
            enriched_calls.append(source_id)
            return {"concepts": 1, "facts": 0, "topics": 1}

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ontology.document_processing.DocumentProcessor",
        _FakeProcessor,
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.engine.IngestionEngine",
        _FakeIngestionEngine,
    )

    result = await enrich_pending_documents(engine, limit=10)

    assert result == {"scanned": 2, "enriched": 2, "failed": 0}
    assert processed == ["doc:a", "doc:b"]
    assert processed_metadata == [
        {"needs_enrichment": False},
        {"needs_enrichment": False},
    ]
    assert enriched_calls == ["doc:a", "doc:b"]


@pytest.mark.asyncio
async def test_enrich_pending_documents_one_bad_doc_does_not_abort_sweep(monkeypatch):
    rows = [
        {"d": {"id": "doc:bad", "text": "x"}},
        {"d": {"id": "doc:good", "text": "y"}},
    ]
    engine = _FakeEngineForSweep(rows)

    class _FailingProcessor:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            pass

        def process(self, _text: str, *, document_id: str, **_kw: Any) -> None:
            if document_id == "doc:bad":
                raise RuntimeError("boom")

    class _FakeIngestionEngine:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            pass

        async def _enrich_text(self, *_a: Any, **_kw: Any) -> dict[str, int]:
            return {"concepts": 0, "facts": 0, "topics": 0}

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ontology.document_processing.DocumentProcessor",
        _FailingProcessor,
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.engine.IngestionEngine",
        _FakeIngestionEngine,
    )

    result = await enrich_pending_documents(engine, limit=10)
    assert result == {"scanned": 2, "enriched": 1, "failed": 1}


@pytest.mark.asyncio
async def test_enrich_pending_documents_no_engine_capability_returns_zeroes():
    result = await enrich_pending_documents(object(), limit=10)
    assert result == {"scanned": 0, "enriched": 0, "failed": 0}


@pytest.mark.asyncio
async def test_enrich_pending_documents_empty_result_is_cheap_noop():
    engine = _FakeEngineForSweep([])
    result = await enrich_pending_documents(engine, limit=10)
    assert result == {"scanned": 0, "enriched": 0, "failed": 0}
