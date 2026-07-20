"""Fail-closed GraphOS parent mediation for Langfuse graph ingestion."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
from fastmcp.exceptions import ToolError

from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    ScopeError,
    SessionRequiredError,
    suspend_session,
    use_session,
)
from agent_utilities.mcp.multiplexer import (
    _mediate_langfuse_kg_ingestion,
    attest_runtime_child_config,
)
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext


def _session(*scopes: str) -> GraphSession:
    actor = ActorContext(
        actor_id="subject:opaque:synthetic",
        actor_type=ActorType.SYSTEM,
        roles=(),
        tenant_id="tenant:opaque:synthetic",
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        scopes=frozenset(scopes),
        graph="graph:opaque:synthetic",
        audience="epistemic-graph",
        policy_version="policy:synthetic",
    )


def _config(enabled: bool = True) -> dict[str, object]:
    return attest_runtime_child_config(
        {
            "command": "synthetic-langfuse-child",
            "_graphos_parent_kg_ingestion": enabled,
            "env": {"LANGFUSE_KG_AUTO_INGEST": "false"},
        }
    )


def _result() -> SimpleNamespace:
    return SimpleNamespace(
        isError=False,
        structuredContent={"data": [{"id": "synthetic-trace"}]},
        content=[],
    )


def _install_fake_provider(monkeypatch, calls: list[tuple[str, object]]) -> None:
    package = ModuleType("langfuse_agent")
    package.__path__ = []  # type: ignore[attr-defined]
    module = ModuleType("langfuse_agent.kg_ingest")

    def ingest_read_result(action: str, payload: object) -> None:
        calls.append((action, payload))

    module.ingest_read_result = ingest_read_result  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse_agent", package)
    monkeypatch.setitem(sys.modules, "langfuse_agent.kg_ingest", module)


class _EnvelopeNodes:
    @staticmethod
    def properties(_node_id: str) -> None:
        return None

    @staticmethod
    def list() -> list[tuple[str, dict[str, object]]]:
        return []


class _EnvelopeChanges:
    def __init__(self) -> None:
        self.applied: list[dict[str, object]] = []
        self.records: dict[str, dict[str, object]] = {}
        self.versions: dict[str, dict[str, object]] = {}
        self.failures: list[Exception] = []

    def get(self, envelope_id: str) -> dict[str, object] | None:
        return self.records.get(envelope_id)

    def content_version(self, object_id: str) -> dict[str, object] | None:
        return self.versions.get(object_id)

    @staticmethod
    def cursor(_source: str, _partition: str = "") -> None:
        return None

    def apply(self, envelope: dict[str, object]) -> dict[str, object]:
        self.applied.append(envelope)
        if self.failures:
            raise self.failures.pop(0)
        envelope_id = str(envelope["envelope_id"])
        content_version = envelope["content_version"]
        assert isinstance(content_version, dict)
        self.records[envelope_id] = envelope
        self.versions[str(content_version["object_id"])] = content_version
        mutation = envelope["mutation"]
        assert isinstance(mutation, dict)
        return {
            "batch_id": mutation["batch_id"],
            "replayed": False,
            "projection_pending": False,
        }


class _EnvelopeRdf:
    @staticmethod
    def validate_shacl(_shapes: str, _data_graph: str) -> dict[str, object]:
        return {"conforms": True, "results": []}


class _EnvelopeClient:
    def __init__(self) -> None:
        self.nodes = _EnvelopeNodes()
        self.changes = _EnvelopeChanges()
        self.rdf = _EnvelopeRdf()

    @staticmethod
    def supports(operation: str) -> bool:
        return operation == "ApplyChangeEnvelope"


def test_parent_ingestion_uses_verified_write_session(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    _install_fake_provider(monkeypatch, calls)

    with use_session(_session("kg:write")):
        _mediate_langfuse_kg_ingestion(
            child_config=_config(),
            original_name="langfuse_observability",
            arguments={"action": "trace_list"},
            result=_result(),
        )

    assert calls == [("trace_list", {"data": [{"id": "synthetic-trace"}]})]


def test_parent_ingestion_live_writer_uses_change_envelope(monkeypatch) -> None:
    import agent_utilities.knowledge_graph.ingestion.envelope_ingest as envelope_ingest
    import agent_utilities.knowledge_graph.memory.native_ingest as native_ingest

    package = ModuleType("langfuse_agent")
    package.__path__ = []  # type: ignore[attr-defined]
    module = ModuleType("langfuse_agent.kg_ingest")

    write_results: list[dict[str, int]] = []

    def ingest_read_result(_action: str, payload: object) -> None:
        assert isinstance(payload, dict)
        records = payload["data"]
        assert isinstance(records, list)
        write_results.append(
            native_ingest.ingest_entities(
                [
                    {
                        "id": f"trace:opaque:synthetic:{index}",
                        "node_type": "Trace",
                    }
                    for index, _record in enumerate(records)
                ],
                source="langfuse-agent",
                domain="langfuse",
            )
        )

    module.ingest_read_result = ingest_read_result  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse_agent", package)
    monkeypatch.setitem(sys.modules, "langfuse_agent.kg_ingest", module)

    client = _EnvelopeClient()
    client.changes.failures.extend(
        RuntimeError(
            f"STALE_VERSION: graph 'graph:opaque:synthetic' expected version "
            f"{version - 1} but authoritative version is {version}"
        )
        for version in range(1, 7)
    )
    authority = SimpleNamespace(client=client)
    monkeypatch.setattr(native_ingest, "native_authority", lambda: authority)
    monkeypatch.setattr(envelope_ingest, "_native_occ_backoff", lambda _attempt: None)
    trace_result = SimpleNamespace(
        isError=False,
        structuredContent={
            "data": [
                {"id": "synthetic-trace-a"},
                {"id": "synthetic-trace-b"},
            ]
        },
        content=[],
    )

    with use_session(_session("kg:write")):
        for _delivery in range(2):
            _mediate_langfuse_kg_ingestion(
                child_config=_config(),
                original_name="langfuse_observability",
                arguments={"action": "trace_list"},
                result=trace_result,
            )

    assert write_results == [
        {"nodes": 2, "edges": 0},
        {"nodes": 2, "edges": 0},
    ]
    assert len(client.changes.applied) == 7
    assert {
        str(applied["mutation"]["idempotency_key"])
        for applied in client.changes.applied
    } == {str(client.changes.applied[0]["mutation"]["idempotency_key"])}
    assert [
        applied["mutation"]["expected_graph_version"]
        for applied in client.changes.applied
    ] == list(range(7))
    mutation = client.changes.applied[-1]["mutation"]
    assert isinstance(mutation, dict)
    operations = mutation["operations"]
    assert isinstance(operations, list)
    assert [operation["method"]["method"] for operation in operations] == [
        "AddNode",
        "AddNode",
    ]


def test_parent_ingestion_rejects_missing_session(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    _install_fake_provider(monkeypatch, calls)

    with suspend_session(), pytest.raises(SessionRequiredError):
        _mediate_langfuse_kg_ingestion(
            child_config=_config(),
            original_name="langfuse_observability",
            arguments={"action": "trace_list"},
            result=_result(),
        )

    assert calls == []


def test_parent_ingestion_rejects_read_only_session(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    _install_fake_provider(monkeypatch, calls)

    with use_session(_session("kg:read")), pytest.raises(ScopeError):
        _mediate_langfuse_kg_ingestion(
            child_config=_config(),
            original_name="langfuse_observability",
            arguments={"action": "trace_list"},
            result=_result(),
        )

    assert calls == []


def test_parent_ingestion_disabled_is_a_noop(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    _install_fake_provider(monkeypatch, calls)

    _mediate_langfuse_kg_ingestion(
        child_config=_config(False),
        original_name="langfuse_observability",
        arguments={"action": "trace_list"},
        result=_result(),
    )

    assert calls == []


def test_parent_ingestion_rejects_post_attestation_opt_in(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    _install_fake_provider(monkeypatch, calls)
    child_config = _config(False)
    child_config["_graphos_parent_kg_ingestion"] = True

    with (
        use_session(_session("kg:write")),
        pytest.raises(ToolError, match="not attested"),
    ):
        _mediate_langfuse_kg_ingestion(
            child_config=child_config,
            original_name="langfuse_observability",
            arguments={"action": "trace_list"},
            result=_result(),
        )

    assert calls == []
