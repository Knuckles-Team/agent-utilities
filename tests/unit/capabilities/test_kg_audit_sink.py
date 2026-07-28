"""KG-native AuditSink adoption (PA-R1.7).

CONCEPT:AU-KG.audit.kg-native-audit-sink — :class:`KgAuditSink` writes/reads the
canonical ``:RunTrace``/``:ToolCall`` provenance (``observability.trace_ontology``), and
:class:`AuditLog` wires it into a live agent run through the composition seam
(``capabilities/composition.py``).
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.capabilities.kg_audit_sink import (
    AuditLog,
    KgAuditSink,
    RunAuditRecord,
    ToolCallRecord,
    _default_kg_sink_resolver,
    identity_redactor,
)
from agent_utilities.core.contextual_model import create_context_agent
from agent_utilities.observability.trace_ontology import TRACE_USED_TOOL_EDGE, trace_id


class FakeEngine:
    """Minimal ``add_node``/``link_nodes``/``query_cypher`` double.

    Mirrors ``tests/unit/fleet_autonomy_fakes.py``'s ``FakeEngine`` shape
    (``ActionPolicy``'s own test double) so the two owned modules' tests are
    consistent, kept local to avoid touching a shared fixture file.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.node_types: dict[str, str] = {}
        self.edges: list[tuple[str, str, str]] = []

    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict[str, Any] | None = None,
        ephemeral: bool = False,
        **_kw: Any,
    ) -> None:
        self.nodes[node_id] = dict(properties or {})
        self.node_types[node_id] = node_type

    def link_nodes(
        self, source_id: str, target_id: str, rel_type: str, **_kw: Any
    ) -> None:
        self.edges.append((source_id, target_id, rel_type))

    def by_type(self, node_type: str) -> list[dict[str, Any]]:
        return [
            props
            for nid, props in self.nodes.items()
            if self.node_types.get(nid) == node_type
        ]

    def query_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        params = params or {}
        tid = params.get("tid")
        if "ToolCall" in query and "RunTrace" in query:
            rows = []
            for src, dst, rel in self.edges:
                if (
                    rel == TRACE_USED_TOOL_EDGE
                    and src == tid
                    and self.node_types.get(dst) == "ToolCall"
                ):
                    props = self.nodes[dst]
                    rows.append(
                        {
                            "tool_call_id": props.get("tool_call_id"),
                            "tool_name": props.get("tool_name"),
                            "audit_arguments": props.get("audit_arguments"),
                            "audit_result": props.get("audit_result"),
                            "audit_error": props.get("audit_error"),
                            "timestamp": props.get("timestamp"),
                            "agent_name": props.get("agent_name"),
                            "conversation_id": props.get("conversation_id"),
                            "parent_run_id": props.get("parent_run_id"),
                            "sequence": props.get("sequence"),
                        }
                    )
            rows.sort(key=lambda r: r.get("sequence") or 0)
            return rows
        if "RunTrace" in query:
            props = self.nodes.get(tid)
            if props is None or self.node_types.get(tid) != "RunTrace":
                return []
            return [
                {
                    "status": props.get("status"),
                    "audit_error": props.get("audit_error"),
                    "input_tokens": props.get("input_tokens"),
                    "output_tokens": props.get("output_tokens"),
                    "total_tokens": props.get("total_tokens"),
                    "conversation_id": props.get("conversation_id"),
                    "parent_run_id": props.get("parent_run_id"),
                    "agent_name": props.get("agent_name"),
                    "timestamp": props.get("timestamp"),
                }
            ]
        return []


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


# ---------------------------------------------------------------------------
# KgAuditSink -- direct sink behavior
# ---------------------------------------------------------------------------


async def test_record_tool_call_writes_toolcall_node_linked_to_runtrace(
    engine: FakeEngine,
) -> None:
    sink = KgAuditSink(engine=engine)
    record = ToolCallRecord(
        run_id="run-1",
        tool_call_id="call-abc",
        tool_name="search_docs",
        arguments='{"q": "hello"}',
        result="3 hits",
    )
    await sink.record_tool_call(record)

    tool_calls = engine.by_type("ToolCall")
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool_call_id"] == "call-abc"
    assert tool_calls[0]["tool_name"] == "search_docs"
    assert tool_calls[0]["audit_arguments"] == '{"q": "hello"}'
    assert tool_calls[0]["audit_result"] == "3 hits"
    assert tool_calls[0]["status"] == "ok"
    assert any(
        e[0] == trace_id("run-1") and e[2] == TRACE_USED_TOOL_EDGE for e in engine.edges
    )


async def test_record_tool_call_error_sets_status_error(engine: FakeEngine) -> None:
    sink = KgAuditSink(engine=engine)
    await sink.record_tool_call(
        ToolCallRecord(
            run_id="run-1b",
            tool_call_id="call-err",
            tool_name="flaky",
            arguments="{}",
            error="boom",
        )
    )
    tool_calls = engine.by_type("ToolCall")
    assert tool_calls[0]["status"] == "error"
    assert tool_calls[0]["audit_error"] == "boom"


async def test_record_run_writes_runtrace_node(engine: FakeEngine) -> None:
    sink = KgAuditSink(engine=engine)
    record = RunAuditRecord(
        run_id="run-2",
        outcome="completed",
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
    )
    await sink.record_run(record)

    runs = engine.by_type("RunTrace")
    assert len(runs) == 1
    assert runs[0]["status"] == "completed"
    assert runs[0]["input_tokens"] == 10
    assert runs[0]["output_tokens"] == 5
    assert runs[0]["total_tokens"] == 15


async def test_record_run_failed_sets_status_failed(engine: FakeEngine) -> None:
    sink = KgAuditSink(engine=engine)
    await sink.record_run(
        RunAuditRecord(run_id="run-3", outcome="failed", error="boom")
    )
    runs = engine.by_type("RunTrace")
    assert runs[0]["status"] == "failed"
    assert runs[0]["audit_error"] == "boom"


async def test_list_tool_calls_round_trips_in_sequence_order(
    engine: FakeEngine,
) -> None:
    sink = KgAuditSink(engine=engine)
    await sink.record_tool_call(
        ToolCallRecord(
            run_id="run-4",
            tool_call_id="c1",
            tool_name="a",
            arguments="{}",
            result="r1",
        )
    )
    await sink.record_tool_call(
        ToolCallRecord(
            run_id="run-4",
            tool_call_id="c2",
            tool_name="b",
            arguments="{}",
            result="r2",
        )
    )

    calls = await sink.list_tool_calls(run_id="run-4")
    assert [c.tool_call_id for c in calls] == ["c1", "c2"]
    assert [c.tool_name for c in calls] == ["a", "b"]
    assert calls[0].result == "r1"
    assert calls[1].result == "r2"


async def test_get_run_returns_recorded_outcome(engine: FakeEngine) -> None:
    sink = KgAuditSink(engine=engine)
    await sink.record_run(
        RunAuditRecord(run_id="run-5", outcome="completed", total_tokens=42)
    )
    got = await sink.get_run(run_id="run-5")
    assert got is not None
    assert got.outcome == "completed"
    assert got.total_tokens == 42


async def test_get_run_missing_returns_none(engine: FakeEngine) -> None:
    sink = KgAuditSink(engine=engine)
    assert await sink.get_run(run_id="does-not-exist") is None


async def test_no_engine_is_a_safe_no_op() -> None:
    sink = KgAuditSink(engine=None)
    await sink.record_tool_call(
        ToolCallRecord(run_id="r", tool_call_id="c", tool_name="t", arguments="{}")
    )
    await sink.record_run(RunAuditRecord(run_id="r", outcome="completed"))
    assert await sink.list_tool_calls(run_id="r") == []
    assert await sink.get_run(run_id="r") is None


async def test_redactor_masks_audit_fields_but_not_the_canonical_ones(
    engine: FakeEngine,
) -> None:
    """Redaction stays a pluggable, no-opinionated-default hook (de-opinionation
    principle): the caller's redactor governs only this sink's ``audit_*`` fields,
    never the already-existing, always-on ``PersistencePrivacyGuard`` blanking
    ``trace_ontology.tool_call_properties`` applies to the canonical ``args`` field."""

    def redactor(field_name: str, value: object) -> object:
        return "***" if field_name == "arguments" else value

    sink = KgAuditSink(engine=engine, redactor=redactor)
    await sink.record_tool_call(
        ToolCallRecord(
            run_id="run-6", tool_call_id="c", tool_name="t", arguments="secret"
        )
    )
    tool_calls = engine.by_type("ToolCall")
    assert tool_calls[0]["audit_arguments"] == "***"
    assert tool_calls[0]["args"] == ""  # canonical field already blanked+digested


def test_identity_redactor_is_passthrough() -> None:
    assert identity_redactor("x", "y") == "y"


def test_default_kg_sink_resolver_reads_ctx_deps_graph_engine(
    engine: FakeEngine,
) -> None:
    class _Deps:
        def __init__(self, graph_engine: Any) -> None:
            self.graph_engine = graph_engine

    class _Ctx:
        def __init__(self, deps: Any) -> None:
            self.deps = deps

    sink = _default_kg_sink_resolver(_Ctx(deps=_Deps(graph_engine=engine)))
    assert isinstance(sink, KgAuditSink)
    assert sink._engine is engine


def test_default_kg_sink_resolver_no_engine_on_deps() -> None:
    class _Deps:
        pass

    class _Ctx:
        def __init__(self, deps: Any) -> None:
            self.deps = deps

    sink = _default_kg_sink_resolver(_Ctx(deps=_Deps()))
    assert isinstance(sink, KgAuditSink)
    assert sink._engine is None


# ---------------------------------------------------------------------------
# AuditLog capability -- live-path wiring through a real agent run
# ---------------------------------------------------------------------------


async def test_audit_log_capability_records_a_live_run(engine: FakeEngine) -> None:
    """Wire-First: the capability must actually fire during a real ``Agent.run``,
    not merely expose hooks that unit-test in isolation."""
    from pydantic_ai.models.test import TestModel

    sink = KgAuditSink(engine=engine)
    agent = create_context_agent(
        TestModel(),
        capabilities=[AuditLog(sink=sink, sink_resolver=None, agent_name="tester")],
    )

    @agent.tool_plain
    def ping() -> str:
        return "pong"

    result = await agent.run("say hi")
    assert result.output is not None

    tool_calls = engine.by_type("ToolCall")
    assert any(tc["tool_name"] == "ping" for tc in tool_calls)
    runs = engine.by_type("RunTrace")
    assert len(runs) == 1
    assert runs[0]["status"] == "completed"
    assert runs[0]["agent_name"] == "tester"


async def test_audit_log_capability_uses_default_kg_sink_resolver(
    engine: FakeEngine,
) -> None:
    """With no explicit sink, the default resolver reads ``ctx.deps.graph_engine`` --
    exactly the composition-seam wiring (``capabilities/composition.py``)."""
    from dataclasses import dataclass

    from pydantic_ai.models.test import TestModel

    @dataclass
    class Deps:
        graph_engine: Any = None

    agent = create_context_agent(
        TestModel(), deps_type=Deps, capabilities=[AuditLog(agent_name="t2")]
    )

    @agent.tool_plain
    def ping() -> str:
        return "pong"

    await agent.run("say hi", deps=Deps(graph_engine=engine))

    assert engine.by_type("ToolCall")
    assert engine.by_type("RunTrace")


async def test_audit_log_capability_no_engine_is_a_no_op() -> None:
    """No ``graph_engine`` on deps -> the default resolver's sink no-ops; the run
    itself is unaffected (matches every other default reliability capability)."""
    from pydantic_ai.models.test import TestModel

    agent = create_context_agent(TestModel(), capabilities=[AuditLog()])

    @agent.tool_plain
    def ping() -> str:
        return "pong"

    result = await agent.run("say hi")
    assert result.output is not None


# ---------------------------------------------------------------------------
# capabilities/composition.py -- kg_audit is wired in and default ON
# ---------------------------------------------------------------------------


def test_default_runtime_capabilities_includes_audit_log_by_default() -> None:
    from agent_utilities.capabilities.composition import default_runtime_capabilities

    defaults = default_runtime_capabilities()
    assert any(isinstance(cap, AuditLog) for cap in defaults)


def test_default_runtime_capabilities_kg_audit_false_excludes_it() -> None:
    from agent_utilities.capabilities.composition import default_runtime_capabilities

    defaults = default_runtime_capabilities(kg_audit=False)
    assert not any(isinstance(cap, AuditLog) for cap in defaults)


def test_merge_capabilities_does_not_double_add_a_caller_supplied_audit_log() -> None:
    from agent_utilities.capabilities.composition import (
        default_runtime_capabilities,
        merge_capabilities,
    )

    custom = AuditLog(agent_name="caller-supplied")
    merged = merge_capabilities([custom], default_runtime_capabilities())
    audit_logs = [cap for cap in merged if isinstance(cap, AuditLog)]
    assert audit_logs == [custom]
