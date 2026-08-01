"""Scope-aware HistorySource over eg's RunTrace/ToolCall provenance (PA-R1.8, Track 5).

CONCEPT:AU-KG.history.scoped-conversation-search — :class:`ScopedEgHistorySource`
implements the native ``pydantic_ai_harness.conversation_search.HistorySource`` protocol
bound to one ``GraphSession`` + one ``root_run_id``, so ``search_conversation_history``
reaches only runs at or below the calling run's own position in the ``parent_run_id``
delegation tree — the access-control gap upstream's own docs name (its own
``scope: 'all' | 'conversation'`` toggle is binary, not hierarchical).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent_utilities.capabilities.eg_history_source import (
    EgStepStore,
    ScopedEgHistorySource,
    build_conversation_search_capability,
)
from agent_utilities.knowledge_graph.core.session import GraphSession, ScopeError
from agent_utilities.observability.trace_ontology import (
    TRACE_NODE_LABEL,
    TRACE_USED_TOOL_EDGE,
    trace_id,
    trace_properties,
)
from agent_utilities.security.brain_context import ActorContext, ActorType


class FakeEngine:
    """Minimal ``add_node``/``link_nodes``/``query_cypher`` double.

    Extends the shape ``tests/unit/capabilities/test_kg_audit_sink.py::FakeEngine``
    already established with the one additional query shape this module's closure
    computation needs: an unfiltered ``MATCH (t:RunTrace) RETURN t.run_id,
    t.parent_run_id`` sweep. Kept local per that file's own precedent ("kept local to
    avoid touching a shared fixture file").
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

    def link_nodes(self, source_id: str, target_id: str, rel_type: str, **_kw: Any) -> None:
        self.edges.append((source_id, target_id, rel_type))

    def by_type(self, node_type: str) -> list[dict[str, Any]]:
        return [p for nid, p in self.nodes.items() if self.node_types.get(nid) == node_type]

    def query_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        params = params or {}
        # Unfiltered run-tree sweep (ScopedEgHistorySource._run_tree_rows).
        if "RETURN t.run_id AS run_id, t.parent_run_id AS parent_run_id" in query:
            return [
                {
                    "run_id": props.get("run_id"),
                    "parent_run_id": props.get("parent_run_id"),
                }
                for nid, props in self.nodes.items()
                if self.node_types.get(nid) == TRACE_NODE_LABEL
            ]
        # Scoped run listing (ScopedEgHistorySource.list_runs): WHERE t.run_id IN $ids.
        if "WHERE t.run_id IN $ids" in query:
            ids = set(params.get("ids") or ())
            return [
                {
                    "run_id": props.get("run_id"),
                    "conversation_id": props.get("conversation_id"),
                    "parent_run_id": props.get("parent_run_id"),
                    "agent_name": props.get("agent_name"),
                    "timestamp": props.get("timestamp"),
                }
                for nid, props in self.nodes.items()
                if self.node_types.get(nid) == TRACE_NODE_LABEL
                and props.get("run_id") in ids
            ]
        # ToolCall-by-trace-node lookup (tool_call_rows_by_trace_node_id).
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
        # Single-node :RunTrace lookup (KgAuditSink.get_run).
        if "RunTrace" in query:
            props = self.nodes.get(tid)
            if props is None or self.node_types.get(tid) != TRACE_NODE_LABEL:
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


def _seed_run_trace(engine: FakeEngine, run_id: str, *, parent_run_id: str | None = None) -> None:
    """Write a ``:RunTrace`` node the way ``agent_runner``/``KgAuditSink`` already do."""
    node_id = trace_id(run_id)
    props = trace_properties(
        run_id=run_id,
        agent_name="tester",
        task="t",
        status="completed",
        timestamp="2026-07-31T00:00:00Z",
    )
    if parent_run_id:
        props["parent_run_id"] = parent_run_id  # stored RAW, matching kg_audit_sink writers
    engine.add_node(node_id, TRACE_NODE_LABEL, properties=props)


def _make_session(*, scopes: frozenset[str] = frozenset({"kg:read", "kg:write"})) -> GraphSession:
    actor = ActorContext(
        actor_id="tester",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("test",),
        tenant_id="tenant-a",
        authenticated=True,
    )
    return GraphSession(actor=actor, tenant="tenant-a", scopes=scopes, graph="tenant-a-graph")


def _seed_tree(engine: FakeEngine) -> None:
    """root -> child -> grandchild, plus an unrelated sibling tree root2 -> child2."""
    _seed_run_trace(engine, "root")
    _seed_run_trace(engine, "child", parent_run_id="root")
    _seed_run_trace(engine, "grandchild", parent_run_id="child")
    _seed_run_trace(engine, "root2")
    _seed_run_trace(engine, "child2", parent_run_id="root2")


# ---------------------------------------------------------------------------
# Fail-closed construction
# ---------------------------------------------------------------------------


def test_construction_fails_closed_without_kg_read_scope(engine: FakeEngine) -> None:
    # `kg:write` alone would satisfy `require_scope("kg:read")` under the coarse-grant
    # lattice (`GraphSession.require_scope`: kg:write -> {kg:write, kg:admin} both
    # accepted), so use a wholly unrelated scope to force the denial this test checks.
    session = _make_session(scopes=frozenset({"unrelated:scope"}))
    with pytest.raises(ScopeError):
        ScopedEgHistorySource(engine=engine, session=session, root_run_id="root")


def test_construction_fails_closed_without_root_run_id(engine: FakeEngine) -> None:
    session = _make_session()
    with pytest.raises(ValueError):
        ScopedEgHistorySource(engine=engine, session=session, root_run_id="")


# ---------------------------------------------------------------------------
# Rank-scoped list_runs: at-or-below-root only, never siblings/ancestors
# ---------------------------------------------------------------------------


async def test_list_runs_from_root_sees_whole_owned_subtree(engine: FakeEngine) -> None:
    _seed_tree(engine)
    session = _make_session()
    source = ScopedEgHistorySource(engine=engine, session=session, root_run_id="root")

    runs = await source.list_runs()
    seen = {r.run_id for r in runs}
    assert seen == {trace_id("root"), trace_id("child"), trace_id("grandchild")}


async def test_list_runs_from_subagent_sees_only_its_own_descendants(
    engine: FakeEngine,
) -> None:
    """The exact gap the article names: a sub-agent's search must not reach its
    parent's other business or a sibling subtree — only at or below itself."""
    _seed_tree(engine)
    session = _make_session()
    source = ScopedEgHistorySource(engine=engine, session=session, root_run_id="child")

    runs = await source.list_runs()
    seen = {r.run_id for r in runs}
    assert seen == {trace_id("child"), trace_id("grandchild")}
    assert trace_id("root") not in seen
    assert trace_id("root2") not in seen
    assert trace_id("child2") not in seen


async def test_list_runs_never_crosses_into_an_unrelated_tree(engine: FakeEngine) -> None:
    _seed_tree(engine)
    session = _make_session()
    source = ScopedEgHistorySource(engine=engine, session=session, root_run_id="root2")

    runs = await source.list_runs()
    seen = {r.run_id for r in runs}
    assert seen == {trace_id("root2"), trace_id("child2")}


async def test_list_runs_engine_unavailable_returns_empty_not_everything(
    engine: FakeEngine,
) -> None:
    _seed_tree(engine)
    session = _make_session()
    source = ScopedEgHistorySource(engine=None, session=session, root_run_id="root")
    assert await source.list_runs() == []


# ---------------------------------------------------------------------------
# run_history: content only for in-scope runs; THE scope-enforcement test
# ---------------------------------------------------------------------------


async def _seed_tool_call(
    engine: FakeEngine, run_id: str, *, tool_name: str, result: str
) -> None:
    from agent_utilities.capabilities.kg_audit_sink import KgAuditSink, ToolCallRecord

    sink = KgAuditSink(engine=engine)
    await sink.record_tool_call(
        ToolCallRecord(
            run_id=run_id,
            tool_call_id=f"call-{run_id}",
            tool_name=tool_name,
            arguments="{}",
            result=result,
        )
    )


async def test_run_history_in_scope_returns_reconstructed_messages(
    engine: FakeEngine,
) -> None:
    _seed_tree(engine)
    await _seed_tool_call(engine, "child", tool_name="search_docs", result="the secret plan is X")
    session = _make_session()
    source = ScopedEgHistorySource(engine=engine, session=session, root_run_id="root")

    messages = await source.run_history(run_id=trace_id("child"))
    assert messages, "expected reconstructed messages for an in-scope run"
    # Content survives through to the ToolReturnPart the BM25 toolset indexes.
    rendered = str(messages)
    assert "search_docs" in rendered
    assert "the secret plan is X" in rendered


async def test_run_history_out_of_scope_returns_nothing_THE_SCOPE_ENFORCEMENT_TEST(
    engine: FakeEngine,
) -> None:
    """The load-bearing test: a session scoped to `child`'s subtree must never recover
    `root2`'s (an unrelated tree's) content, even though it physically exists in the
    same store and `run_history` is a public protocol method a caller could invoke
    directly with any run_id. This is the property that FAILS if scoping is removed —
    delete the `if run_id not in allowed: return []` guard in
    `ScopedEgHistorySource.run_history` and this test fails."""
    _seed_tree(engine)
    await _seed_tool_call(engine, "root2", tool_name="secret_tool", result="root2's private data")
    session = _make_session()
    # Scoped to `child`'s subtree -- root2 is a sibling tree, never below `child`.
    source = ScopedEgHistorySource(engine=engine, session=session, root_run_id="child")

    messages = await source.run_history(run_id=trace_id("root2"))
    assert messages == []
    rendered = str(messages)
    assert "root2's private data" not in rendered


async def test_run_history_unknown_run_id_returns_nothing(engine: FakeEngine) -> None:
    _seed_tree(engine)
    session = _make_session()
    source = ScopedEgHistorySource(engine=engine, session=session, root_run_id="root")
    assert await source.run_history(run_id="trace:does-not-exist") == []


# ---------------------------------------------------------------------------
# End-to-end through the REAL native ConversationSearchToolset (no forking)
# ---------------------------------------------------------------------------


async def test_native_conversation_search_tool_respects_the_scope_boundary(
    engine: FakeEngine,
) -> None:
    """Wire the UNMODIFIED upstream `ConversationSearch` capability to our scoped
    source and call its actual `search_conversation_history` tool function directly
    -- proving the scoping holds through the real BM25 tool contract, not just our
    own adapter's unit-level methods."""
    _seed_tree(engine)
    await _seed_tool_call(
        engine, "child", tool_name="search_docs", result="findable child secret alpha"
    )
    await _seed_tool_call(
        engine, "root2", tool_name="other_tool", result="findable root2 secret bravo"
    )
    session = _make_session()

    capability = build_conversation_search_capability(
        engine, session, root_run_id="child", scope="all"
    )
    toolset = capability.get_toolset()
    assert toolset is not None

    ctx = SimpleNamespace(conversation_id=None)
    result_text = await toolset.search_conversation_history(ctx, query="secret")

    assert "alpha" in result_text
    assert "bravo" not in result_text
    assert "root2" not in result_text


# ---------------------------------------------------------------------------
# EgStepStore -- partial facade, loud about what it does not support
# ---------------------------------------------------------------------------


async def test_step_store_register_and_get_run_round_trips(engine: FakeEngine) -> None:
    from pydantic_ai_harness.step_persistence import RunRecord

    session = _make_session()
    store = EgStepStore(engine, session)
    await store.register_run(RunRecord(run_id="run-x", agent_name="tester"))
    got = await store.get_run(run_id="run-x")
    assert got is not None
    assert got.agent_name == "tester"


def test_step_store_requires_kg_write_scope(engine: FakeEngine) -> None:
    session = _make_session(scopes=frozenset({"unrelated:scope"}))
    with pytest.raises(ScopeError):
        EgStepStore(engine, session)


async def test_step_store_snapshot_methods_fail_loud_not_silent(engine: FakeEngine) -> None:
    session = _make_session()
    store = EgStepStore(engine, session)
    with pytest.raises(NotImplementedError):
        await store.save_snapshot(SimpleNamespace())
    with pytest.raises(NotImplementedError):
        await store.latest_snapshot(run_id="run-x")
    with pytest.raises(NotImplementedError):
        await store.record_tool_effect(SimpleNamespace())


async def test_step_store_list_runs_stays_unimplemented_rather_than_unscoped(
    engine: FakeEngine,
) -> None:
    """A write-side StepStore has no root_run_id to bound a sweep by -- it must not
    grow into a second, unscoped enumeration path next to ScopedEgHistorySource."""
    session = _make_session()
    store = EgStepStore(engine, session)
    with pytest.raises(NotImplementedError):
        await store.list_runs()
