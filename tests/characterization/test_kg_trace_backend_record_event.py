"""Characterization tests for KGTraceBackend.record_event (CCN 35),
agent_utilities/harness/trace_backend.py.

Targets branches not already covered by tests/harness/test_kg_trace_backend.py:
llm-kind generation node cost/token rollup, root-trace persistence + the
on_trace_complete hook (fired + exception-swallowed + not-callable), the
error status flip, and best-effort persistence exception swallowing on both
the trace-node and span/generation-node writes.

Pins OBSERVED behaviour before a complexity-reduction refactor. Must stay
byte-identical across the refactor commit.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from agent_utilities.harness.trace_backend import KGTraceBackend


class _FakeKG:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple[str, str, str]] = []

    def add_node(self, node_id, node_type, properties=None) -> None:
        self.nodes[node_id] = {"node_type": node_type, **(properties or {})}

    def link_nodes(self, src, dst, rel, **_kw) -> None:
        self.edges.append((src, dst, str(rel)))


class _RaisingKG:
    def add_node(self, node_id, node_type, properties=None) -> None:
        raise RuntimeError("kg is down")

    def link_nodes(self, src, dst, rel, **_kw) -> None:
        raise RuntimeError("kg is down")


def test_root_event_persists_trace_node_and_sets_latency_and_content():
    kg = _FakeKG()
    be = KGTraceBackend(backend=kg)

    be.record_event(
        trace_id="t1",
        span_id="root1",
        name="root_op",
        is_root=True,
        latency_ms=12.5,
        input_text="hello input",
        output_text="hello output",
        tags=["a", "b"],
    )

    assert "t1" in kg.nodes
    entry = be.get_trace("t1")
    assert entry["trace"].latency_ms == 12.5
    assert entry["trace"].input == "hello input"
    assert entry["trace"].output == "hello output"


def test_root_event_with_error_sets_status_error():
    be = KGTraceBackend()
    be.record_event(
        trace_id="t2", span_id="root2", name="root_op", is_root=True, error="boom"
    )
    entry = be.get_trace("t2")
    assert entry["trace"].status == "error"


def test_root_event_fires_on_trace_complete_hook():
    be = KGTraceBackend()
    hook = MagicMock()
    be.on_trace_complete = hook

    be.record_event(trace_id="t3", span_id="root3", name="root_op", is_root=True)

    hook.assert_called_once_with("t3")


def test_root_event_swallows_on_trace_complete_hook_exception():
    be = KGTraceBackend()
    hook = MagicMock(side_effect=RuntimeError("hook broke"))
    be.on_trace_complete = hook

    # Must not raise.
    be.record_event(trace_id="t4", span_id="root4", name="root_op", is_root=True)
    hook.assert_called_once()


def test_root_event_with_non_callable_hook_is_skipped_silently():
    be = KGTraceBackend()
    be.on_trace_complete = "not callable"

    # Must not raise.
    be.record_event(trace_id="t5", span_id="root5", name="root_op", is_root=True)


def test_root_event_returns_before_creating_a_span_or_generation_node():
    be = KGTraceBackend()
    be.record_event(trace_id="t6", span_id="root6", name="root_op", is_root=True)
    entry = be.get_trace("t6")
    assert entry["spans"] == []
    assert entry["generations"] == []


def test_llm_kind_child_creates_generation_node_with_cost_and_token_rollup():
    be = KGTraceBackend()
    be.record_event(trace_id="t7", span_id="root7", name="root", is_root=True)
    be.record_event(
        trace_id="t7",
        span_id="gen7",
        name="llm_call",
        is_root=False,
        kind="llm",
        parent_span_id="root7",
        model="gpt-4o",
        provider="openai",
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=10,
        cache_write_tokens=5,
    )
    entry = be.get_trace("t7")
    assert len(entry["generations"]) == 1
    gen = entry["generations"][0]
    assert gen.model == "gpt-4o"
    assert gen.input_tokens == 100
    trace = entry["trace"]
    assert trace.input_tokens == 100
    assert trace.output_tokens == 50
    assert trace.cache_read_tokens == 10
    assert trace.cache_write_tokens == 5


def test_non_llm_kind_child_creates_span_node_not_generation():
    be = KGTraceBackend()
    be.record_event(trace_id="t8", span_id="root8", name="root", is_root=True)
    be.record_event(
        trace_id="t8",
        span_id="span8",
        name="retrieve",
        is_root=False,
        kind="retrieval",
        parent_span_id="root8",
    )
    entry = be.get_trace("t8")
    assert len(entry["spans"]) == 1
    assert entry["generations"] == []


def test_child_event_persists_and_links_to_parent_span():
    kg = _FakeKG()
    be = KGTraceBackend(backend=kg)
    be.record_event(trace_id="t9", span_id="root9", name="root", is_root=True)
    be.record_event(
        trace_id="t9",
        span_id="child9",
        name="op",
        is_root=False,
        kind="general",
        parent_span_id="root9",
    )
    assert "child9" in kg.nodes
    assert ("root9", "child9", "RegistryEdgeType.HAS_SPAN") in kg.edges or any(
        e[0] == "root9" and e[1] == "child9" for e in kg.edges
    )


def test_child_event_without_parent_links_to_trace_id():
    kg = _FakeKG()
    be = KGTraceBackend(backend=kg)
    be.record_event(trace_id="t10", span_id="root10", name="root", is_root=True)
    be.record_event(
        trace_id="t10", span_id="child10", name="op", is_root=False, kind="general"
    )
    assert any(e[0] == "t10" and e[1] == "child10" for e in kg.edges)


def test_trace_persist_exception_is_swallowed():
    be = KGTraceBackend(backend=_RaisingKG())
    # Must not raise even though backend.add_node always raises.
    be.record_event(trace_id="t11", span_id="root11", name="root", is_root=True)
    entry = be.get_trace("t11")
    assert entry is not None  # in-memory mirror still updated


def test_child_persist_exception_is_swallowed():
    be = KGTraceBackend(backend=_RaisingKG())
    be.record_event(trace_id="t12", span_id="root12", name="root", is_root=True)
    # Must not raise even though backend.add_node/link_nodes always raise.
    be.record_event(
        trace_id="t12", span_id="child12", name="op", is_root=False, kind="general"
    )
    entry = be.get_trace("t12")
    assert len(entry["spans"]) == 1  # in-memory mirror still updated


def test_no_backend_configured_still_updates_in_memory_mirror():
    be = KGTraceBackend(backend=None)
    be.record_event(trace_id="t13", span_id="root13", name="root", is_root=True)
    be.record_event(
        trace_id="t13", span_id="child13", name="op", is_root=False, kind="llm"
    )
    entry = be.get_trace("t13")
    assert len(entry["generations"]) == 1


def test_backend_without_add_node_attribute_is_not_persisted():
    class _NoAddNode:
        pass

    be = KGTraceBackend(backend=_NoAddNode())
    # Must not raise (hasattr guard).
    be.record_event(trace_id="t14", span_id="root14", name="root", is_root=True)
    assert be.get_trace("t14") is not None
