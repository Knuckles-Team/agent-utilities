"""Functions runtime — typed registry, runtime I/O, Functions-on-Objects (CONCEPT:KG-2.41).

Covers Palantir ``functions/overview`` semantics: typed/versioned registration
with release/publish + lookup, the single governed :class:`FunctionRuntime`
validating valid and type-mismatched inputs and coercing typed output, and a
Functions-on-Objects read against a small in-memory facade/graph.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.ontology.functions import (
    DEFAULT_FUNCTION_REGISTRY,
    DEFAULT_FUNCTION_RUNTIME,
    FunctionKind,
    FunctionParameter,
    FunctionRegistry,
    FunctionRuntime,
    FunctionSpec,
    ObjectFunctionContext,
)

# ── Registry: typed registration, versioning, release, lookup ───────────────


def test_default_registry_is_live_not_empty() -> None:
    assert "object.summarize" in DEFAULT_FUNCTION_REGISTRY
    assert "numeric.aggregate" in DEFAULT_FUNCTION_REGISTRY
    summ = DEFAULT_FUNCTION_REGISTRY.get("object.summarize")
    assert summ is not None and summ.released is True
    assert summ.kind == FunctionKind.ON_OBJECTS


def test_register_duplicate_version_rejected() -> None:
    reg = FunctionRegistry()
    spec = FunctionSpec(
        name="math.double",
        version="1.0.0",
        inputs=[FunctionParameter(name="x", type="int")],
        output="int",
        handler=lambda x: x * 2,
    )
    reg.register(spec)
    with pytest.raises(ValueError):
        reg.register(spec)
    # replace=True overwrites in place.
    reg.register(spec, replace=True)
    assert len(reg) == 1


def test_release_and_latest_version_lookup() -> None:
    reg = FunctionRegistry()
    reg.register(
        FunctionSpec(name="f", version="1.0.0", output="int", handler=lambda: 1)
    )
    reg.register(
        FunctionSpec(name="f", version="2.0.0", output="int", handler=lambda: 2)
    )
    # No releases yet → latest draft wins by semver.
    assert reg.get("f").version == "2.0.0"
    # Release only the older version → released lookup prefers it.
    reg.release("f", "1.0.0")
    assert reg.get("f", released_only=True).version == "1.0.0"
    # Pinned lookup always returns the exact version.
    assert reg.get("f", "2.0.0").version == "2.0.0"
    assert reg.versions("f") == ["1.0.0", "2.0.0"]


def test_register_rejects_bad_semver() -> None:
    with pytest.raises(ValueError):
        FunctionSpec(name="f", version="1.0", output="int", handler=lambda: 1)


# ── Runtime: valid + invalid typed I/O ──────────────────────────────────────


def _make_runtime() -> tuple[FunctionRuntime, FunctionRegistry]:
    reg = FunctionRegistry()
    reg.register(
        FunctionSpec(
            name="math.add",
            version="1.0.0",
            kind=FunctionKind.PLAIN,
            inputs=[
                FunctionParameter(name="a", type="float"),
                FunctionParameter(name="b", type="float"),
            ],
            output="float",
            handler=lambda a, b: a + b,
            released=True,
        )
    )
    return FunctionRuntime(registry=reg), reg


def test_runtime_valid_invocation_typed_output() -> None:
    rt, _ = _make_runtime()
    res = rt.invoke("math.add", {"a": 2, "b": 3})
    assert res.ok is True
    assert res.value == 5.0
    assert isinstance(res.value, float)  # int widened to declared float
    assert res.audit_ref  # an audit entry was recorded


def test_runtime_type_mismatch_input_is_error() -> None:
    rt, _ = _make_runtime()
    res = rt.invoke("math.add", {"a": "not-a-number", "b": 3})
    assert res.ok is False
    assert "expected float" in res.error
    # Audit still recorded the failure.
    assert res.audit_ref


def test_runtime_missing_required_input_is_error() -> None:
    rt, _ = _make_runtime()
    res = rt.invoke("math.add", {"a": 1})
    assert res.ok is False
    assert "missing required input 'b'" in res.error


def test_runtime_unknown_function_is_error() -> None:
    rt, _ = _make_runtime()
    res = rt.invoke("does.not.exist", {})
    assert res.ok is False
    assert "unknown function" in res.error


def test_runtime_output_coercion_failure_surfaces() -> None:
    reg = FunctionRegistry()
    reg.register(
        FunctionSpec(
            name="bad.output",
            version="1.0.0",
            output="int",
            handler=lambda: {"not": "an int"},
            released=True,
        )
    )
    rt = FunctionRuntime(registry=reg)
    res = rt.invoke("bad.output", {})
    assert res.ok is False
    assert "expected int" in res.error


def test_default_runtime_numeric_aggregate_live_path() -> None:
    res = DEFAULT_FUNCTION_RUNTIME.invoke(
        "numeric.aggregate", {"values": [1, 2, 3, 4], "op": "mean"}
    )
    assert res.ok is True
    assert res.value == 2.5


# ── Functions-on-Objects against a small in-memory facade ───────────────────


class _FakeStore:
    """Minimal in-memory graph store with an execute(cypher, params) interface."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str, str]] = []  # (src, rel, dst)

    def add_node(self, node_id: str, **props: Any) -> None:
        self.nodes[node_id] = {"id": node_id, **props}

    def add_edge(self, src: str, rel: str, dst: str) -> None:
        self.edges.append((src, rel, dst))

    def execute(self, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        # Single-object read: MATCH (n {id: $id}) RETURN n
        if (
            "RETURN n" in cypher
            and "id: $id" in cypher
            and "->" not in cypher
            and "*" not in cypher
            and "<-" not in cypher
            and "-[" not in cypher
        ):
            nid = params.get("id")
            node = self.nodes.get(nid)
            return [{"n": node}] if node else []
        # 1-hop out-neighbors: MATCH (n {id: $id})-[...]->(m) RETURN m
        if "->(m)" in cypher and "RETURN m" in cypher:
            nid = params.get("id")
            out = [
                self.nodes[d]
                for (s, _r, d) in self.edges
                if s == nid and d in self.nodes
            ]
            return [{"m": node} for node in out]
        return []


class _FakeFacade:
    """Stand-in for KnowledgeGraph exposing the guarded query() the helpers use."""

    def __init__(self, store: _FakeStore) -> None:
        self.store = store

    def query(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return self.store.execute(cypher, params or {})


def _seed_facade() -> _FakeFacade:
    store = _FakeStore()
    store.add_node("acct:1", type="account", name="Root", balance=100.0)
    store.add_node("acct:2", type="account", name="Child A", balance=40.0)
    store.add_node("acct:3", type="account", name="Child B", balance=60.0)
    store.add_edge("acct:1", "HOLDS", "acct:2")
    store.add_edge("acct:1", "HOLDS", "acct:3")
    return _FakeFacade(store)


def test_functions_on_objects_read_and_traverse() -> None:
    ctx = ObjectFunctionContext(_seed_facade())
    props = ctx.get_object("acct:1")
    assert props["name"] == "Root"
    assert ctx.get_property("acct:1", "balance") == 100.0

    neighbors = ctx.neighbors("acct:1", rel_type="HOLDS", direction="out")
    names = sorted(n["name"] for n in neighbors)
    assert names == ["Child A", "Child B"]

    total = ctx.aggregate_links("acct:1", "balance", rel_type="HOLDS", op="sum")
    assert total == 100.0


def test_on_objects_function_via_runtime_with_injected_context() -> None:
    """An ON_OBJECTS handler that aggregates over linked objects via the context."""
    reg = FunctionRegistry()

    def _holdings_total(object_id: str, context: ObjectFunctionContext) -> float:
        return context.aggregate_links(object_id, "balance", rel_type="HOLDS", op="sum")

    reg.register(
        FunctionSpec(
            name="account.holdings_total",
            version="1.0.0",
            kind=FunctionKind.ON_OBJECTS,
            inputs=[FunctionParameter(name="object_id", type="string")],
            output="float",
            handler=_holdings_total,
            released=True,
        )
    )
    rt = FunctionRuntime(registry=reg, graph=_seed_facade())
    res = rt.invoke("account.holdings_total", {"object_id": "acct:1"})
    assert res.ok is True
    assert res.value == 100.0


def test_object_summarize_builtin_on_objects() -> None:
    res = DEFAULT_FUNCTION_RUNTIME.invoke(
        "object.summarize",
        {"object_id": "x", "properties": {"name": "Widget", "type": "item"}},
    )
    assert res.ok is True
    assert "Widget" in res.value


def test_offline_facade_degrades_to_empty() -> None:
    """No backend → reads return empty, never raise."""
    ctx = ObjectFunctionContext(None)
    # Default facade with no reachable store yields {} / [] cleanly.
    assert ctx.get_object("nope") == {}
    assert ctx.neighbors("nope") == []
