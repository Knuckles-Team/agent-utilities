"""Unit tests for the named multi-connection graph registry (CONCEPT:AU-KG.backend.multi-connection-registry).

These use lightweight fakes for the engine/backend so the registry's routing,
caching, default-aliasing, fan-out, and partial-success contracts are verified
without standing up real backends or the epistemic-graph daemon.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.connection_registry import (
    DEFAULT_NAME,
    ConnectionRegistry,
)


class _FakeBackend:
    def __init__(self, cypher_support: str = "full", supports_sparql: bool = False):
        self.cypher_support = cypher_support
        self.supports_sparql = supports_sparql
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeEngine:
    """Minimal engine double: records writes, answers queries from a store."""

    def __init__(self, name: str, backend: _FakeBackend | None = None):
        self.name = name
        self.backend = backend or _FakeBackend()
        self.nodes: dict[str, dict] = {}

    def add_node(self, node_id, node_type, props=None):
        self.nodes[node_id] = {"type": node_type, **(props or {})}

    def query_cypher(self, cypher, params=None, as_of=None):
        # Return one row per stored node id (enough to prove isolation/routing).
        return [{"id": nid} for nid in sorted(self.nodes)]


class _Registry(ConnectionRegistry):
    """Registry whose named engines are fakes (no real backend/daemon)."""

    def __init__(self, default_engine):
        super().__init__(default_engine_provider=lambda: default_engine)
        self.built: list[dict] = []

    def _build_engine(self, spec):
        self.built.append(spec)
        backend = _FakeBackend(cypher_support=spec.get("_cypher", "full"))
        return _FakeEngine(spec.get("_name", "named"), backend=backend)


@pytest.fixture
def default_engine():
    return _FakeEngine("default", backend=_FakeBackend(cypher_support="subset"))


@pytest.fixture
def registry(default_engine):
    return _Registry(default_engine)


def test_default_reuses_active_engine(registry, default_engine):
    # The reserved "default" name must alias the injected active engine, never a
    # freshly built one.
    assert registry.get_engine(None) is default_engine
    assert registry.get_engine("") is default_engine
    assert registry.get_engine("default") is default_engine
    assert registry.built == []  # default never triggers a build


def test_register_builds_and_caches(registry):
    registry.register("pg-main", {"backend": "age", "_name": "pg"})
    e1 = registry.get_engine("pg-main")
    e2 = registry.get_engine("pg-main")
    assert e1 is e2  # cached, built once
    assert len(registry.built) == 1
    # "backend" is normalised to the create_backend selector key.
    assert registry.built[0]["backend_type"] == "age"


def test_register_reserved_name_rejected(registry):
    for bad in ("default", "all", "  ", "DEFAULT"):
        with pytest.raises(ValueError):
            registry.register(bad, {"backend": "memory"})


def test_resolve_names_modes(registry):
    registry.register("a", {"backend": "memory"})
    registry.register("b", {"backend": "memory"})
    assert registry.resolve_names("") == ([DEFAULT_NAME], False)
    assert registry.resolve_names("default") == ([DEFAULT_NAME], False)
    assert registry.resolve_names("a") == (["a"], False)  # single named: not fanout
    names, fanout = registry.resolve_names("all")
    assert fanout and set(names) == {DEFAULT_NAME, "a", "b"}
    assert registry.resolve_names("a,b") == (["a", "b"], True)
    assert registry.resolve_names(["a", "b"]) == (["a", "b"], True)
    assert registry.resolve_names(["a"]) == (["a"], False)


def test_non_str_target_routes_to_default(registry):
    # A tool fn called directly (not via _execute_tool) passes the unresolved
    # pydantic FieldInfo default for `target`; that must route to the default
    # connection, never a spurious fan-out.
    class _FieldInfoLike:
        def __str__(self):
            return "annotation=NoneType required=False default='' description='...'"

    assert registry.resolve_names(_FieldInfoLike()) == (["default"], False)
    assert registry.resolve_names(object()) == (["default"], False)


def test_unknown_named_target_raises(registry):
    with pytest.raises(KeyError):
        registry.get_engine("nope")


def test_safe_get_engine_partial_success(registry):
    registry.register("good", {"backend": "memory"})
    eng, err = registry.safe_get_engine("good")
    assert eng is not None and err is None
    eng2, err2 = registry.safe_get_engine("missing")
    assert eng2 is None and "missing" in err2


def test_fanout_isolation_between_connections(registry, default_engine):
    registry.register("other", {"backend": "memory"})
    default_engine.add_node("d1", "Thing")
    registry.get_engine("other").add_node("o1", "Thing")
    # Each engine only sees its own writes.
    assert [r["id"] for r in default_engine.query_cypher("...")] == ["d1"]
    assert [r["id"] for r in registry.get_engine("other").query_cypher("...")] == ["o1"]


def test_set_and_clear_default_target(registry):
    registry.register("c", {"backend": "memory"})
    assert registry.set_default("c") == "c"
    assert registry.default_name() == "c"
    # Removing the current default target resets to "default".
    registry.remove("c")
    assert registry.default_name() == DEFAULT_NAME


def test_remove_closes_backend(registry):
    registry.register("z", {"backend": "memory"})
    eng = registry.get_engine("z")
    assert registry.remove("z") is True
    assert eng.backend.closed is True
    assert registry.remove("z") is False  # already gone


def test_status_surface(registry):
    registry.register("pg", {"backend": "age", "_cypher": "full"})
    registry.get_engine("pg")  # connect so cypher_support is reported
    st = registry.status()
    assert st["default_target"] == DEFAULT_NAME
    by_name = {c["name"]: c for c in st["connections"]}
    assert by_name[DEFAULT_NAME]["cypher_support"] == "subset"
    assert by_name["pg"]["connected"] is True
    assert by_name["pg"]["cypher_support"] == "full"


def test_close_all_clears_cache(registry):
    registry.register("a", {"backend": "memory"})
    eng = registry.get_engine("a")
    registry.close_all()
    assert eng.backend.closed is True
    # A fresh access rebuilds a new engine.
    assert registry.get_engine("a") is not eng
