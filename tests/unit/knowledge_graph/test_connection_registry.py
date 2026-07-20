"""Unit tests for the named multi-connection graph registry (CONCEPT:AU-KG.backend.multi-connection-registry).

These use lightweight fakes for the engine/backend so the registry's routing,
caching, default-aliasing, fan-out, and partial-success contracts are verified
without standing up real backends or the epistemic-graph daemon.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.core.connection_registry import (
    CONNECTION_UNAVAILABLE,
    DEFAULT_NAME,
    ConnectionRegistry,
    ExternalGraphConnection,
    validate_persistable_connection_spec,
)


class _FakeBackend:
    def __init__(self, cypher_support: str = "full", supports_sparql: bool = False):
        self.cypher_support = cypher_support
        self.supports_sparql = supports_sparql
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def execute_read(self, query, params=None):
        return [{"query_ref": "opaque"}]


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


def test_persistable_connection_requires_secret_backed_transport_material():
    validate_persistable_connection_spec(
        {
            "backend": "neo4j",
            "role": "read",
            "connection_profile_ref": "vault://external-graphs/catalog/connection",
        }
    )
    with pytest.raises(ValueError, match="secret reference"):
        validate_persistable_connection_spec(
            {"backend": "neo4j", "role": "read", "uri": "bolt://graph.invalid"}
        )


@pytest.mark.parametrize(
    "backend",
    [
        "neo4j",
        "opencypher",
        "age",
        "ladybug",
        "epistemic_graph",
    ],
)
def test_every_external_property_graph_backend_uses_the_exact_reference_contract(
    backend: str,
) -> None:
    validate_persistable_connection_spec(
        {
            "name": "external-source",
            "backend": backend,
            "role": "read",
            "source_alias": "external-source",
            "connection_profile_ref": "secret://source/connection",
            "mapping_policy_ref": "secret://source/mapping",
            "tls_profile_ref": "secret://source/tls",
            "ingest_page_size": 250,
            "ingest_max_pages": 40,
            "ingest_max_row_bytes": 4_096,
            "ingest_max_total_bytes": 32_768,
            "ingest_max_nesting_depth": 8,
            "ingest_max_collection_items": 2_000,
            "sync_mode": "auto",
            "reconcile_deletions": True,
            "allow_empty_snapshot": False,
            "require_approval": True,
            "schema_drift_policy": "fail_closed",
        }
    )


@pytest.mark.parametrize(
    ("backend", "connection_profile", "auth_profile", "factory_backend"),
    [
        (
            "neo4j",
            {"uri": "bolt://graph.example.test", "database": "catalog"},
            {"user": "runtime-user", "password": "runtime-secret"},
            "neo4j",
        ),
        (
            "opencypher",
            {"uri": "bolt://graph.example.test", "database": "catalog"},
            {"user": "runtime-user", "password": "runtime-secret"},
            "neo4j",
        ),
        (
            "age",
            {"db_name": "catalog"},
            {"uri": "postgresql://graph.example.test/catalog"},
            "age",
        ),
    ],
)
def test_property_graph_auth_profile_is_transiently_merged_into_factory(
    monkeypatch,
    backend: str,
    connection_profile: dict[str, str],
    auth_profile: dict[str, str],
    factory_backend: str,
) -> None:
    from agent_utilities.knowledge_graph import backends
    from agent_utilities.knowledge_graph.core import (
        connection_registry as registry_module,
    )

    documents = {
        "secret://source/connection": json.dumps(connection_profile),
        "secret://source/auth": json.dumps(auth_profile),
    }
    monkeypatch.setattr(
        registry_module,
        "_SECRETS_CLIENT",
        SimpleNamespace(resolve_ref=lambda ref: documents[ref]),
    )
    captured: dict[str, object] = {}

    def create_backend(**kwargs):
        captured.update(kwargs)
        return _FakeBackend()

    monkeypatch.setattr(backends, "create_backend", create_backend)
    registry = ConnectionRegistry()
    registry.register(
        "external-source",
        {
            "backend": backend,
            "role": "read",
            "source_alias": "external-source",
            "connection_profile_ref": "secret://source/connection",
            "auth_profile_ref": "secret://source/auth",
        },
    )

    connection = registry.get_engine("external-source")

    assert isinstance(connection, ExternalGraphConnection)
    assert captured["backend_type"] == factory_backend
    assert all(captured[key] == value for key, value in auth_profile.items())
    assert "auth_profile_ref" not in captured
    exported = registry.export_specs()
    assert exported == [
        {
            "name": "external-source",
            "backend_type": backend,
            "role": "read",
            "source_alias": "external-source",
            "connection_profile_ref": "secret://source/connection",
            "auth_profile_ref": "secret://source/auth",
        }
    ]
    assert "runtime-secret" not in json.dumps(exported)


def test_property_graph_auth_profile_cannot_override_backend_selector(
    monkeypatch,
) -> None:
    from agent_utilities.knowledge_graph.core import (
        connection_registry as registry_module,
    )

    documents = {
        "secret://source/connection": json.dumps(
            {"uri": "bolt://graph.example.test", "database": "catalog"}
        ),
        "secret://source/auth": json.dumps(
            {
                "backend_type": "memory",
                "user": "runtime-user",
                "password": "runtime-secret",
            }
        ),
    }
    monkeypatch.setattr(
        registry_module,
        "_SECRETS_CLIENT",
        SimpleNamespace(resolve_ref=lambda ref: documents[ref]),
    )
    registry = ConnectionRegistry()
    registry.register(
        "external-source",
        {
            "backend": "neo4j",
            "role": "read",
            "source_alias": "external-source",
            "connection_profile_ref": "secret://source/connection",
            "auth_profile_ref": "secret://source/auth",
        },
    )

    with pytest.raises(ValueError, match="selector fields"):
        registry.get_engine("external-source")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ingest_page_size", 0, "out of range"),
        ("ingest_max_pages", 1_001, "out of range"),
        ("ingest_max_row_bytes", 255, "out of range"),
        ("ingest_max_total_bytes", 67_108_865, "out of range"),
        ("ingest_max_nesting_depth", True, "invalid"),
        ("ingest_max_collection_items", 0, "out of range"),
        ("sync_mode", "incremental", "sync_mode"),
        ("reconcile_deletions", "yes", "must be boolean"),
        ("allow_empty_snapshot", 1, "must be boolean"),
    ],
)
def test_property_graph_sync_declaration_fails_closed(field, value, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_persistable_connection_spec(
            {
                "name": "external-source",
                "backend": "neo4j",
                "role": "read",
                "source_alias": "external-source",
                "connection_profile_ref": "secret://source/connection",
                field: value,
            }
        )


def test_property_graph_total_budget_must_cover_one_row() -> None:
    with pytest.raises(ValueError, match="cover one row"):
        validate_persistable_connection_spec(
            {
                "name": "external-source",
                "backend": "neo4j",
                "role": "read",
                "source_alias": "external-source",
                "connection_profile_ref": "secret://source/connection",
                "ingest_max_row_bytes": 1_024,
                "ingest_max_total_bytes": 512,
            }
        )


@pytest.mark.parametrize(
    "field",
    ["custom_ontology", "email", "local_path", "node_query", "profile"],
)
def test_property_graph_declaration_rejects_all_inline_source_material(
    field: str,
) -> None:
    with pytest.raises(ValueError, match="unsupported inline material"):
        validate_persistable_connection_spec(
            {
                "name": "external-source",
                "backend": "neo4j",
                "role": "read",
                "connection_profile_ref": "secret://source/connection",
                field: "must-not-persist",
            }
        )


def test_external_backend_selector_conflict_cannot_bypass_exact_contract() -> None:
    with pytest.raises(ValueError, match="backend selectors disagree"):
        validate_persistable_connection_spec(
            {
                "name": "external-source",
                "backend": "memory",
                "backend_type": "neo4j",
                "role": "read",
                "connection_profile_ref": "secret://source/connection",
                "node_query": "must-not-persist",
            }
        )


def test_persistable_graphql_connection_is_reference_only_and_read_only():
    validate_persistable_connection_spec(
        {
            "backend": "graphql",
            "role": "read",
            "source_alias": "external-source",
            "connection_profile_ref": "secret://source/connection",
            "mapping_policy_ref": "secret://source/policy",
            "auth_profile_ref": "secret://source/auth",
            "tls_profile_ref": "secret://source/tls",
            "variables_ref": "secret://source/variables",
        }
    )
    with pytest.raises(ValueError, match="role='read'"):
        validate_persistable_connection_spec(
            {
                "backend": "graphql",
                "role": "read_write",
                "source_alias": "external-source",
                "connection_profile_ref": "secret://source/connection",
                "mapping_policy_ref": "secret://source/policy",
            }
        )
    with pytest.raises(ValueError, match="unsupported inline material"):
        validate_persistable_connection_spec(
            {
                "backend": "graphql",
                "role": "read",
                "source_alias": "external-source",
                "connection_profile_ref": "secret://source/connection",
                "mapping_policy_ref": "secret://source/policy",
                "query": "query Hidden { records { id } }",
            }
        )
    validate_persistable_connection_spec(
        {
            "backend": "graphql",
            "role": "read",
            "source_alias": "external-source",
            "connection_profile_ref": "secret://source/connection",
            "allow_introspection": True,
        }
    )
    with pytest.raises(ValueError, match="allow_introspection"):
        validate_persistable_connection_spec(
            {
                "backend": "graphql",
                "role": "read",
                "source_alias": "external-source",
                "connection_profile_ref": "secret://source/connection",
            }
        )
    with pytest.raises(ValueError, match="semantic_mapping is unsupported"):
        validate_persistable_connection_spec(
            {
                "backend": "graphql",
                "role": "read",
                "source_alias": "external-source",
                "connection_profile_ref": "secret://source/connection",
                "mapping_policy_ref": "secret://source/policy",
                "semantic_mapping": True,
            }
        )


def test_external_source_cannot_replace_graph_default(registry):
    registry.register(
        "external-source",
        {
            "backend": "graphql",
            "role": "read",
            "source_alias": "external-source",
            "connection_profile_ref": "secret://source/connection",
            "mapping_policy_ref": "secret://source/policy",
        },
    )
    assert not hasattr(registry, "set_default")
    assert registry.default_name() == DEFAULT_NAME

    with pytest.raises(ValueError, match="neutral lowercase"):
        registry.register(
            "Person Name",
            {
                "backend": "graphql",
                "role": "read",
                "source_alias": "external-source",
                "connection_profile_ref": "secret://source/connection",
                "mapping_policy_ref": "secret://source/policy",
            },
        )
    with pytest.raises(ValueError, match="does not match"):
        registry.register(
            "external-source",
            {
                "name": "different-source",
                "backend": "graphql",
                "role": "read",
                "source_alias": "external-source",
                "connection_profile_ref": "secret://source/connection",
                "mapping_policy_ref": "secret://source/policy",
            },
        )


def test_graphql_connection_builds_a_non_authoritative_read_adapter(default_engine):
    from agent_utilities.knowledge_graph.ingestion.graphql_connection import (
        GraphQLSourceAdapter,
    )

    live = ConnectionRegistry(default_engine_provider=lambda: default_engine)
    live.register(
        "external-source",
        {
            "backend": "graphql",
            "role": "read",
            "source_alias": "external-source",
            "connection_profile_ref": "secret://source/connection",
            "mapping_policy_ref": "secret://source/policy",
        },
    )

    source = live.get_engine("external-source")
    assert isinstance(source, GraphQLSourceAdapter)
    assert source.read_only is True
    assert source.cypher_support == "none"
    assert default_engine is live.get_engine(None)


def test_generic_external_connection_never_builds_an_operational_engine(
    default_engine, monkeypatch
):
    from agent_utilities.knowledge_graph import backends

    backend = _FakeBackend()
    monkeypatch.setattr(backends, "create_backend", lambda **_kwargs: backend)
    live = ConnectionRegistry(default_engine_provider=lambda: default_engine)
    live.register("external", {"backend": "neo4j", "role": "read"})

    source = live.get_engine("external")
    assert isinstance(source, ExternalGraphConnection)
    assert source.query_cypher("MATCH (n) RETURN n") == [{"query_ref": "opaque"}]
    assert not hasattr(source, "add_node")
    assert live.get_engine(None) is default_engine


def test_generic_opencypher_uses_the_hardened_bolt_read_transport(
    default_engine, monkeypatch
) -> None:
    from agent_utilities.knowledge_graph import backends

    captured = {}

    def build(**kwargs):
        captured.update(kwargs)
        return _FakeBackend()

    monkeypatch.setattr(backends, "create_backend", build)
    live = ConnectionRegistry(default_engine_provider=lambda: default_engine)
    live.register(
        "external-source",
        {
            "backend": "opencypher",
            "role": "read",
            "ingest_page_size": 250,
            "ingest_max_pages": 9,
            "ingest_max_row_bytes": 4_096,
            "ingest_max_total_bytes": 32_768,
            "ingest_max_nesting_depth": 8,
            "ingest_max_collection_items": 2_000,
            "sync_mode": "snapshot",
            "reconcile_deletions": True,
            "allow_empty_snapshot": False,
        },
    )

    source = live.get_engine("external-source")

    assert isinstance(source, ExternalGraphConnection)
    assert captured["backend_type"] == "neo4j"
    assert not {
        "allow_empty_snapshot",
        "ingest_max_pages",
        "ingest_max_row_bytes",
        "ingest_max_total_bytes",
        "ingest_max_nesting_depth",
        "ingest_max_collection_items",
        "ingest_page_size",
        "reconcile_deletions",
        "sync_mode",
    }.intersection(captured)
    assert live.backend_kind("external-source") == "opencypher"


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


def test_safe_get_engine_partial_success_redacts_backend_error(registry, caplog):
    registry.register("good", {"backend": "memory"})
    eng, err = registry.safe_get_engine("good")
    assert eng is not None and err is None
    eng2, err2 = registry.safe_get_engine("missing")
    assert eng2 is None and err2 == CONNECTION_UNAVAILABLE
    assert "missing" not in caplog.text
    assert "KeyError" in caplog.text


def test_fanout_isolation_between_connections(registry, default_engine):
    registry.register("other", {"backend": "memory"})
    default_engine.add_node("d1", "Thing")
    registry.get_engine("other").add_node("o1", "Thing")
    # Each engine only sees its own writes.
    assert [r["id"] for r in default_engine.query_cypher("...")] == ["d1"]
    assert [r["id"] for r in registry.get_engine("other").query_cypher("...")] == ["o1"]


def test_default_authority_is_immutable(registry):
    registry.register("c", {"backend": "memory"})
    assert registry.default_name() == DEFAULT_NAME
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
