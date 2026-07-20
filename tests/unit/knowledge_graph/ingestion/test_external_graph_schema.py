from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.ingestion.external_graph_schema import (
    ExternalGraphSchemaError,
    GraphQLDiscoveryAdapter,
    RemoteEpistemicGraphReadAdapter,
    approve_mapping_profile,
    discover_external_schema,
    external_graph_readiness,
    external_mapping_policy_digest,
    get_discovery_adapter,
    governed_semantic_mapping_enricher,
    mapping_policy_digest,
    mapping_profile_status,
    propose_mapping_profile,
)


def test_governed_semantic_mapper_uses_lite_bounded_context_transport(
    monkeypatch,
) -> None:
    bundle = SimpleNamespace(cache_key="synthetic-bundle")
    captured: dict[str, object] = {}

    def fake_completion(received_bundle, prompt, **kwargs):
        captured.update(bundle=received_bundle, prompt=prompt, kwargs=kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"Service":"Document"}')
                )
            ]
        )

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.retrieval.context_compiler_serving.bundle_chat_completion",
        fake_completion,
    )

    assert governed_semantic_mapping_enricher(bundle) == {"Service": "Document"}
    assert captured["bundle"] is bundle
    assert captured["kwargs"] == {
        "model": "lite",
        "timeout_s": 30.0,
        "max_retries": 0,
        "max_tokens": 512,
        "temperature": 0,
    }
    assert "JSON object" in str(captured["prompt"])


@pytest.mark.parametrize(
    "content",
    [
        "```json\n{}\n```",
        "[{}]",
        '{"Service": NaN}',
        '{"Service": 7}',
        '{"Service":"Document","Service":"Other"}',
    ],
)
def test_governed_semantic_mapper_rejects_non_strict_output(
    monkeypatch, content
) -> None:
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.retrieval.context_compiler_serving.bundle_chat_completion",
        lambda *_args, **_kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        ),
    )

    with pytest.raises(ExternalGraphSchemaError, match="semantic mapper response"):
        governed_semantic_mapping_enricher(SimpleNamespace())


def test_semantic_proposal_compiles_private_context_and_allowlists_suggestions(
    monkeypatch,
) -> None:
    from agent_utilities.knowledge_graph.retrieval.context_compiler import (
        ContextCompiler,
    )

    bundle = SimpleNamespace(as_text=lambda: "compiled")
    verified_session = object()
    captured: dict[str, object] = {}

    def fake_compile(self, query, **kwargs):
        captured.update(
            query=query,
            kwargs=kwargs,
            rows=list(self.engine._rows),
        )
        return bundle

    monkeypatch.setattr(ContextCompiler, "compile", fake_compile)

    def semantic_enricher(received_bundle):
        assert received_bundle is bundle
        return {
            "Service": "Document",
            "Component": "UnknownTarget",
            "UnknownLabel": "Document",
        }

    store = _Store()
    proposal = propose_mapping_profile(
        _Graph(),
        backend="neo4j",
        connection="external-catalog",
        source_alias="business-graph",
        ontology_classes=["Document"],
        secret_store=store,
        runtime_policy_digest=external_mapping_policy_digest({}),
        semantic_enricher=semantic_enricher,
        context_session=verified_session,
    )

    profile = json.loads(
        store.values["external-graphs/external-catalog/mapping-profile"]
    )
    assert profile["type_map"] == {"Service": "Document"}
    assert proposal["semantic_enrichment"] == "propose-only"
    assert captured["kwargs"]["session"] is verified_session
    assert captured["kwargs"]["token_budget"] == 1_500
    assert {row["name"] for row in captured["rows"]} == {
        "Service",
        "Component",
    }


def test_generic_opencypher_profile_uses_neo4j_compatible_label_function() -> None:
    node_query, edge_query = get_discovery_adapter("opencypher").generated_queries(
        identity_property="id"
    )

    assert "head(labels(n)) AS type" in node_query
    assert node_query.endswith("ORDER BY id SKIP $offset LIMIT $limit")
    assert edge_query.endswith(
        "ORDER BY source, target, type SKIP $offset LIMIT $limit"
    )


class _Store:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def set(self, key: str, value: str, **_metadata) -> None:
        self.values[key] = value


class _Graph:
    def __init__(self, *, drift: bool = False, procedures: bool = True) -> None:
        self.drift = drift
        self.procedures = procedures
        self.queries: list[str] = []

    def execute_read(self, query: str, _params=None):
        self.queries.append(query)
        labels = ["Service", "Component"] + (["NewType"] if self.drift else [])
        if "db.labels" in query:
            return [{"label": value} for value in labels] if self.procedures else []
        if "UNWIND ls" in query:
            return [{"label": value} for value in labels]
        if "db.relationshipTypes" in query:
            return [{"relationshipType": "DEPENDS_ON"}] if self.procedures else []
        if "type(r) AS relationshipType" in query:
            return [{"relationshipType": "DEPENDS_ON"}]
        if "db.propertyKeys" in query:
            return (
                [{"propertyKey": key} for key in ("id", "name", "description")]
                if self.procedures
                else []
            )
        if "UNWIND ks" in query:
            return [{"propertyKey": key} for key in ("id", "name", "description")]
        if "RETURN keys(n)" in query:
            return [{"propertyKeys": ["id", "name", "description"]}]
        return []


def test_foreign_epistemic_graph_uses_central_read_transport(monkeypatch) -> None:
    from agent_utilities.knowledge_graph.core import graph_compute

    captured: dict[str, object] = {}

    class _Client:
        closed = False

        def close(self) -> None:
            self.closed = True

    client = _Client()

    def _connect(**kwargs):
        captured.update(kwargs)
        return client

    monkeypatch.setattr(graph_compute, "connect_external_read_transport", _connect)
    verified_context = {
        "principal": "service:external-reader",
        "tenant": "tenant:test",
        "audience": "epistemic-graph-test",
        "agent_id": "service:external-reader",
        "roles": ["reader"],
        "scopes": ["kg:read"],
        "policy_version": "policy:test",
        "delegation": [],
    }
    adapter = RemoteEpistemicGraphReadAdapter(
        endpoint="tls://engine.example.test:9100",
        auth_secret="runtime-only",
        graph_name="source-graph",
        verified_context=verified_context,
        tls_profile_ref="secret://transport/source",
    )

    assert captured == {
        "endpoint": "tls://engine.example.test:9100",
        "auth_secret": "runtime-only",
        "graph_name": "source-graph",
        "verified_context": verified_context,
        "tls_profile": None,
        "tls_profile_ref": "secret://transport/source",
        "tls_server_name": None,
    }
    adapter.close()
    assert client.closed is True


def test_property_graph_discovery_is_bounded_and_digest_only_public() -> None:
    schema, capabilities = discover_external_schema(
        _Graph(), backend="neo4j", max_types=20
    )

    assert schema.labels == ("Component", "Service")
    assert capabilities.kind == "neo4j"
    public = schema.public_dict()
    assert public["label_count"] == 2
    assert "labels" not in public
    assert len(public["schema_digest"]) == 64


def test_open_cypher_discovery_falls_back_without_becoming_unbounded() -> None:
    graph = _Graph(procedures=False)
    schema, _ = discover_external_schema(graph, backend="age", max_types=7)

    assert schema.labels == ("Component", "Service")
    assert "bounded-label-scan" in schema.fallbacks_used
    assert all("LIMIT" in query for query in graph.queries)


def test_property_graph_discovery_marks_a_sentinel_overflow_partial() -> None:
    class _BoundedGraph:
        def execute_read(self, query: str, params=None):
            bound = int((params or {}).get("limit") or 1)
            if "db.labels" in query:
                return [{"label": f"Type{index}"} for index in range(bound)]
            if "relationshipTypes" in query:
                return [{"relationshipType": "LINKS_TO"}]
            if "propertyKeys" in query:
                return [{"propertyKey": "id"}]
            if "RETURN keys(n)" in query:
                return [{"propertyKeys": ["id"]}]
            return []

    schema, _capabilities = discover_external_schema(
        _BoundedGraph(), backend="neo4j", max_types=2
    )

    assert schema.labels == ("Type0", "Type1")
    assert schema.partial is True


def _approval_profile() -> dict:
    return {
        "profile_format": "external-graph-profile/v1",
        "adapter_version": "external-graph-discovery/v1",
        "backend_kind": "neo4j",
        "discovery_max_types": 200,
        "schema_digest": "a" * 64,
        "source_alias": "business-graph",
        "identity_property": "id",
        "identity_hmac_key_ref": (
            "vault://external-graphs/external-catalog/identity-key"
        ),
        "node_query": "MATCH (n) RETURN n LIMIT $limit",
        "node_mapping": {
            "id_path": "id",
            "type_path": "type",
            "version_path": "version",
            "properties_path": "properties",
            "property_allowlist": ["name"],
        },
        "edge_query": "MATCH (a)-[r]->(b) RETURN r LIMIT $limit",
        "edge_mapping": {
            "source_path": "source",
            "target_path": "target",
            "type_path": "type",
            "properties_path": "properties",
            "property_allowlist": ["confidence"],
        },
        "type_map": {"Service": "Service"},
        "edge_type_map": {"USES": "DEPENDS_ON"},
        "access": {"markings": ["external-import"]},
        "runtime_policy_digest": "b" * 64,
        "sync": {
            "allow_empty_snapshot": False,
            "max_pages": 100,
            "page_size": 500,
            "reconcile_deletions": True,
            "sync_mode": "auto",
        },
    }


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("node_mapping", "id_path"),
        ("node_mapping", "type_path"),
        ("node_mapping", "version_path"),
        ("node_mapping", "properties_path"),
        ("edge_mapping", "source_path"),
        ("edge_mapping", "target_path"),
        ("edge_mapping", "type_path"),
        ("edge_mapping", "properties_path"),
    ],
)
def test_mapping_digest_binds_every_behavioral_selector_path(
    section: str, field: str
) -> None:
    profile = _approval_profile()
    approved = mapping_policy_digest(profile)

    profile[section][field] = f"redirected_{field}"

    assert mapping_policy_digest(profile) != approved


def test_mapping_digest_binds_identity_ref_but_never_secret_value() -> None:
    profile = _approval_profile()
    approved = mapping_policy_digest(profile)
    profile["identity_hmac_key_ref"] = "vault://external-graphs/other/identity-key"
    assert mapping_policy_digest(profile) != approved

    profile = _approval_profile()
    profile["identity_hmac_key"] = "secret-value-a-that-must-never-be-hashed"
    first = mapping_policy_digest(profile)
    profile["identity_hmac_key"] = "secret-value-b-that-must-never-be-hashed"
    assert mapping_policy_digest(profile) == first


def test_proposal_is_pseudonymous_versioned_and_requires_exact_approval() -> None:
    store = _Store()
    proposal = propose_mapping_profile(
        _Graph(),
        backend="neo4j",
        connection="external-catalog",
        source_alias="business-graph",
        ontology_classes=["Service", "Component", "Document"],
        secret_store=store,
        access={"is_public": False, "markings": ["external-import"]},
        property_allowlist=["id", "name", "description", "owner_name"],
        runtime_policy_digest=external_mapping_policy_digest({}),
    )

    assert proposal["status"] == "proposed"
    assert proposal["proposal_version"] == 1
    rendered = json.dumps(proposal, sort_keys=True)
    assert "Component" not in rendered
    assert "Service" not in rendered
    assert '"external_label"' not in rendered
    assert all(
        item["source_token"].startswith("label-") for item in proposal["mappings"]
    )
    stored = json.loads(
        store.values["external-graphs/external-catalog/mapping-profile"]
    )
    assert stored["approval_status"] == "proposed"
    assert "identity_hmac_key" not in stored
    assert stored["identity_hmac_key_ref"] == (
        "vault://external-graphs/external-catalog/identity-key"
    )
    assert "owner_name" not in stored["node_mapping"]["property_allowlist"]
    assert stored["sync"] == {
        "allow_empty_snapshot": False,
        "max_collection_items": 10_000,
        "max_nesting_depth": 16,
        "max_pages": 100,
        "max_row_bytes": 1_048_576,
        "max_total_bytes": 16_777_216,
        "page_size": 500,
        "reconcile_deletions": True,
        "sync_mode": "auto",
    }

    with pytest.raises(ExternalGraphSchemaError, match="does not match"):
        approve_mapping_profile(
            connection="external-catalog",
            proposal_id=proposal["proposal_id"],
            proposal_version=2,
            schema_digest=proposal["schema_digest"],
            mapping_digest=proposal["mapping_digest"],
            secret_store=store,
        )

    approved = approve_mapping_profile(
        connection="external-catalog",
        proposal_id=proposal["proposal_id"],
        proposal_version=proposal["proposal_version"],
        schema_digest=proposal["schema_digest"],
        mapping_digest=proposal["mapping_digest"],
        secret_store=store,
        approver_ref="operator",
    )
    assert approved["status"] == "approved"
    assert (
        mapping_profile_status("external-catalog", secret_store=store)["status"]
        == "approved"
    )

    tampered = json.loads(
        store.values["external-graphs/external-catalog/mapping-profile"]
    )
    tampered["node_mapping"]["id_path"] = "redirected.id"
    store.values["external-graphs/external-catalog/mapping-profile"] = json.dumps(
        tampered
    )
    assert (
        mapping_profile_status("external-catalog", secret_store=store)["status"]
        == "invalid"
    )


def test_readiness_fails_closed_on_schema_drift_without_returning_schema_names() -> (
    None
):
    store = _Store()
    proposal = propose_mapping_profile(
        _Graph(),
        backend="neo4j",
        connection="external-catalog",
        source_alias="business-graph",
        ontology_classes=["Service", "Component"],
        secret_store=store,
        property_allowlist=["id", "description"],
        runtime_policy_digest=external_mapping_policy_digest({}),
    )
    approve_mapping_profile(
        connection="external-catalog",
        proposal_id=proposal["proposal_id"],
        proposal_version=proposal["proposal_version"],
        schema_digest=proposal["schema_digest"],
        mapping_digest=proposal["mapping_digest"],
        secret_store=store,
    )

    readiness = external_graph_readiness(
        _Graph(drift=True),
        backend="neo4j",
        connection="external-catalog",
        secret_store=store,
        runtime_policy_digest=external_mapping_policy_digest({}),
    )
    assert readiness["ready"] is False
    assert readiness["schema_drift"] == "detected"
    assert "NewType" not in json.dumps(readiness)


def test_readiness_fails_closed_when_current_discovery_is_partial() -> None:
    store = _Store()
    proposal = propose_mapping_profile(
        _Graph(),
        backend="neo4j",
        connection="external-catalog",
        source_alias="business-graph",
        ontology_classes=["Service", "Component"],
        secret_store=store,
        property_allowlist=["id", "description"],
        runtime_policy_digest=external_mapping_policy_digest({}),
    )
    approve_mapping_profile(
        connection="external-catalog",
        proposal_id=proposal["proposal_id"],
        proposal_version=proposal["proposal_version"],
        schema_digest=proposal["schema_digest"],
        mapping_digest=proposal["mapping_digest"],
        secret_store=store,
    )

    class _NowPartial(_Graph):
        def execute_read(self, query: str, params=None):
            rows = super().execute_read(query, params)
            if "db.labels" in query:
                return [*rows, {"label": "OverflowSentinel"}]
            return rows

    readiness = external_graph_readiness(
        _NowPartial(),
        backend="neo4j",
        connection="external-catalog",
        secret_store=store,
        runtime_policy_digest=external_mapping_policy_digest({}),
        max_types=2,
    )

    assert readiness["ready"] is False
    assert readiness["discovery"] == "partial"
    assert "OverflowSentinel" not in json.dumps(readiness)


def test_readiness_fails_closed_on_runtime_mapping_policy_drift() -> None:
    store = _Store()
    approved_policy = {"property_allowlist": ["id", "description"]}
    proposal = propose_mapping_profile(
        _Graph(),
        backend="neo4j",
        connection="external-catalog",
        source_alias="business-graph",
        ontology_classes=["Service", "Component"],
        secret_store=store,
        property_allowlist=["id", "description"],
        runtime_policy_digest=external_mapping_policy_digest(approved_policy),
    )
    approve_mapping_profile(
        connection="external-catalog",
        proposal_id=proposal["proposal_id"],
        proposal_version=proposal["proposal_version"],
        schema_digest=proposal["schema_digest"],
        mapping_digest=proposal["mapping_digest"],
        secret_store=store,
    )

    readiness = external_graph_readiness(
        _Graph(),
        backend="neo4j",
        connection="external-catalog",
        secret_store=store,
        runtime_policy_digest=external_mapping_policy_digest(
            {"property_allowlist": ["id"]}
        ),
    )

    assert readiness["ready"] is False
    assert readiness["schema_drift"] == "none"
    assert readiness["mapping_drift"] == "detected"


def test_graphql_introspection_and_bounded_probe_are_transport_neutral() -> None:
    introspection = GraphQLDiscoveryAdapter().discover(
        lambda _document, _variables: {
            "data": {
                "__schema": {
                    "types": [
                        {
                            "kind": "OBJECT",
                            "name": "Asset",
                            "fields": [{"name": "id"}, {"name": "title"}],
                        }
                    ]
                }
            }
        }
    )
    assert introspection.mode == "introspection"
    assert introspection.public_dict()["field_count"] == 2
    assert "Asset" not in json.dumps(introspection.public_dict())

    calls = []

    def no_introspection(document, variables):
        calls.append((document, variables))
        if "__schema" in document:
            return {"errors": [{"message": "disabled"}]}
        return {
            "data": {
                "assets": [
                    {
                        "__typename": "Asset",
                        "id": "raw-value-never-retained",
                        "title": "private sample never retained",
                    }
                ]
            }
        }

    probed = GraphQLDiscoveryAdapter().discover(
        no_introspection,
        probe_document=(
            "query Probe($first: Int!) { "
            "assets(first: $first) { __typename id title } }"
        ),
        probe_variables={"first": 9999},
        max_types=5,
    )
    assert probed.mode == "bounded-probe"
    assert calls[-1][1]["first"] == 5
    raw = json.dumps(probed.raw_dict())
    assert "raw-value-never-retained" not in raw
    assert "private sample never retained" not in raw


def test_graphql_schema_digest_detects_field_type_drift() -> None:
    def discover(field_type: str):
        return GraphQLDiscoveryAdapter().discover(
            lambda _document, _variables: {
                "data": {
                    "__schema": {
                        "types": [
                            {
                                "kind": "OBJECT",
                                "name": "Asset",
                                "fields": [
                                    {
                                        "name": "version",
                                        "args": [],
                                        "type": {
                                            "kind": "SCALAR",
                                            "name": field_type,
                                        },
                                    }
                                ],
                            }
                        ]
                    }
                }
            }
        )

    before = discover("String")
    after = discover("Int")

    assert before.schema_digest != after.schema_digest
    assert before.public_dict()["field_count"] == after.public_dict()["field_count"]


def test_graphql_probe_rejects_writes_and_unbounded_operations() -> None:
    adapter = GraphQLDiscoveryAdapter()

    def disabled(_document, _variables):
        return {"errors": [{"message": "disabled"}]}

    with pytest.raises(ExternalGraphSchemaError, match="read-only"):
        adapter.discover(
            disabled,
            probe_document="mutation Update($limit: Int!) { update(limit: $limit) { id } }",
        )
    with pytest.raises(ExternalGraphSchemaError, match="must bind"):
        adapter.discover(
            disabled,
            probe_document="query Probe { assets { id } }",
        )
    with pytest.raises(ExternalGraphSchemaError, match="must bind"):
        adapter.discover(
            disabled,
            probe_document=(
                "# $limit is commentary, not a bound\nquery Probe { assets { id } }"
            ),
        )
    with pytest.raises(ExternalGraphSchemaError, match="must bind"):
        adapter.discover(
            disabled,
            probe_document=(
                "query Probe($limit: Int!) { "
                "assets(offset: $limit) { id } }"
            ),
        )
    with pytest.raises(ExternalGraphSchemaError, match="must bind"):
        adapter.discover(
            disabled,
            probe_document=(
                "query Probe($limit: Int!) { assets(offset: $limit) { id } } "
                "fragment Unused on Query { assets(limit: $limit) { id } }"
            ),
        )
