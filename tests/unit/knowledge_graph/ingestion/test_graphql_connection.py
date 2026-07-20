from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.ingestion.external_graph_schema import (
    ExternalGraphSchemaError,
    approve_mapping_profile,
    canonical_identity_key_ref,
)
from agent_utilities.knowledge_graph.ingestion.graphql_connection import (
    GraphQLSourceAdapter,
    graphql_mapping_profile_status,
    graphql_source_readiness,
    ingest_registered_graphql,
    propose_graphql_mapping_profile,
)


class _Store:
    def __init__(self, refs: dict[str, object]) -> None:
        self.refs = refs
        self.values: dict[str, str] = {}

    def resolve_ref(self, ref: str) -> str | None:
        value = self.refs.get(ref)
        return json.dumps(value) if value is not None else None

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def set(self, key: str, value: str, **_metadata) -> None:
        self.values[key] = value


class _Response:
    def __init__(self, payload: dict) -> None:
        self.content = json.dumps(payload, allow_nan=False).encode("utf-8")

    def raise_for_status(self) -> None:
        return None


class _Transport:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def post(self, _endpoint: str, **kwargs):
        request = kwargs["json"]
        self.calls.append(request)
        if "RuntimeConnectionProbe" in request["query"]:
            return _Response({"data": {"__typename": "Query"}})
        if "__schema" in request["query"]:
            return _Response(
                {
                    "data": {
                        "__schema": {
                            "types": [
                                {
                                    "kind": "OBJECT",
                                    "name": "Asset",
                                    "fields": [
                                        {"name": "id"},
                                        {"name": "title"},
                                    ],
                                }
                            ]
                        }
                    }
                }
            )
        return _Response(
            {
                "data": {
                    "assets": [{"id": "source-identity", "title": "Synthetic asset"}]
                }
            }
        )


def _named(kind: str, name: str) -> dict:
    return {"kind": kind, "name": name, "ofType": None}


def _non_null(value: dict) -> dict:
    return {"kind": "NON_NULL", "name": None, "ofType": value}


def _list_of(value: dict) -> dict:
    return {"kind": "LIST", "name": None, "ofType": value}


class _BootstrapTransport:
    def __init__(self, roots: list[dict]) -> None:
        self.calls: list[dict] = []
        self.types = [
            {
                "kind": "OBJECT",
                "name": "Query",
                "fields": roots,
            },
            {
                "kind": "OBJECT",
                "name": "SyntheticRecord",
                "fields": [
                    {"name": "id", "args": [], "type": _named("SCALAR", "ID")},
                    {
                        "name": "name",
                        "args": [],
                        "type": _named("SCALAR", "String"),
                    },
                    {
                        "name": "updatedAt",
                        "args": [],
                        "type": _named("SCALAR", "String"),
                    },
                ],
            },
        ]

    def post(self, _endpoint: str, **kwargs):
        request = kwargs["json"]
        self.calls.append(request)
        if "AgentUtilitiesReadBootstrap" in request["query"]:
            return _Response(
                {
                    "data": {
                        "__schema": {
                            "queryType": {"name": "Query"},
                            "mutationType": {"name": "Mutation"},
                            "subscriptionType": {"name": "Subscription"},
                            "types": self.types,
                        }
                    }
                }
            )
        if "__schema" in request["query"]:
            return _Response({"data": {"__schema": {"types": self.types}}})
        return _Response(
            {
                "data": {
                    "records": [
                        {
                            "id": "private-source-id",
                            "name": "Synthetic record",
                            "updatedAt": "version-one",
                        }
                    ]
                }
            }
        )


def _bootstrap_root(
    name: str = "records",
    *,
    bound: bool = True,
    required_selector: bool = False,
) -> dict:
    args = []
    if bound:
        args.append(
            {
                "name": "first",
                "defaultValue": None,
                "type": _non_null(_named("SCALAR", "Int")),
            }
        )
    if required_selector:
        args.append(
            {
                "name": "selector",
                "defaultValue": None,
                "type": _non_null(_named("SCALAR", "String")),
            }
        )
    return {
        "name": name,
        "args": args,
        "type": _non_null(_list_of(_non_null(_named("OBJECT", "SyntheticRecord")))),
    }


def _bootstrap_source(
    store: _Store,
    transport: _BootstrapTransport,
    *,
    allow_empty_snapshot: bool = False,
):
    return GraphQLSourceAdapter(
        connection="external-source",
        source_alias="external-source",
        connection_profile_ref="secret://source/connection",
        auth_profile_ref="secret://source/auth",
        allow_introspection=True,
        allow_empty_snapshot=allow_empty_snapshot,
        resolver=store.resolve_ref,
        transport=transport,
    )


def _refs(
    *, query: str | None = None, bounded_probe_ack: bool | None = None
) -> dict[str, object]:
    discovery = {"enabled": True, "allow_introspection": True}
    if bounded_probe_ack is not None:
        discovery = {
            "enabled": True,
            "allow_introspection": False,
            "probe_query": (
                "query Probe($first: Int!) { assets(first: $first) { "
                "__typename id title } }"
            ),
            "accept_bounded_probe": bounded_probe_ack,
        }
    return {
        "secret://source/connection": {
            "profile_format": "graphql-connection/v1",
            "endpoint": "https://source.example.test/graphql",
        },
        "secret://source/auth": {
            "profile_format": "graphql-auth/v1",
            "headers": {"X-Test-Auth": "synthetic-value"},
        },
        "secret://source/policy": {
            "profile_format": "graphql-document-policy/v1",
            "default_operation": "asset_read",
            "discovery": discovery,
            "operations": {
                "asset_read": {
                    "query": query
                    or "query AssetRead($first: Int!) { assets(first: $first) { id title } }",
                    "root_path": "assets",
                    "id_path": "id",
                    "title_path": "title",
                }
            },
            "governance": {
                "classification": "confidential",
                "retention": "P30D",
                "access": {"markings": ["external-import"]},
            },
        },
        "secret://source/variables": {"first": 5},
    }


def _source(store: _Store, transport: _Transport | None = None) -> GraphQLSourceAdapter:
    return GraphQLSourceAdapter(
        connection="external-source",
        source_alias="external-source",
        connection_profile_ref="secret://source/connection",
        mapping_policy_ref="secret://source/policy",
        auth_profile_ref="secret://source/auth",
        variables_ref="secret://source/variables",
        resolver=store.resolve_ref,
        transport=transport or _Transport(),
    )


def test_graphql_runtime_profile_validation_reuses_native_connector_contract() -> None:
    valid_store = _Store(_refs())
    _source(valid_store).validate_runtime_profiles()

    unsafe_refs = _refs()
    unsafe_refs["secret://source/connection"] = {
        "profile_format": "graphql-connection/v1",
        "endpoint": "http://source.example.test/graphql",
    }
    with pytest.raises(ExternalGraphSchemaError, match="HTTPS"):
        _source(_Store(unsafe_refs)).validate_runtime_profiles()

    unknown_auth_refs = _refs()
    unknown_auth_refs["secret://source/auth"] = {
        "profile_format": "graphql-auth/v1",
        "headers": {},
        "credential": "must-not-be-accepted",
    }
    with pytest.raises(ExternalGraphSchemaError, match="unknown fields"):
        _source(_Store(unknown_auth_refs)).validate_runtime_profiles()

    invalid_tls_refs = _refs()
    invalid_tls_refs["secret://source/tls"] = {"verify": False}
    invalid_tls_store = _Store(invalid_tls_refs)
    invalid_tls_source = GraphQLSourceAdapter(
        connection="external-source",
        source_alias="external-source",
        connection_profile_ref="secret://source/connection",
        mapping_policy_ref="secret://source/policy",
        auth_profile_ref="secret://source/auth",
        tls_profile_ref="secret://source/tls",
        variables_ref="secret://source/variables",
        resolver=invalid_tls_store.resolve_ref,
        transport=_Transport(),
    )
    with pytest.raises(ExternalGraphSchemaError, match="transport security"):
        invalid_tls_source.validate_runtime_profiles()


def test_graphql_connection_probe_is_bounded_and_uses_runtime_profiles() -> None:
    transport = _Transport()
    source = _source(_Store(_refs()), transport)

    assert source.probe_connection() is True
    assert transport.calls == [
        {
            "query": "query RuntimeConnectionProbe { __typename }",
            "variables": {},
        }
    ]


def test_graphql_runtime_documents_reject_duplicate_json_keys() -> None:
    store = _Store(_refs())

    def resolve(ref: str) -> str | None:
        if ref == "secret://source/connection":
            return (
                '{"profile_format":"graphql-connection/v1",'
                '"endpoint":"https://first.example.test/graphql",'
                '"endpoint":"https://second.example.test/graphql"}'
            )
        return store.resolve_ref(ref)

    source = GraphQLSourceAdapter(
        connection="external-source",
        source_alias="external-source",
        connection_profile_ref="secret://source/connection",
        mapping_policy_ref="secret://source/policy",
        auth_profile_ref="secret://source/auth",
        variables_ref="secret://source/variables",
        resolver=resolve,
        transport=_Transport(),
    )

    with pytest.raises(ExternalGraphSchemaError, match="not valid JSON"):
        source.validate_runtime_profiles()


def test_graphql_proposal_approval_and_readiness_are_pseudonymous() -> None:
    store = _Store(_refs())
    source = _source(store)

    proposal = propose_graphql_mapping_profile(
        source,
        connection="external-source",
        source_alias="external-source",
        secret_store=store,
        max_types=20,
    )

    rendered = json.dumps(proposal, sort_keys=True)
    assert proposal["status"] == "proposed"
    assert proposal["schema"]["type_count"] == 1
    assert "Asset" not in rendered
    assert "source.example.test" not in rendered
    assert "AssetRead" not in rendered
    assert "synthetic-value" not in rendered
    stored = json.loads(store.values["external-graphs/external-source/mapping-profile"])
    assert "identity_hmac_key" not in stored
    assert stored["identity_hmac_key_ref"] == canonical_identity_key_ref(
        "external-source"
    )
    assert len(store.values["external-graphs/external-source/identity-key"]) >= 32

    approved = approve_mapping_profile(
        connection="external-source",
        proposal_id=proposal["proposal_id"],
        proposal_version=proposal["proposal_version"],
        schema_digest=proposal["schema_digest"],
        mapping_digest=proposal["mapping_digest"],
        secret_store=store,
        approver_ref="authenticated-operator",
    )
    assert approved["status"] == "approved"
    readiness = graphql_source_readiness(
        source,
        connection="external-source",
        secret_store=store,
    )
    assert readiness["ready"] is True
    assert "Asset" not in json.dumps(readiness)
    status = graphql_mapping_profile_status(
        source,
        connection="external-source",
        secret_store=store,
    )
    assert status["mapping_drift"] == "none"

    changed = dict(store.refs["secret://source/policy"])
    changed["limits"] = {"max_entities": 10}
    store.refs["secret://source/policy"] = changed
    drifted = graphql_source_readiness(
        source,
        connection="external-source",
        secret_store=store,
    )
    assert drifted["ready"] is False
    assert drifted["mapping_drift"] == "detected"
    drifted_status = graphql_mapping_profile_status(
        source,
        connection="external-source",
        secret_store=store,
    )
    assert drifted_status["mapping_drift"] == "detected"


def test_graphql_introspection_generates_bounded_structural_mapping() -> None:
    refs = _refs()
    refs.pop("secret://source/policy")
    store = _Store(refs)
    transport = _BootstrapTransport([_bootstrap_root()])
    source = _bootstrap_source(store, transport)
    assert source.mapping_policy()["profile_format"] == "graphql-document-policy/v1"

    first = propose_graphql_mapping_profile(
        source,
        connection="external-source",
        source_alias="external-source",
        secret_store=store,
    )
    second = propose_graphql_mapping_profile(
        source,
        connection="external-source",
        source_alias="external-source",
        secret_store=store,
    )

    assert first["generation"] == {
        "mode": "introspection-structural",
        "candidate_count": 1,
        "ambiguous": False,
        "approval": "exact-digest",
    }
    assert second["proposal_id"] == first["proposal_id"]
    assert second["mapping_digest"] == first["mapping_digest"]
    rendered = json.dumps(first, sort_keys=True)
    assert "records" not in rendered
    assert "SyntheticRecord" not in rendered
    assert "source.example.test" not in rendered

    stored = json.loads(store.values["external-graphs/external-source/mapping-profile"])
    operation = stored["operations"]["generated_read_001"]
    assert operation["read_bound"] == {"maximum": 100, "variable": "first"}
    assert operation["mappings"]["entities"]["id_path"] == "id"
    assert operation["mappings"]["entities"]["entity_type"] == "SyntheticRecord"
    assert operation["snapshot_authoritative"] is False
    assert operation["query"].startswith("query GeneratedRead")
    assert "mutation" not in operation["query"].lower()
    assert "subscription" not in operation["query"].lower()

    approve_mapping_profile(
        connection="external-source",
        proposal_id=first["proposal_id"],
        proposal_version=first["proposal_version"],
        schema_digest=first["schema_digest"],
        mapping_digest=first["mapping_digest"],
        secret_store=store,
    )
    readiness = graphql_source_readiness(
        source,
        connection="external-source",
        secret_store=store,
    )
    assert readiness["ready"] is True
    assert readiness["mapping_drift"] == "none"


def test_graphql_introspection_compiles_ambiguity_to_exact_digest_proposal() -> None:
    refs = _refs()
    refs.pop("secret://source/policy")
    store = _Store(refs)
    source = _bootstrap_source(
        store,
        _BootstrapTransport([_bootstrap_root("records"), _bootstrap_root("catalog")]),
    )

    proposal = propose_graphql_mapping_profile(
        source,
        connection="external-source",
        source_alias="external-source",
        secret_store=store,
    )

    assert proposal["generation"]["ambiguous"] is True
    assert proposal["generation"]["candidate_count"] == 2
    assert proposal["semantic_enrichment"] == "privacy-compiled-proposal"
    rendered = json.dumps(proposal, sort_keys=True)
    assert "records" not in rendered
    assert "catalog" not in rendered


def test_graphql_introspection_generates_bounded_cursor_pagination() -> None:
    refs = _refs()
    refs.pop("secret://source/policy")
    store = _Store(refs)
    root = {
        "name": "records",
        "args": [
            {
                "name": "first",
                "defaultValue": None,
                "type": _non_null(_named("SCALAR", "Int")),
            },
            {
                "name": "after",
                "defaultValue": None,
                "type": _named("SCALAR", "String"),
            },
        ],
        "type": _non_null(_named("OBJECT", "SyntheticConnection")),
    }
    transport = _BootstrapTransport([root])
    transport.types.extend(
        [
            {
                "kind": "OBJECT",
                "name": "SyntheticConnection",
                "fields": [
                    {
                        "name": "nodes",
                        "args": [],
                        "type": _list_of(_named("OBJECT", "SyntheticRecord")),
                    },
                    {
                        "name": "pageInfo",
                        "args": [],
                        "type": _named("OBJECT", "PageInfo"),
                    },
                ],
            },
            {
                "kind": "OBJECT",
                "name": "PageInfo",
                "fields": [
                    {
                        "name": "endCursor",
                        "args": [],
                        "type": _named("SCALAR", "String"),
                    },
                    {
                        "name": "hasNextPage",
                        "args": [],
                        "type": _non_null(_named("SCALAR", "Boolean")),
                    },
                ],
            },
        ]
    )

    proposal = propose_graphql_mapping_profile(
        _bootstrap_source(store, transport, allow_empty_snapshot=True),
        connection="external-source",
        source_alias="external-source",
        secret_store=store,
    )
    stored = json.loads(store.values["external-graphs/external-source/mapping-profile"])
    operation = stored["operations"]["generated_read_001"]

    assert proposal["generation"]["candidate_count"] == 1
    assert operation["snapshot_authoritative"] is True
    assert operation["allow_empty_snapshot"] is True
    assert operation["mappings"]["entities"]["records_path"] == "nodes"
    assert operation["pagination"] == {
        "cursor_variable": "after",
        "page_size_variable": "first",
        "next_cursor_path": "records.pageInfo.endCursor",
        "has_more_path": "records.pageInfo.hasNextPage",
    }


@pytest.mark.parametrize(
    "root",
    [
        _bootstrap_root(bound=False),
        _bootstrap_root(required_selector=True),
    ],
)
def test_graphql_introspection_rejects_unbounded_or_required_argument_roots(
    root: dict,
) -> None:
    refs = _refs()
    refs.pop("secret://source/policy")
    store = _Store(refs)

    with pytest.raises(ExternalGraphSchemaError, match="structurally safe bounded"):
        propose_graphql_mapping_profile(
            _bootstrap_source(store, _BootstrapTransport([root])),
            connection="external-source",
            source_alias="external-source",
            secret_store=store,
        )
    assert "external-graphs/external-source/mapping-profile" not in store.values
    assert "external-graphs/external-source/identity-key" not in store.values


def test_graphql_policy_rejects_mutation_before_proposal_is_stored() -> None:
    store = _Store(
        _refs(
            query="mutation AssetWrite($first: Int!) { update(first: $first) { id } }"
        )
    )

    with pytest.raises(ExternalGraphSchemaError):
        propose_graphql_mapping_profile(
            _source(store),
            connection="external-source",
            source_alias="external-source",
            secret_store=store,
        )
    assert "external-graphs/external-source/mapping-profile" not in store.values
    assert "external-graphs/external-source/identity-key" not in store.values


def test_graphql_policy_rejects_nested_transport_material() -> None:
    refs = _refs()
    policy = dict(refs["secret://source/policy"])
    operations = dict(policy["operations"])
    operation = dict(operations["asset_read"])
    operation["headers"] = {"X-Unapproved": "must-not-enter-policy"}
    operations["asset_read"] = operation
    policy["operations"] = operations
    refs["secret://source/policy"] = policy
    store = _Store(refs)

    with pytest.raises(ExternalGraphSchemaError, match="transport or credential"):
        _source(store).mapping_policy()


@pytest.mark.parametrize(
    ("ref", "method_name", "unsupported_format"),
    [
        (
            "secret://source/connection",
            "_connection_profile",
            "graphql-connection/v2",
        ),
        (
            "secret://source/policy",
            "mapping_policy",
            "graphql-document-policy/v2",
        ),
        ("secret://source/auth", "_auth_headers", "graphql-auth/v2"),
    ],
)
@pytest.mark.parametrize("format_mode", ["missing", "unknown"])
def test_graphql_runtime_documents_require_exact_versioned_formats(
    ref: str,
    method_name: str,
    unsupported_format: str,
    format_mode: str,
) -> None:
    refs = _refs()
    document = dict(refs[ref])
    if format_mode == "missing":
        document.pop("profile_format")
    else:
        document["profile_format"] = unsupported_format
    refs[ref] = document
    source = _source(_Store(refs))

    with pytest.raises(ExternalGraphSchemaError, match="format is unsupported"):
        getattr(source, method_name)()


def test_graphql_policy_requires_default_for_multiple_operations() -> None:
    refs = _refs()
    policy = dict(refs["secret://source/policy"])
    policy.pop("default_operation")
    operations = dict(policy["operations"])
    operations["asset_summary"] = dict(operations["asset_read"])
    policy["operations"] = operations
    refs["secret://source/policy"] = policy

    with pytest.raises(ExternalGraphSchemaError, match="require a default alias"):
        _source(_Store(refs)).mapping_policy()


@pytest.mark.parametrize(
    ("field", "value"),
    (("discovery", {"enabled": "true"}), ("limits", {"max_pages": 0})),
)
def test_graphql_policy_rejects_ambiguous_or_unbounded_controls(
    field: str, value: object
) -> None:
    refs = _refs()
    policy = dict(refs["secret://source/policy"])
    policy[field] = value
    refs["secret://source/policy"] = policy

    with pytest.raises(ExternalGraphSchemaError):
        _source(_Store(refs)).mapping_policy()


def test_single_operation_default_is_canonical_across_approval() -> None:
    refs = _refs()
    policy = dict(refs["secret://source/policy"])
    policy.pop("default_operation")
    refs["secret://source/policy"] = policy
    store = _Store(refs)
    source = _source(store)

    proposal = propose_graphql_mapping_profile(
        source,
        connection="external-source",
        source_alias="external-source",
        secret_store=store,
    )
    approve_mapping_profile(
        connection="external-source",
        proposal_id=proposal["proposal_id"],
        proposal_version=proposal["proposal_version"],
        schema_digest=proposal["schema_digest"],
        mapping_digest=proposal["mapping_digest"],
        secret_store=store,
    )

    assert (
        graphql_source_readiness(
            source,
            connection="external-source",
            secret_store=store,
        )["ready"]
        is True
    )


def test_graphql_proposal_resolves_each_runtime_reference_once() -> None:
    store = _Store(_refs())
    calls: dict[str, int] = {}

    def resolve(ref: str) -> str | None:
        calls[ref] = calls.get(ref, 0) + 1
        return store.resolve_ref(ref)

    source = GraphQLSourceAdapter(
        connection="external-source",
        source_alias="external-source",
        connection_profile_ref="secret://source/connection",
        mapping_policy_ref="secret://source/policy",
        auth_profile_ref="secret://source/auth",
        variables_ref="secret://source/variables",
        resolver=resolve,
        transport=_Transport(),
    )

    propose_graphql_mapping_profile(
        source,
        connection="external-source",
        source_alias="external-source",
        secret_store=store,
    )

    assert set(calls.values()) == {1}


def test_discovery_only_policy_requires_synthesizable_read_root() -> None:
    refs = _refs()
    policy = dict(refs["secret://source/policy"])
    policy.pop("default_operation")
    policy.pop("operations")
    refs["secret://source/policy"] = policy
    store = _Store(refs)
    source = _source(store)

    schema, _capabilities, _accepted = source.discover(max_types=20)
    assert schema.public_dict()["type_count"] == 1
    with pytest.raises(ExternalGraphSchemaError, match="read root"):
        propose_graphql_mapping_profile(
            source,
            connection="external-source",
            source_alias="external-source",
            secret_store=store,
        )


def test_bounded_probe_requires_explicit_partial_schema_acknowledgement() -> None:
    store = _Store(_refs(bounded_probe_ack=False))
    transport = _Transport()

    with pytest.raises(ExternalGraphSchemaError, match="partial"):
        propose_graphql_mapping_profile(
            _source(store, transport),
            connection="external-source",
            source_alias="external-source",
            secret_store=store,
        )

    acknowledged = _Store(_refs(bounded_probe_ack=True))
    proposal = propose_graphql_mapping_profile(
        _source(acknowledged, _Transport()),
        connection="external-source",
        source_alias="external-source",
        secret_store=acknowledged,
    )
    assert proposal["schema"]["mode"] == "bounded-probe"


def test_graphql_ingest_uses_approved_profile_and_native_connector_seam(
    monkeypatch,
) -> None:
    store = _Store(_refs())
    source = _source(store)
    proposal = propose_graphql_mapping_profile(
        source,
        connection="external-source",
        source_alias="external-source",
        secret_store=store,
    )
    approve_mapping_profile(
        connection="external-source",
        proposal_id=proposal["proposal_id"],
        proposal_version=proposal["proposal_version"],
        schema_digest=proposal["schema_digest"],
        mapping_digest=proposal["mapping_digest"],
        secret_store=store,
    )
    captured = {}

    async def fake_ingest(_self, manifest):
        captured["manifest"] = manifest
        return SimpleNamespace(
            status="success",
            error=None,
            nodes_created=3,
            edges_created=2,
            details={"envelopes_ingested": 1, "checkpoint_advanced": True},
        )

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.engine.IngestionEngine.ingest",
        fake_ingest,
    )
    result = ingest_registered_graphql(
        SimpleNamespace(backend=None),
        source,
        connection="external-source",
        secret_store=store,
        variables_ref="secret://source/variables",
        contextual=False,
    )

    assert result["status"] == "success"
    assert result["nodes_created"] == 3
    manifest = captured["manifest"]
    assert manifest.source_uri == "graphql_document"
    config = manifest.metadata["connector_config"]
    assert config["profile_ref"].startswith("secret://runtime/")
    assert "endpoint" not in config
    assert "query" not in config
    assert "headers" not in config
    runtime_profile = json.loads(config["profile_resolver"](config["profile_ref"]))
    assert (
        runtime_profile["identity_hmac_key"]
        == store.values["external-graphs/external-source/identity-key"]
    )
    stored_profile = json.loads(
        store.values["external-graphs/external-source/mapping-profile"]
    )
    assert "identity_hmac_key" not in stored_profile


def test_graphql_identity_material_drift_fails_closed() -> None:
    store = _Store(_refs())
    source = _source(store)
    proposal = propose_graphql_mapping_profile(
        source,
        connection="external-source",
        source_alias="external-source",
        secret_store=store,
    )
    approve_mapping_profile(
        connection="external-source",
        proposal_id=proposal["proposal_id"],
        proposal_version=proposal["proposal_version"],
        schema_digest=proposal["schema_digest"],
        mapping_digest=proposal["mapping_digest"],
        secret_store=store,
    )
    profile_key = "external-graphs/external-source/mapping-profile"
    stored = json.loads(store.values[profile_key])
    stored["identity_hmac_key"] = "embedded-secret-material-must-not-be-trusted"
    store.values[profile_key] = json.dumps(stored)

    readiness = graphql_source_readiness(
        source,
        connection="external-source",
        secret_store=store,
    )
    assert readiness["ready"] is False
    assert readiness["status"] == "not_ready"
    with pytest.raises(ExternalGraphSchemaError, match="cannot embed"):
        ingest_registered_graphql(
            SimpleNamespace(backend=None),
            source,
            connection="external-source",
            secret_store=store,
            dry_run=True,
        )
