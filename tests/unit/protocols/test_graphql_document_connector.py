from __future__ import annotations

import json

import pytest

from agent_utilities.protocols.source_connectors import build_connector, list_sources
from agent_utilities.protocols.source_connectors.connectors.graphql_document import (
    GraphQLDocumentError,
)


class _Response:
    def __init__(self, payload: dict, *, status_error: Exception | None = None) -> None:
        self.content = json.dumps(payload, allow_nan=False).encode("utf-8")
        self._status_error = status_error

    def raise_for_status(self) -> None:
        if self._status_error is not None:
            raise self._status_error

    @property
    def headers(self) -> dict[str, str]:
        return {"content-length": str(len(self.content))}

    def iter_bytes(self):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

class _Transport:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def post(self, endpoint: str, **kwargs):
        self.calls.append({"endpoint": endpoint, **kwargs})
        return _Response(self.payload)


class _CursorTransport:
    def __init__(self, first: dict, second: dict) -> None:
        self.first = first
        self.second = second
        self.calls: list[dict] = []

    def post(self, endpoint: str, **kwargs):
        self.calls.append({"endpoint": endpoint, **kwargs})
        cursor = kwargs["json"]["variables"].get("after")
        return _Response(self.first if cursor is None else self.second)


class _SequenceTransport:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def post(self, endpoint: str, **kwargs):
        self.calls.append({"endpoint": endpoint, **kwargs})
        return _Response(self.payloads.pop(0))


def _profile() -> dict:
    return {
        "endpoint": "https://knowledge.example.test/graphql",
        "identity_hmac_key": "synthetic-test-key-material-32-bytes",
        "headers": {"Authorization": "Bearer synthetic-token"},
        "operations": {
            "entity_document": {
                "query": (
                    "query EntityDocument($slug: String!) { "
                    "entity(slug: $slug) { name slug document { title "
                    "frontmatter sections { title level content } } } }"
                ),
                "root_path": "entity",
                "id_path": "slug",
                "title_path": "document.title",
                "frontmatter_path": "document.frontmatter",
                "sections_path": "document.sections",
            }
        },
    }


def _payload() -> dict:
    return {
        "data": {
            "entity": {
                "name": "Synthetic Lifecycle",
                "slug": "synthetic-lifecycle",
                "document": {
                    "title": "Synthetic Lifecycle Guide",
                    "frontmatter": {
                        "classification": "internal",
                        "created_by": "Example Person",
                    },
                    "sections": [
                        {
                            "title": "Overview",
                            "level": 1,
                            "content": (
                                "Contact contact@example.test. "
                                "Local draft: /home/example/private/draft.md"
                            ),
                        }
                    ],
                },
            }
        }
    }


def _hierarchy_profile() -> dict:
    return {
        "endpoint": "https://hierarchy.example.test/graphql",
        "identity_hmac_key": "synthetic-hierarchy-key-material-32-bytes",
        "headers": {"Authorization": "Bearer synthetic-hierarchy-token"},
        "limits": {
            "max_pages": 4,
            "page_size": 2,
            "max_entities": 50,
            "max_documents": 10,
        },
        "governance": {
            "classification": "confidential",
            "retention": "P30D",
            "legal_hold": False,
            "tenant": "synthetic-tenant",
            "schema_version": "2",
            "ontology_mapping_version": "synthetic-v1",
            "access": {
                "is_public": False,
                "group_ids": ["synthetic-readers"],
                "markings": ["synthetic-restricted"],
            },
        },
        "operations": {
            "hierarchy_sync": {
                "query": (
                    "query Hierarchy($after: String, $first: Int!) { "
                    "catalog(after: $after, first: $first) { units documents "
                    "applications dependencies pageInfo { endCursor hasNextPage } } }"
                ),
                "root_path": "catalog",
                "pagination": {
                    "cursor_variable": "after",
                    "page_size_variable": "first",
                    "next_cursor_path": "catalog.pageInfo.endCursor",
                    "has_more_path": "catalog.pageInfo.hasNextPage",
                },
                "partial_errors": {
                    "codes": ["SOURCE_FIELD_UNAVAILABLE"],
                    "paths": ["catalog.advisory"],
                },
                "mappings": {
                    "hierarchy": {
                        "records_path": "units",
                        "id_path": "id",
                        "children_path": "children",
                        "version_path": "updated",
                        "property_allowlist": ["name"],
                        "entity_type": "HierarchyUnit",
                    },
                    "documents": {
                        "records_path": "documents",
                        "id_path": "id",
                        "version_path": "updated",
                        "title_path": "title",
                        "content_path": "content",
                        "parent_id_path": "unit_id",
                        "application_id_path": "application_id",
                        "property_allowlist": ["category"],
                        "doc_type": "hierarchy_document",
                    },
                    "applications": {
                        "records_path": "applications",
                        "id_path": "id",
                        "version_path": "updated",
                        "parent_id_path": "unit_id",
                        "property_allowlist": ["name", "runtime"],
                    },
                    "dependencies": {
                        "records_path": "dependencies",
                        "id_path": "id",
                        "version_path": "updated",
                        "source_id_path": "source_id",
                        "target_id_path": "target_id",
                        "property_allowlist": ["criticality"],
                    },
                },
            }
        },
    }


def _hierarchy_pages() -> tuple[dict, dict]:
    first = {
        "data": {
            "catalog": {
                "units": [
                    {
                        "id": "unit-root",
                        "name": "Synthetic Root",
                        "updated": "v1",
                        "children": [
                            {
                                "id": "unit-child",
                                "name": "Synthetic Child",
                                "updated": "v1",
                                "children": [],
                            }
                        ],
                    }
                ],
                "documents": [
                    {
                        "id": "doc-one",
                        "title": "Synthetic Guide One",
                        "content": "Bounded synthetic content one.",
                        "category": "guide",
                        "unit_id": "unit-root",
                        "application_id": "app-one",
                        "updated": "v1",
                    }
                ],
                "applications": [
                    {
                        "id": "app-one",
                        "name": "Synthetic Application One",
                        "runtime": "synthetic",
                        "unit_id": "unit-root",
                        "updated": "v1",
                    }
                ],
                "dependencies": [],
                "pageInfo": {"endCursor": "cursor-one", "hasNextPage": True},
            }
        },
        "errors": [
            {
                "message": "synthetic optional field unavailable",
                "path": ["catalog", "advisory"],
                "extensions": {"code": "SOURCE_FIELD_UNAVAILABLE"},
            }
        ],
    }
    second = {
        "data": {
            "catalog": {
                "units": [],
                "documents": [
                    {
                        "id": "doc-two",
                        "title": "Synthetic Guide Two",
                        "content": "Bounded synthetic content two.",
                        "category": "guide",
                        "unit_id": "unit-child",
                        "application_id": "app-two",
                        "updated": "v1",
                    }
                ],
                "applications": [
                    {
                        "id": "app-two",
                        "name": "Synthetic Application Two",
                        "runtime": "synthetic",
                        "unit_id": "unit-child",
                        "updated": "v1",
                    }
                ],
                "dependencies": [
                    {
                        "id": "dependency-one",
                        "source_id": "app-one",
                        "target_id": "app-two",
                        "criticality": "synthetic",
                        "updated": "v1",
                    }
                ],
                "pageInfo": {"endCursor": None, "hasNextPage": False},
            }
        }
    }
    return first, second


@pytest.mark.concept("AU-KG.ingest.universal-data-connector")
def test_graphql_document_is_registered_and_requires_secret_backed_profile() -> None:
    assert "graphql_document" in set(list_sources())
    with pytest.raises(ValueError, match="profile_ref"):
        build_connector(
            "graphql_document",
            {
                "source_alias": "knowledge-catalog",
                "operation": "entity_document",
                "profile": _profile(),
            },
        )


def test_graphql_discovery_callable_uses_runtime_tls_profile(monkeypatch) -> None:
    profile = _profile()
    profile["discovery"] = {"enabled": True, "allow_introspection": True}
    profile["tls"] = {"settings": {"system_trust": True}}
    captured: dict = {}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def stream(self, _method: str, _endpoint: str, **_kwargs):
            return _Response(
                {"data": {"__schema": {"queryType": {"name": "Query"}}}}
            )

    def _client_factory(**kwargs):
        captured.update(kwargs)
        return _Client()

    monkeypatch.setattr(
        "agent_utilities.core.http_client.create_http_client", _client_factory
    )
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.http_safety.require_safe_source_url",
        lambda *_args, **_kwargs: None,
    )
    connector = build_connector(
        "graphql_document",
        {
            "source_alias": "schema-catalog",
            "operation": "entity_document",
            "profile_ref": "secret://integrations/schema-catalog/graphql",
            "profile_resolver": lambda _ref: json.dumps(profile),
        },
    )

    result = connector.execute(
        "query SchemaProbe { __schema { queryType { name } } }", {}
    )

    assert result["data"]["__schema"]["queryType"]["name"] == "Query"
    assert captured["verify"] is connector._resolved_tls.ssl_context
    assert captured["trust_env"] is True


@pytest.mark.concept("AU-KG.ingest.universal-data-connector")
def test_graphql_document_ingests_full_document_without_persisting_source_details() -> (
    None
):
    transport = _Transport(_payload())
    profile_json = json.dumps(_profile())
    connector = build_connector(
        "graphql_document",
        {
            "source_alias": "knowledge-catalog",
            "operation": "entity_document",
            "variables": {"slug": "synthetic-lifecycle"},
            "profile_ref": "vault://integrations/knowledge-catalog/graphql",
            "profile_resolver": lambda _ref: profile_json,
            "transport": transport,
        },
    )

    documents = list(connector.load())

    assert len(documents) == 1
    assert "verify" not in transport.calls[0]
    assert "trust_env" not in transport.calls[0]
    doc = documents[0]
    assert doc.title == "Synthetic Lifecycle Guide"
    assert "# Synthetic Lifecycle Guide" in doc.text
    assert "[REDACTED_PERSON]" in doc.text
    assert "contact@example.test" not in doc.text
    assert "/home/example" not in doc.text
    assert doc.source_uri.startswith("external-source://knowledge-catalog/")
    assert doc.external_access is not None
    assert doc.external_access.is_public is False
    assert doc.external_access.markings == ["connector-unconfigured-acl"]

    persisted_shape = json.dumps(doc.model_dump(), sort_keys=True)
    assert "knowledge.example.test" not in persisted_shape
    assert "synthetic-token" not in persisted_shape
    assert "synthetic-test-key-material" not in persisted_shape
    assert "vault://" not in persisted_shape
    assert "query EntityDocument" not in persisted_shape
    assert doc.metadata["privacy"]["redactions"] == 3

    request = transport.calls[0]
    assert request["json"]["variables"] == {"slug": "synthetic-lifecycle"}
    assert request["headers"]["Authorization"] == "Bearer synthetic-token"


@pytest.mark.concept("AU-KG.ingest.universal-data-connector")
def test_graphql_document_rejects_profile_without_private_identity_key() -> None:
    profile = _profile()
    profile.pop("identity_hmac_key")
    connector = build_connector(
        "graphql_document",
        {
            "source_alias": "knowledge-catalog",
            "operation": "entity_document",
            "profile": profile,
            "transport": _Transport(_payload()),
        },
    )

    with pytest.raises(GraphQLDocumentError, match="identity HMAC"):
        list(connector.load())


@pytest.mark.concept("AU-KG.ingest.universal-data-connector")
def test_graphql_document_rejects_private_egress_without_exact_allowlist() -> None:
    profile = _profile()
    profile["endpoint"] = "https://127.0.0.1/graphql"
    connector = build_connector(
        "graphql_document",
        {
            "source_alias": "synthetic-source",
            "operation": "entity_document",
            "profile": profile,
            "transport": _Transport(_payload()),
        },
    )

    with pytest.raises(GraphQLDocumentError, match="not permitted"):
        list(connector.load())


@pytest.mark.concept("AU-KG.ingest.universal-data-connector")
def test_graphql_document_rejects_header_injection() -> None:
    profile = _profile()
    profile["headers"] = {"X-Source": "value\r\nInjected: value"}
    connector = build_connector(
        "graphql_document",
        {
            "source_alias": "synthetic-source",
            "operation": "entity_document",
            "profile": profile,
            "transport": _Transport(_payload()),
        },
    )

    with pytest.raises(GraphQLDocumentError, match="headers are invalid"):
        list(connector.load())


@pytest.mark.concept("AU-KG.ingest.universal-data-connector")
def test_graphql_document_rejects_hop_by_hop_headers() -> None:
    profile = _profile()
    profile["headers"] = {"Proxy-Authorization": "synthetic-value"}
    connector = build_connector(
        "graphql_document",
        {
            "source_alias": "synthetic-source",
            "operation": "entity_document",
            "profile": profile,
            "transport": _Transport(_payload()),
        },
    )

    with pytest.raises(GraphQLDocumentError, match="headers are invalid"):
        list(connector.load())


@pytest.mark.concept("AU-KG.ingest.universal-data-connector")
def test_graphql_source_errors_do_not_echo_upstream_messages() -> None:
    transport = _Transport(
        {
            "errors": [
                {
                    "message": (
                        "private endpoint rejected contact@example.test "
                        "from /home/example/private"
                    )
                }
            ]
        }
    )
    connector = build_connector(
        "graphql_document",
        {
            "source_alias": "knowledge-catalog",
            "operation": "entity_document",
            "profile": _profile(),
            "transport": transport,
        },
    )

    with pytest.raises(GraphQLDocumentError) as exc:
        list(connector.load())

    message = str(exc.value)
    assert message == "GraphQL source returned an error"
    assert "contact@example.test" not in message
    assert "/home/example" not in message


@pytest.mark.parametrize(
    "query",
    [
        "mutation UpdateEntity { updateEntity { id } }",
        "subscription EntityChanges { entityChanged { id } }",
        "query Inspect { __schema { queryType { name } } }",
    ],
)
def test_graphql_document_rejects_non_ingest_operations(query: str) -> None:
    profile = _profile()
    profile["operations"]["entity_document"]["query"] = query
    connector = build_connector(
        "graphql_document",
        {
            "source_alias": "knowledge-catalog",
            "operation": "entity_document",
            "profile": profile,
            "transport": _Transport(_payload()),
        },
    )

    with pytest.raises(GraphQLDocumentError):
        list(connector.load())


def test_graphql_document_ast_allows_keywords_only_inside_query_data() -> None:
    profile = _profile()
    profile["operations"]["entity_document"]["query"] = (
        'query MutationReport { entity(note: "mutation subscription") { '
        "slug document { title sections { title level content } } } }"
    )
    connector = build_connector(
        "graphql_document",
        {
            "source_alias": "knowledge-catalog",
            "operation": "entity_document",
            "profile": profile,
            "transport": _Transport(_payload()),
        },
    )

    assert len(list(connector.load())) == 1


def test_graphql_document_ast_rejects_ambiguous_multi_operation_document() -> None:
    profile = _profile()
    profile["operations"]["entity_document"]["query"] = (
        "query First { entity { slug } } query Second { entity { slug } }"
    )
    connector = build_connector(
        "graphql_document",
        {
            "source_alias": "knowledge-catalog",
            "operation": "entity_document",
            "profile": profile,
            "transport": _Transport(_payload()),
        },
    )

    with pytest.raises(GraphQLDocumentError, match="read query"):
        list(connector.load())


def test_graphql_document_custom_transport_requires_raw_bounded_bytes() -> None:
    class JsonOnlyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return _payload()

    class JsonOnlyTransport:
        def post(self, _endpoint: str, **_kwargs):
            return JsonOnlyResponse()

    connector = build_connector(
        "graphql_document",
        {
            "source_alias": "knowledge-catalog",
            "operation": "entity_document",
            "profile": _profile(),
            "transport": JsonOnlyTransport(),
        },
    )

    with pytest.raises(GraphQLDocumentError, match="bounded byte response"):
        list(connector.load())


def test_graphql_document_rejects_response_over_bound() -> None:
    payload = _payload()
    payload["data"]["entity"]["document"]["sections"][0]["content"] = "x" * 5000
    connector = build_connector(
        "graphql_document",
        {
            "source_alias": "knowledge-catalog",
            "operation": "entity_document",
            "profile": _profile(),
            "transport": _Transport(payload),
            "max_response_bytes": 1024,
        },
    )

    with pytest.raises(GraphQLDocumentError, match="configured bound"):
        list(connector.load())


@pytest.mark.concept("AU-ECO.connector.incremental-poll-watermark")
def test_graphql_document_poll_is_idempotent() -> None:
    connector = build_connector(
        "graphql_document",
        {
            "source_alias": "knowledge-catalog",
            "operation": "entity_document",
            "profile": _profile(),
            "transport": _Transport(_payload()),
        },
    )

    first = connector.poll()
    second = connector.poll(first.checkpoint)

    assert len(first.documents) == 1
    assert second.documents == []
    assert second.checkpoint.watermark == first.checkpoint.watermark
    assert second.checkpoint.state["snapshot_sequence"] == 1


@pytest.mark.concept("AU-KG.ingest.change-envelope")
def test_graphql_hierarchy_maps_bounded_governed_envelopes_and_documents() -> None:
    first_page, second_page = _hierarchy_pages()
    transport = _CursorTransport(first_page, second_page)
    connector = build_connector(
        "graphql_document",
        {
            "source_alias": "synthetic-hierarchy",
            "operation": "hierarchy_sync",
            "profile": _hierarchy_profile(),
            "transport": transport,
        },
    )

    first = connector.poll()

    assert len(first.documents) == 2
    assert len(connector.last_envelopes) == 7
    assert first.checkpoint.has_more is False
    assert len(first.checkpoint.state["versions"]) == 7
    assert first.documents[0].metadata["embedding_handoff"] is True
    assert first.documents[0].external_access.group_ids == ["synthetic-readers"]
    assert {envelope.classification.value for envelope in connector.last_envelopes} == {
        "confidential"
    }
    assert {envelope.retention for envelope in connector.last_envelopes} == {"P30D"}
    assert all(envelope.checkpoint for envelope in connector.last_envelopes)
    assert all(
        envelope.provenance["identity_scheme"] == "hmac-sha256"
        for envelope in connector.last_envelopes
    )
    dependency = next(
        envelope
        for envelope in connector.last_envelopes
        if envelope.typed_payload["entity_kind"] == "dependency"
    )
    assert dependency.typed_payload["_links"][0]["type"] == "DEPENDS_ON"

    persisted = json.dumps(
        {
            "documents": [document.model_dump() for document in first.documents],
            "envelopes": [envelope.as_dict() for envelope in connector.last_envelopes],
            "checkpoint": first.checkpoint.model_dump(),
        },
        sort_keys=True,
        default=str,
    )
    for private_value in (
        "unit-root",
        "unit-child",
        "doc-one",
        "doc-two",
        "app-one",
        "app-two",
        "dependency-one",
        "synthetic-hierarchy-key-material",
        "synthetic-hierarchy-token",
        "query Hierarchy",
    ):
        assert private_value not in persisted

    assert transport.calls[0]["json"]["variables"] == {
        "after": None,
        "first": 2,
    }
    assert transport.calls[1]["json"]["variables"] == {
        "after": "cursor-one",
        "first": 2,
    }

    second = connector.poll(first.checkpoint)
    assert second.documents == []
    assert connector.last_envelopes == []


@pytest.mark.parametrize(
    ("has_more", "cursor", "message"),
    [
        ("false", "cursor-one", "continuation flag is not boolean"),
        (1, "cursor-one", "continuation flag is not boolean"),
        (True, 7, "invalid continuation"),
        (True, "x" * 4_097, "invalid continuation"),
        (True, "cursor\x00one", "invalid continuation"),
        (True, " cursor-one", "invalid continuation"),
    ],
)
def test_graphql_pagination_requires_bounded_typed_continuation(
    has_more, cursor, message
) -> None:
    profile = _hierarchy_profile()
    first_page, second_page = _hierarchy_pages()
    first_page["data"]["catalog"]["pageInfo"] = {
        "endCursor": cursor,
        "hasNextPage": has_more,
    }
    connector = build_connector(
        "graphql_document",
        {
            "source_alias": "synthetic-hierarchy",
            "operation": "hierarchy_sync",
            "profile": profile,
            "transport": _CursorTransport(first_page, second_page),
        },
    )

    with pytest.raises(GraphQLDocumentError, match=message):
        connector.poll()


@pytest.mark.concept("AU-KG.ingest.change-envelope")
def test_graphql_authoritative_snapshot_emits_governed_tombstone_once() -> None:
    profile = _hierarchy_profile()
    profile["operations"]["hierarchy_sync"]["snapshot_authoritative"] = True
    first_page, second_page = _hierarchy_pages()
    first_page.pop("errors")
    transport = _CursorTransport(first_page, second_page)
    connector = build_connector(
        "graphql_document",
        {
            "source_alias": "synthetic-hierarchy",
            "operation": "hierarchy_sync",
            "profile": profile,
            "transport": transport,
        },
    )

    initial = connector.poll_envelopes()
    assert initial.checkpoint.state["checkpoint_format"] == (
        "graphql-snapshot-checkpoint/v1"
    )
    assert initial.checkpoint.state["snapshot_sequence"] == 1

    removed_node_id = next(
        envelope.source_object_id
        for envelope in initial.envelopes
        if envelope.typed_payload
        and envelope.typed_payload.get("entity_kind") == "document"
        and envelope.typed_payload.get("title") == "Synthetic Guide Two"
    )
    second_page["data"]["catalog"]["documents"] = []
    reconciled = connector.poll_envelopes(initial.checkpoint)
    tombstones = [
        envelope for envelope in reconciled.envelopes if envelope.operation == "delete"
    ]

    assert len(tombstones) == 1
    tombstone = tombstones[0]
    assert tombstone.source_object_id == removed_node_id
    assert tombstone.tenant == "synthetic-tenant"
    assert tombstone.classification.value == "confidential"
    assert tombstone.retention == "P30D"
    assert tombstone.legal_hold is False
    assert tombstone.source_acl is not None
    assert tombstone.source_acl.group_ids == ["synthetic-readers"]
    assert tombstone.provenance["snapshot_reconciliation"] is True
    assert reconciled.checkpoint.state["snapshot_sequence"] == 2

    stable = connector.poll_envelopes(reconciled.checkpoint)
    assert [item for item in stable.envelopes if item.operation == "delete"] == []


def test_graphql_authoritative_snapshot_fails_closed_when_partial() -> None:
    profile = _hierarchy_profile()
    profile["operations"]["hierarchy_sync"]["snapshot_authoritative"] = True
    first_page, second_page = _hierarchy_pages()
    connector = build_connector(
        "graphql_document",
        {
            "source_alias": "synthetic-hierarchy",
            "operation": "hierarchy_sync",
            "profile": profile,
            "transport": _CursorTransport(first_page, second_page),
        },
    )

    with pytest.raises(GraphQLDocumentError, match="did not complete"):
        connector.poll_envelopes()


def test_graphql_authoritative_empty_snapshot_requires_explicit_approval() -> None:
    profile = _hierarchy_profile()
    profile["operations"]["hierarchy_sync"]["snapshot_authoritative"] = True
    first_page, second_page = _hierarchy_pages()
    first_page.pop("errors")
    transport = _CursorTransport(first_page, second_page)
    connector = build_connector(
        "graphql_document",
        {
            "source_alias": "synthetic-hierarchy",
            "operation": "hierarchy_sync",
            "profile": profile,
            "transport": transport,
        },
    )
    initial = connector.poll_envelopes()
    transport.first = {
        "data": {
            "catalog": {
                "units": [],
                "documents": [],
                "applications": [],
                "dependencies": [],
                "pageInfo": {"endCursor": None, "hasNextPage": False},
            }
        }
    }

    with pytest.raises(GraphQLDocumentError, match="requires explicit approval"):
        connector.poll_envelopes(initial.checkpoint)


def test_graphql_approved_authoritative_empty_snapshot_tombstones_baseline() -> None:
    profile = _hierarchy_profile()
    operation = profile["operations"]["hierarchy_sync"]
    operation["snapshot_authoritative"] = True
    operation["allow_empty_snapshot"] = True
    first_page, second_page = _hierarchy_pages()
    first_page.pop("errors")
    transport = _CursorTransport(first_page, second_page)
    connector = build_connector(
        "graphql_document",
        {
            "source_alias": "synthetic-hierarchy",
            "operation": "hierarchy_sync",
            "profile": profile,
            "transport": transport,
        },
    )
    initial = connector.poll_envelopes()
    transport.first = {
        "data": {
            "catalog": {
                "units": [],
                "documents": [],
                "applications": [],
                "dependencies": [],
                "pageInfo": {"endCursor": None, "hasNextPage": False},
            }
        }
    }

    reconciled = connector.poll_envelopes(initial.checkpoint)
    tombstones = [
        envelope for envelope in reconciled.envelopes if envelope.operation == "delete"
    ]

    assert len(tombstones) == 7
    assert all(
        envelope.provenance["authoritative_empty_approved"] is True
        for envelope in tombstones
    )
    assert reconciled.checkpoint.state["allow_empty_snapshot"] is True


def test_graphql_hierarchy_optional_field_fallback_is_allowlisted_and_bounded() -> None:
    profile = _hierarchy_profile()
    operation = profile["operations"]["hierarchy_sync"]
    operation.pop("pagination")
    operation["optional_field_fallbacks"] = [
        {
            "query": (
                "query HierarchyFallback { catalog { units documents "
                "applications dependencies } }"
            ),
            "codes": ["GRAPHQL_VALIDATION_FAILED"],
            "paths": ["catalog.documents.optionalSummary"],
        }
    ]
    failed = {
        "errors": [
            {
                "message": "synthetic validation detail",
                "path": ["catalog", "documents", "optionalSummary"],
                "extensions": {"code": "GRAPHQL_VALIDATION_FAILED"},
            }
        ]
    }
    first_page, _ = _hierarchy_pages()
    first_page.pop("errors")
    first_page["data"]["catalog"]["pageInfo"]["hasNextPage"] = False
    transport = _SequenceTransport([failed, first_page])
    connector = build_connector(
        "graphql_document",
        {
            "source_alias": "synthetic-hierarchy",
            "operation": "hierarchy_sync",
            "profile": profile,
            "transport": transport,
        },
    )

    plan = connector.plan()

    assert len(transport.calls) == 2
    assert plan["counts"]["fallbacks"] == 1
    assert plan["counts"]["entities"] == 4


def test_graphql_hierarchy_rejects_non_allowlisted_partial_errors() -> None:
    profile = _hierarchy_profile()
    first_page, second_page = _hierarchy_pages()
    first_page["errors"][0]["extensions"]["code"] = "UNAPPROVED_ERROR"
    connector = build_connector(
        "graphql_document",
        {
            "source_alias": "synthetic-hierarchy",
            "operation": "hierarchy_sync",
            "profile": profile,
            "transport": _CursorTransport(first_page, second_page),
        },
    )

    with pytest.raises(GraphQLDocumentError, match="source returned an error"):
        connector.plan()


def test_graphql_hierarchy_requires_explicit_retention_governance() -> None:
    profile = _hierarchy_profile()
    profile["governance"].pop("retention")
    first_page, second_page = _hierarchy_pages()
    connector = build_connector(
        "graphql_document",
        {
            "source_alias": "synthetic-hierarchy",
            "operation": "hierarchy_sync",
            "profile": profile,
            "transport": _CursorTransport(first_page, second_page),
        },
    )

    with pytest.raises(GraphQLDocumentError, match="retention policy"):
        connector.plan()


def test_graphql_hierarchy_dry_run_returns_only_counts_and_digests() -> None:
    first_page, second_page = _hierarchy_pages()
    connector = build_connector(
        "graphql_document",
        {
            "source_alias": "synthetic-hierarchy",
            "operation": "hierarchy_sync",
            "profile": _hierarchy_profile(),
            "transport": _CursorTransport(first_page, second_page),
            "dry_run": True,
        },
    )

    plan = connector.plan()
    persisted = json.dumps(plan, sort_keys=True)

    assert plan["status"] == "planned"
    assert plan["counts"]["entities"] == 7
    assert plan["entity_counts"] == {
        "hierarchy": 2,
        "document": 2,
        "application": 2,
        "dependency": 1,
    }
    assert list(connector.load()) == []
    assert connector.last_envelopes == []
    for private_value in (
        "unit-root",
        "doc-one",
        "app-one",
        "Bounded synthetic content",
        "hierarchy.example.test",
        "synthetic-hierarchy-key-material",
        "synthetic-hierarchy-token",
        "query Hierarchy",
    ):
        assert private_value not in persisted


def test_graphql_hierarchy_rejects_repeated_pagination_cursor() -> None:
    first_page, second_page = _hierarchy_pages()
    second_page["data"]["catalog"]["pageInfo"] = {
        "endCursor": "cursor-one",
        "hasNextPage": True,
    }
    connector = build_connector(
        "graphql_document",
        {
            "source_alias": "synthetic-hierarchy",
            "operation": "hierarchy_sync",
            "profile": _hierarchy_profile(),
            "transport": _CursorTransport(first_page, second_page),
        },
    )

    with pytest.raises(GraphQLDocumentError, match="invalid continuation"):
        connector.plan()


def test_graphql_hierarchy_depth_is_bounded() -> None:
    profile = _hierarchy_profile()
    operation = profile["operations"]["hierarchy_sync"]
    operation.pop("pagination")
    first_page, _ = _hierarchy_pages()
    first_page.pop("errors")
    first_page["data"]["catalog"]["pageInfo"]["hasNextPage"] = False
    connector = build_connector(
        "graphql_document",
        {
            "source_alias": "synthetic-hierarchy",
            "operation": "hierarchy_sync",
            "profile": profile,
            "transport": _Transport(first_page),
            "max_hierarchy_depth": 1,
        },
    )

    plan = connector.plan()

    assert plan["entity_counts"]["hierarchy"] == 1
    assert plan["counts"]["truncated"] == 1
