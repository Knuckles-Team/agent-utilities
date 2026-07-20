from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.ingestion.external_graph import (
    ExternalGraphIngestionError,
    ExternalGraphIngestionRequest,
    ingest_registered_graph,
)
from agent_utilities.knowledge_graph.ingestion.external_graph_schema import (
    external_mapping_policy_digest,
    mapping_policy_digest,
)
from agent_utilities.models.company_brain import DataClassification


@pytest.fixture(autouse=True)
def _certified_external_graph_bundle(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ontology.connector_manifest_gate.precheck_source",
        lambda source: {
            "checked": True,
            "ok": source == "external_graph",
            "connector": "native-source-connectors",
        },
    )


class _ExternalEngine:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, dict]] = []

    def execute_read(self, query: str, params: dict):
        self.calls.append((query, params))
        if self.fail:
            raise RuntimeError(
                "remote path /home/example/private and contact@example.test"
            )
        if " SET " in query:
            raise RuntimeError("read-only transaction")
        if "ExternalNode" in query:
            return [
                {
                    "id": "raw-node-a",
                    "kind": "Capability",
                    "version": "2026-07-01",
                    "properties": {
                        "title": "Synthetic Capability",
                        "description": (
                            "Contact contact@example.test; draft "
                            "/home/example/private/capability.md"
                        ),
                        "owner_name": "Example Person",
                        "not_allowed": "must not cross the allowlist",
                    },
                },
                {
                    "id": "raw-node-b",
                    "kind": "Process",
                    "version": "2026-07-02",
                    "properties": {
                        "title": "Synthetic Process",
                        "description": "A governed process",
                        "owner_name": "Example Owner",
                    },
                },
                {
                    "id": "raw-person",
                    "kind": "Person",
                    "version": "2026-07-02",
                    "properties": {"title": "Example Individual"},
                },
            ]
        return [
            {
                "source": "raw-node-a",
                "target": "raw-node-b",
                "kind": "DEPENDS_ON",
                "properties": {"confidence": 0.9, "owner_name": "Example Owner"},
            },
            {
                "source": "raw-person",
                "target": "raw-node-b",
                "kind": "OWNS",
                "properties": {"confidence": 1.0},
            },
        ]


class _Registry:
    def __init__(self, engine, *, role: str = "read") -> None:
        self.engine = engine
        self.connection_role = role

    def role(self, name: str) -> str:
        assert name == "external-catalog"
        return self.connection_role

    def get_engine(self, name: str):
        assert name == "external-catalog"
        return self.engine


def _profile() -> dict:
    return {
        "identity_hmac_key_ref": (
            "vault://external-graphs/external-catalog/identity-key"
        ),
        "node_query": (
            "MATCH (n:ExternalNode) RETURN n.id AS id, n.kind AS kind, "
            "n.version AS version, properties(n) AS properties "
            "ORDER BY id SKIP $offset LIMIT $limit"
        ),
        "node_mapping": {
            "id_path": "id",
            "type_path": "kind",
            "version_path": "version",
            "properties_path": "properties",
            "property_allowlist": ["title", "description", "owner_name"],
        },
        "edge_query": (
            "MATCH (a)-[r]->(b) RETURN a.id AS source, b.id AS target, "
            "type(r) AS kind, properties(r) AS properties "
            "ORDER BY source, target, kind SKIP $offset LIMIT $limit"
        ),
        "edge_mapping": {
            "source_path": "source",
            "target_path": "target",
            "type_path": "kind",
            "properties_path": "properties",
            "property_allowlist": ["confidence", "owner_name"],
        },
        "type_map": {"Capability": "Capability", "Process": "BusinessProcess"},
        "access": {"is_public": False, "markings": ["external-import"]},
    }


def _request(*, dry_run: bool = False) -> ExternalGraphIngestionRequest:
    return ExternalGraphIngestionRequest(
        connection="external-catalog",
        source_alias="business-graph",
        profile_ref="vault://integrations/business-graph/import-profile",
        variables={"scope": "synthetic"},
        max_records=50,
        classification=DataClassification.CONFIDENTIAL,
        retention="P30D",
        dry_run=dry_run,
    )


def _approved_profile() -> tuple[dict, str]:
    runtime_policy_digest = external_mapping_policy_digest(
        {"property_allowlist": ["title"]}
    )
    profile = {
        **_profile(),
        "profile_format": "external-graph-profile/v1",
        "approval_status": "approved",
        "source_alias": "business-graph",
        "runtime_policy_digest": runtime_policy_digest,
        "sync": {
            "allow_empty_snapshot": False,
            "max_collection_items": 10_000,
            "max_nesting_depth": 16,
            "max_pages": 100,
            "max_row_bytes": 1_048_576,
            "max_total_bytes": 16_777_216,
            "page_size": 500,
            "reconcile_deletions": True,
            "sync_mode": "auto",
        },
    }
    profile["mapping_digest"] = mapping_policy_digest(profile)
    return profile, runtime_policy_digest


def _approved_profile_resolver(profile: dict):
    def resolve(ref: str) -> str | None:
        if ref == "vault://integrations/business-graph/import-profile":
            return json.dumps(profile)
        if ref == "vault://external-graphs/external-catalog/identity-key":
            return "synthetic-test-key-material-32-bytes"
        return None

    return resolve


@pytest.fixture(autouse=True)
def _identity_key_secret(monkeypatch):
    class _Secrets:
        @staticmethod
        def resolve_ref(ref: str) -> str | None:
            if ref == "vault://external-graphs/external-catalog/identity-key":
                return "synthetic-test-key-material-32-bytes"
            return None

    monkeypatch.setattr(
        "agent_utilities.security.secrets_client.create_secrets_client",
        lambda: _Secrets(),
    )


def test_external_graph_manifest_gate_fails_before_profile_or_source_read(
    monkeypatch,
) -> None:
    external = _ExternalEngine()
    registry = _Registry(external)
    profile_resolved = False

    def resolve_profile(_ref: str) -> str:
        nonlocal profile_resolved
        profile_resolved = True
        return "{}"

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ontology.connector_manifest_gate.precheck_source",
        lambda _source: {"checked": True, "ok": False},
    )

    with pytest.raises(ExternalGraphIngestionError, match="certified capability"):
        ingest_registered_graph(
            object(), registry, _request(), profile_resolver=resolve_profile
        )

    assert profile_resolved is False
    assert external.calls == []


@pytest.mark.concept("AU-KG.ingest.external-graph-federation")
def test_external_graph_ingestion_uses_envelopes_and_never_persists_raw_identity(
    monkeypatch,
) -> None:
    captured = []

    def fake_ingest(_engine, envelope):
        captured.append(envelope)
        return {"status": "success"}

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.external_graph.ingest_envelope",
        fake_ingest,
    )
    result = ingest_registered_graph(
        object(),
        _Registry(_ExternalEngine()),
        _request(),
        profile=_profile(),
    )

    assert result["status"] == "success"
    assert result["nodes"] == 2  # the Person row is dropped fail-closed
    assert result["edges"] == 1
    assert result["privacy"]["redactions"] == 6
    assert set(result["privacy"]["detected_types"]) >= {
        "email",
        "personal_entity",
        "personal_field",
        "posix_user_path",
    }
    assert len(captured) == 3

    serialized = json.dumps(
        [envelope.as_dict() for envelope in captured], sort_keys=True
    )
    for forbidden in (
        "raw-node-a",
        "raw-node-b",
        "raw-person",
        "contact@example.test",
        "/home/example",
        "Example Person",
        "not_allowed",
        "vault://",
        "synthetic-test-key-material",
        "MATCH (n:ExternalNode)",
    ):
        assert forbidden not in serialized

    first = captured[0]
    assert first.source_object_id.startswith("external:business-graph:")
    assert first.classification == DataClassification.CONFIDENTIAL
    assert first.retention == "P30D"
    assert first.source_acl is not None
    assert first.source_acl.markings == ["external-import"]
    assert first.typed_payload is not None
    assert first.typed_payload["externalToolId"] == first.source_object_id
    assert first.typed_payload["owner_name"] == "[REDACTED_PERSON]"
    assert len(first.typed_payload["_links"]) == 1
    marker = captured[-1]
    assert marker.operation == "snapshot_complete"
    assert set(marker.live_ids) == {
        envelope.source_object_id for envelope in captured[:-1]
    }
    assert marker.checkpoint is not None
    assert all(envelope.checkpoint is None for envelope in captured[:-1])


@pytest.mark.concept("AU-KG.ingest.external-graph-federation")
def test_external_graph_dry_run_returns_only_aliases_digests_and_counts() -> None:
    result = ingest_registered_graph(
        object(),
        _Registry(_ExternalEngine()),
        _request(dry_run=True),
        profile=_profile(),
    )

    assert result["status"] == "dry_run"
    assert result["planned_nodes"] == 2
    assert result["planned_edges"] == 1
    serialized = json.dumps(result, sort_keys=True)
    assert "raw-node" not in serialized
    assert "MATCH" not in serialized
    assert "vault://" not in serialized


def test_external_graph_material_version_changes_for_edge_only_delta(
    monkeypatch,
) -> None:
    captured = []

    def fake_ingest(_engine, envelope):
        captured.append(envelope)
        return {"status": "success"}

    class _EdgeDeltaGraph(_ExternalEngine):
        def __init__(self, confidence: float) -> None:
            super().__init__()
            self.confidence = confidence

        def execute_read(self, query: str, params: dict):
            rows = super().execute_read(query, params)
            if "ExternalNode" not in query:
                rows[0]["properties"]["confidence"] = self.confidence
            return rows

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.external_graph.ingest_envelope",
        fake_ingest,
    )
    ingest_registered_graph(
        object(), _Registry(_EdgeDeltaGraph(0.4)), _request(), profile=_profile()
    )
    before = captured[0].source_version
    captured.clear()
    ingest_registered_graph(
        object(), _Registry(_EdgeDeltaGraph(0.8)), _request(), profile=_profile()
    )
    after = captured[0].source_version

    assert before != after
    assert len(before) == len(after) == 64


def test_external_graph_ingest_fails_closed_on_runtime_mapping_policy_drift() -> None:
    profile, _approved_policy_digest = _approved_profile()
    request = ExternalGraphIngestionRequest(
        **{
            **_request(dry_run=True).__dict__,
            "runtime_policy_digest": external_mapping_policy_digest(
                {"property_allowlist": ["description"]}
            ),
        }
    )

    with pytest.raises(ExternalGraphIngestionError, match="policy drift"):
        ingest_registered_graph(
            object(),
            _Registry(_ExternalEngine()),
            request,
            profile_resolver=_approved_profile_resolver(profile),
        )


def test_external_graph_ingest_rejects_partial_schema_rediscovery(monkeypatch) -> None:
    profile, runtime_policy_digest = _approved_profile()
    profile["backend_kind"] = "neo4j"
    profile["schema_digest"] = "a" * 64
    profile["mapping_digest"] = mapping_policy_digest(profile)
    request = ExternalGraphIngestionRequest(
        **{
            **_request(dry_run=True).__dict__,
            "runtime_policy_digest": runtime_policy_digest,
        }
    )

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.external_graph_schema.discover_external_schema",
        lambda *_args, **_kwargs: (
            type(
                "PartialSchema",
                (),
                {"partial": True, "schema_digest": profile["schema_digest"]},
            )(),
            object(),
        ),
    )

    with pytest.raises(ExternalGraphIngestionError, match="incomplete"):
        ingest_registered_graph(
            object(),
            _Registry(_ExternalEngine()),
            request,
            profile_resolver=_approved_profile_resolver(profile),
        )


def test_external_graph_ingest_rejects_mapping_path_tampering_after_approval() -> None:
    profile, runtime_policy_digest = _approved_profile()
    profile["node_mapping"]["id_path"] = "redirected.id"
    request = ExternalGraphIngestionRequest(
        **{
            **_request(dry_run=True).__dict__,
            "runtime_policy_digest": runtime_policy_digest,
        }
    )

    with pytest.raises(ExternalGraphIngestionError, match="changed after approval"):
        ingest_registered_graph(
            object(),
            _Registry(_ExternalEngine()),
            request,
            profile_resolver=_approved_profile_resolver(profile),
        )


def test_external_graph_runtime_profile_rejects_embedded_identity_secret() -> None:
    profile, runtime_policy_digest = _approved_profile()
    profile["identity_hmac_key"] = "embedded-secret-material-must-not-be-trusted"
    request = ExternalGraphIngestionRequest(
        **{
            **_request(dry_run=True).__dict__,
            "runtime_policy_digest": runtime_policy_digest,
        }
    )

    with pytest.raises(ExternalGraphIngestionError, match="cannot embed"):
        ingest_registered_graph(
            object(),
            _Registry(_ExternalEngine()),
            request,
            profile_resolver=_approved_profile_resolver(profile),
        )


@pytest.mark.concept("AU-KG.ingest.external-graph-federation")
def test_external_graph_rejects_mutating_or_unbounded_queries() -> None:
    profile = _profile()
    profile["node_query"] = (
        "MATCH (n:ExternalNode) SET n.flag = true RETURN n "
        "ORDER BY n.id SKIP $offset LIMIT $limit"
    )
    with pytest.raises(ExternalGraphIngestionError, match="read failed"):
        ingest_registered_graph(
            object(),
            _Registry(_ExternalEngine()),
            _request(dry_run=True),
            profile=profile,
        )

    profile = _profile()
    profile["node_query"] = (
        "MATCH (n:ExternalNode) RETURN '$limit' AS note /* LIMIT $limit */"
    )
    with pytest.raises(ExternalGraphIngestionError, match=r"LIMIT \$limit"):
        ingest_registered_graph(
            object(),
            _Registry(_ExternalEngine()),
            _request(dry_run=True),
            profile=profile,
        )

    profile = _profile()
    profile["node_query"] = "MATCH (n:ExternalNode) RETURN n LIMIT $limit; DELETE n"
    with pytest.raises(ExternalGraphIngestionError, match="exactly one"):
        ingest_registered_graph(
            object(),
            _Registry(_ExternalEngine()),
            _request(dry_run=True),
            profile=profile,
        )

    profile = _profile()
    profile["node_query"] = "MATCH (n:ExternalNode) RETURN n LIMIT $limit + 1000000"
    with pytest.raises(ExternalGraphIngestionError, match=r"LIMIT \$limit"):
        ingest_registered_graph(
            object(),
            _Registry(_ExternalEngine()),
            _request(dry_run=True),
            profile=profile,
        )

    profile = _profile()
    profile["node_query"] = "MATCH (n:ExternalNode) RETURN n"
    with pytest.raises(ExternalGraphIngestionError, match=r"\$limit"):
        ingest_registered_graph(
            object(),
            _Registry(_ExternalEngine()),
            _request(dry_run=True),
            profile=profile,
        )


@pytest.mark.concept("AU-KG.ingest.external-graph-federation")
def test_external_graph_remote_error_does_not_echo_remote_content() -> None:
    with pytest.raises(ExternalGraphIngestionError) as exc:
        ingest_registered_graph(
            object(),
            _Registry(_ExternalEngine(fail=True)),
            _request(dry_run=True),
            profile=_profile(),
        )

    message = str(exc.value)
    assert "contact@example.test" not in message
    assert "/home/example" not in message
    assert message == "External graph read failed (RuntimeError)"


def test_external_graph_rejects_a_backend_that_ignores_the_row_bound() -> None:
    request = ExternalGraphIngestionRequest(
        **{
            **_request(dry_run=True).__dict__,
            "max_records": 2,
        }
    )

    with pytest.raises(ExternalGraphIngestionError, match="bound"):
        ingest_registered_graph(
            object(),
            _Registry(_ExternalEngine()),
            request,
            profile=_profile(),
        )


def test_external_graph_rejects_ambiguous_query_surface() -> None:
    class _AmbiguousEngine:
        def query_cypher(self, _query, _params):
            return []

    with pytest.raises(ExternalGraphIngestionError, match="read-only surface"):
        ingest_registered_graph(
            object(),
            _Registry(_AmbiguousEngine()),
            _request(dry_run=True),
            profile=_profile(),
        )


@pytest.mark.concept("AU-KG.ingest.external-graph-federation")
def test_external_graph_rejects_mirror_sources_and_email_acls() -> None:
    with pytest.raises(ExternalGraphIngestionError, match="Mirror"):
        ingest_registered_graph(
            object(),
            _Registry(_ExternalEngine(), role="mirror"),
            _request(dry_run=True),
            profile=_profile(),
        )

    profile = _profile()
    profile["access"] = {
        "is_public": False,
        "user_emails": ["synthetic@example.test"],
    }
    with pytest.raises(ExternalGraphIngestionError, match="user-email"):
        ingest_registered_graph(
            object(),
            _Registry(_ExternalEngine()),
            _request(dry_run=True),
            profile=profile,
        )


class _PagedGraph:
    def __init__(self, count: int) -> None:
        self.nodes = [
            {
                "id": f"raw-{index}",
                "kind": "Capability",
                "version": str(index),
                "properties": {"title": f"Synthetic {index}"},
            }
            for index in range(count)
        ]
        self.calls: list[dict[str, int]] = []

    def execute_read(self, query: str, params: dict):
        if "ExternalNode" not in query:
            return []
        self.calls.append(dict(params))
        offset = int(params["offset"])
        limit = int(params["limit"])
        return self.nodes[offset : offset + limit]

    def read_snapshot_page(
        self,
        *,
        query: str,
        params: dict,
        max_records: int,
        snapshot_token: str | None,
    ) -> dict:
        token = snapshot_token or "snapshot-token-1"
        assert token == "snapshot-token-1"
        rows = self.execute_read(query, params)
        assert len(rows) <= max_records
        return {"rows": rows, "snapshot_token": token}


def test_external_graph_drains_deterministic_pages_before_writing(monkeypatch) -> None:
    captured = []
    graph = _PagedGraph(5)
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.external_graph.ingest_envelope",
        lambda _engine, envelope: captured.append(envelope) or {"status": "success"},
    )
    request = ExternalGraphIngestionRequest(
        **{**_request().__dict__, "page_size": 2, "max_pages": 3}
    )

    result = ingest_registered_graph(
        object(), _Registry(graph), request, profile=_profile()
    )

    assert result["nodes"] == 5
    assert result["sync_strategy"] == "snapshot"
    assert [call["offset"] for call in graph.calls] == [0, 2, 4]
    assert [call["limit"] for call in graph.calls] == [3, 3, 3]
    assert len(captured) == 6
    assert captured[-1].operation == "snapshot_complete"


def test_external_graph_page_overflow_fails_before_any_write(monkeypatch) -> None:
    captured = []
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.external_graph.ingest_envelope",
        lambda _engine, envelope: captured.append(envelope) or {"status": "success"},
    )
    request = ExternalGraphIngestionRequest(
        **{**_request().__dict__, "page_size": 2, "max_pages": 2}
    )

    with pytest.raises(ExternalGraphIngestionError, match="page bound"):
        ingest_registered_graph(
            object(), _Registry(_PagedGraph(5)), request, profile=_profile()
        )

    assert captured == []


def test_external_graph_tokenless_snapshot_uses_one_bounded_read() -> None:
    class _OffsetOnlyGraph(_PagedGraph):
        read_snapshot_page = None

    graph = _OffsetOnlyGraph(750)
    request = ExternalGraphIngestionRequest(
        **{**_request(dry_run=True).__dict__, "max_records": 1_000}
    )

    result = ingest_registered_graph(
        object(), _Registry(graph), request, profile=_node_only_profile()
    )

    assert result["planned_nodes"] == 750
    assert result["snapshot_authoritative"] is True
    assert graph.calls == [{"scope": "synthetic", "offset": 0, "limit": 1_001}]


def test_external_graph_rejects_snapshot_token_drift_before_any_write(
    monkeypatch,
) -> None:
    captured = []

    class _DriftingSnapshotGraph(_PagedGraph):
        def read_snapshot_page(
            self,
            *,
            query: str,
            params: dict,
            max_records: int,
            snapshot_token: str | None,
        ) -> dict:
            rows = self.execute_read(query, params)
            assert len(rows) <= max_records
            return {
                "rows": rows,
                "snapshot_token": (
                    "snapshot-token-1" if snapshot_token is None else "snapshot-token-2"
                ),
            }

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.external_graph.ingest_envelope",
        lambda _engine, envelope: captured.append(envelope) or {"status": "success"},
    )
    request = ExternalGraphIngestionRequest(
        **{**_request().__dict__, "page_size": 2, "max_pages": 3}
    )

    with pytest.raises(ExternalGraphIngestionError, match="token changed"):
        ingest_registered_graph(
            object(), _Registry(_DriftingSnapshotGraph(5)), request, profile=_profile()
        )

    assert captured == []


class _RowsGraph:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def execute_read(self, query: str, params: dict):
        if "ExternalNode" not in query:
            return []
        offset = int(params["offset"])
        limit = int(params["limit"])
        return self.rows[offset : offset + limit]


def _node_only_profile() -> dict:
    profile = _profile()
    profile.pop("edge_query")
    profile.pop("edge_mapping")
    return profile


def test_external_graph_missing_identity_makes_snapshot_nonauthoritative(
    monkeypatch,
) -> None:
    captured = []
    rows = [
        {
            "id": "raw-valid",
            "kind": "Capability",
            "version": "1",
            "properties": {"title": "Synthetic"},
        },
        {
            "kind": "Capability",
            "version": "2",
            "properties": {"title": "Missing identity"},
        },
    ]
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.external_graph.ingest_envelope",
        lambda _engine, envelope: captured.append(envelope) or {"status": "success"},
    )

    result = ingest_registered_graph(
        object(), _Registry(_RowsGraph(rows)), _request(), profile=_node_only_profile()
    )

    assert result["status"] == "partial"
    assert result["snapshot_authoritative"] is False
    assert captured[-1].operation == "snapshot_complete"
    assert not captured[-1].live_ids
    assert captured[-1].provenance["fetch_ok"] is False


def test_external_graph_missing_edge_identity_suppresses_reconciliation(
    monkeypatch,
) -> None:
    captured = []

    class _MissingEdgeIdentityGraph(_ExternalEngine):
        def execute_read(self, query: str, params: dict):
            rows = super().execute_read(query, params)
            if "ExternalNode" not in query:
                rows[0].pop("target")
            return rows

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.external_graph.ingest_envelope",
        lambda _engine, envelope: captured.append(envelope) or {"status": "success"},
    )

    result = ingest_registered_graph(
        object(), _Registry(_MissingEdgeIdentityGraph()), _request(), profile=_profile()
    )

    assert result["status"] == "partial"
    assert result["snapshot_authoritative"] is False
    assert captured[-1].operation == "snapshot_complete"
    assert not captured[-1].live_ids
    assert captured[-1].provenance["fetch_ok"] is False


@pytest.mark.parametrize(
    ("rows", "limits", "message"),
    [
        (
            [
                {
                    "id": "raw-large",
                    "kind": "Capability",
                    "version": "1",
                    "properties": {"title": "x" * 300},
                }
            ],
            {"max_row_bytes": 256, "max_total_bytes": 2_048},
            "per-row byte bound",
        ),
        (
            [
                {
                    "id": f"raw-{index}",
                    "kind": "Capability",
                    "version": "1",
                    "properties": {"title": "x" * 80},
                }
                for index in range(2)
            ],
            {"max_row_bytes": 256, "max_total_bytes": 256},
            "cumulative byte bound",
        ),
        (
            [
                {
                    "id": "raw-deep",
                    "kind": "Capability",
                    "version": "1",
                    "properties": {"title": {"nested": "value"}},
                }
            ],
            {
                "max_row_bytes": 512,
                "max_total_bytes": 512,
                "max_nesting_depth": 2,
            },
            "nesting-depth bound",
        ),
        (
            [
                {
                    "id": "raw-wide",
                    "kind": "Capability",
                    "version": "1",
                    "properties": {"title": [1, 2, 3, 4, 5]},
                }
            ],
            {
                "max_row_bytes": 512,
                "max_total_bytes": 512,
                "max_collection_items": 4,
            },
            "collection-size bound",
        ),
    ],
)
def test_external_graph_enforces_structural_payload_budgets_before_writes(
    monkeypatch, rows, limits, message
) -> None:
    captured = []
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.external_graph.ingest_envelope",
        lambda _engine, envelope: captured.append(envelope) or {"status": "success"},
    )
    request = ExternalGraphIngestionRequest(**{**_request().__dict__, **limits})

    with pytest.raises(ExternalGraphIngestionError, match=message):
        ingest_registered_graph(
            object(), _Registry(_RowsGraph(rows)), request, profile=_node_only_profile()
        )

    assert captured == []


def test_external_graph_enforces_payload_budget_on_cdc_events(monkeypatch) -> None:
    captured = []

    class _OversizedCDCGraph:
        @staticmethod
        def read_change_page(*, cursor: str | None, limit: int):
            assert cursor == "cursor-1"
            assert limit > 0
            return {
                "events": [
                    {
                        "operation": "upsert",
                        "entity": "node",
                        "record": {
                            "id": "raw-node-a",
                            "kind": "Capability",
                            "version": "1",
                            "properties": {"title": "x" * 300},
                        },
                    }
                ],
                "next_cursor": "cursor-2",
                "has_more": False,
            }

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.external_graph.read_change_cursor",
        lambda _engine, _connector, *, source_instance: "cursor-1",
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.external_graph.ingest_envelope",
        lambda _engine, envelope: captured.append(envelope) or {"status": "success"},
    )
    request = ExternalGraphIngestionRequest(
        **{
            **_request().__dict__,
            "max_row_bytes": 256,
            "max_total_bytes": 2_048,
        }
    )

    with pytest.raises(ExternalGraphIngestionError, match="per-row byte bound"):
        ingest_registered_graph(
            object(), _Registry(_OversizedCDCGraph()), request, profile=_profile()
        )

    assert captured == []


def test_external_graph_uses_discovered_native_cdc_and_advances_cursor_once(
    monkeypatch,
) -> None:
    captured = []

    class _CDCGraph:
        def __init__(self) -> None:
            self.cursors: list[str | None] = []

        def execute_read(self, _query: str, _params: dict):
            raise AssertionError("snapshot query must not run when CDC is available")

        def read_change_page(self, *, cursor: str | None, limit: int):
            assert limit == 2
            self.cursors.append(cursor)
            if cursor == "cursor-1":
                return {
                    "events": [
                        {
                            "operation": "upsert",
                            "entity": "node",
                            "record": {
                                "id": "raw-node-a",
                                "kind": "Capability",
                                "version": "1",
                                "properties": {"title": "Synthetic A"},
                            },
                        },
                        {
                            "operation": "delete",
                            "entity": "node",
                            "id": "raw-node-old",
                        },
                    ],
                    "next_cursor": "cursor-2",
                    "has_more": True,
                }
            return {
                "events": [
                    {
                        "operation": "upsert",
                        "entity": "node",
                        "record": {
                            "id": "raw-node-b",
                            "kind": "Process",
                            "version": "2",
                            "properties": {"title": "Synthetic B"},
                        },
                    }
                ],
                "next_cursor": "cursor-3",
                "has_more": False,
            }

    graph = _CDCGraph()
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.external_graph.read_change_cursor",
        lambda _engine, _connector, *, source_instance: "cursor-1",
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.external_graph.ingest_envelope",
        lambda _engine, envelope: captured.append(envelope) or {"status": "success"},
    )
    request = ExternalGraphIngestionRequest(
        **{**_request().__dict__, "page_size": 2, "max_pages": 2}
    )

    result = ingest_registered_graph(
        object(), _Registry(graph), request, profile=_profile()
    )

    assert graph.cursors == ["cursor-1", "cursor-2"]
    assert result["sync_strategy"] == "cdc"
    assert result["nodes"] == 2
    assert result["deletes"] == 1
    assert [envelope.operation for envelope in captured] == [
        "upsert",
        "upsert",
        "delete",
        "snapshot_complete",
    ]
    assert all(envelope.checkpoint is None for envelope in captured[:-1])
    assert captured[-1].checkpoint == "cursor-3"
    assert captured[-1].provenance["fetch_ok"] is False
    serialized = json.dumps([item.as_dict() for item in captured], sort_keys=True)
    assert "raw-node" not in serialized


@pytest.mark.parametrize(
    ("next_cursor", "message"),
    [
        (None, "cursor did not advance"),
        ("cursor-1", "cursor did not advance"),
        (7, "cursor is invalid"),
        (" cursor-2", "cursor is invalid"),
        ("cursor\n2", "cursor is invalid"),
        ("x" * 4_097, "cursor is invalid"),
    ],
)
def test_external_graph_rejects_nonempty_terminal_cdc_page_without_advanced_cursor(
    monkeypatch, next_cursor, message
) -> None:
    captured = []

    class _TerminalCDCGraph:
        @staticmethod
        def read_change_page(*, cursor: str | None, limit: int):
            assert cursor == "cursor-1"
            assert limit > 0
            return {
                "events": [
                    {
                        "operation": "upsert",
                        "entity": "node",
                        "record": {
                            "id": "raw-node-a",
                            "kind": "Capability",
                            "version": "1",
                            "properties": {"title": "Synthetic A"},
                        },
                    }
                ],
                "next_cursor": next_cursor,
                "has_more": False,
            }

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.external_graph.read_change_cursor",
        lambda _engine, _connector, *, source_instance: "cursor-1",
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.external_graph.ingest_envelope",
        lambda _engine, envelope: captured.append(envelope) or {"status": "success"},
    )

    with pytest.raises(ExternalGraphIngestionError, match=message):
        ingest_registered_graph(
            object(),
            _Registry(_TerminalCDCGraph()),
            _request(),
            profile=_profile(),
        )

    assert captured == []


def test_external_graph_rejects_unsafe_persisted_cursor_before_source_read(
    monkeypatch,
) -> None:
    class _UnreadCDCGraph:
        calls = 0

        @classmethod
        def read_change_page(cls, **_kwargs):
            cls.calls += 1
            raise AssertionError("unsafe cursor reached the source")

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.external_graph.read_change_cursor",
        lambda _engine, _connector, *, source_instance: " cursor-1",
    )

    with pytest.raises(ExternalGraphIngestionError, match="cursor is invalid"):
        ingest_registered_graph(
            object(),
            _Registry(_UnreadCDCGraph()),
            _request(),
            profile=_profile(),
        )

    assert _UnreadCDCGraph.calls == 0


def test_external_graph_empty_snapshot_requires_explicit_reconcile_approval(
    monkeypatch,
) -> None:
    captured = []
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.external_graph.ingest_envelope",
        lambda _engine, envelope: captured.append(envelope) or {"status": "success"},
    )

    with pytest.raises(ExternalGraphIngestionError, match="not approved"):
        ingest_registered_graph(
            object(), _Registry(_PagedGraph(0)), _request(), profile=_profile()
        )
    assert captured == []

    request = ExternalGraphIngestionRequest(
        **{**_request().__dict__, "allow_empty_snapshot": True}
    )
    result = ingest_registered_graph(
        object(), _Registry(_PagedGraph(0)), request, profile=_profile()
    )
    assert result["nodes"] == 0
    assert len(captured) == 1
    assert captured[0].provenance["authoritative_empty_approved"] is True
