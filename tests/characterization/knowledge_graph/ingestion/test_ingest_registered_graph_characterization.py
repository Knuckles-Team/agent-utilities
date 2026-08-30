"""Characterization tests for ``ingest_registered_graph`` (CX-AU-04).

CCN 150 at time of writing -- these tests pin the OBSERVED, black-box
behaviour of the public function before any decomposition, so a later
refactor can be checked byte-for-byte against real (not aspirational)
behaviour. Only the public entry point is exercised; no private helper is
imported here (see CX-AU-04's dispatch brief: a characterization test for an
extracted private helper is out of scope and would itself become a
`test_only_symbols` finding).

Per the two-commit discipline, this file must be added and pass GREEN
against the UNMODIFIED ``external_graph.py`` before any refactor commit, and
must not change during the refactor commit that follows.
"""

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


class _ExternalEngine:
    """A read-only stub graph exposing two node rows and one edge row."""

    def __init__(self, *, include_person: bool = True) -> None:
        self.include_person = include_person
        self.calls: list[tuple[str, dict]] = []

    def execute_read(self, query: str, params: dict):
        self.calls.append((query, params))
        if "ExternalNode" in query:
            rows = [
                {
                    "id": "raw-node-a",
                    "kind": "Capability",
                    "version": "2026-07-01",
                    "properties": {
                        "title": "Synthetic Capability",
                        "description": "A governed capability",
                        "owner_name": "Example Owner",
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
            ]
            if self.include_person:
                rows.append(
                    {
                        "id": "raw-person",
                        "kind": "Person",
                        "version": "2026-07-02",
                        "properties": {"title": "Example Individual"},
                    }
                )
            return rows
        return [
            {
                "source": "raw-node-a",
                "target": "raw-node-b",
                "kind": "DEPENDS_ON",
                "properties": {"confidence": 0.9, "owner_name": "Example Owner"},
            }
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


def _request(*, dry_run: bool = False, **overrides) -> ExternalGraphIngestionRequest:
    base = dict(
        connection="external-catalog",
        source_alias="business-graph",
        profile_ref="vault://integrations/business-graph/import-profile",
        variables={"scope": "synthetic"},
        max_records=50,
        classification=DataClassification.CONFIDENTIAL,
        retention="P30D",
        dry_run=dry_run,
    )
    base.update(overrides)
    return ExternalGraphIngestionRequest(**base)


def _approved_profile() -> dict:
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
    return profile


def _capture_ingest(monkeypatch):
    captured: list = []

    def fake_ingest_batch(_engine, envelopes):
        captured.extend(envelopes)
        return [{"status": "success"} for _ in envelopes]

    def fake_ingest(_engine, envelope):
        captured.append(envelope)
        return {"status": "success"}

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.external_graph.ingest_envelopes",
        fake_ingest_batch,
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.external_graph.ingest_envelope",
        fake_ingest,
    )
    return captured


# ---------------------------------------------------------------------------
# Golden-path snapshot ingestion: pins the full public result-dict shape.
# ---------------------------------------------------------------------------


def test_snapshot_ingestion_pins_full_result_dict(monkeypatch) -> None:
    captured = _capture_ingest(monkeypatch)

    result = ingest_registered_graph(
        object(),
        _Registry(_ExternalEngine()),
        _request(),
        profile=_profile(),
    )

    assert result == {
        "status": "success",
        "source_alias": "business-graph",
        "connection": "external-catalog",
        "nodes": 2,
        "edges": 1,
        "deletes": 0,
        "sync_strategy": "snapshot",
        "snapshot_authoritative": True,
        "results": {"success": 3},
        "profile_digest": result["profile_digest"],
        "privacy": {
            "redactions": 4,
            "detected_types": ["personal_entity", "personal_field"],
        },
    }
    # 2 node envelopes in one batch + 1 snapshot-complete marker == 3 envelopes.
    assert len(captured) == 3
    marker = captured[-1]
    assert marker.operation == "snapshot_complete"
    assert set(marker.live_ids) == {
        envelope.source_object_id for envelope in captured[:-1]
    }


def test_snapshot_ingestion_never_persists_raw_identity(monkeypatch) -> None:
    captured = _capture_ingest(monkeypatch)

    ingest_registered_graph(
        object(),
        _Registry(_ExternalEngine()),
        _request(),
        profile=_profile(),
    )

    serialized = json.dumps(
        [envelope.as_dict() for envelope in captured], sort_keys=True
    )
    for forbidden in (
        "raw-node-a",
        "raw-node-b",
        "raw-person",
        "vault://",
        "synthetic-test-key-material",
        "MATCH (n:ExternalNode)",
    ):
        assert forbidden not in serialized


# ---------------------------------------------------------------------------
# dry_run: pins the reduced-fields result shape.
# ---------------------------------------------------------------------------


def test_dry_run_pins_result_shape() -> None:
    result = ingest_registered_graph(
        object(),
        _Registry(_ExternalEngine()),
        _request(dry_run=True),
        profile=_profile(),
    )

    assert result == {
        "status": "dry_run",
        "source_alias": "business-graph",
        "connection": "external-catalog",
        "planned_nodes": 2,
        "planned_edges": 1,
        "planned_deletes": 0,
        "sync_strategy": "snapshot",
        "snapshot_authoritative": True,
        "profile_digest": result["profile_digest"],
        "privacy": {
            "redactions": 4,
            "detected_types": ["personal_entity", "personal_field"],
        },
    }


# ---------------------------------------------------------------------------
# CDC path: pins cursor advancement and the resulting envelope operations.
# ---------------------------------------------------------------------------


def test_cdc_ingestion_pins_result_and_advances_cursor_once(monkeypatch) -> None:
    captured = _capture_ingest(monkeypatch)

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
    assert result["edges"] == 0
    assert [envelope.operation for envelope in captured] == [
        "upsert",
        "upsert",
        "delete",
        "snapshot_complete",
    ]
    assert captured[-1].checkpoint == "cursor-3"
    assert captured[-1].provenance["fetch_ok"] is False


# ---------------------------------------------------------------------------
# Validation branches: pin the exact ExternalGraphIngestionError text so a
# decomposition that reorders/renames a raise site is caught immediately.
# ---------------------------------------------------------------------------


def test_invalid_page_size_message() -> None:
    with pytest.raises(
        ExternalGraphIngestionError, match=r"^page_size must be between 1 and 1000$"
    ):
        ingest_registered_graph(
            object(),
            _Registry(_ExternalEngine()),
            _request(page_size=0),
            profile=_profile(),
        )


def test_invalid_sync_mode_message() -> None:
    with pytest.raises(
        ExternalGraphIngestionError,
        match=r"^sync_mode must be auto, cdc, or snapshot$",
    ):
        ingest_registered_graph(
            object(),
            _Registry(_ExternalEngine()),
            _request(sync_mode="weekly"),
            profile=_profile(),
        )


def test_manifest_gate_fails_before_profile_or_source_read(monkeypatch) -> None:
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

    with pytest.raises(
        ExternalGraphIngestionError,
        match=r"^External graph connector requires a certified capability bundle$",
    ):
        ingest_registered_graph(
            object(), registry, _request(), profile_resolver=resolve_profile
        )

    assert profile_resolved is False
    assert external.calls == []


def test_duplicate_identity_raises_exact_message(monkeypatch) -> None:
    class _DuplicateGraph(_ExternalEngine):
        def execute_read(self, query: str, params: dict):
            if "ExternalNode" in query:
                return [
                    {
                        "id": "same-id",
                        "kind": "Capability",
                        "version": "1",
                        "properties": {"title": "A"},
                    },
                    {
                        "id": "same-id",
                        "kind": "Capability",
                        "version": "2",
                        "properties": {"title": "B"},
                    },
                ]
            return []

    with pytest.raises(
        ExternalGraphIngestionError,
        match=r"^External graph snapshot contains a duplicate identity$",
    ):
        ingest_registered_graph(
            object(),
            _Registry(_DuplicateGraph()),
            _request(),
            profile=_profile(),
        )


def test_cdc_duplicate_identity_identifies_the_cdc_batch(monkeypatch) -> None:
    class _DuplicateCDCGraph:
        def execute_read(self, _query: str, _params: dict):
            raise AssertionError("snapshot query must not run when CDC is available")

        def read_change_page(self, *, cursor: str | None, limit: int):
            assert cursor is None
            assert limit == 50
            return {
                "events": [
                    {
                        "operation": "upsert",
                        "entity": "node",
                        "record": {
                            "id": "same-id",
                            "kind": "Capability",
                            "version": "1",
                            "properties": {"title": "A"},
                        },
                    },
                    {
                        "operation": "upsert",
                        "entity": "node",
                        "record": {
                            "id": "same-id",
                            "kind": "Capability",
                            "version": "2",
                            "properties": {"title": "B"},
                        },
                    },
                ],
                "next_cursor": "cursor-1",
                "has_more": False,
            }

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.external_graph.read_change_cursor",
        lambda _engine, _connector, *, source_instance: None,
    )

    with pytest.raises(
        ExternalGraphIngestionError,
        match=r"^External graph CDC batch contains a duplicate identity$",
    ):
        ingest_registered_graph(
            object(),
            _Registry(_DuplicateCDCGraph()),
            _request(sync_mode="cdc"),
            profile=_profile(),
        )


# ---------------------------------------------------------------------------
# Missing identity: pins the "nonauthoritative snapshot" (partial) behaviour,
# not just an error -- this is an OBSERVED result shape, not a failure.
# ---------------------------------------------------------------------------


def test_missing_node_identity_marks_snapshot_nonauthoritative(monkeypatch) -> None:
    class _MissingIdentityGraph(_ExternalEngine):
        def execute_read(self, query: str, params: dict):
            if "ExternalNode" in query:
                return [
                    {
                        "id": "raw-node-a",
                        "kind": "Capability",
                        "version": "1",
                        "properties": {"title": "A"},
                    },
                    {
                        "id": "",
                        "kind": "Capability",
                        "version": "1",
                        "properties": {"title": "no identity"},
                    },
                ]
            return []

    captured = _capture_ingest(monkeypatch)
    result = ingest_registered_graph(
        object(),
        _Registry(_MissingIdentityGraph()),
        _request(),
        profile=_profile(),
    )

    assert result["status"] == "partial"
    assert result["snapshot_authoritative"] is False
    assert result["nodes"] == 1
    # The reconciliation marker is NOT authoritative when identity is missing:
    # observed (possibly surprising) behaviour, pinned as-is.
    marker = captured[-1]
    assert marker.live_ids == ()
    assert marker.provenance["fetch_ok"] is False


def test_empty_snapshot_requires_explicit_reconcile_approval(monkeypatch) -> None:
    class _EmptyGraph(_ExternalEngine):
        def execute_read(self, query: str, params: dict):
            return []

    with pytest.raises(
        ExternalGraphIngestionError,
        match=r"^External graph empty snapshot is not approved for reconciliation$",
    ):
        ingest_registered_graph(
            object(),
            _Registry(_EmptyGraph()),
            _request(),
            profile=_profile(),
        )

    # With explicit approval, the empty snapshot is accepted (observed both
    # sides of the same guard in one test).
    _capture_ingest(monkeypatch)
    result = ingest_registered_graph(
        object(),
        _Registry(_EmptyGraph()),
        _request(allow_empty_snapshot=True),
        profile=_profile(),
    )
    assert result["nodes"] == 0
    assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Runtime profile drift: pins the approved-profile validation path (this is
# the only path exercised when ``profile=`` is NOT given, i.e. production).
# ---------------------------------------------------------------------------


def test_runtime_mapping_policy_drift_rejected() -> None:
    profile = _approved_profile()
    with pytest.raises(
        ExternalGraphIngestionError,
        match=r"^External graph mapping policy drift requires a new proposal$",
    ):
        ingest_registered_graph(
            object(),
            _Registry(_ExternalEngine()),
            _request(runtime_policy_digest="0" * 64),
            profile_resolver=lambda ref: (
                json.dumps(profile)
                if ref == "vault://integrations/business-graph/import-profile"
                else "synthetic-test-key-material-32-bytes"
            ),
        )
