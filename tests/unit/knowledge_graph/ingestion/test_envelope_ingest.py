"""Tests for :func:`ingest_envelope` — the atomic ChangeEnvelope ingest transaction

(AU-P1-5, CONCEPT:AU-KG.ingest.envelope-atomic-transaction).

Covers the round-trip success path, mid-transaction rollback (no partial state +
watermark unadvanced), the ``snapshot_complete`` reconcile/tombstone path, and
idempotent replay.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.ingestion.change_envelope import ChangeEnvelope
from agent_utilities.knowledge_graph.ingestion.envelope_ingest import ingest_envelope


class FakeBackend:
    """A small real in-memory graph store — enough to exercise every query shape

    ``ingest_envelope`` / ``write_entities`` / ``source_sync`` watermark+reconcile
    issue, so the atomicity assertions are checked against real state transitions
    rather than a call-count mock.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        self.watermark: dict[str, str] = {}
        self.lineage: dict[str, dict[str, Any]] = {}

    # -- execute (single-row Cypher-ish queries) --------------------------

    def execute(self, query: str, params: dict[str, Any] | None = None) -> list[dict]:
        params = params or {}

        if "ChangeLineage" in query:
            lid = params.get("id")
            if "RETURN n.status" in query:
                rec = self.lineage.get(lid)
                return [rec] if rec else []
            if "DETACH DELETE" in query:
                self.lineage.pop(lid, None)
                return []
            if "MERGE" in query:
                rec = self.lineage.setdefault(lid, {})
                rec.update({k: v for k, v in params.items() if k != "id"})
                return []
            return []

        if "SourceSyncState" in query:
            if "RETURN" in query:
                w = self.watermark.get(params.get("id"))
                return [{"w": w}] if w else []
            if "MERGE" in query:
                self.watermark[params["id"]] = params.get("wm")
                return []
            return []

        if "n.content_hash AS h" in query:
            ids = params.get("ids") or []
            return [
                {"id": i, "h": self.nodes[i]["content_hash"]}
                for i in ids
                if i in self.nodes and "content_hash" in self.nodes[i]
            ]

        if "n.externalToolId AS guid" in query:
            return [
                {"id": nid, "guid": props.get("externalToolId")}
                for nid, props in self.nodes.items()
                if props.get("domain") == params.get("src") and props.get("externalToolId")
            ]

        nid = params.get("id")
        if "RETURN n.id AS id" in query:
            return [{"id": nid}] if nid in self.nodes else []
        if "RETURN n.archived AS archived" in query:
            if nid in self.nodes:
                return [{"archived": self.nodes[nid].get("archived", False)}]
            return []
        if "SET n.archived = true" in query:
            if nid in self.nodes:
                self.nodes[nid]["archived"] = True
                self.nodes[nid]["archivedReason"] = params.get("reason")
            return []
        if "SET n.archived = $archived" in query:
            if nid in self.nodes:
                self.nodes[nid]["archived"] = params.get("archived")
            return []
        if "DETACH DELETE n" in query:
            self.nodes.pop(nid, None)
            self.edges = [
                e for e in self.edges if e.get("source") != nid and e.get("target") != nid
            ]
            return []
        return []

    # -- execute_batch (UNWIND MERGE — the materialization write path) ----

    def execute_batch(self, query: str, rows: list[dict[str, Any]]) -> list:
        if "MATCH (s" in query and "MERGE (s)-[r:" in query:
            for row in rows:
                self.edges.append(dict(row))
            return []
        if "MERGE (n:" in query:
            for row in rows:
                node = self.nodes.setdefault(row["id"], {})
                node.update(row)
            return []
        return []


class FakeEngine:
    def __init__(self, backend: FakeBackend) -> None:
        self.backend = backend


def _upsert_env(**overrides: Any) -> ChangeEnvelope:
    record = {"id": "obj-1", "type": "AgentMemory", "name": "Widget", "updatedAt": "2026-01-01"}
    record.update(overrides.pop("record_overrides", {}))
    return ChangeEnvelope.from_connector_record(
        record,
        connector="acme-connector",
        id_field="id",
        version_field="updatedAt",
        checkpoint=record.get("updatedAt"),
        **overrides,
    )


def test_round_trip_success_writes_object_lineage_and_advances_watermark():
    backend = FakeBackend()
    engine = FakeEngine(backend)
    env = _upsert_env()

    result = ingest_envelope(engine, env)

    assert result["status"] == "success"
    assert result["watermark_advanced"] is True
    assert backend.nodes["obj-1"]["name"] == "Widget"
    lineage = backend.lineage[f"envelope:{env.idempotency_key}"]
    assert lineage["status"] == "applied"
    assert backend.watermark["sync:acme-connector"] == "2026-01-01"


def test_idempotent_replay_is_a_noop_skip():
    backend = FakeBackend()
    engine = FakeEngine(backend)
    env = _upsert_env()

    first = ingest_envelope(engine, env)
    assert first["status"] == "success"

    second = ingest_envelope(engine, env)
    assert second["status"] == "skipped"
    assert second["watermark_advanced"] is False
    # no duplicate node, no duplicate edge bookkeeping
    assert len(backend.nodes) == 1


def test_mid_transaction_failure_leaves_no_partial_state(monkeypatch):
    """A NEW object: if lineage-write fails after the object write succeeded,

    the object write is rolled back (deleted) and the watermark never advances —
    a resumed sync sees no trace of this attempt.
    """
    backend = FakeBackend()
    engine = FakeEngine(backend)
    env = _upsert_env()

    import agent_utilities.knowledge_graph.ingestion.envelope_ingest as mod

    def _boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("lineage backend unreachable")

    monkeypatch.setattr(mod, "_lineage_write", _boom)

    result = ingest_envelope(engine, env)

    assert result["status"] == "failed"
    assert result["watermark_advanced"] is False
    # NO partial state: the object write was rolled back entirely.
    assert "obj-1" not in backend.nodes
    assert not backend.lineage
    assert not backend.watermark


def test_mid_transaction_failure_after_cdc_leaves_write_durable_but_unadvanced():
    """A failure AFTER cdc (e.g. the final watermark-advance step) must NOT undo

    the already-durable write+lineage — undoing a correct write would be worse
    than leaving it. The watermark must still not advance, so a retry redoes the
    (idempotent) envelope rather than losing it.
    """
    backend = FakeBackend()
    engine = FakeEngine(backend)
    env = _upsert_env()

    import agent_utilities.knowledge_graph.ingestion.envelope_ingest as mod

    def _boom(*_a: Any, **_k: Any) -> bool:
        raise RuntimeError("watermark store unreachable")

    # Patch only for the first (failing) attempt — restored before the retry
    # below, which must use the REAL implementation to prove convergence.
    original = mod._advance_watermark
    mod._advance_watermark = _boom
    try:
        result = ingest_envelope(engine, env)
    finally:
        mod._advance_watermark = original

    assert result["status"] == "failed"
    assert result["watermark_advanced"] is False
    # the write + lineage("pending") already durably succeeded — NOT undone.
    assert backend.nodes["obj-1"]["name"] == "Widget"
    assert backend.lineage[f"envelope:{env.idempotency_key}"]["status"] == "pending"
    assert not backend.watermark

    # A retry converges to "applied" (idempotent MERGE), proving crash-resume safety.
    retry = ingest_envelope(engine, env)
    assert retry["status"] == "success"
    assert backend.lineage[f"envelope:{env.idempotency_key}"]["status"] == "applied"


def test_snapshot_complete_envelope_tombstones_correctly():
    backend = FakeBackend()
    backend.nodes["gone"] = {"domain": "acme-connector", "externalToolId": "gone"}
    backend.nodes["kept"] = {"domain": "acme-connector", "externalToolId": "kept"}
    engine = FakeEngine(backend)

    marker = ChangeEnvelope.snapshot_complete(
        connector="acme-connector", live_ids=["kept"], checkpoint="2026-02-01"
    )
    result = ingest_envelope(engine, marker)

    assert result["status"] == "success"
    assert backend.nodes["gone"].get("archived") is True
    assert not backend.nodes["kept"].get("archived")
    assert result["watermark_advanced"] is True
    assert backend.watermark["sync:acme-connector"] == "2026-02-01"


def test_snapshot_complete_never_tombstones_on_unverified_empty_snapshot():
    """CONCEPT:AU-P0-4 fail-closed — fetch_ok=False must refuse to tombstone even

    with an empty live_ids set (mirrors ``_reconcile``'s own guarantee)."""
    backend = FakeBackend()
    backend.nodes["kept"] = {"domain": "acme-connector", "externalToolId": "kept"}
    engine = FakeEngine(backend)

    marker = ChangeEnvelope.snapshot_complete(
        connector="acme-connector", live_ids=[], fetch_ok=False
    )
    result = ingest_envelope(engine, marker)

    assert result["status"] == "success"
    assert not backend.nodes["kept"].get("archived")


def test_successive_snapshot_complete_passes_each_apply_not_just_the_first():
    """Two DISTINCT reconcile passes (different checkpoints) must each run — a

    snapshot_complete marker has no source_object_id, so without a
    checkpoint-derived source_version every pass would collide on the same
    idempotency key and the 2nd+ pass would be wrongly skipped as a replay.
    """
    backend = FakeBackend()
    backend.nodes["gone1"] = {"domain": "acme-connector", "externalToolId": "gone1"}
    backend.nodes["gone2"] = {"domain": "acme-connector", "externalToolId": "gone2"}
    engine = FakeEngine(backend)

    first = ChangeEnvelope.snapshot_complete(
        connector="acme-connector", live_ids=["gone2"], checkpoint="2026-01-01"
    )
    second = ChangeEnvelope.snapshot_complete(
        connector="acme-connector", live_ids=[], checkpoint="2026-02-01"
    )
    assert first.idempotency_key != second.idempotency_key

    r1 = ingest_envelope(engine, first)
    assert r1["status"] == "success"
    assert backend.nodes["gone1"].get("archived") is True  # gone1 not in live_ids

    r2 = ingest_envelope(engine, second)
    assert r2["status"] == "success"  # NOT skipped as a replay of the first pass


def test_delete_operation_tombstones_and_rollback_restores_prior_archived(monkeypatch):
    backend = FakeBackend()
    backend.nodes["obj-1"] = {"name": "Widget", "archived": False}
    engine = FakeEngine(backend)
    env = ChangeEnvelope(
        connector="acme-connector", operation="delete", source_object_id="obj-1"
    )

    import agent_utilities.knowledge_graph.ingestion.envelope_ingest as mod

    def _boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("lineage backend unreachable")

    monkeypatch.setattr(mod, "_lineage_write", _boom)
    result = ingest_envelope(engine, env)

    assert result["status"] == "failed"
    # rollback restored the pre-delete archived value (False), not left tombstoned.
    assert backend.nodes["obj-1"]["archived"] is False


def test_rejected_envelope_never_touches_the_backend():
    """A PUBLIC classification with no proof of public ACL is rejected before

    any write (CONCEPT:AU-P0-4 fail-closed connector permissions)."""
    from agent_utilities.models.company_brain import DataClassification

    backend = FakeBackend()
    engine = FakeEngine(backend)
    env = _upsert_env(classification=DataClassification.PUBLIC)

    result = ingest_envelope(engine, env)

    assert result["status"] == "rejected"
    assert result["watermark_advanced"] is False
    assert not backend.nodes


def test_blob_backed_artifact_envelope_writes_a_pointer_node():
    """A ``blob_ref`` envelope (no ``typed_payload``) writes a minimal artifact

    pointer node — the "artifact" leg of "write object/link/artifact"."""
    backend = FakeBackend()
    engine = FakeEngine(backend)
    env = ChangeEnvelope(
        connector="acme-connector",
        source_object_id="blob-1",
        payload_type="pdf",
        blob_ref="s3://bucket/report.pdf",
        checkpoint="2026-01-01",
    )

    result = ingest_envelope(engine, env)

    assert result["status"] == "success"
    assert backend.nodes["blob-1"]["blob_ref"] == "s3://bucket/report.pdf"
    assert backend.nodes["blob-1"]["type"] == "pdf"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
