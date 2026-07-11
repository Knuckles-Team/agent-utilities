"""Unit tests for the Blob/Rendition/AssetOccurrence identity chain (AU-P1-4).

CONCEPT:AU-KG.identity.asset-occurrence. Exercises :class:`MediaStore` against a fake
engine client (no live engine required — mirrors the ``_FakeClient`` pattern used by
``tests/knowledge_graph/test_native_ingest.py``), proving:

* the SAME bytes stored under two different messages/tenants dedup to ONE ``:Blob``
  but mint TWO distinct ``:AssetOccurrence`` nodes, each with its own independent
  tenant/owner/ACL/provenance (never digest-collapsed);
* an occurrence carries tenant/owner/ACL/retention/legal-hold/provenance;
* the refcount bracket (incref-before-commit, unref-compensation-on-failure) around
  the cross-modal ACID commit;
* ``store_rendition`` dedups derived bytes but still mints a distinct rendition id;
* ``migrate_legacy_asset`` upgrades a pre-AU-P1-4 digest-keyed asset without
  mutating it;
* ``record_extraction`` is a working seam for async extraction/embedding results.
"""

from __future__ import annotations

import hashlib

import pytest

from agent_utilities.knowledge_graph.core.session import GraphSession
from agent_utilities.knowledge_graph.memory.media_store import MediaStore
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext

pytestmark = [pytest.mark.concept("AU-KG.identity.asset-occurrence")]


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _FakeBlob:
    """Fake ``client.blob`` — content-addressed store + refcount, no chunk streaming."""

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.refcounts: dict[str, int] = {}
        self.incref_calls: list[str] = []
        self.unref_calls: list[str] = []
        self.fail_incref_for: set[str] = set()

    def store(self, data: bytes) -> str:
        digest = _digest(data)
        self.data[digest] = data
        return digest

    def incref(self, digest: str) -> int:
        self.incref_calls.append(digest)
        if digest in self.fail_incref_for:
            raise RuntimeError("simulated incref failure")
        self.refcounts[digest] = self.refcounts.get(digest, 0) + 1
        return self.refcounts[digest]

    def unref(self, digest: str) -> int:
        self.unref_calls.append(digest)
        self.refcounts[digest] = max(0, self.refcounts.get(digest, 0) - 1)
        return self.refcounts[digest]

    def fetch(self, digest: str) -> bytes:
        return self.data[digest]


class _FakeTxn:
    """Fake ``client.txn``. Writes land in ``nodes``/``blob_refs``/``embeddings``
    immediately (not staged-until-commit like the real engine) — sufficient to
    assert on the Python-level control flow this module owns (incref bracketing,
    node/props/blob_ref/embedding calls), not the engine's own ACID guarantee."""

    def __init__(self, *, fail_commit: bool = False) -> None:
        self.nodes: dict[str, dict] = {}
        self.blob_refs: list[tuple[str, str]] = []
        self.embeddings: dict[str, list[float]] = {}
        self.committed: list[str] = []
        self.fail_commit = fail_commit
        self._n = 0

    def begin(self, graph: str | None = None) -> str:
        self._n += 1
        return f"txn{self._n}"

    def add_node(self, txn, node_id, props):
        self.nodes[node_id] = dict(props)

    def blob_ref(self, txn, node_id, digest):
        self.blob_refs.append((node_id, digest))
        return True

    def add_embedding(self, txn, node_id, embedding):
        self.embeddings[node_id] = list(embedding)
        return True

    def commit(self, txn) -> bool:
        if self.fail_commit:
            return False
        self.committed.append(txn)
        return True


class _FakeNodes:
    """Fake ``client.nodes`` backed by the SAME dict the txn writes into, so a
    read-after-write within one test sees what was just committed."""

    def __init__(self, backing: dict[str, dict]) -> None:
        self._backing = backing

    def has(self, node_id: str) -> bool:
        return node_id in self._backing

    def properties(self, node_id: str) -> dict | None:
        return self._backing.get(node_id)


class _FakeEdges:
    def __init__(self) -> None:
        self.edges: list[tuple[str, str, dict]] = []

    def add(self, source, dest, props):
        self.edges.append((source, dest, props))


class _FakeClient:
    def __init__(self, *, fail_commit: bool = False) -> None:
        self.blob = _FakeBlob()
        self.txn = _FakeTxn(fail_commit=fail_commit)
        self.nodes = _FakeNodes(self.txn.nodes)
        self.edges = _FakeEdges()


class _FakeCompute:
    def __init__(
        self, client: _FakeClient | None = None, graph_name: str = "__commons__"
    ):
        self._client = client or _FakeClient()
        self.graph_name = graph_name

    def nodes(self, data: bool = False):
        """Fallback node-enumeration surface ``iter_nodes_by_types`` uses when a
        graph exposes no ``get_nodes_by_label`` (this fake has none) — exercises
        that local/test-graph path for :meth:`MediaStore.migrate_legacy_assets_bulk`."""
        items = list(self._client.txn.nodes.items())
        return items if data else [nid for nid, _ in items]


def _session(tenant: str, actor_id: str = "user:1") -> GraphSession:
    return GraphSession(
        actor=ActorContext(
            actor_id=actor_id, actor_type=ActorType.HUMAN, tenant_id=tenant
        ),
        tenant=tenant,
    )


IMG = b"\x89PNG\r\n\x1a\n" + bytes(range(64)) + b"chart-bytes"


# --------------------------------------------------------------------------- #
# Dedup-at-blob / distinct-occurrence identity                                #
# --------------------------------------------------------------------------- #


def test_same_bytes_two_messages_one_blob_two_occurrences():
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))

    res1 = store.store_media(
        IMG,
        media_type="image",
        mime_type="image/png",
        message_id="mem:msg1",
        session=_session("tenant-a", "user:alice"),
        acl=["tenant-a:read"],
        retention="30d",
    )
    res2 = store.store_media(
        IMG,
        media_type="image",
        mime_type="image/png",
        message_id="mem:msg2",
        session=_session("tenant-b", "user:bob"),
        acl=["tenant-b:read"],
        retention="90d",
        legal_hold=True,
    )

    assert res1 is not None and res2 is not None
    # ONE blob (content dedup) ...
    assert res1.digest == res2.digest
    assert res1.blob_id == res2.blob_id
    assert res1.deduped is False  # first sighting of these bytes
    assert res2.deduped is True  # second sighting — no new chunks

    # ... but TWO distinct occurrences, never digest-collapsed.
    assert res1.occurrence_id != res2.occurrence_id
    assert res1.digest not in res1.occurrence_id
    assert res1.occurrence_id.startswith("occurrence:")
    # Back-compat alias still resolves to the (now distinct) occurrence id.
    assert res1.asset_id == res1.occurrence_id

    # Independent provenance survived — the second write did NOT overwrite the first.
    p1 = client.txn.nodes[res1.occurrence_id]
    p2 = client.txn.nodes[res2.occurrence_id]
    assert p1["tenant"] == "tenant-a" and p2["tenant"] == "tenant-b"
    assert p1["owner"] == "user:alice" and p2["owner"] == "user:bob"
    assert p1["acl"] == ["tenant-a:read"] and p2["acl"] == ["tenant-b:read"]
    assert p1["retention"] == "30d" and p2["retention"] == "90d"
    assert p1["legal_hold"] is False and p2["legal_hold"] is True
    assert p1["message_id"] == "mem:msg1" and p2["message_id"] == "mem:msg2"

    # Only ONE :Blob node was ever staged (the second call saw it already present).
    blob_nodes = [n for n in client.txn.nodes if n.startswith("blob:")]
    assert blob_nodes == [res1.blob_id]


def test_occurrence_carries_full_provenance_bundle():
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))
    res = store.store_media(
        b"unique-provenance-bytes",
        media_type="image",
        session=_session("acme"),
        owner="user:carol",
        acl={"read": ["acme"], "write": ["acme:admin"]},
        event_time="2026-01-01T00:00:00Z",
        retention="7y",
        legal_hold=True,
        provenance={"platform": "slack", "channel_id": "C1"},
    )
    assert res is not None
    props = client.txn.nodes[res.occurrence_id]
    assert props["type"] == "AssetOccurrence"
    assert props["tenant"] == "acme"
    assert props["owner"] == "user:carol"
    assert props["acl"] == {"read": ["acme"], "write": ["acme:admin"]}
    assert props["event_time"] == "2026-01-01T00:00:00Z"
    assert props["retention"] == "7y"
    assert props["legal_hold"] is True
    assert props["provenance"] == {"platform": "slack", "channel_id": "C1"}

    # hasBlob edge is present.
    assert (res.occurrence_id, res.blob_id, {"type": "hasBlob"}) in client.edges.edges


def test_tenant_isolated_blob_salts_the_blob_node_id():
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))
    res = store.store_media(
        b"tenant-isolated-bytes",
        session=_session("acme"),
        tenant_isolated_blob=True,
    )
    assert res is not None
    assert res.blob_id == f"blob:acme:{res.digest}"


# --------------------------------------------------------------------------- #
# Refcount atomicity (incref-before-commit, unref-on-failure)                 #
# --------------------------------------------------------------------------- #


def test_refcount_incref_before_commit_then_kept_on_success():
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))
    res = store.store_media(b"atomic-bytes", session=_session("acme"))
    assert res is not None
    assert client.blob.incref_calls == [res.digest]
    assert client.blob.unref_calls == []  # no compensation needed
    assert client.blob.refcounts[res.digest] == 1


def test_refcount_compensated_on_commit_failure():
    client = _FakeClient(fail_commit=True)
    store = MediaStore(_FakeCompute(client))
    res = store.store_media(b"will-fail-bytes", session=_session("acme"))
    assert res is None
    digest = _digest(b"will-fail-bytes")
    # incref happened BEFORE the doomed commit, and was compensated after.
    assert client.blob.incref_calls == [digest]
    assert client.blob.unref_calls == [digest]
    assert client.blob.refcounts[digest] == 0


def test_incref_failure_aborts_before_any_txn_is_opened():
    client = _FakeClient()
    data = b"incref-will-fail"
    client.blob.fail_incref_for.add(_digest(data))
    store = MediaStore(_FakeCompute(client))
    res = store.store_media(data, session=_session("acme"))
    assert res is None
    # No txn was ever begun/committed — the abort happened pre-txn.
    assert client.txn.committed == []
    assert client.txn.nodes == {}


# --------------------------------------------------------------------------- #
# Renditions                                                                   #
# --------------------------------------------------------------------------- #


def test_store_rendition_dedups_bytes_but_mints_distinct_ids():
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))
    occ = store.store_media(IMG, session=_session("acme"))
    assert occ is not None

    thumb = b"thumbnail-bytes"
    r1 = store.store_rendition(
        thumb,
        source_digest=occ.digest,
        rendition_type="thumbnail",
        occurrence_id=occ.occurrence_id,
        model="thumbnailer-v1",
        session=_session("acme"),
    )
    r2 = store.store_rendition(
        thumb,
        source_digest=occ.digest,
        rendition_type="thumbnail",
        occurrence_id=occ.occurrence_id,
        model="thumbnailer-v2",
        session=_session("acme"),
    )
    assert r1 is not None and r2 is not None
    assert r1.digest == r2.digest  # same derived bytes -> same blob
    assert r1.deduped is False and r2.deduped is True
    assert r1.rendition_id != r2.rendition_id  # distinct lineage per model

    props1 = client.txn.nodes[r1.rendition_id]
    props2 = client.txn.nodes[r2.rendition_id]
    assert props1["model"] == "thumbnailer-v1"
    assert props2["model"] == "thumbnailer-v2"
    assert props1["derived_from_digest"] == occ.digest

    # occurrence -> rendition edges recorded for both.
    assert (occ.occurrence_id, r1.rendition_id, {"type": "hasRendition"}) in (
        client.edges.edges
    )
    assert (occ.occurrence_id, r2.rendition_id, {"type": "hasRendition"}) in (
        client.edges.edges
    )


# --------------------------------------------------------------------------- #
# Async extraction/embedding seam                                             #
# --------------------------------------------------------------------------- #


def test_record_extraction_attaches_embedding_and_model_lineage():
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))
    occ = store.store_media(IMG, session=_session("acme"))
    assert occ is not None

    ok = store.record_extraction(
        occ.occurrence_id,
        model="clip-vit-b32",
        extracted_text="a chart showing revenue",
        embedding=[0.1, 0.2, 0.3],
        session=_session("acme"),
    )
    assert ok is True
    props = client.txn.nodes[occ.occurrence_id]
    assert props["extraction_model"] == "clip-vit-b32"
    assert props["extracted_text"] == "a chart showing revenue"
    assert client.txn.embeddings[occ.occurrence_id] == [0.1, 0.2, 0.3]


def test_record_extraction_missing_node_returns_false():
    store = MediaStore(_FakeCompute())
    assert store.record_extraction("occurrence:does-not-exist", model="m") is False


# --------------------------------------------------------------------------- #
# Legacy migration shim                                                        #
# --------------------------------------------------------------------------- #


def test_migrate_legacy_asset_creates_distinct_occurrence_without_mutating_legacy():
    client = _FakeClient()
    digest = _digest(b"legacy-bytes")
    legacy_id = f"media:{digest}"
    # Seed a pre-AU-P1-4 digest-keyed asset directly (as the OLD store_media used to
    # write it) plus its backing blob node.
    client.txn.nodes[f"blob:{digest}"] = {
        "type": "Blob",
        "content_digest": digest,
        "file_size_bytes": 11,
        "created_at": "2025-01-01T00:00:00Z",
    }
    client.txn.nodes[legacy_id] = {
        "type": "MediaAsset",
        "content_digest": digest,
        "media_type": "image",
        "mime_type": "image/png",
        "source": "legacy-platform",
        "message_id": "mem:old1",
        "created_at": "2025-01-01T00:00:00Z",
        "file_size_bytes": 11,
    }
    legacy_snapshot = dict(client.txn.nodes[legacy_id])

    store = MediaStore(_FakeCompute(client))
    res = store.migrate_legacy_asset(legacy_id, session=_session("acme"))

    assert res is not None
    assert res.occurrence_id.startswith("occurrence:")
    assert res.occurrence_id != legacy_id
    assert res.digest == digest
    assert res.deduped is True  # the blob already existed

    # The legacy node is untouched.
    assert client.txn.nodes[legacy_id] == legacy_snapshot

    new_props = client.txn.nodes[res.occurrence_id]
    assert new_props["type"] == "AssetOccurrence"
    assert new_props["legacy_asset_id"] == legacy_id
    assert new_props["provenance"]["migrated_from"] == legacy_id
    assert new_props["source"] == "legacy-platform"
    assert new_props["message_id"] == "mem:old1"

    assert (
        res.occurrence_id,
        legacy_id,
        {"type": "migratedFrom"},
    ) in client.edges.edges


def test_migrate_legacy_asset_missing_node_returns_none():
    store = MediaStore(_FakeCompute())
    assert store.migrate_legacy_asset("media:doesnotexist") is None


def test_migrate_legacy_asset_without_digest_returns_none():
    client = _FakeClient()
    client.txn.nodes["media:bad"] = {"type": "MediaAsset"}
    store = MediaStore(_FakeCompute(client))
    assert store.migrate_legacy_asset("media:bad") is None


# --------------------------------------------------------------------------- #
# Bulk migration sweep (AU-P1-4 follow-up)                                    #
# --------------------------------------------------------------------------- #


def _seed_legacy_asset(
    client: _FakeClient, data: bytes, *, source: str = "legacy-platform"
) -> str:
    """Seed a pre-AU-P1-4 digest-keyed ``media:<digest>`` node + its blob node
    directly into the fake backing store (as the OLD ``store_media`` used to
    write it)."""
    digest = _digest(data)
    legacy_id = f"media:{digest}"
    client.txn.nodes[f"blob:{digest}"] = {
        "type": "Blob",
        "content_digest": digest,
        "file_size_bytes": len(data),
        "created_at": "2025-01-01T00:00:00Z",
    }
    client.txn.nodes[legacy_id] = {
        "type": "MediaAsset",
        "content_digest": digest,
        "media_type": "image",
        "mime_type": "image/png",
        "source": source,
        "created_at": "2025-01-01T00:00:00Z",
        "file_size_bytes": len(data),
    }
    return legacy_id


def test_migrate_legacy_assets_bulk_migrates_all_and_rerun_is_noop():
    client = _FakeClient()
    legacy1 = _seed_legacy_asset(client, b"legacy-bulk-bytes-one")
    legacy2 = _seed_legacy_asset(client, b"legacy-bulk-bytes-two")
    store = MediaStore(_FakeCompute(client))

    result = store.migrate_legacy_assets_bulk(session=_session("acme"))

    assert result.scanned == 2
    assert result.migrated == 2
    assert result.skipped_already_migrated == 0
    assert result.failed == 0
    assert len(result.occurrence_ids) == 2

    occurrences = {
        nid: data
        for nid, data in client.txn.nodes.items()
        if data.get("type") == "AssetOccurrence"
    }
    assert len(occurrences) == 2
    legacy_ids_seen = {props["legacy_asset_id"] for props in occurrences.values()}
    assert legacy_ids_seen == {legacy1, legacy2}

    # Legacy nodes are completely untouched (non-destructive).
    assert client.txn.nodes[legacy1]["type"] == "MediaAsset"
    assert client.txn.nodes[legacy2]["type"] == "MediaAsset"
    assert "legacy_asset_id" not in client.txn.nodes[legacy1]

    # Re-running the sweep is a full no-op: idempotent, no new occurrences minted.
    result2 = store.migrate_legacy_assets_bulk(session=_session("acme"))
    assert result2.scanned == 2
    assert result2.migrated == 0
    assert result2.skipped_already_migrated == 2
    assert result2.failed == 0
    assert result2.occurrence_ids == []

    occurrences_after = [
        data
        for _nid, data in client.txn.nodes.items()
        if data.get("type") == "AssetOccurrence"
    ]
    assert len(occurrences_after) == 2  # unchanged by the re-run


def test_migrate_legacy_assets_bulk_batches_and_reports_progress():
    client = _FakeClient()
    for i in range(5):
        _seed_legacy_asset(client, f"legacy-bulk-batch-{i}".encode())
    store = MediaStore(_FakeCompute(client))

    progress_calls: list[dict] = []
    result = store.migrate_legacy_assets_bulk(
        batch_size=2, session=_session("acme"), progress=progress_calls.append
    )

    assert result.scanned == 5
    assert result.migrated == 5
    # 5 items / batch_size 2 -> batches of [2, 2, 1] -> 3 progress calls.
    assert len(progress_calls) == 3
    assert [c["processed"] for c in progress_calls] == [2, 4, 5]
    assert progress_calls[-1]["migrated"] == 5
    assert progress_calls[-1]["scanned"] == 5


def test_migrate_legacy_assets_bulk_counts_failures_without_aborting():
    client = _FakeClient()
    legacy_ok = _seed_legacy_asset(client, b"legacy-bulk-good-bytes")
    client.txn.nodes["media:bad"] = {"type": "MediaAsset"}  # no content_digest
    store = MediaStore(_FakeCompute(client))

    result = store.migrate_legacy_assets_bulk(session=_session("acme"))

    assert result.scanned == 2
    assert result.migrated == 1
    assert result.failed == 1
    assert "media:bad" in result.failed_ids
    occurrences = [
        data
        for _nid, data in client.txn.nodes.items()
        if data.get("type") == "AssetOccurrence"
    ]
    assert len(occurrences) == 1
    assert occurrences[0]["legacy_asset_id"] == legacy_ok


def test_migrate_legacy_assets_bulk_empty_graph_is_a_noop():
    store = MediaStore(_FakeCompute())
    result = store.migrate_legacy_assets_bulk()
    assert result.scanned == 0
    assert result.migrated == 0
    assert result.as_dict()["scanned"] == 0


# --------------------------------------------------------------------------- #
# Basic round-trip (unchanged surface)                                        #
# --------------------------------------------------------------------------- #


def test_fetch_bytes_and_fetch_asset_roundtrip():
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))
    res = store.store_media(IMG, session=_session("acme"))
    assert res is not None
    assert store.fetch_bytes(res.digest) == IMG
    assert store.fetch_asset(res.occurrence_id) == IMG
    assert store.fetch_occurrence(res.occurrence_id) == IMG


def test_empty_bytes_is_noop():
    store = MediaStore(_FakeCompute())
    assert store.store_media(b"") is None
    assert (
        store.store_rendition(b"", source_digest="x", rendition_type="thumbnail")
        is None
    )
