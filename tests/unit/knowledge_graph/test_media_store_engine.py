#!/usr/bin/python
"""Real-engine proof for durable multimodal memory (CONCEPT:AU-KG.identity.asset-occurrence).

Against the ACTUAL ephemeral engine (KG-2.238, pi-max tier so blob+tsdb are served):

* a small image/audio payload stored via :class:`MediaStore` round-trips back byte-
  for-byte from the engine BLOB substrate;
* content-addressed DEDUP at the :Blob layer ONLY — the same bytes stored twice
  yield ONE blob (deduped, same digest) but TWO DISTINCT :AssetOccurrence nodes
  (AU-P1-4 — occurrence identity is never digest-collapsed);
* the occurrence node + blob-ref (+ embedding) land in ONE cross-modal ACID commit —
  a reader sees the :AssetOccurrence with its content_digest, the :Blob node, and the
  :hasBlob edge.

These exercise the engine's blob feature; if the resolved test binary lacked it the
fixture would have rebuilt pi-max (tests/_test_engine.py).
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.engine, pytest.mark.concept("AU-KG.identity.asset-occurrence")]


# A tiny but non-trivial payload: includes 0x0A (newline) and 0xFF bytes so we prove
# binary survives the framed transport + the chunk Raw-bin path.
_IMG = bytes(range(256)) * 4 + b"\x0a\x0a chart \xff\xd8\xff pixels"
_AUDIO = b"OggS\x00\x02" + bytes(range(200)) + b"\x0a voice note \xff"


def _store(engine_graph):
    from agent_utilities.knowledge_graph.memory.media_store import MediaStore

    return MediaStore(engine_graph)


def test_media_blob_roundtrip(engine_graph):
    """Stored media reads back byte-for-byte from the engine BLOB substrate."""
    store = _store(engine_graph)
    res = store.store_media(_IMG, media_type="image", mime_type="image/jpeg")
    assert res is not None
    assert res.size_bytes == len(_IMG)
    assert res.deduped is False  # first time these bytes are seen

    # Fetch back by digest AND by occurrence id (``.asset_id`` back-compat alias
    # too) — both recover the exact bytes.
    assert store.fetch_bytes(res.digest) == _IMG
    assert store.fetch_asset(res.occurrence_id) == _IMG
    assert store.fetch_asset(res.asset_id) == _IMG


def test_media_content_addressed_dedup(engine_graph):
    """The same bytes stored twice → one blob (deduped) but TWO distinct occurrences."""
    store = _store(engine_graph)
    first = store.store_media(_AUDIO, media_type="audio", mime_type="audio/ogg")
    second = store.store_media(_AUDIO, media_type="audio", mime_type="audio/ogg")
    assert first is not None and second is not None
    # Content-addressed: identical bytes ⇒ identical digest ⇒ identical blob.
    assert first.digest == second.digest
    assert first.blob_id == second.blob_id
    # AU-P1-4: occurrence identity is NEVER digest-collapsed — two calls, two ids.
    assert first.occurrence_id != second.occurrence_id
    # The second store saw the blob already present (dedup — no new chunks).
    assert second.deduped is True
    assert store.fetch_bytes(first.digest) == _AUDIO


def test_media_cross_modal_acid_node_and_blob(engine_graph):
    """The :AssetOccurrence node + :Blob node + :hasBlob edge land atomically + are readable."""
    compute = engine_graph
    store = _store(compute)
    res = store.store_media(
        _IMG,
        media_type="image",
        mime_type="image/png",
        message_id="mem:msg1",
        embedding=[0.1, 0.2, 0.3, 0.4],
    )
    assert res is not None

    # The occurrence node committed with its content digest (the ACID node write).
    occurrence = compute._client.nodes.properties(res.occurrence_id)
    assert occurrence is not None
    assert occurrence.get("type") == "AssetOccurrence"
    assert occurrence.get("content_digest") == res.digest
    assert occurrence.get("message_id") == "mem:msg1"

    # The :Blob handle node committed too (same txn).
    blob = compute._client.nodes.properties(f"blob:{res.digest}")
    assert blob is not None
    assert blob.get("type") == "Blob"
    assert blob.get("content_digest") == res.digest

    # The :hasBlob edge links occurrence → blob.
    assert compute.has_edge(res.occurrence_id, f"blob:{res.digest}") is True


def test_media_isolation_per_test(engine_graph):
    """A fresh tenant per test: a uniquely-tagged asset written here is the only one.

    The other tests store ``_IMG``/``_AUDIO`` in their OWN tenant graphs, so this
    fresh graph has neither asset node — proving per-test isolation.
    """
    compute = engine_graph
    store = _store(compute)
    res = store.store_media(b"unique-iso-bytes-\x00\x01\x0a", media_type="image")
    assert res is not None
    assert compute.has_node(res.occurrence_id) is True
    # Assets from the dedup/roundtrip tests must NOT leak into this graph.
    import hashlib

    other = "blob:" + hashlib.sha256(_AUDIO).hexdigest()  # not this test's blob
    assert compute.has_node(other) is False
