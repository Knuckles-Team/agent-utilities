"""First-class multimodal memory over the engine BLOB substrate (CONCEPT:AU-KG.ingest.list-durable-media).

Media (an image a user sent, a voice note, a chart) used to be ephemeral and absent
from the KG: the messaging layer transcribed audio then ``os.unlink``ed it, wrapped
images as inline ``BinaryContent`` then discarded them, and persisted only TEXT. This
module makes media **durable and first-class** by storing the raw bytes in the engine's
content-addressed BLOB store (CONCEPT:EG-KG.storage.blob-namespace) and recording a ``:MediaAsset`` KG node
that references the blob — so "show me the chart they sent yesterday" becomes a real
query.

The store is built on the ONE engine authority (no second database):

* **Content-addressed + deduped.** ``client.blob.store(bytes)`` returns a digest; the
  same bytes always yield the same digest and store ZERO new chunks. A ``:Blob`` node
  carries that digest as its graph-side handle.
* **One cross-modal ACID txn.** The ``:MediaAsset`` node, its ``__blob__`` reference
  (``txn.blob_ref``), and — when an embedding is supplied — its vector all land in ONE
  ``client.txn`` commit (CONCEPT:EG-KG.txn.reader-never-sees-node): a reader never sees a node without its blob.
* **GC-safe.** ``blob.incref`` is called for each referencing ``:MediaAsset`` so the
  engine's refcount GC never reclaims a blob the graph still points at.

This is a CORE capability (per "Universal capability — one core, thin entrypoints"):
the messaging stack, the webui, the terminal — every entrypoint persists media through
THIS, contributing only how it receives the bytes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: Node-id prefixes — stable, content-derived where possible so re-persisting the
#: same bytes is idempotent at the graph level too (one :Blob per digest).
_BLOB_PREFIX = "blob:"
_ASSET_PREFIX = "media:"


@dataclass(frozen=True)
class StoredMedia:
    """The result of persisting one media payload.

    Attributes:
        asset_id: The ``:MediaAsset`` node id.
        digest: The content-addressed blob digest (the dedup/fetch key).
        deduped: ``True`` when the bytes were already present (no new chunks stored).
        size_bytes: The payload size.
    """

    asset_id: str
    digest: str
    deduped: bool
    size_bytes: int


class MediaStore:
    """Persist + retrieve media as durable, content-addressed KG-linked blobs.

    CONCEPT:AU-KG.ingest.list-durable-media. Bind to a live :class:`GraphComputeEngine` (or anything exposing
    a ``._client`` sync engine client with ``.blob``/``.txn``/``.nodes`` and a
    ``graph_name``). All methods are best-effort safe to call from the messaging
    background-persist path: a failure logs and returns ``None`` rather than raising.
    """

    def __init__(self, compute: Any) -> None:
        self._compute = compute

    # -- internals -----------------------------------------------------------
    @property
    def _client(self) -> Any:
        client = getattr(self._compute, "_client", None)
        if client is None:
            raise RuntimeError("MediaStore: bound engine has no live client")
        return client

    @property
    def _graph(self) -> str:
        return getattr(self._compute, "graph_name", "__commons__")

    def _blob_exists(self, digest: str) -> bool:
        """Whether a ``:Blob`` node for ``digest`` already exists in this graph."""
        try:
            return bool(self._client.nodes.has(f"{_BLOB_PREFIX}{digest}"))
        except Exception:  # noqa: BLE001 — treat unknown as "not yet present"
            return False

    # -- store ---------------------------------------------------------------
    def store_media(
        self,
        data: bytes,
        *,
        media_type: str = "",
        mime_type: str = "",
        source: str = "",
        message_id: str | None = None,
        name: str = "",
        embedding: list[float] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> StoredMedia | None:
        """Store ``data`` durably and create a ``:MediaAsset`` linked to its blob.

        Steps, all on the ONE engine authority:

        1. ``blob.store(data)`` — content-addressed + deduped → ``digest``.
        2. ONE cross-modal ACID txn (CONCEPT:EG-KG.txn.reader-never-sees-node): stage the ``:Blob`` node (if
           new), the ``:MediaAsset`` node, the ``blob_ref`` linking asset→blob, and the
           asset embedding when supplied — then commit atomically.
        3. ``blob.incref(digest)`` so the asset's reference is counted for GC.
        4. When ``message_id`` is given, add the ``:attachedToMessage`` edge so the
           media is reachable from the conversation memory.

        Returns a :class:`StoredMedia` (or ``None`` on failure — never raises).
        """
        if not data:
            return None
        client = self._client
        try:
            digest = client.blob.store(data)
        except Exception as e:  # noqa: BLE001
            logger.warning("[CONCEPT:AU-KG.ingest.list-durable-media] blob store failed: %s", e)
            return None

        blob_id = f"{_BLOB_PREFIX}{digest}"
        asset_id = f"{_ASSET_PREFIX}{digest}"
        blob_is_new = not self._blob_exists(digest)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        asset_props: dict[str, Any] = {
            "type": "MediaAsset",
            "name": name or f"media {media_type or mime_type or digest[:8]}",
            "content_digest": digest,
            "media_type": media_type,
            "mime_type": mime_type,
            "file_size_bytes": len(data),
            "source": source,
            "created_at": now,
        }
        if message_id:
            asset_props["message_id"] = message_id
        if extra:
            asset_props.update(extra)

        try:
            txn = client.txn.begin(graph=self._graph)
            if blob_is_new:
                client.txn.add_node(
                    txn,
                    blob_id,
                    {
                        "type": "Blob",
                        "content_digest": digest,
                        "file_size_bytes": len(data),
                        "created_at": now,
                    },
                )
            client.txn.add_node(txn, asset_id, asset_props)
            # The durable graph-side reference asset→blob (CONCEPT:EG-KG.txn.reader-never-sees-node). Lands
            # atomically with the node (+ embedding) in this one commit.
            client.txn.blob_ref(txn, asset_id, digest)
            if embedding:
                client.txn.add_embedding(txn, asset_id, list(embedding))
            committed = client.txn.commit(txn)
        except Exception as e:  # noqa: BLE001
            logger.warning("[CONCEPT:AU-KG.ingest.list-durable-media] media ACID txn failed: %s", e)
            return None
        if not committed:
            logger.warning("[CONCEPT:AU-KG.ingest.list-durable-media] media txn conflict (not committed)")
            return None

        # Count this asset's reference so GC never reclaims the blob underneath it.
        try:
            client.blob.incref(digest)
        except Exception as e:  # noqa: BLE001
            logger.debug("[CONCEPT:AU-KG.ingest.list-durable-media] blob incref skipped: %s", e)

        # Link asset→blob (:hasBlob) and asset→message (:attachedToMessage) edges so the
        # graph is navigable. Best-effort, outside the txn (pure graph edges).
        try:
            client.edges.add(asset_id, blob_id, {"type": "hasBlob"})
            if message_id:
                client.edges.add(asset_id, message_id, {"type": "attachedToMessage"})
        except Exception as e:  # noqa: BLE001
            logger.debug("[CONCEPT:AU-KG.ingest.list-durable-media] media edge link skipped: %s", e)

        logger.info(
            "[CONCEPT:AU-KG.ingest.list-durable-media] stored media asset %s (digest=%s, %d bytes, new=%s)",
            asset_id,
            digest[:12],
            len(data),
            blob_is_new,
        )
        return StoredMedia(
            asset_id=asset_id,
            digest=digest,
            deduped=not blob_is_new,
            size_bytes=len(data),
        )

    # -- fetch ---------------------------------------------------------------
    def fetch_bytes(self, digest: str) -> bytes | None:
        """Fetch the raw bytes of a stored blob by digest (``None`` on failure)."""
        try:
            return self._client.blob.fetch(digest)
        except Exception as e:  # noqa: BLE001
            logger.warning("[CONCEPT:AU-KG.ingest.list-durable-media] blob fetch failed (%s): %s", digest, e)
            return None

    def fetch_asset(self, asset_id: str) -> bytes | None:
        """Fetch the bytes for a ``:MediaAsset`` by its node id (resolves the digest)."""
        try:
            props = self._client.nodes.properties(asset_id) or {}
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[CONCEPT:AU-KG.ingest.list-durable-media] asset lookup failed (%s): %s", asset_id, e
            )
            return None
        digest = props.get("content_digest")
        if not digest:
            return None
        return self.fetch_bytes(str(digest))
