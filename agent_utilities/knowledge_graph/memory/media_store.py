"""First-class multimodal memory over the engine BLOB substrate (CONCEPT:AU-KG.ingest.list-durable-media).

Media (an image a user sent, a voice note, a chart) used to be ephemeral and absent
from the KG: the messaging layer transcribed audio then ``os.unlink``ed it, wrapped
images as inline ``BinaryContent`` then discarded them, and persisted only TEXT. This
module makes media **durable and first-class** by storing the raw bytes in the engine's
content-addressed BLOB store (CONCEPT:EG-KG.storage.blob-namespace) and recording a KG
node that references the blob — so "show me the chart they sent yesterday" becomes a
real query.

Identity chain (CONCEPT:AU-KG.identity.asset-occurrence — AU-P1-4)
--------------------------------------------------------------------
Earlier versions of this module derived BOTH the blob id AND the asset node id from
the content digest. That's wrong: it means the SAME bytes seen in a second message,
tenant, or legal context silently **collapsed onto ONE node**, overwriting whatever
source/tenant/ACL/retention/legal-hold the first occurrence had recorded — a real
provenance loss, not a cache hit. The model now separates *what the bytes are* from
*how/when/by-whom they occurred*:

* **``:Blob``** (id ``blob:<digest>``, optionally ``blob:<tenant>:<digest>`` — see
  ``tenant_isolated_blob``) — the ONLY thing that dedups. Content-addressed: the same
  bytes always yield the same digest and store zero new chunks.
* **``:Rendition``** (id ``rendition:<uuid>``, :meth:`MediaStore.store_rendition`) —
  a DERIVED form of a blob (thumbnail, transcode, OCR/ASR extraction). Its bytes
  dedup like any blob, but the ``:Rendition`` node id is a distinct uuid (never
  digest-derived) so two renditions of identical derived bytes produced by different
  models/pipelines keep separate ``model`` lineage.
* **``:AssetOccurrence``** (id ``occurrence:<uuid>``, :meth:`MediaStore.store_media`)
  — the thing that actually owns provenance: ``source``/``tenant``/``owner``/``acl``/
  ``event_time``/``retention``/``legal_hold``. Its id is a fresh uuid EVERY call —
  never derived from the digest — so the same bytes attached to two different
  messages/tenants yield ONE ``:Blob`` (dedup) but TWO distinct ``:AssetOccurrence``
  nodes, each with its own independent provenance.
* **Message/Document** references the occurrence (``:attachedToMessage``), not the
  blob directly.

One cross-modal ACID txn (CONCEPT:EG-KG.txn.reader-never-sees-node): the node, its
``__blob__`` reference (``txn.blob_ref``), and — when supplied — its vector all land
in ONE ``client.txn`` commit, so a reader never sees a node without its blob.

**GC-safety is a bracket, not a single call.** The engine's blob refcount
(``blob.incref``/``unref``) is a SEPARATE RPC from the txn commit — there is no
server-side 2PC spanning both. This module brackets the two app-side: ``incref``
happens BEFORE the txn is opened (so the blob can never be reclaimed out from under
an in-flight write), and is compensated with ``unref`` if the commit then fails or
conflicts (so a failed write never leaks a permanent reference). See
``MediaStore._commit_atomic`` for the exact sequencing and its one residual gap
(a crash between ``incref`` and a failed commit's compensating ``unref``).

**Migration.** Pre-AU-P1-4 digest-keyed ``media:<digest>`` nodes (``type ==
"MediaAsset"``) are left exactly as they are — nothing reads or rewrites them
automatically, so any existing consumer still querying ``type = 'MediaAsset'``
keeps working unchanged. :meth:`MediaStore.migrate_legacy_asset` is the opt-in
upgrade path: given a legacy asset id, it reads its ``content_digest`` (no bytes
re-fetch — the blob already exists) and mints a NEW, distinct ``:AssetOccurrence``
pointing at that same blob, carrying the legacy node's fields forward as
provenance and stamping ``legacy_asset_id`` for audit. It is intentionally NOT
idempotent/bulk — each call mints one new occurrence, matching the "never
digest-collapsed" invariant; a bulk sweep is a follow-up (see AGENTS.md).

This is a CORE capability (per "Universal capability — one core, thin entrypoints"):
the messaging stack, the webui, the terminal — every entrypoint persists media through
THIS, contributing only how it receives the bytes.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.session import GraphSession

logger = logging.getLogger(__name__)

#: Node-id prefixes.
#:
#: ``_BLOB_PREFIX`` is (optionally tenant-salted) content-derived — the one thing
#: that's allowed to collapse onto a single node. ``_OCCURRENCE_PREFIX`` and
#: ``_RENDITION_PREFIX`` are uuid-derived — NEVER digest-derived — so distinct
#: occurrences/renditions of identical bytes never collapse (CONCEPT:AU-KG.identity.asset-occurrence).
_BLOB_PREFIX = "blob:"
_OCCURRENCE_PREFIX = "occurrence:"
_RENDITION_PREFIX = "rendition:"
#: Pre-AU-P1-4 digest-keyed asset id prefix — kept ONLY for the migration shim
#: (:meth:`MediaStore.migrate_legacy_asset`) and reader back-compat; never used by
#: new writes.
_LEGACY_ASSET_PREFIX = "media:"


@dataclass(frozen=True)
class StoredMedia:
    """The result of persisting one media occurrence.

    Attributes:
        occurrence_id: The ``:AssetOccurrence`` node id — a fresh uuid every call,
            never derived from the content digest (CONCEPT:AU-KG.identity.asset-occurrence).
        digest: The content-addressed blob digest (the dedup/fetch key).
        deduped: ``True`` when the bytes were already present (no new chunks stored).
        size_bytes: The payload size.
        blob_id: The ``:Blob`` node id this occurrence references (``blob:<digest>``,
            or ``blob:<tenant>:<digest>`` when tenant-isolated).
    """

    occurrence_id: str
    digest: str
    deduped: bool
    size_bytes: int
    blob_id: str = ""

    @property
    def asset_id(self) -> str:
        """Back-compat alias for pre-AU-P1-4 callers that read ``.asset_id``.

        The value is now the distinct ``:AssetOccurrence`` id, not a digest-derived
        one — see the module docstring's identity-chain section.
        """
        return self.occurrence_id


@dataclass(frozen=True)
class StoredRendition:
    """The result of persisting one derived rendition of a blob.

    Attributes:
        rendition_id: The ``:Rendition`` node id — a fresh uuid every call.
        digest: The content-addressed digest of the DERIVED bytes.
        deduped: ``True`` when the derived bytes were already present.
        size_bytes: The derived payload size.
        blob_id: The ``:Blob`` node id backing the derived bytes.
        derived_from_digest: The digest of the SOURCE blob this was derived from.
        rendition_type: What kind of derivation this is (``"thumbnail"``,
            ``"transcode"``, ``"transcript"``, ``"ocr"``, ...).
    """

    rendition_id: str
    digest: str
    deduped: bool
    size_bytes: int
    blob_id: str
    derived_from_digest: str
    rendition_type: str


class MediaStore:
    """Persist + retrieve media as durable, content-addressed, KG-linked blobs.

    CONCEPT:AU-KG.ingest.list-durable-media / CONCEPT:AU-KG.identity.asset-occurrence.
    Bind to a live :class:`GraphComputeEngine` (or anything exposing a ``._client``
    sync engine client with ``.blob``/``.txn``/``.nodes``/``.edges`` and a
    ``graph_name``). All methods are best-effort safe to call from the messaging
    background-persist path: a failure logs and returns ``None``/``False`` rather
    than raising.
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

    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    @staticmethod
    def _blob_node_id(digest: str, tenant_salt: str) -> str:
        """The ``:Blob`` node id for ``digest`` — tenant-salted when ``tenant_salt``
        is given (CONCEPT:AU-KG.identity.asset-occurrence — cross-tenant matching
        prohibited). Salting only changes the GRAPH-side node id/dedup scope; the
        underlying content-addressed bytes are still shared engine-wide (there is
        no way to avoid that at the storage layer, nor a reason to: raw bytes
        carry no provenance by themselves)."""
        if tenant_salt:
            return f"{_BLOB_PREFIX}{tenant_salt}:{digest}"
        return f"{_BLOB_PREFIX}{digest}"

    def _blob_exists(self, blob_id: str) -> bool:
        """Whether a ``:Blob`` node for ``blob_id`` already exists in this graph."""
        try:
            return bool(self._client.nodes.has(blob_id))
        except Exception:  # noqa: BLE001 — treat unknown as "not yet present"
            return False

    def _compensate_unref(self, digest: str) -> None:
        """Best-effort undo of a pre-commit ``incref`` after a failed/conflicted commit."""
        try:
            self._client.blob.unref(digest)
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "[CONCEPT:AU-KG.identity.asset-occurrence] compensating unref skipped: %s",
                e,
            )

    def _commit_atomic(
        self,
        *,
        graph: str,
        digest: str,
        blob_id: str,
        blob_is_new: bool,
        blob_props: dict[str, Any],
        node_id: str,
        node_props: dict[str, Any],
        embedding: list[float] | None,
    ) -> bool:
        """Commit ``node_id`` + its blob reference as one unit, GC-bracketed.

        The engine's cross-modal ACID txn already makes the node/property/blob-ref/
        embedding writes land atomically WITH each other in one commit
        (CONCEPT:EG-KG.txn.reader-never-sees-node). What it does NOT cover is the
        blob's GC refcount — ``blob.incref``/``unref`` are SEPARATE RPCs outside any
        txn, so there is no server-side 2PC across the two.

        This closes that gap app-side: ``incref`` runs BEFORE the txn is opened (so
        the blob can never be reclaimed while the write is in flight), and is
        compensated with ``unref`` if the commit then fails or conflicts (so a
        failed write never leaks a permanent reference). The one residual gap a true
        distributed transaction would close is a process crash strictly between the
        successful ``incref`` and a failed commit's compensating ``unref`` — an
        acceptable residual since its only failure mode is "a blob is GC'd one sweep
        late", never a torn graph read (CONCEPT:AU-KG.identity.asset-occurrence).
        """
        client = self._client
        try:
            client.blob.incref(digest)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[CONCEPT:AU-KG.identity.asset-occurrence] blob incref failed, aborting write of %s: %s",
                node_id,
                e,
            )
            return False
        try:
            txn = client.txn.begin(graph=graph)
            if blob_is_new:
                client.txn.add_node(txn, blob_id, blob_props)
            client.txn.add_node(txn, node_id, node_props)
            client.txn.blob_ref(txn, node_id, digest)
            if embedding:
                client.txn.add_embedding(txn, node_id, list(embedding))
            committed = client.txn.commit(txn)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[CONCEPT:AU-KG.identity.asset-occurrence] ACID txn failed for %s, compensating unref: %s",
                node_id,
                e,
            )
            self._compensate_unref(digest)
            return False
        if not committed:
            logger.warning(
                "[CONCEPT:AU-KG.identity.asset-occurrence] txn conflict for %s (not committed), compensating unref",
                node_id,
            )
            self._compensate_unref(digest)
            return False
        return True

    # -- store: occurrence ----------------------------------------------------
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
        session: GraphSession | None = None,
        tenant: str | None = None,
        owner: str = "",
        acl: Any = None,
        event_time: str | None = None,
        retention: str = "",
        legal_hold: bool = False,
        provenance: dict[str, Any] | None = None,
        tenant_isolated_blob: bool | None = None,
    ) -> StoredMedia | None:
        """Store ``data`` durably and create a DISTINCT ``:AssetOccurrence`` linked to its blob.

        CONCEPT:AU-KG.identity.asset-occurrence (AU-P1-4). Steps, all on the ONE
        engine authority:

        1. ``blob.store(data)`` — content-addressed + deduped → ``digest``. This is
           the ONLY dedup point; a second call with the SAME bytes yields the same
           digest/blob but a BRAND NEW occurrence id.
        2. GC-bracketed cross-modal ACID commit (:meth:`_commit_atomic`): the
           ``:Blob`` node (if new), the ``:AssetOccurrence`` node, its ``blob_ref``,
           and the embedding (when supplied) land in ONE commit, bracketed by an
           incref-before/unref-on-failure refcount window.
        4. When ``message_id`` is given, add the ``:attachedToMessage`` edge so the
           media is reachable from the conversation memory.

        Args:
            session: Optional explicit
                :class:`~agent_utilities.knowledge_graph.core.session.GraphSession`
                (CONCEPT:AU-P0-1). When omitted, one is derived from today's ambient
                actor/trace via ``GraphSession.from_ambient()`` so ``tenant``/
                ``owner``/``source`` default sensibly even for callers that don't
                construct one explicitly. When ``session.graph`` is set, the media
                lands in that named graph instead of the bound engine's default.
            tenant: Explicit tenant id for THIS occurrence. Defaults to
                ``session.tenant``. Distinct from blob tenant-isolation (see
                ``tenant_isolated_blob``) — an occurrence always carries its own
                tenant regardless of blob sharing.
            owner: The identity that owns this occurrence (defaults to
                ``session.actor.actor_id``).
            acl: Occurrence-level access control (any JSON-serializable shape —
                a role list, a per-scope dict, etc.). Stored verbatim.
            event_time: When the underlying event occurred (ISO-8601). Defaults to
                "now" when not given (distinct from ``created_at``, which is always
                "now" — the write time).
            retention: A retention-policy label/duration for this occurrence.
            legal_hold: Whether this occurrence is under legal hold (exempt from
                retention-driven deletion).
            provenance: Free-form provenance detail (platform/channel/thread/etc.)
                distinct from the coarse ``source`` string.
            tenant_isolated_blob: When ``True``, the ``:Blob`` node id is salted
                with ``tenant`` so cross-tenant byte-identical uploads do NOT share
                a graph-side blob node (still share engine-side bytes/chunks — see
                :meth:`_blob_node_id`). Defaults to the
                ``KG_MEDIA_TENANT_ISOLATED_BLOBS`` setting (off by default, matching
                prior single-namespace behavior).

        Returns a :class:`StoredMedia` (or ``None`` on failure — never raises).
        """
        if not data:
            return None
        from ..core.session import GraphSession

        if session is None:
            session = GraphSession.from_ambient()
        if not source and session.actor is not None:
            source = session.actor.actor_id
        if tenant is None:
            tenant = session.tenant or (
                session.actor.tenant_id if session.actor is not None else ""
            )
        if not owner and session.actor is not None:
            owner = session.actor.actor_id

        client = self._client
        try:
            digest = client.blob.store(data)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[CONCEPT:AU-KG.ingest.list-durable-media] blob store failed: %s", e
            )
            return None

        isolate = (
            tenant_isolated_blob
            if tenant_isolated_blob is not None
            else _tenant_isolated_blobs_setting()
        )
        blob_id = self._blob_node_id(digest, tenant if isolate else "")
        blob_is_new = not self._blob_exists(blob_id)
        now = self._now()

        occurrence_id = f"{_OCCURRENCE_PREFIX}{uuid.uuid4().hex}"
        occurrence_props: dict[str, Any] = {
            "type": "AssetOccurrence",
            "name": name or f"media {media_type or mime_type or digest[:8]}",
            "content_digest": digest,
            "blob_id": blob_id,
            "media_type": media_type,
            "mime_type": mime_type,
            "file_size_bytes": len(data),
            "source": source,
            "tenant": tenant,
            "owner": owner,
            "event_time": event_time or now,
            "retention": retention,
            "legal_hold": bool(legal_hold),
            "created_at": now,
        }
        if acl is not None:
            occurrence_props["acl"] = acl
        if message_id:
            occurrence_props["message_id"] = message_id
        if provenance:
            occurrence_props["provenance"] = provenance
        if session.trace_context:
            occurrence_props["trace_context"] = session.trace_context
        if extra:
            occurrence_props.update(extra)

        effective_graph = session.graph or self._graph
        blob_props = {
            "type": "Blob",
            "content_digest": digest,
            "file_size_bytes": len(data),
            "created_at": now,
        }

        committed = self._commit_atomic(
            graph=effective_graph,
            digest=digest,
            blob_id=blob_id,
            blob_is_new=blob_is_new,
            blob_props=blob_props,
            node_id=occurrence_id,
            node_props=occurrence_props,
            embedding=embedding,
        )
        if not committed:
            return None

        # Link occurrence→blob (:hasBlob) and occurrence→message (:attachedToMessage)
        # edges so the graph is navigable. Best-effort, outside the txn (pure graph edges).
        try:
            client.edges.add(occurrence_id, blob_id, {"type": "hasBlob"})
            if message_id:
                client.edges.add(
                    occurrence_id, message_id, {"type": "attachedToMessage"}
                )
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "[CONCEPT:AU-KG.identity.asset-occurrence] occurrence edge link skipped: %s",
                e,
            )

        logger.info(
            "[CONCEPT:AU-KG.identity.asset-occurrence] stored occurrence %s (digest=%s, %d bytes, blob_new=%s, tenant=%s)",
            occurrence_id,
            digest[:12],
            len(data),
            blob_is_new,
            tenant,
        )
        return StoredMedia(
            occurrence_id=occurrence_id,
            digest=digest,
            deduped=not blob_is_new,
            size_bytes=len(data),
            blob_id=blob_id,
        )

    # -- store: rendition -------------------------------------------------------
    def store_rendition(
        self,
        data: bytes,
        *,
        source_digest: str,
        rendition_type: str,
        occurrence_id: str | None = None,
        model: str = "",
        mime_type: str = "",
        extra: dict[str, Any] | None = None,
        session: GraphSession | None = None,
    ) -> StoredRendition | None:
        """Persist a DERIVED form of a blob (thumbnail/transcode/extraction) as its
        own content-addressed blob + a ``:Rendition`` node (CONCEPT:AU-KG.identity.asset-occurrence).

        Renditions dedup on their OWN bytes exactly like a top-level blob — two
        callers deriving byte-identical thumbnails from the same (or different)
        source share one blob — but each call still mints a DISTINCT ``:Rendition``
        node id (never digest-derived), so two renditions of identical derived bytes
        produced by different models/pipelines keep separate ``model`` lineage. This
        is the seam async extraction/embedding pipelines write into: the heavy model
        call happens wherever it runs (remote/optional); it hands its OUTPUT bytes
        here.

        Args:
            data: The derived bytes (e.g. a generated thumbnail, a transcoded clip).
            source_digest: The digest of the blob this rendition was derived FROM.
            rendition_type: What kind of derivation this is (``"thumbnail"``,
                ``"transcode"``, ``"transcript"``, ``"ocr"``, ...).
            occurrence_id: Optional owning ``:AssetOccurrence`` id — when given, a
                ``:hasRendition`` edge links occurrence → rendition.
            model: The model/pipeline that produced this rendition (lineage).

        Returns a :class:`StoredRendition` (or ``None`` on failure — never raises).
        """
        if not data:
            return None
        from ..core.session import GraphSession

        if session is None:
            session = GraphSession.from_ambient()

        client = self._client
        try:
            digest = client.blob.store(data)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[CONCEPT:AU-KG.identity.asset-occurrence] rendition blob store failed: %s",
                e,
            )
            return None

        blob_id = f"{_BLOB_PREFIX}{digest}"
        blob_is_new = not self._blob_exists(blob_id)
        now = self._now()
        rendition_id = f"{_RENDITION_PREFIX}{uuid.uuid4().hex}"

        rendition_props: dict[str, Any] = {
            "type": "Rendition",
            "content_digest": digest,
            "blob_id": blob_id,
            "rendition_type": rendition_type,
            "derived_from_digest": source_digest,
            "model": model,
            "mime_type": mime_type,
            "file_size_bytes": len(data),
            "created_at": now,
        }
        if occurrence_id:
            rendition_props["occurrence_id"] = occurrence_id
        if extra:
            rendition_props.update(extra)

        effective_graph = session.graph or self._graph
        blob_props = {
            "type": "Blob",
            "content_digest": digest,
            "file_size_bytes": len(data),
            "created_at": now,
        }

        committed = self._commit_atomic(
            graph=effective_graph,
            digest=digest,
            blob_id=blob_id,
            blob_is_new=blob_is_new,
            blob_props=blob_props,
            node_id=rendition_id,
            node_props=rendition_props,
            embedding=None,
        )
        if not committed:
            return None

        try:
            client.edges.add(rendition_id, blob_id, {"type": "hasBlob"})
            client.edges.add(
                rendition_id, f"{_BLOB_PREFIX}{source_digest}", {"type": "derivedFrom"}
            )
            if occurrence_id:
                client.edges.add(
                    occurrence_id, rendition_id, {"type": "hasRendition"}
                )
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "[CONCEPT:AU-KG.identity.asset-occurrence] rendition edge link skipped: %s",
                e,
            )

        return StoredRendition(
            rendition_id=rendition_id,
            digest=digest,
            deduped=not blob_is_new,
            size_bytes=len(data),
            blob_id=blob_id,
            derived_from_digest=source_digest,
            rendition_type=rendition_type,
        )

    # -- async extraction/embedding seam -----------------------------------------
    def record_extraction(
        self,
        node_id: str,
        *,
        model: str = "",
        extracted_text: str = "",
        embedding: list[float] | None = None,
        extra: dict[str, Any] | None = None,
        session: GraphSession | None = None,
    ) -> bool:
        """Attach an (optionally embedded) extraction result to an EXISTING
        occurrence or rendition node (CONCEPT:AU-KG.identity.asset-occurrence).

        The seam an async extraction/embedding pipeline calls back into: the actual
        heavy model call (OCR/ASR/vision-embedding) is expected to run wherever it
        runs — remote, a background worker, optional entirely — and hands its
        result here. Retains model lineage (``model``) so a reader can tell WHICH
        model produced the extraction. Read-modify-write on the node's existing
        properties (the node already exists; this is not a new occurrence).

        Returns ``True`` on a successful commit, ``False`` otherwise (never raises).
        """
        if not node_id:
            return False
        from ..core.session import GraphSession

        if session is None:
            session = GraphSession.from_ambient()

        client = self._client
        try:
            props = dict(client.nodes.properties(node_id) or {})
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[CONCEPT:AU-KG.identity.asset-occurrence] extraction target lookup failed (%s): %s",
                node_id,
                e,
            )
            return False
        if not props:
            logger.warning(
                "[CONCEPT:AU-KG.identity.asset-occurrence] extraction target not found: %s",
                node_id,
            )
            return False

        props["extraction_model"] = model
        if extracted_text:
            props["extracted_text"] = extracted_text
        props["extracted_at"] = self._now()
        if extra:
            props.update(extra)

        effective_graph = session.graph or self._graph
        try:
            txn = client.txn.begin(graph=effective_graph)
            client.txn.add_node(txn, node_id, props)
            if embedding:
                client.txn.add_embedding(txn, node_id, list(embedding))
            committed = client.txn.commit(txn)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[CONCEPT:AU-KG.identity.asset-occurrence] extraction commit failed for %s: %s",
                node_id,
                e,
            )
            return False
        return bool(committed)

    # -- migration shim -----------------------------------------------------
    def migrate_legacy_asset(
        self,
        legacy_asset_id: str,
        *,
        session: GraphSession | None = None,
    ) -> StoredMedia | None:
        """Upgrade a pre-AU-P1-4 digest-keyed ``media:<digest>`` node into a proper
        ``:AssetOccurrence`` (CONCEPT:AU-KG.identity.asset-occurrence — the back-compat
        upgrade path for AU-P1-4).

        Before this change, ``store_media`` derived BOTH the blob id and the asset
        id from the content digest, so the same bytes seen a second time silently
        collapsed onto ONE node. Existing ``media:<digest>`` nodes are left exactly
        as they are — nothing here mutates or deletes them, so any reader still
        querying ``type = 'MediaAsset'`` keeps working unchanged. This method reads
        one legacy node's ``content_digest`` (the blob already exists — no bytes
        re-fetch/re-store) and mints a NEW, distinct ``:AssetOccurrence`` pointing
        at that SAME blob, carrying the legacy node's ``source``/``media_type``/
        ``mime_type``/``message_id``/``created_at`` forward as ``provenance`` and
        stamping ``legacy_asset_id`` for audit.

        This is intentionally NOT idempotent and NOT a bulk migration: each call
        mints one new occurrence for one legacy asset (a bulk sweep over every
        ``type = 'MediaAsset'`` node is a follow-up — see AGENTS.md). Calling it
        twice for the same legacy id creates two occurrences, which is consistent
        with (not a violation of) the "occurrence identity is never digest-
        collapsed" invariant.

        Returns the new :class:`StoredMedia` (or ``None`` when the legacy node is
        missing/has no digest, or the write fails — never raises).
        """
        from ..core.session import GraphSession

        if session is None:
            session = GraphSession.from_ambient()

        client = self._client
        try:
            legacy = client.nodes.properties(legacy_asset_id)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[CONCEPT:AU-KG.identity.asset-occurrence] legacy asset lookup failed (%s): %s",
                legacy_asset_id,
                e,
            )
            return None
        if not legacy:
            logger.warning(
                "[CONCEPT:AU-KG.identity.asset-occurrence] legacy asset not found: %s",
                legacy_asset_id,
            )
            return None
        digest = legacy.get("content_digest")
        if not digest:
            logger.warning(
                "[CONCEPT:AU-KG.identity.asset-occurrence] legacy asset %s has no content_digest, cannot migrate",
                legacy_asset_id,
            )
            return None

        blob_id = f"{_BLOB_PREFIX}{digest}"
        blob_is_new = not self._blob_exists(blob_id)
        now = self._now()
        occurrence_id = f"{_OCCURRENCE_PREFIX}{uuid.uuid4().hex}"

        occurrence_props: dict[str, Any] = {
            "type": "AssetOccurrence",
            "name": legacy.get("name", "") or f"media {digest[:8]}",
            "content_digest": digest,
            "blob_id": blob_id,
            "media_type": legacy.get("media_type", ""),
            "mime_type": legacy.get("mime_type", ""),
            "file_size_bytes": legacy.get("file_size_bytes", 0),
            "source": legacy.get("source", ""),
            "tenant": session.tenant,
            "owner": session.actor.actor_id if session.actor is not None else "",
            "event_time": legacy.get("created_at", now),
            "retention": "",
            "legal_hold": False,
            "created_at": now,
            "legacy_asset_id": legacy_asset_id,
            "provenance": {
                "migrated_from": legacy_asset_id,
                "legacy_source": legacy.get("source", ""),
                "legacy_created_at": legacy.get("created_at", ""),
            },
        }
        if legacy.get("message_id"):
            occurrence_props["message_id"] = legacy["message_id"]

        effective_graph = session.graph or self._graph
        blob_props = {
            "type": "Blob",
            "content_digest": digest,
            "file_size_bytes": legacy.get("file_size_bytes", 0),
            "created_at": now,
        }

        committed = self._commit_atomic(
            graph=effective_graph,
            digest=digest,
            blob_id=blob_id,
            blob_is_new=blob_is_new,
            blob_props=blob_props,
            node_id=occurrence_id,
            node_props=occurrence_props,
            embedding=None,
        )
        if not committed:
            return None

        try:
            client.edges.add(occurrence_id, blob_id, {"type": "hasBlob"})
            client.edges.add(
                occurrence_id, legacy_asset_id, {"type": "migratedFrom"}
            )
            if legacy.get("message_id"):
                client.edges.add(
                    occurrence_id,
                    legacy["message_id"],
                    {"type": "attachedToMessage"},
                )
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "[CONCEPT:AU-KG.identity.asset-occurrence] migrated occurrence edge link skipped: %s",
                e,
            )

        return StoredMedia(
            occurrence_id=occurrence_id,
            digest=digest,
            deduped=not blob_is_new,
            size_bytes=int(legacy.get("file_size_bytes", 0) or 0),
            blob_id=blob_id,
        )

    # -- fetch ---------------------------------------------------------------
    def fetch_bytes(self, digest: str) -> bytes | None:
        """Fetch the raw bytes of a stored blob by digest (``None`` on failure)."""
        try:
            return self._client.blob.fetch(digest)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[CONCEPT:AU-KG.ingest.list-durable-media] blob fetch failed (%s): %s",
                digest,
                e,
            )
            return None

    def fetch_asset(self, asset_id: str) -> bytes | None:
        """Fetch the bytes for an ``:AssetOccurrence``/``:Rendition``/legacy
        ``:MediaAsset`` node by its node id (resolves the digest). Kept under its
        pre-AU-P1-4 name for back-compat; works unchanged for any node id carrying
        a ``content_digest`` property."""
        try:
            props = self._client.nodes.properties(asset_id) or {}
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[CONCEPT:AU-KG.ingest.list-durable-media] asset lookup failed (%s): %s",
                asset_id,
                e,
            )
            return None
        digest = props.get("content_digest")
        if not digest:
            return None
        return self.fetch_bytes(str(digest))

    # ``fetch_occurrence`` is the AU-P1-4-native name for ``fetch_asset``.
    fetch_occurrence = fetch_asset


def _tenant_isolated_blobs_setting() -> bool:
    """Whether ``:Blob`` node ids should be tenant-salted by default (CONCEPT:AU-KG.identity.asset-occurrence).

    Deferred import — ``core.config`` pulls in a heavier chain than this module
    otherwise needs, and this is only read on the (uncommon) tenant-isolation path.
    """
    from agent_utilities.core.config import setting

    return bool(setting("KG_MEDIA_TENANT_ISOLATED_BLOBS", False))
