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
keeps working unchanged. :meth:`MediaStore.migrate_legacy_asset` is the opt-in,
per-id upgrade path: given a legacy asset id, it reads its ``content_digest`` (no
bytes re-fetch — the blob already exists) and mints a NEW, distinct
``:AssetOccurrence`` pointing at that same blob, carrying the legacy node's
fields forward as provenance and stamping ``legacy_asset_id`` for audit. Calling
it twice for the SAME legacy id mints two occurrences — consistent with (not a
violation of) the "never digest-collapsed" invariant.

:meth:`MediaStore.migrate_legacy_assets_bulk` is the bulk follow-up: it sweeps
EVERY ``type == 'MediaAsset'`` node and migrates each through that same per-id
shim, batched and idempotent — it skips legacy ids that already have a migrated
``:AssetOccurrence`` (detected by scanning existing occurrences' stamped
``legacy_asset_id``), so re-running the sweep after a partial run (or just for
safety) only migrates what's left. Legacy nodes are never mutated or deleted by
either path.

This is a CORE capability (per "Universal capability — one core, thin entrypoints"):
the messaging stack, the webui, the terminal — every entrypoint persists media through
THIS, contributing only how it receives the bytes.

**Evidence-spine convergence (Seam 2, CONCEPT:AU-KG.identity.evidence-spine-convergence /
EG-X1).** Storing an ``:AssetOccurrence`` records *that* some bytes occurred; it says
nothing about *where inside those bytes* a claim's evidence sits. epistemic-graph's
own evidence-graph (``eg_epistemic::evidence.rs``, feature ``evidence-graph``) already
resolves that from an ``:Evidence``-role node's ``evidence_span``/``occurrence_id``/
``blob_ref`` properties via ``Method::ExplainEvidence`` — the SAME typed-node-by-
convention ``SourceObject -> AssetOccurrence -> Blob`` identity chain this module's
docstring describes above, just not previously linked to a located locus. Before this,
an occurrence stored via AU had no path into that resolver: a claim citing it could
name the occurrence but never recover the exact page/box/span.
:meth:`MediaStore.store_document_page_evidence` closes that gap for the document
modality: it stores the media (as :meth:`store_media` always has), then ALSO writes a
``:SourceObject`` node for the owning document, an ``:Evidence`` node carrying a
``PageBox`` ``EvidenceSpan`` locus plus the occurrence/blob identity chain, and —
when a ``claim_id`` is given — the SAME ``relationship_type: "SUPPORTS"`` edge
convention ``eg_epistemic``'s own claim materialization
(``src/server/handlers/mining.rs::materialize_claim``) writes. No new engine
write endpoint was needed: the generic ``nodes.add``/``edges.add`` RPCs the rest of
this module already uses are sufficient to produce the EXACT property/edge shape
``BeliefGraph::from_graph_view`` decodes — the resolver is entirely engine-side and is
reused unchanged (see ``crates/eg-epistemic/tests/x1_evidence_chain.rs`` for the
engine's own acceptance test of that decode path, and
``tests/unit/knowledge_graph/test_media_store_evidence_spine.py`` for this module's
half of the round-trip). Opt-in by construction: nothing about :meth:`store_media`
changes, and a caller that never calls :meth:`store_document_page_evidence` writes
nothing extra. Scoped to ONE modality (document page-box) end-to-end; the SAME
pattern (an ``:Evidence`` node's ``evidence_span`` set to the modality's own
``EvidenceSpan`` variant, plus ``occurrence_id``/``blob_ref``) extends unchanged to
the other ten loci ``eg_modality::EvidenceSpan`` defines (image region, audio/video
segment, table cell range, code symbol, …) — see ``docs/architecture/
evidence_spine_convergence.md``.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class BulkMigrationResult:
    """Outcome of a :meth:`MediaStore.migrate_legacy_assets_bulk` sweep.

    Attributes:
        scanned: Total legacy ``type == 'MediaAsset'`` nodes found this run.
        migrated: NEW ``:AssetOccurrence`` nodes minted this run (excludes
            skips/failures).
        skipped_already_migrated: Legacy ids that already had a migrated
            occurrence from a prior run — the idempotency no-op count.
        failed: Legacy ids :meth:`MediaStore.migrate_legacy_asset` could not
            migrate (e.g. missing/no ``content_digest`` — never raises, just
            counted here).
        occurrence_ids: The newly minted ``:AssetOccurrence`` ids, in migration
            order.
        failed_ids: The legacy ids that failed to migrate, for retry/audit.
    """

    scanned: int = 0
    migrated: int = 0
    skipped_already_migrated: int = 0
    failed: int = 0
    occurrence_ids: list[str] = field(default_factory=list)
    failed_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "migrated": self.migrated,
            "skipped_already_migrated": self.skipped_already_migrated,
            "failed": self.failed,
            "occurrence_ids": list(self.occurrence_ids),
            "failed_ids": list(self.failed_ids),
        }


@dataclass(frozen=True)
class DocumentEvidenceLocus:
    """The result of :meth:`MediaStore.store_document_page_evidence` — an
    ``:AssetOccurrence`` whose exact page+box locus is now resolvable through the
    ONE epistemic-graph evidence spine (CONCEPT:AU-KG.identity.evidence-spine-convergence,
    EG-X1).

    Attributes:
        source_object_id: The ``:SourceObject`` node id for the owning document
            (``sourceobject:<document_id>``) — the top of the identity chain.
        occurrence_id: The ``:AssetOccurrence`` node id (from the underlying
            :class:`StoredMedia`).
        blob_id: The ``:Blob`` node id the occurrence resolves to.
        evidence_id: The ``:Evidence`` node id carrying the located ``PageBox``
            locus plus the ``occurrence_id``/``blob_ref`` identity chain —
            the id `Method::ExplainEvidence`/``evidence_citations`` walk to.
        claim_id: The claim this evidence was linked to via a ``SUPPORTS`` edge,
            when one was given.
        digest: The content-addressed digest of the stored bytes.
    """

    source_object_id: str
    occurrence_id: str
    blob_id: str
    evidence_id: str
    digest: str
    claim_id: str | None = None


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

    # -- evidence-spine through-write (Seam 2, CONCEPT:AU-KG.identity.evidence-spine-convergence) --
    def store_document_page_evidence(
        self,
        data: bytes,
        *,
        document_id: str,
        page: int,
        x: float,
        y: float,
        width: float,
        height: float,
        mime_type: str = "application/pdf",
        source: str = "",
        claim_id: str | None = None,
        confidence: float = 1.0,
        session: GraphSession | None = None,
        **store_media_kwargs: Any,
    ) -> DocumentEvidenceLocus | None:
        """Store a document page's bytes AND through-write the EG evidence-graph
        typed-node chain for its ``PageBox`` locus (Seam 2, CONCEPT:AU-KG.identity.
        evidence-spine-convergence / EG-X1) — opt-in: nothing about
        :meth:`store_media` changes; call THIS when the caller has a located
        page+box to record, so the resulting occurrence becomes resolvable via
        epistemic-graph's OWN citation resolver (``Method::ExplainEvidence`` /
        ``eg_epistemic::evidence_citations``) rather than a second, AU-side one.

        Writes, beyond the usual :meth:`store_media` occurrence/blob:

        1. A ``:SourceObject`` node for the owning document (``sourceobject:
           <document_id>``, upserted once — a repeat call for the same
           ``document_id`` reuses it) plus a structural ``hasOccurrence`` edge to
           the new ``:AssetOccurrence``.
        2. An ``:Evidence`` node carrying the located ``PageBox``
           ``eg_modality::EvidenceSpan`` locus (as the externally-tagged
           ``{"PageBox": {...}}`` shape ``BeliefGraph::from_graph_view`` decodes)
           plus ``occurrence_id``/``blob_ref`` — the SAME identity-chain
           convention ``eg_epistemic::evidence`` documents — and a structural
           ``extractedFrom`` edge back to the occurrence.
        3. When ``claim_id`` is given, a ``relationship_type: "SUPPORTS"`` edge
           from the evidence node to it — the SAME convention
           ``src/server/handlers/mining.rs::materialize_claim``'s own
           ``supports_edge`` writes, so ``eg_epistemic``'s support/contradiction/
           attack walk (and hence ``evidence_citations``) recognizes it with no
           engine-side change.

        Args:
            data: The page's rendered/extracted bytes (e.g. a page image or PDF
                slice) — stored exactly like :meth:`store_media`.
            document_id: The owning document's id — becomes the ``PageBox``
                locus's ``document_id`` AND the ``:SourceObject`` node's key.
            page: 1-indexed (or however the caller numbers) page number.
            x, y, width, height: The box's coordinates on that page, in
                whatever unit the caller's page-rendering pipeline uses
                (mirrors ``eg_modality::EvidenceSpan::PageBox`` — pass-through,
                no unit conversion here).
            claim_id: An existing ``:Claim``/belief-bearing node id to
                SUPPORTS-link this evidence to. When omitted, the evidence node
                is written but not yet cited by any claim (a later call can link
                it by writing that edge directly).
            confidence: This evidence node's own belief prior (read by
                ``BeliefGraph::from_graph_view`` as ``confidence``, default
                ``1.0`` — the media was actually observed, not inferred).
            store_media_kwargs: Forwarded verbatim to :meth:`store_media`
                (``embedding``, ``tenant``, ``owner``, ``acl``, ``message_id``, …).

        Returns a :class:`DocumentEvidenceLocus` (or ``None`` on failure — never
        raises, matching every other write in this module).
        """
        stored = self.store_media(
            data,
            media_type="document_page",
            mime_type=mime_type,
            source=source,
            session=session,
            **store_media_kwargs,
        )
        if stored is None:
            return None

        client = self._client
        now = self._now()
        source_object_id = f"sourceobject:{document_id}"
        try:
            if not bool(client.nodes.has(source_object_id)):
                client.nodes.add(
                    source_object_id,
                    {
                        "type": "SourceObject",
                        "document_id": document_id,
                        "mime_type": mime_type,
                        "created_at": now,
                    },
                )
            client.edges.add(
                source_object_id, stored.occurrence_id, {"type": "hasOccurrence"}
            )
        except Exception as e:  # noqa: BLE001 — best-effort, mirrors the module's posture
            logger.warning(
                "[CONCEPT:AU-KG.identity.evidence-spine-convergence] SourceObject write failed for %s: %s",
                document_id,
                e,
            )
            return None

        evidence_id = f"evidence:{uuid.uuid4().hex}"
        evidence_props: dict[str, Any] = {
            "type": "Evidence",
            "about": document_id,
            "confidence": float(confidence),
            "evidence_span": {
                "PageBox": {
                    "document_id": document_id,
                    "page": int(page),
                    "x": float(x),
                    "y": float(y),
                    "width": float(width),
                    "height": float(height),
                }
            },
            "occurrence_id": stored.occurrence_id,
            "blob_ref": stored.blob_id,
            "created_at": now,
        }
        try:
            client.nodes.add(evidence_id, evidence_props)
            client.edges.add(
                evidence_id, stored.occurrence_id, {"type": "extractedFrom"}
            )
            if claim_id:
                client.edges.add(
                    evidence_id, claim_id, {"relationship_type": "SUPPORTS"}
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[CONCEPT:AU-KG.identity.evidence-spine-convergence] Evidence write failed for occurrence %s: %s",
                stored.occurrence_id,
                e,
            )
            return None

        logger.info(
            "[CONCEPT:AU-KG.identity.evidence-spine-convergence] wrote evidence-graph chain "
            "%s -> %s -> %s -> evidence %s (page=%s, claim=%s)",
            source_object_id,
            stored.occurrence_id,
            stored.blob_id,
            evidence_id,
            page,
            claim_id or "-",
        )
        return DocumentEvidenceLocus(
            source_object_id=source_object_id,
            occurrence_id=stored.occurrence_id,
            blob_id=stored.blob_id,
            evidence_id=evidence_id,
            digest=stored.digest,
            claim_id=claim_id,
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
                client.edges.add(occurrence_id, rendition_id, {"type": "hasRendition"})
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

        This per-id call is intentionally NOT idempotent: calling it twice for the
        same legacy id creates two occurrences, which is consistent with (not a
        violation of) the "occurrence identity is never digest-collapsed"
        invariant. :meth:`migrate_legacy_assets_bulk` is the idempotent BULK sweep
        built on top of this shim (it skips legacy ids already migrated).

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
            client.edges.add(occurrence_id, legacy_asset_id, {"type": "migratedFrom"})
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

    # -- bulk migration --------------------------------------------------------
    def _migrated_legacy_ids(self) -> set[str]:
        """Legacy asset ids that already have a migrated ``:AssetOccurrence``.

        The idempotency check for :meth:`migrate_legacy_assets_bulk`: scans
        existing ``:AssetOccurrence`` nodes (a bounded per-type fetch via
        :func:`~..core.bounded_read.iter_nodes_by_types` — never a whole-graph
        scan) for the ``legacy_asset_id`` :meth:`migrate_legacy_asset` stamps onto
        every occurrence it mints, so a bulk sweep never mints a second occurrence
        for an already-migrated legacy asset. Best-effort: a scan failure yields an
        empty set (worst case a re-run mints one duplicate occurrence for the
        affected ids, rather than raising).
        """
        from ..core.bounded_read import iter_nodes_by_types

        migrated: set[str] = set()
        try:
            for _nid, data in iter_nodes_by_types(self._compute, "AssetOccurrence"):
                legacy_id = (
                    data.get("legacy_asset_id") if isinstance(data, dict) else None
                )
                if legacy_id:
                    migrated.add(str(legacy_id))
        except Exception as e:  # noqa: BLE001 — best-effort idempotency check
            logger.warning(
                "[CONCEPT:AU-KG.identity.asset-occurrence] scan for already-migrated legacy assets failed: %s",
                e,
            )
        return migrated

    def migrate_legacy_assets_bulk(
        self,
        *,
        batch_size: int = 100,
        session: GraphSession | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> BulkMigrationResult:
        """Sweep EVERY legacy ``type == 'MediaAsset'`` node and migrate it.

        CONCEPT:AU-KG.identity.asset-occurrence — the bulk follow-up to the
        per-id :meth:`migrate_legacy_asset` shim (AU-P1-4):

        * **Idempotent** — a legacy asset that already has a migrated
          ``:AssetOccurrence`` (per :meth:`_migrated_legacy_ids`) is skipped, so
          re-running the sweep (after a partial run, or just for safety) migrates
          only what's left and mints no duplicates.
        * **Batched** — processes ``batch_size`` legacy ids at a time, invoking
          ``progress`` (if given) with a running summary after each batch, so a
          caller can report progress on a large sweep without waiting for it to
          finish.
        * **Non-destructive** — every migration goes through the existing per-id
          :meth:`migrate_legacy_asset` shim, which never mutates or deletes the
          legacy node; the sweep itself never touches legacy nodes either.

        Args:
            batch_size: Legacy ids processed per batch before a ``progress`` call.
            session: Optional explicit session forwarded to every
                :meth:`migrate_legacy_asset` call (actor/tenant default from the
                ambient session when omitted, as usual).
            progress: Optional callback invoked after each batch with a running
                summary dict (``scanned``/``processed``/``migrated``/
                ``skipped_already_migrated``/``failed``). A raising callback is
                logged and ignored — it never aborts the sweep.

        Returns a :class:`BulkMigrationResult` summarizing the whole sweep.
        """
        from ..core.bounded_read import iter_nodes_by_types

        already_migrated = self._migrated_legacy_ids()
        legacy_ids = [
            nid for nid, _data in iter_nodes_by_types(self._compute, "MediaAsset")
        ]
        scanned = len(legacy_ids)
        migrated_occurrence_ids: list[str] = []
        failed_ids: list[str] = []
        skipped = 0

        batch_size = max(1, int(batch_size))
        for start in range(0, scanned, batch_size):
            batch = legacy_ids[start : start + batch_size]
            for legacy_id in batch:
                if legacy_id in already_migrated:
                    skipped += 1
                    continue
                migration = self.migrate_legacy_asset(legacy_id, session=session)
                if migration is None:
                    failed_ids.append(legacy_id)
                else:
                    migrated_occurrence_ids.append(migration.occurrence_id)
                    # Guard against a duplicate within this same run (defensive;
                    # legacy_ids has no repeats in practice).
                    already_migrated.add(legacy_id)

            if progress is not None:
                try:
                    progress(
                        {
                            "scanned": scanned,
                            "processed": min(start + batch_size, scanned),
                            "migrated": len(migrated_occurrence_ids),
                            "skipped_already_migrated": skipped,
                            "failed": len(failed_ids),
                        }
                    )
                except Exception as e:  # noqa: BLE001 — a bad progress callback never aborts the sweep
                    logger.debug(
                        "[CONCEPT:AU-KG.identity.asset-occurrence] migrate_legacy_assets_bulk progress callback failed: %s",
                        e,
                    )

        result = BulkMigrationResult(
            scanned=scanned,
            migrated=len(migrated_occurrence_ids),
            skipped_already_migrated=skipped,
            failed=len(failed_ids),
            occurrence_ids=migrated_occurrence_ids,
            failed_ids=failed_ids,
        )
        logger.info(
            "[CONCEPT:AU-KG.identity.asset-occurrence] bulk migration: scanned=%d migrated=%d skipped=%d failed=%d",
            result.scanned,
            result.migrated,
            result.skipped_already_migrated,
            result.failed,
        )
        return result

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
