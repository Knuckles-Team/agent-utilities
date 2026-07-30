"""KV-cache checkpoints as first-class, content-addressed graph resources.

CONCEPT:AU-KG.memory.kv-checkpoint-resource — the operator's design: checkpoint a KV
cache at a point of ideal understanding, then either initialise a NEW agent from it
or reset an EXISTING conversation back to it to restore a context window, without
re-prefilling from scratch.

Mirrors :class:`~agent_utilities.knowledge_graph.memory.media_store.MediaStore`'s
established pattern for exactly this reason: a heavy binary payload (there, arbitrary
media; here, a serialized KV-cache blob) belongs behind the engine's content-addressed
blob store — dedup'd, GC-bracketed, fetched by digest — never inlined into a node
property ("the graph is the semantic master, not a byte store"). A ``:KVCheckpoint``
node carries the provenance (which run, which point, the full key components) and a
``hasBlob`` edge to the blob; ``initializedFrom``/``restoredFrom`` edges record which
new agent run or restored conversation consumed it — genuine graph-queryable lineage,
not just a byte copy.

Security boundary, not a cache key
----------------------------------
:class:`KVCheckpointKey` hashes SIX components — model identity, quantization, serving
engine + version, prefix digest, tenant, and policy version — into the checkpoint id.
Any single component changing (a model swap, a requantization, an engine upgrade, a
different prefix, a different tenant, a rotated policy version) produces a completely
different id, so there is no accidental cross-context reuse via the normal creation
path. Because a checkpoint id can ALSO be handed around directly (an operator resuming
a run, a scheduler restoring a conversation) bypassing key re-derivation,
:meth:`KVCheckpointStore.get_checkpoint` — the single load primitive every operation
routes through — independently re-checks the loaded node's stored ``tenant`` against
the caller's ``requesting_tenant`` and (when supplied) its stored ``policy_version``
against the caller's current one, refusing the load outright on either mismatch. This
is deliberately defense-in-depth: cross-tenant checkpoint reuse is treated as a
security incident, not a correctness nicety.

Fail-closed, with an explicit, traced cold-start opt-in
--------------------------------------------------------
Every load raises a typed :class:`KVCheckpointError` (:class:`CheckpointNotFoundError`
/ :class:`CrossTenantCheckpointError` / :class:`StaleCheckpointError`) rather than
silently degrading. :meth:`KVCheckpointStore.restore_conversation` is the one place a
cold start is available, and only when the caller explicitly passes
``allow_cold_start=True`` — the fallback is then logged and recorded via
``KVCACHE_CHECKPOINT_OPS{outcome="cold_start_fallback"}`` and the returned
:class:`RestoreResult` carries ``cold_start=True`` plus the specific reason, so a
caller can never mistake a cold start for a real restore.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agent_utilities.knowledge_graph.core.session import GraphSession

logger = logging.getLogger(__name__)

#: Node-id prefix for a ``:KVCheckpoint`` node — always keyed by the deterministic,
#: content-addressed :attr:`KVCheckpointKey.checkpoint_id` (never a uuid): a repeat
#: ``create_checkpoint`` call with the SAME key components upserts the same node.
_CHECKPOINT_PREFIX = "kvcheckpoint:"
#: Node-id prefix for the heavy KV blob. ALWAYS tenant-salted (unlike
#: ``MediaStore``'s opt-in ``tenant_isolated_blob``) — checkpoint bytes are a
#: security boundary by default, so two tenants that happen to produce
#: byte-identical KV pages never share a graph-side blob node.
_BLOB_PREFIX = "kvblob:"
#: Field separator joining a :class:`KVCheckpointKey`'s components before hashing.
#: A non-printable byte no legitimate identifier/digest/version string contains, so
#: no combination of component values can collide with a different combination
#: (e.g. ``("a", "bc")`` vs ``("ab", "c")`` hash to different ids).
_KEY_SEPARATOR = "\x1f"

__all__ = [
    "CheckpointNotFoundError",
    "CrossTenantCheckpointError",
    "KVCheckpointError",
    "KVCheckpointKey",
    "KVCheckpointRecord",
    "KVCheckpointStore",
    "RestoreResult",
    "StaleCheckpointError",
]


def _record_checkpoint_op(op: str, outcome: str) -> None:
    """Best-effort KVCheckpoint telemetry (WS-4 idiom — never breaks the caller)."""
    try:
        from agent_utilities.observability.gateway_metrics import (
            KVCACHE_CHECKPOINT_OPS,
        )

        KVCACHE_CHECKPOINT_OPS.labels(op=op, outcome=outcome).inc()
    except Exception as exc:  # noqa: BLE001 — metrics must never break the caller
        logger.debug("KVCheckpoint telemetry recording failed: %s", exc)


class KVCheckpointError(RuntimeError):
    """Base for every KVCheckpoint failure. Always raised — never swallowed to a
    silent default; a caller that wants a fallback must opt in explicitly (see
    :meth:`KVCheckpointStore.restore_conversation`'s ``allow_cold_start``)."""


class CheckpointNotFoundError(KVCheckpointError):
    """No ``:KVCheckpoint`` node exists for the given id (invalid checkpoint)."""


class CrossTenantCheckpointError(KVCheckpointError):
    """The checkpoint's stored tenant does not match the requesting tenant.

    Always refused — there is no bypass flag. Serving one tenant's KV context to
    another is treated as a security incident, not a correctness nicety.
    """


class StaleCheckpointError(KVCheckpointError):
    """The checkpoint's stored policy version no longer matches the caller's
    current one — the authorization surface has moved on since it was minted."""


class KVCheckpointKey(BaseModel):
    """Every component whose change MUST invalidate the checkpoint.

    CONCEPT:AU-KG.memory.kv-checkpoint-resource. Treated as a SECURITY BOUNDARY, not a
    cache key (see module docstring): :attr:`checkpoint_id` hashes ALL SIX fields, so a
    change to any one of them mints a wholly different id — there is no way to
    accidentally resolve a checkpoint minted under different conditions.
    """

    model_identity: str = Field(
        description="Stable model identifier (e.g. 'qwen2.5-72b-instruct')."
    )
    quantization: str = Field(
        description="Quantization scheme (e.g. 'fp16', 'awq-int4', 'none')."
    )
    serving_engine: str = Field(
        description="Serving engine name (e.g. 'vllm', 'lmcache')."
    )
    engine_version: str = Field(description="Serving engine version string.")
    prefix_digest: str = Field(
        description="Content digest of the exact prompt-prefix/token sequence this "
        "checkpoint captures."
    )
    tenant: str = Field(
        min_length=1,
        description="Owning tenant id. Mandatory — never defaulted or inferred; a "
        "checkpoint with no tenant could never be safely loaded back.",
    )
    policy_version: str = Field(
        default="",
        description="Authorization/policy revision active when the checkpoint was "
        "minted (mirrors GraphSession.policy_version / KG_POLICY_VERSION).",
    )

    model_config = ConfigDict(frozen=True, extra="forbid")

    @property
    def checkpoint_id(self) -> str:
        """Deterministic content-addressed id: sha256 over every key component,
        joined with :data:`_KEY_SEPARATOR` so distinct component combinations can
        never collide onto the same id."""
        canonical = _KEY_SEPARATOR.join(
            (
                self.model_identity,
                self.quantization,
                self.serving_engine,
                self.engine_version,
                self.prefix_digest,
                self.tenant,
                self.policy_version,
            )
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class KVCheckpointRecord(BaseModel):
    """A loaded/created checkpoint's provenance (never the heavy blob bytes — see
    :meth:`KVCheckpointStore.fetch_bytes`)."""

    checkpoint_id: str
    key: KVCheckpointKey
    blob_id: str
    digest: str
    run_id: str
    point: str
    size_bytes: int
    created_at: str
    provenance: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class RestoreResult(BaseModel):
    """Outcome of :meth:`KVCheckpointStore.restore_conversation`.

    ``cold_start`` is the explicit, traced fallback signal (see module docstring) —
    a caller must check it rather than assume ``restored=False`` alone means a real
    restore attempt happened.
    """

    conversation_id: str
    checkpoint_id: str
    restored: bool
    cold_start: bool = False
    fallback_reason: str = ""
    record: KVCheckpointRecord | None = None
    data: bytes | None = None

    model_config = ConfigDict(extra="forbid")


class KVCheckpointStore:
    """Persist + load KV-cache checkpoints as durable, content-addressed, KG-linked
    graph resources (CONCEPT:AU-KG.memory.kv-checkpoint-resource).

    Bind to a live engine facade the same way
    :class:`~agent_utilities.knowledge_graph.memory.media_store.MediaStore` does — an
    object exposing a ``._client`` sync engine client with ``.blob``/``.txn``/
    ``.nodes``/``.edges`` and a ``graph_name``.
    """

    def __init__(self, compute: Any) -> None:
        self._compute = compute

    # -- internals -------------------------------------------------------------
    @property
    def _client(self) -> Any:
        client = getattr(self._compute, "_client", None)
        if client is None:
            raise RuntimeError("KVCheckpointStore: bound engine has no live client")
        return client

    @property
    def _graph(self) -> str:
        return getattr(self._compute, "graph_name", "__commons__")

    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    @staticmethod
    def _node_id(checkpoint_id: str) -> str:
        return f"{_CHECKPOINT_PREFIX}{checkpoint_id}"

    @staticmethod
    def _blob_node_id(digest: str, tenant: str) -> str:
        return f"{_BLOB_PREFIX}{tenant}:{digest}"

    def _blob_exists(self, blob_id: str) -> bool:
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
                "[CONCEPT:AU-KG.memory.kv-checkpoint-resource] compensating unref skipped: %s",
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
    ) -> bool:
        """Commit the checkpoint node + its blob reference as one GC-bracketed unit.

        Identical bracketing to ``MediaStore._commit_atomic``: ``incref`` runs BEFORE
        the txn opens (so the blob can never be reclaimed mid-write) and is
        compensated with ``unref`` on any failure/conflict (so a failed write never
        leaks a permanent reference).
        """
        client = self._client
        try:
            client.blob.incref(digest)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[CONCEPT:AU-KG.memory.kv-checkpoint-resource] blob incref failed, aborting write of %s: %s",
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
            committed = client.txn.commit(txn)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[CONCEPT:AU-KG.memory.kv-checkpoint-resource] ACID txn failed for %s, compensating unref: %s",
                node_id,
                e,
            )
            self._compensate_unref(digest)
            return False
        if not committed:
            logger.warning(
                "[CONCEPT:AU-KG.memory.kv-checkpoint-resource] txn conflict for %s (not committed), compensating unref",
                node_id,
            )
            self._compensate_unref(digest)
            return False
        return True

    # -- create ------------------------------------------------------------
    def create_checkpoint(
        self,
        data: bytes,
        *,
        key: KVCheckpointKey,
        run_id: str,
        point: str,
        session: GraphSession | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> KVCheckpointRecord | None:
        """Checkpoint ``data`` (the serialized KV-cache blob) at an explicit point in
        a live run.

        Steps, all on the ONE engine authority (mirrors
        :meth:`~agent_utilities.knowledge_graph.memory.media_store.MediaStore.store_media`):

        1. ``blob.store(data)`` — content-addressed + deduped → ``digest``.
        2. GC-bracketed cross-modal ACID commit (:meth:`_commit_atomic`): the
           tenant-salted ``:Blob`` node (if new), the ``:KVCheckpoint`` node, and its
           ``blob_ref`` land in ONE commit.
        3. A structural ``hasBlob`` edge for graph navigability.

        Returns ``None`` (never raises) on an empty payload or an engine-side
        failure — creation failures are non-fatal to the caller's run, unlike a
        LOAD failure, which is always fail-closed.
        """
        if not data:
            return None
        if not key.tenant:
            raise KVCheckpointError(
                "KVCheckpointKey.tenant is mandatory — a checkpoint with no tenant "
                "can never be safely loaded back"
            )

        from agent_utilities.knowledge_graph.core.session import (
            GraphSession as _GraphSession,
        )

        if session is None:
            session = _GraphSession.from_ambient()

        client = self._client
        try:
            digest = client.blob.store(data)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[CONCEPT:AU-KG.memory.kv-checkpoint-resource] blob store failed: %s",
                e,
            )
            _record_checkpoint_op("create", "error")
            return None

        blob_id = self._blob_node_id(digest, key.tenant)
        blob_is_new = not self._blob_exists(blob_id)
        now = self._now()
        checkpoint_id = key.checkpoint_id
        node_id = self._node_id(checkpoint_id)

        node_props: dict[str, Any] = {
            "type": "KVCheckpoint",
            "checkpoint_id": checkpoint_id,
            "model_identity": key.model_identity,
            "quantization": key.quantization,
            "serving_engine": key.serving_engine,
            "engine_version": key.engine_version,
            "prefix_digest": key.prefix_digest,
            "tenant": key.tenant,
            "policy_version": key.policy_version,
            "content_digest": digest,
            "blob_id": blob_id,
            "run_id": run_id,
            "point": point,
            "file_size_bytes": len(data),
            "created_at": now,
        }
        if provenance:
            node_props["provenance"] = provenance
        if session.trace_context:
            node_props["trace_context"] = session.trace_context

        blob_props = {
            "type": "Blob",
            "content_digest": digest,
            "file_size_bytes": len(data),
            "created_at": now,
        }

        committed = self._commit_atomic(
            graph=session.graph or self._graph,
            digest=digest,
            blob_id=blob_id,
            blob_is_new=blob_is_new,
            blob_props=blob_props,
            node_id=node_id,
            node_props=node_props,
        )
        if not committed:
            _record_checkpoint_op("create", "error")
            return None

        try:
            client.edges.add(node_id, blob_id, {"type": "hasBlob"})
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "[CONCEPT:AU-KG.memory.kv-checkpoint-resource] checkpoint→blob edge link skipped: %s",
                e,
            )

        _record_checkpoint_op("create", "ok")
        logger.info(
            "[CONCEPT:AU-KG.memory.kv-checkpoint-resource] checkpoint %s created "
            "(run=%s, point=%s, tenant=%s, %d bytes, blob_new=%s)",
            checkpoint_id[:12],
            run_id,
            point,
            key.tenant,
            len(data),
            blob_is_new,
        )
        return KVCheckpointRecord(
            checkpoint_id=checkpoint_id,
            key=key,
            blob_id=blob_id,
            digest=digest,
            run_id=run_id,
            point=point,
            size_bytes=len(data),
            created_at=now,
            provenance=provenance or {},
        )

    # -- load: the single fail-closed primitive ------------------------------
    def get_checkpoint(
        self,
        checkpoint_id: str,
        *,
        requesting_tenant: str,
        current_policy_version: str | None = None,
    ) -> KVCheckpointRecord:
        """Fail-closed checkpoint load — the ONE primitive every operation routes
        through.

        Raises :class:`CheckpointNotFoundError` for an unknown id,
        :class:`CrossTenantCheckpointError` for a tenant mismatch (ALWAYS checked,
        no bypass), and :class:`StaleCheckpointError` when
        ``current_policy_version`` is supplied and disagrees with the checkpoint's
        stored policy version. Never degrades silently — see the module docstring
        for why this is a security boundary, not a cache-miss.
        """
        if not requesting_tenant:
            raise KVCheckpointError(
                "requesting_tenant is mandatory for a checkpoint load"
            )

        node_id = self._node_id(checkpoint_id)
        try:
            props = self._client.nodes.properties(node_id)
        except Exception as exc:
            _record_checkpoint_op("get", "not_found")
            raise CheckpointNotFoundError(
                f"checkpoint lookup failed for {checkpoint_id}"
            ) from exc
        if not props:
            _record_checkpoint_op("get", "not_found")
            raise CheckpointNotFoundError(f"no checkpoint found for id {checkpoint_id}")

        stored_tenant = props.get("tenant", "")
        if stored_tenant != requesting_tenant:
            _record_checkpoint_op("get", "cross_tenant_denied")
            logger.warning(
                "[CONCEPT:AU-KG.memory.kv-checkpoint-resource] REFUSED cross-tenant "
                "checkpoint load: checkpoint=%s owner_tenant=%s requester_tenant=%s",
                checkpoint_id,
                stored_tenant,
                requesting_tenant,
            )
            raise CrossTenantCheckpointError(
                f"checkpoint {checkpoint_id} belongs to tenant {stored_tenant!r}, "
                f"refusing load for {requesting_tenant!r}"
            )

        stored_policy_version = props.get("policy_version", "")
        if (
            current_policy_version is not None
            and stored_policy_version != current_policy_version
        ):
            _record_checkpoint_op("get", "stale")
            raise StaleCheckpointError(
                f"checkpoint {checkpoint_id} was minted under "
                f"policy_version={stored_policy_version!r}, current is "
                f"{current_policy_version!r} — refusing to load a stale checkpoint"
            )

        _record_checkpoint_op("get", "ok")
        return KVCheckpointRecord(
            checkpoint_id=checkpoint_id,
            key=KVCheckpointKey(
                model_identity=props.get("model_identity", ""),
                quantization=props.get("quantization", ""),
                serving_engine=props.get("serving_engine", ""),
                engine_version=props.get("engine_version", ""),
                prefix_digest=props.get("prefix_digest", ""),
                tenant=stored_tenant,
                policy_version=stored_policy_version,
            ),
            blob_id=props.get("blob_id", ""),
            digest=props.get("content_digest", ""),
            run_id=props.get("run_id", ""),
            point=props.get("point", ""),
            size_bytes=int(props.get("file_size_bytes", 0) or 0),
            created_at=props.get("created_at", ""),
            provenance=props.get("provenance") or {},
        )

    def fetch_bytes(self, record: KVCheckpointRecord) -> bytes:
        """Fetch a loaded checkpoint's heavy KV blob bytes by content digest.

        Always a separate call from :meth:`get_checkpoint` — the graph node (cheap,
        provenance-only) is loaded and validated FIRST; only a caller that actually
        needs the bytes pays the blob fetch, keeping "the graph is the semantic
        master, not a byte store" true in both directions.
        """
        return self._client.blob.fetch(record.digest)

    # -- operations ----------------------------------------------------------
    def instantiate_agent(
        self,
        checkpoint_id: str,
        *,
        requesting_tenant: str,
        new_run_id: str,
        current_policy_version: str | None = None,
    ) -> KVCheckpointRecord:
        """Load a checkpoint (fail-closed via :meth:`get_checkpoint`) and link a NEW
        agent run's provenance to it.

        Writes an ``initializedFrom`` edge from a fresh ``:AgentRun`` node
        (``agentrun:<new_run_id>``) to the checkpoint node, so the graph records
        which run was seeded from which checkpoint — queryable lineage. The lineage
        edge is best-effort (logged, not raised) — a load that already succeeded is
        not undone by a failure to link the edge; only the load itself is
        fail-closed.
        """
        record = self.get_checkpoint(
            checkpoint_id,
            requesting_tenant=requesting_tenant,
            current_policy_version=current_policy_version,
        )
        run_node_id = f"agentrun:{new_run_id}"
        try:
            client = self._client
            txn = client.txn.begin(graph=self._graph)
            if not client.nodes.has(run_node_id):
                client.txn.add_node(
                    txn,
                    run_node_id,
                    {
                        "type": "AgentRun",
                        "run_id": new_run_id,
                        "created_at": self._now(),
                    },
                )
            client.txn.commit(txn)
            client.edges.add(
                run_node_id, self._node_id(checkpoint_id), {"type": "initializedFrom"}
            )
        except Exception as e:  # noqa: BLE001 — lineage edge is best-effort
            logger.warning(
                "[CONCEPT:AU-KG.memory.kv-checkpoint-resource] initializedFrom lineage "
                "edge failed for run=%s checkpoint=%s: %s",
                new_run_id,
                checkpoint_id,
                e,
            )
        _record_checkpoint_op("instantiate_agent", "ok")
        return record

    def restore_conversation(
        self,
        checkpoint_id: str,
        *,
        conversation_id: str,
        requesting_tenant: str,
        current_policy_version: str | None = None,
        allow_cold_start: bool = False,
    ) -> RestoreResult:
        """Reset ``conversation_id``'s context window back to a prior checkpoint.

        FAIL-CLOSED by default: an invalid/foreign/stale checkpoint, or a blob-fetch
        failure, raises the underlying :class:`KVCheckpointError`. Pass
        ``allow_cold_start=True`` to instead catch that failure and return an
        EXPLICIT, TRACED cold-start :class:`RestoreResult` (``cold_start=True`` +
        ``fallback_reason``, plus a logged warning and a
        ``KVCACHE_CHECKPOINT_OPS{outcome="cold_start_fallback"}`` telemetry event)
        rather than raising — never a SILENT degrade to a cold start.
        """
        try:
            record = self.get_checkpoint(
                checkpoint_id,
                requesting_tenant=requesting_tenant,
                current_policy_version=current_policy_version,
            )
        except KVCheckpointError as exc:
            if not allow_cold_start:
                raise
            return self._traced_cold_start(conversation_id, checkpoint_id, exc)

        try:
            data = self.fetch_bytes(record)
        except Exception as exc:  # noqa: BLE001 — fetch failure is ALSO fail-closed
            if not allow_cold_start:
                raise KVCheckpointError(
                    f"checkpoint {checkpoint_id} blob fetch failed"
                ) from exc
            return self._traced_cold_start(conversation_id, checkpoint_id, exc)

        try:
            self._client.edges.add(
                f"conversation:{conversation_id}",
                self._node_id(checkpoint_id),
                {"type": "restoredFrom"},
            )
        except Exception as e:  # noqa: BLE001 — lineage edge is best-effort
            logger.debug(
                "[CONCEPT:AU-KG.memory.kv-checkpoint-resource] restoredFrom lineage "
                "edge failed: %s",
                e,
            )

        _record_checkpoint_op("restore_conversation", "ok")
        return RestoreResult(
            conversation_id=conversation_id,
            checkpoint_id=checkpoint_id,
            restored=True,
            record=record,
            data=data,
        )

    @staticmethod
    def _traced_cold_start(
        conversation_id: str, checkpoint_id: str, exc: Exception
    ) -> RestoreResult:
        """Build the EXPLICIT, TRACED cold-start :class:`RestoreResult` for
        ``restore_conversation``'s ``allow_cold_start=True`` path."""
        reason = type(exc).__name__
        logger.warning(
            "[CONCEPT:AU-KG.memory.kv-checkpoint-resource] restore_conversation "
            "cold-start fallback for conversation=%s checkpoint=%s: %s (%s)",
            conversation_id,
            checkpoint_id,
            exc,
            reason,
        )
        _record_checkpoint_op("restore_conversation", "cold_start_fallback")
        return RestoreResult(
            conversation_id=conversation_id,
            checkpoint_id=checkpoint_id,
            restored=False,
            cold_start=True,
            fallback_reason=reason,
        )
