#!/usr/bin/python
from __future__ import annotations

"""Queue-driven agent dispatch — the enqueue side.

CONCEPT:AU-ORCH.dispatch.queue-agent-dispatch — Queue-driven agent dispatch with session-keyed partitions
consumed by a stateless dispatch-worker fleet

The in-process asyncio scheduler (``core/cognitive_scheduler.py``) caps agent
concurrency at one process on one host: sessions are pinned to the host that
created them, and a busy gateway cannot hand work to an idle peer. This module
externalizes the *dispatch* of an agent turn (a goal-loop run or an
orchestrator job) onto the SAME durable task-queue stack the KG ingest plane
already scales on (CONCEPT:AU-KG.backend.selectable-queue-backend/2.56/2.57):

* the queue carries a small typed :class:`AgentTurnEnvelope` — REFERENCES only
  (``payload_ref`` points at the goal/WorkItem in the shared state store or
  graph); large bodies never ride the queue;
* the partition key is the **session id** (see
  :func:`~agent_utilities.knowledge_graph.core.kafka_queue_backend.partition_key_for`)
  so all turns of one session land on one partition and execute serially —
  per-session ordering is a turn-coherence REQUIREMENT, stronger than the
  ingest plane's per-tenant ordering;
* any host running an ``agent-dispatch-worker``
  (:mod:`agent_utilities.orchestration.agent_dispatch_worker`) claims turns,
  rehydrates session state from the OS-5.16 shared state store, executes
  through the EXISTING goal/agent execution paths, and writes results back
  durably — the scheduler tier becomes horizontally scalable and sessions are
  no longer pinned to their birth host.

Placement is deliberately **queue-pull** (workers claim work when they have
capacity) rather than a central placer pushing turns at workers: with
session-keyed partitions the broker/queue already provides per-session
serialization and uniform load spreading, so a placer would add a coordination
point and a failure mode without adding correctness. Affinity-aware placement
(HRW on warm caches) is future work layered on the same envelope.

Agent dispatch is always queue-backed: dispatch returns a job handle and a
worker fleet executes it. The queue *transport* reuses the KG-2.55 resolution
(``TASK_QUEUE_BACKEND``/auto): Kafka topic ``agent_turns``, Postgres SKIP
LOCKED table ``agent_dispatch_queue``, or the zero-infra per-host SQLite file.
"""

import hashlib
import hmac
import json
import logging
import math
import secrets
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

#: Kafka topic / consumer group for dispatched agent turns. Postgres and SQLite
#: transports use an equivalently-named table / db file instead of a topic.
AGENT_TURNS_TOPIC = "agent_turns"
DISPATCH_GROUP = "agent-dispatch"

#: Envelope kinds the dispatch workers know how to execute.
KIND_GOAL_LOOP = "goal_loop"
KIND_ORCHESTRATOR_TASK = "orchestrator_task"

# The broker carrier is deliberately a small, versioned, independently
# verifiable envelope.  It is not a replacement for the native WorkItem
# authority: the consumer verifies this transport proof first, then re-reads
# the admitted WorkItem and claims/fences it through the engine.
DISPATCH_CARRIER_VERSION = 1
DISPATCH_CARRIER_TTL_S = 300.0
DISPATCH_CARRIER_MAX_CLOCK_SKEW = 30.0
DISPATCH_CARRIER_MAX_FIELD_BYTES = 256


class DispatchCarrierError(ValueError):
    """A dispatch message has no current, authentic session carrier."""


class DispatchCarrierSecretUnavailable(DispatchCarrierError):
    """No deployment-wide key is available to authenticate broker messages."""


_dispatch_ephemeral_secret: bytes | None = None
_dispatch_ephemeral_secret_lock = threading.Lock()


def _dispatch_carrier_secret(secret: str | bytes | None = None) -> bytes:
    """Resolve the one deployment-wide HMAC authority for dispatch carriers.

    ``AGENT_UTILITIES_TOKEN_SECRET`` is the explicit cross-process authority;
    ``GRAPH_SERVICE_AUTH_SECRET`` is accepted as the already-shared service
    identity key for deployments that intentionally use one.  A development
    fallback reuses the run-token secret's process-local random key so the
    zero-infrastructure single-process profile remains usable.  Production
    profiles fail closed when neither configured secret exists: a process-local
    key can never authenticate a carrier after a replica migration.
    """
    if isinstance(secret, bytes):
        key = secret
    elif secret is not None:
        key = str(secret).encode("utf-8")
    else:
        from agent_utilities.core.config import setting

        configured = str(
            setting("AGENT_UTILITIES_TOKEN_SECRET", "")
            or setting("GRAPH_SERVICE_AUTH_SECRET", "")
            or ""
        ).strip()
        key = configured.encode("utf-8") if configured else b""
    if key:
        return key

    from agent_utilities.core.profile_guard import is_production_profile

    if is_production_profile():
        raise DispatchCarrierSecretUnavailable(
            "dispatch carrier authentication requires AGENT_UTILITIES_TOKEN_SECRET "
            "or GRAPH_SERVICE_AUTH_SECRET in production"
        )

    # Match run-token's zero-config development posture without introducing a
    # second deterministic or guessable key.  This path is intentionally not
    # accepted by production profiles and therefore cannot be used for a
    # cross-replica deployment by accident.
    global _dispatch_ephemeral_secret
    with _dispatch_ephemeral_secret_lock:
        if _dispatch_ephemeral_secret is None:
            _dispatch_ephemeral_secret = secrets.token_bytes(32)
        return _dispatch_ephemeral_secret


def _carrier_text(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    rendered = str(value or "")
    encoded = rendered.encode("utf-8")
    if not rendered and allow_empty:
        return rendered
    if (
        not rendered
        or len(encoded) > DISPATCH_CARRIER_MAX_FIELD_BYTES
        or "\x00" in rendered
    ):
        raise DispatchCarrierError(f"dispatch carrier {field_name} is invalid")
    return rendered


class DispatchCarrier(BaseModel):
    """Signed broker proof binding one turn to tenant/session/job and time.

    The signature covers every field except itself.  A duplicate delivery may
    therefore carry the same proof, but it still cannot change identity or
    target; native WorkItem idempotency/fencing remains the authority that
    turns that duplicate into a skip.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(default=DISPATCH_CARRIER_VERSION)
    tenant: str
    session_id: str
    job_id: str
    kind: str
    payload_ref: str
    issued_at: float
    expires_at: float
    nonce: str
    signature: str
    # Execution selectors are signed too; otherwise a broker tamper could
    # keep the same job/session proof while swapping the selected agent or
    # removing the dispatch deadline.
    agent_name: str = ""
    deadline_unix: float | None = None

    @field_validator("version")
    @classmethod
    def _version(cls, value: int) -> int:
        if value != DISPATCH_CARRIER_VERSION:
            raise DispatchCarrierError(
                f"unsupported dispatch carrier version {value!r}"
            )
        return value

    @field_validator(
        "tenant",
        "session_id",
        "job_id",
        "kind",
        "payload_ref",
        "nonce",
        "signature",
        "agent_name",
    )
    @classmethod
    def _bounded_text(cls, value: str, info: Any) -> str:
        return _carrier_text(
            value,
            info.field_name,
            allow_empty=info.field_name
            in {"tenant", "kind", "payload_ref", "agent_name"},
        )

    @staticmethod
    def _canonical_payload(
        *,
        version: int,
        tenant: str,
        session_id: str,
        job_id: str,
        kind: str,
        payload_ref: str,
        agent_name: str,
        deadline_unix: float | None,
        issued_at: float,
        expires_at: float,
        nonce: str,
    ) -> bytes:
        return json.dumps(
            {
                "expires_at": expires_at,
                "deadline_unix": deadline_unix,
                "issued_at": issued_at,
                "agent_name": agent_name,
                "job_id": job_id,
                "kind": kind,
                "nonce": nonce,
                "payload_ref": payload_ref,
                "session_id": session_id,
                "tenant": tenant,
                "version": version,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @classmethod
    def mint(
        cls,
        *,
        tenant: str,
        session_id: str,
        job_id: str,
        kind: str = "",
        payload_ref: str = "",
        agent_name: str = "",
        deadline_unix: float | None = None,
        ttl_seconds: float = DISPATCH_CARRIER_TTL_S,
        now: float | None = None,
        nonce: str | None = None,
        secret: str | bytes | None = None,
        allow_empty_tenant: bool = False,
    ) -> DispatchCarrier:
        tenant = _carrier_text(tenant, "tenant", allow_empty=allow_empty_tenant)
        session_id = _carrier_text(session_id, "session_id")
        job_id = _carrier_text(job_id, "job_id")
        kind = _carrier_text(kind, "kind", allow_empty=True)
        payload_ref = _carrier_text(payload_ref, "payload_ref", allow_empty=True)
        agent_name = _carrier_text(agent_name, "agent_name", allow_empty=True)
        issued_at = float(time.time() if now is None else now)
        ttl = float(ttl_seconds)
        if (
            not math.isfinite(issued_at)
            or not math.isfinite(ttl)
            or (deadline_unix is not None and not math.isfinite(float(deadline_unix)))
        ):
            raise DispatchCarrierError("dispatch carrier timestamps must be finite")
        if ttl <= 0 or ttl > DISPATCH_CARRIER_TTL_S:
            raise DispatchCarrierError("dispatch carrier TTL is outside its bound")
        expires_at = issued_at + ttl
        carrier_nonce = _carrier_text(nonce or secrets.token_urlsafe(18), "nonce")
        payload = cls._canonical_payload(
            version=DISPATCH_CARRIER_VERSION,
            tenant=tenant,
            session_id=session_id,
            job_id=job_id,
            kind=kind,
            payload_ref=payload_ref,
            agent_name=agent_name,
            deadline_unix=(float(deadline_unix) if deadline_unix is not None else None),
            issued_at=issued_at,
            expires_at=expires_at,
            nonce=carrier_nonce,
        )
        signature = hmac.new(
            _dispatch_carrier_secret(secret), payload, hashlib.sha256
        ).hexdigest()
        return cls(
            tenant=tenant,
            session_id=session_id,
            job_id=job_id,
            kind=kind,
            payload_ref=payload_ref,
            agent_name=agent_name,
            deadline_unix=(float(deadline_unix) if deadline_unix is not None else None),
            issued_at=issued_at,
            expires_at=expires_at,
            nonce=carrier_nonce,
            signature=signature,
        )

    def verify(
        self,
        *,
        tenant: str,
        session_id: str,
        job_id: str,
        kind: str = "",
        payload_ref: str = "",
        agent_name: str = "",
        deadline_unix: float | None = None,
        now: float | None = None,
        secret: str | bytes | None = None,
        require_tenant: bool = True,
    ) -> DispatchCarrier:
        expected_tenant = _carrier_text(
            tenant, "tenant", allow_empty=not require_tenant
        )
        expected_session = _carrier_text(session_id, "session_id")
        expected_job = _carrier_text(job_id, "job_id")
        expected_kind = _carrier_text(kind, "kind", allow_empty=True)
        expected_payload_ref = _carrier_text(
            payload_ref, "payload_ref", allow_empty=True
        )
        expected_agent_name = _carrier_text(agent_name, "agent_name", allow_empty=True)
        if deadline_unix is not None and not math.isfinite(float(deadline_unix)):
            raise DispatchCarrierError("dispatch deadline must be finite")
        if self.version != DISPATCH_CARRIER_VERSION:
            raise DispatchCarrierError("unsupported dispatch carrier version")
        if require_tenant and not self.tenant:
            raise DispatchCarrierError("dispatch carrier tenant is required")
        if (
            self.tenant,
            self.session_id,
            self.job_id,
            self.kind,
            self.payload_ref,
            self.agent_name,
            self.deadline_unix,
        ) != (
            expected_tenant,
            expected_session,
            expected_job,
            expected_kind,
            expected_payload_ref,
            expected_agent_name,
            float(deadline_unix) if deadline_unix is not None else None,
        ):
            raise DispatchCarrierError("dispatch carrier identity binding mismatch")
        if not all(math.isfinite(value) for value in (self.issued_at, self.expires_at)):
            raise DispatchCarrierError("dispatch carrier timestamps must be finite")
        if self.expires_at <= self.issued_at:
            raise DispatchCarrierError("dispatch carrier expiry must follow issuance")
        if self.expires_at - self.issued_at > DISPATCH_CARRIER_TTL_S:
            raise DispatchCarrierError("dispatch carrier lifetime exceeds its bound")
        moment = float(time.time() if now is None else now)
        if not math.isfinite(moment):
            raise DispatchCarrierError("dispatch carrier verification time is invalid")
        if self.issued_at > moment + DISPATCH_CARRIER_MAX_CLOCK_SKEW:
            raise DispatchCarrierError("dispatch carrier was issued in the future")
        if moment >= self.expires_at:
            raise DispatchCarrierError("dispatch carrier has expired")
        payload = self._canonical_payload(
            version=self.version,
            tenant=self.tenant,
            session_id=self.session_id,
            job_id=self.job_id,
            kind=self.kind,
            payload_ref=self.payload_ref,
            agent_name=self.agent_name,
            deadline_unix=self.deadline_unix,
            issued_at=self.issued_at,
            expires_at=self.expires_at,
            nonce=self.nonce,
        )
        expected_signature = hmac.new(
            _dispatch_carrier_secret(secret), payload, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(self.signature, expected_signature):
            raise DispatchCarrierError("dispatch carrier signature is invalid")
        return self


class DispatchQueueFull(RuntimeError):
    """The bounded dispatch queue rejected a new turn."""


class AgentTurnEnvelope(BaseModel):
    """One dispatched agent turn on the ``agent_turns`` queue.

    CONCEPT:AU-ORCH.dispatch.queue-agent-dispatch — the queue carries references, not bodies: the durable
    record (the ``goals``/``sessions`` rows for a goal run, the WorkItem for an
    orchestrator job) is the payload's source of truth, addressed
    by ``payload_ref``. ``job_id`` doubles as the idempotency key — redelivery
    of an already-claimed/finished job is skipped by the worker's claim check
    (at-least-once delivery, idempotent claims).
    """

    #: Full-width 128-bit id (AU-P0-3): the prior ``.hex[:8]`` truncation kept
    #: only 32 bits of entropy — at ~77k dispatched jobs the birthday bound
    #: crosses 50% collision probability, and a collided ``job_id`` is silently
    #: treated as the SAME idempotency key (a redelivery skip) by the queue's
    #: claim check, dropping a distinct turn. ``uuid.uuid4().hex`` is the full
    #: 128-bit value; no truncation. A future upgrade to a time-ordered
    #: UUIDv7/ULID would additionally make ``job_id`` sort chronologically,
    #: but neither ships in this repo's dependencies yet (stdlib ``uuid7`` only
    #: lands in Python 3.14, and this package supports >=3.11) — tracked as
    #: follow-up, not blocking this fix.
    job_id: str = Field(default_factory=lambda: f"dispatch-{uuid.uuid4().hex}")
    session_id: str
    kind: str = KIND_GOAL_LOOP
    payload_ref: str = ""
    agent_name: str = ""
    tenant: str = ""
    #: Claim priority as the ONE discrete integer bucket (0=critical ..
    #: 3=background), identical to the WorkItem ``prio_bucket``
    #: (CONCEPT:AU-KG.ingest.hardened-priority-scheduled-task).
    prio_bucket: int = 2
    deadline_unix: float | None = None
    attempt: int = 0
    enqueued_at: float = Field(default_factory=time.time)
    #: Signed transport proof.  It is added at the explicit enqueue boundary;
    #: direct model construction remains useful for local inspection/fixtures,
    #: but the consumer refuses a delivery whose proof is absent or stale.
    carrier: DispatchCarrier | None = None

    @field_validator("prio_bucket", mode="before")
    @classmethod
    def _coerce_prio(cls, v: Any) -> int:
        """Validate the shared integer priority bucket.

        Lazy-imports ``_coerce_prio_bucket`` (the single normalizer) to avoid a
        construction-time import cycle with the engine module — the same pattern
        ``state_tools`` / ``schedule_engine`` / ``bus`` use to reach it.
        """
        from agent_utilities.knowledge_graph.core.engine_tasks import (
            _coerce_prio_bucket,
        )

        return _coerce_prio_bucket(v)

    def to_item(self) -> dict[str, Any]:
        """Serialize for the queue, including the authenticated carrier.

        ``session_id`` stays top-level so ``partition_key_for`` keys the
        message without decoding metadata.  Direct local callers that have not
        gone through :func:`enqueue_agent_turn` get a development-safe carrier
        for round-trip inspection; the consumer still requires a non-empty
        tenant and verifies it against the durable WorkItem.
        """
        if self.carrier is None:
            self.carrier = DispatchCarrier.mint(
                tenant=self.tenant,
                session_id=self.session_id,
                job_id=self.job_id,
                kind=self.kind,
                payload_ref=self.payload_ref,
                agent_name=self.agent_name,
                deadline_unix=self.deadline_unix,
                now=self.enqueued_at,
                allow_empty_tenant=True,
            )
        return self.model_dump()

    def ensure_authenticated_carrier(
        self,
        *,
        now: float | None = None,
        secret: str | bytes | None = None,
    ) -> DispatchCarrier:
        """Mint/verify the broker proof before any WorkItem is submitted."""
        if not self.tenant:
            raise DispatchCarrierError(
                "dispatch enqueue requires a non-empty authenticated tenant"
            )
        if self.carrier is None:
            self.carrier = DispatchCarrier.mint(
                tenant=self.tenant,
                session_id=self.session_id,
                job_id=self.job_id,
                kind=self.kind,
                payload_ref=self.payload_ref,
                agent_name=self.agent_name,
                deadline_unix=self.deadline_unix,
                now=now,
                secret=secret,
            )
        return self.carrier.verify(
            tenant=self.tenant,
            session_id=self.session_id,
            job_id=self.job_id,
            kind=self.kind,
            payload_ref=self.payload_ref,
            agent_name=self.agent_name,
            deadline_unix=self.deadline_unix,
            now=now,
            secret=secret,
        )

    def authenticate_carrier(
        self,
        *,
        now: float | None = None,
        secret: str | bytes | None = None,
    ) -> DispatchCarrier:
        """Verify a delivered proof without changing the envelope."""
        if self.carrier is None:
            raise DispatchCarrierError("dispatch delivery has no carrier")
        if not self.tenant:
            raise DispatchCarrierError("dispatch delivery has no tenant")
        return self.carrier.verify(
            tenant=self.tenant,
            session_id=self.session_id,
            job_id=self.job_id,
            kind=self.kind,
            payload_ref=self.payload_ref,
            agent_name=self.agent_name,
            deadline_unix=self.deadline_unix,
            now=now,
            secret=secret,
        )

    @classmethod
    def from_item(cls, item: dict[str, Any]) -> AgentTurnEnvelope:
        return cls.model_validate(item)


# ── queue construction ─────────────────────────────────────────────────────

_queue_lock = threading.Lock()
_queue: Any = None


def create_dispatch_queue(config: Any = None) -> Any:
    """Build the ``agent_turns`` queue on the KG-2.55-selected transport.

    Composes the existing task-queue stack rather than introducing a second
    queue technology: the SAME ``TASK_QUEUE_BACKEND``/auto resolution picks
    kafka (keyed ``agent_turns`` topic), postgres (SKIP LOCKED claims on the
    ``agent_dispatch_queue`` table of the shared state store), or the per-host
    SQLite file. A configured kafka/postgres transport that is unreachable raises
    :class:`~agent_utilities.knowledge_graph.core.queue_backend.TaskQueueUnavailable`.
    """
    from agent_utilities.knowledge_graph.core.queue_backend import (
        resolve_task_queue_backend,
    )

    if config is None:
        from agent_utilities.core.config import config as _cfg

        config = _cfg

    choice = resolve_task_queue_backend(config)
    sqlite_db_path = _sqlite_queue_path()

    if choice == "kafka":
        from agent_utilities.knowledge_graph.core.kafka_queue_backend import (
            KafkaQueueBackend,
        )

        return KafkaQueueBackend(
            bootstrap_servers=getattr(config, "kafka_bootstrap_servers", None),
            partitions=int(getattr(config, "agent_turns_partitions", 6) or 6),
            tasks_topic=AGENT_TURNS_TOPIC,
            consumer_group=DISPATCH_GROUP,
        )

    if choice == "postgres":
        from agent_utilities.knowledge_graph.core.queue_backend import (
            TaskQueueUnavailable,
        )

        try:
            from agent_utilities.knowledge_graph.core.postgres_queue_backend import (
                PostgresTaskQueue,
            )

            return PostgresTaskQueue(queue_table="agent_dispatch_queue")
        except Exception as e:  # noqa: BLE001 - fail closed at the authority boundary
            raise TaskQueueUnavailable(
                "configured Postgres dispatch authority is unavailable "
                f"(error_type={type(e).__name__})"
            ) from None

    from agent_utilities.knowledge_graph.core.engine_tasks import SQLiteTaskQueue

    return SQLiteTaskQueue(sqlite_db_path)


def _sqlite_queue_path() -> str:
    from agent_utilities.core.paths import data_dir

    return str(data_dir() / "agent_dispatch_queue.db")


def get_dispatch_queue(config: Any = None) -> Any:
    """Process-wide cached dispatch queue (lazily constructed)."""
    global _queue
    with _queue_lock:
        if _queue is None:
            _queue = create_dispatch_queue(config)
        return _queue


def reset_dispatch_queue_for_tests(queue: Any = None) -> None:
    """Swap/clear the cached dispatch queue (test isolation seam)."""
    global _queue
    with _queue_lock:
        _queue = queue


def dispatch_queue_depth(queue: Any = None) -> int:
    """Return authoritative ``agent_turns`` depth.

    Probe failures propagate. Treating an unavailable depth authority as zero
    would disable admission control precisely when the queue is unhealthy.
    """
    q = queue if queue is not None else get_dispatch_queue()
    depth = int(q.get_queue_size())
    if depth < 0:
        raise RuntimeError("dispatch queue returned a negative depth")
    return depth


# ── enqueue ────────────────────────────────────────────────────────────────


def enqueue_agent_turn(
    envelope: AgentTurnEnvelope, queue: Any = None, engine: Any = None
) -> dict[str, Any]:
    """Publish one agent turn and return its job handle.

    The handle is what queue-mode dispatch returns to the caller instead of
    executing in-process: poll ``graph_jobs action=status`` or the collapsed
    ``/api/graph/jobs`` REST surface (orchestrator jobs), or
    the goals API (goal runs) for progress and the executing worker/host.

    ``engine``, when the caller already holds a resolved engine (e.g.
    ``graph_jobs``'s ``dispatch`` action, which resolves one via
    ``kg_server._get_engine()`` to build its ``Orchestrator`` two lines
    above), is reused as-is instead of independently deriving a second one
    through ``IntelligenceGraphEngine.get_or_create()``. Callers with no
    engine already in hand (``None``, the default) get the same
    process-singleton lookup as before — this is additive, not a behavior
    change for them.

    D-03/GOC-39: a bare ``get_or_create()`` here constructed its own engine
    independent of whatever the caller already resolved, which is merely
    redundant in production (both resolve the same process singleton) but a
    real bug under test doubles — a caller-injected fake engine (e.g.
    ``tests/unit/test_agent_dispatch.py``'s ``orchestrate_tool`` fixture,
    which patches ``kg_server._get_engine`` for the ``Orchestrator``/WorkItem
    calls either side of this one) was silently bypassed here, so this call
    alone reached for the real process-wide engine and raced its
    connection/lifecycle against the test's own isolated one.
    """
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.core.session import resolve_session
    from agent_utilities.orchestration.work_item import (
        cancel_work_item,
        submit_work_item,
    )

    session = resolve_session(required_scope="kg:write")
    tenant = envelope.tenant or session.tenant
    if envelope.tenant and envelope.tenant != session.tenant:
        raise PermissionError("AgentTurnEnvelope tenant differs from GraphSession")
    envelope.tenant = tenant
    # Authenticate the broker carrier BEFORE the durable WorkItem admission.
    # A missing deployment-wide key or a stale/tampered caller-provided carrier
    # must not leave an orphan WorkItem that no worker can safely consume.
    envelope.ensure_authenticated_carrier()
    q = queue if queue is not None else get_dispatch_queue()
    from agent_utilities.core.config import config

    max_depth = int(config.agent_dispatch_max_depth)
    if dispatch_queue_depth(q) >= max_depth:
        raise DispatchQueueFull("agent dispatch queue is at its admission bound")

    engine = engine if engine is not None else IntelligenceGraphEngine.get_or_create()
    work_item_id = f"workitem:dispatch:{envelope.job_id}"
    submit_work_item(
        engine,
        kind="agent_turn",
        payload_ref=envelope.payload_ref,
        tenant=tenant,
        priority=envelope.prio_bucket,
        deadline_unix=envelope.deadline_unix,
        resource_class="agent_dispatch",
        fairness_group=envelope.session_id,
        work_item_id=work_item_id,
        idempotency_key=envelope.job_id,
    )
    try:
        accepted = q.put_if_below(envelope.to_item(), max_depth)
    except Exception:
        if not cancel_work_item(
            engine, work_item_id, reason="dispatch_queue_admission_failed"
        ):
            raise RuntimeError(
                "dispatch queue admission failed and WorkItem rollback was rejected"
            ) from None
        raise
    if not accepted:
        if not cancel_work_item(
            engine, work_item_id, reason="dispatch_queue_capacity_rejected"
        ):
            raise RuntimeError(
                "dispatch queue capacity rejection could not cancel its WorkItem"
            )
        raise DispatchQueueFull("agent dispatch queue is at its admission bound")
    from agent_utilities.messaging.bus_privacy import bus_reference

    logger.info(
        "Agent turn enqueued: job_ref=%s session_ref=%s kind=%s",
        bus_reference("dispatch_job", envelope.job_id, tenant=tenant),
        bus_reference("session", envelope.session_id, tenant=tenant),
        envelope.kind,
    )
    return {
        "job_id": envelope.job_id,
        "work_item_id": work_item_id,
        "session_id": envelope.session_id,
        "kind": envelope.kind,
        "dispatch": "queued",
        "status": "pending",
    }


# ── per-session mutual exclusion ───────────────────────────────────────────

# A session lock is a short-lived local coordination aid, never a durable
# lease.  The old dict retained one ``threading.Lock`` forever for every
# session ever observed, which made high-cardinality session churn a process
# memory leak.  Entries are reference counted and removed as soon as their
# final holder leaves; the hard cap protects the simultaneous-session case.
MAX_SESSION_LOCK_ENTRIES = 4096
MAX_SESSION_ID_BYTES = 256


class SessionLockCapacityError(RuntimeError):
    """The process-local session-lock registry reached its bounded capacity."""


@dataclass
class _SessionLockEntry:
    lock: threading.Lock = field(default_factory=threading.Lock)
    references: int = 0


_session_locks_lock = threading.Lock()
_session_locks: dict[str, _SessionLockEntry] = {}


def _validate_session_id(session_id: str) -> str:
    rendered = str(session_id or "")
    if (
        not rendered
        or len(rendered.encode("utf-8")) > MAX_SESSION_ID_BYTES
        or "\x00" in rendered
    ):
        raise ValueError("dispatch session id is invalid or exceeds its bound")
    return rendered


class _SessionLockHandle:
    """Ref-counted handle that removes its registry entry on final release."""

    def __init__(self, session_id: str) -> None:
        self.session_id = _validate_session_id(session_id)
        self._entry: _SessionLockEntry | None = None

    def __enter__(self) -> None:
        with _session_locks_lock:
            entry = _session_locks.get(self.session_id)
            if entry is None:
                if len(_session_locks) >= MAX_SESSION_LOCK_ENTRIES:
                    raise SessionLockCapacityError(
                        "dispatch session-lock registry is at its bounded capacity"
                    )
                entry = _SessionLockEntry()
                _session_locks[self.session_id] = entry
            entry.references += 1
            self._entry = entry
        try:
            entry.lock.acquire()
        except BaseException:
            self._release_reference(entry)
            raise
        return None

    def _release_reference(self, entry: _SessionLockEntry) -> None:
        with _session_locks_lock:
            entry.references -= 1
            if entry.references <= 0 and _session_locks.get(self.session_id) is entry:
                _session_locks.pop(self.session_id, None)

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        entry = self._entry
        if entry is None:
            return
        try:
            entry.lock.release()
        finally:
            self._release_reference(entry)
            self._entry = None


def _session_lock(session_id: str) -> _SessionLockHandle:
    """Return a bounded, lifecycle-managed per-session lock handle."""
    return _SessionLockHandle(session_id)


def session_lock_registry_size() -> int:
    """Return the current local lock-entry cardinality for health/TCKs."""
    with _session_locks_lock:
        return len(_session_locks)


@contextmanager
def session_execution_guard(session_id: str) -> Iterator[None]:
    """One executing worker per session at a time (CONCEPT:AU-ORCH.dispatch.queue-agent-dispatch).

    At-least-once delivery means two workers can briefly hold the same
    session's turns (e.g. a redelivery racing the original consumer). Turn
    coherence requires per-session mutual exclusion, layered:

    * a process-local per-session lock serializes worker threads in ONE
      process (covers the SQLite/single-host transports);
    * ``state_claim_guard`` extends the critical section fleet-wide via a
      Postgres advisory lock when durable state is externalized
      (CONCEPT:AU-OS.state.unified-durable-state-externalization) — two hosts can never execute one session at once.

    A crashed holder releases both automatically (process death drops the
    advisory lock server-side), so crash recovery is redelivery + re-claim,
    never a stuck session.  The process-local handle is reference-counted and
    removed after its final holder, with a hard simultaneous-entry bound so
    high-cardinality session churn cannot grow this registry without limit.
    """
    from agent_utilities.core.state_store import state_claim_guard

    with _session_lock(session_id), state_claim_guard(f"agent-session:{session_id}"):
        yield


# ── fleet-visible worker registry ──────────────────────────────────────────

#: A worker whose last heartbeat is older than this is presumed gone and is
#: excluded from topology/metrics (its in-flight claim recovers separately via
#: the stale-claim re-claim path).
WORKER_HEARTBEAT_TTL_S = 90.0


def record_dispatch_worker_heartbeat(
    worker_id: str,
    *,
    host: str = "",
    capacity: int = 1,
    active_sessions: list[str] | tuple[str, ...] = (),
    queue_backend: str = "",
) -> None:
    """Upsert this worker's liveness row in the fleet registry.

    CONCEPT:AU-ORCH.dispatch.queue-agent-dispatch — the registry lives in the SAME sessions store the
    OS-5.18 supervisory plane already reads (per-host SQLite, or the shared
    Postgres under ``STATE_DB_URI`` — where every gateway sees every host's
    workers). ``/api/fleet/topology`` surfaces these rows.
    """
    import json as _json

    from agent_utilities.core import sessions as _sessions
    from agent_utilities.messaging.bus_privacy import bus_reference

    now = time.time()
    conn = _sessions._connect_db()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO dispatch_workers
                (worker_id, host, capacity, active_sessions, queue_backend,
                 started_at, last_heartbeat)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(worker_id) DO UPDATE SET
                host = excluded.host,
                capacity = excluded.capacity,
                active_sessions = excluded.active_sessions,
                queue_backend = excluded.queue_backend,
                last_heartbeat = excluded.last_heartbeat
            """,
            (
                worker_id,
                bus_reference("dispatch_host", host or worker_id),
                int(capacity),
                _json.dumps(list(active_sessions)),
                queue_backend,
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def list_dispatch_workers(
    ttl_s: float = WORKER_HEARTBEAT_TTL_S,
) -> list[dict[str, Any]]:
    """Live dispatch workers (heartbeat within ``ttl_s``), newest first."""
    import json as _json

    from agent_utilities.core import sessions as _sessions

    cutoff = time.time() - ttl_s
    conn = _sessions._connect_db()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT worker_id, host, capacity, active_sessions, queue_backend, "
            "started_at, last_heartbeat FROM dispatch_workers "
            "WHERE last_heartbeat >= ? ORDER BY last_heartbeat DESC",
            (cutoff,),
        )
        workers: list[dict[str, Any]] = []
        for row in cursor.fetchall():
            entry = dict(row)
            try:
                entry["active_sessions"] = _json.loads(
                    entry.get("active_sessions") or "[]"
                )
            except (TypeError, ValueError):
                entry["active_sessions"] = []
            workers.append(entry)
        return workers
    finally:
        conn.close()
