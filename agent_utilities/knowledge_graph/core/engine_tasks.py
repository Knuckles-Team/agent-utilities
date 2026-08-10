import asyncio
import base64
import json
import logging
import re
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, cast

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)
_DATABASE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_GRAPH_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


def _require_verified_background_session(session: Any) -> Any:
    """Reject non-session or incomplete authority before a thread is created."""
    from agent_utilities.knowledge_graph.core.session import (
        GraphSession,
        SessionRequiredError,
    )

    if not isinstance(session, GraphSession):
        raise SessionRequiredError(
            "Background graph work requires a verified GraphSession"
        )
    session.engine_verified_context()
    if not session.actor.authenticated:
        raise SessionRequiredError(
            "Background graph work requires an authenticated actor"
        )
    return session


def _capture_verified_background_session() -> Any:
    """Capture the already-verified authority a daemon thread must retain.

    ``ContextVar`` values do not cross :class:`threading.Thread` boundaries.
    Background graph work therefore captures the immutable process session
    before spawning and binds both that session and its actor inside the new
    thread.  Absence or disagreement is an authorization failure, never a
    reason to synthesize a system identity.
    """
    from agent_utilities.knowledge_graph.core.session import (
        GraphSession,
        SessionRequiredError,
    )
    from agent_utilities.security.brain_context import (
        IdentityRequiredError,
        current_actor,
    )

    session = _require_verified_background_session(GraphSession.from_ambient())
    try:
        actor = current_actor()
    except IdentityRequiredError:
        raise SessionRequiredError(
            "Background graph work requires a verified actor"
        ) from None
    if actor != session.actor or not actor.authenticated:
        raise SessionRequiredError(
            "Background graph actor and session authority must match"
        )
    return session


def _run_with_background_authority(
    session: Any, target: Any, *args: Any, **kwargs: Any
) -> Any:
    """Run one thread entrypoint under its captured graph authority."""
    from agent_utilities.knowledge_graph.core.session import use_session
    from agent_utilities.security.brain_context import use_actor

    session = _require_verified_background_session(session)
    with use_actor(session.actor), use_session(session):
        return target(*args, **kwargs)


def _authorized_background_thread(
    session: Any,
    target: Any,
    *,
    name: str,
    args: tuple[Any, ...] = (),
) -> threading.Thread:
    """Build a daemon thread whose entrypoint explicitly restores authority."""
    session = _require_verified_background_session(session)
    return threading.Thread(
        target=_run_with_background_authority,
        args=(session, target, *args),
        daemon=True,
        name=name,
    )


def _require_database_identifier(value: object) -> str:
    rendered = str(value or "")
    if not _DATABASE_IDENTIFIER.fullmatch(rendered):
        raise ValueError("Database identifier is not permitted")
    return rendered


def _safe_graph_identifier(value: object, *, default: str = "") -> str:
    rendered = re.sub(r"\W+", "_", str(value or default)).strip("_")
    return rendered if _GRAPH_IDENTIFIER.fullmatch(rendered) else default


def daemon_role() -> str:
    """Resolve this process's KG background-daemon role (CONCEPT:AU-KG.coordination.embedder-breaker / OS-5.0).

    The KG runs ONE consolidated background daemon (queue drain + graph writer +
    task workers + maintenance scheduler + file-watch poll). This selects who
    runs it:

    * ``host``   — run the full UnifiedDaemon (the API gateway sets this).
    * ``client`` — run NOTHING; submit work to the durable queue that the host
      daemon drains (MCP server / CLI / one-shot scripts set this).
    * ``auto``   — default: run the consolidated daemon in-process for
      single-process development usage.

    ``KG_DAEMON_ROLE`` overrides (default ``auto``). Note: test mode and
    ``--stage-to-queue`` independently suppress *auto-start* of the daemon in
    ``__init__`` without changing the role, so explicit ``start_task_workers()``
    calls in tests still work.
    """
    from agent_utilities.core.config import setting

    role = (setting("KG_DAEMON_ROLE", "auto") or "auto").strip().lower()
    return role if role in {"host", "client", "auto"} else "auto"


# Supported file extensions for document ingestion (LlamaIndex SimpleDirectoryReader)
SUPPORTED_EXTENSIONS: set[str] = {
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".txt",
    ".md",
    ".csv",
    ".epub",
    ".json",
    ".jsonl",
    ".html",
    ".htm",
    ".xml",
    ".yaml",
    ".yml",
    ".rst",
    ".rtf",
    ".ipynb",
}


class _BoundedPypdfReader:
    """LlamaIndex file-reader adapter for the governed pypdf path."""

    def load_data(
        self,
        file: str | Path | None = None,
        *,
        file_path: str | Path | None = None,
        **_: Any,
    ) -> list[Any]:
        from llama_index.core import Document

        from ..extraction.pdf import read_pdf_text

        source = file if file is not None else file_path
        if source is None:
            return []
        text = read_pdf_text(source)
        return [Document(text=text)] if text else []


def _pdf_file_extractor() -> dict[str, Any]:
    """Map ``.pdf`` to the one bounded, license-approved pypdf reader."""

    return {".pdf": _BoundedPypdfReader()}


def _encode_metadata(data: dict[str, Any]) -> str:
    """Encode metadata dict as base64 JSON for safe Cypher storage."""
    return base64.b64encode(json.dumps(data).encode()).decode()


def _decode_metadata(raw: str | None) -> dict[str, Any]:
    """Robustly decode metadata from any stored format.

    Handles:
        1. Valid JSON strings
        2. Base64-encoded JSON
        3. Malformed key-value strings (e.g. ``{error: some msg, key: val}``)
        4. None / empty → returns ``{}``
    """
    if not raw:
        return {}

    # Attempt 1: Direct JSON parse
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass  # nosec B110

    # Attempt 2: Base64-encoded JSON
    try:
        decoded = base64.b64decode(raw).decode()
        result = json.loads(decoded)
        if isinstance(result, dict):
            return result
    except Exception:
        pass  # nosec B110

    # Attempt 3: Regex fallback for malformed key-value strings
    # Handles patterns like: {error: some message, target: /path/to/file}
    try:
        stripped = raw.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            inner = stripped[1:-1]
            pairs = {}
            # Split on ", " that precedes a key pattern (word followed by colon)
            parts = re.split(r",\s*(?=\w+:)", inner)
            for part in parts:
                match = re.match(r"(\w+):\s*(.+)", part.strip())
                if match:
                    pairs[match.group(1)] = match.group(2).strip()
            if pairs:
                return pairs
    except Exception:
        pass  # nosec B110

    logger.warning("Failed to decode task metadata: %.100s...", raw)
    return {"_raw": raw}


_WORKSPACE_TARGET_PREFIX = "workspace:"


def _portable_task_target(value: str) -> str:
    """Persist workspace paths without machine roots or user identities."""
    raw = str(value)
    path = Path(raw)
    if not path.is_absolute():
        return raw
    from agent_utilities.core.workspace import get_agent_workspace

    try:
        relative = path.resolve().relative_to(get_agent_workspace().resolve())
    except ValueError as exc:
        raise ValueError(
            "durable ingestion targets must be inside the configured workspace"
        ) from exc
    return f"{_WORKSPACE_TARGET_PREFIX}{relative.as_posix()}"


def _resolve_task_target(value: str) -> Path:
    """Resolve a portable WorkItem target against runtime configuration."""
    raw = str(value)
    if not raw.startswith(_WORKSPACE_TARGET_PREFIX):
        return Path(raw)
    from agent_utilities.core.workspace import get_workspace_path

    return get_workspace_path(raw.removeprefix(_WORKSPACE_TARGET_PREFIX))


def _submit_kafka_notification(
    queue: Any, task_type: str, envelope: dict[str, Any]
) -> None:
    """Publish a Kafka task-submission notification, routed by task type
    (D-42, CONCEPT:AU-ORCH.scheduling.acquisition-lane-fairness).

    A :data:`~agent_utilities.core.resource_priority.HYDRATION_TASK_TYPES` task
    goes to the queue's ``put_hydration`` (the dedicated hydration-priority
    topic the reserved consumer subset polls first — see
    ``ingest_worker.py::start_ingest_consumer_pool``); everything else uses the
    ordinary ``put``. ``put_hydration`` only exists on the Kafka backend
    (:class:`~agent_utilities.knowledge_graph.core.kafka_queue_backend.
    KafkaQueueBackend`), so a queue without it (a test double, or a future
    non-Kafka caller of this helper) transparently falls back to ``put`` —
    this function is only ever reached from the Kafka branch of
    :meth:`TaskManagerMixin.submit_task`.
    """
    from agent_utilities.core.resource_priority import HYDRATION_TASK_TYPES

    put_hydration = getattr(queue, "put_hydration", None)
    if task_type in HYDRATION_TASK_TYPES and callable(put_hydration):
        put_hydration(envelope)
    else:
        queue.put(envelope)


def _coerce_prio_bucket(value: Any, default: int = 2) -> int:
    """Validate a current WorkItem claim bucket in the closed interval 0..3."""
    if value is None:
        return default
    if isinstance(value, bool):
        raise TypeError("WorkItem prio_bucket must be an integer")
    if isinstance(value, int):
        if 0 <= value <= 3:
            return value
        raise ValueError("WorkItem prio_bucket must be between 0 and 3")
    raise TypeError("WorkItem prio_bucket must be an integer")


import sqlite3

from .queue_backend import QueueBackend


def _kg_dev_mode() -> bool:
    """True when ``KG_DEV_MODE`` disables all KG background daemons.

    One switch replaces the per-daemon ``KG_*_DAEMON`` env toggles (which all
    defaulted on): production runs every daemon; dev can silence the lot. Read
    via ``AgentConfig`` so there's a single typed source of truth, not scattered
    ``os.environ`` reads. (CONCEPT:AU-KG.coordination.embedder-breaker / config discipline)
    """
    try:
        from agent_utilities.core.config import config

        return bool(getattr(config, "kg_dev_mode", False))
    except Exception:  # noqa: BLE001 — config unavailable → daemons on (prod default)
        return False


def compute_ingest_worker_count(configured: int | None = None) -> int:
    """Autosize the ingest worker pool for THIS host (CPU + memory bounded).

    The single sizing policy shared by the in-process task workers and the
    decoupled ``kg-ingest`` consumer pool (CONCEPT:AU-KG.ingest.decoupled-kg-ingest-consumer): ~36% of the cores,
    capped by available memory at ~3 GB per heavy worker, floor of 2. An
    explicit ``configured`` value (``KG_INGESTION_WORKERS``) wins outright.
    """
    if configured is None:
        from agent_utilities.core.config import DEFAULT_KG_INGESTION_WORKERS

        configured = DEFAULT_KG_INGESTION_WORKERS
    if configured:
        return int(configured)
    try:
        import os

        import psutil

        # Memory bound: assume ~3GB RAM per heavy (parse/LLM) worker.
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        max_mem_workers = max(1, int(available_gb / 3.0))

        # CPU bound: target ~36% of the cores so ingest never starves the box.
        cores = os.cpu_count() or 4
        max_cpu_workers = max(1, int(cores * 0.36))

        return max(2, min(max_cpu_workers, max_mem_workers))
    except Exception as e:  # noqa: BLE001 — sizing is best-effort
        logger.debug("Dynamic worker scaling failed, falling back to 4: %s", e)
        return 4


# Embedding-backfill sizing. Previously the single overloaded
# ``KG_EMBED_BACKFILL_BATCH`` env was read in two places with CONFLICTING
# defaults (256 vs 512) for two genuinely different knobs — a config bug. They
# are now two named constants: the per-tick node budget and the per-query DB
# fetch size. (CONCEPT:AU-KG.coordination.embedder-breaker / config discipline)
_EMBED_BACKFILL_BUDGET = 256
_EMBED_BACKFILL_FETCH = 512

# PerformanceAnomaly consumer cadence (CONCEPT:AU-AHE.optimization.performance-anomaly-consumer): a bounded, LLM-free
# scan, so a fixed moderate interval suffices — no env knob needed.
_ANOMALY_CONSUMER_INTERVAL = 900.0
# TMS staleness consumer cadence (CONCEPT:EG-KG.epistemic.truth-maintenance, Seam 3 follow-up, W3.2): a bounded,
# LLM-free scan of the engine's own durable reasoning projection — a fixed
# moderate interval suffices, no env knob needed.
_TMS_REVALIDATION_INTERVAL = 300.0

# Runtime-reliability detect→gap cadence (CONCEPT:AU-AHE.harness.runtime-reliability-loop):
# a bounded, LLM-free pass that aggregates recent :RuntimeSignal events into SOURCE_RUNTIME
# gaps — a few minutes keeps a building pattern fresh without churn. One correct value, not
# an env knob; the aggregation window (15min) spans several ticks.
_RUNTIME_RELIABILITY_INTERVAL = 180.0

# Background-daemon cadences (config discipline): each has
# one correct default and no per-host correctness requirement, so they are named
# module constants rather than env knobs (replacing KG_*_INTERVAL / KG_TASK_*).
# Seconds.
_EVOLUTION_INTERVAL = 3600.0
_RECONCILE_INTERVAL = 900.0
_ENRICH_INTERVAL = 20.0
_FILE_WATCH_INTERVAL = 30.0
# Fast cadence for the reactive autoscale poll (CONCEPT:AU-KG.compute.reactive-push): it only does a
# cheap non-blocking WorkItem change-feed poll and short-circuits when nothing
# changed, so it can run far more often than the slow ``_tick_fleet_autoscaler``
# safety-net interval — turning "scale on the change" from minutes into seconds.
_AUTOSCALE_REACTIVE_INTERVAL = 5.0
# Reactive placement-mining poll cadence (report §9 #6, X-5): a cheap
# non-blocking ``:ToolCall`` change-feed poll, far more often than the giant
# hourly Loop-engine cycle the (unchanged) mining pass used to depend on
# exclusively. Slightly slower than the autoscale poll — a ``:ToolCall`` fires
# far more often fleet-wide than a WorkItem/queue-depth change, and each
# actual mining pass (when ``PLACEMENT_CONTROL_LOOP_ENABLED``) runs real
# Cypher, so the tick trades a little latency for materially less redundant
# work under load.
_PLACEMENT_MINING_REACTIVE_INTERVAL = 30.0
_HYGIENE_INTERVAL = 86400.0
# Warm-fork parent + dev-workspace idle reap (CONCEPT:AU-OS.host.so-they-are-idle). Background; never preempts work.
_WARM_PARENT_REAP_INTERVAL = 300.0
# Package-install manifest watch (CONCEPT:AU-KG.ingest.package-install-autoingest): a
# cheap manifest-hash check (dedup no-ops when nothing installed since last tick), so
# it can poll far more often than a heavy sweep without wasted work.
_PACKAGE_INSTALL_INGEST_INTERVAL = 300.0
_EMBED_BACKFILL_IDLE_INTERVAL = 30.0
_EMBED_BACKFILL_BUSY_SLEEP = 1.0

# D-EMB: how often the non-pgvector fallback tick re-scans for `embedding`
# PROPERTY rows not yet in the ANN index. Deliberately much coarser than the
# pgvector path's 1s/30s cadence -- `hydrate_engine_embeddings` has no
# incremental cursor (it walks every node's properties each call), so running
# it on the tight cadence would add a full-graph scan to an already-contended
# engine every cycle (D-PERF-2).
_EMBED_BACKFILL_GENERIC_INTERVAL_S = 600.0

# Embedder circuit-breaker (CONCEPT:AU-KG.coordination.embedder-breaker): when the embedding endpoint is down
# (e.g. the GPU host power-cycles → vLLM 502s), the backfill tick must NOT keep
# calling it every 30s across N tables (each with client-side retries) — that
# retry-storm pegs the daemon and makes the whole KG surface time out. After this
# many consecutive embed failures the circuit OPENS: ticks become cheap no-ops
# (zero embed calls) for the cooldown, then one probe batch tests recovery.
_EMBED_CB_THRESHOLD = 3
_EMBED_CB_COOLDOWN = 300.0
# Card-enrichment circuit-breaker (CONCEPT:AU-KG.enrichment.card-attempt-status): mirrors the embedder
# breaker so a down card LLM (vLLM 502s) doesn't get retry-stormed every 20s. After this
# many consecutive fully-failed backfill batches the circuit OPENS: ticks become cheap
# no-ops for the cooldown, then one probe batch tests recovery.
_CARD_CB_THRESHOLD = 3
_CARD_CB_COOLDOWN = 300.0
_TASK_ORPHAN_GRACE_SEC = 90.0
_TASK_MAX_RUNTIME_SEC = 7200.0
# A WorkItem lease is an ownership/recovery boundary, not an execution limit.
# Keep it short so a crashed pod's ingestion job is reclaimable promptly; the
# separate two-hour watchdog above still bounds legitimate task execution.
_TASK_WORK_ITEM_LEASE_SEC = 60.0
_TASK_WORK_ITEM_HEARTBEAT_SEC = 20.0
_TASK_MAX_REQUEUE = 3
_USAGE_SYNC_INTERVAL = 900.0
_USAGE_PRICING_REFRESH_INTERVAL = 86400.0

# CONCEPT:AU-KG.ingest.hardened-priority-scheduled-task — Hardened priority and scheduled task queue with retry and dead-letter.
# Priority is one discrete integer bucket (0=critical .. 3=background). The
# native WorkItem claim orders and fences these buckets; no string priority or
# client-side claim selector exists.
_PRIORITY_BUCKETS: tuple[int, ...] = (0, 1, 2, 3)
_PRIO_CRITICAL, _PRIO_HIGH, _PRIO_NORMAL, _PRIO_BACKGROUND = 0, 1, 2, 3
_DEFAULT_PRIO_BUCKET = _PRIO_NORMAL

_TASK_MAX_ATTEMPTS = 3

# AU-P1-CL: ``_update_task_status``'s terminal-status arg -> the WorkItem
# outcome committed through the active native claim. All
# calls through ``_update_task_status`` are treated as non-retryable (the
# APP-LEVEL retry/backoff/DLQ decision lives in ``_fail_or_retry_task``,
# which commits through ``work_item.commit_result`` itself with
# ``retryable=True`` and only calls back into ``_update_task_status`` once
# IT has already decided the outcome is terminal — see that method).
_INGEST_TERMINAL_STATUS_TO_WORK_ITEM: dict[str, str] = {
    "completed": "succeeded",
    "failed": "failed",
    "cancelled": "cancelled",
    "dead_letter": "failed",
}


def _task_status_from_work_item(item: dict[str, Any] | None) -> str:
    """Render the public job-status vocabulary from the sole WorkItem."""
    status = str((item or {}).get("status") or "")
    if (
        status == "ready"
        and float((item or {}).get("next_retry_at") or 0) > time.time()
    ):
        return "scheduled"
    return {
        "submitted": "blocked",
        "ready": "pending",
        "leased": "running",
        "running": "running",
        "succeeded": "completed",
        "failed": "failed",
        "cancelled": "cancelled",
        "dead_letter": "dead_letter",
    }.get(status, "unknown")


def _retryable_partial_materialization(error: BaseException) -> dict[str, Any] | None:
    """Return the engine's typed hydration signal, never a text lookalike.

    A catalog-known graph deliberately rejects every graph operation while its
    bounded lazy-open rebuild is incomplete.  That is availability state, not
    an ingestion failure.  Only the exact retryable wire payload is safe to
    defer; malformed, stale, and terminal materialization errors continue
    through ordinary failure handling.
    """
    try:
        payload = json.loads(str(error))
    except (TypeError, json.JSONDecodeError):
        return None
    if (
        isinstance(payload, dict)
        and payload.get("code") == "PARTIAL_MATERIALIZATION"
        and payload.get("retryable") is True
    ):
        return payload
    return None


# Enrichment pass sizing (config discipline): per-tick LLM-card batch budget. The
# per-batch summarization concurrency is CPU/mem auto-sized via
# compute_ingest_worker_count(); these batch caps are bounded constants, not env
# knobs (replacing KG_ENRICH_BATCH / KG_ENRICH_MAX_BATCHES).
#
# CONCEPT:AU-KG.ontology.capability-card-backfill-lane — the per-TICK chunk is sized to drain the ``cards_pending``
# backlog at scale, not just trickle it. Each tick re-checks the foreground
# throttle BETWEEN batches and yields promptly, so a large MAX_BATCHES never
# blocks interactive work — it only bounds how much ONE enrichment-lane task
# does before completing and freeing the worker for fair re-claim. At
# 16 * 64 = 1024 symbols/tick and a 20s interval, a single lane worker drains
# ~180k symbols/hr — enough to clear an 85k backlog inside an hour, then the
# delta-skip (summary != '') keeps subsequent ticks cheap.
_ENRICH_BATCH = 16
_ENRICH_MAX_BATCHES = 64


class SQLiteTaskQueue(QueueBackend):
    """Thread-safe, persistent SQLite-backed queue for tasks to prevent memory loss on restarts."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.Lock()
        with self.lock:
            self._connect().close()

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with the schema ENSURED.

        Tables are (re)created on every connect (cheap ``IF NOT EXISTS``) so the
        queue self-heals if its parent directory or db file is deleted/recreated
        after init — otherwise every method would fail forever with
        ``no such table: staging`` once the file is gone. (CONCEPT:AU-KG.compute.registered-edge-type)
        """
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        with conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS queue (id INTEGER PRIMARY KEY, data TEXT)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS staging "
                "(id INTEGER PRIMARY KEY, job_id TEXT, graph_data TEXT)"
            )
        return conn

    def put(self, item: dict):
        self.put_many([item])

    def put_many(self, items: list[dict[str, Any]]) -> None:
        with self.lock:
            conn = self._connect()
            try:
                with conn:
                    conn.executemany(
                        "INSERT INTO queue (data) VALUES (?)",
                        ((json.dumps(item),) for item in items),
                    )
            finally:
                conn.close()

    def put_if_below(self, item: dict[str, Any], max_depth: int) -> bool:
        """Atomically admit one item under the local durable depth bound."""
        if max_depth < 1:
            raise ValueError("max_depth must be positive")
        with self.lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute("BEGIN IMMEDIATE")
                    row = conn.execute("SELECT COUNT(*) FROM queue").fetchone()
                    if row and int(row[0]) >= max_depth:
                        return False
                    conn.execute(
                        "INSERT INTO queue (data) VALUES (?)", (json.dumps(item),)
                    )
                    return True
            finally:
                conn.close()

    def get(self) -> tuple[int, dict] | None:
        with self.lock:
            conn = self._connect()
            try:
                with conn:
                    cur = conn.execute(
                        "SELECT id, data FROM queue ORDER BY id ASC LIMIT 1"
                    )
                    row = cur.fetchone()
                    if row:
                        return row[0], json.loads(row[1])
                    return None
            finally:
                conn.close()

    def ack(self, item_id: int):
        with self.lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute("DELETE FROM queue WHERE id = ?", (item_id,))
            finally:
                conn.close()

    def get_queue_size(self) -> int:
        with self.lock:
            conn = self._connect()
            try:
                with conn:
                    cur = conn.execute("SELECT COUNT(*) FROM queue")
                    row = cur.fetchone()
                    return row[0] if row else 0
            finally:
                conn.close()

    def put_staged_graph(self, job_id: str, nodes: list, edges: list):
        """Insert a serialized graph into the staging queue for the GraphWriterDaemon."""
        payload = json.dumps({"nodes": nodes, "edges": edges})
        with self.lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        "INSERT INTO staging (job_id, graph_data) VALUES (?, ?)",
                        (job_id, payload),
                    )
            finally:
                conn.close()

    def get_staged_graph(self) -> tuple[int, str, dict] | None:
        """Fetch the oldest staged graph payload."""
        with self.lock:
            conn = self._connect()
            try:
                with conn:
                    cur = conn.execute(
                        "SELECT id, job_id, graph_data FROM staging ORDER BY id ASC LIMIT 1"
                    )
                    row = cur.fetchone()
                    if row:
                        return row[0], row[1], json.loads(row[2])
                    return None
            finally:
                conn.close()

    def ack_staged_graph(self, item_id: int):
        """Acknowledge and remove a processed staged graph."""
        with self.lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute("DELETE FROM staging WHERE id = ?", (item_id,))
            finally:
                conn.close()


class GraphEngineProtocol(Protocol):
    backend: Any

    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict[str, Any] | None = None,
        ephemeral: bool = False,
    ) -> Any:
        if properties is None:
            properties = {}
        props = {"node_type": node_type, **properties, "ephemeral": ephemeral}
        if hasattr(self, "backend") and self.backend is not None:
            if hasattr(self.backend, "add_node"):
                return self.backend.add_node(node_id, **props)
        return {"id": node_id, "properties": props}

    def link_nodes(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict | None = None,
        ephemeral: bool = False,
    ) -> None:
        if properties is None:
            properties = {}
        props = {"relationship": rel_type, **properties, "ephemeral": ephemeral}
        if hasattr(self, "backend") and self.backend is not None:
            if hasattr(self.backend, "add_edge"):
                self.backend.add_edge(source_id, target_id, **props)

    def query_cypher(
        self, cypher: str, params: dict | None = None
    ) -> list[dict[str, Any]]:
        if hasattr(self, "backend") and self.backend is not None:
            if hasattr(self.backend, "execute"):
                return self.backend.execute(cypher, params)
        return []


class _ControlPlaneWorkItemEngine:
    """Adapts a ``TaskManagerMixin`` host to the ``engine`` protocol
    :mod:`agent_utilities.orchestration.work_item` expects (``add_node``/
    ``query_cypher``/``link_nodes``/``compare_and_set_node_fields``), bound
    ENTIRELY to the host's configured control authority
    (CONCEPT:AU-KG.backend.schedule-on-control-graph).

    Ingestion WorkItems live in the control plane. Reusing ``engine.add_node``/
    ``engine.query_cypher`` directly would route scheduling writes onto
    ``__commons__`` and reintroduce exactly the
    content-ingestion write-lock contention the control/commons split exists
    to avoid. This adapter is the ONLY thing that changes: every state-
    machine transition in ``work_item.py`` is reused unmodified.

    BUG-059 disposition — JUSTIFIED BYPASS, evaluated and confirmed standing.
    ``add_node``/``link_nodes`` below write directly through ``self._host._control``
    (a raw backend), never reaching ``IntelligenceGraphEngine._upsert_node`` /
    ``GraphComputeEngine.add_node``'s ``stamp_ownership`` gate — the same
    chokepoint-bypass shape as the other 12 BUG-059 sites, but here the reason
    is stronger than "nobody wired it": routing through the engine wrapper
    is the one thing this class exists to avoid (see above — it would put
    scheduling writes back on the commons-graph lock). WorkItems are
    control-plane scheduling bookkeeping (a durable work queue entry), not
    anyone's owned content — the same "system/platform data, not user data"
    shape ``stamp_ownership`` already leaves intentionally unowned for a
    privileged actor, and control-plane WorkItem creation runs from ingestion
    scheduling code paths with no per-request human/agent actor to bind in
    the first place. Pinned by
    ``tests/unit/knowledge_graph/core/test_engine_tasks_bug059.py`` so this
    stays a deliberate, documented exemption and is never "fixed" by
    accident into routing through ``engine.add_node``.
    """

    def __init__(self, host: "TaskManagerMixin") -> None:
        self._host = host

    def query_cypher(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return self._host._control_cypher(cypher, params)

    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict[str, Any] | None = None,
        ephemeral: bool = False,
    ) -> None:
        backend = self._host._control
        add = getattr(backend, "add_node", None)
        if not callable(add):
            from agent_utilities.orchestration.work_item import (
                WorkItemBackendUnavailable,
            )

            raise WorkItemBackendUnavailable(
                "control authority does not expose durable WorkItem creation"
            )
        # CONCEPT:AU-KG.ontology.node-type-casing-convergence — normalizes a node's class identity to the schema label so two engine adapters can no longer diverge on its casing.
        # The ``node_type`` PARAMETER (the schema label, e.g. ``"WorkItem"``)
        # is this call's single source of truth for the node's class identity,
        # matching
        # ``IntelligenceGraphEngine.add_node``'s own ``props["node_type"] =
        # node_type`` stamp one layer up. ``work_item.py``'s callers build
        # ``properties`` from ``RegistryNode.to_graph_properties()``, which
        # ALSO writes a ``node_type`` key — the lowercase snake_case
        # ``RegistryNodeType`` enum value (e.g. ``"work_item"``), not the
        # PascalCase label this adapter's caller passed. Spreading that dict
        # unreconciled let it silently override the label on this
        # control-plane path only, so ingestion-scheduled WorkItems landed as
        # ``node_type="work_item"`` while every other WorkItem (submitted
        # through the main-graph adapter, which already normalizes this) landed
        # as ``node_type="WorkItem"`` — the exact casing split measured live
        # (WorkItem 4,590 vs work_item 3,760). Overriding here, at the one
        # place both adapters converge before reaching their respective
        # backends, is the fix at the chokepoint: neither adapter can diverge
        # from the caller-supplied class identity again, regardless of what a
        # ``RegistryNode`` subclass happens to fold into its own properties.
        props = dict(properties or {})
        props["node_type"] = node_type
        add(node_id, label=node_type, **props)

    def create_node_if_absent(
        self, node_id: str, properties: dict[str, Any] | None = None
    ) -> bool:
        """Expose the control graph's atomic WorkItem create primitive.

        Repository-development admission needs the native membership-test and
        insert result so concurrent first writers can distinguish the winner
        from a deduplicated loser. The control-plane wrapper must delegate this
        directly to the host's native graph client; falling back to
        ``add_node`` would reintroduce the read-then-write race this adapter
        exists to close.
        """

        return bool(
            self._native_work_item_method("create_node_if_absent")(
                node_id, properties=properties
            )
        )

    def link_nodes(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict | None = None,
        ephemeral: bool = False,
    ) -> None:
        backend = self._host._control
        add_edge = getattr(backend, "add_edge", None)
        if not callable(add_edge):
            return
        try:
            add_edge(source_id, target_id, rel_type=str(rel_type), **(properties or {}))
        except Exception as e:  # noqa: BLE001 — viz edge is best-effort
            logger.debug("control-plane work-item view: link_nodes failed: %s", e)

    def compare_and_set_node_fields(
        self, node_id: str, conditions: dict[str, Any], updates: dict[str, Any]
    ) -> bool:
        backend = self._host._control
        fn = getattr(backend, "compare_and_set_node_fields", None)
        if not callable(fn):
            from agent_utilities.orchestration.work_item import (
                WorkItemBackendUnavailable,
            )

            raise WorkItemBackendUnavailable(
                f"{type(backend).__name__} has no compare_and_set_node_fields — "
                "the ingestion WorkItem requires an engine-native "
                "atomic CAS control backend."
            )
        return bool(fn(node_id, conditions, updates))

    def _native_work_items(self) -> Any:
        """Return the host's one process-owned native WorkItem client."""
        backend_graph = getattr(self._host._control, "graph", None)
        if backend_graph is None:
            from agent_utilities.orchestration.work_item import (
                WorkItemBackendUnavailable,
            )

            raise WorkItemBackendUnavailable(
                "control authority does not expose the native WorkItem client"
            )
        return backend_graph

    def _native_work_item_method(self, name: str) -> Any:
        """Return one required native WorkItem operation or fail closed."""
        target = self._native_work_items()
        method = getattr(target, name, None)
        if not callable(method):
            from agent_utilities.orchestration.work_item import (
                WorkItemBackendUnavailable,
            )

            raise WorkItemBackendUnavailable(
                f"control authority does not expose required WorkItem operation {name}"
            )
        return method

    def claim_work_item(self, request: dict[str, Any]) -> Any:
        return self._native_work_item_method("claim_work_item")(request)

    def renew_work_item_lease(self, request: dict[str, Any]) -> Any:
        return self._native_work_item_method("renew_work_item_lease")(request)

    def commit_work_item_result(self, request: dict[str, Any]) -> Any:
        return self._native_work_item_method("commit_work_item_result")(request)

    def cancel_work_item(self, request: dict[str, Any]) -> Any:
        return self._native_work_item_method("cancel_work_item")(request)

    def defer_work_item(self, request: dict[str, Any]) -> Any:
        return self._native_work_item_method("defer_work_item")(request)


class TaskManagerMixin(GraphEngineProtocol):
    """Mixin for the native persistent WorkItem queue.

    CONCEPT:AU-KG.compute.persistent-task-tracking - Persistent Task Tracking
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._workers_running = False
        self._worker_lock = threading.Lock()
        self._background_daemons_lock = threading.Lock()
        self._active_work_item_claims: dict[str, dict[str, Any]] = {}
        self._active_work_item_claims_lock = threading.Lock()
        self._work_item_lease_heartbeats: dict[
            str, tuple[threading.Event, threading.Event]
        ] = {}
        self._work_item_lease_heartbeats_lock = threading.Lock()

        # Pre-import LlamaIndex components in main thread to avoid parallel worker import race conditions
        try:
            from llama_index.core import SimpleDirectoryReader  # noqa: F401
            from llama_index.core.embeddings import BaseEmbedding  # noqa: F401
        except ImportError as exc:  # noqa: BLE001 — ImportError-guarded optional-dependency pre-import (already labeled 'optional dependency' in the log message); LlamaIndex readers are looked up again, lazily, wherever they're actually used
            logger.debug("LlamaIndex pre-import skipped (optional dependency): %s", exc)

        # Initialize pluggable persistent task queue
        from agent_utilities.core.config import config
        from agent_utilities.core.paths import data_dir

        queue_db_path = data_dir() / "kg_task_queue.db"

        # Queue selection is ONE explicit, fail-loud path (CONCEPT:AU-KG.backend.selectable-queue-backend):
        # TASK_QUEUE_BACKEND=sqlite|postgres|kafka, default auto (postgres when
        # STATE_DB_URI is set — CONCEPT:AU-OS.state.unified-durable-state-externalization/KG-2.54 — else sqlite). An
        # explicitly selected kafka/postgres queue that is unreachable raises
        # TaskQueueUnavailable here instead of silently degrading.
        from .queue_backend import create_task_queue

        self._submission_queue: QueueBackend
        self._submission_queue, self._task_queue_backend_name = create_task_queue(
            config, str(queue_db_path)
        )

        import sys

        # ── Role-gated background daemon (CONCEPT:AU-KG.coordination.embedder-breaker / OS-5.0) ───────────
        # The KG runs ONE consolidated daemon. ``client`` processes (the MCP
        # server, CLI, one-shot scripts, and tests) spawn NOTHING — they enqueue
        # work to the durable queue that the ``host`` daemon (the API gateway)
        # drains. ``host``/``auto`` run the daemon here. This replaces the former
        # five independently-spawned thread families (submitter, graph-writer,
        # task workers, per-job maintenance daemons, and the SDD/scholarx file
        # watcher) with one lifecycle.
        self._daemon_role = daemon_role()
        _test_or_staging = bool(
            setting("AGENT_UTILITIES_TESTING") or "--stage-to-queue" in sys.argv
        )
        # Singleton election (CONCEPT:AU-KG.coordination.embedder-breaker / OS-5.9): only the flock holder runs
        # the consolidated daemon. ``auto`` self-heals to ``client`` when a host
        # already holds the lock; an explicit ``host`` that loses raises
        # KGHostAlreadyRunning (descriptive). Test/staging never elect or lock —
        # they skip auto-start but keep a non-client effective role so explicit
        # start_task_workers() still works.
        if _test_or_staging:
            self._effective_role = "client" if self._daemon_role == "client" else "host"
        else:
            from .host_lock import resolve_daemon_role

            self._effective_role = resolve_daemon_role(self._daemon_role)
        if self._effective_role == "client" or _test_or_staging:
            logger.info(
                "KG daemon auto-start skipped (requested=%s, effective=%s, "
                "test/staging=%s); the host daemon drains the durable queue.",
                self._daemon_role,
                self._effective_role,
                _test_or_staging,
            )
            return

        if getattr(self, "_defer_background_start", False):
            logger.info(
                "KG background daemons deferred until the graph materialization "
                "barrier completes."
            )
            return
        self.start_background_daemons()

    def start_background_daemons(self) -> None:
        """Start the consolidated daemon families once graph startup is writable.

        GraphOS cold-open sets ``_defer_background_start`` before engine
        construction, then calls this only after the native materialization
        manifest is complete and valid. Other host entrypoints retain the
        historical eager behavior.
        """
        if getattr(self, "_effective_role", "client") == "client":
            return
        with self._background_daemons_lock:
            existing = getattr(self, "_graph_writer_thread", None)
            if existing is not None and existing.is_alive():
                return
            self._defer_background_start = False
        self._background_worker_session = _capture_verified_background_session()
        self._graph_writer_thread = _authorized_background_thread(
            self._background_worker_session,
            self._graph_writer_loop,
            name="KG-Graph-Writer",
        )
        self._graph_writer_thread.start()

        # KG background daemons are always on in production; the single
        # KG_DEV_MODE switch disables the whole set (it replaced the per-daemon
        # KG_*_DAEMON env toggles). (CONCEPT:AU-KG.coordination.embedder-breaker / config discipline)
        if _kg_dev_mode():
            return

        # Single consolidated maintenance scheduler (CONCEPT:AU-KG.coordination.embedder-breaker): runs ALL
        # periodic KG jobs (analysis, compaction, evolution, enrichment, AND the
        # SDD/skills/scholarx file-watch scan — see ``_maintenance_jobs``) in ONE
        # throttled thread behind one shared foreground gate. No separate file
        # watcher thread.
        self._maintenance_thread = _authorized_background_thread(
            self._background_worker_session,
            self._maintenance_scheduler_loop,
            name="KG-Maintenance-Scheduler",
        )
        self._maintenance_thread.start()

        # Dedicated vector-embedding backfill drain (separate from the periodic
        # scheduler so it is never starved behind slow LLM ticks). (KG-2.8)
        self._embed_backfill_thread = _authorized_background_thread(
            self._background_worker_session,
            self._embedding_backfill_loop,
            name="KG-Embedding-Backfill",
        )
        self._embed_backfill_thread.start()

    def _background_session_for_spawn(self) -> Any:
        """Return the immutable daemon authority, capturing it exactly once."""
        session = getattr(self, "_background_worker_session", None)
        if session is None:
            session = _capture_verified_background_session()
            self._background_worker_session = session
        return _require_verified_background_session(session)

    # ── Control-plane backend routing (CONCEPT:AU-KG.backend.schedule-on-control-graph) ─────────────────
    # Native WorkItem operations and :Schedule state are CONTROL plane. The
    # engine configures exactly one control authority; absence is a hard error,
    # never an invitation to create a second work-state location in the content
    # graph. Content/document/codebase ingestion deliberately uses ``self.backend``.

    @property
    def _control(self) -> Any:
        """Return the configured control-plane authority or fail closed.

        A missing authority cannot fall back to the content backend: doing so
        would split WorkItem truth across graph locations.
        """
        control = getattr(self, "control_backend", None)
        if control is None:
            from agent_utilities.orchestration.work_item import (
                WorkItemBackendUnavailable,
            )

            raise WorkItemBackendUnavailable(
                "the configured WorkItem control authority is unavailable"
            )
        return control

    def _control_cypher(
        self, cypher: str, params: dict | None = None
    ) -> list[dict[str, Any]]:
        """Run a CONTROL-PLANE Cypher read/write against ``__control__``.

        Mirrors ``query_cypher`` but targets the isolated control backend
        (CONCEPT:AU-KG.backend.schedule-on-control-graph). Used for WorkItem / :Schedule / queue / claim ops so
        they never block on the content-ingestion write lock.
        """
        ctrl = self._control
        execute_read = getattr(ctrl, "execute_read", None)
        if not callable(execute_read):
            from agent_utilities.orchestration.work_item import (
                WorkItemBackendUnavailable,
            )

            raise WorkItemBackendUnavailable(
                "the WorkItem control authority has no governed read surface"
            )
        return execute_read(cypher, params)

    @property
    def _work_item_engine(self) -> _ControlPlaneWorkItemEngine:
        """Cached :class:`_ControlPlaneWorkItemEngine` view for the ingestion
        queue's sole WorkItem authority."""
        view = getattr(self, "_work_item_engine_cache", None)
        if view is None:
            view = _ControlPlaneWorkItemEngine(self)
            self._work_item_engine_cache = view
        return view

    def unified_daemon_status(self) -> dict[str, Any]:
        """Status of the single consolidated background daemon (CONCEPT:AU-KG.coordination.embedder-breaker).

        Reports this process's role and which daemon threads are alive, so the
        API gateway can surface one '/daemon/status' view instead of scattered
        per-thread state.
        """

        def _alive(attr: str) -> bool:
            t = getattr(self, attr, None)
            return bool(t and t.is_alive())

        role = getattr(self, "_daemon_role", None) or daemon_role()
        from .host_lock import effective_daemon_role, host_lock_holder

        threads = {
            "submission": _alive("_submission_thread"),
            "graph_writer": _alive("_graph_writer_thread"),
            "maintenance": _alive("_maintenance_thread"),
            "embed_backfill": _alive("_embed_backfill_thread"),
            "task_workers": bool(getattr(self, "_workers_running", False)),
        }
        status: dict[str, Any] = {
            "role": role,
            "effective_role": getattr(self, "_effective_role", None)
            or effective_daemon_role(),
            "host_lock_holder": host_lock_holder(),
            "running": any(threads.values()),
            "threads": threads,
            "maintenance_jobs": [n for n, _, _ in self._maintenance_jobs()],
        }
        status["queue_backend"] = getattr(self, "_task_queue_backend_name", None)
        try:
            q = getattr(self, "_submission_queue", None)
            if q is not None:
                status["queue_depth"] = q.get_queue_size()
        except Exception as exc:  # noqa: BLE001 — status surface stays best-effort
            logger.debug("queue depth probe failed: %s", exc)
        # Engine shard topology + per-shard reachability (CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw /
        # CONCEPT:AU-OS.scaling.shard-topology-visibility-per)
        # The flock host role above governs only the LOCAL engine; remote
        # shards are probed (short transport-level connect) and reported
        # here, never managed.
        try:
            from .shard_topology import shard_topology_status

            status["shards"] = shard_topology_status()
        except Exception:  # noqa: BLE001 - status surface stays best-effort
            pass
        return status

    def _tick_kg_analysis(self) -> None:
        """One autonomous-analysis tick (CONCEPT:AU-KG.compute.cross-pillar-synergy).

        Schedules a relevance sweep hourly, then selects the highest-degree
        stale ``Concept`` for background deep analysis. Run by the consolidated
        maintenance scheduler (no own thread / sleeps / throttle gate).
        """
        import time

        RELEVANCE_SWEEP_INTERVAL = 3600.0  # 60 minutes
        now = time.time()
        # CONCEPT:AU-ECO.messaging.debounce-relevance-sweep — do NOT fire the heavy relevance sweep immediately on every
        # restart (the prior 0.0 default did): co-located with the messaging router, a
        # startup sweep saturates the process and starves the inbound reply loop. Defer the
        # first sweep by one full interval after start.
        last_relevance_sweep = getattr(self, "_last_relevance_sweep", None)
        if last_relevance_sweep is None:
            self._last_relevance_sweep = now
            return
        if now - last_relevance_sweep >= RELEVANCE_SWEEP_INTERVAL:
            try:
                primary = self._detect_primary_codebase()
                if primary:
                    logger.info(
                        "KGAnalysis: scheduling relevance sweep for '%s'", primary
                    )
                    self.submit_task(
                        target_path=primary,
                        is_codebase=False,
                        task_type="relevance_sweep",
                        provenance={
                            "source": "autonomous_kg_daemon",
                            "mode": "scheduled",
                        },
                    )
            except Exception as e:
                logger.error(f"Relevance sweep scheduling error: {e}")
            self._last_relevance_sweep = now

        from datetime import datetime, timedelta

        cutoff = (datetime.now(UTC) - timedelta(days=7)).isoformat()
        query = (
            "MATCH (n:Concept) "
            "WHERE (n.last_analyzed IS NULL OR n.last_analyzed < $cutoff) "
            "WITH n, size((n)--()) as degree "
            "ORDER BY degree DESC "
            "LIMIT 1 "
            "RETURN n.id as id, n.name as name"
        )
        results = self.query_cypher(query, {"cutoff": cutoff})
        if not results:
            return

        node_id = results[0]["id"]
        node_name = results[0].get("name") or node_id
        logger.info(
            "KGAnalysis: selected '%s' (%s) for background deep analysis.",
            node_name,
            node_id,
        )
        self.backend.execute(
            "MATCH (n:Concept {id: $id}) SET n.last_analyzed = current_timestamp()",
            {"id": node_id},
        )
        from agent_utilities.core.config import DEFAULT_KG_ANALYSIS_MAX_DEPTH

        self.submit_task(
            target_path=node_name,
            is_codebase=False,
            task_type="deep_analysis",
            provenance={
                "current_depth": 0,
                "max_depth": DEFAULT_KG_ANALYSIS_MAX_DEPTH,
                "source": "autonomous_kg_daemon",
            },
        )

    def _detect_primary_codebase(self) -> str | None:
        """Detect the primary codebase by finding the repository with the most Code nodes."""
        try:
            results = self.query_cypher(
                "MATCH (c:Code) WHERE c.file_path IS NOT NULL "
                "RETURN c.id AS id, c.file_path AS path LIMIT 500"
            )
            if not results:
                return None

            # Extract repository roots from paths
            repo_counts: dict[str, int] = {}
            for row in results:
                path = row.get("path", "")
                if not path:
                    continue
                # Resolve the repository by the portable tree marker, independent
                # of checkout depth, account name, or operating system.
                parts = [part for part in path.replace("\\", "/").split("/") if part]
                if "agent-packages" in parts:
                    marker = parts.index("agent-packages")
                    repo_name = parts[marker + 1] if marker + 1 < len(parts) else ""
                    if not repo_name:
                        continue
                    repo_counts[repo_name] = repo_counts.get(repo_name, 0) + 1

            if repo_counts:
                return max(repo_counts, key=repo_counts.get)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001 — best-effort heuristic; returns None (the documented 'unknown' case) exactly like when repo_counts is empty, so callers already handle this return uniformly
            logger.debug(f"Primary codebase detection failed: {e}")
        return None

    # ── Consolidated maintenance scheduler (CONCEPT:AU-KG.coordination.embedder-breaker) ──────────────

    def _maintenance_jobs(self) -> list[tuple[str, float, Any]]:
        """Inline plumbing the maintenance thread runs DIRECTLY (CONCEPT:AU-OS.state.unified-scheduling-one-intelligent).

        Everything else recurring (analysis, the self-evolution loop, enrichment,
        evolution, the fleet ticks, usage/file/hygiene/tenant-gc sweeps, and the
        declarative ``deploy/schedules.yml`` entries) is now a durable
        ``:Schedule`` that the ``scheduler`` tick ENQUEUES onto the unified queue
        — those bodies run in the worker pool under native WorkItem leases, not
        in this thread (see :meth:`_register_maintenance_schedules`). Only the
        queue's OWN plumbing stays inline, because it must run even when the
        queue/workers are saturated (it is what feeds and heals them):

          * ``scheduler``       — evaluate :Schedule nodes and enqueue due jobs
        Native ``ClaimWorkItem`` owns lease recovery, dependency release, and
        delayed availability. No Python maintenance writer is permitted to
        compete with that authority.
        """
        return [
            ("scheduler", 60.0, self._tick_scheduler),
        ]

    def _register_maintenance_schedules(self) -> None:
        """Register the former fixed-interval maintenance ticks as durable
        ``:Schedule`` nodes so the unified scheduler enqueues them (CONCEPT:AU-OS.state.unified-scheduling-one-intelligent).

        Each becomes an ``interval`` schedule whose ``scheduled_job`` runs the
        engine ``_tick_<ref>`` method (``kind: maint``) — or, for the
        self-evolution loop, a ``kind: maint`` schedule pointing at ``_tick_loop``.
        The config gates that used to decide whether a tick was *registered* now
        decide whether its *schedule* is registered, so the opt-in/opt-out
        defaults are unchanged. Maintenance runs at background priority (bucket 3)
        so it never preempts real ingestion/research; the loop runs at bucket 2.
        Run once at startup; idempotent (registration preserves live run state).
        """
        from agent_utilities.core.config import DEFAULT_KG_MODEL_ID
        from agent_utilities.core.config import config as _cfg
        from agent_utilities.core.schedule_engine import (
            ScheduleSpec,
            register_schedule,
        )

        specs: list[ScheduleSpec] = []

        def _maint(name, ref, interval, *, enabled=True, prio=3, task_type=None):
            # Always upsert the node (with ``enabled`` reflecting the config gate)
            # so toggling a flag off across a restart disables the schedule too.
            # CONCEPT:AU-KG.ontology.capability-card-backfill-lane — ``task_type`` lets a high-volume maint job run in
            # its OWN functional lane (default ``scheduled_job`` = the maint lane).
            specs.append(
                ScheduleSpec(
                    name=name,
                    payload={"kind": "maint", "ref": ref},
                    trigger="interval",
                    interval_s=float(interval),
                    prio_bucket=prio,
                    enabled=bool(enabled),
                    task_type=task_type or "scheduled_job",
                )
            )

        _maint("analysis", "kg_analysis", 120.0, enabled=bool(DEFAULT_KG_MODEL_ID))
        # Self-evolution Loop engine cycle (CONCEPT:AU-KG.research.these-properties-carry), OPT-IN via KG_LOOP=1;
        # runs _tick_loop as a task at research priority.
        _maint(
            "loop_cycle", "loop", _cfg.kg_loop_interval, enabled=_cfg.kg_loop, prio=2
        )
        # ScholarX RSS research-feed screen (CONCEPT:AU-KG.research.scholarx-rss-research-feed): grade incoming RSS
        # items, skip already-seen, enqueue prioritized full-paper fetch+ingest.
        # Default-ON (no-ops without ScholarX); KG_RESEARCH_FEED=0 disables.
        specs.append(
            ScheduleSpec(
                name="research_feed",
                payload={"kind": "research_feed"},
                trigger="interval",
                interval_s=float(getattr(_cfg, "kg_research_feed_interval", 1800.0)),
                prio_bucket=2,
                enabled=bool(getattr(_cfg, "kg_research_feed", True)),
            )
        )
        _maint(
            "sai_factory",
            "sai_factory",
            _cfg.kg_sai_factory_interval,
            enabled=_cfg.kg_sai_factory,
        )
        _maint(
            "failure_ingest",
            "failure_ingest",
            _cfg.kg_failure_evolution_interval,
            enabled=_cfg.kg_failure_evolution,
        )
        _maint(
            "optimization",
            "optimize_components",
            _cfg.kg_optimization_interval,
            enabled=_cfg.kg_optimization_enabled,
        )
        _maint(
            "anomaly_consumer",
            "anomaly_consumer",
            _ANOMALY_CONSUMER_INTERVAL,
            enabled=_cfg.kg_anomaly_consumer,
        )
        # TMS staleness consumer (CONCEPT:EG-KG.epistemic.truth-maintenance, Seam 3
        # follow-up, W3.2) — bounded, LLM-free, propose-only; native by default,
        # like ``compaction``/``evolution`` below (no config flag needed).
        _maint("tms_revalidation", "tms_revalidation", _TMS_REVALIDATION_INTERVAL)
        # Runtime-reliability detect→gap loop (CONCEPT:AU-AHE.harness.runtime-reliability-loop)
        # — bounded, LLM-free, propose-only; native by default (no flag), background
        # priority, like ``tms_revalidation``/``anomaly_consumer``. Turns the four runtime
        # signals (engine latency / listener restart / retrieval degrade / delegation
        # over-budget) into SOURCE_RUNTIME gaps + safe heals/recommendations.
        _maint(
            "runtime_reliability", "runtime_reliability", _RUNTIME_RELIABILITY_INTERVAL
        )
        _maint(
            "fuseki_publish",
            "fuseki_publish",
            _cfg.kg_fuseki_publish_interval,
            enabled=_cfg.kg_fuseki_publish,
        )
        _maint(
            "fleet_reconciler",
            "fleet_reconciler",
            _cfg.fleet_reconciler_interval,
            enabled=_cfg.fleet_reconciler,
        )
        _maint(
            "fleet_autoscaler",
            "fleet_autoscaler",
            _cfg.fleet_autoscaler_interval,
            enabled=_cfg.fleet_autoscaler,
        )
        # CONCEPT:AU-KG.compute.reactive-push — reactive push half of OS-5.29: a fast, cheap poll of the
        # engine's WorkItem change-feed that evaluates only on a queue-depth change.
        # Same opt-in gate as the autoscaler; the slow tick above is the safety net.
        _maint(
            "fleet_autoscale_reactive",
            "fleet_autoscale_reactive",
            _AUTOSCALE_REACTIVE_INTERVAL,
            enabled=_cfg.fleet_autoscaler,
        )
        # Placement-mining reactive trigger (report §9 #6, X-5) — Native by
        # default: the TRIGGER itself is unconditionally scheduled (no flag),
        # so a cheap ``:ToolCall`` change-feed poll always runs; the actual
        # mine->propose->govern->canary pass it fires on a change still passes
        # through ``placement_control_loop``'s own existing, deliberately
        # conservative ``PLACEMENT_CONTROL_LOOP_ENABLED`` gate (default off) —
        # this closes "nothing calls run_placement_mining_cycle automatically"
        # without changing that gate's default.
        _maint(
            "placement_mining_reactive",
            "placement_mining_reactive",
            _PLACEMENT_MINING_REACTIVE_INTERVAL,
        )
        _maint("compaction", "compaction", 1800.0)
        _maint("evolution", "evolution", _EVOLUTION_INTERVAL)
        from ..backends.fanout_backend import FanOutBackend

        _mirror_backend = getattr(getattr(self, "backend", None), "inner", None)
        if _mirror_backend is None:
            _mirror_backend = getattr(self, "backend", None)
        _maint(
            "reconcile_mirrors",
            "reconcile_mirrors",
            _RECONCILE_INTERVAL,
            enabled=isinstance(_mirror_backend, FanOutBackend),
        )
        # CONCEPT:AU-KG.ontology.capability-card-backfill-lane — OWL capability-card backfill runs in its OWN
        # ``enrichment`` lane (task_type ``enrichment_backfill``), NOT the
        # best-effort maint lane. Previously it rode ``scheduled_job`` and so was
        # capped at the maint floor (1 worker shared with ~17 ticks), leaving ~85k
        # Code symbols un-carded. As its own non-best-effort lane it drains the
        # cards_pending backlog in parallel while the background-throttle semaphore
        # + per-lane reservation keep it from starving the control/query planes.
        _maint(
            "enrichment",
            "enrichment",
            _ENRICH_INTERVAL,
            prio=2,
            task_type="enrichment_backfill",
        )
        _usage = bool(getattr(_cfg, "usage_tracking_enabled", True))
        _maint("usage_log_sync", "usage_log_sync", _USAGE_SYNC_INTERVAL, enabled=_usage)
        _maint(
            "usage_pricing_refresh",
            "usage_pricing_refresh",
            _USAGE_PRICING_REFRESH_INTERVAL,
            enabled=_usage,
        )
        _maint(
            "file_watch",
            "file_watch",
            _FILE_WATCH_INTERVAL,
            enabled=bool(getattr(_cfg, "enable_sdd_watcher", True)),
        )
        _maint("hygiene", "hygiene", _HYGIENE_INTERVAL)
        _maint("tenant_gc", "tenant_gc", _cfg.kg_tenant_gc_interval)
        # Goals-as-contracts SLA watch (CONCEPT:AU-ORCH.session.escalate-breached-goals): escalate breached goals.
        # Default-on; no-ops when no goals carry an sla_seconds.
        _maint("goal_sla", "goal_sla", 300.0)
        # Warm-fork parent + dev-workspace idle reap (CONCEPT:AU-OS.host.so-they-are-idle). Default-on;
        # no-ops when no warm parents / idle workspaces exist.
        _maint("warm_parent_reap", "warm_parent_reap", _WARM_PARENT_REAP_INTERVAL)
        # Package-install manifest watch (CONCEPT:AU-KG.ingest.package-install-autoingest): auto-extend
        # the KG when a package is installed — watches the universal-installer's
        # install-manifest.json and re-drives the prompt/ontology/skill reloads on
        # change. Default-on; the manifest-hash watermark makes an unchanged manifest
        # (the common case — nothing installed since the last tick) a cheap no-op.
        _maint(
            "package_install_ingest",
            "package_install_ingest",
            _PACKAGE_INSTALL_INGEST_INTERVAL,
        )

        for spec in specs:
            try:
                register_schedule(self, spec)
            except Exception as exc:  # noqa: BLE001 — one schedule never blocks others
                logger.debug(
                    "register maintenance schedule failed (exception_type=%s)",
                    type(exc).__name__,
                )

    def _tick_warm_parent_reap(self) -> None:
        """Reap idle warm-fork parents + idle dev-workspace containers (CONCEPT:AU-OS.host.so-they-are-idle).

        Background maintenance closes governed warm parents idle past the registry TTL and runs
        ``DockerWorkspace.reap_idle`` (OS-5.33) so leaked dev-workspace containers are cleaned on
        the same tick. No-ops cheaply when nothing is pooled.
        """
        try:
            from agent_utilities.runtime.warm_registry import WarmParentRegistry

            reaped = WarmParentRegistry.reap_active()
            if reaped:
                logger.info("Reaped %d idle warm-fork parent(s).", len(reaped))
        except Exception as e:  # noqa: BLE001 — reap is best-effort
            logger.debug("warm-parent reap skipped: %s", e)
        try:
            from agent_utilities.runtime.docker_workspace import DockerWorkspace

            workspaces = DockerWorkspace.reap_idle()
            if workspaces:
                logger.info(
                    "Reaped %d idle dev-workspace container(s).", len(workspaces)
                )
        except Exception as e:  # noqa: BLE001 — reap is best-effort (mirrors the warm-parent reap above); a skipped tick just leaves idle containers for the NEXT scheduled tick to reap, no state is falsely advanced
            logger.debug("dev-workspace reap skipped: %s", e)

    def _tick_package_install_ingest(self) -> None:
        """Auto-extend the KG when a package is installed (CONCEPT:AU-KG.ingest.package-install-autoingest).

        Runs the ``package_install`` connector (:mod:`..ingestion.package_install_ingest`)
        against the live engine — the same handler ``source_sync
        source=package_install`` calls, so this scheduled tick and the on-demand
        MCP/REST trigger share one implementation. Watermarked on the
        ``install-manifest.json`` content hash, so a tick where nothing new was
        installed since the last run is a cheap no-op.
        """
        try:
            from agent_utilities.knowledge_graph.core.source_sync import sync_source

            report = sync_source(self, "package_install", mode="delta")
            if not report.get("skipped_unchanged"):
                logger.info("package_install_ingest: %s", report)
        except Exception as e:  # noqa: BLE001 — one job's failure never stops others
            logger.debug("package_install_ingest tick error: %s", e)

    def _tick_goal_sla(self) -> None:
        """Evaluate open goals against their SLA + escalate breaches (ORCH-1.78)."""
        try:
            from agent_utilities.core.goal_sla import evaluate_goal_slas

            report = evaluate_goal_slas(self)
            if report.get("breached") or report.get("at_risk"):
                logger.info(
                    "goal_sla: %d breached, %d at-risk (of %d open)",
                    len(report["breached"]),
                    len(report["at_risk"]),
                    report["checked"],
                )
        except Exception as e:  # noqa: BLE001 — one job's failure never stops others
            logger.debug("goal_sla tick error: %s", e)

    def _tick_usage_log_sync(self) -> None:
        """Auto-detect + sync local agent logs into the usage store (ECO-4.42).

        Best-effort and bounded; skips silently when the collector is
        unavailable. The collector itself no-ops on a remote-client engine
        (the client pushes instead), so this only does work on the log host.
        """
        try:
            from agent_utilities.ingestion.collector import collect_local_sessions

            result = collect_local_sessions()
            if result.get("ingested"):
                logger.info("usage_log_sync: %s", result)
        except Exception as e:  # noqa: BLE001 — one maintenance-tick usage-log sync; the next tick retries with no state to reconcile (collect_local_sessions is safe to call repeatedly)
            logger.debug("usage_log_sync skipped: %s", e)

    def _tick_usage_pricing_refresh(self) -> None:
        """Refresh the LiteLLM pricing catalog into the usage store (ECO-4.40)."""
        try:
            from agent_utilities.pricing import refresh_catalog
            from agent_utilities.usage import get_usage_backend

            try:
                backend = get_usage_backend()
            except Exception:  # noqa: BLE001
                backend = None
            n = refresh_catalog(backend=backend)
            logger.debug("usage_pricing_refresh: merged %d models", n)
        except Exception as e:  # noqa: BLE001 — one maintenance-tick pricing-catalog refresh; the catalog simply stays at its last-known values until the next tick retries
            logger.debug("usage_pricing_refresh skipped: %s", e)

    def _tick_hygiene(self) -> None:
        """One memory-hygiene pass (CONCEPT:EG-KG.compute.compiled-semantic-reasoner).

        Archives stale AI-generated memory by closing its bi-temporal ``valid_to`` (never deletes;
        alerts high-confidence stale items) and merges near-duplicates. Run by the consolidated
        maintenance scheduler behind the shared foreground-throttle gate.
        """
        try:
            from agent_utilities.knowledge_graph.memory.hygiene import MemoryHygiene

            summary = MemoryHygiene(self).run()
            if summary.get("archived") or summary.get("merged"):
                logger.info(
                    "[KG-2.17] hygiene: archived=%s alerted=%s merged=%s scanned=%s",
                    summary.get("archived"),
                    summary.get("alerted"),
                    summary.get("merged"),
                    summary.get("scanned"),
                )
        except Exception as e:  # noqa: BLE001 — one job's failure never stops others
            logger.debug("hygiene tick error: %s", e)

    def _tick_fleet_reconciler(self) -> None:
        """One desired-state fleet reconcile pass (CONCEPT:AU-OS.config.desired-state-fleet-reconciler).

        Diffs the fleet registry (+ optional desired-state override) against
        the pluggable FleetObserver and converges each divergence through the
        ActionPolicy decision point (CONCEPT:AU-OS.deployment.fleet-lifecycle-control) and the FleetActuator
        seam; also drains human-granted ActionApproval entries. Storm-guarded
        (FLEET_RECONCILER_MAX_ACTIONS per tick); leader-only via the
        consolidated maintenance scheduler.
        """
        try:
            from agent_utilities.orchestration.fleet_reconciler import reconcile_fleet

            report = reconcile_fleet(self)
            if report.get("divergences") or report.get("approved_drained"):
                logger.info(
                    "[OS-5.25] fleet reconcile: divergences=%s processed=%s "
                    "deferred=%s approved_drained=%s actuator=%s",
                    report.get("divergences"),
                    report.get("processed"),
                    len(report.get("deferred") or []),
                    len(report.get("approved_drained") or []),
                    report.get("actuator"),
                )
        except Exception as e:  # noqa: BLE001 — one job's failure never stops others
            logger.debug("fleet_reconciler tick error: %s", e)

    def _tick_fleet_autoscaler(self) -> None:
        """One reactive autoscale pass (CONCEPT:AU-OS.scaling.reactive-replica-autoscaling).

        For each registry service with a ``scaling:`` block: read its load
        signal through the pluggable ScalingSignalProvider, target-track a
        desired replica count within the declared min/max bounds (step-capped,
        cooldown/flap-guarded against the durable action ledger), diff against
        the FleetObserver and propose ``scale_service`` through the
        ActionPolicy decision point (CONCEPT:AU-OS.deployment.fleet-lifecycle-control) + FleetActuator seam;
        scale-ups get an OS-5.27 deploy watch. Leader-only via the
        consolidated maintenance scheduler.
        """
        try:
            from agent_utilities.orchestration.fleet_autoscaler import autoscale_fleet

            report = autoscale_fleet(self)
            if report.get("actions"):
                logger.info(
                    "[OS-5.29] fleet autoscale: evaluated=%s actions=%s scaled=%s "
                    "actuator=%s signals=%s",
                    report.get("evaluated"),
                    report.get("actions"),
                    report.get("scaled"),
                    report.get("actuator"),
                    report.get("signal_provider"),
                )
        except Exception as e:  # noqa: BLE001 — one job's failure never stops others
            logger.debug("fleet_autoscaler tick error: %s", e)

    def _fleet_autoscale_subscription(self) -> Any:
        """Lazily-built reactive control-plane WorkItem change-feed.

        One subscription per daemon process, cached on the engine, so the reactive
        autoscale tick fires on the engine's pushed WorkItem change-event (the
        queue-depth signal moved) rather than waiting out the slow safety-net
        interval. Rebuilt if it couldn't resolve a streaming surface on first use.
        """
        sub = getattr(self, "_autoscale_subscription", None)
        if sub is None or not getattr(sub, "available", False):
            from agent_utilities.orchestration.fleet_autoscaler import (
                fleet_autoscale_subscription,
            )

            sub = fleet_autoscale_subscription(self)
            self._autoscale_subscription = sub
        return sub

    def _tick_fleet_autoscale_reactive(self) -> None:
        """Fire an autoscale evaluation on a control-plane WorkItem change.

        The push half of OS-5.29 autoscaling: poll the engine's WorkItem change-feed
        (non-blocking, O(new changes)) and run one ``autoscale_fleet`` pass ONLY when
        the engine pushed a queue-depth-moving change since the last poll — so a burst
        of enqueued work scales the fleet at change-time, not at the next slow
        ``_tick_fleet_autoscaler`` interval (which remains the safety-net reconcile).
        A no-op when the engine has no streaming surface (the periodic tick covers it).
        Leader-only via the consolidated maintenance scheduler.
        """
        try:
            sub = self._fleet_autoscale_subscription()
            if not sub.available:
                return
            sub.poll(block_ms=0)
            if sub.pending_state["pending"] == 0:
                return
            sub.pending_state["pending"] = 0

            from agent_utilities.orchestration.fleet_autoscaler import autoscale_fleet

            report = autoscale_fleet(self)
            if report.get("actions"):
                logger.info(
                    "[KG-2.253] reactive autoscale on WorkItem change: evaluated=%s "
                    "actions=%s scaled=%s",
                    report.get("evaluated"),
                    report.get("actions"),
                    report.get("scaled"),
                )
        except Exception as e:  # noqa: BLE001 — one job's failure never stops others
            logger.debug("fleet_autoscale_reactive tick error: %s", e)

    def _placement_mining_subscription(self) -> Any:
        """Lazily-built reactive ``:ToolCall`` change-feed (report §9 #6, X-5).

        One subscription per daemon process, cached on the engine (mirrors
        :meth:`_fleet_autoscale_subscription`), so the reactive placement-mining
        tick fires on the engine's pushed ``:ToolCall`` change-event rather than
        waiting out the giant hourly Loop-engine cycle. Rebuilt if it couldn't
        resolve a streaming surface on first use.
        """
        sub = getattr(self, "_placement_mining_sub", None)
        if sub is None or not getattr(sub, "available", False):
            from agent_utilities.knowledge_graph.research.placement_mining import (
                placement_mining_subscription,
            )

            sub = placement_mining_subscription(self)
            self._placement_mining_sub = sub
        return sub

    def _tick_placement_mining_reactive(self) -> None:
        """Fire a placement-mining cycle on a new ``:ToolCall`` change (report §9 #6, X-5).

        The trigger half of X-5: poll the engine's ``:ToolCall`` change-feed
        (non-blocking, O(new changes)) and run one
        :func:`~agent_utilities.knowledge_graph.research.placement_mining.
        placement_control_loop` pass ONLY when the engine pushed a new
        ``:ToolCall`` since the last poll — closing "nothing calls
        run_placement_mining_cycle automatically" (report's exact finding).
        This tick's OWN scheduling is unconditional/default-on (Native by
        default — see its registration in ``_register_maintenance_schedules``);
        ``placement_control_loop`` still resolves its existing
        ``PLACEMENT_CONTROL_LOOP_ENABLED`` gate internally on every call, so a
        deployment that hasn't opted in pays only the cheap poll — the mining/
        canary pass itself never runs until that flag is set, unchanged from
        before this trigger existed. A no-op when the engine has no streaming
        surface (the existing manual/Loop-cycle triggers cover it). Leader-only
        via the consolidated maintenance scheduler.
        """
        try:
            sub = self._placement_mining_subscription()
            if not sub.available:
                return
            sub.poll(block_ms=0)
            if sub.pending_state["pending"] == 0:
                return
            sub.pending_state["pending"] = 0

            from agent_utilities.knowledge_graph.research.placement_mining import (
                placement_control_loop,
            )

            report = placement_control_loop(self)
            if report.get("enabled") and (
                report.get("persisted") or report.get("applied")
            ):
                logger.info(
                    "[X-5] reactive placement mining on ToolCall change: "
                    "proposals=%s persisted=%s applied=%s",
                    report.get("proposals"),
                    report.get("persisted"),
                    report.get("applied"),
                )
        except Exception as e:  # noqa: BLE001 — one job's failure never stops others
            logger.debug("placement_mining_reactive tick error: %s", e)

    def _get_host_token(self) -> str:
        """Opaque process identity for WorkItem lease ownership.

        Native lease expiry/fencing distinguishes a dead process from a live
        holder. Persisting a host name or operating-system identity is neither
        necessary nor permitted by the privacy contract.
        """
        tok = getattr(self, "_host_token_cache", None)
        if tok is None:
            from agent_utilities.orchestration.agent_dispatch_worker import (
                worker_token,
            )

            tok = worker_token()
            self._host_token_cache = tok
        return tok

    def _deps_state(self, deps: list[str]) -> str:
        """Read dependency state exclusively from their WorkItems."""
        if not deps:
            return "ready"
        from agent_utilities.orchestration import work_item as _wi

        broken = {
            "failed",
            "dead_letter",
            "cancelled",
        }
        all_done = True
        for dep in deps:
            item = _wi.get_work_item(
                self._work_item_engine, _wi.ingest_task_work_item_id(dep)
            )
            state = (item or {}).get("status")
            if state in broken:
                return "broken"
            if state != "succeeded":
                all_done = False
        return "ready" if all_done else "waiting"

    def _tick_file_watch(self) -> None:
        """One SDD/skills/scholarx/config file-watch scan (CONCEPT:AU-KG.research.research-pipeline-runner / OS-5.0).

        Replaces the former dedicated ``KGPlanWatcherThread``: a single
        synchronous ``run_watcher_scan`` pass, run by the consolidated
        maintenance scheduler behind the shared foreground-throttle gate so it
        no longer floods ingestion on startup or competes with interactive runs.
        """
        try:
            from agent_utilities.sdd.watcher import (
                get_workspace_path,
                run_watcher_scan,
            )

            run_watcher_scan(self, get_workspace_path())
        except Exception as e:  # noqa: BLE001 — one job's failure never stops others
            logger.debug("file_watch tick failed (%s)", type(e).__name__)

    def _record_queue_telemetry(self, queue_size: int) -> None:
        """Publish ingest queue depth (+ Kafka consumer lag) as Prometheus
        gauges on the OS-5.23 gateway metrics registry (CONCEPT:AU-KG.ingest.decoupled-kg-ingest-consumer).

        No-op-cheap: without ``prometheus_client`` the gauges are shared no-ops.
        Sampled by the maintenance scheduler on the leader host (the process
        that also serves ``GET /metrics``); for Kafka the backend's queue size
        IS the ``kg-ingest`` group lag, recorded under both names so dashboards
        can alert on lag without knowing the selected backend.
        """
        try:
            from agent_utilities.observability.gateway_metrics import (
                KG_INGEST_CONSUMER_LAG,
                KG_INGEST_QUEUE_DEPTH,
            )

            backend_name = getattr(self, "_task_queue_backend_name", "sqlite")
            KG_INGEST_QUEUE_DEPTH.labels(backend=backend_name).set(float(queue_size))
            if backend_name == "kafka":
                from .kafka_queue_backend import INGEST_GROUP, TASKS_TOPIC

                KG_INGEST_CONSUMER_LAG.labels(
                    topic=TASKS_TOPIC, group=INGEST_GROUP
                ).set(float(queue_size))
        except Exception:  # noqa: BLE001 — telemetry must never break the loop
            pass

    def _maintenance_scheduler_loop(self) -> None:
        """Single thread running all periodic KG maintenance jobs.

        Replaces the former per-job daemon threads (analysis / compaction /
        evolution / enrichment). One backend-readiness check and one
        foreground-throttle gate guard every job, so background work uniformly
        yields the GPU/LLM to interactive runs. (CONCEPT:AU-KG.compute.registered-edge-type / KG-2.8)

        Tick classification (CONCEPT:AU-OS.state.cross-host-daemon-leadership): every job in this scheduler is
        **leader-only**. The remaining inline scheduler tick must have one fleet
        owner so N hosts do not enqueue the same recurring work.
        With ``state_db_uri`` set, a Postgres advisory lock elects exactly one
        leader fleet-wide; followers idle here and still contribute **per-host**
        capacity (task workers + submission/graph-writer queue drains, whose
        claims are cross-host atomic — CONCEPT:AU-KG.ingest.cross-host-safe-kg). Under the SQLite
        default ``is_leader()`` is always true (flock already enforces a single
        per-host daemon).
        """
        import time

        from agent_utilities.core.leadership import get_leadership

        jobs = self._maintenance_jobs()
        if not jobs:
            return
        names = ", ".join(n for n, _, _ in jobs)
        logger.info("KG maintenance scheduler started with jobs: %s", names)

        POLL = 5.0
        leadership = get_leadership("kg-maintenance")
        # Stagger first runs so a startup burst doesn't fire everything at once.
        last_run = {name: time.time() - interval + 15.0 for name, interval, _ in jobs}

        while True:
            try:
                if not getattr(self, "backend", None):
                    time.sleep(10.0)
                    continue

                # Leader-only gate (CONCEPT:AU-OS.state.cross-host-daemon-leadership): non-leader hosts skip all
                # singleton maintenance ticks and re-check for fail-over.
                if not leadership.is_leader():
                    time.sleep(10.0)
                    continue

                # Backpressure visibility (CONCEPT:AU-KG.ingest.decoupled-kg-ingest-consumer): sample the durable
                # submission-queue depth every pass so depth (and, for Kafka,
                # kg-ingest consumer lag) lands on the OS-5.23 gateway Prometheus
                # registry — including under load, exactly when it matters most.
                q = getattr(self, "_submission_queue", None)
                if q is not None:
                    try:
                        self._record_queue_telemetry(q.get_queue_size())
                    except Exception:  # noqa: BLE001 — queue probe best-effort
                        pass

                # This loop now runs ONLY the scheduler, including stale-tick
                # collapse (CONCEPT:AU-OS.state.stale-tick-collapse). Native
                # ClaimWorkItem owns lease recovery, dependency release, and
                # delayed availability. Unlike the heavy job *bodies* it
                # enqueues, scheduler plumbing must run even when workers are
                # saturated. It is therefore
                # deliberately NOT gated by the foreground throttle or a
                # bulk-ingest auto-defer: gating it was the regression that let a
                # stale-tick backlog and dead-worker leases pile up *precisely*
                # while ingestion was busy and the queue most needed healing.

                now = time.time()
                for name, interval, tick in jobs:
                    if now - last_run[name] < interval:
                        continue
                    logger.info("[maint-loop] running job %r", name)
                    try:
                        tick()
                        logger.info("[maint-loop] job %r done", name)
                    except Exception as e:  # one job's failure never stops others
                        logger.error("Maintenance job '%s' error: %s", name, e)
                    last_run[name] = time.time()
                time.sleep(POLL)
            except Exception as e:
                logger.error(f"MaintenanceScheduler error: {e}")
                time.sleep(30.0)

    def _card_store(self) -> Any:
        """Lazy, process-wide persistent card cache (keyed by ast_hash) so identical
        code is LLM-summarised once across runs/repos. Best-effort → ``None`` on
        failure. Engine-only: routes to ``:CardCache`` nodes on the one engine
        authority (no SQLite fallback). (CONCEPT:AU-KG.coordination.embedder-breaker/KG-2.244)
        """
        store = getattr(self, "_card_store_inst", None)
        if store is None:
            try:
                from ..enrichment.cards import CardStore

                store = CardStore(backend=getattr(self, "backend", None))
            except Exception:  # noqa: BLE001 - cache is best-effort
                store = None
            self._card_store_inst = store
        return store

    def _tick_enrichment(self) -> None:
        """One Phase-2 enrichment tick: backfill LLM capability cards onto
        structurally-ingested ``Code`` nodes whose ``summary`` is still empty.

        CONCEPT:AU-KG.coordination.embedder-breaker. Cards are cached by ``ast_hash`` so unchanged code is
        never re-summarised; only non-empty summaries are written back (so a
        transient LLM outage doesn't poison nodes). Drains up to
        ``KG_ENRICH_MAX_BATCHES`` batches per tick, re-checking the foreground
        throttle between batches so it yields promptly to interactive runs.
        """
        import json
        import time

        backend = getattr(self, "backend", None)
        if not backend:
            return
        if not hasattr(self, "_enrich_card_cache"):
            self._enrich_card_cache: dict[str, Any] = {}

        # Circuit breaker: while OPEN (a down card LLM), this tick is a cheap no-op so a
        # broken endpoint is never retry-stormed (CONCEPT:AU-KG.enrichment.card-attempt-status).
        now = time.monotonic()
        if self._card_circuit_open(now):
            return

        from ..enrichment.cards import (
            generate_symbol_cards,
            make_lite_llm_fn,
            make_llm_fn,
        )
        from ..enrichment.models import CodeEntity

        BATCH = _ENRICH_BATCH
        MAX_BATCHES = _ENRICH_MAX_BATCHES
        max_workers = compute_ingest_worker_count()
        llm_fn = getattr(self, "_enrich_llm_fn", None)

        for _ in range(MAX_BATCHES):
            # Yield between batches to interactive runs AND to a bulk ingest.
            try:
                from agent_utilities.core.background_throttle import get_throttle

                if get_throttle().should_yield_background:
                    return
            except ImportError as exc:  # noqa: BLE001 — ImportError-guarded optional background-throttle signal; when unavailable the tick proceeds without the early-yield check rather than the enrichment batch being lost
                logger.debug(
                    "background-throttle check skipped (optional dependency): %s", exc
                )

            # Select nodes STILL needing a card: no summary AND not yet attempted
            # (``card_status`` is set to 'ok'/'skip' once a node is resolved, so a trivial
            # or genuinely-un-summarizable symbol drops out and the window ADVANCES instead
            # of re-fetching the same rows forever). A transient LLM failure leaves
            # card_status unset, so it is retried next tick — governed by the breaker.
            # ORDER BY makes the scan deterministic. (CONCEPT:AU-KG.enrichment.card-attempt-status)
            rows = self.query_cypher(
                "MATCH (n:Code) WHERE n.summary = '' AND n.ast_hash IS NOT NULL "
                "AND n.card_status IS NULL "
                "RETURN n.id AS id, n.name AS name, n.kind AS kind, "
                "n.file_path AS file_path, n.patterns AS patterns, "
                "n.language AS language, n.ast_hash AS ast_hash "
                "ORDER BY n.id LIMIT " + str(BATCH)
            )
            if not rows:
                return
            if llm_fn is None:
                # Card summaries are a structured extraction task — route to the
                # LITE chat model by default (markedly faster than the heavy KG
                # model, which is what saturated the engine on a full backfill).
                # ``KG_CARD_MODEL=heavy`` forces the heavy model. (CONCEPT:AU-KG.coordination.embedder-breaker)
                from agent_utilities.core.config import setting

                use_heavy = setting("KG_CARD_MODEL", "lite").lower() == "heavy"
                llm_fn = make_llm_fn() if use_heavy else make_lite_llm_fn()
                self._enrich_llm_fn = llm_fn

            ents = [
                CodeEntity(
                    id=r["id"],
                    name=r.get("name") or r["id"],
                    qualname=r.get("name") or r["id"],
                    kind=r.get("kind") or "function",
                    language=r.get("language") or "",
                    file_path=r.get("file_path") or "",
                    line=0,
                    ast_hash=r.get("ast_hash") or "",
                    patterns=[p for p in (r.get("patterns") or "").split(",") if p],
                )
                for r in rows
            ]
            # Respect the global background throttle: skip this tick if foreground
            # (interactive) work is active, and cap concurrent background LLM load
            # via the shared semaphore so card backfill can't saturate the engine
            # (CONCEPT:AU-KG.compute.registered-edge-type). The per-batch foreground check above stays as a
            # fast-path; this adds the concurrency cap shared with other daemons.
            from agent_utilities.core.background_throttle import get_throttle

            with get_throttle().background_slot(wait_foreground=False) as slot:
                if not slot:
                    return
                cards = generate_symbol_cards(
                    ents,
                    llm_fn,
                    cache=self._enrich_card_cache,
                    max_workers=max_workers,
                    store=self._card_store(),
                )
            written = 0
            attempted = (
                0  # nodes resolved this batch (ok + skip) — real forward progress
            )
            failed = 0  # transient LLM failures — NOT marked done, retried next tick
            for card in cards:
                status = getattr(card, "status", "ok" if card.summary else "skip")
                if status == "failed":
                    failed += 1
                    continue
                try:
                    if status == "ok" and card.summary:
                        backend.execute(
                            "MATCH (n:Code {id: $id}) SET n.summary = $summary, "
                            "n.responsibilities = $resp, n.card_status = 'ok'",
                            {
                                "id": card.id,
                                "summary": card.summary,
                                "resp": json.dumps(card.responsibilities),
                            },
                        )
                        written += 1
                    else:
                        # PERMANENT empty (trivial accessor / un-summarizable): mark
                        # 'skip' so it never re-selects (fixes the never-100% stall).
                        backend.execute(
                            "MATCH (n:Code {id: $id}) SET n.card_status = 'skip'",
                            {"id": card.id},
                        )
                    attempted += 1
                except Exception:
                    logger.debug("card writeback failed for %s", card.id, exc_info=True)
            logger.info(
                "KG enrichment: %d summarized, %d skipped, %d failed (of %d)",
                written,
                attempted - written,
                failed,
                len(rows),
            )
            # Breaker: a batch that produced ONLY failures (no node resolved) signals a
            # down LLM → count toward opening the circuit and stop this tick. Any forward
            # progress (a summary OR a skip mark) resets it. Because attempted nodes are
            # now marked, an all-trivial window no longer freezes the tick (the old
            # ``written == 0 → return`` bug); only a genuine outage stops it.
            if attempted == 0 and failed > 0:
                self._card_circuit_record(False, now)
                return
            self._card_circuit_record(True, now)

    # Candidate text columns used to build embedding input, in priority order.
    _EMBED_TEXT_COLS = (
        "name",
        "title",
        "summary",
        "description",
        "content",
        "qualname",
    )

    def _embed_circuit_open(self, now: float) -> bool:
        """True while the embedder circuit breaker is OPEN (skip embed work).

        CONCEPT:AU-KG.coordination.embedder-breaker — keeps a down embedder from being retry-stormed.
        """
        return getattr(self, "_embed_cb_open_until", 0.0) > now

    def _embed_circuit_record(self, success: bool, now: float) -> None:
        """Record an embed attempt outcome; open the breaker after repeated fails."""
        if success:
            self._embed_cb_failures = 0
            self._embed_cb_open_until = 0.0
            return
        fails = int(getattr(self, "_embed_cb_failures", 0)) + 1
        self._embed_cb_failures = fails
        if fails >= _EMBED_CB_THRESHOLD:
            self._embed_cb_open_until = now + _EMBED_CB_COOLDOWN
            logger.warning(
                "embed backfill: embedder unhealthy (%d consecutive failures) — "
                "circuit OPEN for %.0fs (skipping embed work to avoid a retry-storm)",
                fails,
                _EMBED_CB_COOLDOWN,
            )

    def _card_circuit_open(self, now: float) -> bool:
        """True while the card-enrichment breaker is OPEN (skip card work).

        CONCEPT:AU-KG.enrichment.card-attempt-status — keeps a down card LLM from being retry-stormed.
        """
        return getattr(self, "_card_cb_open_until", 0.0) > now

    def _card_circuit_record(self, success: bool, now: float) -> None:
        """Record a card-backfill batch outcome; open the breaker after repeated all-fail
        batches, reset it on any forward progress."""
        if success:
            self._card_cb_failures = 0
            self._card_cb_open_until = 0.0
            return
        fails = int(getattr(self, "_card_cb_failures", 0)) + 1
        self._card_cb_failures = fails
        if fails >= _CARD_CB_THRESHOLD:
            self._card_cb_open_until = now + _CARD_CB_COOLDOWN
            logger.warning(
                "card backfill: card LLM unhealthy (%d consecutive all-fail batches) — "
                "circuit OPEN for %.0fs (skipping card work to avoid a retry-storm)",
                fails,
                _CARD_CB_COOLDOWN,
            )

    # CONCEPT:AU-KG.compute.per-channel-embedding-backfill — Per-channel embedding backfill: round-robin unembedded
    # nodes by source_system + fan out to embedding capacity, so a tiny url/doc
    # crawl's chunks aren't FIFO-starved behind millions of codebase chunks.
    # Per-table source-rotation cursors: which channel leads next tick.
    _EMBED_SOURCE_CURSORS: dict[str, int] = {}  # noqa: RUF012

    def _collect_unembedded_rows(
        self, conn_factory: Any, tbl: str, take: int
    ) -> list[tuple[Any, str]]:
        """Pull up to ``take`` NULL-embedding ``(id, text)`` rows from ``tbl``,
        round-robin across ingestion *channels* (``source_system``).

        CONCEPT:AU-KG.compute.per-channel-embedding-backfill — a single ``WHERE embedding IS NULL LIMIT n`` FIFO lets
        one huge channel (822K codebase ``Code`` chunks) starve a small url/doc
        crawl's chunks that share the table. Instead, for a table that carries a
        ``source_system`` column we find the distinct channels with unembedded
        rows and give each a slice of the budget every tick, rotating which
        channel leads (so the per-channel remainder is shared fairly over time).
        Tables without ``source_system`` (internal codebase writes) fall back to
        the plain bounded scan. Interpreter-safe: only equality filters + LIMIT, no ORDER
        BY (the interpreter strips ORDER BY) — exactly like the lane claim.
        """
        tbl = _require_database_identifier(tbl)
        with conn_factory() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = %s",
                (tbl,),
            )
            cols = {r[0] for r in cur.fetchall()}
            text_cols = [c for c in self._EMBED_TEXT_COLS if c in cols]
            if not text_cols or "embedding" not in cols:
                return []
            expr = " || ' ' || ".join(f"COALESCE(\"{c}\",'')" for c in text_cols)
            has_source = "source_system" in cols

            unfiltered = object()

            def _fetch(source_system: object, limit: int) -> list[tuple[Any, str]]:
                if source_system is unfiltered:
                    source_clause = ""
                    params: tuple[object, ...] = (limit,)
                elif source_system is None:
                    source_clause = " AND source_system IS NULL"
                    params = (limit,)
                else:
                    source_clause = " AND source_system = %s"
                    params = (str(source_system), limit)
                cur.execute(
                    # Both identifiers are selected from fixed allowlists and
                    # ``tbl`` was validated by ``_require_database_identifier``.
                    f'SELECT id, {expr} FROM "{tbl}" '  # nosec B608
                    f"WHERE embedding IS NULL{source_clause} LIMIT %s",
                    params,
                )
                got = [(r[0], (r[1] or "").strip()) for r in cur.fetchall()]
                return [(nid, txt) for nid, txt in got if txt]

            if not has_source:
                return _fetch(unfiltered, take)

            # Distinct channels that still have unembedded rows (bounded scan, no
            # GROUP BY ORDER BY — equality-friendly DISTINCT only).
            cur.execute(
                # ``tbl`` was validated by ``_require_database_identifier``.
                f'SELECT DISTINCT source_system FROM "{tbl}" '  # nosec B608
                "WHERE embedding IS NULL LIMIT 64"
            )
            channels = sorted(
                str(r[0]) if r[0] is not None else "" for r in cur.fetchall()
            )
            if len(channels) <= 1:
                # One (or zero) channel — nothing to round-robin; plain scan.
                return _fetch(unfiltered, take)

            # Rotate which channel leads this tick (fair sharing of the remainder).
            cur_idx = self._EMBED_SOURCE_CURSORS.get(tbl, 0) % len(channels)
            self._EMBED_SOURCE_CURSORS[tbl] = (cur_idx + 1) % len(channels)
            ordered_ch = channels[cur_idx:] + channels[:cur_idx]

            per_channel = max(1, take // len(ordered_ch))
            items: list[tuple[Any, str]] = []
            for ch in ordered_ch:
                if len(items) >= take:
                    break
                slot = min(per_channel, take - len(items))
                if ch == "":
                    rows = _fetch(None, slot)
                else:
                    rows = _fetch(ch, slot)
                items.extend(rows)
            return items[:take]

    def _tick_embedding_backfill_generic(self) -> int:
        """Non-pgvector fallback for :meth:`_tick_embedding_backfill` (D-EMB).

        This "dedicated vector-embedding backfill drain" thread has been
        running in production the whole time — but ``_tick_embedding_backfill``
        below returns 0 immediately on any backend that isn't
        ``PostgreSQLBackend`` (checks ``pgvector_available``/``_conn``/
        ``_get_embedding_tables``, all pgvector-only attributes), so on the
        native-engine (Ladybug/redb, ``BrainGuardedBackend``) topology this
        production actually runs on, the daemon has been a silent, permanent
        no-op since it started. That is the primary mechanism behind the
        measured 0.5% embedding coverage (136/26,680 nodes, D-PERF-5): a
        backfill daemon that LOOKS like it's running (thread alive, ticking
        every 30s) but never does anything on this deployment's backend.

        This reconciles nodes that already carry an ``embedding`` PROPERTY
        (written by the ingest-time chokepoint in ``ingestion/envelope_ingest.py``,
        or by any other writer) into the engine's ANN/HNSW index — a SEPARATE
        store the property write alone does not populate (see
        ``epistemic_graph_backend.add_embedding``'s docstring: "distinct from
        storing an embedding node property"; ``semantic_search`` reads the ANN
        index, not the property). Rate-limited to
        :data:`_EMBED_BACKFILL_GENERIC_INTERVAL_S` — see that constant's
        comment for why.

        Deliberately does NOT generate new embeddings for the many legacy
        nodes that have no ``embedding`` property at all: that needs an
        embedding-endpoint call per node, touches classification/ACL policy if
        done through the governed ChangeEnvelope write path, and is an
        explicit, operator-approved backfill (``scripts/backfill_embeddings.py``),
        never a silent background daemon.
        """
        target = self.backend
        hydrate = getattr(target, "hydrate_engine_embeddings", None)
        if not callable(hydrate):
            return 0
        now = time.monotonic()
        last = getattr(self, "_last_generic_embed_hydrate", 0.0)
        if now - last < _EMBED_BACKFILL_GENERIC_INTERVAL_S:
            return 0
        self._last_generic_embed_hydrate = now
        try:
            count = int(hydrate())
        except Exception as e:  # noqa: BLE001 — best-effort reconciliation tick; a failure here must never kill the daemon loop, only skip this pass
            logger.debug("generic embedding-index hydrate failed: %s", e)
            return 0
        if count:
            logger.info(
                "KG embedding-index hydrate: indexed %d node(s) into the ANN store",
                count,
            )
        return count

    def _tick_embedding_backfill(self) -> int:
        """Backfill vector embeddings onto configured pgvector nodes that lack them.

        Vector features (semantic_search, concept→code RELATES_TO linking,
        designation, latent retrieval) need embeddings on pgvector node
        tables, but the structural codebase pass and concept extraction create
        nodes WITHOUT embeddings. This embeds unembedded rows incrementally in
        bounded batches with the configured model, behind the shared foreground
        gate. Idempotent (only ``embedding IS NULL`` rows). (CONCEPT:AU-KG.coordination.embedder-breaker)
        """
        target = self.backend
        conn_factory = getattr(target, "_conn", None)
        get_tables = getattr(target, "_get_embedding_tables", None)
        if (
            not callable(conn_factory)
            or not callable(get_tables)
            or not getattr(target, "pgvector_available", False)
        ):
            # D-EMB: not a pgvector backend — try the native-engine fallback
            # instead of unconditionally returning 0 (see
            # `_tick_embedding_backfill_generic`'s docstring for why this
            # branch was previously a silent permanent no-op in production).
            return self._tick_embedding_backfill_generic()

        budget = _EMBED_BACKFILL_BUDGET

        tables = get_tables()
        if not tables:
            return 0
        # Retrieval-critical labels first, then the rest.
        prio = ["Code", "Concept", "Document", "Feature", "Skill", "Message"]
        ordered = [t for t in prio if t in tables] + [
            t for t in tables if t not in prio
        ]

        from ..enrichment.semantic import make_embed_fn

        embed_fn = getattr(self, "_backfill_embed_fn", None)
        if embed_fn is None:
            embed_fn = make_embed_fn()
            self._backfill_embed_fn = embed_fn

        # Circuit breaker: while OPEN (embedder recently failed repeatedly), skip
        # all embed work so we don't retry-storm a dead endpoint and peg the daemon.
        import time as _t

        now = _t.monotonic()
        if self._embed_circuit_open(now):
            return 0

        # Fair per-table share so retrieval-critical labels (e.g. Concept) aren't
        # starved behind a huge table (e.g. Code). Each table gets up to
        # ``per_table`` rows per tick, still bounded by the total budget.
        per_table = max(16, budget // max(1, len(ordered)))
        total = 0
        remaining = budget
        for tbl in ordered:
            if remaining <= 0:
                break
            take = min(per_table, remaining)
            try:
                items = self._collect_unembedded_rows(conn_factory, tbl, take)
            except Exception as e:  # noqa: BLE001 — one table's row-selection query for this backfill tick; `continue`s to the next table so a single bad table doesn't stop the whole backfill pass, and nothing is marked embedded for the skipped table
                logger.debug("embed backfill query failed: %s", e)
                continue

            if not items:
                continue
            try:
                # KG-2.144: fan the embed calls out to the embedding model's
                # parallel capacity — ``make_embed_fn`` batches at 64 and runs up
                # to ``capacity`` batches concurrently via the shared controller,
                # so with capacity 1 it stays sequential and with K it does K at
                # once (scales with the number of vLLM instances). Same nodes,
                # same vectors, idempotent.
                vecs = embed_fn([t for _, t in items])
                with conn_factory() as conn, conn.cursor() as cur:
                    for (nid, _), vec in zip(items, vecs, strict=False):
                        cur.execute(
                            # ``tbl`` was validated by
                            # ``_require_database_identifier``.
                            f'UPDATE "{tbl}" SET embedding = %s::vector '  # nosec B608
                            "WHERE id = %s AND embedding IS NULL",
                            (str(vec), nid),
                        )
                    conn.commit()
                total += len(items)
                remaining -= len(items)
                self._embed_circuit_record(True, now)  # healthy → close breaker
            except Exception as e:  # noqa: BLE001 — already the safe direction: `total`/`remaining` are only advanced above this except (never inside it), and the circuit breaker is explicitly recorded False + the tick breaks rather than continuing to hammer a down endpoint
                logger.debug("embed backfill store failed: %s", e)
                # An embed/store failure means the endpoint is likely down for
                # every table — record it and stop hammering the rest this tick.
                self._embed_circuit_record(False, now)
                break
        if total:
            logger.info("KG embedding backfill: embedded %d nodes", total)
        return total

    def _tick_loop(self) -> None:
        """One propose-only self-evolution cycle (CONCEPT:AU-KG.compute.registered-edge-type).

        Runs ``LoopController.run_one_cycle`` (intake active Loops →
        acquire related sources → ADDRESSES resolve → optional distill/synthesize
        as DRAFTS/proposals). Always propose-only: nothing is auto-merged or
        executed. Throttled + opt-in via ``KG_LOOP``.
        """
        try:
            from agent_utilities.core.config import config as _cfg

            from ..research.loop_controller import LoopController

            rep = LoopController(self).run_one_cycle(max_topics=_cfg.kg_loop_topics)
            logger.info(
                "Loop cycle: intake=%s resolved=%s sources=%s team=%s",
                rep.get("topics_intake"),
                rep.get("topics_resolved"),
                rep.get("sources_linked"),
                bool(rep.get("team")),
            )
        except Exception as e:  # noqa: BLE001
            logger.error("loop tick error: %s", e)

    def _world_model_subscription(self) -> Any:
        """Lazily-built reactive ``WorldModelTransition`` change-feed (CONCEPT:AU-KG.compute.reactive-push).

        One subscription per daemon process, cached on the engine, so the SAI tick
        consumes the engine's pushed change-events instead of re-scanning the whole
        transition history every tick. Rebuilt transparently if it couldn't resolve
        a streaming surface on first use (engine not yet connected).
        """
        sub = getattr(self, "_wm_subscription", None)
        if sub is None or not getattr(sub, "available", False):
            from agent_utilities.harness.world_model_task import (
                world_model_subscription,
            )

            sub = world_model_subscription(self)
            self._wm_subscription = sub
        return sub

    def _tick_sai_factory(self) -> None:
        """One SAI-factory world-model specialization cycle (CONCEPT:AU-AHE.harness.sai-controller).

        REACTIVE (CONCEPT:AU-KG.compute.reactive-push): instead of re-querying the ENTIRE
        ``WorldModelTransition`` history every tick, poll the engine's change-feed
        subscription and only re-specialize when the engine pushed a NEW transition
        since the last tick (the change that caused it) — or on cold-start
        catch-up. When the engine has no streaming surface (``available`` False),
        fall back to the periodic specialization so behaviour is never worse than
        before. Grounds a learned dynamics model in the transition history and
        persists a ``SaiFactoryCycle`` node. AU-native (no LLM/GPU). Throttled +
        opt-in via ``KG_SAI_FACTORY``; a no-op when too little history exists.
        """
        try:
            from agent_utilities.harness.superhuman_gate import SuperhumanCertifier
            from agent_utilities.harness.world_model_task import (
                specialize_world_model_from_engine,
            )

            sub = self._world_model_subscription()
            if sub.available:
                # Non-blocking poll: O(new transitions) on the engine's pushed
                # feed, NOT a full re-scan. Skip the expensive specialization when
                # nothing changed since the last tick.
                sub.poll(block_ms=0)
                pending = sub.pending_state["pending"]
                if pending == 0:
                    logger.debug("SAI factory tick: no new transitions (reactive skip)")
                    return
                sub.pending_state["pending"] = 0

            summary = specialize_world_model_from_engine(
                self, certifier=SuperhumanCertifier()
            )
            if summary is None:
                logger.debug("SAI factory tick: insufficient transition history")
                return
            logger.info(
                "SAI factory cycle: task=%s reward=%s reached=%s transitions=%s",
                summary.get("task_id"),
                summary.get("final_specialist_reward"),
                summary.get("reached"),
                summary.get("transitions"),
            )
        except Exception as e:  # noqa: BLE001
            logger.error("sai_factory tick failed: %s", e)

    def _tick_failure_ingest(self) -> None:
        """Ingest Langfuse failures → gap topics → regression-gated remediation.

        Pulls error/low-score/cost-latency telemetry from Langfuse, materializes
        ``ExecutionSummary`` / ``PerformanceAnomaly`` nodes and synthetic
        ``failure_gap`` ``Concept`` topics, then — when new gaps appear — runs one
        golden-loop cycle whose auto-merge is gated by a regression check bound to
        those failures. Opt-in via KG_FAILURE_EVOLUTION (CONCEPT:AU-AHE.harness.failure-evolution).
        """
        try:
            from ..adaptation.failure_analyzer import run_failure_ingest

            report = run_failure_ingest(self)
            logger.info(
                "Failure ingest: pulled=%s patterns=%s gaps=%s anomalies=%s",
                report.get("records_pulled"),
                report.get("patterns"),
                len(report.get("gap_concepts", [])),
                report.get("anomalies"),
            )
        except Exception as e:  # noqa: BLE001
            logger.error("failure_ingest tick error: %s", e)

    def _tick_optimize_components(self) -> None:
        """Propose-only optimization sweep over the self-supervised targets.

        The scheduled twin of ``graph_evolution action=optimize_component`` (CONCEPT:
        AHE-3.46): gathers live graph data and runs the extraction / concept_match /
        routing optimizers, recording optimization trajectories. Nothing is auto-applied —
        promotion stays behind ``should_promote`` and a future auto-apply gate. Default ON
        via KG_OPTIMIZATION_ENABLED. The provider-free native program job is the
        sole execution contract.
        """
        try:
            from ...harness.program_optimization import run_optimization_sweep

            report = run_optimization_sweep(self)
            if report.get("failed"):
                logger.error(
                    "Optimization sweep tick: FAILURES=%s optimized=%s "
                    "duration=%ss (propose-only)",
                    report.get("failed"),
                    report.get("optimized"),
                    report.get("duration_s"),
                )
            else:
                logger.info(
                    "Optimization sweep tick: optimized=%s duration=%ss (propose-only)",
                    report.get("optimized"),
                    report.get("duration_s"),
                )
        except Exception as e:  # noqa: BLE001
            logger.error("optimization tick error: %s", e)

    def _tick_anomaly_consumer(self) -> None:
        """Drain unconsumed PerformanceAnomaly nodes into failure_gap topics.

        One bounded consumer pass (CONCEPT:AU-AHE.optimization.performance-anomaly-consumer): clusters fresh anomalies
        by target + type, files one failure_gap Concept per cluster through the
        failure analyzer's shared gap-topic path (so the golden loop's intake
        remediates them), and stamps every scanned anomaly ``consumed``.
        Propose-only and LLM-free; on by default via KG_ANOMALY_CONSUMER.
        """
        try:
            from ..adaptation.anomaly_consumer import consume_anomalies

            report = consume_anomalies(self)
            if report.get("scanned"):
                logger.info(
                    "Anomaly consumer: scanned=%s gaps=%s consumed=%s",
                    report.get("scanned"),
                    report.get("gaps_filed"),
                    report.get("consumed"),
                )
        except Exception as e:  # noqa: BLE001
            logger.error("anomaly_consumer tick error: %s", e)

    def _tick_tms_revalidation(self) -> None:
        """Revalidate every TMS materialization the engine has marked stale.

        One bounded consumer pass (CONCEPT:EG-KG.epistemic.truth-maintenance, Seam 3
        follow-up, W3.2): reads the engine's durable ``stale_materializations``
        projection, and for each stale mined claim / capability-index entry /
        cached context bundle, routes to that owner's re-validation action —
        propose a ``:BeliefRevisionProposal`` for a claim (never mutates the
        claim itself), evict a capability-index cache row, or drop a context
        bundle from the KV cache. Stateless and propose-only; on by default,
        background priority.
        """
        try:
            from ..adaptation.tms_revalidation import revalidate_stale_materializations

            report = revalidate_stale_materializations(self)
            if report.get("stale"):
                logger.info(
                    "TMS revalidation: scanned=%s stale=%s revalidated=%s",
                    report.get("scanned"),
                    report.get("stale"),
                    report.get("revalidated"),
                )
        except Exception as e:  # noqa: BLE001
            logger.error("tms_revalidation tick error: %s", e)

    def _tick_runtime_reliability(self) -> None:
        """One runtime-reliability detect→gap pass (CONCEPT:AU-AHE.harness.runtime-reliability-loop).

        Drains the hot-path runtime-signal buffer, persists it as :RuntimeSignal nodes,
        aggregates recent signals by (kind, subject), and for a pattern crossing threshold
        OPENS a SOURCE_RUNTIME :Gap through the canonical flywheel — or, for a recognized
        class, records a resolved heal (listener_restart, already auto-healed by the
        messaging supervisor) or a config/perf recommendation gap (engine_latency /
        retrieval_degraded). Bounded, LLM-free, propose-only (never mutates prod); native by
        default, background priority — like ``anomaly_consumer``/``tms_revalidation`` above.
        """
        try:
            from ..research.runtime_reliability import runtime_reliability_analyzer

            report = runtime_reliability_analyzer(self)
            if report.get("patterns"):
                logger.info(
                    "Runtime reliability: scanned=%s patterns=%s gaps=%s "
                    "recommendations=%s heals=%s",
                    report.get("scanned"),
                    report.get("patterns"),
                    report.get("gaps_opened"),
                    report.get("recommendations"),
                    report.get("heals"),
                )
        except Exception as e:  # noqa: BLE001
            logger.error("runtime_reliability tick error: %s", e)

    def _tick_scheduler(self) -> None:
        """Evaluate every durable ``:Schedule`` and ENQUEUE the jobs that are due.

        The ONE scheduler tick (CONCEPT:AU-OS.state.unified-scheduling-one-intelligent): it reads the durable
        ``:Schedule`` registry (seeded from ``deploy/schedules.yml`` plus the
        former fixed-interval maintenance ticks registered programmatically) and
        for every due schedule enqueues a ``scheduled_job`` WorkItem onto the
        unified queue — it does not run any job inline. Cron, interval, and
        adaptive triggers are all handled here. ``/cron calendar`` reads the
        same registry.
        """
        try:
            from agent_utilities.core.schedule_engine import run_scheduler_tick

            # Register the former fixed-interval maintenance ticks as durable
            # :Schedule nodes once (after the backend is ready). Idempotent.
            if not getattr(self, "_maint_schedules_registered", False):
                logger.info("[maint] registering maintenance schedules…")
                self._register_maintenance_schedules()
                self._maint_schedules_registered = True
                logger.info("[maint] maintenance schedules registered")
            result = run_scheduler_tick(self)
            if result.get("fired"):
                logger.info("scheduler fired: %s", result["fired"])
        except Exception as e:  # noqa: BLE001
            logger.error("scheduler tick error: %s", e)

        # Phase-0 daemon telemetry (CONCEPT:AU-ORCH.execution.two-level-fair-rotation): republish the same
        # pending/in-flight snapshot admission control already computes as
        # per-lane gauges, on the scheduler's own 60s cadence. Best-effort and
        # fully isolated from the tick above — never affects scheduling.
        try:
            from agent_utilities.knowledge_graph.core.task_lanes import (
                record_lane_metrics,
            )

            reg = getattr(self, "_worker_reg", None)
            running_by_lane = reg.running_by_lane() if reg is not None else {}
            record_lane_metrics(self._pending_by_lane(), running_by_lane)
        except Exception:  # noqa: BLE001
            logger.debug("scheduler tick: lane metrics failed", exc_info=True)

    def _tick_fuseki_publish(self) -> None:
        """Push the bundled ontology modules to Apache Jena Fuseki.

        One bounded distribution pass (CONCEPT:AU-KG.ontology.authoritative-tbox): merges every shipped
        ``ontology*.ttl`` module and PUTs it to the configured Fuseki dataset
        via :func:`publish_ontology_to_fuseki`, so an optional enterprise
        triplestore stays in sync with the authoritative ontology. Opt-in via
        ``KG_FUSEKI_PUBLISH``; endpoint from ``KG_FUSEKI_ENDPOINT`` (falling
        back to the publisher's own resolution).
        """
        try:
            from agent_utilities.core.config import config as _cfg

            from .ontology_publisher import publish_ontology_to_fuseki

            report = publish_ontology_to_fuseki(endpoint=_cfg.kg_fuseki_endpoint)
            if report.get("status") == "success":
                logger.info(
                    "Fuseki publish: %s triples -> %s/%s",
                    report.get("triple_count"),
                    report.get("endpoint"),
                    report.get("dataset"),
                )
            else:
                logger.warning(
                    "Fuseki publish did not complete: %s",
                    report.get("error") or report.get("reason"),
                )
        except Exception as e:  # noqa: BLE001 — one job's failure never stops others
            logger.error("fuseki_publish tick error: %s", e)

    def _embedding_backfill_loop(self) -> None:
        """Dedicated drain loop for vector-embedding backfill (CONCEPT:AU-KG.coordination.embedder-breaker).

        Runs independently of the periodic maintenance scheduler so it is NOT
        blocked behind slow LLM ticks: it embeds a batch, and if work remains
        (a full batch landed) loops again almost immediately; when the graph is
        fully embedded it idles at a long interval. Yields to interactive runs
        via the shared foreground throttle.

        Leader-only (CONCEPT:AU-OS.state.cross-host-daemon-leadership): two hosts would select the same
        unembedded batch and duplicate embedding work, so only the fleet
        leader drains it.
        """
        import time

        from agent_utilities.core.leadership import get_leadership

        leadership = get_leadership("kg-maintenance")
        batch = _EMBED_BACKFILL_FETCH
        try:
            idle = _EMBED_BACKFILL_IDLE_INTERVAL
        except ValueError:
            idle = 30.0
        busy = _EMBED_BACKFILL_BUSY_SLEEP

        while True:
            try:
                if not getattr(self, "backend", None):
                    time.sleep(idle)
                    continue
                if not leadership.is_leader():
                    time.sleep(idle)
                    continue
                # Yield to interactive/foreground work AND to a bulk ingest — the
                # 512-node embed batch is a prime swamper of a post-restart backlog.
                try:
                    from agent_utilities.core.background_throttle import get_throttle

                    if get_throttle().should_yield_background:
                        time.sleep(busy)
                        continue
                except ImportError as exc:  # noqa: BLE001 — background_throttle is an optional yield-to-foreground hint; when absent the loop just skips the yield and proceeds straight to _tick_embedding_backfill() below, it does not skip or mark any embedding work
                    logger.debug(
                        "background-throttle check skipped (optional dependency): %s",
                        exc,
                    )
                embedded = self._tick_embedding_backfill()
                # Full batch ⇒ likely more to do ⇒ loop fast; else back off.
                time.sleep(busy if embedded >= batch else idle)
            except Exception as e:  # noqa: BLE001 — never let the loop die
                logger.error("EmbeddingBackfillLoop error: %s", e)
                time.sleep(idle)

    def _tick_tenant_gc(self) -> None:
        """Drop leaked per-job community-detection tenants (CONCEPT:AU-KG.coordination.embedder-breaker).

        Structural ingest runs community detection in an ephemeral
        ``{graph}__enrich_comm_{uuid}`` tenant and deletes it in a ``finally`` — but
        a process kill (a daemon redeploy mid-ingest) skips that, leaking the tenant.
        Every leaked tenant is then re-serialized on EVERY checkpoint, which is what
        bloated checkpoint cost into multi-second write freezes. These tenants are
        per-job ephemeral, so when no bulk ingest is in flight they are ALL orphans
        and safe to drop. Only the ``__enrich_comm_`` pattern is touched — never a
        real graph.
        """
        from agent_utilities.core.background_throttle import get_throttle

        if get_throttle().bulk_ingest_active:
            return  # a live ingest may own its comm tenant right now
        backend = getattr(self, "backend", None)
        graph = getattr(backend, "graph", None) or getattr(backend, "_graph", None)
        client = getattr(graph, "_client", None)
        if client is None:
            return
        try:
            tenants = client.tenants.list()
        except Exception:  # noqa: BLE001 — best-effort sweep
            return
        leaked = [
            t["name"]
            for t in tenants
            if isinstance(t, dict) and "__enrich_comm_" in t.get("name", "")
        ]
        deleted = 0
        for name in leaked:
            try:
                client.tenants.delete(name)
                deleted += 1
            except Exception:  # noqa: BLE001 — one failure never stops the sweep
                pass
        if deleted:
            logger.info(
                "tenant GC: dropped %d leaked community tenant(s) (checkpoint sprawl)",
                deleted,
            )

    def _tick_reconcile_mirrors(self) -> None:
        """Repair drift from the engine authority into configured mirrors."""
        from ..backends.fanout_backend import FanOutBackend

        backend = getattr(self, "backend", None)
        fanout = getattr(backend, "inner", backend)
        if not isinstance(fanout, FanOutBackend):
            return
        try:
            reports = fanout.reconcile()
            summaries = [r for r in reports.values() if isinstance(r, dict)]
            nodes_missing = sum(int(r.get("nodes_missing", 0)) for r in summaries)
            edges_missing = sum(int(r.get("edges_missing", 0)) for r in summaries)
            errs = sum(int(r.get("errors", 0)) for r in summaries) + sum(
                1 for r in summaries if "error" in r
            )
            missing = nodes_missing + edges_missing
            if missing or errs:
                logger.warning(
                    "mirror reconcile: drift remains after repair — "
                    "%d nodes / %d edges missing, %d write errors (%s)",
                    nodes_missing,
                    edges_missing,
                    errs,
                    reports,
                )
            else:
                logger.debug("mirror reconcile: all configured mirrors are in sync")
        except Exception as e:  # noqa: BLE001
            logger.warning("mirror reconcile tick failed: %s", e)

    def _tick_compaction(self) -> None:
        """One LCM compaction tick (CONCEPT:AU-KG.memory.tiered-memory-caching).

        Finds ``Thread`` nodes with more than ``COMPACTION_THRESHOLD``
        uncompacted messages and delegates to ``AgentContextManager``. Run by
        the consolidated maintenance scheduler.
        """
        COMPACTION_THRESHOLD = 30
        threads = self.query_cypher(
            "MATCH (t:Thread)-[:CONTAINS]->(m:Message) "
            "WITH t, count(m) AS msg_count "
            "WHERE msg_count > $threshold "
            "AND (t.last_compacted IS NULL) "
            "RETURN t.id AS id, msg_count "
            "ORDER BY msg_count DESC LIMIT 3",
            {"threshold": COMPACTION_THRESHOLD},
        )
        if not threads:
            return
        from agent_utilities.knowledge_graph.memory import AgentContextManager

        ecm = AgentContextManager(max_tokens=32000)
        for thread in threads:
            thread_id = thread.get("id", "")
            msg_count = thread.get("msg_count", 0)
            if not thread_id:
                continue
            try:
                result = ecm.compact_thread(
                    thread_id=thread_id,
                    engine=self,
                    strategy="progressive",
                    compaction_threshold=COMPACTION_THRESHOLD,
                )
                logger.info(
                    "Compaction: thread %s (%d msgs) → %s",
                    thread_id,
                    msg_count,
                    result.get("status", "unknown"),
                )
            except Exception as e:
                logger.warning(f"Compaction: failed to compact {thread_id}: {e}")

    def _tick_evolution(self) -> None:
        """One research-evolution cycle tick (CONCEPT:AU-KG.retrieval.per-item-relevance-ranking).

        Scans unresolved research topics, counts scorable items against the
        primary codebase, logs an ``EvolutionCycle`` node, and triggers the
        telemetry-ingestion sweep. Run by the consolidated maintenance scheduler.
        """
        from datetime import datetime

        EVOLUTION_INTERVAL = _EVOLUTION_INTERVAL
        cycle_start = datetime.now(UTC)
        cycle_id = f"evo_cycle_{cycle_start.strftime('%Y%m%d_%H%M%S')}"
        logger.info("Evolution: starting cycle %s", cycle_id)

        # 1. Detect unresolved research topics
        topics = self.query_cypher(
            "MATCH (c:Concept) OPTIONAL MATCH (c)-[:ADDRESSED_BY]->(p) "
            "WHERE p IS NULL RETURN c.id AS id, c.name AS name ORDER BY c.name LIMIT 15"
        )
        topic_count = len(topics) if topics else 0
        logger.info("Evolution: found %d unresolved topics", topic_count)

        # 2. Detect primary codebase
        primary_codebase = self._detect_primary_codebase()

        # 3. Count scorable items if we have a codebase target
        papers_scored = 0
        if primary_codebase and topic_count > 0:
            try:
                count_result = self.query_cypher(
                    "MATCH (n) WHERE n:Document OR n:Codebase RETURN count(n) AS total",
                )
                papers_scored = count_result[0].get("total", 0) if count_result else 0
                logger.info(
                    "Evolution: %d items available for relevance sweep against '%s'",
                    papers_scored,
                    primary_codebase,
                )
            except Exception as e:
                logger.warning(f"Evolution: relevance count failed: {e}")

        # 4. Log evolution cycle as a KG node
        try:
            from agent_utilities.knowledge_graph.core.engine import (
                IntelligenceGraphEngine,
            )

            throughput = 0
            try:
                throughput_query = self.query_cypher(
                    "MATCH (n:OptimizationTrajectory) WHERE n.created_at >= $timestamp "
                    "RETURN count(n) AS throughput",
                    params={
                        "timestamp": (
                            cycle_start - timedelta(seconds=EVOLUTION_INTERVAL)
                        ).isoformat()
                    },
                )
                throughput = (
                    throughput_query[0].get("throughput", 0) if throughput_query else 0
                )
                logger.info(
                    "Evolution: OptimizationTrajectoryNode throughput = %d", throughput
                )
            except Exception as e:
                logger.warning(f"Evolution: failed to get throughput: {e}")

            if isinstance(self, IntelligenceGraphEngine):
                self.add_node(
                    node_id=cycle_id,
                    node_type="EvolutionCycle",
                    properties={
                        "triggered_by": "daemon",
                        "topics_scanned": topic_count,
                        "papers_scored": papers_scored,
                        "primary_codebase": primary_codebase or "unknown",
                        "optimization_throughput": throughput,
                        "created_at": cycle_start.isoformat(),
                    },
                )
                logger.info(
                    "Evolution: logged cycle %s (topics=%d, scored=%d)",
                    cycle_id,
                    topic_count,
                    papers_scored,
                )
        except Exception as e:
            logger.warning(f"Evolution: failed to log cycle node: {e}")

        # 5. Telemetry/failure ingestion now runs as its own dedicated maintenance
        # job (``failure_ingest`` → _tick_failure_ingest, CONCEPT:AU-AHE.harness.failure-evolution), opt-in
        # via KG_FAILURE_EVOLUTION. The previous inline ``telemetry_ingestion``
        # workflow sweep referenced a workflow that was never defined (it raised
        # ValueError every cycle), so it has been removed in favor of that tick.

    def _graph_writer_loop(self):
        """Background daemon thread to drain the staging SQLite queue and insert heavy graph payloads sequentially to prevent lock contention."""
        import time

        from agent_utilities.knowledge_graph.pipeline.phases.sync import _TYPE_TO_TABLE
        from agent_utilities.models.schema_definition import SCHEMA

        # Build schema cache
        schema_cache = {}
        for node_schema in SCHEMA.nodes:
            schema_cache[node_schema.name] = set(node_schema.columns.keys())

        while True:
            try:
                if not getattr(self, "backend", None):
                    time.sleep(1.0)
                    continue

                item = self._submission_queue.get_staged_graph()
                if item is None:
                    time.sleep(1.0)
                    continue

                item_id, job_id, graph_data = item
                nodes = graph_data.get("nodes", [])
                edges = graph_data.get("edges", [])

                logger.info(
                    f"GraphWriterDaemon processing payload for {job_id}: {len(nodes)} nodes, {len(edges)} edges"
                )

                node_type_map = {}

                # Execute all nodes sequentially.
                for node in nodes:
                    if "id" in node and "type" in node:
                        nid = node.pop("id")
                        raw_type = str(node.pop("type")).lower()
                        label = _TYPE_TO_TABLE.get(raw_type) or "".join(
                            word.capitalize()
                            for word in raw_type.replace("_", " ").split()
                        )
                        label = _safe_graph_identifier(label, default="Code")
                        if not label:
                            label = "Code"

                        node_type_map[nid] = label

                        # Filter valid properties
                        valid_keys = schema_cache.get(label)
                        props = {k: v for k, v in node.items() if v is not None}
                        # Preserve original semantic type for Code nodes (file/symbol/module).
                        # The Code table declares ``symbol_type`` (not a bare ``type``, which
                        # the schema retired in favor of ``node_type``) for exactly this — the
                        # same column parse.py/graph_compute.py/blast_radius.py already read
                        # and write. Writing "type" here used to alias the schema's generic
                        # column; now it would be silently dropped by the valid_keys filter
                        # below, so route it to the dedicated column instead.
                        if label == "Code" and raw_type and raw_type != "code":
                            props["symbol_type"] = raw_type

                        # Collect extra properties into metadata dict, mirroring sync.py logic
                        if valid_keys is not None and "metadata" in valid_keys:
                            extra_props = {}
                            for k in list(props.keys()):
                                if k != "id" and k not in valid_keys:
                                    extra_props[k] = props.pop(k)
                            if extra_props:
                                curr_meta = props.get("metadata", {})
                                if isinstance(curr_meta, str):
                                    try:
                                        import json

                                        curr_meta = json.loads(curr_meta)
                                    except Exception:
                                        curr_meta = {}
                                if not isinstance(curr_meta, dict):
                                    curr_meta = {}
                                curr_meta.update(extra_props)
                                props["metadata"] = curr_meta

                        if valid_keys:
                            props = {k: v for k, v in props.items() if k in valid_keys}

                        # Serialize dict/list values to JSON strings
                        for k, v in list(props.items()):
                            if isinstance(v, dict | list):
                                import json

                                props[k] = json.dumps(v)

                        # Execute MERGE
                        # Using query_cypher to pass props nicely
                        safe_properties = {
                            key: value
                            for key, value in props.items()
                            if isinstance(key, str) and _GRAPH_IDENTIFIER.fullmatch(key)
                        }
                        set_clause = ", ".join(
                            [f"n.{key} = $props_{key}" for key in safe_properties]
                        )
                        if set_clause:
                            set_clause = " SET " + set_clause
                        query = f"MERGE (n:{label} {{id: $id}}){set_clause}"

                        params = {"id": nid}
                        for key, value in safe_properties.items():
                            params[f"props_{key}"] = value

                        self.backend.execute(query, params)

                # Execute all edges sequentially. A comma-pattern MATCH plus an
                # edge MERGE both exceed the engine's native Cypher write
                # subset (one leading MATCH, MERGE on a single bare node only;
                # epistemic-graph/crates/eg-query/src/cypher/parser.rs:1184);
                # ``link_nodes`` dispatches through the typed engine API for a
                # native authority (which requires both endpoints to already
                # exist, unlike the portable Cypher fallback used for a
                # non-native store) and per-edge best effort (matching the
                # original MATCH's silent no-op for a dangling reference): the
                # staged item is retried wholesale on failure (below), so one
                # edge referencing a node outside this batch must not doom the
                # whole item to an infinite retry loop.
                for edge in edges:
                    if "source" in edge and "target" in edge and "type" in edge:
                        src = edge.pop("source")
                        tgt = edge.pop("target")
                        etype = _safe_graph_identifier(str(edge.pop("type")).upper())

                        if not etype:
                            continue

                        try:
                            self.link_nodes(src, tgt, etype)
                        except Exception as exc:  # noqa: BLE001 — dangling edge reference against a native authority; logged and skipped so it can't wedge this staged item in an infinite retry loop
                            logger.debug(
                                "Skipped staged edge %s -[%s]-> %s: %s",
                                src,
                                etype,
                                tgt,
                                exc,
                            )

                # Only acknowledge and remove from staging if successful
                self._submission_queue.ack_staged_graph(item_id)
            except Exception as exc:
                logger.error(
                    "Error persisting staged graph; will retry: %s",
                    exc,
                )
                time.sleep(2.0)

    def submit_task(
        self,
        target_path: str,
        is_codebase: bool,
        provenance: dict,
        task_type: str | None = None,
        skip_dedupe: bool = False,
        priority: int | None = None,
        scheduled_for: float | None = None,
        depends_on: list[str] | None = None,
        max_attempts: int = _TASK_MAX_ATTEMPTS,
        job_id: str | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> str:
        """Submit a background task to the unified durable queue (CONCEPT:AU-KG.ingest.hardened-priority-scheduled-task).

        ``priority`` picks a claim bucket (0=critical .. 3=background).
        ``scheduled_for`` and ``depends_on`` are evaluated atomically by native
        WorkItem selection. ``job_id`` lets a
        caller supply a deterministic id (the unified Scheduler uses
        ``sched:<name>:<minute>`` so a double-fire is an idempotent upsert).
        """
        from agent_utilities.orchestration import work_item as _wi

        durable_target = _portable_task_target(target_path)
        # WorkItem owns both the immutable execution definition and lifecycle.
        if not skip_dedupe:
            for item in self._ingest_work_item_index().values():
                meta = item.get("metadata") or {}
                if meta and meta.get("target") == durable_target:
                    if item.get("status") not in _wi.TERMINAL_WORK_ITEM_STATUSES:
                        return str(item.get("payload_ref") or "")

        if not job_id:
            job_id = f"job-{uuid.uuid4().hex}"

        if not task_type:
            task_type = "codebase" if is_codebase else "document"

        now = time.time()
        task_data: dict[str, Any] = {
            "target": durable_target,
            "type": task_type,
            "submitted_at": datetime.now(UTC).isoformat(),
            "attempts": 0,
            "max_attempts": int(max_attempts),
        }
        if extra_meta:
            task_data.update(extra_meta)
            only_files = task_data.get("only_files")
            if isinstance(only_files, list):
                task_data["only_files"] = [
                    _portable_task_target(str(path)) for path in only_files
                ]

        prio_bucket = _coerce_prio_bucket(priority)
        if depends_on:
            task_data["depends_on"] = list(depends_on)
        if scheduled_for and float(scheduled_for) > now:
            task_data["eta_unix"] = float(scheduled_for)

        from agent_utilities.knowledge_graph.core.task_lanes import lane_for_task_type

        lane = lane_for_task_type(task_type)
        if provenance:
            task_data["provenance"] = dict(provenance)

        try:
            from agent_utilities.knowledge_graph.core.session import current_session

            active_session = current_session()
            tenant = active_session.tenant if active_session is not None else ""
        except Exception:  # pragma: no cover - local bootstrap
            tenant = ""
        work_item_id = _wi.ingest_task_work_item_id(job_id)
        task_data["work_item_id"] = work_item_id
        _wi.ensure_ingest_task_work_item(
            self._work_item_engine,
            job_id,
            prio_bucket=prio_bucket,
            resource_class=lane or "default",
            fairness_group=task_type,
            tenant=tenant,
            depends_on=depends_on or (),
            available_at=float(scheduled_for) if scheduled_for else None,
            max_attempts=max_attempts,
            metadata=task_data,
        )
        # Kafka remains a notification transport only. Its consumers must win
        # the same native WorkItem claim before executing; non-Kafka workers
        # discover ready WorkItems directly and need no second queue record.
        if getattr(self, "_task_queue_backend_name", "sqlite") == "kafka":
            from agent_utilities.security.persistence_privacy import (
                persistence_reference,
            )

            _submit_kafka_notification(
                self._submission_queue,
                task_type,
                {
                    "job_id": job_id,
                    "partition_ref": persistence_reference(
                        "ingest_partition", durable_target, namespace=tenant
                    ),
                },
            )

        # Pre-ingestion: drop ONLY the HNSW indexes for tables this task writes to.
        # (Kuzu can't SET on indexed columns.) Unaffected indexes stay active.
        _TASK_TABLE_MAP = {
            "codebase": ["Code"],
            "document": ["Article"],
            "conversation": ["Message"],
        }
        affected_tables = _TASK_TABLE_MAP.get(task_type, [])
        if (
            affected_tables
            and self.backend
            and hasattr(self.backend, "drop_vector_indices")
        ):
            if not hasattr(self, "_dropped_tables"):
                self._dropped_tables: set[str] = set()
            new_tables = [t for t in affected_tables if t not in self._dropped_tables]
            if new_tables:
                try:
                    self.backend.drop_vector_indices(tables=new_tables)
                    # D-DST-3 (CONCEPT:AU-AHE.evaluation.debug-swallow-justification): only
                    # mark these tables dropped AFTER drop_vector_indices actually succeeds.
                    # Marking them first (the prior order) meant a failed drop was never
                    # retried on the next submit_task call for this task_type — the write-
                    # then-mark-seen shape this triage was scoped to find — while the still-
                    # indexed columns kept failing the ingestion SET the comment above warns
                    # about ("Kuzu can't SET on indexed columns").
                    self._dropped_tables.update(new_tables)
                except Exception as e:  # noqa: BLE001 — the drop stays un-marked on failure (see above), so the next task submission for this task_type retries it instead of permanently believing the indexes are gone
                    logger.debug(f"Pre-ingestion index drop skipped: {e}")

        # Lazily start workers if they aren't already running
        self.start_task_workers()
        return job_id

    def _maybe_fanout_codebase(
        self, job_id: str, target: Path, meta: dict[str, Any]
    ) -> bool:
        """Split a too-large whole-repo codebase task into shard-routed sub-tasks.

        CONCEPT:AU-KG.ingest.subtask-routing-key — the big-repo tail fix. A whole-repo ``codebase`` task for
        a repo above :data:`~...repo_split.SPLIT_MIN_FILES` files is fanned out into
        K balanced sub-tasks, each scoped to a file bucket (``only_files``) and routed
        to its own per-shard graph (``code:<repo>__s<i>``), so the buckets commit in
        PARALLEL across the engine's K redb shard writers instead of one repo pinning
        one worker/shard for minutes. Returns ``True`` when it fanned out (the caller
        marks this parent done and stops); ``False`` to ingest inline as before.

        Guards keep the median safe and the recursion bounded:

        * a sub-task (carries ``route_repo``/``split_child``) never re-splits;
        * an explicitly-scoped task (``only_files`` already set, e.g. the dirty
          self-ingest) is left exactly as-is;
        * small/medium repos (the healthy p50) fall straight through to the inline
          path, untouched.
        """
        # Already a sub-task, or an explicitly-scoped ingest → never fan out.
        if meta.get("route_repo") or meta.get("split_child") or meta.get("only_files"):
            return False
        repo_root = Path(target)
        if not repo_root.is_dir():
            return False
        try:
            from agent_utilities.knowledge_graph.enrichment.pipeline import (
                discover_source_files,
            )
            from agent_utilities.knowledge_graph.ingestion.repo_split import (
                SPLIT_MIN_FILES,
                plan_repo_split,
                split_graph_suffix,
            )

            files = discover_source_files(repo_root)
        except Exception:  # noqa: BLE001 — discovery failure → ingest inline
            return False
        if len(files) <= SPLIT_MIN_FILES:
            return False

        from agent_utilities.knowledge_graph.core.worker_scheduler import (
            durable_shard_writers,
        )

        # Fan across the engine's shard-writer width (≥2 so a split is meaningful).
        k = max(2, durable_shard_writers())
        buckets = plan_repo_split(repo_root, files, k)
        if len(buckets) <= 1:
            return False

        repo_name = repo_root.name
        child_ids: list[str] = []
        for i, bucket in enumerate(buckets):
            child_ids.append(
                self.submit_task(
                    target_path=str(repo_root),
                    is_codebase=True,
                    provenance={},
                    task_type="codebase",
                    # All children share the repo target — the per-bucket identity is
                    # the routing key, so the target-based dedupe must be bypassed.
                    skip_dedupe=True,
                    extra_meta={
                        "only_files": [str(p) for p in bucket],
                        "route_repo": f"{repo_name}{split_graph_suffix(i)}",
                        "split_child": True,
                        "split_parent": job_id,
                        "split_bucket": i,
                    },
                )
            )
        self._update_task_status(
            job_id,
            "completed",
            {
                "target": str(repo_root),
                "type": "codebase",
                "status": "fanned_out",
                "split_children": child_ids,
                "split_buckets": len(buckets),
                "split_files": len(files),
            },
        )
        logger.info(
            "[KG-2.287] split big repo %s (%d files) into %d shard-routed sub-tasks",
            repo_name,
            len(files),
            len(buckets),
        )
        return True

    def _bulk_ingest_active(self, threshold: int = 1) -> bool:
        """True if ``threshold``+ codebase ingest tasks are pending/running.

        Used to gate recursive ``deep_analysis`` fan-out: while a bulk codebase
        ingest is draining, ``deep_analysis`` (0-node, recursive, blocking-LLM)
        runs flat (no fan-out) so it can't flood the queue ahead of structural
        ingest. (CONCEPT:AU-KG.compute.registered-edge-type / KG-2.8)
        """
        n = 0
        for item in self._ingest_work_item_index().values():
            meta = item.get("metadata") or {}
            state = _task_status_from_work_item(item)
            if (
                meta
                and meta.get("type") == "codebase"
                and state in {"pending", "running", "scheduled", "blocked"}
            ):
                n += 1
                if n >= threshold:
                    return True
        return False

    def ingest_queue_depth(self) -> int:
        """Uniform ingest backlog depth across queue backends (CONCEPT:AU-KG.ingest.decoupled-kg-ingest-consumer).

        Counts native ingestion WorkItems in a non-terminal state. This is the
        single number backpressure consumers (the batch
        orchestrator's deferral, the maintenance bulk-defer gate, the lag
        metrics) should read, regardless of which backend is selected.
        """
        active = {"pending", "running", "scheduled", "blocked"}
        return sum(
            1
            for item in self._ingest_work_item_index().values()
            if _task_status_from_work_item(item) in active
        )

    def submit_directory_tasks(
        self, directory: Path, provenance: dict
    ) -> tuple[list[dict[str, str]], list[str]]:
        """Enumerate supported files in a directory and create per-file jobs.

        Args:
            directory: Path to the directory to scan.
            provenance: Provenance metadata for tracking.

        Returns:
            Tuple of (queued_jobs, skipped_files).
        """
        queued_jobs: list[dict[str, str]] = []
        skipped: list[str] = []

        # Pre-fetch WorkItem definitions once.
        active_targets = set()
        for item in self._ingest_work_item_index().values():
            meta = item.get("metadata") or {}
            state = _task_status_from_work_item(item)
            if (
                meta
                and "target" in meta
                and state
                in {
                    "pending",
                    "running",
                    "scheduled",
                    "blocked",
                }
            ):
                active_targets.add(meta["target"])

        for file_path in sorted(directory.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                target_str = str(file_path)
                if target_str in active_targets:
                    skipped.append(target_str)
                    continue

                job_id = self.submit_task(
                    target_str,
                    is_codebase=False,
                    provenance=provenance,
                    skip_dedupe=True,
                )
                queued_jobs.append({"job_id": job_id, "target": target_str})
                active_targets.add(target_str)
            else:
                skipped.append(str(file_path))

        return queued_jobs, skipped

    def start_task_workers(self, worker_count: int | None = None):
        """Start background workers to poll and execute tasks from the graph."""
        from agent_utilities.core.config import (
            DEFAULT_KNOWLEDGE_GRAPH_SYNC_BACKGROUND,
        )

        # Role gate: ``client`` processes never run task workers — the host
        # daemon (the singleton flock holder) drains the shared queue. Uses the
        # *effective* role so an ``auto`` process that lost the host election also
        # behaves as a client. (CONCEPT:AU-KG.coordination.embedder-breaker / OS-5.9)
        from .host_lock import effective_daemon_role

        if effective_daemon_role() == "client":
            logger.debug("effective daemon role=client; not starting task workers.")
            return

        if not DEFAULT_KNOWLEDGE_GRAPH_SYNC_BACKGROUND:
            logger.debug(
                "knowledge_graph_sync_background is false, skipping task workers."
            )
            return

        if worker_count is None:
            worker_count = compute_ingest_worker_count()

        if not self.backend:
            # We can't do distributed worker locks safely without a persistent backend
            return

        with self._worker_lock:
            # Check if we should start workers (if queue has items)
            # or if we are already running.
            if self._workers_running:
                return

            background_session = self._background_session_for_spawn()
            # Start workers
            self._workers_running = True

        if getattr(self, "_task_queue_backend_name", "sqlite") == "kafka":
            # CONCEPT:AU-KG.ingest.decoupled-kg-ingest-consumer — Kafka mode: the host's worker pool joins the
            # ``kg-ingest`` consumer group instead of polling WorkItems, so
            # it shares partitions (and per-key ordering) with any decoupled
            # `kg-ingest-worker` processes added for scale-out.
            from ..ingest_worker import start_ingest_consumer_pool

            logger.info(
                "Starting %d kg-ingest consumer workers (kafka task queue)...",
                worker_count,
            )
            start_ingest_consumer_pool(
                self,
                worker_count=worker_count,
                background_session=background_session,
            )
            return

        # ORCH-1.81: record the live pool size so the admission registry/policy
        # (hot spare, per-lane min coverage, codebase cap) size to this host's pool.
        self._ingest_worker_count = int(worker_count)

        # CONCEPT:AU-ORCH.scheduling.acquisition-lane-fairness — reserve a small floor of
        # this pool for HYDRATION-priority work (:data:`~agent_utilities.core.
        # resource_priority.HYDRATION_TASK_TYPES`, e.g. the fleet tool-schema
        # boot/scheduled probe) so it always has a worker that checks it FIRST,
        # regardless of how many ordinary connector-sync jobs occupy the rest of
        # the pool. Reuses the existing hot-spare sizing
        # (``SchedulerConfig.reserved`` / ``KG_SCHED_RESERVED``, same knob the
        # interactive-lane floor uses) rather than introducing a second reservation
        # knob. A dedicated ``worker_count - 1`` cap keeps at least one thread
        # NEVER hydration-first even on a large pool, so a hydration firehose
        # can't fully starve ordinary connector-sync work either.
        from .worker_scheduler import scheduler_config_from_env

        hydration_reserved_count = 0
        # D-43: a single-worker pool used to reserve 0 (the ``worker_count - 1``
        # cap forces this — you cannot dedicate a whole 1-worker pool to
        # hydration-first without a *second* thread to still guarantee general
        # work, which doesn't exist here). That left minimal/laptop deployments
        # with the original starvation exposure this whole mechanism exists to
        # close. Since ``_claim_next_task(hydration_reserved=True)`` already
        # FALLS THROUGH to the normal unrestricted claim whenever no hydration
        # work is pending (see its docstring), a hydration-first check never
        # drops general-work coverage by itself — the only real risk is a
        # sustained hydration-type firehose keeping the sole worker permanently
        # in the hydration branch, starving general work in the OTHER
        # direction. Alternating the check (hydration-first on every other
        # poll) gives hydration work a genuine floor without that full-reversal
        # risk: even under a firehose, half the polls still go straight to the
        # general claim.
        single_worker_alternate = worker_count == 1
        if worker_count > 1:
            hydration_reserved_count = min(
                max(1, scheduler_config_from_env(worker_count).reserved),
                worker_count - 1,
            )

        logger.info(
            f"Starting {worker_count} TaskManager workers "
            f"({hydration_reserved_count} hydration-reserved"
            f"{', alternating on the sole worker' if single_worker_alternate else ''})..."
        )
        for i in range(worker_count):
            t = _authorized_background_thread(
                background_session,
                self._task_worker_loop,
                name=f"KGTaskWorker-{i}",
                args=(i < hydration_reserved_count, single_worker_alternate),
            )
            t.start()

    def _ingest_work_item_index(self) -> dict[str, dict[str, Any]]:
        """Load ingestion WorkItems keyed by public job id in one graph read."""
        rows = self._work_item_engine.query_cypher(
            "MATCH (w:WorkItem) RETURN w.id AS id, w.kind AS kind, "
            "w.payload_ref AS payload_ref, w.status AS status, "
            "w.next_retry_at AS next_retry_at, w.resource_class AS resource_class, "
            "w.fairness_group AS fairness_group, w.prio_bucket AS prio_bucket, "
            "w.attempt AS attempt, w.max_attempts AS max_attempts, "
            "w.submitted_at AS submitted_at, w.completed_at AS completed_at, "
            "w.error_ref AS error_ref, "
            "w.result_ref AS result_ref, w.metadata AS metadata"
        )
        out: dict[str, dict[str, Any]] = {}
        for row in rows or []:
            if (
                not isinstance(row, dict)
                or row.get("kind") != "ingest_task"
                or not row.get("payload_ref")
            ):
                continue
            item = dict(row)
            if not isinstance(item.get("metadata"), dict):
                item["metadata"] = {}
            out[str(item["payload_ref"])] = item
        return out

    def lane_metrics(self) -> dict[str, Any]:
        """Per-lane congestion snapshot (CONCEPT:AU-ORCH.execution.two-level-fair-rotation): pending depth + in-flight per
        functional lane, so congestion is VISIBLE before it starves work — the observability
        that was missing when codebase ingestion silently sat at 75-pending/0-running. Returns
        ``{lane: {pending, running, model_role}}`` + a ``lane_less`` bucket for un-stamped tasks.
        """
        from agent_utilities.knowledge_graph.core.task_lanes import (
            LANE_NAMES,
            lane_model_role,
        )

        work = self._ingest_work_item_index()
        counts: dict[tuple[str, str], int] = {}
        type_pending: dict[str, int] = {}
        for item in work.values():
            state = _task_status_from_work_item(item)
            lane = str(item.get("resource_class") or "")
            counts[(lane, state)] = counts.get((lane, state), 0) + 1
            if state == "pending":
                kind = str(item.get("fairness_group") or "")
                type_pending[kind] = type_pending.get(kind, 0) + 1

        # ORCH-1.81: overlay the LIVE in-process worker registry so the snapshot
        # also shows how many workers each lane is *actually occupying right now*
        # (the queue's ``running`` status is set on claim; ``live_running`` is the
        # admission registry's view, which also drives the reservation/cap math).
        reg = getattr(self, "_worker_reg", None)
        live_running = reg.running_by_lane() if reg is not None else {}

        out: dict[str, Any] = {}
        for lane in LANE_NAMES:
            p = counts.get((lane, "pending"), 0)
            r = counts.get((lane, "running"), 0)
            out[lane] = {
                "pending": p,
                "running": r,
                "live_running": int(live_running.get(lane, 0)),
                "model_role": lane_model_role(lane),
            }
        total_pending = sum(
            v for (lane, status), v in counts.items() if status == "pending"
        )
        total_running = sum(
            v for (lane, status), v in counts.items() if status == "running"
        )
        out["lane_less"] = {
            "pending": max(
                0,
                total_pending
                - sum(v["pending"] for v in out.values() if "pending" in v),
            ),
            "running": max(
                0,
                total_running
                - sum(v["running"] for v in out.values() if "running" in v),
            ),
            "model_role": None,
        }
        # KG-2.145: surface the adaptive LLM/embedding concurrency targets next to
        # lane congestion, so over/under-utilisation of the vLLM serving tier is
        # visible in the same snapshot. Throttled internally; best-effort.
        try:
            from agent_utilities.core.model_capacity_autoscale import get_utilization

            out["model_concurrency"] = {
                role: get_utilization(role) for role in ("embedding", "lite", "default")
            }
        except Exception:  # noqa: BLE001 — observability is best-effort, never fatal
            out["model_concurrency"] = {}

        # ORCH-1.81: surface the scheduler's pool/reservation picture for ops.
        cfg = getattr(self, "_sched_config", None)
        out["scheduler"] = {
            "worker_count": getattr(cfg, "worker_count", None),
            "reserved": getattr(cfg, "reserved", None),
            "per_lane_min": getattr(cfg, "per_lane_min", None),
            "codebase_cap": getattr(cfg, "codebase_cap", None),
            "busy_workers": reg.busy_count() if reg is not None else 0,
            "free_workers": (
                reg.free_count(getattr(cfg, "worker_count", 0))
                if reg is not None and cfg is not None
                else None
            ),
            "running_by_type": reg.running_by_type() if reg is not None else {},
        }

        # CONCEPT:AU-ORCH.dispatch.two-pool — per-pool congestion + budget, so an
        # operator can see whether memory-gen is at its cap (back-pressured on the
        # write lock) while acquisition still has headroom. Pending is summed over
        # each pool's lanes (+ the content_url override); running is the live
        # registry's per-pool view.
        from agent_utilities.knowledge_graph.core.task_lanes import (
            POOLS,
            pool_for_task_type,
        )

        live_by_pool = reg.running_by_pool() if reg is not None else {}
        pool_out: dict[str, Any] = {}
        for pool, lanes in POOLS.items():
            pending = sum(out.get(ln, {}).get("pending", 0) for ln in lanes)
            # content_url rides the ingestion lane but is budgeted as acquisition;
            # move its pending count to the acquisition rollup for an accurate view.
            pool_out[pool] = {
                "pending": pending,
                "live_running": int(live_by_pool.get(pool, 0)),
            }
        # Reflect the per-type pool override in the pending rollup (content_url).
        cu_pending = type_pending.get("content_url", 0)
        cu_pool = pool_for_task_type("content_url")
        if cu_pool in pool_out and cu_pool != "memory_gen":
            pool_out[cu_pool]["pending"] += cu_pending
            if "memory_gen" in pool_out:
                pool_out["memory_gen"]["pending"] = max(
                    0, pool_out["memory_gen"]["pending"] - cu_pending
                )
        if cfg is not None:
            pool_out["acquisition_floor"] = getattr(cfg, "acquisition_floor", None)
            pool_out["memory_gen_cap"] = getattr(cfg, "memory_gen_cap", None)
        out["pools"] = pool_out
        return out

    # -- Reserved-worker fair scheduler (CONCEPT:AU-ORCH.dispatch.worker-scheduling) ------------------
    def _worker_registry(self):
        """Lazy in-process worker→(lane, type) registry for admission control.

        Created on first use and sized to the live worker pool. The pool size is
        autosized once (``compute_ingest_worker_count``); the registry only tracks
        what each worker is *currently* processing, so it never needs resizing.
        """
        reg = getattr(self, "_worker_reg", None)
        if reg is None:
            from .worker_scheduler import (
                WorkerRegistry,
                resolve_engine_shard_writers,
                scheduler_config_from_env,
            )

            wc = int(getattr(self, "_ingest_worker_count", 0) or 0)
            if wc <= 0:
                wc = compute_ingest_worker_count()
                self._ingest_worker_count = wc
            reg = WorkerRegistry()
            self._worker_reg = reg
            self._sched_config = scheduler_config_from_env(wc)
            # resolve the ENGINE's real durable shard-writer
            # width K once, from the engine that owns the redb backend (it may be a
            # remote box with a different cpu count than this scheduling host in
            # split-storage). Cached inside worker_scheduler so the codebase
            # admission floor reflects the engine's actual K, not this host's cpus.
            try:
                resolve_engine_shard_writers(self.backend)
            except Exception:  # noqa: BLE001 — best-effort; falls back to cpu/env
                pass
        return reg

    def _admission_policy(self):
        """Lazy, cached :class:`AdmissionPolicy` bound to the live registry/config.

        Reuses the SAME :class:`SchedulerConfig`/:class:`WorkerRegistry` pair
        :meth:`_worker_registry` lazily builds (calling it here ensures
        ``_sched_config`` exists too) — the reservation/heavy-type/cap knobs and
        in-process running-worker picture an admission decision reads must be
        the ones the rest of the scheduler is using, not a second, unbound
        instance that would drift from the registry workers actually claim into.
        """
        policy = getattr(self, "_admission_pol", None)
        if policy is None:
            from .worker_scheduler import AdmissionPolicy

            registry = self._worker_registry()
            policy = AdmissionPolicy(self._sched_config, registry)
            self._admission_pol = policy
        return policy

    def _pending_by_lane(self) -> dict[str, int]:
        """Ready WorkItem count per functional resource-class lane.

        PERF (BUG-047 — CONCEPT:AU-ORCH.scheduling.resource-priority-edict): this
        used to route through :meth:`_ingest_work_item_index`, whose
        ``MATCH (w:WorkItem) RETURN <15 wide fields, incl. metadata>`` has no
        WHERE/LIMIT and returns EVERY WorkItem ever created. This is the
        engine's dominant sustained-slow-query source (measured 8-21s against a
        500ms threshold): :meth:`_claim_next_task` calls this on every
        non-hydration-reserved claim attempt, by every worker in the pool, on
        every poll (idle backoff is only 2-15s) — so the full unbounded scan
        re-ran continuously, tripping the engine circuit breaker and taking the
        whole KG read path down with it (BUG-047/BUG-048).
        A per-lane PENDING count needs only ``resource_class`` grouped under a
        ``status``/``next_retry_at`` predicate, not the full row set — this
        mirrors the aggregate-count convention already established by
        :func:`agent_utilities.orchestration.work_item.machine_state_distribution`
        (``RETURN <col>, count(w)`` instead of materializing every row in
        Python). Only ``ingest_task`` WorkItems feed lane admission, matching
        :meth:`_ingest_work_item_index`'s existing filter; the native "ready and
        due" predicate reproduces ``_task_status_from_work_item``'s "pending"
        classification exactly (status == "ready" and next_retry_at <= now).

        No property-value index exists for ``status``/``next_retry_at`` at the
        Cypher layer (confirmed: the schema defines no secondary index beyond
        the ``id`` primary key, and the engine's own ``_upsert`` docstring notes
        there is "no write-path id index" either) — this bound reduces the
        RESPONSE SIZE from O(all WorkItem history × 15 wide fields, including a
        JSON metadata blob) to O(number of lanes) aggregated rows, which is the
        overwhelming share of the measured cost, but the underlying scan still
        visits every WorkItem node server-side. A genuine indexed access path
        (the native, keyset-paginated ``GetNodesByLabel``/``list_by_label``
        verb epistemic-graph already exposes) would remove that residual cost
        too; wiring it through :class:`_ControlPlaneWorkItemEngine` is a
        follow-up, not required for this fix.
        """
        from agent_utilities.knowledge_graph.core.task_lanes import LANE_NAMES

        out: dict[str, int] = {lane: 0 for lane in LANE_NAMES}
        rows = self._work_item_engine.query_cypher(
            "MATCH (w:WorkItem) WHERE w.kind = 'ingest_task' "
            "AND w.payload_ref IS NOT NULL AND w.status = 'ready' "
            "AND (w.next_retry_at IS NULL OR w.next_retry_at <= $now) "
            "RETURN w.resource_class AS resource_class, count(w) AS n",
            {"now": time.time()},
        )
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            lane = str(row.get("resource_class") or "")
            if lane in out:
                out[lane] += int(row.get("n") or 0)
        return out

    def _remember_work_item_claim(self, job_id: str, claim: dict[str, Any]) -> None:
        """Keep the native fencing tuple in process memory while executing.

        Claims are capabilities, not durable task metadata. Persisting them in
        another node would create a second writable ownership authority.
        """
        with self._active_work_item_claims_lock:
            self._active_work_item_claims[job_id] = dict(claim)

    def _active_work_item_claim(
        self, job_id: str, *, pop: bool = False
    ) -> dict[str, Any] | None:
        lock = getattr(self, "_active_work_item_claims_lock", None)
        if lock is None:
            return None
        with lock:
            if pop:
                return self._active_work_item_claims.pop(job_id, None)
            claim = self._active_work_item_claims.get(job_id)
            return dict(claim) if claim is not None else None

    def _start_work_item_lease_heartbeat(
        self,
        job_id: str,
        *,
        cancellation_event: threading.Event | None = None,
    ) -> None:
        """Renew one claimed ingestion WorkItem's lease — until its task body
        returns, OR until ``cancellation_event`` fires, whichever comes first.

        CONCEPT:AU-ORCH.scheduling.soft-timeout-lease-quarantine — a claimed task that has already
        exceeded its lane's soft timeout (:mod:`.task_lanes`) has, by definition,
        overrun its expected envelope; renewing its lease forever just because the
        cooperative cancellation it was sent went unheeded turns one uncooperative
        synchronous call into a PERMANENT hole in the pool (proven live: both the
        boot fleet-tool-schema hydration job and its hourly schedule counterpart
        were observed renewing past their soft timeout with the worker still
        wedged). Once ``cancellation_event`` is set, renewal STOPS here — the
        lease is left to expire on its own TTL (``_TASK_WORK_ITEM_LEASE_SEC``) so
        another worker (this host or another) can reclaim the WorkItem and make
        progress on it, instead of the job being quarantined indefinitely
        alongside a thread Python cannot forcibly reclaim. This is safe against a
        duplicate/stale commit from the abandoned original run:
        ``_require_live_work_item_lease`` re-checks (and re-attempts) the lease
        immediately before any terminal write, so a run whose lease has already
        moved on can never commit a result out from under the new owner.
        """
        claim = self._active_work_item_claim(job_id)
        if claim is None:
            raise RuntimeError(f"ingestion job {job_id!r} has no active WorkItem claim")
        stop = threading.Event()
        lost = threading.Event()
        with self._work_item_lease_heartbeats_lock:
            self._work_item_lease_heartbeats[job_id] = (stop, lost)

        def _heartbeat_loop() -> None:
            from agent_utilities.orchestration import work_item as _wi

            while not stop.wait(_TASK_WORK_ITEM_HEARTBEAT_SEC):
                if cancellation_event is not None and cancellation_event.is_set():
                    lost.set()
                    logger.warning(
                        "ingestion WorkItem lease renewal stopped for %s — its "
                        "soft timeout already fired; the lease will expire and "
                        "the job becomes reclaimable rather than being renewed "
                        "indefinitely",
                        job_id,
                    )
                    return
                active_claim = self._active_work_item_claim(job_id)
                if active_claim is None:
                    return
                if not _wi.heartbeat(
                    self._work_item_engine,
                    str(active_claim["work_item_id"]),
                    active_claim,
                    lease_ttl_s=_TASK_WORK_ITEM_LEASE_SEC,
                ):
                    lost.set()
                    logger.warning(
                        "ingestion WorkItem lease lost while task %s was running",
                        job_id,
                    )
                    return

        _authorized_background_thread(
            self._background_session_for_spawn(),
            _heartbeat_loop,
            name=f"KGTaskLease-{job_id}",
        ).start()

    def _stop_work_item_lease_heartbeat(self, job_id: str) -> None:
        with self._work_item_lease_heartbeats_lock:
            heartbeat = self._work_item_lease_heartbeats.pop(job_id, None)
        if heartbeat is not None:
            heartbeat[0].set()

    def _require_live_work_item_lease(self, job_id: str, claim: dict[str, Any]) -> None:
        """Fence terminal state changes on a lease renewed immediately before commit."""
        from agent_utilities.orchestration import work_item as _wi

        with self._work_item_lease_heartbeats_lock:
            heartbeat = self._work_item_lease_heartbeats.get(job_id)
        if heartbeat is not None and heartbeat[1].is_set():
            raise _wi.WorkItemBackendUnavailable(
                f"ingestion job {job_id!r} lost its WorkItem lease"
            )
        work_item_id = str(
            claim.get("work_item_id") or _wi.ingest_task_work_item_id(job_id)
        )
        if not _wi.heartbeat(
            self._work_item_engine,
            work_item_id,
            claim,
            lease_ttl_s=_TASK_WORK_ITEM_LEASE_SEC,
        ):
            if heartbeat is not None:
                heartbeat[1].set()
            raise _wi.WorkItemBackendUnavailable(
                f"ingestion job {job_id!r} lease renewal was rejected"
            )

    def _ingest_task_metadata(self, job_id: str) -> dict[str, Any]:
        """Read an ingestion definition from its sole WorkItem."""
        from agent_utilities.orchestration import work_item as _wi

        item = _wi.get_work_item(
            self._work_item_engine, _wi.ingest_task_work_item_id(job_id)
        )
        if item is None:
            raise _wi.WorkItemBackendUnavailable(
                f"ingestion WorkItem for {job_id!r} does not exist"
            )
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            raise _wi.WorkItemBackendUnavailable(
                f"ingestion WorkItem for {job_id!r} has invalid metadata"
            )
        return dict(metadata)

    def _claim_next_task(
        self,
        worker_id: str | None = None,
        *,
        hydration_reserved: bool = False,
    ) -> tuple[str, dict[str, Any]] | None:
        """Claim the next runnable ingestion WorkItem natively.

        ``hydration_reserved`` (CONCEPT:AU-ORCH.scheduling.acquisition-lane-fairness) — this
        worker is one of the pool's reserved hydration slots
        (:func:`start_task_workers`): it FIRST tries to claim one of the small,
        foundational :data:`~agent_utilities.core.resource_priority.HYDRATION_TASK_TYPES`
        (e.g. ``capability_hydration``), scoped per type via ``resource_class``/
        ``fairness_group`` so it never picks up an ordinary ``connector_sync``
        sharing the same lane. Only when NONE of those are pending does it fall
        through to the normal unrestricted claim, so a reserved worker still does
        useful background work rather than idling — it just never lets that work
        block it from claiming hydration work the instant some is enqueued. This
        is what gives priority-1 capability/boot work a real floor: unlike the
        ordinary rotation, this worker's NEXT claim attempt (which happens as
        soon as its current task ends) always checks hydration lanes first, so
        it cannot be starved by however many concurrent legacy connector syncs
        are occupying the rest of the pool.
        """
        from agent_utilities.orchestration import work_item as _wi

        token = self._get_host_token()
        claim = None
        if hydration_reserved:
            from agent_utilities.core.resource_priority import HYDRATION_TASK_TYPES
            from agent_utilities.knowledge_graph.core.task_lanes import (
                lane_for_task_type,
            )

            for hydration_type in sorted(HYDRATION_TASK_TYPES):
                claim = _wi.claim_next(
                    self._work_item_engine,
                    queue="ingest_task",
                    resource_class=lane_for_task_type(hydration_type),
                    fairness_group=hydration_type,
                    token=token,
                    lease_ttl_s=_TASK_WORK_ITEM_LEASE_SEC,
                )
                if claim is not None:
                    break
        if claim is None:
            claim = _wi.claim_next(
                self._work_item_engine,
                queue="ingest_task",
                token=token,
                lease_ttl_s=_TASK_WORK_ITEM_LEASE_SEC,
            )
        if claim is None:
            return None  # authoritative negative; no secondary scan/fallback
        if not _wi.mark_running(self._work_item_engine, claim["work_item_id"], claim):
            return None

        job_id = str(
            claim.get("payload_ref")
            or _wi.ingest_task_job_id_from_work_item_id(claim["work_item_id"])
            or ""
        )
        if not job_id:
            raise _wi.WorkItemBackendUnavailable(
                "ClaimWorkItem returned an ingest item without payload_ref"
            )
        # Claim authority must be locally visible before the metadata read. The
        # read is graph-scoped and can be the first operation to observe a lazy
        # materialization transition; remembering only afterwards stranded an
        # already-consumed native attempt with no job id on the worker frame.
        self._remember_work_item_claim(job_id, claim)
        try:
            meta = self._ingest_task_metadata(job_id)
        except Exception as exc:
            materialization = _retryable_partial_materialization(exc)
            if materialization is not None:
                # Mirror _task_worker_loop's in-body materialization handling:
                # an unexpected failure releasing the native lease must still
                # drop the in-memory claim so it cannot be mistaken for a live
                # local reservation. The native WorkItem itself self-heals via
                # its own lease TTL + the pre-existing expired-lease reaper —
                # never converted into an application failed/dead-letter path.
                try:
                    self._defer_task_for_materialization(job_id, materialization)
                except Exception as defer_error:  # noqa: BLE001 - infrastructure transition is logged below
                    self._active_work_item_claim(job_id, pop=True)
                    logger.error(
                        "TaskManager could not defer %s while the graph was "
                        "materializing: %s",
                        job_id,
                        defer_error,
                    )
                return None
            if isinstance(exc, _wi.WorkItemBackendUnavailable):
                try:
                    _wi.commit_result(
                        self._work_item_engine,
                        claim["work_item_id"],
                        claim,
                        outcome="failed",
                        error_ref=f"invalid_ingest_definition:{job_id}",
                        retryable=False,
                    )
                finally:
                    self._active_work_item_claim(job_id, pop=True)
            else:
                # Every OTHER failure of the metadata read (a transient engine /
                # connection error surfacing as a bare RuntimeError, say) must
                # drop the in-memory claim too, or it strands here forever: the
                # claim is now remembered BEFORE the read (see above), so unlike
                # the previous ordering this except path can leak one. The
                # native WorkItem lease still self-heals via its TTL + the
                # expired-lease reaper; this only keeps the local bookkeeping
                # honest so a dead claim is never mistaken for a live local
                # reservation. The exception itself is always re-raised.
                self._active_work_item_claim(job_id, pop=True)
            raise
        # Stamp the winning claim's own lease identity onto the returned
        # metadata — ``claim`` (from ``_wi.claim_next``/``claim_specific``)
        # already carries the authoritative ``lease_owner``/``lease_epoch``
        # native WorkItem fields; nothing downstream previously surfaced them,
        # so a caller/log line had no way to say WHO holds this task or WHICH
        # fencing generation it's running under. This is an in-memory
        # enrichment of the dict handed back to the caller ONLY — it is never
        # persisted onto another node's durable metadata (that would create a
        # second writable ownership authority; the native WorkItem lease
        # remains the sole source of truth, per ``_remember_work_item_claim``).
        meta["claimed_by"] = claim.get("lease_owner")
        meta["work_item_epoch"] = claim.get("lease_epoch")
        meta["work_item_id"] = claim.get("work_item_id")
        tkind = str(meta.get("type") or "document")
        if worker_id is not None:
            from agent_utilities.knowledge_graph.core.task_lanes import (
                lane_for_task_type,
            )

            lane = lane_for_task_type(tkind)
            # CONCEPT:AU-ORCH.dispatch.worker-scheduling — gate the claim through the
            # reserved-worker fair AdmissionPolicy before this worker commits to the
            # task. Only applied to the general/unrestricted claim: the
            # ``hydration_reserved`` priority-floor path above must never be
            # second-guessed here — that floor exists specifically so hydration
            # work can't be starved, which is the opposite of what admission's
            # hot-spare/heavy-type/coverage rules are for. A denied admission
            # releases the native lease without consuming a retry attempt, so a
            # later (better-suited) poll — by this worker or another — picks the
            # task back up once the pool's live picture allows it.
            if not hydration_reserved:
                decision = self._admission_policy().decide(
                    lane, tkind, self._pending_by_lane()
                )
                if not decision.admit:
                    logger.debug(
                        "worker %s admission denied for %s/%s: %s — deferring claim",
                        worker_id,
                        lane,
                        tkind,
                        decision.reason,
                    )
                    self._defer_task_for_admission(job_id)
                    return None
            self._worker_registry().start(worker_id, lane, tkind)
        return job_id, meta

    def _task_worker_loop(
        self, hydration_reserved: bool = False, hydration_alternate: bool = False
    ):
        """Distributed polling loop that picks up pending tasks natively.

        ``hydration_reserved`` (CONCEPT:AU-ORCH.scheduling.acquisition-lane-fairness) marks this
        as one of the pool's reserved hydration-priority workers — see
        :meth:`_claim_next_task` and :meth:`start_task_workers`.

        ``hydration_alternate`` (D-43) is the degenerate single-worker case: this
        worker checks the hydration lane first on every OTHER poll instead of
        never (``hydration_reserved`` stays permanently ``False`` when there is
        no second thread to keep pinned to general work). Bounded so a sustained
        hydration-type firehose can't fully starve ordinary connector-sync work
        either — only every other poll is hydration-first.
        """
        # ORCH-1.81: a stable per-thread id keys this worker in the admission
        # registry, so the policy knows what THIS worker is processing.
        worker_id = threading.current_thread().name
        poll_count = 0
        # BUG-047 gap-fill (CONCEPT:AU-ORCH.scheduling.resource-priority-edict): this
        # dedicated background thread polls and claims work for its entire life —
        # tag it BACKGROUND_INGESTION ONCE so every engine call the claim decision
        # itself makes (:meth:`_claim_next_task`'s native ``ClaimWorkItem`` RPC,
        # :meth:`_pending_by_lane`'s admission read) carries that QoS priority
        # claim too. Before this, the claim/admission path ran with NO priority
        # scope entered at all: :func:`resource_priority._effective` resolves an
        # untagged context to ``ORCHESTRATION`` (HIGH, never yields) — the poll
        # loop's own engine reads were therefore indistinguishable from
        # interactive/orchestration traffic to the engine's reserved read lane,
        # even though this thread does nothing but background ingestion work.
        # The edict WAS already wired for an already-claimed task's own body
        # (``_run_body``'s ``priority_scope(priority_for_task_type(task_type))``
        # below, entered only after a claim succeeds) — this closes the gap for
        # the claim/admission decision that runs before that. ``set_priority``
        # (not a ``with`` block) is correct here: this OS thread never does
        # anything else, so setting its default once is equivalent to wrapping
        # the whole loop, without reindenting it; the per-task ``priority_scope``
        # below still correctly nests — it overrides this default for the
        # claimed task's duration, then restores it on exit.
        from agent_utilities.core.resource_priority import PriorityClass, set_priority

        set_priority(PriorityClass.BACKGROUND_INGESTION)
        while True:
            try:
                job_id = None
                target_path = None
                is_codebase = False
                task_type = "document"

                effective_hydration_reserved = hydration_reserved or (
                    hydration_alternate and poll_count % 2 == 0
                )
                poll_count += 1
                claimed = self._claim_next_task(
                    worker_id=worker_id,
                    hydration_reserved=effective_hydration_reserved,
                )
                if claimed:
                    job_id, meta = claimed
                    if meta:
                        if "target" in meta:
                            target_path = _resolve_task_target(str(meta["target"]))
                        task_type = meta.get("type", "document")
                        is_codebase = task_type == "codebase"

                if not job_id:
                    # Idle backoff. During a bulk ingest, back off HARD: one worker
                    # holds the ingest while the other idle workers repeatedly
                    # polled the queue every 2s, flooding the single
                    # client event loop + engine and starving the ingest worker
                    # (profiled: 24% of daemon CPU in poll query_cypher vs 10% in the
                    # actual ingest). A new task then waits at most one backoff to be
                    # claimed — fine while a multi-minute ingest drains. (CONCEPT:AU-KG.compute.registered-edge-type)
                    from agent_utilities.core.background_throttle import get_throttle

                    time.sleep(15.0 if get_throttle().should_yield_background else 2.0)
                    continue

                if not target_path:
                    logger.error(f"Task {job_id} has no target in metadata, skipping.")
                    self._update_task_status(
                        job_id,
                        "failed",
                        {
                            "error": "Missing target in task metadata",
                            "type": "unknown",
                        },
                    )
                    # ORCH-1.81: free this worker in the admission registry.
                    self._worker_registry().finish(worker_id)
                    time.sleep(2.0)
                    continue

                try:
                    self._execute_claimed_task(
                        job_id, target_path, is_codebase, task_type
                    )
                finally:
                    # ORCH-1.81: mark the worker free the moment its task is done
                    # (success or raise), so the next worker's admission and the
                    # codebase cap see the freed slot immediately.
                    self._worker_registry().finish(worker_id)

            except Exception as e:
                materialization = _retryable_partial_materialization(e)
                if materialization is not None:
                    if job_id:
                        try:
                            deferred = self._defer_task_for_materialization(
                                job_id, materialization
                            )
                        except Exception as defer_error:  # noqa: BLE001 - infrastructure transition is logged below
                            self._active_work_item_claim(job_id, pop=True)
                            logger.error(
                                "TaskManager could not defer %s while the graph was "
                                "materializing: %s",
                                job_id,
                                defer_error,
                            )
                        else:
                            if deferred:
                                logger.info(
                                    "TaskManager deferred materializing task %s "
                                    "(phase=%s cursor=%s)",
                                    job_id,
                                    materialization.get("phase"),
                                    materialization.get("completeness_cursor"),
                                )
                            else:
                                logger.warning(
                                    "TaskManager discarded fenced claim for "
                                    "materializing task %s (phase=%s cursor=%s)",
                                    job_id,
                                    materialization.get("phase"),
                                    materialization.get("completeness_cursor"),
                                )
                        # A materialization transition is infrastructure state,
                        # never an application attempt. Do not fall through to
                        # the generic retry/dead-letter path even if the fenced
                        # control transition was lost to another owner.
                        continue
                    else:
                        logger.info(
                            "TaskManager waiting for graph materialization before "
                            "claiming work (phase=%s cursor=%s)",
                            materialization.get("phase"),
                            materialization.get("completeness_cursor"),
                        )
                        time.sleep(5)
                        continue

                logger.error(f"TaskManager worker error: {e}")
                if job_id:
                    try:
                        self._fail_or_retry_task(job_id, str(e))
                    except Exception as inner_e:
                        logger.error(
                            f"Failed to update task status to failed for {job_id}: {inner_e}"
                        )
                # ORCH-1.81: ensure the worker is freed even on the error path.
                try:
                    self._worker_registry().finish(worker_id)
                except Exception as exc:  # noqa: BLE001 — worker-registry cleanup is best-effort
                    logger.debug(
                        "worker registry finish() failed for %s: %s", worker_id, exc
                    )  # nosec B110
                time.sleep(5)

    def _execute_claimed_task(
        self,
        job_id: str,
        target_path: Path,
        is_codebase: bool,
        task_type: str = "document",
    ) -> None:
        """Run ONE already-claimed task to completion (the shared worker body).

        Used by both the in-process graph-polling workers and the decoupled
        ``kg-ingest`` Kafka consumers (CONCEPT:AU-KG.ingest.decoupled-kg-ingest-consumer) so the processing logic
        exists exactly once. Heavy task types (parse storms / background LLM /
        analysis) run through the shared background throttle so they yield to
        interactive (foreground) work and stay within the global concurrency
        cap — a bulk ingest can no longer consume the engine's whole in-flight
        budget and starve live queries (CONCEPT:AU-KG.compute.registered-edge-type read/ingest plane
        isolation). Lightweight types (diff/conversation/…) run unthrottled.
        """
        # CONCEPT:AU-ORCH.scheduling.soft-timeout-lease-quarantine — shared with the lease
        # heartbeat below so renewal STOPS the moment this task's soft timeout fires
        # (see :meth:`_start_work_item_lease_heartbeat`), instead of an unconditional
        # heartbeat keeping an already-exceeded task's lease alive indefinitely.
        cancellation_event = threading.Event()
        lease_heartbeat_active = self._active_work_item_claim(job_id) is not None
        if lease_heartbeat_active:
            self._start_work_item_lease_heartbeat(
                job_id, cancellation_event=cancellation_event
            )
        try:
            self._execute_claimed_task_body(
                job_id,
                target_path,
                is_codebase,
                task_type,
                cancellation_event=cancellation_event,
            )
        finally:
            if lease_heartbeat_active:
                self._stop_work_item_lease_heartbeat(job_id)

    def _execute_claimed_task_body(
        self,
        job_id: str,
        target_path: Path,
        is_codebase: bool,
        task_type: str = "document",
        *,
        cancellation_event: threading.Event,
    ) -> None:
        """Execute the bounded task body while its enclosing lease stays live."""
        _HEAVY_TASK_TYPES = {
            "codebase",
            "document",
            "content_url",
            "feed_ingest",
            "feed_sweep",
            "deep_analysis",
            "synthesize",
            "deep_extract",
            "background_research",
            "relevance_sweep",
            "skill_workflows",
            "self_tool_surface",
            # Async session-bundle upload (CONCEPT:AU-KG.ingest.drain-session-bundle): each session fans out
            # to many usage-store rows, so it drains under the background throttle.
            "session_upload",
            # Scheduled jobs (source syncs, loop cycles, the RSS feed screen) run
            # under the background throttle so a heavy cycle yields to foreground
            # work like any other background task. (CONCEPT:AU-OS.state.unified-scheduling-one-intelligent)
            "scheduled_job",
            # Full-paper download + ingest enqueued by the RSS feed screen.
            "research_paper_fetch",
            # Cohort barrier finalize → assimilation pass + feature matrix (KG-2.172).
            "cohort_synthesize",
        }
        # CONCEPT:AU-KG.compute.lane-bound-task — request cooperative cancellation
        # when EVERY claimed task reaches its lane's soft timeout.
        #
        # The body deliberately stays on the durable queue's existing, bounded
        # worker instead of spawning a per-attempt daemon.  A tiny authorized
        # watchdog only sets the cancellation event and then exits.  Cooperative
        # loops stop promptly.  An uncooperative synchronous call remains
        # quarantined in its original fixed-capacity worker — Python cannot
        # safely kill an arbitrary running thread, so that worker slot is not
        # freed here (bound the call itself at its transport layer instead; see
        # ``_sync_fleet``'s ``_run_async(..., timeout=...)`` for a worked
        # example). What IS bounded here is the WorkItem's *lease*: this same
        # ``cancellation_event`` stops :meth:`_start_work_item_lease_heartbeat`
        # from renewing past the soft timeout (CONCEPT:AU-ORCH.scheduling.soft-timeout-lease-quarantine), so the
        # lease expires and another worker/host can pick the job back up instead
        # of it being quarantined forever alongside the stuck thread. A late
        # write from the abandoned original run is fenced out by
        # ``_require_live_work_item_lease`` before any terminal commit.
        from .task_lanes import task_soft_timeout

        timeout = task_soft_timeout(task_type)
        heavy = task_type in _HEAVY_TASK_TYPES
        outcome: dict[str, BaseException] = {}
        body_done = threading.Event()

        def _run_body() -> None:
            # CONCEPT:AU-KG.compute.task-priority-tag — tag this task's whole execution with its resource
            # PriorityClass (derived from the SAME lane taxonomy as the worker
            # AdmissionPolicy), so every shared-LLM call it makes inherits the class:
            # an ingestion task's enrichment calls run as BACKGROUND_INGESTION and
            # yield the reserved LLM headroom to interactive/orchestration work, while
            # an on-pool ``queries`` task (conversation/kg_memory) runs INTERACTIVE.
            # Set inside the durable queue worker so the complete task inherits it.
            from agent_utilities.core.resource_priority import (
                priority_for_task_type,
                priority_scope,
            )
            from agent_utilities.core.task_cancellation import (
                raise_if_task_cancelled,
                use_task_cancellation,
            )

            async def _run_owned_task() -> None:
                task = asyncio.create_task(
                    self._run_background_task(
                        job_id, target_path, is_codebase, task_type
                    )
                )
                try:
                    while not task.done():
                        await asyncio.wait({task}, timeout=0.05)
                        raise_if_task_cancelled()
                    await task
                finally:
                    if not task.done():
                        task.cancel()

            try:
                with (
                    priority_scope(priority_for_task_type(task_type)),
                    use_task_cancellation(cancellation_event),
                ):
                    asyncio.run(_run_owned_task())
            except BaseException as exc:  # noqa: BLE001 — relayed to the worker loop
                outcome["exc"] = exc

        def _request_cancellation_at_bound() -> None:
            if body_done.wait(timeout):
                return
            cancellation_event.set()
            logger.warning(
                "[KG-2.286] task %s (%s) exceeded soft timeout %.0fs — "
                "cooperative cancellation requested; its fixed worker stays "
                "quarantined until the in-flight body exits, but its WorkItem "
                "lease STOPS being renewed from here "
                "(CONCEPT:AU-ORCH.scheduling.soft-timeout-lease-quarantine) so "
                "the job becomes reclaimable on lease expiry",
                job_id,
                task_type,
                timeout,
            )

        timeout_watchdog = _authorized_background_thread(
            self._background_session_for_spawn(),
            _request_cancellation_at_bound,
            name=f"kg-task-timeout-{job_id}",
        )
        timeout_watchdog.start()
        try:
            if heavy:
                # Hold the background concurrency slot for the task's whole owned
                # lifetime.  Releasing it while an uncooperative timed-out call was
                # still running would hide real load and over-subscribe the engine.
                #
                # CONCEPT:AU-KG.ontology.capability-card-backfill-lane — ``enrichment_backfill`` is deliberately NOT in
                # ``_HEAVY_TASK_TYPES``, so it runs WITHOUT this outer permit:
                # ``_tick_enrichment`` acquires the background_slot PER BATCH.
                from agent_utilities.core.background_throttle import get_throttle

                with get_throttle().background_slot():
                    _run_body()
            else:
                _run_body()
        finally:
            body_done.set()
            timeout_watchdog.join()

        if cancellation_event.is_set():
            raise RuntimeError(f"soft timeout: {task_type} exceeded {timeout:.0f}s")
        if "exc" in outcome:
            # Re-raise the task's real failure so the worker loop's retry path runs.
            raise outcome["exc"]

        # Post-ingestion: auto-build HNSW indexes when queue drains
        self._maybe_build_vector_indexes()

    def _drain_session_upload(
        self, job_id: str, task_type: str = "session_upload"
    ) -> dict[str, int]:
        """Persist an enqueued session-bundle upload into the usage store.

        CONCEPT:AU-KG.ingest.drain-session-bundle — the ``ingest_sessions`` MCP/REST handler enqueues large
        uploads as a ``session_upload`` WorkItem metadata payload (same shape
        as ``kg_memory``); this runs on the host
        worker, off the request path. ``record_bundle`` is idempotent (replaces
        existing rows) so a retry is safe.
        """
        from agent_utilities.usage.models import ParsedSessionBundle
        from agent_utilities.usage.recorder import get_usage_recorder

        umeta = self._ingest_task_metadata(job_id)
        payload = umeta.get("payload", {}) or {}
        bundles = payload.get("bundles", []) or []
        up_tenant = str(payload.get("tenant_id") or "")
        recorder = get_usage_recorder()
        ok = 0
        for item in bundles:
            try:
                bundle = ParsedSessionBundle.model_validate(item)
            except Exception as exc:  # noqa: BLE001
                logger.warning("session_upload_bad_bundle: %s", exc)
                continue
            if up_tenant:
                bundle.session.tenant_id = up_tenant
            if recorder.record_bundle(bundle):
                ok += 1
        result = {"received": len(bundles), "ingested": ok}
        # The durable queue needed the already privacy-normalized bundle only
        # until acknowledged ingestion.  Do not retain even sanitized session
        # payloads after the WorkItem completes.
        self._update_task_status(
            job_id, "completed", {"type": task_type, "payload": None, **result}
        )
        return result

    async def _run_background_task(
        self, job_id: str, target: Path, is_codebase: bool, task_type: str = "document"
    ):
        """Execute the ingestion logic."""
        try:
            if task_type in ("scheduled_job", "enrichment_backfill"):
                # A recurring job enqueued by the unified scheduler (CONCEPT:AU-OS.state.unified-scheduling-one-intelligent).
                # ``enrichment_backfill`` is the same dispatch, only landed in the
                # dedicated enrichment lane so it isn't capped at the maint floor
                # (CONCEPT:AU-KG.ontology.capability-card-backfill-lane).
                # The payload (the dispatch descriptor) rides on the task metadata;
                # run it through the single dispatcher and let the schedule's own
                # failure backoff govern cadence (so we do NOT route a job failure
                # through the task-level retry — that would double-retry).
                from agent_utilities.core.schedule_engine import (
                    record_schedule_result,
                    run_scheduled_job,
                )

                meta = self._ingest_task_metadata(job_id)
                sched_name = meta.get("schedule", "")
                payload = meta.get("payload", {})
                try:
                    result = run_scheduled_job(self, payload)
                    ok = str(result.get("status", "ok")) not in {"error", "failed"}
                except Exception as e:  # noqa: BLE001 — recorded as a schedule failure
                    result = {"status": "error", "error": str(e)}
                    ok = False
                if sched_name:
                    record_schedule_result(
                        self,
                        sched_name,
                        ok,
                        duration_s=result.get("duration_s"),
                        status=result.get("status"),
                    )
                self._update_task_status(
                    job_id,
                    "completed" if ok else "failed",
                    {
                        "target": str(target),
                        "type": task_type,
                        "schedule": sched_name,
                        "result": result,
                    },
                )
                return
            if task_type == "research_paper_fetch":
                # A high-graded RSS item: download the full paper and ingest it
                # (CONCEPT:AU-KG.research.scholarx-rss-research-feed). Enqueued by the RSS feed screen with a
                # grade-derived priority, so the best papers are fetched first.
                from agent_utilities.automation.research_pipeline import (
                    ResearchPipelineRunner,
                )

                meta = self._ingest_task_metadata(job_id)
                paper = meta.get("paper", {})
                runner = ResearchPipelineRunner(engine=self)  # type: ignore[arg-type]  # self is the engine
                from ..research.cohort import resolve_ephemeral_paper_pdf
                from .ingest_profile import profile_ingest

                # OS-5.69/70 — profile token usage + per-stage timing for this paper.
                with profile_ingest(str(paper.get("id", ""))) as _prof:
                    article_id = await runner.ingest_paper_full(
                        paper.get("id", ""),
                        paper.get("title", ""),
                        paper.get("abstract", ""),
                        paper.get("authors", []),
                        # Resolve a pre-downloaded PDF only in worker memory. The
                        # durable task carries the paper id, never a machine path.
                        pdf_path=resolve_ephemeral_paper_pdf(str(paper.get("id", ""))),
                        source_url=paper.get("url", ""),
                        relevance_score=float(paper.get("score", 0.0) or 0.0),
                        domains=paper.get("domains"),
                    )
                self._update_task_status(
                    job_id,
                    "completed",
                    {
                        "target": paper.get("id", ""),
                        "type": task_type,
                        "article_id": article_id,
                        "score": paper.get("score"),
                        "profile": _prof.to_dict(),
                    },
                )
                return
            if task_type == "kg_memory":
                # CONCEPT:AU-KG.compute.offloaded-memory-write — a memory write offloaded from a SERVING process. The
                # host performs the embed+write here (inline, _local=True so it never
                # re-enqueues), isolating heavy ingestion from the serving/read plane.
                meta = self._ingest_task_metadata(job_id)
                p = meta.get("payload", {})
                mid = self.store_memory(  # type: ignore[attr-defined]  # MemoryMixin, composed onto the engine
                    content=p.get("content", ""),
                    memory_type=p.get("memory_type", "episodic"),
                    name=p.get("name", ""),
                    tags=p.get("tags", []),
                    trust_score=p.get("trust_score", 0.8),
                    agent_id=p.get("agent_ref", ""),
                    extra_props=p.get("extra_props") or None,
                    _local=True,
                    _memory_id=p.get("memory_id"),
                )
                self._update_task_status(
                    job_id, "completed", {"memory_id": mid, "type": "kg_memory"}
                )
                return
            if task_type == "conversation":
                # Process a single conversation from a JSON or overview file
                from agent_utilities.knowledge_graph.core.conversation_ingestion import (
                    ingest_conversations_to_kg,
                    parse_antigravity_logs,
                    parse_claude_logs,
                    parse_codex_logs,
                    parse_windsurf_logs,
                )

                # Determine source from target path
                target_str = str(target)
                convs = []

                if "antigravity" in target_str:
                    # Antigravity target is the parent dir of overview.txt
                    convs = parse_antigravity_logs(target.parent.parent.parent)
                elif "windsurf" in target_str:
                    convs = parse_windsurf_logs(target.parent)
                elif "claude" in target_str:
                    convs = parse_claude_logs(target.parent)
                elif "codex" in target_str:
                    convs = parse_codex_logs(target.parent)

                # Filter for the specific target file
                convs = [c for c in convs if c.get("path") == target_str]

                if not convs:
                    raise Exception(f"Could not parse conversation at {target_str}")

                result = ingest_conversations_to_kg(conversations=convs)
                self._update_task_status(
                    job_id,
                    "completed",
                    {
                        "total_ingested": result.get("total_ingested", 0),
                        "total_messages": result.get("total_messages", 0),
                        "target": target_str,
                        "type": "conversation",
                    },
                )

            elif task_type == "content_url":
                # Content-aware URL ingest OFF the request path (CONCEPT:AU-KG.compute.registered-edge-type):
                # route through the unified IngestionEngine DOCUMENT path so the page
                # is fetched via the resolver (ArchiveBox→crawl4ai→requests) and a
                # research roundup auto-acquires the papers it cites. The real URL
                # rides in WorkItem metadata because the claim path
                # wraps ``target`` in Path() (which would collapse ``https://``).
                from agent_utilities.knowledge_graph.ingestion.engine import (
                    ContentType,
                    IngestionEngine,
                    IngestionManifest,
                )

                tprops = self._ingest_task_metadata(job_id)
                url = str(tprops.get("source_url") or "").strip()
                if not url:
                    # Fallback: repair the Path()-mangled scheme separator.
                    url = re.sub(r"^(https?):/(?!/)", r"\1://", str(target))
                meta = {}
                ep = tprops.get("extract_papers")
                if ep is not None:
                    meta["extract_papers"] = (
                        ep if isinstance(ep, bool) else str(ep).lower() == "true"
                    )
                # CONCEPT:AU-KG.ingest.chunk-overlap-stage — ``ingest_url`` defaults these ON (set by the
                # MCP tool) so a URL ingest gets first-class embedded Chunk objects
                # + contextual-retrieval enrichment, at parity with connector
                # ingestion (KG-2.50) rather than only the plain idea_block chunks.
                for _bool_key in ("chunk_objects", "contextual"):
                    _val = tprops.get(_bool_key)
                    if _val is not None:
                        meta[_bool_key] = (
                            _val
                            if isinstance(_val, bool)
                            else str(_val).lower() == "true"
                        )
                ing = IngestionEngine(kg_engine=self)
                r = await ing.ingest(
                    IngestionManifest(
                        content_type=ContentType.DOCUMENT,
                        source_uri=url,
                        metadata=meta,
                    )
                )
                self._update_task_status(
                    job_id,
                    "completed" if r.status == "success" else "failed",
                    {
                        "target": url,
                        "type": task_type,
                        "status": r.status,
                        "nodes": r.nodes_created,
                        "details": r.details,
                        "error": r.error,
                    },
                )

            elif task_type == "feed_ingest":
                # Async full-ingest of a relevance-gated feed article OFF the sweep
                # path (CONCEPT:AU-KG.ingest.rss-feed-connector). The world-model gate enqueues; the worker
                # pool drains these in parallel, so "reviews" (the sweep) scale
                # independently of "ingest" (chunk + embed + contextual-enrich),
                # and ingest scales 1→N with the model-concurrency controller. The
                # already-fetched article text rides on the task — no re-crawl. Run
                # the (sync) DocumentProcessor in a worker thread so concurrent
                # feed_ingest tasks don't serialize on the event loop.
                from agent_utilities.knowledge_graph.ontology.document_processing import (
                    ChunkingConfig,
                    DocumentProcessor,
                )

                meta_t = self._ingest_task_metadata(job_id)
                fd = (meta_t or {}).get("feed_doc") or {}
                if not fd.get("document_id"):
                    self._update_task_status(
                        job_id,
                        "failed",
                        {"type": task_type, "error": "no feed_doc payload"},
                    )
                else:
                    proc = DocumentProcessor(
                        getattr(self, "backend", None),
                        engine=self,
                        chunking=ChunkingConfig(),
                        contextual=True,
                    )
                    try:
                        await asyncio.to_thread(
                            proc.process,
                            fd.get("text", "") or "",
                            document_id=fd["document_id"],
                            title=fd.get("title") or fd["document_id"],
                            doc_type=fd.get("doc_type", "news_article"),
                            source=fd.get("source", ""),
                            metadata=fd.get("metadata") or {},
                            connector="feed",
                            source_instance="feed-ingest-worker",
                        )
                        # Unified always-on intelligence layer (CONCEPT:AU-KG.enrichment.topic-classification-topology):
                        # DocumentProcessor.process only chunks + contextual-
                        # enriches — route the article body through the SAME
                        # central seam every other ingestion adaptor drains
                        # (concepts + facts + WorldView topic classification) so a
                        # feed_ingest article isn't a shallower write than a
                        # directly-ingested document. Best-effort; never fails the task.
                        try:
                            from agent_utilities.knowledge_graph.ingestion.engine import (
                                IngestionEngine as _IngestionEngine,
                            )

                            await _IngestionEngine(kg_engine=self).enrich_text(
                                fd["document_id"],
                                fd.get("text", "") or "",
                                fd.get("doc_type", "news_article"),
                                fd.get("title") or fd["document_id"],
                            )
                        except Exception:  # noqa: BLE001 — enrichment never breaks the task
                            logger.debug(
                                "[feed_ingest] central enrichment seam failed for %s",
                                fd["document_id"],
                                exc_info=True,
                            )
                        self._update_task_status(
                            job_id,
                            "completed",
                            {"target": fd["document_id"], "type": task_type},
                        )
                    except Exception as fe:  # noqa: BLE001
                        self._update_task_status(
                            job_id,
                            "failed",
                            {
                                "target": fd["document_id"],
                                "type": task_type,
                                "error": str(fe),
                            },
                        )

            elif task_type == "feed_sweep":
                # The RSS/FreshRSS sweep run OFF the request path (CONCEPT:AU-KG.ingest.rss-feed-connector).
                # The sweep is the "review" producer: it fetches (concurrently),
                # runs the world-model gate, and ENQUEUES per-article worldview/
                # research tasks. It does NOT ride the 300s MCP call — graph_feeds
                # sync enqueues this and returns immediately. The gate loop does
                # per-item engine work, so run it in a worker thread.
                from agent_utilities.knowledge_graph.core.source_sync import sync_source

                meta_t = self._ingest_task_metadata(job_id)
                source = str((meta_t or {}).get("feed_source") or "rss")
                fmode = str((meta_t or {}).get("feed_mode") or "delta")
                try:
                    res = await asyncio.to_thread(sync_source, self, source, mode=fmode)
                    self._update_task_status(
                        job_id,
                        "completed",
                        {"target": f"feed:{source}", "type": task_type, "result": res},
                    )
                except Exception as se:  # noqa: BLE001
                    self._update_task_status(
                        job_id,
                        "failed",
                        {
                            "target": f"feed:{source}",
                            "type": task_type,
                            "error": str(se),
                        },
                    )

            elif task_type == "skill_workflows":
                # CONCEPT:AU-KG.ingest.skill-workflow-corpus — ingest the universal-skills workflow corpus as
                # dispatchable WorkflowDefinition DAGs, OFF the request path. The
                # per-node durable writes (~150s for ~315 workflows) exceed the MCP
                # 300s call ceiling, so the action enqueues this job and returns a
                # job_id; the worker runs it to completion here. ``target`` is the
                # corpus root, or the ``"universal-skills"`` sentinel = default
                # installed package.
                from agent_utilities.knowledge_graph.core.engine import (
                    IntelligenceGraphEngine,
                )
                from agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest import (
                    ingest_skill_workflows,
                )

                root = None if str(target) == "universal-skills" else str(target)
                # ``self`` is the engine (this mixin is mixed into it).
                summary = ingest_skill_workflows(
                    cast(IntelligenceGraphEngine, self), root=root
                )
                self._update_task_status(
                    job_id,
                    "completed",
                    {
                        "workflows": summary.get("workflows", 0),
                        "steps": summary.get("steps", 0),
                        "skill_links": summary.get("skill_links", 0),
                        "skipped": summary.get("skipped", 0),
                        "errors": summary.get("errors", 0),
                        "target": str(target),
                        "type": "skill_workflows",
                    },
                )

            elif task_type == "diff":
                # Process a patch file or diff string
                import hashlib

                from agent_utilities.core.embedding_utilities import (
                    create_embedding_model,
                )

                embed_model = create_embedding_model()

                diff_content = (
                    target.read_text(encoding="utf-8", errors="replace")
                    if target.is_file()
                    else str(target)
                )
                if not diff_content.strip():
                    raise Exception("Empty diff content")

                nid = f"diff-{hashlib.sha256(diff_content.encode()).hexdigest()[:8]}"
                embedding = embed_model.get_text_embedding(diff_content)

                props: dict[str, Any] = {
                    "content": diff_content,
                    "embedding": embedding,
                    "target_path": str(target),
                    "last_seen_timestamp": datetime.now(UTC).isoformat(),
                }
                self.add_node(nid, "DiffEntry", properties=props)

                self._update_task_status(
                    job_id,
                    "completed",
                    {
                        "diffs_added": 1,
                        "target": str(target),
                        "type": "diff",
                    },
                )
            elif task_type == "deep_analysis":
                from agent_utilities.core.config import DEFAULT_KG_ANALYSIS_MAX_DEPTH

                # 'target' path is repurposed as the 'query' or 'concept_id' for deep_analysis
                query = str(target)

                # Fetch metadata to track depth
                t_props = self._ingest_task_metadata(job_id)
                current_depth = int(t_props.get("current_depth", 0))
                max_depth = int(t_props.get("max_depth", DEFAULT_KG_ANALYSIS_MAX_DEPTH))

                # While a bulk codebase ingest is draining, run deep_analysis flat
                # (no recursive fan-out) so its 0-node, blocking-LLM jobs don't
                # flood the queue ahead of structural ingest. (CONCEPT:AU-KG.compute.registered-edge-type)
                if max_depth > 0 and self._bulk_ingest_active():
                    logger.info(
                        "deep_analysis: bulk ingest active — capping max_depth to 0 "
                        "(was %d) to defer recursive fan-out",
                        max_depth,
                    )
                    max_depth = 0

                logger.info(
                    f"Executing deep_analysis for {query} (depth {current_depth}/{max_depth})"
                )

                # Call the method from IntelligenceGraphEngine (which this class is mixed into)
                exec_fn = getattr(self, "execute_deep_analysis", None)
                if exec_fn:
                    result = exec_fn(query, max_depth)
                else:
                    result = {
                        "status": "error",
                        "reason": "execute_deep_analysis not found",
                    }

                if result.get("status") == "success":
                    new_targets = result.get("discovered_targets", [])
                    if current_depth < max_depth and new_targets:
                        # Queue subsequent background jobs for discovered concepts
                        for new_target in new_targets:
                            # Avoid immediate loops by checking if it's the exact same query
                            if new_target != query:
                                self.submit_task(
                                    target_path=new_target,
                                    is_codebase=False,
                                    task_type="deep_analysis",
                                    provenance={
                                        "current_depth": current_depth + 1,
                                        "max_depth": max_depth,
                                        "parent_concept": query,
                                    },
                                )

                self._update_task_status(
                    job_id,
                    "completed",
                    {
                        "target": query,
                        "type": "deep_analysis",
                        "depth": current_depth,
                        "result": result,
                    },
                )

            elif is_codebase or task_type == "codebase":
                # Unified path: the async worker and the synchronous MCP/engine
                # callers share ONE implementation — the structural
                # EnrichmentPipeline via IngestionEngine (CONCEPT:AU-KG.coordination.embedder-breaker). The
                # old per-repo subprocess (`--maintain --stage-to-queue`) is
                # gone; LLM enrichment is deferred to the background card daemon.
                from ..ingestion.engine import (
                    ContentType,
                    IngestionEngine,
                    IngestionManifest,
                )

                # Per-repo call-graph community detection is always on. The
                # engine's community_detection is now deterministically bounded
                # (15s wall-clock + iteration cap, epistemic-graph KG-2.16) and
                # loads its scratch tenant in one batch round-trip, so it can no
                # longer hang or stall a bulk load — the old KG_INGEST_FEATURES /
                # KG_INGEST_PROFILE opt-out knobs are gone. (CONCEPT:AU-KG.compute.registered-edge-type)
                # Forward a caller-scoped file subset (CONCEPT:AU-KG.ingest.agent-utilities-checkout): the
                # agent-utilities self-ingest scopes a DIRTY tree to its
                # git-status-modified files via ``only_files`` on the task
                # metadata; pass it through so the ingest engine parses only those.
                cb_meta = self._ingest_task_metadata(job_id)

                # CONCEPT:AU-KG.ingest.subtask-routing-key — big-repo tail: if this is a whole-repo task for a
                # repo large enough to pin one worker/shard for minutes, fan it out
                # into K shard-routed sub-tasks instead of ingesting inline. Returns
                # True when it fanned out (this parent is done); the children run in
                # parallel across the K redb shard writers.
                if self._maybe_fanout_codebase(job_id, target, cb_meta):
                    return

                cb_manifest_meta: dict[str, Any] = {"features": True}
                only_files = cb_meta.get("only_files")
                if isinstance(only_files, list) and only_files:
                    cb_manifest_meta["only_files"] = [
                        str(_resolve_task_target(str(path))) for path in only_files
                    ]
                # CONCEPT:AU-KG.ingest.subtask-routing-key — a split sub-task carries its own routing key so
                # its structural writes land on a distinct per-shard graph
                # (``code:<repo>__s<i>``) instead of the shared ``code:<repo>``.
                route_repo = cb_meta.get("route_repo")
                if route_repo:
                    cb_manifest_meta["route_repo"] = route_repo
                ing = IngestionEngine(kg_engine=self)
                cb_res = await ing.ingest(
                    IngestionManifest(
                        content_type=ContentType.CODEBASE,
                        source_uri=str(target),
                        metadata=cb_manifest_meta,
                    )
                )
                if cb_res.status == "failed":
                    raise Exception(f"Codebase ingestion failed: {cb_res.error}")

                self._update_task_status(
                    job_id,
                    "completed",
                    {
                        "nodes_added": cb_res.nodes_created,
                        "edges_added": cb_res.edges_created,
                        "target": str(target),
                        "type": "codebase",
                        "status": cb_res.status,
                        "cards_pending": cb_res.details.get("cards_pending", 0),
                    },
                )
            elif task_type == "relevance_sweep":
                # Score all ingested papers and codebases against a target
                result = await self._run_relevance_sweep(job_id, str(target))
                self._update_task_status(job_id, "completed", result)
            elif task_type == "self_tool_surface":
                # graph-os registers the provider before publishing this
                # priority-one WorkItem. Keep the ChangeEnvelope write on the
                # bounded memory-generation lane so cold materialization or
                # write contention never blocks the boot-plan producer.
                from agent_utilities.knowledge_graph.ingestion.engine import (
                    IngestionEngine,
                )

                self_tools = await IngestionEngine(kg_engine=self)._ingest_self_tools()
                if self_tools.status == "failed":
                    raise RuntimeError("self tool-surface ingestion failed")
                self._update_task_status(
                    job_id,
                    "completed",
                    {
                        "target": str(target),
                        "type": task_type,
                        "status": self_tools.status,
                        "nodes_added": self_tools.nodes_created,
                        "edges_added": self_tools.edges_created,
                    },
                )
            elif task_type in ("connector_sync", "capability_hydration"):
                # CONCEPT:AU-ORCH.scheduling.connector-sync-lane — one external connector's delta sync, run as a LANED task
                # (the 'connectors' lane). The */20m fleet sweep enqueues one of these per
                # connector so they fan out in PARALLEL instead of one slow connector
                # (gitlab/servicenow) blocking the rest in a sequential inline loop.
                #
                # ``capability_hydration`` (CONCEPT:AU-ORCH.scheduling.acquisition-lane-fairness) is the SAME
                # dispatch — its metadata's ``target`` is always ``"fleet"`` — but a
                # distinct task type so it can be given its own reserved-worker floor
                # (see ``start_task_workers``) instead of competing 1:1 with however
                # many ordinary ``connector_sync`` jobs the */20m sweep has in flight.
                from agent_utilities.knowledge_graph.core.source_sync import sync_source

                connector_meta = self._ingest_task_metadata(job_id)
                mode = str(connector_meta.get("sync_mode") or "delta")
                sync_res = sync_source(self, str(target), mode=mode)
                self._update_task_status(
                    job_id,
                    "completed",
                    {
                        "target": str(target),
                        "type": task_type,
                        **(
                            sync_res
                            if isinstance(sync_res, dict)
                            else {"result": sync_res}
                        ),
                    },
                )
            elif task_type == "connector_drain":
                # CONCEPT:AU-KG.ontology.single-source-full-drain — ONE paginated page of a chunked full-corpus drain. The
                # WorkItem metadata carries the drain identity and resumable
                # connector checkpoint; ``run_drain_page`` drains this page, ingests
                # it, and self-continues by enqueuing the NEXT page-task while the cursor has
                # more — so a single ``source_sync(full)`` drains the whole corpus across many
                # capacity-guarded background tasks without ever blocking the request.
                from agent_utilities.knowledge_graph.core.chunked_drain import (
                    run_drain_page,
                )

                dmeta = self._ingest_task_metadata(job_id)
                drain_res = run_drain_page(
                    self,
                    source=str(dmeta.get("drain_source") or target),
                    mode=str(dmeta.get("sync_mode") or "full"),
                    drain_id=str(dmeta.get("drain_id") or ""),
                    page=int(dmeta.get("drain_page") or 0),
                    checkpoint_json=dmeta.get("drain_checkpoint"),
                )
                self._update_task_status(
                    job_id,
                    "completed",
                    {"target": str(target), "type": task_type, **drain_res},
                )
            elif task_type == "fleet_event_triage":
                # Fleet-event triage (CONCEPT:AU-OS.config.fleet-event-ingress): 'target' is the
                # FleetEvent node id enqueued by the gateway's
                # POST /api/fleet/events webhook receiver, not a filesystem
                # path. Correlates the event to known KG entities and files a
                # failure_gap topic when severity warrants. Remediation
                # playbooks (CONCEPT:AU-OS.host.remediation-playbooks) register on the dispatch seam
                # here, so wherever triage runs they are live.
                from agent_utilities.knowledge_graph.adaptation.fleet_event_triage import (
                    triage_fleet_event,
                )
                from agent_utilities.knowledge_graph.adaptation.remediation_playbooks import (
                    ensure_registered as _ensure_playbooks,
                )

                _ensure_playbooks()
                result = triage_fleet_event(self, str(target))
                self._update_task_status(
                    job_id,
                    "completed",
                    {"target": str(target), "type": task_type, **result},
                )
            elif task_type == "deploy_watch":
                # Health-gated deploy watch (CONCEPT:AU-OS.config.health-gated-deploy-rollback): 'target' is the
                # watched service name; the watch spec (window, deadline,
                # rollback params) rides on the WorkItem, so a reclaimed watch
                # resumes against its original
                # deadline. Failure invokes the policy-gated rollback.
                from agent_utilities.orchestration.deploy_watch import (
                    run_deploy_watch,
                )

                result = run_deploy_watch(self, str(target), job_id)
                self._update_task_status(
                    job_id,
                    "completed",
                    {"target": str(target), "type": task_type, **result},
                )
            elif task_type in ("synthesize", "deep_extract", "background_research"):
                from agent_utilities.analysis.analyzer import GraphAnalyzer

                analyzer = GraphAnalyzer(self)
                query = str(target)

                # Fetch metadata to track top_k if provided
                t_props = self._ingest_task_metadata(job_id)
                top_k = int(t_props.get("top_k", 10))

                try:
                    if task_type == "synthesize":
                        result = await analyzer.synthesize(query, top_k)
                    elif task_type == "deep_extract":
                        result = await analyzer.deep_extract(query)
                    elif task_type == "background_research":
                        result = await analyzer.background_research(query)

                    self._update_task_status(
                        job_id,
                        "completed",
                        {
                            "target": query,
                            "type": task_type,
                            "result": result,
                        },
                    )
                except Exception as e:
                    self._fail_or_retry_task(job_id, str(e), {"type": task_type})

            elif task_type == "cohort_synthesize":
                # Self-polling barrier gate for a research cohort (CONCEPT:AU-KG.coordination.research-cohort-barrier):
                # once every member task is terminal (completed OR failed — a poison
                # member never wedges the cohort) or the deadline passes, run the
                # assimilation pass + materialize the feature matrix over whatever was
                # ingested. Until then re-defer ONE poll interval as 'scheduled' (NOT
                # a failure attempt); native availability makes it claimable later.
                from agent_utilities.knowledge_graph.research.cohort import (
                    cohort_ready,
                    finalize_cohort,
                )

                cmeta = self._ingest_task_metadata(job_id)
                cohort_id = str(cmeta.get("cohort_id") or "")
                deadline = float(cmeta.get("deadline_unix", 0.0) or 0.0)
                ready, member_st = cohort_ready(self, cohort_id, deadline_unix=deadline)
                if not ready:
                    eta = time.time() + 60.0
                    cmeta["eta_unix"] = eta
                    cmeta["member_status"] = member_st
                    from agent_utilities.orchestration import work_item as _wi

                    work_item_id = str(cmeta.get("work_item_id") or "")
                    if not work_item_id:
                        raise _wi.WorkItemBackendUnavailable(
                            f"cohort barrier {job_id} has no authoritative WorkItem"
                        )
                    claim = self._active_work_item_claim(job_id)
                    if claim is None:
                        raise _wi.WorkItemBackendUnavailable(
                            f"cohort barrier {job_id} has no active native claim"
                        )
                    if not _wi.defer_work_item(
                        self._work_item_engine,
                        work_item_id,
                        claim,
                        next_retry_at=eta,
                        reason_ref="cohort_barrier",
                    ):
                        raise _wi.WorkItemBackendUnavailable(
                            f"cohort barrier {job_id} deferral was fenced"
                        )
                    self._active_work_item_claim(job_id, pop=True)
                else:
                    try:
                        result = finalize_cohort(self, cohort_id)
                        self._update_task_status(
                            job_id,
                            "completed",
                            {
                                "type": task_type,
                                "cohort_id": cohort_id,
                                "members": member_st,
                                "feature_matrix": (
                                    result.get("feature_matrix") or {}
                                ).get("counts", {}),
                            },
                        )
                    except Exception as e:
                        self._fail_or_retry_task(job_id, str(e), {"type": task_type})

            elif task_type == "session_upload":
                # CONCEPT:AU-KG.ingest.drain-session-bundle — drain a remote session-bundle upload that the
                # ``ingest_sessions`` MCP/REST handler enqueued (its synchronous
                # record_bundle loop blew past the 60s MCP window). Body extracted
                # to a helper so it is unit-testable without a live worker loop.
                self._drain_session_upload(job_id, task_type)
                return

            else:
                import hashlib

                from llama_index.core import SimpleDirectoryReader

                from agent_utilities.core.embedding_utilities import (
                    create_embedding_model,
                )

                embed_model = create_embedding_model()
                # Override the library default with the governed pypdf adapter,
                # which enforces file, page, and extracted-character bounds.
                pdf_extractor = _pdf_file_extractor()
                if target.is_dir():
                    # exclude_hidden=False is REQUIRED: the research store lives
                    # under ``~/.local/share/...`` and SimpleDirectoryReader treats
                    # any file beneath a dot-dir (``.local``) as hidden, excluding
                    # everything → "No files found" despite PDFs present.
                    # recursive=False skips the ``.metadata`` sidecar dir;
                    # required_exts limits to real documents. (CONCEPT:AU-KG.coordination.embedder-breaker)
                    docs = SimpleDirectoryReader(
                        input_dir=str(target),
                        recursive=False,
                        exclude_hidden=False,
                        required_exts=sorted(SUPPORTED_EXTENSIONS),
                        file_extractor=pdf_extractor,
                    ).load_data()
                else:
                    docs = SimpleDirectoryReader(
                        input_files=[str(target)],
                        exclude_hidden=False,
                        file_extractor=pdf_extractor,
                    ).load_data()

                created = []
                skipped = 0
                ingestion_timestamp = datetime.now(UTC).isoformat()

                # Pass 1 — dedup (O(1) id-keyed lookup per chunk) and collect the NEW
                # chunks. Embeddings are NOT computed here: a per-chunk
                # ``get_text_embedding`` is one network round-trip to the embedding
                # service, and doing it inside this loop made a single PDF take
                # minutes. We gather first, then embed the whole document in one
                # batched call below. (CONCEPT:AU-KG.coordination.embedder-breaker ingestion throughput; see
                # [[epistemic-graph-transport]] — batch over the wire, never per-element.)
                pending: list[tuple[str, str, int, dict[str, Any]]] = []
                for idx, doc in enumerate(docs):
                    chunk_text = doc.text
                    # Sanitize to prevent UnicodeEncodeError (surrogates) when sending to LLM
                    chunk_text = chunk_text.encode("utf-8", errors="replace").decode(
                        "utf-8"
                    )
                    if not chunk_text.strip():
                        continue
                    file_path = doc.metadata.get("file_path", str(target))
                    raw_id = f"{file_path}::{chunk_text}".encode(errors="replace")
                    nid = f"doc-{hashlib.sha256(raw_id).hexdigest()[:8]}"

                    existing = self.query_cypher(
                        "MATCH (n:Article {id: $nid}) RETURN n.id as id", {"nid": nid}
                    )
                    if existing:
                        self.backend.execute(
                            "MATCH (n:Article {id: $nid}) SET n.last_seen_timestamp = $ts",
                            {"nid": nid, "ts": ingestion_timestamp},
                        )
                        skipped += 1
                        continue
                    pending.append((nid, chunk_text, idx, doc.metadata))

                # Pass 2 — batch-embed every new chunk in one shot (sub-batched). The
                # LlamaIndex embedding models expose ``get_text_embedding_batch`` which
                # packs many chunks into a single request; this replaces N serial
                # round-trips with ~N/64, the change that takes a document from minutes
                # to seconds. Fall back to per-chunk only if the model lacks the batch API.
                texts = [c[1] for c in pending]
                embeddings: list = []
                _embed_batch = getattr(embed_model, "get_text_embedding_batch", None)
                if callable(_embed_batch):
                    _BATCH = 64
                    for _i in range(0, len(texts), _BATCH):
                        embeddings.extend(_embed_batch(texts[_i : _i + _BATCH]))
                else:
                    embeddings = [embed_model.get_text_embedding(t) for t in texts]

                for (nid, chunk_text, idx, meta), embedding in zip(
                    pending, embeddings, strict=False
                ):
                    props = {
                        "content": chunk_text,
                        "embedding": embedding,
                        "metadata": json.dumps(meta),
                        "last_seen_timestamp": ingestion_timestamp,
                        "target_path": str(target),
                        "chunk_index": idx,
                    }
                    self.add_node(nid, "Article", properties=props)
                    created.append(nid)

                self.backend.execute(
                    "MATCH (n:Article) WHERE n.target_path = $target AND n.last_seen_timestamp < $ts DETACH DELETE n",
                    {"target": str(target), "ts": ingestion_timestamp},
                )
                self._update_task_status(
                    job_id,
                    "completed",
                    {
                        # ``nodes_added``/``edges_added`` are the canonical keys the
                        # per-category metrics aggregator reads (see
                        # aggregate_ingest_metrics). The async document worker writes
                        # one Article node per new chunk and no edges; surface those
                        # counts here so completed document jobs no longer report 0
                        # nodes. ``chunks_added`` is retained as a descriptive alias.
                        "nodes_added": len(created),
                        "edges_added": 0,
                        "chunks_added": len(created),
                        "chunks_skipped": skipped,
                        "skip_reason": "Hash match exists in DB",
                        "target": str(target),
                        "type": "document",
                    },
                )

        except Exception as e:
            import traceback

            error_msg = str(e)
            error_tb = traceback.format_exc()
            logger.error(f"Task {job_id} failed: {error_tb}")
            # App-level failure: retry with backoff, then dead-letter past the cap
            # (CONCEPT:AU-KG.ingest.hardened-priority-scheduled-task). Native expired-lease recovery is separate.
            self._fail_or_retry_task(
                job_id,
                error_msg,
                {
                    "traceback": error_tb[-4000:],  # last 4000 chars of traceback
                    "target": str(target),
                    "type": task_type,
                },
            )
        finally:
            # Force WAL checkpoint to ensure data persists across server restarts for ALL task types
            self._checkpoint_db()

    async def _run_relevance_sweep(self, job_id: str, target_codebase: str) -> dict:
        """Score all ingested papers and codebases against a target codebase.

        Groups Article nodes by source paper (target_path), groups Code nodes by
        repository. Computes composite relevance scores and persists as
        RELEVANCE_SCORED edges in the KG.

        CONCEPT:AU-KG.retrieval.per-item-relevance-ranking — Per-Item Relevance Ranking
        """
        # Defer while a bulk ingest is in flight: this sweep scores every paper +
        # repo (heavy queries + embeddings) and, as a worker-pool task, runs
        # CONCURRENTLY with ingest on the single-writer engine. It's periodic, so
        # skipping a cycle is cheap — the maintenance scheduler re-enqueues it once
        # the ingest drains. (CONCEPT:AU-KG.compute.registered-edge-type)
        try:
            from agent_utilities.core.background_throttle import get_throttle

            if get_throttle().should_yield_background:
                logger.info(
                    "RelevanceSweep: deferring — bulk ingest/foreground active "
                    "(will retry next cycle)."
                )
                return {"status": "deferred", "reason": "bulk_ingest_or_foreground"}
        except ImportError as exc:  # noqa: BLE001 — ImportError-guarded optional background-throttle signal, same pattern as the enrichment tick above; the sweep proceeds without the early-defer check rather than being lost
            logger.debug(
                "background-throttle check skipped (optional dependency): %s", exc
            )

        logger.info(f"RelevanceSweep: starting sweep against '{target_codebase}'")

        # ── Step 1: Compute target codebase centroid embedding ──
        target_articles = self.query_cypher(
            "MATCH (c:Code) WHERE c.file_path CONTAINS $name "
            "RETURN c.id AS id, c.embedding AS emb LIMIT 200",
            {"name": target_codebase},
        )

        target_embeddings = []
        for row in target_articles:
            emb = row.get("emb")
            if emb and isinstance(emb, list):
                target_embeddings.append(emb)

        if not target_embeddings:
            # Fallback: try Article nodes related to the target
            target_articles = self.query_cypher(
                "MATCH (a:Article) WHERE a.target_path CONTAINS $name "
                "RETURN a.id AS id, a.embedding AS emb LIMIT 100",
                {"name": target_codebase},
            )
            for row in target_articles:
                emb = row.get("emb")
                if emb and isinstance(emb, list):
                    target_embeddings.append(emb)

        if not target_embeddings:
            return {
                "status": "no_target_data",
                "target": target_codebase,
                "message": f"No embeddings found for target '{target_codebase}'",
            }

        # Compute centroid
        from agent_utilities.numeric import xp as np

        centroid = np.mean(target_embeddings, axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm

        # ── Step 2: Gather all unique papers (grouped by target_path) ──
        paper_rows = self.query_cypher(
            "MATCH (a:Article) WHERE a.target_path IS NOT NULL "
            "RETURN DISTINCT a.target_path AS paper_path"
        )
        unique_papers = [r["paper_path"] for r in paper_rows if r.get("paper_path")]

        # ── Step 3: Gather all unique repositories (grouped by file_path prefix) ──
        code_rows = self.query_cypher(
            "MATCH (c:Code) WHERE c.file_path IS NOT NULL "
            "RETURN c.id AS id, c.file_path AS path LIMIT 2000"
        )
        repo_set: set[str] = set()
        for row in code_rows:
            path = row.get("path", "")
            if not path:
                continue
            parts = path.split("/")
            if len(parts) >= 6:
                repo_name = parts[5] if "agent-packages" in path else parts[4]
                if repo_name != target_codebase:
                    repo_set.add(repo_name)

        logger.info(
            f"RelevanceSweep: scoring {len(unique_papers)} papers + {len(repo_set)} repos"
        )

        # ── Step 4: Score each paper ──
        scored_items = []
        timestamp = datetime.now(UTC).isoformat()

        for paper_path in unique_papers:
            try:
                # Get all chunks for this paper
                chunks = self.query_cypher(
                    "MATCH (a:Article) WHERE a.target_path = $path "
                    "RETURN a.id AS id, a.embedding AS emb, a.content AS content LIMIT 50",
                    {"path": paper_path},
                )

                if not chunks:
                    continue

                # Compute paper-level embedding (mean of chunk embeddings)
                paper_embeddings = []
                paper_content_sample = ""
                for chunk in chunks:
                    emb = chunk.get("emb")
                    if emb and isinstance(emb, list):
                        paper_embeddings.append(emb)
                    if not paper_content_sample and chunk.get("content"):
                        paper_content_sample = chunk["content"][:500]

                if not paper_embeddings:
                    continue

                paper_centroid = np.mean(paper_embeddings, axis=0)
                paper_norm = np.linalg.norm(paper_centroid)
                if paper_norm > 0:
                    paper_centroid = paper_centroid / paper_norm

                # Semantic similarity (cosine)
                semantic_score = float(np.dot(centroid, paper_centroid)) * 30.0
                semantic_score = max(0.0, min(30.0, semantic_score))

                # Content keyword overlap (concept-level)
                content_lower = paper_content_sample.lower()
                concept_keywords = [
                    "knowledge graph",
                    "orchestration",
                    "agent",
                    "mcp",
                    "pydantic",
                    "memory",
                    "embedding",
                    "protocol",
                    "reasoning",
                    "multi-agent",
                    "context",
                    "planning",
                    "tool",
                    "inference",
                    "coordination",
                ]
                overlap_count = sum(1 for kw in concept_keywords if kw in content_lower)
                concept_score = min(20.0, overlap_count * 4.0)

                # Architecture match (heuristic based on content signals)
                arch_keywords = [
                    "plugin",
                    "mixin",
                    "factory",
                    "protocol",
                    "registry",
                    "dependency injection",
                    "event-driven",
                    "microservice",
                ]
                arch_count = sum(1 for kw in arch_keywords if kw in content_lower)
                arch_score = min(20.0, arch_count * 5.0)

                # Innovation potential (unique concepts)
                innovation_keywords = [
                    "novel",
                    "propose",
                    "introduce",
                    "framework",
                    "benchmark",
                    "state-of-the-art",
                    "outperform",
                    "sota",
                    "contribution",
                ]
                innov_count = sum(
                    1 for kw in innovation_keywords if kw in content_lower
                )
                innovation_score = min(20.0, innov_count * 5.0)

                # Feasibility (integration ease)
                feasibility_keywords = [
                    "python",
                    "pip",
                    "api",
                    "library",
                    "open-source",
                    "github",
                ]
                feas_count = sum(
                    1 for kw in feasibility_keywords if kw in content_lower
                )
                feasibility_score = min(10.0, feas_count * 2.5)

                composite = (
                    semantic_score
                    + concept_score
                    + arch_score
                    + innovation_score
                    + feasibility_score
                )
                composite = round(min(100.0, composite), 2)

                item_id = f"paper:{Path(paper_path).stem}"
                scored_items.append(
                    {
                        "id": item_id,
                        "type": "paper",
                        "path": paper_path,
                        "score": composite,
                        "semantic": round(semantic_score, 2),
                        "concept_overlap": round(concept_score, 2),
                        "arch_compat": round(arch_score, 2),
                        "innovation": round(innovation_score, 2),
                        "feasibility": round(feasibility_score, 2),
                    }
                )

                # Persist as edge in KG
                self._persist_relevance_score(
                    item_id,
                    target_codebase,
                    composite,
                    semantic_score,
                    concept_score,
                    arch_score,
                    innovation_score,
                    feasibility_score,
                    timestamp,
                )

            except Exception as e:
                logger.warning("RelevanceSweep: paper scoring failed: %s", e)

        # ── Step 5: Score each repository ──
        for repo_name in repo_set:
            try:
                repo_chunks = self.query_cypher(
                    "MATCH (c:Code) WHERE c.file_path CONTAINS $name "
                    "RETURN c.id AS id, c.embedding AS emb, c.content AS content LIMIT 100",
                    {"name": repo_name},
                )

                if not repo_chunks:
                    continue

                repo_embeddings = []
                repo_content_sample = ""
                for chunk in repo_chunks:
                    emb = chunk.get("emb")
                    if emb and isinstance(emb, list):
                        repo_embeddings.append(emb)
                    if not repo_content_sample and chunk.get("content"):
                        repo_content_sample = chunk["content"][:500]

                if not repo_embeddings:
                    continue

                repo_centroid = np.mean(repo_embeddings, axis=0)
                repo_norm = np.linalg.norm(repo_centroid)
                if repo_norm > 0:
                    repo_centroid = repo_centroid / repo_norm

                semantic_score = float(np.dot(centroid, repo_centroid)) * 30.0
                semantic_score = max(0.0, min(30.0, semantic_score))

                content_lower = repo_content_sample.lower()
                concept_keywords = [
                    "knowledge graph",
                    "orchestration",
                    "agent",
                    "mcp",
                    "pydantic",
                    "memory",
                    "embedding",
                    "protocol",
                    "reasoning",
                    "multi-agent",
                ]
                concept_score = min(
                    20.0, sum(1 for kw in concept_keywords if kw in content_lower) * 4.0
                )

                arch_keywords = [
                    "plugin",
                    "mixin",
                    "factory",
                    "protocol",
                    "registry",
                    "dependency injection",
                ]
                arch_score = min(
                    20.0, sum(1 for kw in arch_keywords if kw in content_lower) * 5.0
                )

                innovation_score = 10.0  # Codebases get baseline innovation score
                feasibility_score = 8.0  # Codebases are inherently more feasible

                composite = (
                    semantic_score
                    + concept_score
                    + arch_score
                    + innovation_score
                    + feasibility_score
                )
                composite = round(min(100.0, composite), 2)

                item_id = f"repo:{repo_name}"
                scored_items.append(
                    {
                        "id": item_id,
                        "type": "codebase",
                        "name": repo_name,
                        "score": composite,
                        "semantic": round(semantic_score, 2),
                        "concept_overlap": round(concept_score, 2),
                        "arch_compat": round(arch_score, 2),
                        "innovation": round(innovation_score, 2),
                        "feasibility": round(feasibility_score, 2),
                    }
                )

                self._persist_relevance_score(
                    item_id,
                    target_codebase,
                    composite,
                    semantic_score,
                    concept_score,
                    arch_score,
                    innovation_score,
                    feasibility_score,
                    timestamp,
                )

            except Exception as e:
                logger.warning(f"RelevanceSweep: error scoring repo {repo_name}: {e}")

        # Sort by composite score descending
        scored_items.sort(key=lambda x: x["score"], reverse=True)

        logger.info(
            f"RelevanceSweep: completed — {len(scored_items)} items scored against '{target_codebase}'"
        )

        return {
            "status": "completed",
            "target_codebase": target_codebase,
            "items_scored": len(scored_items),
            "top_10": scored_items[:10],
            "scored_at": timestamp,
            "type": "relevance_sweep",
        }

    def _persist_relevance_score(
        self,
        item_id: str,
        target_codebase: str,
        composite: float,
        semantic: float,
        concept_overlap: float,
        arch_compat: float,
        innovation: float,
        feasibility: float,
        timestamp: str,
    ) -> None:
        """Persist a relevance score as a node + edge in the KG."""
        try:
            # Ensure the item node exists
            self.add_node(
                item_id,
                "Article",
                properties={
                    "relevance_score": composite,
                    "relevance_target": target_codebase,
                    "relevance_scored_at": timestamp,
                },
            )

            # Ensure target codebase node exists
            target_id = f"codebase:{target_codebase}"
            self.add_node(
                target_id,
                "Code",
                properties={
                    "name": target_codebase,
                    "node_type": "codebase_root",
                },
            )

            # Create RELEVANCE_SCORED edge (CONCEPT:AU-KG.compute.registered-edge-type — registered edge type)
            from ...models.knowledge_graph import RegistryEdgeType

            self.link_nodes(
                item_id,
                target_id,
                RegistryEdgeType.RELEVANCE_SCORED,
                properties={
                    "score": composite,
                    "semantic": semantic,
                    "concept_overlap": concept_overlap,
                    "arch_compat": arch_compat,
                    "innovation": innovation,
                    "feasibility": feasibility,
                    "scored_at": timestamp,
                    "scorer_version": "0.12.0",
                },
            )
        except Exception as e:  # noqa: BLE001 — one relevance-scoring edge write inside the sweep's per-item loop; a failed write is simply absent from the next query_relevance_rankings read, it does not falsely appear scored
            logger.debug(f"RelevanceSweep: edge persistence error for {item_id}: {e}")

    def query_relevance_rankings(
        self, target_codebase: str, top_k: int = 20
    ) -> list[dict]:
        """Query pre-computed relevance rankings from the KG.

        CONCEPT:AU-KG.retrieval.per-item-relevance-ranking — Per-Item Relevance Ranking
        """
        try:
            results = self.query_cypher(
                "MATCH (item)-[r:RELEVANCE_SCORED]->(target:Code) "
                "WHERE target.name = $codebase "
                "RETURN item.id AS id, r.score AS score, r.semantic AS semantic, "
                "r.concept_overlap AS concept_overlap, r.arch_compat AS arch_compat, "
                "r.innovation AS innovation, r.feasibility AS feasibility, "
                "r.scored_at AS scored_at "
                "ORDER BY r.score DESC LIMIT $top_k",
                {"codebase": target_codebase, "top_k": top_k},
            )
            return results
        except Exception as e:
            logger.error(f"Relevance ranking query failed: {e}")
            return []

    def _maybe_build_vector_indexes(self) -> None:
        """Auto-build HNSW vector indexes when the ingestion queue is fully drained.

        Only rebuilds indexes for tables that were dropped during this batch.
        Checks if there are no pending or running tasks left. If so, builds
        HNSW indexes in a separate background thread to avoid blocking the worker.
        Uses a flag to ensure this only fires once per ingestion batch.
        """
        if not self.backend:
            return

        # WorkItem decides whether queue work remains.
        if self.ingest_queue_depth() > 0:
            return

        # Use a lock + flag so only one worker triggers the build
        if not hasattr(self, "_index_build_lock"):
            self._index_build_lock = threading.Lock()
        if not hasattr(self, "_indexes_built"):
            self._indexes_built = False

        # Capture which tables need rebuilding
        tables_to_build = list(getattr(self, "_dropped_tables", set()))

        with self._index_build_lock:
            if self._indexes_built:
                return
            self._indexes_built = True

        def _build():
            try:
                if tables_to_build:
                    logger.info(
                        "Ingestion queue drained — rebuilding HNSW indexes for: %s",
                        ", ".join(tables_to_build),
                    )
                else:
                    logger.info(
                        "Ingestion queue drained — building all HNSW vector indexes..."
                    )
                if hasattr(self.backend, "build_vector_indices"):
                    self.backend.build_vector_indices(tables=tables_to_build or None)
                    logger.info("HNSW vector indexes built successfully.")
                else:
                    logger.debug("Backend does not support vector index building.")
            except Exception as e:
                logger.warning(f"Post-ingestion vector index build failed: {e}")
            finally:
                # Reset flags so future ingestion batches re-trigger the cycle
                with self._index_build_lock:
                    self._indexes_built = False
                if hasattr(self, "_dropped_tables"):
                    self._dropped_tables = set()

        _authorized_background_thread(
            self._background_session_for_spawn(),
            _build,
            name="KG-IndexBuilder",
        ).start()

    def _update_task_status(
        self, job_id: str, status: str, metadata: dict[str, Any]
    ) -> None:
        """Commit a terminal outcome through the active native WorkItem claim."""
        outcome = _INGEST_TERMINAL_STATUS_TO_WORK_ITEM.get(status)
        if outcome is None:
            raise ValueError(f"unsupported terminal ingestion status: {status!r}")
        from agent_utilities.orchestration import work_item as _wi

        claim = self._active_work_item_claim(job_id)
        if claim is None:
            raise _wi.WorkItemBackendUnavailable(
                f"ingestion job {job_id!r} has no active native WorkItem claim"
            )
        self._require_live_work_item_lease(job_id, claim)
        work_item_id = str(
            claim.get("work_item_id") or _wi.ingest_task_work_item_id(job_id)
        )
        result = _wi.commit_result(
            self._work_item_engine,
            work_item_id,
            claim,
            outcome=outcome,
            retryable=False,
            result_ref=f"outcome:ingest_task:{job_id}"
            if outcome == "succeeded"
            else None,
            error_ref=f"ingest_task:{job_id}:{status}"
            if outcome != "succeeded"
            else None,
        )
        if result not in {"committed", "noop", "dead_letter"}:
            raise _wi.WorkItemBackendUnavailable(
                f"WorkItem commit for {job_id} was rejected ({result})"
            )
        self._active_work_item_claim(job_id, pop=True)
        self._checkpoint_db()

    def _fail_or_retry_task(
        self, job_id: str, error: str, details: dict[str, Any] | None = None
    ) -> None:
        """Commit an application failure through native retry policy."""
        from agent_utilities.orchestration import work_item as _wi

        claim = self._active_work_item_claim(job_id)
        if claim is None:
            raise _wi.WorkItemBackendUnavailable(
                f"ingestion job {job_id!r} has no active native WorkItem claim"
            )
        self._require_live_work_item_lease(job_id, claim)
        work_item_id = str(
            claim.get("work_item_id") or _wi.ingest_task_work_item_id(job_id)
        )
        result = _wi.commit_result(
            self._work_item_engine,
            work_item_id,
            claim,
            outcome="failed",
            retryable=True,
            error_ref=f"ingest_task:{job_id}:{error}",
        )
        if result not in {"committed", "noop", "dead_letter", "retry_scheduled"}:
            raise _wi.WorkItemBackendUnavailable(
                f"WorkItem failure commit for {job_id} was rejected ({result})"
            )
        self._active_work_item_claim(job_id, pop=True)
        if result == "dead_letter":
            logger.warning("ingestion WorkItem %s exhausted retries", job_id)

    def _defer_task_for_materialization(
        self, job_id: str, materialization: dict[str, Any]
    ) -> bool:
        """Release a claimed task immediately without consuming an attempt.

        GraphOS does its bounded materialization wait before workers start, so
        an in-flight task must never wait behind the same graph-wide gate while
        holding a short renewable lease.  The engine-native defer is one fenced
        transition: success restores the exact prior attempt count; a rejected
        transition means this worker no longer owns the lease and is never
        converted into an application failure.
        """
        from agent_utilities.orchestration import work_item as _wi

        claim = self._active_work_item_claim(job_id)
        if claim is None:
            raise _wi.WorkItemBackendUnavailable(
                f"ingestion job {job_id!r} has no active native WorkItem claim"
            )
        work_item_id = str(
            claim.get("work_item_id") or _wi.ingest_task_work_item_id(job_id)
        )
        logger.debug(
            "Deferring ingestion job %s at materialization phase=%s cursor=%s",
            job_id,
            materialization.get("phase"),
            materialization.get("completeness_cursor"),
        )
        try:
            deferred = _wi.defer_work_item(
                self._work_item_engine,
                work_item_id,
                claim,
                next_retry_at=time.time() + 1.0,
                reason_ref="partial_materialization",
            )
        finally:
            self._active_work_item_claim(job_id, pop=True)
        return bool(deferred)

    def _defer_task_for_admission(self, job_id: str) -> bool:
        """Release a claim the :class:`AdmissionPolicy` denied, without consuming an attempt.

        The native ``claim_next`` CAS commits the lease before the task's lane/
        type is known, so a candidate the pool's own admission rules refuse
        (heavy-type cap, hot-spare reservation, interactive floor, best-effort
        lane cap, memory-gen pool cap, …) is un-claimed the same way a
        materialization-blocked task is: a fenced, near-immediate-retry defer
        rather than running work this host's own scheduler just decided it
        should hold back on. Mirrors :meth:`_defer_task_for_materialization`.
        """
        from agent_utilities.orchestration import work_item as _wi

        claim = self._active_work_item_claim(job_id)
        if claim is None:
            return False
        work_item_id = str(
            claim.get("work_item_id") or _wi.ingest_task_work_item_id(job_id)
        )
        try:
            deferred = _wi.defer_work_item(
                self._work_item_engine,
                work_item_id,
                claim,
                next_retry_at=time.time() + 1.0,
                reason_ref="admission_denied",
            )
        finally:
            self._active_work_item_claim(job_id, pop=True)
        return bool(deferred)

    def aggregate_ingest_metrics(self, window_sec: int = 86400) -> dict[str, Any]:
        """Per-category ingest metrics from ingestion WorkItems.

        Powers the MCP ``graph_ingest`` jobs/job_status breakdown so polling shows
        time/nodes/edges/failures per content type — the same view the harness
        writes to ``progress.json``.
        """
        work = self._ingest_work_item_index()
        cutoff = None
        if window_sec:
            try:
                cutoff = datetime.now(UTC) - timedelta(seconds=window_sec)
            except Exception:  # noqa: BLE001
                cutoff = None
        cats: dict[str, dict[str, Any]] = {}
        for item in work.values():
            meta = item.get("metadata") or {}
            if cutoff is not None:
                ca = item.get("completed_at")
                if ca:
                    try:
                        completed = (
                            datetime.fromtimestamp(float(ca), UTC)
                            if isinstance(ca, int | float)
                            else datetime.fromisoformat(str(ca))
                        )
                        if completed < cutoff:
                            continue
                    except (ValueError, TypeError) as exc:  # noqa: BLE001 — unparseable completed_at timestamp; the item is included un-window-filtered (as the log message and the comment above it describe) rather than silently dropped from the aggregate — the safer direction for a metrics report
                        logger.debug(
                            "ingest metrics: unparseable completed_at %r, not window-filtered: %s",
                            ca,
                            exc,
                        )
            cat = meta.get("type") or meta.get("content_type") or "unknown"
            c = cats.setdefault(
                cat,
                {
                    "jobs": 0,
                    "completed": 0,
                    "failed": 0,
                    "nodes": 0,
                    "edges": 0,
                    "duration_ms": 0.0,
                },
            )
            c["jobs"] += 1
            st = _task_status_from_work_item(item)
            if st in ("completed", "done", "success"):
                c["completed"] += 1
            elif st in ("failed", "dead_letter", "error"):
                c["failed"] += 1
            c["nodes"] += int(
                meta.get("nodes_added", meta.get("nodes_created", 0)) or 0
            )
            c["edges"] += int(
                meta.get("edges_added", meta.get("edges_created", 0)) or 0
            )
            submitted = item.get("submitted_at")
            completed_at = item.get("completed_at")
            if isinstance(submitted, int | float) and isinstance(
                completed_at, int | float
            ):
                c["duration_ms"] += max(0.0, (completed_at - submitted) * 1000.0)
        for c in cats.values():
            c["duration_ms"] = round(c["duration_ms"], 1)
        return cats

    def profile_report(
        self, window_sec: int = 86400, group_by: str = "lane"
    ) -> dict[str, Any]:
        """Per-lane / per-stage latency profile from WorkItems and profile spans.

        Where ``aggregate_ingest_metrics`` sums per content TYPE, this groups by a
        chosen dimension — ``lane`` (the functional task lane), ``type`` (the task
        type / pipeline stage), or ``tkind`` — and reports latency PERCENTILES
        (p50/p95/max) plus token/cost totals and a **parallelism factor** (sum of
        per-task durations ÷ wall-clock span). That is exactly the measurement a
        profiling run needs to PROVE a speed-up: the same corpus before vs after an
        optimization, and how much pipelining the staged lanes actually buy.

        This is a read-only projection and adds no competing work-state writer.
        """

        def _pct(values: list[float], p: float) -> float:
            if not values:
                return 0.0
            if len(values) == 1:
                return values[0]
            k = (len(values) - 1) * (p / 100.0)
            lo = int(k)
            hi = min(lo + 1, len(values) - 1)
            frac = k - lo
            return values[lo] * (1 - frac) + values[hi] * frac

        work = self._ingest_work_item_index()
        rows: list[dict[str, Any]] = []
        for job_id, item in work.items():
            meta = dict(item.get("metadata") or {})
            meta.setdefault("lane", item.get("resource_class"))
            meta.setdefault("tkind", item.get("fairness_group"))
            rows.append(
                {
                    "id": job_id,
                    "status": _task_status_from_work_item(item),
                    "meta": meta,
                    "submitted_at": item.get("submitted_at"),
                    "completed_at": item.get("completed_at"),
                }
            )
        # OS-5.71 — fold in off-queue profile spans (assimilation, embed-backfill,
        # concept-registry embedding) so the report covers paths that never become
        # WorkItems. They carry a work-shaped envelope (type='offqueue:<kind>').
        try:
            spans = self._control_cypher(
                "MATCH (s:ProfileSpan) RETURN 'completed' as status, s.metadata as meta"
            )
            if spans:
                rows = list(rows or []) + list(spans)
        except Exception:  # noqa: BLE001 — spans are best-effort, never block the report
            pass
        cutoff = None
        if window_sec:
            try:
                cutoff = datetime.now(UTC) - timedelta(seconds=window_sec)
            except Exception:  # noqa: BLE001
                cutoff = None

        key = group_by if group_by in ("lane", "type", "tkind") else "lane"
        groups: dict[str, dict[str, Any]] = {}
        starts: list[float] = []
        ends: list[float] = []
        # CONCEPT:AU-KG.compute.p99-latency-metric — keep each work identity+duration so the
        # report can name the slowest-N outliers (the p95/max offenders), not just
        # per-lane percentiles. This is what makes a 13-min codebase pin or a 456s
        # hung connector VISIBLE as a specific task, not a lane statistic.
        tail_tasks: list[dict[str, Any]] = []
        for r in rows or []:
            raw_meta = r.get("meta")
            meta = (
                dict(raw_meta)
                if isinstance(raw_meta, dict)
                else _decode_metadata(raw_meta)
            )
            ca = r.get("completed_at") or meta.get("completed_at")
            if cutoff is not None and ca:
                try:
                    completed_dt = (
                        datetime.fromtimestamp(float(ca), UTC)
                        if isinstance(ca, int | float)
                        else datetime.fromisoformat(str(ca))
                    )
                    if completed_dt < cutoff:
                        continue
                except (ValueError, TypeError) as exc:  # noqa: BLE001 — unparseable completed_at timestamp in the tail-tasks profile report — same documented un-window-filtered fallback as aggregate_ingest_metrics above
                    logger.debug(
                        "ingest tail tasks: unparseable completed_at %r, not window-filtered: %s",
                        ca,
                        exc,
                    )
            g = (
                meta.get(key)
                or meta.get("type")
                or meta.get("content_type")
                or "unknown"
            )
            grp = groups.setdefault(
                g,
                {
                    "count": 0,
                    "completed": 0,
                    "failed": 0,
                    "dead_letter": 0,
                    "_durations": [],
                    "tokens": 0,
                    "cost": 0.0,
                    "nodes": 0,
                    "edges": 0,
                    "llm_calls": 0,
                    "embed_calls": 0,
                    "_stages": {},
                },
            )
            grp["count"] += 1
            st = (r.get("status") or "").lower()
            if st in ("completed", "done", "success"):
                grp["completed"] += 1
            elif st in ("failed", "error"):
                grp["failed"] += 1
            elif st == "dead_letter":
                grp["dead_letter"] += 1
            dur = float(meta.get("duration_ms", 0) or 0)
            submitted = r.get("submitted_at")
            completed = r.get("completed_at")
            if (
                dur <= 0
                and isinstance(submitted, int | float)
                and isinstance(completed, int | float)
            ):
                dur = max(0.0, (completed - submitted) * 1000.0)
            if dur > 0:
                grp["_durations"].append(dur)
                # CONCEPT:AU-KG.compute.p99-latency-metric — record the per-task tail entry.
                tail_tasks.append(
                    {
                        "id": r.get("id"),
                        "duration_ms": round(dur, 1),
                        "type": meta.get("type")
                        or meta.get("content_type")
                        or "unknown",
                        "lane": meta.get("lane") or g,
                        "status": st,
                        "target": str(meta.get("target", ""))[:120],
                    }
                )
            usage = meta.get("usage") or {}
            # OS-5.69/70 — the ingest profile carries real token usage + per-stage
            # timing (read/extract/embed/write), so the report is no longer tokens=0
            # and can show WHERE ingest time goes.
            prof = meta.get("profile") or {}
            grp["tokens"] += int(
                meta.get("tokens", usage.get("total", prof.get("total_tokens", 0))) or 0
            )
            grp["cost"] += float(
                meta.get("cost", usage.get("cost", prof.get("cost", 0))) or 0
            )
            grp["llm_calls"] += int(prof.get("llm_calls", 0) or 0)
            grp["embed_calls"] += int(prof.get("embed_calls", 0) or 0)
            for _sname, _sms in (prof.get("stages_ms") or {}).items():
                grp["_stages"].setdefault(_sname, []).append(float(_sms or 0))
            grp["nodes"] += int(
                meta.get("nodes_added", meta.get("nodes_created", 0)) or 0
            )
            grp["edges"] += int(
                meta.get("edges_added", meta.get("edges_created", 0)) or 0
            )
            for ts, bucket in (
                (r.get("submitted_at") or meta.get("started_at"), starts),
                (ca, ends),
            ):
                if ts:
                    try:
                        bucket.append(
                            float(ts)
                            if isinstance(ts, int | float)
                            else datetime.fromisoformat(str(ts)).timestamp()
                        )
                    except (ValueError, TypeError) as exc:  # noqa: BLE001 — unparseable timestamp is excluded from this one latency bucket (as the log message says) — a metrics-precision loss, not a correctness issue, since the item is simply absent from the bucket
                        logger.debug(
                            "ingest metrics: unparseable timestamp %r, excluded from bucket: %s",
                            ts,
                            exc,
                        )

        for grp in groups.values():
            durs = sorted(grp.pop("_durations"))
            grp["total_ms"] = round(sum(durs), 1)
            grp["p50_ms"] = round(_pct(durs, 50), 1)
            grp["p95_ms"] = round(_pct(durs, 95), 1)
            # CONCEPT:AU-KG.compute.p99-latency-metric — surface p99 alongside p95/max so a thin tail (a
            # few outliers) is distinguishable from a fat one at the lane level.
            grp["p99_ms"] = round(_pct(durs, 99), 1)
            grp["max_ms"] = round(durs[-1], 1) if durs else 0.0
            grp["cost"] = round(grp["cost"], 4)
            # per-stage p50 / total across the group's ingests (OS-5.70)
            grp["stages_ms"] = {
                s: {
                    "p50": round(_pct(sorted(v), 50), 1),
                    "total": round(sum(v), 1),
                    "n": len(v),
                }
                for s, v in grp.pop("_stages").items()
            }

        total_ms = sum(g["total_ms"] for g in groups.values())
        wall_ms = (max(ends) - min(starts)) * 1000.0 if starts and ends else 0.0
        # CONCEPT:AU-KG.compute.p99-latency-metric — the slowest-N tasks overall: the concrete outliers a
        # profiling run hunts (the big-repo pin, the hung connector/maint tick).
        tail_tasks.sort(key=lambda t: t["duration_ms"], reverse=True)
        slowest_n = 10
        return {
            "group_by": key,
            "groups": groups,
            "parallelism_factor": round(total_ms / wall_ms, 2) if wall_ms > 0 else 0.0,
            "wall_ms": round(wall_ms, 1),
            "total_task_ms": round(total_ms, 1),
            "slowest": tail_tasks[:slowest_n],
        }

    def _checkpoint_db(self) -> None:
        """Force a WAL checkpoint so a SQLite-backed store persists across restarts.

        Only a backend that exposes an explicit ``wal_checkpoint()`` (a real
        SQLite WAL) is checkpointed. Graph and mirror backends route
        ``execute()`` through the Cypher engine, so the previous raw
        ``execute("CHECKPOINT;")`` fallback misparsed that string into a node
        query and **blocked indefinitely on the engine** — deadlocking every
        task worker after each ``_update_task_status``. There is nothing to
        WAL-checkpoint on those backends, so they are skipped. (CONCEPT:AU-KG.coordination.embedder-breaker)
        """
        wal = getattr(self.backend, "wal_checkpoint", None)
        if not callable(wal):
            return
        try:
            wal()
            logger.debug("WAL checkpoint completed (native).")
        except Exception as e:  # noqa: BLE001 — checkpoint is best-effort
            logger.debug("WAL checkpoint skipped: %s", e)

    def get_task_status(self, job_id: str) -> dict | None:
        """Render one ingestion WorkItem using the public job vocabulary."""
        from agent_utilities.orchestration import work_item as _wi

        item = _wi.get_work_item(
            self._work_item_engine, _wi.ingest_task_work_item_id(job_id)
        )
        if item is None or item.get("kind") != "ingest_task":
            return None
        status = _task_status_from_work_item(item)

        return {
            "job_id": job_id,
            "status": status,
            "metadata": dict(item.get("metadata") or {}),
            "attempt": item.get("attempt"),
            "max_attempts": item.get("max_attempts"),
            "resource_class": item.get("resource_class"),
            "lease_expires_at": item.get("lease_expires_at"),
            "heartbeat_at": item.get("heartbeat_at"),
            "updated_at": item.get("updated_at"),
        }

    def list_tasks(self) -> dict:
        """Group ingestion WorkItems by their rendered public status."""
        work = self._ingest_work_item_index()
        response: dict[str, Any] = {
            "running": [],
            "pending": [],
            "scheduled": [],
            "blocked": [],
            "completed": [],
            "failed": [],
            "cancelled": [],
            "dead_letter": [],
            "unknown": [],
        }

        for job_id, item in work.items():
            status = _task_status_from_work_item(item)
            meta = item.get("metadata") or {}
            job_info: dict[str, Any] = {
                "job_id": job_id,
                "target": meta.get("target", "unknown"),
            }
            if status in {"failed", "dead_letter"}:
                job_info["error"] = item.get("error_ref") or "Unknown error"
                response[status].append(job_info)
            elif status in response:
                if status == "completed":
                    # Include result summary for completed jobs
                    for key in (
                        "chunks_added",
                        "nodes_added",
                        "edges_added",
                        "diffs_added",
                        "chunks_skipped",
                        "skip_reason",
                    ):
                        if key in meta:
                            job_info[key] = meta[key]
                response[status].append(job_info)

        total_tasks = sum(len(items) for items in response.values())

        if total_tasks > 0:
            completed_count = len(response["completed"])
            progress = round((completed_count / total_tasks) * 100, 2)
            response["progress_percentage"] = f"{progress}% complete"
            response["progress_stats"] = {
                "total_tasks": total_tasks,
                "completed": completed_count,
                "pending_in_graph": len(response["pending"]),
                "running_in_graph": len(response["running"]),
                "scheduled": len(response["scheduled"]),
                "blocked": len(response["blocked"]),
            }

        return response

    def remove_task(self, job_id: str) -> bool:
        """WorkItem audit records are immutable and cannot be removed."""
        return False

    def clear_completed_tasks(self) -> dict:
        """Reject deletion of immutable WorkItem audit records."""
        return {
            "status": "error",
            "error": "WorkItem audit records cannot be cleared",
            "cleared": 0,
            "remaining": len(self._ingest_work_item_index()),
        }

    def cancel_task(self, job_id: str) -> dict:
        """Cancel a single queued/running task by id (terminal 'cancelled').

        The native engine owns cancellation and preserves the audit record.
        """
        if not job_id:
            return {"status": "error", "error": "job_id required"}
        try:
            from agent_utilities.orchestration import work_item as _wi

            item_id = _wi.ingest_task_work_item_id(job_id)
            prior = _wi.get_work_item(self._work_item_engine, item_id)
            if prior is None:
                return {"status": "error", "error": f"job {job_id} not found"}
            cancelled = _wi.cancel_work_item(
                self._work_item_engine,
                item_id,
                reason="cancel_task",
            )
        except Exception as e:  # noqa: BLE001 — public control API is structured
            return {"status": "error", "error": f"WorkItem cancel failed: {e}"}
        if not cancelled:
            return {
                "status": "error",
                "error": "WorkItem cancellation was rejected by its current lease",
            }
        self._active_work_item_claim(job_id, pop=True)
        return {
            "status": "success",
            "job_id": job_id,
            "prev_status": _task_status_from_work_item(prior),
        }

    def clear_tasks(self, status: str = "completed") -> dict:
        """Reject deletion of immutable WorkItem audit records."""
        status = (status or "completed").strip().lower()
        valid = {
            "pending",
            "running",
            "scheduled",
            "blocked",
            "completed",
            "failed",
            "cancelled",
            "dead_letter",
            "all",
        }
        if status not in valid:
            return {
                "status": "error",
                "error": f"status must be one of {sorted(valid)}",
            }

        return {
            "status": "error",
            "error": "WorkItem audit records cannot be cleared",
            "cleared": 0,
            "filter": status,
            "remaining": len(self._ingest_work_item_index()),
        }

    def prioritize_task(self, job_id: str, priority: int = 1) -> dict:
        """Re-prioritize a task by setting its claim bucket (CONCEPT:AU-KG.ingest.hardened-priority-scheduled-task).

        The worker claim iterates integer buckets 0..3 in ascending order, so
        a lower bucket runs first. Named priority aliases are not accepted.
        """
        try:
            bucket = _coerce_prio_bucket(priority)
        except (TypeError, ValueError):
            return {
                "status": "error",
                "error": "priority must be an integer bucket from 0 through 3",
            }
        from agent_utilities.orchestration import work_item as _wi

        item_id = _wi.ingest_task_work_item_id(job_id)
        if _wi.get_work_item(self._work_item_engine, item_id) is None:
            return {"status": "error", "error": f"job {job_id} not found"}
        if not _wi.set_work_item_priority(self._work_item_engine, item_id, bucket):
            return {
                "status": "error",
                "error": "WorkItem priority update was rejected",
            }
        return {
            "status": "success",
            "job_id": job_id,
            "prio_bucket": bucket,
            "task_status": _task_status_from_work_item(
                _wi.get_work_item(self._work_item_engine, item_id)
            ),
        }
