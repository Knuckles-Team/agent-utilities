import asyncio
import base64
import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, cast

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)


def daemon_role() -> str:
    """Resolve this process's KG background-daemon role (CONCEPT:KG-2.8 / OS-5.0).

    The KG runs ONE consolidated background daemon (queue drain + graph writer +
    task workers + maintenance scheduler + file-watch poll). This selects who
    runs it:

    * ``host``   — run the full UnifiedDaemon (the API gateway sets this).
    * ``client`` — run NOTHING; submit work to the durable queue that the host
      daemon drains (MCP server / CLI / one-shot scripts set this).
    * ``auto``   — default: run the consolidated daemon in-process (single-
      process / dev usage, backward compatible).

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


def _pdf_file_extractor() -> dict[str, Any]:
    """Map ``.pdf`` to PyMuPDF instead of SimpleDirectoryReader's default (pypdf).

    pypdf's pure-Python ``extract_text`` is pathologically slow on some PDFs — a
    single 1 MB / 23-page file was observed pinning a core for **5+ minutes** in
    ``read_from_stream``. Worse, because it never releases the GIL, that one file
    starves every other KGTaskWorker on the host, so the whole durable queue stalls
    behind it. PyMuPDF (``fitz``) extracts the same file in ~0.2s and is a C
    extension that releases the GIL during parsing, so it neither stalls nor
    serializes other workers. Returns an empty mapping (default reader) if the
    PyMuPDF reader isn't installed. (CONCEPT:KG-2.8)
    """
    try:
        from llama_index.readers.file import PyMuPDFReader

        return {".pdf": PyMuPDFReader()}
    except Exception:  # pragma: no cover - optional dependency / import guard
        return {}


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


def _coerce_prio_bucket(value: Any, default: int = 2) -> int:
    """Map a priority spec to a discrete claim bucket 0..3 (CONCEPT:KG-2.113).

    Accepts an int bucket, a numeric string, or the legacy ``priority`` string
    (``critical``/``high``/``normal``/``background``/``low``). Out-of-range ints
    are clamped into ``[0, 3]`` so a caller can never wedge the claim loop.
    """
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, int):
        return max(0, min(value, 3))
    text = str(value).strip().lower()
    if text.isdigit():
        return max(0, min(int(text), 3))
    return {
        "critical": 0,
        "high": 1,
        "normal": 2,
        "background": 3,
        "low": 3,
    }.get(text, default)


import sqlite3

from .queue_backend import QueueBackend


def _kg_dev_mode() -> bool:
    """True when ``KG_DEV_MODE`` disables all KG background daemons.

    One switch replaces the per-daemon ``KG_*_DAEMON`` env toggles (which all
    defaulted on): production runs every daemon; dev can silence the lot. Read
    via ``AgentConfig`` so there's a single typed source of truth, not scattered
    ``os.environ`` reads. (CONCEPT:KG-2.8 / config discipline)
    """
    try:
        from agent_utilities.core.config import config

        return bool(getattr(config, "kg_dev_mode", False))
    except Exception:  # noqa: BLE001 — config unavailable → daemons on (prod default)
        return False


def compute_ingest_worker_count(configured: int | None = None) -> int:
    """Autosize the ingest worker pool for THIS host (CPU + memory bounded).

    The single sizing policy shared by the in-process task workers and the
    decoupled ``kg-ingest`` consumer pool (CONCEPT:KG-2.57): ~36% of the cores,
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
# fetch size. (CONCEPT:KG-2.8 / config discipline)
_EMBED_BACKFILL_BUDGET = 256
_EMBED_BACKFILL_FETCH = 512

# PerformanceAnomaly consumer cadence (CONCEPT:AHE-3.19): a bounded, LLM-free
# scan, so a fixed moderate interval suffices — no env knob needed.
_ANOMALY_CONSUMER_INTERVAL = 900.0

# Background-daemon cadences and task-reaper limits (config discipline): each has
# one correct default and no per-host correctness requirement, so they are named
# module constants rather than env knobs (replacing KG_*_INTERVAL / KG_TASK_*).
# Seconds.
_EVOLUTION_INTERVAL = 3600.0
_RECONCILE_INTERVAL = 900.0
_ENRICH_INTERVAL = 20.0
_FILE_WATCH_INTERVAL = 30.0
# Fast cadence for the reactive autoscale poll (CONCEPT:KG-2.253): it only does a
# cheap non-blocking ``:Task`` change-feed poll and short-circuits when nothing
# changed, so it can run far more often than the slow ``_tick_fleet_autoscaler``
# safety-net interval — turning "scale on the change" from minutes into seconds.
_AUTOSCALE_REACTIVE_INTERVAL = 5.0
_HYGIENE_INTERVAL = 86400.0
_TASK_REAPER_INTERVAL = 120.0
# Warm-fork parent + dev-workspace idle reap (CONCEPT:OS-5.58). Background; never preempts work.
_WARM_PARENT_REAP_INTERVAL = 300.0
_EMBED_BACKFILL_IDLE_INTERVAL = 30.0
_EMBED_BACKFILL_BUSY_SLEEP = 1.0

# Embedder circuit-breaker (CONCEPT:KG-2.8): when the embedding endpoint is down
# (e.g. the GPU host power-cycles → vLLM 502s), the backfill tick must NOT keep
# calling it every 30s across N tables (each with client-side retries) — that
# retry-storm pegs the daemon and makes the whole KG surface time out. After this
# many consecutive embed failures the circuit OPENS: ticks become cheap no-ops
# (zero embed calls) for the cooldown, then one probe batch tests recovery.
_EMBED_CB_THRESHOLD = 3
_EMBED_CB_COOLDOWN = 300.0
_TASK_ORPHAN_GRACE_SEC = 90.0
_TASK_MAX_RUNTIME_SEC = 7200.0
_TASK_MAX_REQUEUE = 3
_USAGE_SYNC_INTERVAL = 900.0
_USAGE_PRICING_REFRESH_INTERVAL = 86400.0

# CONCEPT:KG-2.113 — Hardened priority and scheduled task queue with retry and dead-letter.
# Priority is a discrete
# integer *bucket* (0=critical .. 3=background) rather than a numeric field,
# because the L1 graph interpreter strips ORDER BY and supports only equality —
# so the worker claim iterates buckets ascending with one equality query each
# (the generalization of the old binary high/normal two-query tier). Legacy
# nodes carrying only the ``priority`` string map high→1, normal→2.
_PRIORITY_BUCKETS: tuple[int, ...] = (0, 1, 2, 3)
_PRIO_CRITICAL, _PRIO_HIGH, _PRIO_NORMAL, _PRIO_BACKGROUND = 0, 1, 2, 3
_DEFAULT_PRIO_BUCKET = _PRIO_NORMAL

# Max candidate rows a single claim attempt will CAS before giving up (returning
# idle). Each miss means a peer host won that row via the engine CAS, so we re-
# select and try the next pending candidate. Bounds the contention sweep so a
# burst of competing workers can't spin. (CONCEPT:KG-2.141)
_CLAIM_MAX_RETRIES = 8
# App-level retry: a task that *raises* (vs. a host crash, handled by the reaper)
# is retried with exponential backoff by re-scheduling it for a future minute,
# then dead-lettered past the cap. Distinct from the reaper's crash-requeue
# counter (``reaper_resets`` / ``_TASK_MAX_REQUEUE``) — two failure modes.
_TASK_MAX_ATTEMPTS = 3
_TASK_RETRY_BASE_SEC = 30.0
# Delayed/blocked → pending promotion sweep cadence (the only range comparison
# in the queue, done in Python over the small scheduled/blocked set).
_PROMOTION_SWEEP_INTERVAL = 60.0
# Terminal task statuses (no further work). ``dead_letter`` is a poison task
# that exhausted its app-level retries; kept distinct from ``failed`` (the
# reaper's crash-requeue terminal) so the two failure modes triage separately.
_TERMINAL_TASK_STATUS = frozenset({"completed", "failed", "cancelled", "dead_letter"})

# Enrichment pass sizing (config discipline): per-tick LLM-card batch budget. The
# per-batch summarization concurrency is CPU/mem auto-sized via
# compute_ingest_worker_count(); these batch caps are bounded constants, not env
# knobs (replacing KG_ENRICH_BATCH / KG_ENRICH_MAX_BATCHES).
#
# CONCEPT:KG-2.153 — the per-TICK chunk is sized to drain the ``cards_pending``
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
        queue self-heals if its db file is deleted/recreated/corrupted after
        init — otherwise every method would fail forever with
        ``no such table: staging`` once the file is gone. (CONCEPT:KG-2.7)
        """
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
        with self.lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        "INSERT INTO queue (data) VALUES (?)", (json.dumps(item),)
                    )
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
        props = {"rel_type": rel_type, **properties, "ephemeral": ephemeral}
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


class TaskManagerMixin(GraphEngineProtocol):
    """Mixin for native persistent Task Queues in the Intelligence Graph.

    CONCEPT:KG-2.0 - Persistent Task Tracking
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._workers_running = False
        self._worker_lock = threading.Lock()
        # Task claiming is now arbitrated by the engine compare-and-set (it holds
        # the graph write lock for the flip), so the former in-process
        # ``_claim_lock`` and the Postgres advisory ``state_claim_guard`` are no
        # longer needed to serialize the claim. (CONCEPT:KG-2.141)

        # Pre-import LlamaIndex components in main thread to avoid parallel worker import race conditions
        try:
            from llama_index.core import SimpleDirectoryReader  # noqa: F401
            from llama_index.core.embeddings import BaseEmbedding  # noqa: F401
        except ImportError:
            pass

        # Initialize pluggable persistent task queue
        from agent_utilities.core.config import config
        from agent_utilities.core.paths import data_dir

        queue_db_path = data_dir() / "kg_task_queue.db"

        # Queue selection is ONE explicit, fail-loud path (CONCEPT:KG-2.55):
        # TASK_QUEUE_BACKEND=sqlite|postgres|kafka, default auto (postgres when
        # STATE_DB_URI is set — CONCEPT:OS-5.16/KG-2.54 — else sqlite). An
        # explicitly selected kafka/postgres queue that is unreachable raises
        # TaskQueueUnavailable here instead of silently degrading.
        from .queue_backend import create_task_queue

        self._submission_queue: QueueBackend
        self._submission_queue, self._task_queue_backend_name = create_task_queue(
            config, str(queue_db_path)
        )

        import sys

        # ── Role-gated background daemon (CONCEPT:KG-2.8 / OS-5.0) ───────────
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
        # Singleton election (CONCEPT:KG-2.8 / OS-5.9): only the flock holder runs
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

        # Continuous queue drainers that actually move ingested data.
        # Kafka mode (CONCEPT:KG-2.57): the ``kg-ingest`` consumer group owns
        # the whole task lifecycle (Task node is created AT CLAIM TIME by the
        # consuming worker), so the submission drain — which would race the
        # group for the same messages just to mint a node — does not run.
        if self._task_queue_backend_name != "kafka":
            self._submission_thread = threading.Thread(
                target=self._submission_worker_loop,
                daemon=True,
                name="KG-Job-Submitter",
            )
            self._submission_thread.start()

        self._graph_writer_thread = threading.Thread(
            target=self._graph_writer_loop, daemon=True, name="KG-Graph-Writer"
        )
        self._graph_writer_thread.start()

        # KG background daemons are always on in production; the single
        # KG_DEV_MODE switch disables the whole set (it replaced the per-daemon
        # KG_*_DAEMON env toggles). (CONCEPT:KG-2.8 / config discipline)
        if _kg_dev_mode():
            return

        # Single consolidated maintenance scheduler (CONCEPT:KG-2.8): runs ALL
        # periodic KG jobs (analysis, compaction, evolution, enrichment, AND the
        # SDD/skills/scholarx file-watch scan — see ``_maintenance_jobs``) in ONE
        # throttled thread behind one shared foreground gate. No separate file
        # watcher thread.
        self._maintenance_thread = threading.Thread(
            target=self._maintenance_scheduler_loop,
            daemon=True,
            name="KG-Maintenance-Scheduler",
        )
        self._maintenance_thread.start()

        # Dedicated vector-embedding backfill drain (separate from the periodic
        # scheduler so it is never starved behind slow LLM ticks). (KG-2.8)
        self._embed_backfill_thread = threading.Thread(
            target=self._embedding_backfill_loop,
            daemon=True,
            name="KG-Embedding-Backfill",
        )
        self._embed_backfill_thread.start()

    # ── Control-plane backend routing (CONCEPT:KG-2.148) ─────────────────
    # The scheduler / task-claim / status / reaper / promotion-sweep / :Schedule
    # writes are CONTROL plane and must run on the isolated ``__control__`` engine
    # graph's write lock, NOT ``__commons__``'s — which sustained content
    # ingestion holds and otherwise starves the control plane through. The engine
    # builds ``self.control_backend`` (a backend bound to ``__control__``); these
    # helpers route the control-plane ops to it, degrading to ``self.backend``
    # when no isolated control backend exists (so behaviour is unchanged on
    # deployments where construction failed). Content/document/codebase ingestion
    # writes deliberately keep using ``self.backend`` (``__commons__``).

    @property
    def _control(self) -> Any:
        """The control-plane backend (``__control__``), or ``self.backend``.

        CONCEPT:KG-2.148 — single accessor so every control-plane call site
        routes through the isolated control graph when available and falls back
        cleanly to the shared content backend when it isn't.
        """
        return getattr(self, "control_backend", None) or self.backend

    def _control_cypher(
        self, cypher: str, params: dict | None = None
    ) -> list[dict[str, Any]]:
        """Run a CONTROL-PLANE Cypher read/write against ``__control__``.

        Mirrors ``query_cypher`` but targets the isolated control backend
        (CONCEPT:KG-2.148). Used for :Task / :Schedule / queue / claim ops so
        they never block on the content-ingestion write lock.
        """
        ctrl = self._control
        if ctrl is not None and hasattr(ctrl, "execute"):
            return ctrl.execute(cypher, params)
        return []

    def unified_daemon_status(self) -> dict[str, Any]:
        """Status of the single consolidated background daemon (CONCEPT:KG-2.8).

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
        except Exception:  # noqa: BLE001
            pass
        # Engine shard topology + per-shard reachability (CONCEPT:KG-2.58 /
        # CONCEPT:OS-5.28)
        # The flock host role above governs only the LOCAL engine; remote
        # shards are probed (short transport-level connect) and reported
        # here, never managed.
        try:
            from .shard_topology import shard_topology_status

            status["shards"] = shard_topology_status()
        except Exception:  # noqa: BLE001 - status surface stays best-effort
            pass
        return status

    def start_sdd_watcher(self):
        """Deprecated: the SDD/plan/skills/scholarx file-watch is now a periodic
        job inside the consolidated maintenance scheduler (``_tick_file_watch``,
        registered in ``_maintenance_jobs``), not a dedicated thread.

        Kept as a no-op so existing callers (e.g. the MCP server) don't spawn a
        second watcher thread. The scan runs only in the daemon ``host``/``auto``
        process. (CONCEPT:KG-2.6 / KG-2.8)
        """
        logger.debug(
            "start_sdd_watcher() is a no-op; file-watch runs as the 'file_watch' "
            "maintenance job in the consolidated scheduler."
        )

    def _tick_kg_analysis(self) -> None:
        """One autonomous-analysis tick (CONCEPT:KG-2.4).

        Schedules a relevance sweep hourly, then selects the highest-degree
        stale ``Concept`` for background deep analysis. Run by the consolidated
        maintenance scheduler (no own thread / sleeps / throttle gate).
        """
        import time

        RELEVANCE_SWEEP_INTERVAL = 3600.0  # 60 minutes
        now = time.time()
        # CONCEPT:ECO-4.72 — do NOT fire the heavy relevance sweep immediately on every
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
                "RETURN c.file_path AS path LIMIT 500"
            )
            if not results:
                return None

            # Extract repository roots from paths
            repo_counts: dict[str, int] = {}
            for row in results:
                path = row.get("path", "")
                if not path:
                    continue
                # Heuristic: repo root is 6th component of /home/apps/workspace/agent-packages/<name>
                parts = path.split("/")
                if len(parts) >= 6:
                    repo_name = parts[5] if "agent-packages" in path else parts[4]
                    repo_counts[repo_name] = repo_counts.get(repo_name, 0) + 1

            if repo_counts:
                return max(repo_counts, key=repo_counts.get)  # type: ignore[arg-type]
        except Exception as e:
            logger.debug(f"Primary codebase detection failed: {e}")
        return None

    # ── Consolidated maintenance scheduler (CONCEPT:KG-2.8) ──────────────

    def _maintenance_jobs(self) -> list[tuple[str, float, Any]]:
        """Inline plumbing the maintenance thread runs DIRECTLY (CONCEPT:OS-5.44).

        Everything else recurring (analysis, the self-evolution loop, enrichment,
        evolution, the fleet ticks, usage/file/hygiene/tenant-gc sweeps, and the
        declarative ``deploy/schedules.yml`` entries) is now a durable
        ``:Schedule`` that the ``scheduler`` tick ENQUEUES onto the unified queue
        — those bodies run in the worker pool under the throttle/lease/reaper, not
        in this thread (see :meth:`_register_maintenance_schedules`). Only the
        queue's OWN plumbing stays inline, because it must run even when the
        queue/workers are saturated (it is what feeds and heals them):

          * ``scheduler``       — evaluate :Schedule nodes and enqueue due jobs
          * ``task_reaper``     — requeue tasks orphaned by a dead worker/host
          * ``promotion_sweep`` — promote due/unblocked scheduled & blocked tasks
        """
        return [
            ("scheduler", 60.0, self._tick_scheduler),
            ("task_reaper", _TASK_REAPER_INTERVAL, self._tick_task_reaper),
            ("promotion_sweep", _PROMOTION_SWEEP_INTERVAL, self._tick_promotion_sweep),
        ]

    def _register_maintenance_schedules(self) -> None:
        """Register the former fixed-interval maintenance ticks as durable
        ``:Schedule`` nodes so the unified scheduler enqueues them (CONCEPT:OS-5.44).

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
            # CONCEPT:KG-2.153 — ``task_type`` lets a high-volume maint job run in
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
        # Self-evolution Loop engine cycle (CONCEPT:KG-2.78), OPT-IN via KG_LOOP=1;
        # runs _tick_loop as a task at research priority.
        _maint(
            "loop_cycle", "loop", _cfg.kg_loop_interval, enabled=_cfg.kg_loop, prio=2
        )
        # ScholarX RSS research-feed screen (CONCEPT:KG-2.114): grade incoming RSS
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
            "dspy_optimization",
            "optimize_components",
            _cfg.kg_dspy_optimization_interval,
            enabled=_cfg.kg_dspy_optimization,
        )
        _maint(
            "anomaly_consumer",
            "anomaly_consumer",
            _ANOMALY_CONSUMER_INTERVAL,
            enabled=_cfg.kg_anomaly_consumer,
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
        # CONCEPT:KG-2.253 — reactive push half of OS-5.29: a fast, cheap poll of the
        # engine's ``:Task`` change-feed that evaluates only on a queue-depth change.
        # Same opt-in gate as the autoscaler; the slow tick above is the safety net.
        _maint(
            "fleet_autoscale_reactive",
            "fleet_autoscale_reactive",
            _AUTOSCALE_REACTIVE_INTERVAL,
            enabled=_cfg.fleet_autoscaler,
        )
        _maint("compaction", "compaction", 1800.0)
        _maint("evolution", "evolution", _EVOLUTION_INTERVAL)
        _maint(
            "reconcile_durable",
            "reconcile_durable",
            _RECONCILE_INTERVAL,
            enabled=callable(
                getattr(getattr(self, "backend", None), "reconcile_to_durable", None)
            ),
        )
        # CONCEPT:KG-2.153 — OWL capability-card backfill runs in its OWN
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
        # Goals-as-contracts SLA watch (CONCEPT:ORCH-1.78): escalate breached goals.
        # Default-on; no-ops when no goals carry an sla_seconds.
        _maint("goal_sla", "goal_sla", 300.0)
        # Warm-fork parent + dev-workspace idle reap (CONCEPT:OS-5.58). Default-on;
        # no-ops when no warm parents / idle workspaces exist.
        _maint("warm_parent_reap", "warm_parent_reap", _WARM_PARENT_REAP_INTERVAL)

        for spec in specs:
            try:
                register_schedule(self, spec)
            except Exception as e:  # noqa: BLE001 — one schedule never blocks others
                logger.debug(
                    "register maintenance schedule %s failed: %s", spec.name, e
                )

    def _tick_warm_parent_reap(self) -> None:
        """Reap idle warm-fork parents + idle dev-workspace containers (CONCEPT:OS-5.58).

        Background maintenance: closes warm parents (forkserver processes, warmed containers,
        microVMs) idle past the registry TTL, and adopts the previously-orphaned
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
        except Exception as e:  # noqa: BLE001
            logger.debug("dev-workspace reap skipped: %s", e)
        try:
            # CONCEPT:ORCH-1.94 — stateless backstop: sweep warm-fork sandbox containers the
            # in-memory registry can no longer see (orphaned by a daemon restart) by label + age.
            from agent_utilities.rlm.sandboxes.container_fork_backend import (
                reap_orphaned_sandboxes,
            )

            orphans = reap_orphaned_sandboxes()
            if orphans:
                logger.info(
                    "Reaped %d orphaned warm-fork sandbox container(s).", len(orphans)
                )
        except Exception as e:  # noqa: BLE001 — sweep is best-effort
            logger.debug("orphaned-sandbox sweep skipped: %s", e)

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
        except Exception as e:  # noqa: BLE001
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
        except Exception as e:  # noqa: BLE001
            logger.debug("usage_pricing_refresh skipped: %s", e)

    def _tick_hygiene(self) -> None:
        """One memory-hygiene pass (CONCEPT:KG-2.17).

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
        """One desired-state fleet reconcile pass (CONCEPT:OS-5.25).

        Diffs the fleet registry (+ optional desired-state override) against
        the pluggable FleetObserver and converges each divergence through the
        ActionPolicy decision point (CONCEPT:OS-5.24) and the FleetActuator
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
        """One reactive autoscale pass (CONCEPT:OS-5.29).

        For each registry service with a ``scaling:`` block: read its load
        signal through the pluggable ScalingSignalProvider, target-track a
        desired replica count within the declared min/max bounds (step-capped,
        cooldown/flap-guarded against the durable action ledger), diff against
        the FleetObserver and propose ``scale_service`` through the
        ActionPolicy decision point (CONCEPT:OS-5.24) + FleetActuator seam;
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
        """Lazily-built reactive control-plane ``:Task`` change-feed (CONCEPT:KG-2.253).

        One subscription per daemon process, cached on the engine, so the reactive
        autoscale tick fires on the engine's pushed ``:Task`` change-event (the
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
        """Fire an autoscale evaluation ON a control-plane ``:Task`` change (KG-2.253).

        The push half of OS-5.29 autoscaling: poll the engine's ``:Task`` change-feed
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
                    "[KG-2.253] reactive autoscale on :Task change: evaluated=%s "
                    "actions=%s scaled=%s",
                    report.get("evaluated"),
                    report.get("actions"),
                    report.get("scaled"),
                )
        except Exception as e:  # noqa: BLE001 — one job's failure never stops others
            logger.debug("fleet_autoscale_reactive tick error: %s", e)

    def _get_host_token(self) -> str:
        """Stable per-process identity for task-claim ownership (zombie reaper).

        Unique across process restarts (hostname + pid + boot second), so a task
        claimed by a now-dead host is distinguishable from one claimed by the live
        host. Cached on the engine singleton, so every worker thread + the reaper in
        this process share one value. (CONCEPT:KG-2.8)
        """
        tok = getattr(self, "_host_token_cache", None)
        if tok is None:
            import os
            import socket

            tok = f"{socket.gethostname()}:{os.getpid()}:{int(time.time())}"
            self._host_token_cache = tok
        return tok

    def _tick_task_reaper(self) -> None:
        """Requeue zombie/stuck 'running' tasks (CONCEPT:KG-2.8 ingestion durability).

        A task is marked 'running' when a worker claims it; if that worker/host
        process dies mid-task (crash / SIGKILL / redeploy) the Task is stranded in
        'running' forever and never re-claimed, silently wedging that ingestion. The
        singleton host lock guarantees exactly one host runs workers, so any
        'running' task whose ``claimed_by`` is NOT this host's live token (after a
        short grace, to tolerate election hand-off) is an orphan from a dead host →
        reset to 'pending' for re-claim. A same-host task exceeding an absolute
        runtime cap is also requeued (backstop for a wedged-but-alive worker). A task
        requeued more than the cap is marked 'failed' (poison-pill guard) so a task
        that reliably kills its worker cannot loop forever. Host-only; driven by the
        consolidated maintenance scheduler.
        """

        from .host_lock import effective_daemon_role

        if effective_daemon_role() != "host":
            return
        try:
            grace = _TASK_ORPHAN_GRACE_SEC
            max_runtime = _TASK_MAX_RUNTIME_SEC
            max_resets = _TASK_MAX_REQUEUE
            now = time.time()
            token = self._get_host_token()

            from agent_utilities.core.state_store import postgres_state_enabled

            _multi_host = postgres_state_enabled()

            # CONCEPT:KG-2.148 — the reaper scans + resets :Task on the CONTROL
            # plane (__control__), never the content graph.
            rows = self._control_cypher(
                "MATCH (t:Task {status: 'running'}) RETURN t.id as id, t.metadata as meta"
            )
            requeued = failed = 0
            for row in rows or []:
                tid = row.get("id")
                if not tid:
                    continue
                meta = _decode_metadata(row.get("meta")) or {}
                claimed_by = meta.get("claimed_by")
                # Claim age: prefer claim_unix, else parse started_at; else unknown.
                claim_unix = meta.get("claim_unix")
                if claim_unix is None and meta.get("started_at"):
                    try:
                        claim_unix = datetime.fromisoformat(
                            meta["started_at"]
                        ).timestamp()
                    except (ValueError, TypeError):
                        claim_unix = None
                try:
                    age = now - float(claim_unix) if claim_unix is not None else None
                except (ValueError, TypeError):
                    age = None

                # Orphan: a 'running' task NOT owned by the live host. The singleton
                # host lock guarantees exactly one host runs workers, so any running
                # task whose owner isn't this host's token is being processed by
                # nobody — its worker died. This covers both a *foreign* token (a
                # previous host) and an *unstamped* task (claimed before the reaper
                # existed — the first-deploy case), so a fresh host cleans pre-existing
                # zombies instead of waiting out the absolute cap. A foreign explicit
                # token is reaped even if its age is unknown (provably a dead host); an
                # unstamped task needs a known age past the hand-off grace to avoid
                # racing any malformed-but-fresh claim.
                #
                # Multi-host (CONCEPT:OS-5.16/KG-2.54): with ``state_db_uri`` set,
                # N hosts legitimately run workers, so a foreign token is NOT
                # proof of a dead worker — another live host may be processing
                # it. The reaper (leader-only) then degrades to conservative
                # age-based reaping: unstamped past the grace, or ANY claim past
                # the absolute runtime cap.
                if _multi_host:
                    orphan = (
                        claimed_by is None and age is not None and age >= grace
                    ) or (age is not None and age >= max_runtime)
                else:
                    not_live = claimed_by != token  # token is never None
                    orphan = not_live and (
                        (claimed_by is not None and age is None)
                        or (age is not None and age >= grace)
                    )
                # Backstop: this host's own task wedged beyond the absolute cap.
                backstop = (
                    claimed_by == token and age is not None and age >= max_runtime
                )
                if not (orphan or backstop):
                    continue

                resets = int(meta.get("reaper_resets", 0)) + 1
                reason = "orphan(dead-host)" if orphan else "stuck(runtime-cap)"
                if resets > max_resets:
                    meta["error"] = (
                        f"reaper: exceeded {max_resets} requeues ({reason}); "
                        "failing to break the loop"
                    )
                    meta["reaper_resets"] = resets
                    self._control_cypher(
                        "MATCH (t:Task {id: $id}) SET t.status = 'failed', t.metadata = $meta",
                        {"id": tid, "meta": _encode_metadata(meta)},
                    )
                    failed += 1
                    logger.warning(
                        "TaskReaper: FAILED %s after %d requeues (%s)",
                        tid,
                        resets - 1,
                        reason,
                    )
                    continue

                meta["reaper_resets"] = resets
                meta["reaper_last_reason"] = reason
                meta["reaper_last_at"] = datetime.now(UTC).isoformat()
                meta.pop("claimed_by", None)
                meta.pop("claim_unix", None)
                # Guard on status='running' so we never clobber a task a worker just
                # legitimately completed between the scan and this write.
                self._control_cypher(
                    "MATCH (t:Task {id: $id, status: 'running'}) SET t.status = 'pending', t.metadata = $meta",
                    {"id": tid, "meta": _encode_metadata(meta)},
                )
                # Kafka mode (CONCEPT:KG-2.57): nothing polls 'pending' :Task
                # nodes — the kg-ingest consumer group drives processing from
                # the topic — so a reaped orphan is RE-PUBLISHED for re-claim.
                # The claim is idempotent (status-checked), so a duplicate
                # delivery of the original message is harmless.
                if getattr(self, "_task_queue_backend_name", None) == "kafka":
                    try:
                        self._submission_queue.put(
                            {
                                "job_id": tid,
                                "props": {
                                    "status": "pending",
                                    "metadata": _encode_metadata(meta),
                                },
                            }
                        )
                    except Exception as e:  # noqa: BLE001 — next reaper tick retries
                        logger.warning(
                            "TaskReaper: kafka re-publish of %s failed: %s", tid, e
                        )
                requeued += 1
                logger.warning(
                    "TaskReaper: requeued %s (%s, age=%ss, claimed_by=%s)",
                    tid,
                    reason,
                    round(age) if age is not None else "?",
                    claimed_by,
                )
            if requeued or failed:
                logger.info(
                    "TaskReaper: requeued=%d failed=%d (running scanned=%d)",
                    requeued,
                    failed,
                    len(rows or []),
                )
        except Exception as e:  # noqa: BLE001 — one job's failure never stops others
            logger.debug("task_reaper tick error: %s", e)

    def _deps_state(self, deps: list[str]) -> str:
        """Resolve a dependency set to ``ready`` / ``waiting`` / ``broken``.

        ``ready`` = every dep ``completed``; ``broken`` = a dep reached a
        terminal non-completed state (failed/dead_letter/cancelled) so the
        dependent must never run on a broken precondition; ``waiting``
        otherwise. (CONCEPT:KG-2.113)
        """
        if not deps:
            return "ready"
        # Per-dep id-scoped equality queries: the L1 graph interpreter refuses an
        # unscoped ``WHERE t.id IN [...]`` (full-graph scan), so we look each dep
        # up by id (deps are few). (CONCEPT:KG-2.113)
        broken = {"failed", "dead_letter", "cancelled"}
        all_done = True
        for dep in deps:
            # CONCEPT:KG-2.148 — dependency :Task lookups are CONTROL plane.
            rows = self._control_cypher(
                "MATCH (t:Task {id: $id}) RETURN t.status as s", {"id": dep}
            )
            state = rows[0].get("s") if rows else None
            if state in broken:
                return "broken"
            if state != "completed":
                all_done = False
        return "ready" if all_done else "waiting"

    def _tick_promotion_sweep(self) -> None:
        """Promote due 'scheduled' and unblocked 'blocked' tasks to 'pending'.

        The unified queue defers work two ways without ORDER BY/range queries on
        the hot path: a delayed/retrying task waits as ``scheduled`` (carrying
        ``eta_unix``) and a dependent task waits as ``blocked`` (carrying
        ``depends_on``). This per-minute, leader-only sweep is the ONE place an
        eta/dependency comparison happens — in Python over the small
        scheduled/blocked set — so the worker claim stays pure equality.
        (CONCEPT:KG-2.113)
        """
        from .host_lock import effective_daemon_role

        if effective_daemon_role() != "host":
            return
        try:
            now = time.time()
            promoted = 0
            cancelled = 0
            # CONCEPT:KG-2.148 — the promotion sweep reads + flips :Task on the
            # CONTROL plane (__control__), isolated from content ingestion.
            # scheduled → pending once eta is due (or eta missing/garbled → now).
            rows = self._control_cypher(
                "MATCH (t:Task {status: 'scheduled'}) "
                "RETURN t.id as id, t.metadata as meta"
            )
            for row in rows or []:
                tid = row.get("id")
                if not tid:
                    continue
                meta = _decode_metadata(row.get("meta")) or {}
                eta = meta.get("eta_unix")
                try:
                    due = eta is None or float(eta) <= now
                except (TypeError, ValueError):
                    due = True
                if due:
                    self._control_cypher(
                        "MATCH (t:Task {id: $id, status: 'scheduled'}) "
                        "SET t.status = 'pending'",
                        {"id": tid},
                    )
                    promoted += 1
            # blocked → pending once all deps completed; broken deps → cancel.
            rows = self._control_cypher(
                "MATCH (t:Task {status: 'blocked'}) "
                "RETURN t.id as id, t.metadata as meta"
            )
            for row in rows or []:
                tid = row.get("id")
                if not tid:
                    continue
                meta = _decode_metadata(row.get("meta")) or {}
                deps = meta.get("depends_on") or []
                state = self._deps_state(deps)
                if state == "ready":
                    self._control_cypher(
                        "MATCH (t:Task {id: $id, status: 'blocked'}) "
                        "SET t.status = 'pending'",
                        {"id": tid},
                    )
                    promoted += 1
                elif state == "broken":
                    meta["error"] = "dependency failed; cancelling dependent task"
                    self._update_task_status(tid, "cancelled", meta)
                    cancelled += 1
            if promoted or cancelled:
                logger.info(
                    "PromotionSweep: promoted=%d cancelled=%d", promoted, cancelled
                )
        except Exception as e:  # noqa: BLE001 — one job's failure never stops others
            logger.debug("promotion_sweep tick error: %s", e)

    def _tick_file_watch(self) -> None:
        """One SDD/skills/scholarx/config file-watch scan (CONCEPT:KG-2.6 / OS-5.0).

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
            logger.debug("file_watch tick error: %s", e)

    def _record_queue_telemetry(self, queue_size: int) -> None:
        """Publish ingest queue depth (+ Kafka consumer lag) as Prometheus
        gauges on the OS-5.23 gateway metrics registry (CONCEPT:KG-2.57).

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
        yields the GPU/LLM to interactive runs. (CONCEPT:KG-2.7 / KG-2.8)

        Tick classification (CONCEPT:OS-5.17): every job in this scheduler is
        **leader-only** — whole-graph/singleton passes (analysis, golden loop,
        failure ingest, anomaly consumer, fuseki publish, compaction, evolution,
        durable reconcile, enrichment, SDD/file watch, hygiene, task reaper)
        where N hosts running them means duplicated LLM spend or double writes.
        With ``state_db_uri`` set, a Postgres advisory lock elects exactly one
        leader fleet-wide; followers idle here and still contribute **per-host**
        capacity (task workers + submission/graph-writer queue drains, whose
        claims are cross-host atomic — CONCEPT:KG-2.54). Under the SQLite
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

                # Leader-only gate (CONCEPT:OS-5.17): non-leader hosts skip all
                # singleton maintenance ticks and re-check for fail-over.
                if not leadership.is_leader():
                    time.sleep(10.0)
                    continue

                # Backpressure visibility (CONCEPT:KG-2.57): sample the durable
                # submission-queue depth every pass so depth (and, for Kafka,
                # kg-ingest consumer lag) lands on the OS-5.23 gateway Prometheus
                # registry — including under load, exactly when it matters most.
                q = getattr(self, "_submission_queue", None)
                if q is not None:
                    try:
                        self._record_queue_telemetry(q.get_queue_size())
                    except Exception:  # noqa: BLE001 — queue probe best-effort
                        pass

                # This loop now runs ONLY queue PLUMBING — the scheduler (which
                # also collapses stale ticks, CONCEPT:OS-5.53), the task reaper,
                # and the promotion sweep. Unlike the heavy job *bodies* they
                # enqueue (which run in the worker pool under the background
                # throttle), the plumbing feeds and heals the queue, so it MUST
                # run even when the queue/workers are saturated. It is therefore
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
        authority (no SQLite fallback). (CONCEPT:KG-2.8/KG-2.244)
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

        CONCEPT:KG-2.8. Cards are cached by ``ast_hash`` so unchanged code is
        never re-summarised; only non-empty summaries are written back (so a
        transient LLM outage doesn't poison nodes). Drains up to
        ``KG_ENRICH_MAX_BATCHES`` batches per tick, re-checking the foreground
        throttle between batches so it yields promptly to interactive runs.
        """
        import json

        backend = getattr(self, "backend", None)
        if not backend:
            return
        if not hasattr(self, "_enrich_card_cache"):
            self._enrich_card_cache: dict[str, Any] = {}

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
            except ImportError:
                pass

            rows = self.query_cypher(
                "MATCH (n:Code) WHERE n.summary = '' AND n.ast_hash IS NOT NULL "
                "RETURN n.id AS id, n.name AS name, n.kind AS kind, "
                "n.file_path AS file_path, n.patterns AS patterns, "
                "n.language AS language, n.ast_hash AS ast_hash LIMIT " + str(BATCH)
            )
            if not rows:
                return
            if llm_fn is None:
                # Card summaries are a structured extraction task — route to the
                # LITE chat model by default (markedly faster than the heavy KG
                # model, which is what saturated the engine on a full backfill).
                # ``KG_CARD_MODEL=heavy`` forces the heavy model. (CONCEPT:KG-2.8)
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
            # (CONCEPT:KG-2.7). The per-batch foreground check above stays as a
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
            for card in cards:
                if not card.summary:
                    continue
                try:
                    backend.execute(
                        "MATCH (n:Code {id: $id}) SET n.summary = $summary, "
                        "n.responsibilities = $resp",
                        {
                            "id": card.id,
                            "summary": card.summary,
                            "resp": json.dumps(card.responsibilities),
                        },
                    )
                    written += 1
                except Exception:
                    logger.debug("card writeback failed for %s", card.id, exc_info=True)
            logger.info("KG enrichment: backfilled %d/%d cards", written, len(rows))
            # If nothing landed (LLM likely down), stop this tick; retry later.
            if written == 0:
                return

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

        CONCEPT:KG-2.8 — keeps a down embedder from being retry-stormed.
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

    # CONCEPT:KG-2.144 — Per-channel embedding backfill: round-robin unembedded
    # nodes by source_system + fan out to embedding capacity, so a tiny url/doc
    # crawl's chunks aren't FIFO-starved behind millions of codebase chunks.
    # Per-table source-rotation cursors: which channel leads next tick.
    _EMBED_SOURCE_CURSORS: dict[str, int] = {}  # noqa: RUF012

    def _collect_unembedded_rows(
        self, conn_factory: Any, tbl: str, take: int
    ) -> list[tuple[Any, str]]:
        """Pull up to ``take`` NULL-embedding ``(id, text)`` rows from ``tbl``,
        round-robin across ingestion *channels* (``source_system``).

        CONCEPT:KG-2.144 — a single ``WHERE embedding IS NULL LIMIT n`` FIFO lets
        one huge channel (822K codebase ``Code`` chunks) starve a small url/doc
        crawl's chunks that share the table. Instead, for a table that carries a
        ``source_system`` column we find the distinct channels with unembedded
        rows and give each a slice of the budget every tick, rotating which
        channel leads (so the per-channel remainder is shared fairly over time).
        Tables without ``source_system`` (internal codebase writes) fall back to
        the plain bounded scan. L1-safe: only equality filters + LIMIT, no ORDER
        BY (the interpreter strips ORDER BY) — exactly like the lane claim.
        """
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

            def _fetch(where: str, params: tuple, limit: int) -> list[tuple[Any, str]]:
                cur.execute(
                    f'SELECT id, {expr} FROM "{tbl}" '  # nosec B608
                    f"WHERE embedding IS NULL{where} LIMIT %s",
                    (*params, limit),
                )
                got = [(r[0], (r[1] or "").strip()) for r in cur.fetchall()]
                return [(nid, txt) for nid, txt in got if txt]

            if not has_source:
                return _fetch("", (), take)

            # Distinct channels that still have unembedded rows (bounded scan, no
            # GROUP BY ORDER BY — equality-friendly DISTINCT only).
            cur.execute(
                f'SELECT DISTINCT source_system FROM "{tbl}" '  # nosec B608
                "WHERE embedding IS NULL LIMIT 64"
            )
            channels = sorted(
                str(r[0]) if r[0] is not None else "" for r in cur.fetchall()
            )
            if len(channels) <= 1:
                # One (or zero) channel — nothing to round-robin; plain scan.
                return _fetch("", (), take)

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
                    rows = _fetch(" AND source_system IS NULL", (), slot)
                else:
                    rows = _fetch(" AND source_system = %s", (ch,), slot)
                items.extend(rows)
            return items[:take]

    def _tick_embedding_backfill(self) -> int:
        """Backfill vector embeddings onto durable nodes that lack them.

        Vector features (semantic_search, concept→code RELATES_TO linking,
        designation, latent retrieval) need embeddings on the L3/pgvector node
        tables, but the structural codebase pass and concept extraction create
        nodes WITHOUT embeddings. This embeds unembedded rows incrementally in
        bounded batches with the configured model, behind the shared foreground
        gate. Idempotent (only ``embedding IS NULL`` rows). (CONCEPT:KG-2.8)
        """
        l3 = getattr(self.backend, "l3", self.backend)
        conn_factory = getattr(l3, "_conn", None)
        get_tables = getattr(l3, "_get_embedding_tables", None)
        if (
            not callable(conn_factory)
            or not callable(get_tables)
            or not getattr(l3, "pgvector_available", False)
        ):
            return 0  # not a pgvector-backed L3

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
            except Exception as e:  # noqa: BLE001
                logger.debug("embed backfill: query %s failed: %s", tbl, e)
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
                            f'UPDATE "{tbl}" SET embedding = %s::vector '  # nosec B608
                            "WHERE id = %s AND embedding IS NULL",
                            (str(vec), nid),
                        )
                    conn.commit()
                total += len(items)
                remaining -= len(items)
                self._embed_circuit_record(True, now)  # healthy → close breaker
            except Exception as e:  # noqa: BLE001
                logger.debug("embed backfill: store %s failed: %s", tbl, e)
                # An embed/store failure means the endpoint is likely down for
                # every table — record it and stop hammering the rest this tick.
                self._embed_circuit_record(False, now)
                break
        if total:
            logger.info("KG embedding backfill: embedded %d nodes", total)
        return total

    def _tick_loop(self) -> None:
        """One propose-only self-evolution cycle (CONCEPT:KG-2.7).

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
        """Lazily-built reactive ``WorldModelTransition`` change-feed (CONCEPT:KG-2.253).

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
        """One SAI-factory world-model specialization cycle (CONCEPT:AHE-3.29).

        REACTIVE (CONCEPT:KG-2.253): instead of re-querying the ENTIRE
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
            logger.error("sai_factory tick error: %s", e)

    def _tick_failure_ingest(self) -> None:
        """Ingest Langfuse failures → gap topics → regression-gated remediation.

        Pulls error/low-score/cost-latency telemetry from Langfuse, materializes
        ``ExecutionSummary`` / ``PerformanceAnomaly`` nodes and synthetic
        ``failure_gap`` ``Concept`` topics, then — when new gaps appear — runs one
        golden-loop cycle whose auto-merge is gated by a regression check bound to
        those failures. Opt-in via KG_FAILURE_EVOLUTION (CONCEPT:AHE-3.18).
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
        """Propose-only DSPy optimization sweep over the self-supervised targets.

        The scheduled twin of ``graph_orchestrate action=optimize_component`` (CONCEPT:
        AHE-3.46): gathers live graph data and runs the extraction / concept_match /
        routing optimizers, recording optimization trajectories. Nothing is auto-applied —
        promotion stays behind ``should_promote`` and a future auto-apply gate. Opt-in via
        KG_DSPY_OPTIMIZATION.
        """
        try:
            from ...harness.dspy_optimization import run_optimization_sweep

            report = run_optimization_sweep(self)
            logger.info(
                "DSPy optimization sweep: optimized=%s (propose-only)",
                report.get("optimized"),
            )
        except Exception as e:  # noqa: BLE001
            logger.error("dspy_optimization tick error: %s", e)

    def _tick_anomaly_consumer(self) -> None:
        """Drain unconsumed PerformanceAnomaly nodes into failure_gap topics.

        One bounded consumer pass (CONCEPT:AHE-3.19): clusters fresh anomalies
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

    def _tick_scheduler(self) -> None:
        """Evaluate every durable ``:Schedule`` and ENQUEUE the jobs that are due.

        The ONE scheduler tick (CONCEPT:OS-5.44): it reads the durable
        ``:Schedule`` registry (seeded from ``deploy/schedules.yml`` plus the
        former fixed-interval maintenance ticks registered programmatically) and
        for every due schedule enqueues a ``scheduled_job`` ``:Task`` onto the
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

    def _tick_fuseki_publish(self) -> None:
        """Push the bundled ontology modules to Apache Jena Fuseki.

        One bounded distribution pass (CONCEPT:KG-2.52): merges every shipped
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
        """Dedicated drain loop for vector-embedding backfill (CONCEPT:KG-2.8).

        Runs independently of the periodic maintenance scheduler so it is NOT
        blocked behind slow LLM ticks: it embeds a batch, and if work remains
        (a full batch landed) loops again almost immediately; when the graph is
        fully embedded it idles at a long interval. Yields to interactive runs
        via the shared foreground throttle.

        Leader-only (CONCEPT:OS-5.17): two hosts would select the same
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
                except ImportError:
                    pass
                embedded = self._tick_embedding_backfill()
                # Full batch ⇒ likely more to do ⇒ loop fast; else back off.
                time.sleep(busy if embedded >= batch else idle)
            except Exception as e:  # noqa: BLE001 — never let the loop die
                logger.error("EmbeddingBackfillLoop error: %s", e)
                time.sleep(idle)

    def _tick_tenant_gc(self) -> None:
        """Drop leaked per-job community-detection tenants (CONCEPT:KG-2.8).

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

    def _tick_reconcile_durable(self) -> None:
        """Autoheal the L1→L2 durable mirror (CONCEPT:KG-2.8).

        Backfills any nodes/edges present in the L1 compute graph but missing from
        the durable Postgres tier, so the two stores converge after an L1-only run,
        a restart, or a newly-introduced node type. Best-effort + idempotent (writes
        are upserts; auto-DDL creates any missing type table). Logs a drift summary.
        """
        backend = getattr(self, "backend", None)
        fn = getattr(backend, "reconcile_to_durable", None)
        if not callable(fn):
            return
        try:
            summary = fn() or {}
            # Exact post-condition drift — what truly remained unmirrored after the
            # pass (not the always-positive count of best-effort writes).
            missing = summary.get("nodes_missing", 0) + summary.get("edges_missing", 0)
            errs = summary.get("errors", 0)
            if missing or errs:
                logger.warning(
                    "durable reconcile: drift remains after sync — "
                    "%d nodes / %d edges missing, %d write errors (%s)",
                    summary.get("nodes_missing", 0),
                    summary.get("edges_missing", 0),
                    errs,
                    summary,
                )
            else:
                logger.debug(
                    "durable reconcile: in sync (%d nodes, %d edges mirrored)",
                    summary.get("nodes", 0),
                    summary.get("edges", 0),
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("durable reconcile tick failed: %s", e)

    def _tick_compaction(self) -> None:
        """One LCM compaction tick (CONCEPT:KG-2.1).

        Finds ``Thread`` nodes with more than ``COMPACTION_THRESHOLD``
        uncompacted messages and delegates to ``ElasticContextManager``. Run by
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
        from agent_utilities.knowledge_graph.memory import ElasticContextManager

        ecm = ElasticContextManager(max_tokens=32000)
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
        """One research-evolution cycle tick (CONCEPT:KG-2.5).

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
        # job (``failure_ingest`` → _tick_failure_ingest, CONCEPT:AHE-3.18), opt-in
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
                        if not label:
                            label = "Code"

                        node_type_map[nid] = label

                        # Filter valid properties
                        valid_keys = schema_cache.get(label)
                        props = {k: v for k, v in node.items() if v is not None}
                        # Preserve original semantic type for Code nodes (file/symbol/module)
                        if label == "Code" and raw_type and raw_type != "code":
                            props["type"] = raw_type

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
                        set_clause = ", ".join(
                            [f"n.{k} = $props_{k}" for k in props.keys()]
                        )
                        if set_clause:
                            set_clause = " SET " + set_clause
                        query = f"MERGE (n:{label} {{id: $id}}){set_clause}"

                        params = {"id": nid}
                        for k, v in props.items():
                            params[f"props_{k}"] = v

                        self.backend.execute(query, params)

                # Execute all edges sequentially
                for edge in edges:
                    if "source" in edge and "target" in edge and "type" in edge:
                        src = edge.pop("source")
                        tgt = edge.pop("target")
                        etype = str(edge.pop("type")).upper()
                        etype = "".join(c for c in etype if c.isalnum() or c == "_")

                        if not etype:
                            continue

                        u_label = node_type_map.get(src, "Code")
                        v_label = node_type_map.get(tgt, "Code")

                        query = f"MATCH (a:{u_label} {{id: $uid}}), (b:{v_label} {{id: $vid}}) MERGE (a)-[r:{etype}]->(b)"
                        self.backend.execute(query, {"uid": src, "vid": tgt})

                # Only acknowledge and remove from staging if successful
                self._submission_queue.ack_staged_graph(item_id)
            except Exception as e:
                logger.error(f"Error persisting staged graph (will retry): {e}")
                time.sleep(2.0)

    def _submission_worker_loop(self):
        """Background daemon thread to drain the SQLite queue and insert tasks into the graph."""
        import time

        while True:
            try:
                item = self._submission_queue.get()
                if item is None:
                    time.sleep(0.1)
                    continue

                item_id, task_data = item
                job_id = task_data["job_id"]
                props = task_data["props"]

                # CONCEPT:KG-2.148 — the :Task node is CONTROL plane: write it to
                # the isolated ``__control__`` graph (via the control backend) so
                # task creation never blocks behind sustained content ingestion on
                # ``__commons__``'s write lock.
                ctrl = self._control
                if (
                    ctrl is not None
                    and ctrl is not self.backend
                    and hasattr(ctrl, "add_node")
                ):
                    # Replicate EXACTLY what the protocol add_node() does (it wraps
                    # node_type + ephemeral into the property bag), just bound to
                    # the control backend instead of self.backend.
                    ctrl.add_node(job_id, node_type="Task", ephemeral=False, **props)
                else:
                    # Degrade to the shared content backend (unchanged behaviour).
                    self.add_node(job_id, "Task", properties=props)

                # Only acknowledge and remove from queue if successful
                self._submission_queue.ack(item_id)
                self._checkpoint_db()
            except Exception as e:
                logger.error(f"Error persisting queued task (will retry): {e}")
                time.sleep(1.0)

    def submit_task(
        self,
        target_path: str,
        is_codebase: bool,
        provenance: dict,
        task_type: str | None = None,
        skip_dedupe: bool = False,
        priority: str | int | None = None,
        scheduled_for: float | None = None,
        depends_on: list[str] | None = None,
        max_attempts: int = _TASK_MAX_ATTEMPTS,
        job_id: str | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> str:
        """Submit a background task to the unified durable queue (CONCEPT:KG-2.113).

        ``priority`` picks a claim bucket (0=critical .. 3=background, or the
        legacy ``high``/``normal`` strings). ``scheduled_for`` (a unix ts in the
        future) enqueues the task as ``scheduled`` until the per-minute promotion
        sweep makes it ``pending``. ``depends_on`` (other job ids) enqueues it as
        ``blocked`` until every dependency has ``completed``. ``job_id`` lets a
        caller supply a deterministic id (the unified Scheduler uses
        ``sched:<name>:<minute>`` so a double-fire is an idempotent upsert).
        """
        # Statuses that still represent un-finished work for dedupe + the
        # promotion sweep: pending/running plus the new delayed/blocked lanes.
        if not skip_dedupe:
            # CONCEPT:KG-2.148 — :Task dedupe read is CONTROL plane → __control__.
            existing = self._control_cypher(
                "MATCH (t:Task) WHERE t.status IN "
                "['pending', 'running', 'scheduled', 'blocked'] "
                "RETURN t.id as id, t.metadata as meta"
            )
            for row in existing:
                meta = _decode_metadata(row.get("meta"))
                if meta and meta.get("target") == target_path:
                    return row["id"]

        if not job_id:
            job_id = f"job-{uuid.uuid4().hex[:8]}"

        if not task_type:
            task_type = "codebase" if is_codebase else "document"

        now = time.time()
        task_data: dict[str, Any] = {
            "target": target_path,
            "type": task_type,
            "submitted_at": datetime.now(UTC).isoformat(),
            "attempts": 0,
            "max_attempts": int(max_attempts),
        }
        if extra_meta:
            task_data.update(extra_meta)

        prio_bucket = _coerce_prio_bucket(priority)
        # Resolve the initial lane: blocked (deps) > scheduled (eta) > pending.
        status = "pending"
        due_bucket: int | None = None
        if depends_on:
            task_data["depends_on"] = list(depends_on)
            status = "blocked"
        elif scheduled_for and float(scheduled_for) > now:
            task_data["eta_unix"] = float(scheduled_for)
            due_bucket = int(float(scheduled_for) // 60)
            status = "scheduled"

        from agent_utilities.knowledge_graph.core.task_lanes import lane_for_task_type

        encoded_meta = _encode_metadata(task_data)
        props: dict[str, Any] = {
            "status": status,
            "metadata": encoded_meta,
            "prio_bucket": prio_bucket,
            # CONCEPT:ORCH-1.75 — stamp the functional lane (top-level, queryable) so the
            # worker can claim fairly per-lane and we can surface per-lane congestion.
            "lane": lane_for_task_type(task_type),
            # CONCEPT:ORCH-1.76 — stamp the task TYPE top-level too, so claiming can rotate
            # fairly across types WITHIN a lane (a fast diff/document not stuck behind codebase).
            "tkind": task_type,
        }
        if due_bucket is not None:
            props["due_bucket"] = due_bucket
        if provenance:
            props.update(provenance)

        # Add the Task node to the persistence layer via the dedicated queue
        self._submission_queue.put({"job_id": job_id, "props": props})

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
                self._dropped_tables.update(new_tables)
                try:
                    self.backend.drop_vector_indices(tables=new_tables)
                except Exception as e:
                    logger.debug(f"Pre-ingestion index drop skipped: {e}")

        # Lazily start workers if they aren't already running
        self.start_task_workers()
        return job_id

    def _maybe_fanout_codebase(
        self, job_id: str, target: Path, meta: dict[str, Any]
    ) -> bool:
        """Split a too-large whole-repo codebase task into shard-routed sub-tasks.

        CONCEPT:KG-2.287 — the big-repo tail fix. A whole-repo ``codebase`` task for
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
        * splitting is skipped unless graph routing is enabled (distinct graphs are
          what buys the shard parallelism — with routing off every bucket would land
          on the same graph, so the split would add tasks for no gain);
        * small/medium repos (the healthy p50) fall straight through to the inline
          path, untouched.
        """
        # Already a sub-task, or an explicitly-scoped ingest → never fan out.
        if meta.get("route_repo") or meta.get("split_child") or meta.get("only_files"):
            return False
        try:
            from agent_utilities.knowledge_graph.core import ingest_routing

            if not ingest_routing.routing_enabled():
                return False
        except Exception:  # noqa: BLE001 — routing probe is best-effort
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
        ingest. (CONCEPT:KG-2.7 / KG-2.8)
        """
        try:
            rows = self._control_cypher(
                "MATCH (t:Task) WHERE t.status IN ['pending','running'] "
                "RETURN t.metadata as meta"
            )
        except Exception:  # noqa: BLE001
            return False
        n = 0
        for row in rows or []:
            meta = _decode_metadata(row.get("meta")) if isinstance(row, dict) else None
            if meta and meta.get("type") == "codebase":
                n += 1
                if n >= threshold:
                    return True
        return False

    def ingest_queue_depth(self) -> int:
        """Uniform ingest backlog depth across queue backends (CONCEPT:KG-2.57).

        ``queued-but-not-yet-claimed`` (the selected backend's queue size — for
        Kafka that is the ``kg-ingest`` consumer-group lag, for SQLite/Postgres
        the row count) **plus** in-graph ``pending``/``running`` ``:Task``
        nodes. This is the single number backpressure consumers (the batch
        orchestrator's deferral, the maintenance bulk-defer gate, the lag
        metrics) should read, regardless of which backend is selected.
        """
        depth = 0
        q = getattr(self, "_submission_queue", None)
        if q is not None:
            try:
                depth += int(q.get_queue_size())
            except Exception:  # noqa: BLE001 — depth probe is best-effort
                pass
        try:
            rows = self._control_cypher(
                "MATCH (t:Task) WHERE t.status IN ['pending','running'] "
                "RETURN count(t) AS c"
            )
            if rows:
                row = rows[0]
                depth += int(row.get("c", 0) or 0) if isinstance(row, dict) else 0
        except Exception:  # noqa: BLE001
            pass
        return depth

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

        # Pre-fetch active targets to deduplicate efficiently
        active_targets = set()
        for task in self._control_cypher(
            "MATCH (t:Task) WHERE t.status IN ['pending', 'running'] RETURN t.metadata as meta"
        ):
            meta = _decode_metadata(task.get("meta"))
            if meta and "target" in meta:
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
        # behaves as a client. (CONCEPT:KG-2.8 / OS-5.9)
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

            # Start workers
            self._workers_running = True

        if getattr(self, "_task_queue_backend_name", "sqlite") == "kafka":
            # CONCEPT:KG-2.57 — Kafka mode: the host's worker pool joins the
            # ``kg-ingest`` consumer group instead of polling :Task nodes, so
            # it shares partitions (and per-key ordering) with any decoupled
            # `kg-ingest-worker` processes added for scale-out.
            from ..ingest_worker import start_ingest_consumer_pool

            logger.info(
                "Starting %d kg-ingest consumer workers (kafka task queue)...",
                worker_count,
            )
            start_ingest_consumer_pool(self, worker_count=worker_count)
            return

        # ORCH-1.81: record the live pool size so the admission registry/policy
        # (hot spare, per-lane min coverage, codebase cap) size to this host's pool.
        self._ingest_worker_count = int(worker_count)
        logger.info(f"Starting {worker_count} TaskManager workers...")
        for i in range(worker_count):
            t = threading.Thread(
                target=self._task_worker_loop, name=f"KGTaskWorker-{i}", daemon=True
            )
            t.start()

    def _select_pending_task(
        self,
        admit: Callable[[str, str], bool] | None = None,
    ) -> dict[str, Any] | None:
        """Return one claimable pending Task row, highest priority bucket first.

        Bucketed equality queries (0=critical .. 3=background) replace
        ``ORDER BY priority`` because the L1 graph interpreter strips ORDER BY
        and supports only equality. A final untyped sweep picks up legacy nodes
        that predate ``prio_bucket`` and carry only the ``priority`` string. This
        single path works on both the SQLite default and pg-age (equality only),
        so priority + reordering hold everywhere. (CONCEPT:KG-2.113)

        CONCEPT:ORCH-1.75/1.76 — TWO-LEVEL fair rotation: rotate which functional lane gets
        first dibs each claim (so a backed-up lane can't head-of-line-block another — codebase
        ingestion was starved 75-pending/0-run), AND within the chosen lane rotate across its
        task TYPES (so a fast ``diff``/``document`` is not stuck behind a big ``codebase`` batch
        sharing the ingestion lane). Inside a (lane,type) the priority buckets still order work;
        lane-stamped-but-untyped and fully-legacy tasks fall through to the broader sweeps.

        CONCEPT:ORCH-1.81 — ``admit(lane, task_type) -> bool`` is the reserved-worker admission
        gate: rotation proposes the candidate, ``admit`` decides whether THIS free worker may
        claim that lane/type *now* (hot-spare reservation, per-lane min coverage, codebase cap).
        A denied (lane, type) is skipped so the rotation can offer a lane that needs coverage;
        if every typed/lane candidate is denied we fall through to the legacy sweeps (which a
        ``None`` admit, or an admit that has nothing left to deny, also reaches). When ``admit``
        is ``None`` the behaviour is exactly the pre-ORCH-1.81 rotation.
        """
        from agent_utilities.knowledge_graph.core.task_lanes import (
            LANE_NAMES,
            lane_task_types,
        )

        # Track whether the admission gate denied any candidate this pass — if it
        # denied at least one real (lane, type) and admitted none, the worker is
        # being held back as a spare, so the lane-less LEGACY sweeps must NOT grab a
        # task and spend that spare. With no admit (None), nothing is ever denied,
        # so legacy sweeps run exactly as before. (CONCEPT:ORCH-1.81)
        denied_any = False

        def _ok(lane: str, task_type: str) -> bool:
            nonlocal denied_any
            if admit is None:
                return True
            ok = admit(lane, task_type)
            if not ok:
                denied_any = True
            return ok

        if not hasattr(self, "_lane_cursor"):
            self._lane_cursor = 0
            self._type_cursors: dict[str, int] = {}
        n = len(LANE_NAMES)
        order = [LANE_NAMES[(self._lane_cursor + i) % n] for i in range(n)]
        self._lane_cursor = (self._lane_cursor + 1) % n
        for lane in order:
            # Per-TYPE fair rotation within the lane (ORCH-1.76).
            types = lane_task_types(lane)
            if types:
                tc = self._type_cursors.get(lane, 0)
                m = len(types)
                torder = [types[(tc + i) % m] for i in range(m)]
                self._type_cursors[lane] = (tc + 1) % m
                for tk in torder:
                    # ORCH-1.81: admission gate — a denied (lane, type) is skipped
                    # so this free worker stays a spare / steers to an uncovered
                    # lane instead of piling onto a capped/covered one.
                    if not _ok(lane, tk):
                        continue
                    for bucket in _PRIORITY_BUCKETS:
                        # CONCEPT:KG-2.148 — :Task claim selection is CONTROL plane.
                        rows = self._control_cypher(
                            "MATCH (t:Task {status: 'pending', tkind: $tk, prio_bucket: $b}) "
                            "RETURN t.id as id, t.metadata as meta LIMIT 1",
                            {"tk": tk, "b": bucket},
                        )
                        if rows:
                            return rows[0]
            # Lane-stamped but un-typed (pre-ORCH-1.76): claim by lane.
            if not _ok(lane, ""):
                continue
            for bucket in _PRIORITY_BUCKETS:
                rows = self._control_cypher(
                    "MATCH (t:Task {status: 'pending', lane: $lane, prio_bucket: $b}) "
                    "RETURN t.id as id, t.metadata as meta LIMIT 1",
                    {"lane": lane, "b": bucket},
                )
                if rows:
                    return rows[0]
        # ORCH-1.81: the admission gate denied work and admitted none → this worker
        # is being kept as a hot spare; don't let the lane-less legacy sweeps spend
        # it. (Min-coverage already relaxed the spare inside admit when needed.)
        if denied_any:
            return None
        # Lane-less legacy tasks (pre-ORCH-1.75 stamp): plain bucket sweep.
        for bucket in _PRIORITY_BUCKETS:
            rows = self._control_cypher(
                "MATCH (t:Task {status: 'pending', prio_bucket: $b}) "
                "RETURN t.id as id, t.metadata as meta LIMIT 1",
                {"b": bucket},
            )
            if rows:
                return rows[0]
        # Legacy fallback (pre-bucket nodes): honor the old high-then-any tiering.
        rows = self._control_cypher(
            "MATCH (t:Task {status: 'pending', priority: 'high'}) "
            "RETURN t.id as id, t.metadata as meta LIMIT 1"
        )
        if rows:
            return rows[0]
        rows = self._control_cypher(
            "MATCH (t:Task {status: 'pending'}) "
            "RETURN t.id as id, t.metadata as meta LIMIT 1"
        )
        return rows[0] if rows else None

    def lane_metrics(self) -> dict[str, Any]:
        """Per-lane congestion snapshot (CONCEPT:ORCH-1.75): pending depth + in-flight per
        functional lane, so congestion is VISIBLE before it starves work — the observability
        that was missing when codebase ingestion silently sat at 75-pending/0-running. Returns
        ``{lane: {pending, running, model_role}}`` + a ``lane_less`` bucket for un-stamped tasks.
        """
        from agent_utilities.knowledge_graph.core.task_lanes import (
            LANE_NAMES,
            lane_model_role,
        )

        def _count(where: str, params: dict[str, Any]) -> int:
            # CONCEPT:KG-2.148 — lane congestion counts read :Task on __control__.
            rows = self._control_cypher(
                f"MATCH (t:Task {{{where}}}) RETURN count(t) as c", params
            )
            return int((rows[0].get("c") if rows else 0) or 0) if rows else 0

        # ORCH-1.81: overlay the LIVE in-process worker registry so the snapshot
        # also shows how many workers each lane is *actually occupying right now*
        # (the queue's ``running`` status is set on claim; ``live_running`` is the
        # admission registry's view, which also drives the reservation/cap math).
        reg = getattr(self, "_worker_reg", None)
        live_running = reg.running_by_lane() if reg is not None else {}

        out: dict[str, Any] = {}
        for lane in LANE_NAMES:
            p = _count("status: 'pending', lane: $l", {"l": lane})
            r = _count("status: 'running', lane: $l", {"l": lane})
            out[lane] = {
                "pending": p,
                "running": r,
                "live_running": int(live_running.get(lane, 0)),
                "model_role": lane_model_role(lane),
            }
        total_pending = _count("status: 'pending'", {})
        total_running = _count("status: 'running'", {})
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
        return out

    # -- Reserved-worker fair scheduler (CONCEPT:ORCH-1.81) ------------------
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
            # CONCEPT:KG-2.281 — resolve the ENGINE's real durable shard-writer
            # width K once, from the engine that owns the redb backend (it may be a
            # remote box with a different cpu count than this scheduling host in
            # split-storage). Cached inside worker_scheduler so the codebase
            # admission floor reflects the engine's actual K, not this host's cpus.
            try:
                resolve_engine_shard_writers(self.backend)
            except Exception:  # noqa: BLE001 — best-effort; falls back to cpu/env
                pass
        return reg

    def _pending_by_lane(self) -> dict[str, int]:
        """Pending-task count per functional lane (for admission control).

        One equality count query per lane (LANE_NAMES is small). Lane-stamped
        tasks are counted directly; typed-but-not-yet-lane-stamped tasks are
        already covered because every enqueue stamps ``lane``. Best-effort: a
        query error yields 0 for that lane (degrade to "no pending" → permissive).
        """
        from agent_utilities.knowledge_graph.core.task_lanes import LANE_NAMES

        out: dict[str, int] = {}
        for lane in LANE_NAMES:
            try:
                rows = self._control_cypher(
                    "MATCH (t:Task {status: 'pending', lane: $l}) RETURN count(t) as c",
                    {"l": lane},
                )
                out[lane] = int((rows[0].get("c") if rows else 0) or 0)
            except Exception:  # noqa: BLE001 — best-effort; permissive on error
                out[lane] = 0
        return out

    def _make_admission(self) -> Callable[[str, str], bool] | None:
        """Build this-claim's admission predicate, or ``None`` to disable the gate.

        Returns ``None`` (pre-ORCH-1.81 behaviour) when the pool is too small for
        any reservation to make sense (1 worker can't keep a spare AND do work) or
        when the feature is disabled (``KG_SCHED_RESERVED=0`` and no cap). Otherwise
        binds an :class:`AdmissionPolicy` over the live registry + a freshly-sampled
        ``pending_by_lane`` snapshot.
        """
        reg = self._worker_registry()
        cfg = self._sched_config
        # With a single worker, a hot spare would mean never working; and with no
        # reservation and no explicit cap there's nothing to enforce.
        if cfg.worker_count <= 1:
            return None
        if cfg.reserved <= 0 and cfg.codebase_cap is None:
            return None
        from .worker_scheduler import AdmissionPolicy

        policy = AdmissionPolicy(cfg, reg)
        pending = self._pending_by_lane()

        def _admit(lane: str, task_type: str) -> bool:
            return policy.admit(lane, task_type, pending)

        return _admit

    def _claim_next_task(
        self, worker_id: str | None = None
    ) -> tuple[str, dict[str, Any]] | None:
        """Claim the next runnable Task and stamp ownership (CONCEPT:KG-2.141).

        Atomicity is now arbitrated by the engine's compare-and-set, which holds
        the graph write lock for the flip — so a row is claimed exactly once
        *across hosts*, backend-agnostically, without the former Postgres
        advisory lock (``state_claim_guard``) or in-process ``threading.Lock``.
        The bucket-ascending candidate selection is unchanged; for each
        candidate we CAS ``status: pending → running`` and stamp ownership. A
        CAS that returns ``False`` means another worker won that row — we skip to
        the next candidate. The ownership stamp (live host token + claim_unix)
        is what the zombie reaper uses to requeue a dead host's work. Returns
        ``(job_id, stamped_meta)`` or ``None`` when idle. (CONCEPT:KG-2.113)
        """
        # ORCH-1.81: build this claim's admission gate from the live worker→lane
        # registry. Composes WITH the rotation+CAS: rotation proposes a candidate,
        # ``admit`` decides whether THIS free worker may take that lane/type now
        # (keep a hot spare, cap codebase, guarantee per-lane min coverage). A
        # ``None`` gate (tiny pool / disabled) preserves the prior behaviour.
        try:
            admit = self._make_admission()
        except Exception:  # noqa: BLE001 — scheduling is best-effort; never block claims
            admit = None

        # Bound the retry sweep so a burst of contending workers can't spin
        # forever; each miss means a peer claimed that candidate, so the next
        # selection returns a different pending row (the claimed one is now
        # 'running' and no longer matches the pending filter).
        for _ in range(_CLAIM_MAX_RETRIES):
            row = self._select_pending_task(admit=admit)
            if not row:
                return None
            job_id = row["id"]
            meta = _decode_metadata(row.get("meta"))
            meta["started_at"] = datetime.now(UTC).isoformat()
            meta["claimed_by"] = self._get_host_token()
            meta["claim_unix"] = time.time()
            encoded_meta = _encode_metadata(meta)
            # CONCEPT:KG-2.148 — the claim CAS (status: pending→running) is the
            # hottest control-plane write; route it to the isolated __control__
            # graph so claims hold a write lock that content ingestion never
            # touches. Falls back to self.backend when no control backend exists.
            won = self._control.compare_and_set_node_fields(
                job_id,
                {"status": "pending"},
                {"status": "running", "metadata": encoded_meta},
            )
            if won:
                # ORCH-1.81: stamp the registry the instant the CAS wins so the
                # NEXT worker's admission sees this lane/type as covered/busy.
                if worker_id is not None:
                    from agent_utilities.knowledge_graph.core.task_lanes import (
                        lane_for_task_type,
                    )

                    tk = str(meta.get("type") or meta.get("tkind") or "document")
                    self._worker_registry().start(worker_id, lane_for_task_type(tk), tk)
                return job_id, meta
            # Lost the race for this row — try the next candidate.
        return None

    def _task_worker_loop(self):
        """Distributed polling loop that picks up pending tasks natively."""
        # ORCH-1.81: a stable per-thread id keys this worker in the admission
        # registry, so the policy knows what THIS worker is processing.
        worker_id = threading.current_thread().name
        while True:
            try:
                job_id = None
                target_path = None
                is_codebase = False
                task_type = "document"

                claimed = self._claim_next_task(worker_id=worker_id)
                if claimed:
                    job_id, meta = claimed
                    if meta:
                        if "target" in meta:
                            target_path = Path(meta["target"])
                        task_type = meta.get("type", "document")
                        is_codebase = task_type == "codebase"

                if not job_id:
                    # Idle backoff. During a bulk ingest, back off HARD: one worker
                    # holds the ingest while the other ~7 idle workers were each
                    # busy-polling two Task graph-scans every 2s, flooding the single
                    # client event loop + engine and starving the ingest worker
                    # (profiled: 24% of daemon CPU in poll query_cypher vs 10% in the
                    # actual ingest). A new task then waits at most one backoff to be
                    # claimed — fine while a multi-minute ingest drains. (CONCEPT:KG-2.7)
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
                except Exception:  # noqa: BLE001
                    pass  # nosec B110
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
        ``kg-ingest`` Kafka consumers (CONCEPT:KG-2.57) so the processing logic
        exists exactly once. Heavy task types (parse storms / background LLM /
        analysis) run through the shared background throttle so they yield to
        interactive (foreground) work and stay within the global concurrency
        cap — a bulk ingest can no longer consume the engine's whole in-flight
        budget and starve live queries (CONCEPT:KG-2.7 read/ingest plane
        isolation). Lightweight types (diff/conversation/…) run unthrottled.
        """
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
            # Async session-bundle upload (CONCEPT:KG-2.272): each session fans out
            # to many usage-store rows, so it drains under the background throttle.
            "session_upload",
            # Scheduled jobs (source syncs, loop cycles, the RSS feed screen) run
            # under the background throttle so a heavy cycle yields to foreground
            # work like any other background task. (CONCEPT:OS-5.44)
            "scheduled_job",
            # Full-paper download + ingest enqueued by the RSS feed screen.
            "research_paper_fetch",
            # Cohort barrier finalize → assimilation pass + feature matrix (KG-2.172).
            "cohort_synthesize",
        }
        # CONCEPT:KG-2.286 — bound EVERY claimed task by its lane's soft timeout so a
        # hung task (a connector with no per-call timeout, a wedged maint tick) frees
        # its worker FAST instead of pinning it until the reaper's 2h absolute cap.
        #
        # Why a watchdog THREAD and not ``asyncio.wait_for``: the work is run via
        # ``asyncio.run`` and a hang may be a *synchronous* blocking call (a connector
        # with no socket timeout) with no await point to cancel — and even when it is
        # cancellable, ``asyncio.run``'s loop-close JOINS the default executor (up to
        # ``THREAD_JOIN_TIMEOUT``), so the worker would still block on the hung thread.
        # Running the task body in a daemon thread and ``join(timeout)``-ing it lets
        # the worker RETURN at the bound regardless of where the hang is; the hung
        # thread is abandoned (daemon → never blocks shutdown) and the task is routed
        # through the KG-2.113 retry→backoff→dead_letter machinery by the worker loop.
        import threading as _threading

        from .task_lanes import task_soft_timeout

        timeout = task_soft_timeout(task_type)
        heavy = task_type in _HEAVY_TASK_TYPES
        outcome: dict[str, BaseException] = {}

        def _run_body() -> None:
            # CONCEPT:KG-2.293 — tag this task's whole execution with its resource
            # PriorityClass (derived from the SAME lane taxonomy as the worker
            # AdmissionPolicy), so every shared-LLM call it makes inherits the class:
            # an ingestion task's enrichment calls run as BACKGROUND_INGESTION and
            # yield the reserved LLM headroom to interactive/orchestration work, while
            # an on-pool ``queries`` task (conversation/kg_memory) runs INTERACTIVE.
            # Set inside the worker thread because contextvars don't cross threads.
            from agent_utilities.core.resource_priority import (
                priority_for_task_type,
                priority_scope,
            )

            try:
                with priority_scope(priority_for_task_type(task_type)):
                    asyncio.run(
                        self._run_background_task(
                            job_id, target_path, is_codebase, task_type
                        )
                    )
            except BaseException as exc:  # noqa: BLE001 — relayed to the worker loop
                outcome["exc"] = exc

        worker_thread = _threading.Thread(
            target=_run_body, name=f"kg-task-{job_id}", daemon=True
        )
        if heavy:
            # Hold the background concurrency slot only while we WAIT — released the
            # instant the worker is freed (success or timeout), so an abandoned hung
            # thread can't leak a slot forever (it merely over-subscribes by one
            # transiently, which the reaper/dead-letter then resolves).
            #
            # CONCEPT:KG-2.153 — ``enrichment_backfill`` is deliberately NOT in
            # ``_HEAVY_TASK_TYPES``, so it falls to the ``else`` branch below and runs
            # WITHOUT this outer permit: ``_tick_enrichment`` acquires the
            # background_slot PER BATCH (released between batches) so the dedicated
            # enrichment lane isn't capped by one tick-long outer permit while other
            # background/maint ticks still interleave — its per-batch throttle is the
            # real gate, and the soft-timeout above still bounds it.
            from agent_utilities.core.background_throttle import get_throttle

            with get_throttle().background_slot():
                worker_thread.start()
                worker_thread.join(timeout)
        else:
            worker_thread.start()
            worker_thread.join(timeout)

        if worker_thread.is_alive():
            # Overran the bound — abandon the daemon thread, free the worker.
            logger.warning(
                "[KG-2.286] task %s (%s) exceeded soft timeout %.0fs — abandoning "
                "for retry/dead_letter",
                job_id,
                task_type,
                timeout,
            )
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

        CONCEPT:KG-2.272 — the ``ingest_sessions`` MCP/REST handler enqueues large
        uploads as a ``session_upload`` task with the bundles on the Task node's
        metadata payload (same shape as ``kg_memory``); this runs on the host
        worker, off the request path. ``record_bundle`` is idempotent (replaces
        existing rows) so a retry is safe.
        """
        from agent_utilities.usage.models import ParsedSessionBundle
        from agent_utilities.usage.recorder import get_usage_recorder

        urows = self._control_cypher(
            "MATCH (t:Task {id: $id}) RETURN t.metadata as m", {"id": job_id}
        )
        umeta = _decode_metadata(urows[0]["m"]) if urows else {}
        payload = umeta.get("payload", {}) or {}
        bundles = payload.get("bundles", []) or []
        up_tenant = str(payload.get("tenant_id") or "")
        recorder = get_usage_recorder()
        ok = 0
        for item in bundles:
            try:
                bundle = ParsedSessionBundle.model_validate(item)
            except Exception as e:  # noqa: BLE001
                logger.warning("session_upload bad bundle skipped: %s", e)
                continue
            if up_tenant:
                bundle.session.tenant_id = up_tenant
            if recorder.record_bundle(bundle):
                ok += 1
        result = {"received": len(bundles), "ingested": ok}
        self._update_task_status(job_id, "completed", {"type": task_type, **result})
        return result

    async def _run_background_task(
        self, job_id: str, target: Path, is_codebase: bool, task_type: str = "document"
    ):
        """Execute the ingestion logic."""
        try:
            if task_type in ("scheduled_job", "enrichment_backfill"):
                # A recurring job enqueued by the unified scheduler (CONCEPT:OS-5.44).
                # ``enrichment_backfill`` is the same dispatch, only landed in the
                # dedicated enrichment lane so it isn't capped at the maint floor
                # (CONCEPT:KG-2.153).
                # The payload (the dispatch descriptor) rides on the task metadata;
                # run it through the single dispatcher and let the schedule's own
                # failure backoff govern cadence (so we do NOT route a job failure
                # through the task-level retry — that would double-retry).
                from agent_utilities.core.schedule_engine import (
                    record_schedule_result,
                    run_scheduled_job,
                )

                rows = self._control_cypher(
                    "MATCH (t:Task {id: $id}) RETURN t.metadata as m", {"id": job_id}
                )
                meta = _decode_metadata(rows[0]["m"]) if rows else {}
                sched_name = meta.get("schedule", "")
                payload = meta.get("payload", {})
                try:
                    result = run_scheduled_job(self, payload)
                    ok = str(result.get("status", "ok")) not in {"error", "failed"}
                except Exception as e:  # noqa: BLE001 — recorded as a schedule failure
                    result = {"status": "error", "error": str(e)}
                    ok = False
                if sched_name:
                    record_schedule_result(self, sched_name, ok)
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
                # (CONCEPT:KG-2.114). Enqueued by the RSS feed screen with a
                # grade-derived priority, so the best papers are fetched first.
                from agent_utilities.automation.research_pipeline import (
                    ResearchPipelineRunner,
                )

                rows = self._control_cypher(
                    "MATCH (t:Task {id: $id}) RETURN t.metadata as m", {"id": job_id}
                )
                meta = _decode_metadata(rows[0]["m"]) if rows else {}
                paper = meta.get("paper", {})
                runner = ResearchPipelineRunner(engine=self)  # type: ignore[arg-type]  # self is the engine
                from .ingest_profile import profile_ingest

                # OS-5.69/70 — profile token usage + per-stage timing for this paper.
                with profile_ingest(str(paper.get("id", ""))) as _prof:
                    article_id = await runner.ingest_paper_full(
                        paper.get("id", ""),
                        paper.get("title", ""),
                        paper.get("abstract", ""),
                        paper.get("authors", []),
                        # honor a pre-downloaded PDF (CONCEPT:KG-2.194) so a cohort
                        # ingests the full paper TEXT as an Article, not an abstract.
                        pdf_path=paper.get("pdf_path") or None,
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
                # CONCEPT:KG-2.130 — a memory write offloaded from a SERVING process. The
                # host performs the embed+write here (inline, _local=True so it never
                # re-enqueues), isolating heavy ingestion from the serving/read plane.
                rows = self._control_cypher(
                    "MATCH (t:Task {id: $id}) RETURN t.metadata as m", {"id": job_id}
                )
                meta = _decode_metadata(rows[0]["m"]) if rows else {}
                p = meta.get("payload", {})
                mid = self.store_memory(  # type: ignore[attr-defined]  # MemoryMixin, composed onto the engine
                    content=p.get("content", ""),
                    memory_type=p.get("memory_type", "episodic"),
                    name=p.get("name", ""),
                    tags=p.get("tags", []),
                    trust_score=p.get("trust_score", 0.8),
                    agent_id=p.get("agent_id", ""),
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
                # Content-aware URL ingest OFF the request path (CONCEPT:KG-2.7):
                # route through the unified IngestionEngine DOCUMENT path so the page
                # is fetched via the resolver (ArchiveBox→crawl4ai→requests) and a
                # research roundup auto-acquires the papers it cites. The real URL
                # rides on the Task node's ``source_url`` prop because the claim path
                # wraps ``target`` in Path() (which would collapse ``https://``).
                from agent_utilities.knowledge_graph.ingestion.engine import (
                    ContentType,
                    IngestionEngine,
                    IngestionManifest,
                )

                trow = self._control_cypher(
                    "MATCH (t:Task {id: $id}) RETURN t", {"id": job_id}
                )
                tprops = trow[0]["t"] if trow else {}
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
                # path (CONCEPT:KG-2.121). The world-model gate enqueues; the worker
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

                trow = self._control_cypher(
                    "MATCH (t:Task {id: $id}) RETURN t", {"id": job_id}
                )
                meta_t = _decode_metadata(trow[0]["t"].get("metadata")) if trow else {}
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
                # The RSS/FreshRSS sweep run OFF the request path (CONCEPT:KG-2.121).
                # The sweep is the "review" producer: it fetches (concurrently),
                # runs the world-model gate, and ENQUEUES per-article worldview/
                # research tasks. It does NOT ride the 300s MCP call — graph_feeds
                # sync enqueues this and returns immediately. The gate loop does
                # per-item engine work, so run it in a worker thread.
                from agent_utilities.knowledge_graph.core.source_sync import sync_source

                trow = self._control_cypher(
                    "MATCH (t:Task {id: $id}) RETURN t", {"id": job_id}
                )
                meta_t = _decode_metadata(trow[0]["t"].get("metadata")) if trow else {}
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
                # CONCEPT:KG-2.97 — ingest the universal-skills workflow corpus as
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
                trows = self._control_cypher(
                    "MATCH (t:Task {id: $id}) RETURN t", {"id": job_id}
                )
                t_props = trows[0]["t"] if trows else {}
                current_depth = int(t_props.get("current_depth", 0))
                max_depth = int(t_props.get("max_depth", DEFAULT_KG_ANALYSIS_MAX_DEPTH))

                # While a bulk codebase ingest is draining, run deep_analysis flat
                # (no recursive fan-out) so its 0-node, blocking-LLM jobs don't
                # flood the queue ahead of structural ingest. (CONCEPT:KG-2.7)
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
                # EnrichmentPipeline via IngestionEngine (CONCEPT:KG-2.8). The
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
                # KG_INGEST_PROFILE opt-out knobs are gone. (CONCEPT:KG-2.7)
                # Forward a caller-scoped file subset (CONCEPT:KG-2.150): the
                # agent-utilities self-ingest scopes a DIRTY tree to its
                # git-status-modified files via ``only_files`` on the task
                # metadata; pass it through so the ingest engine parses only those.
                cb_rows = self._control_cypher(
                    "MATCH (t:Task {id: $id}) RETURN t.metadata as m", {"id": job_id}
                )
                cb_meta = _decode_metadata(cb_rows[0]["m"]) if cb_rows else {}

                # CONCEPT:KG-2.287 — big-repo tail: if this is a whole-repo task for a
                # repo large enough to pin one worker/shard for minutes, fan it out
                # into K shard-routed sub-tasks instead of ingesting inline. Returns
                # True when it fanned out (this parent is done); the children run in
                # parallel across the K redb shard writers.
                if self._maybe_fanout_codebase(job_id, target, cb_meta):
                    return

                cb_manifest_meta: dict[str, Any] = {"features": True}
                only_files = cb_meta.get("only_files")
                if isinstance(only_files, list) and only_files:
                    cb_manifest_meta["only_files"] = only_files
                # CONCEPT:KG-2.287 — a split sub-task carries its own routing key so
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
            elif task_type == "connector_sync":
                # CONCEPT:ORCH-1.77 — one external connector's delta sync, run as a LANED task
                # (the 'connectors' lane). The */20m fleet sweep enqueues one of these per
                # connector so they fan out in PARALLEL instead of one slow connector
                # (gitlab/servicenow) blocking the rest in a sequential inline loop.
                from agent_utilities.knowledge_graph.core.source_sync import sync_source

                mrows = self._control_cypher(
                    "MATCH (t:Task {id: $id}) RETURN t.sync_mode as m", {"id": job_id}
                )
                mode = str((mrows[0].get("m") if mrows else None) or "delta")
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
                # CONCEPT:KG-2.301 — ONE paginated page of a chunked full-corpus drain. The
                # Task carries drain_id/source/mode top-level + the resumable connector
                # checkpoint in its metadata blob; ``run_drain_page`` drains this page, ingests
                # it, and self-continues by enqueuing the NEXT page-task while the cursor has
                # more — so a single ``source_sync(full)`` drains the whole corpus across many
                # capacity-guarded background tasks without ever blocking the request.
                from agent_utilities.knowledge_graph.core.chunked_drain import (
                    run_drain_page,
                )

                drows = self._control_cypher(
                    "MATCH (t:Task {id: $id}) RETURN t.sync_mode AS mode, "
                    "t.drain_id AS did, t.drain_source AS src, t.drain_page AS page, "
                    "t.metadata AS meta",
                    {"id": job_id},
                )
                drow = drows[0] if drows else {}
                dmeta = _decode_metadata(drow.get("meta")) if drow.get("meta") else {}
                drain_res = run_drain_page(
                    self,
                    source=str(drow.get("src") or target),
                    mode=str(drow.get("mode") or "full"),
                    drain_id=str(drow.get("did") or ""),
                    page=int(drow.get("page") or dmeta.get("drain_page") or 0),
                    checkpoint_json=dmeta.get("drain_checkpoint"),
                )
                self._update_task_status(
                    job_id,
                    "completed",
                    {"target": str(target), "type": task_type, **drain_res},
                )
            elif task_type == "fleet_event_triage":
                # Fleet-event triage (CONCEPT:OS-5.15): 'target' is the
                # FleetEvent node id enqueued by the gateway's
                # POST /api/fleet/events webhook receiver, not a filesystem
                # path. Correlates the event to known KG entities and files a
                # failure_gap topic when severity warrants. Remediation
                # playbooks (CONCEPT:OS-5.26) register on the dispatch seam
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
                # Health-gated deploy watch (CONCEPT:OS-5.27): 'target' is the
                # watched service name; the watch spec (window, deadline,
                # rollback params) rides on this Task node, so a watch
                # requeued by the zombie reaper resumes against its ORIGINAL
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
                srows = self._control_cypher(
                    "MATCH (t:Task {id: $id}) RETURN t", {"id": job_id}
                )
                t_props = srows[0]["t"] if srows else {}
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
                # Self-polling barrier gate for a research cohort (CONCEPT:KG-2.172):
                # once every member task is terminal (completed OR failed — a poison
                # member never wedges the cohort) or the deadline passes, run the
                # assimilation pass + materialize the feature matrix over whatever was
                # ingested. Until then re-defer ONE poll interval as 'scheduled' (NOT
                # a failure attempt) so the promotion sweep re-promotes it.
                from agent_utilities.knowledge_graph.research.cohort import (
                    cohort_ready,
                    finalize_cohort,
                )

                crow = self._control_cypher(
                    "MATCH (t:Task {id: $id}) RETURN t.metadata as meta", {"id": job_id}
                )
                cmeta = (
                    _decode_metadata(crow[0]["meta"])
                    if crow and crow[0].get("meta")
                    else {}
                )
                cohort_id = str(cmeta.get("cohort_id") or "")
                deadline = float(cmeta.get("deadline_unix", 0.0) or 0.0)
                ready, member_st = cohort_ready(self, cohort_id, deadline_unix=deadline)
                if not ready:
                    eta = time.time() + 60.0
                    cmeta["eta_unix"] = eta
                    cmeta["member_status"] = member_st
                    self._control_cypher(
                        "MATCH (t:Task {id: $id}) SET t.status = 'scheduled', "
                        "t.due_bucket = $due, t.metadata = $meta",
                        {
                            "id": job_id,
                            "due": int(eta // 60),
                            "meta": _encode_metadata(cmeta),
                        },
                    )
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
                # CONCEPT:KG-2.272 — drain a remote session-bundle upload that the
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
                # PyMuPDF for PDFs: C-based, GIL-releasing, ~0.2s vs pypdf's
                # multi-minute stall that would otherwise wedge the whole host.
                pdf_extractor = _pdf_file_extractor()
                if target.is_dir():
                    # exclude_hidden=False is REQUIRED: the research store lives
                    # under ``~/.local/share/...`` and SimpleDirectoryReader treats
                    # any file beneath a dot-dir (``.local``) as hidden, excluding
                    # everything → "No files found" despite PDFs present.
                    # recursive=False skips the ``.metadata`` sidecar dir;
                    # required_exts limits to real documents. (CONCEPT:KG-2.8)
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
                # batched call below. (CONCEPT:KG-2.8 ingestion throughput; see
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
            # (CONCEPT:KG-2.113). The reaper's crash-requeue is a separate path.
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

        CONCEPT:KG-2.5 — Per-Item Relevance Ranking
        """
        # Defer while a bulk ingest is in flight: this sweep scores every paper +
        # repo (heavy queries + embeddings) and, as a worker-pool task, runs
        # CONCURRENTLY with ingest on the single-writer engine. It's periodic, so
        # skipping a cycle is cheap — the maintenance scheduler re-enqueues it once
        # the ingest drains. (CONCEPT:KG-2.7)
        try:
            from agent_utilities.core.background_throttle import get_throttle

            if get_throttle().should_yield_background:
                logger.info(
                    "RelevanceSweep: deferring — bulk ingest/foreground active "
                    "(will retry next cycle)."
                )
                return {"status": "deferred", "reason": "bulk_ingest_or_foreground"}
        except ImportError:
            pass

        logger.info(f"RelevanceSweep: starting sweep against '{target_codebase}'")

        # ── Step 1: Compute target codebase centroid embedding ──
        target_articles = self.query_cypher(
            "MATCH (c:Code) WHERE c.file_path CONTAINS $name "
            "RETURN c.embedding AS emb LIMIT 200",
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
                "RETURN a.embedding AS emb LIMIT 100",
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
            "RETURN c.file_path AS path LIMIT 2000"
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
                    "RETURN a.embedding AS emb, a.content AS content LIMIT 50",
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

                # Architecture compatibility (heuristic based on content signals)
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
                logger.warning(f"RelevanceSweep: error scoring paper {paper_path}: {e}")

        # ── Step 5: Score each repository ──
        for repo_name in repo_set:
            try:
                repo_chunks = self.query_cypher(
                    "MATCH (c:Code) WHERE c.file_path CONTAINS $name "
                    "RETURN c.embedding AS emb, c.content AS content LIMIT 100",
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

            # Create RELEVANCE_SCORED edge (CONCEPT:KG-2.7 — registered edge type)
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
        except Exception as e:
            logger.debug(f"RelevanceSweep: edge persistence error for {item_id}: {e}")

    def query_relevance_rankings(
        self, target_codebase: str, top_k: int = 20
    ) -> list[dict]:
        """Query pre-computed relevance rankings from the KG.

        CONCEPT:KG-2.5 — Per-Item Relevance Ranking
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

        # Quick check: are there still pending/running tasks?
        remaining = self._control_cypher(
            "MATCH (t:Task) WHERE t.status IN ['pending', 'running'] RETURN count(t) as cnt"
        )
        if remaining and remaining[0].get("cnt", 0) > 0:
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

        threading.Thread(target=_build, daemon=True, name="KG-IndexBuilder").start()

    def _update_task_status(
        self, job_id: str, status: str, metadata: dict[str, Any]
    ) -> None:
        """Update a task's status and metadata using base64-encoded JSON.

        CONCEPT:KG-2.148 — :Task status/metadata is CONTROL plane → __control__.
        """
        if not self.backend:
            return

        # Preserve existing metadata timestamps (control-plane read).
        existing = self._control_cypher(
            "MATCH (t:Task {id: $id}) RETURN t.metadata as meta", {"id": job_id}
        )
        if existing and existing[0].get("meta"):
            old_meta = _decode_metadata(existing[0]["meta"])
            old_meta.update(metadata)
            metadata = old_meta

        if status in ("completed", "failed"):
            metadata.setdefault("completed_at", datetime.now(UTC).isoformat())
            # Central per-job duration (CONCEPT:KG-2.8 metrics) — computed from
            # started_at (stamped at claim time) so every category gets timing
            # without editing each completion site.
            started = metadata.get("started_at")
            if started and "duration_ms" not in metadata:
                try:
                    st = datetime.fromisoformat(started)
                    ct = datetime.fromisoformat(metadata["completed_at"])
                    metadata["duration_ms"] = round((ct - st).total_seconds() * 1000, 1)
                except (ValueError, TypeError):
                    pass

        encoded = _encode_metadata(metadata)
        self._control_cypher(
            "MATCH (t:Task {id: $id}) SET t.status = $status, t.metadata = $meta",
            {"id": job_id, "status": status, "meta": encoded},
        )
        self._checkpoint_db()

    def _fail_or_retry_task(
        self, job_id: str, error: str, details: dict[str, Any] | None = None
    ) -> None:
        """Handle an application-level task failure with retry/backoff/dead-letter.

        A task that *raises* (vs. a host crash, which the reaper handles via
        ``reaper_resets``) is retried with exponential backoff by re-scheduling
        it for a future minute (``status='scheduled'`` + ``due_bucket``), then
        dead-lettered once it exhausts ``max_attempts``. The two counters are
        deliberately separate: ``attempts`` answers "does this task reliably
        throw?", ``reaper_resets`` answers "did its host die?". (CONCEPT:KG-2.113)
        """
        if not self.backend:
            return
        # CONCEPT:KG-2.148 — :Task retry/dead-letter is CONTROL plane → __control__.
        rows = self._control_cypher(
            "MATCH (t:Task {id: $id}) RETURN t.metadata as meta", {"id": job_id}
        )
        meta = _decode_metadata(rows[0]["meta"]) if rows and rows[0].get("meta") else {}
        if details:
            meta.update(details)
        attempts = int(meta.get("attempts", 0)) + 1
        max_attempts = int(meta.get("max_attempts", _TASK_MAX_ATTEMPTS))
        meta["attempts"] = attempts
        meta["error"] = error
        if attempts >= max_attempts:
            meta["dead_letter_at"] = datetime.now(UTC).isoformat()
            logger.warning(
                "Task %s dead-lettered after %d attempts: %s",
                job_id,
                attempts,
                error,
            )
            self._update_task_status(job_id, "dead_letter", meta)
            return
        # Exponential backoff with jitter; re-route through the delayed-visibility
        # machinery so the promotion sweep makes it pending again when due.
        delay = _TASK_RETRY_BASE_SEC * (2 ** (attempts - 1))
        delay += (hash(job_id) % 1000) / 1000.0 * _TASK_RETRY_BASE_SEC  # nosec B311
        eta = time.time() + delay
        meta["eta_unix"] = eta
        meta.pop("claimed_by", None)
        meta.pop("claim_unix", None)
        due_bucket = int(eta // 60)
        self._control_cypher(
            "MATCH (t:Task {id: $id}) "
            "SET t.status = 'scheduled', t.due_bucket = $due, t.metadata = $meta",
            {"id": job_id, "due": due_bucket, "meta": _encode_metadata(meta)},
        )
        logger.info(
            "Task %s retry %d/%d scheduled in %.0fs (%s)",
            job_id,
            attempts,
            max_attempts,
            delay,
            error,
        )

    def aggregate_ingest_metrics(self, window_sec: int = 86400) -> dict[str, Any]:
        """Per-category ingest metrics from completed Task nodes (CONCEPT:KG-2.8).

        Powers the MCP ``graph_ingest`` jobs/job_status breakdown so polling shows
        time/nodes/edges/failures per content type — the same view the harness
        writes to ``progress.json``.
        """
        try:
            rows = self._control_cypher(
                "MATCH (t:Task) RETURN t.status as status, t.metadata as meta"
            )
        except Exception:  # noqa: BLE001
            return {}
        cutoff = None
        if window_sec:
            try:
                cutoff = datetime.now(UTC) - timedelta(seconds=window_sec)
            except Exception:  # noqa: BLE001
                cutoff = None
        cats: dict[str, dict[str, Any]] = {}
        for r in rows or []:
            meta = _decode_metadata(r.get("meta"))
            if cutoff is not None:
                ca = meta.get("completed_at")
                if ca:
                    try:
                        if datetime.fromisoformat(ca) < cutoff:
                            continue
                    except (ValueError, TypeError):
                        pass
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
            st = (r.get("status") or "").lower()
            if st in ("completed", "done", "success"):
                c["completed"] += 1
            elif st in ("failed", "error"):
                c["failed"] += 1
            c["nodes"] += int(
                meta.get("nodes_added", meta.get("nodes_created", 0)) or 0
            )
            c["edges"] += int(
                meta.get("edges_added", meta.get("edges_created", 0)) or 0
            )
            c["duration_ms"] += float(meta.get("duration_ms", 0) or 0)
        for c in cats.values():
            c["duration_ms"] = round(c["duration_ms"], 1)
        return cats

    def profile_report(
        self, window_sec: int = 86400, group_by: str = "lane"
    ) -> dict[str, Any]:
        """Per-lane / per-stage latency + cost profile from Task nodes (CONCEPT:OS-5.55).

        Where ``aggregate_ingest_metrics`` sums per content TYPE, this groups by a
        chosen dimension — ``lane`` (the functional task lane), ``type`` (the task
        type / pipeline stage), or ``tkind`` — and reports latency PERCENTILES
        (p50/p95/max) plus token/cost totals and a **parallelism factor** (sum of
        per-task durations ÷ wall-clock span). That is exactly the measurement a
        profiling run needs to PROVE a speed-up: the same corpus before vs after an
        optimization, and how much pipelining the staged lanes actually buy.

        Reads only metadata every task already carries — ``duration_ms`` (computed in
        ``_update_task_status``), ``lane``/``type`` (stamped at submit), and optional
        ``tokens``/``cost``/``usage`` when an LLM stage recorded them — so it adds no
        write path and covers EVERY ingestion lane uniformly.
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

        try:
            rows = self._control_cypher(
                "MATCH (t:Task) RETURN t.id as id, t.status as status, t.metadata as meta"
            )
        except Exception:  # noqa: BLE001
            return {}
        # OS-5.71 — fold in off-queue profile spans (assimilation, embed-backfill,
        # concept-registry embedding) so the report covers paths that never become
        # :Task nodes. They carry a task-shaped envelope (type='offqueue:<kind>').
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
        # CONCEPT:KG-2.288 — per-TASK tail: keep each task's identity+duration so the
        # report can name the slowest-N outliers (the p95/max offenders), not just
        # per-lane percentiles. This is what makes a 13-min codebase pin or a 456s
        # hung connector VISIBLE as a specific task, not a lane statistic.
        tail_tasks: list[dict[str, Any]] = []
        for r in rows or []:
            meta = _decode_metadata(r.get("meta"))
            ca = meta.get("completed_at")
            if cutoff is not None and ca:
                try:
                    if datetime.fromisoformat(ca) < cutoff:
                        continue
                except (ValueError, TypeError):
                    pass
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
            if dur > 0:
                grp["_durations"].append(dur)
                # CONCEPT:KG-2.288 — record the per-task tail entry.
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
            for ts, bucket in ((meta.get("started_at"), starts), (ca, ends)):
                if ts:
                    try:
                        bucket.append(datetime.fromisoformat(ts).timestamp())
                    except (ValueError, TypeError):
                        pass

        for grp in groups.values():
            durs = sorted(grp.pop("_durations"))
            grp["total_ms"] = round(sum(durs), 1)
            grp["p50_ms"] = round(_pct(durs, 50), 1)
            grp["p95_ms"] = round(_pct(durs, 95), 1)
            # CONCEPT:KG-2.288 — surface p99 alongside p95/max so a thin tail (a
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
        # CONCEPT:KG-2.288 — the slowest-N tasks overall: the concrete outliers a
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
        SQLite WAL) is checkpointed. The graph / tiered / Postgres backends route
        ``execute()`` through the Cypher engine, so the previous raw
        ``execute("CHECKPOINT;")`` fallback misparsed that string into a node
        query and **blocked indefinitely on the engine** — deadlocking every
        task worker after each ``_update_task_status`` (the live
        ``TieredGraphBackend`` wasn't in the old skip-list). There is nothing to
        WAL-checkpoint on those backends, so they are skipped. (CONCEPT:KG-2.8)
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
        """Get the status and decoded metadata for a specific task."""
        results = self._control_cypher(
            "MATCH (t:Task {id: $id}) RETURN t.status as status, t.metadata as meta",
            {"id": job_id},
        )
        if not results:
            return None

        status = results[0]["status"]
        meta = _decode_metadata(results[0].get("meta"))

        return {
            "job_id": job_id,
            "status": status,
            "metadata": meta,
        }

    def list_tasks(self) -> dict:
        """List all tasks grouped by status with decoded metadata."""
        results = self._control_cypher(
            "MATCH (t:Task) RETURN t.id as id, t.status as status, t.metadata as meta"
        )
        print(f"DEBUG: list_tasks results: {results}")
        response: dict[str, Any] = {
            "running": [],
            "pending": [],
            "completed": [],
            "failed": [],
        }

        for row in results:
            status = row["status"]
            meta = _decode_metadata(row.get("meta"))
            job_info: dict[str, Any] = {
                "job_id": row["id"],
                "target": meta.get("target", "unknown"),
            }
            if status == "failed":
                job_info["error"] = meta.get("error", "Unknown error")
                if meta.get("traceback"):
                    job_info["traceback"] = meta["traceback"]
                response["failed"].append(job_info)
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

        sqlite_queue_size = (
            self._submission_queue.get_queue_size()
            if hasattr(self, "_submission_queue")
            else 0
        )
        total_tasks = (
            sqlite_queue_size
            + len(response["running"])
            + len(response["pending"])
            + len(response["completed"])
            + len(response["failed"])
        )

        if total_tasks > 0:
            completed_count = len(response["completed"])
            progress = round((completed_count / total_tasks) * 100, 2)
            response["progress_percentage"] = f"{progress}% complete"
            response["progress_stats"] = {
                "total_tasks": total_tasks,
                "completed": completed_count,
                "pending_in_graph": len(response["pending"]),
                "running_in_graph": len(response["running"]),
                "queued_in_sqlite": sqlite_queue_size,
            }

        return response

    def remove_task(self, job_id: str) -> bool:
        """Remove a task from the graph."""
        res = self._control_cypher(
            "MATCH (t:Task {id: $id}) RETURN t.id as id", {"id": job_id}
        )
        if not res:
            return False

        self._control_cypher("MATCH (t:Task {id: $id}) DETACH DELETE t", {"id": job_id})
        return True

    def clear_completed_tasks(self) -> dict:
        """Clear all completed or failed tasks from the queue."""
        results = self._control_cypher(
            "MATCH (t:Task) WHERE t.status IN ['completed', 'failed'] "
            "RETURN count(t) as count"
        )
        cleared = results[0]["count"] if results else 0

        self._control_cypher(
            "MATCH (t:Task) WHERE t.status IN ['completed', 'failed'] DETACH DELETE t"
        )

        rem_results = self._control_cypher("MATCH (t:Task) RETURN count(t) as count")
        remaining = rem_results[0]["count"] if rem_results else 0

        return {"status": "success", "cleared": cleared, "remaining": remaining}

    def cancel_task(self, job_id: str) -> dict:
        """Cancel a single queued/running task by id (terminal 'cancelled').

        Removes it from the worker poll and the reaper's view without deleting the
        record (audit trail). A 'running' task's in-flight thread isn't interrupted,
        but the task is never re-claimed or requeued. (CONCEPT:KG-2.8 queue control)
        """
        if not job_id:
            return {"status": "error", "error": "job_id required"}
        rows = self._control_cypher(
            "MATCH (t:Task {id: $id}) RETURN t.status as s", {"id": job_id}
        )
        if not rows:
            return {"status": "error", "error": f"job {job_id} not found"}
        self._control_cypher(
            "MATCH (t:Task {id: $id}) SET t.status = 'cancelled'", {"id": job_id}
        )
        return {"status": "success", "job_id": job_id, "prev_status": rows[0].get("s")}

    def clear_tasks(self, status: str = "completed") -> dict:
        """Delete Task nodes from the queue by status filter (CONCEPT:KG-2.8 queue control).

        ``status`` ∈ pending|running|completed|failed|cancelled|zombie|all.
        ``zombie`` deletes only 'running' tasks NOT owned by the live host token
        (orphans from a dead host) — a targeted clear that never removes a task this
        host is actively processing. ``all`` clears every task. Returns counts.
        """
        status = (status or "completed").strip().lower()
        valid = {
            "pending",
            "running",
            "completed",
            "failed",
            "cancelled",
            "zombie",
            "all",
        }
        if status not in valid:
            return {
                "status": "error",
                "error": f"status must be one of {sorted(valid)}",
            }

        if status == "all":
            rows = self._control_cypher("MATCH (t:Task) RETURN t.id as id")
            ids = [r["id"] for r in (rows or []) if r.get("id")]
        elif status == "zombie":
            token = self._get_host_token()
            rows = self._control_cypher(
                "MATCH (t:Task {status: 'running'}) RETURN t.id as id, t.metadata as meta"
            )
            ids = []
            for r in rows or []:
                if not r.get("id"):
                    continue
                meta = _decode_metadata(r.get("meta")) or {}
                if meta.get("claimed_by") != token:  # foreign or unstamped → orphan
                    ids.append(r["id"])
        else:
            rows = self._control_cypher(
                "MATCH (t:Task {status: $s}) RETURN t.id as id", {"s": status}
            )
            ids = [r["id"] for r in (rows or []) if r.get("id")]

        for tid in ids:
            self._control_cypher(
                "MATCH (t:Task {id: $id}) DETACH DELETE t", {"id": tid}
            )

        rem = self._control_cypher("MATCH (t:Task) RETURN count(t) as count")
        remaining = rem[0]["count"] if rem else 0
        return {
            "status": "success",
            "cleared": len(ids),
            "filter": status,
            "remaining": remaining,
        }

    def prioritize_task(self, job_id: str, priority: str | int = "high") -> dict:
        """Re-prioritize a task by setting its claim bucket (CONCEPT:KG-2.113).

        Accepts a numeric bucket (0=critical .. 3=background) or a named level
        (``critical``/``high``/``normal``/``background``). The worker claim
        iterates buckets ascending, so a bumped job runs ahead of higher
        buckets. The legacy ``priority`` string is kept in lockstep for any
        node/tool that still reads it.
        """
        valid_names = {"critical", "high", "normal", "background", "low"}
        if (
            isinstance(priority, str)
            and not priority.strip().lstrip("-").isdigit()
            and priority.strip().lower() not in valid_names
        ):
            return {
                "status": "error",
                "error": "priority must be one of "
                f"{sorted(valid_names)} or a bucket 0-3",
            }
        bucket = _coerce_prio_bucket(priority)
        legacy = {0: "high", 1: "high", 2: "normal", 3: "normal"}[bucket]
        rows = self._control_cypher(
            "MATCH (t:Task {id: $id}) RETURN t.status as s", {"id": job_id}
        )
        if not rows:
            return {"status": "error", "error": f"job {job_id} not found"}
        self._control_cypher(
            "MATCH (t:Task {id: $id}) SET t.prio_bucket = $b, t.priority = $p",
            {"id": job_id, "b": bucket, "p": legacy},
        )
        return {
            "status": "success",
            "job_id": job_id,
            "priority": legacy,
            "prio_bucket": bucket,
            "task_status": rows[0].get("s"),
        }
