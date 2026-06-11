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
from typing import Any, Protocol

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
    import os

    role = (os.environ.get("KG_DAEMON_ROLE") or "auto").strip().lower()
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


# Embedding-backfill sizing. Previously the single overloaded
# ``KG_EMBED_BACKFILL_BATCH`` env was read in two places with CONFLICTING
# defaults (256 vs 512) for two genuinely different knobs — a config bug. They
# are now two named constants: the per-tick node budget and the per-query DB
# fetch size. (CONCEPT:KG-2.8 / config discipline)
_EMBED_BACKFILL_BUDGET = 256
_EMBED_BACKFILL_FETCH = 512

# A bulk ingest is "in progress" when the durable submission queue is at least
# this deep; the maintenance scheduler auto-defers its whole-graph passes while
# bulk-loading rather than contending with ingestion. Replaces the manual
# KG_BULK_INGEST flag — the engine already knows the queue depth. (CONCEPT:KG-2.7)
_BULK_QUEUE_THRESHOLD = 5


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
        self._claim_lock = threading.Lock()

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
        backend_type = str(getattr(config, "queue_backend", "sqlite")).lower()

        self._submission_queue: QueueBackend

        if backend_type == "nats":
            from .nats_queue_backend import NatsQueueBackend

            self._submission_queue = NatsQueueBackend(
                fallback_db_path=str(queue_db_path),
                nats_url=getattr(config, "nats_url", None),
            )
        elif backend_type == "kafka":
            from .kafka_queue_backend import KafkaQueueBackend

            self._submission_queue = KafkaQueueBackend(
                fallback_db_path=str(queue_db_path),
                bootstrap_servers=getattr(config, "kafka_bootstrap_servers", None),
            )
        else:
            self._submission_queue = SQLiteTaskQueue(str(queue_db_path))

        import os
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
            os.environ.get("AGENT_UTILITIES_TESTING") or "--stage-to-queue" in sys.argv
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
        try:
            q = getattr(self, "_submission_queue", None)
            if q is not None and hasattr(q, "depth"):
                status["queue_depth"] = q.depth()
        except Exception:  # noqa: BLE001
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
        last_relevance_sweep = getattr(self, "_last_relevance_sweep", 0.0)
        now = time.time()
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
        """Registry of periodic KG maintenance jobs: ``(name, interval_s, tick)``.

        One place to see/declare every background job. The scheduler runs them
        all in a single thread, each on its own interval, behind one shared
        foreground-throttle gate.
        """
        import os

        from agent_utilities.core.config import DEFAULT_KG_MODEL_ID

        jobs: list[tuple[str, float, Any]] = []
        # NOTE: embedding backfill is NOT a periodic maintenance job — it runs in
        # its own dedicated drain loop (``_embedding_backfill_loop``) so it isn't
        # starved behind the slower LLM ticks (analysis/enrichment) in this
        # single sequential scheduler thread.
        if DEFAULT_KG_MODEL_ID:
            jobs.append(("analysis", 120.0, self._tick_kg_analysis))
        # Self-evolution golden loop (propose-only) — throttled and OPT-IN
        # (autonomous LLM work). Enable with KG_GOLDEN_LOOP=1. (KG-2.7)
        from agent_utilities.core.config import config as _cfg

        if _cfg.kg_golden_loop:
            jobs.append(
                (
                    "golden_loop",
                    _cfg.kg_golden_loop_interval,
                    self._tick_golden_loop,
                )
            )
        # Failure-driven evolution (CONCEPT:AHE-3.18) — opt-in (KG_FAILURE_EVOLUTION=True).
        # Ingests Langfuse failures into failure-gap topics and runs a
        # regression-gated remediation cycle. This is the real telemetry sweep that
        # _tick_evolution used to (broken) stub out.
        if _cfg.kg_failure_evolution:
            jobs.append(
                (
                    "failure_ingest",
                    _cfg.kg_failure_evolution_interval,
                    self._tick_failure_ingest,
                )
            )
        jobs.append(("compaction", 1800.0, self._tick_compaction))
        jobs.append(
            (
                "evolution",
                float(os.getenv("KG_EVOLUTION_INTERVAL", "3600")),
                self._tick_evolution,
            )
        )
        # Durable-tier autoheal (CONCEPT:KG-2.8): backfill L1 (compute) → L2/L3
        # (durable Postgres) so the stores converge and an L1-only run / restart /
        # new node type can never silently diverge. Self-healing: runs ~15s after
        # startup then every KG_RECONCILE_INTERVAL. Registered only when a durable
        # reconcile exists (tiered backend). Disabled wholesale via KG_DEV_MODE.
        if callable(
            getattr(getattr(self, "backend", None), "reconcile_to_durable", None)
        ):
            jobs.append(
                (
                    "reconcile_durable",
                    float(os.getenv("KG_RECONCILE_INTERVAL", "900")),
                    self._tick_reconcile_durable,
                )
            )
        jobs.append(
            (
                "enrichment",
                float(os.getenv("KG_ENRICH_INTERVAL", "20")),
                self._tick_enrichment,
            )
        )
        # SDD/skills/scholarx/config file-watch as a periodic scan job — folded
        # in here instead of its own KGPlanWatcherThread + watchdog. Gated by
        # ``config.enable_sdd_watcher``. (CONCEPT:KG-2.6 / OS-5.0)
        try:
            from agent_utilities.core.config import config as _cfg

            watch_enabled = getattr(_cfg, "enable_sdd_watcher", True)
        except Exception:  # noqa: BLE001
            watch_enabled = True
        if watch_enabled:
            jobs.append(
                (
                    "file_watch",
                    float(os.getenv("KG_FILE_WATCH_INTERVAL", "30")),
                    self._tick_file_watch,
                )
            )
        # Memory hygiene: decay-archive stale AI memory + semantic-merge dedup (CONCEPT:KG-2.17).
        # Long interval (default daily) — bounded maintenance.
        jobs.append(
            (
                "hygiene",
                float(os.getenv("KG_HYGIENE_INTERVAL", "86400")),
                self._tick_hygiene,
            )
        )
        # Zombie/stuck task reaper: requeue 'running' tasks orphaned by a dead
        # worker/host so a killed/redeployed host's in-flight ingestions are
        # recovered within minutes instead of stranding forever. Short interval.
        # (CONCEPT:KG-2.8 ingestion durability)
        jobs.append(
            (
                "task_reaper",
                float(os.getenv("KG_TASK_REAPER_INTERVAL", "120")),
                self._tick_task_reaper,
            )
        )
        return jobs

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
        import os

        from .host_lock import effective_daemon_role

        if effective_daemon_role() != "host":
            return
        try:
            grace = float(os.getenv("KG_TASK_ORPHAN_GRACE_SEC", "90"))
            max_runtime = float(os.getenv("KG_TASK_MAX_RUNTIME_SEC", "7200"))
            max_resets = int(os.getenv("KG_TASK_MAX_REQUEUE", "3"))
            now = time.time()
            token = self._get_host_token()

            rows = self.query_cypher(
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
                    self.backend.execute(
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
                self.backend.execute(
                    "MATCH (t:Task {id: $id, status: 'running'}) SET t.status = 'pending', t.metadata = $meta",
                    {"id": tid, "meta": _encode_metadata(meta)},
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

    def _maintenance_scheduler_loop(self) -> None:
        """Single thread running all periodic KG maintenance jobs.

        Replaces the former per-job daemon threads (analysis / compaction /
        evolution / enrichment). One backend-readiness check and one
        foreground-throttle gate guard every job, so background work uniformly
        yields the GPU/LLM to interactive runs. (CONCEPT:KG-2.7 / KG-2.8)
        """
        import time

        jobs = self._maintenance_jobs()
        if not jobs:
            return
        names = ", ".join(n for n, _, _ in jobs)
        logger.info("KG maintenance scheduler started with jobs: %s", names)

        POLL = 5.0
        # Stagger first runs so a startup burst doesn't fire everything at once.
        last_run = {name: time.time() - interval + 15.0 for name, interval, _ in jobs}

        while True:
            try:
                if not getattr(self, "backend", None):
                    time.sleep(10.0)
                    continue

                # Single shared foreground gate for ALL background jobs.
                try:
                    from agent_utilities.core.background_throttle import get_throttle

                    if get_throttle().foreground_active:
                        time.sleep(POLL)
                        continue
                except ImportError:
                    pass

                # Auto-defer while a bulk ingest drains: the whole-graph passes
                # contend with ingestion for the single-writer engine. Detected
                # from the durable submission-queue depth, not a manual flag.
                q = getattr(self, "_submission_queue", None)
                if q is not None:
                    try:
                        if q.get_queue_size() > _BULK_QUEUE_THRESHOLD:
                            time.sleep(POLL)
                            continue
                    except Exception:  # noqa: BLE001 — queue probe best-effort
                        pass

                now = time.time()
                for name, interval, tick in jobs:
                    if now - last_run[name] < interval:
                        continue
                    try:
                        tick()
                    except Exception as e:  # one job's failure never stops others
                        logger.error("Maintenance job '%s' error: %s", name, e)
                    last_run[name] = time.time()
                time.sleep(POLL)
            except Exception as e:
                logger.error(f"MaintenanceScheduler error: {e}")
                time.sleep(30.0)

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
        import os

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

        BATCH = int(os.environ.get("KG_ENRICH_BATCH", "16"))
        MAX_BATCHES = int(os.environ.get("KG_ENRICH_MAX_BATCHES", "8"))
        max_workers = int(os.environ.get("KG_LLM_CONCURRENCY", "6"))
        llm_fn = getattr(self, "_enrich_llm_fn", None)

        for _ in range(MAX_BATCHES):
            # Yield to interactive runs between batches.
            try:
                from agent_utilities.core.background_throttle import get_throttle

                if get_throttle().foreground_active:
                    return
            except ImportError:
                pass

            rows = self.query_cypher(
                "MATCH (n:Code) WHERE n.summary = '' AND n.ast_hash IS NOT NULL "
                "RETURN n.id AS id, n.name AS name, n.kind AS kind, "
                "n.file_path AS file_path, n.patterns AS patterns, "
                "n.ast_hash AS ast_hash LIMIT " + str(BATCH)
            )
            if not rows:
                return
            if llm_fn is None:
                # Card summaries are a structured extraction task — route to the
                # LITE chat model by default (markedly faster than the heavy KG
                # model, which is what saturated the engine on a full backfill).
                # ``KG_CARD_MODEL=heavy`` forces the heavy model. (CONCEPT:KG-2.8)
                use_heavy = os.environ.get("KG_CARD_MODEL", "lite").lower() == "heavy"
                llm_fn = make_llm_fn() if use_heavy else make_lite_llm_fn()
                self._enrich_llm_fn = llm_fn

            ents = [
                CodeEntity(
                    id=r["id"],
                    name=r.get("name") or r["id"],
                    qualname=r.get("name") or r["id"],
                    kind=r.get("kind") or "function",
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
                with conn_factory() as conn, conn.cursor() as cur:
                    cur.execute(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = %s",
                        (tbl,),
                    )
                    cols = {r[0] for r in cur.fetchall()}
                    text_cols = [c for c in self._EMBED_TEXT_COLS if c in cols]
                    if not text_cols or "embedding" not in cols:
                        continue
                    expr = " || ' ' || ".join(
                        f"COALESCE(\"{c}\",'')" for c in text_cols
                    )
                    cur.execute(
                        f'SELECT id, {expr} FROM "{tbl}" '  # nosec B608
                        "WHERE embedding IS NULL LIMIT %s",
                        (take,),
                    )
                    rows = cur.fetchall()
            except Exception as e:  # noqa: BLE001
                logger.debug("embed backfill: query %s failed: %s", tbl, e)
                continue

            items = [(r[0], (r[1] or "").strip()) for r in rows]
            items = [(nid, txt) for nid, txt in items if txt]
            if not items:
                continue
            try:
                vecs = embed_fn([t for _, t in items])
                with conn_factory() as conn, conn.cursor() as cur:
                    for (nid, _), vec in zip(items, vecs, strict=False):
                        cur.execute(
                            f'UPDATE "{tbl}" SET embedding = %s::vector '  # nosec B608
                            "WHERE id = %s",
                            (str(vec), nid),
                        )
                    conn.commit()
                total += len(items)
                remaining -= len(items)
            except Exception as e:  # noqa: BLE001
                logger.debug("embed backfill: store %s failed: %s", tbl, e)
        if total:
            logger.info("KG embedding backfill: embedded %d nodes", total)
        return total

    def _tick_golden_loop(self) -> None:
        """One propose-only self-evolution cycle (CONCEPT:KG-2.7).

        Runs ``GoldenLoopController.run_one_cycle`` (intake unresolved topics →
        acquire related sources → ADDRESSES resolve → optional distill/synthesize
        as DRAFTS/proposals). Always propose-only: nothing is auto-merged or
        executed. Throttled + opt-in via ``KG_GOLDEN_LOOP``.
        """
        try:
            from agent_utilities.core.config import config as _cfg

            from ..research.golden_loop import GoldenLoopController

            rep = GoldenLoopController(self).run_one_cycle(
                max_topics=_cfg.kg_golden_loop_topics
            )
            logger.info(
                "Golden loop cycle: intake=%s resolved=%s sources=%s team=%s",
                rep.get("topics_intake"),
                rep.get("topics_resolved"),
                rep.get("sources_linked"),
                bool(rep.get("team")),
            )
        except Exception as e:  # noqa: BLE001
            logger.error("golden_loop tick error: %s", e)

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

    def _embedding_backfill_loop(self) -> None:
        """Dedicated drain loop for vector-embedding backfill (CONCEPT:KG-2.8).

        Runs independently of the periodic maintenance scheduler so it is NOT
        blocked behind slow LLM ticks: it embeds a batch, and if work remains
        (a full batch landed) loops again almost immediately; when the graph is
        fully embedded it idles at a long interval. Yields to interactive runs
        via the shared foreground throttle.
        """
        import os
        import time

        batch = _EMBED_BACKFILL_FETCH
        try:
            idle = float(os.getenv("KG_EMBED_BACKFILL_INTERVAL", "30"))
        except ValueError:
            idle = 30.0
        busy = float(os.getenv("KG_EMBED_BACKFILL_BUSY_SLEEP", "1"))

        while True:
            try:
                if not getattr(self, "backend", None):
                    time.sleep(idle)
                    continue
                # Yield to interactive/foreground work.
                try:
                    from agent_utilities.core.background_throttle import get_throttle

                    if get_throttle().foreground_active:
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
        import os
        from datetime import datetime

        EVOLUTION_INTERVAL = float(os.getenv("KG_EVOLUTION_INTERVAL", "3600"))
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

                # This call will block if the DB is locked by worker threads,
                # but it won't hang the MCP endpoint!
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
    ) -> str:
        """Submit a background ingestion task to the KG natively."""
        if not skip_dedupe:
            existing = self.query_cypher(
                "MATCH (t:Task) WHERE t.status IN ['pending', 'running'] RETURN t.id as id, t.metadata as meta"
            )
            for row in existing:
                meta = _decode_metadata(row.get("meta"))
                if meta and meta.get("target") == target_path:
                    return row["id"]

        job_id = f"job-{uuid.uuid4().hex[:8]}"

        if not task_type:
            task_type = "codebase" if is_codebase else "document"

        task_data = {
            "target": target_path,
            "type": task_type,
            "submitted_at": datetime.now(UTC).isoformat(),
        }

        encoded_meta = _encode_metadata(task_data)
        props = {"status": "pending", "metadata": encoded_meta}
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

    def _bulk_ingest_active(self, threshold: int = 1) -> bool:
        """True if ``threshold``+ codebase ingest tasks are pending/running.

        Used to gate recursive ``deep_analysis`` fan-out: while a bulk codebase
        ingest is draining, ``deep_analysis`` (0-node, recursive, blocking-LLM)
        runs flat (no fan-out) so it can't flood the queue ahead of structural
        ingest. (CONCEPT:KG-2.7 / KG-2.8)
        """
        try:
            rows = self.query_cypher(
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
        for task in self.query_cypher(
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
            DEFAULT_KG_INGESTION_WORKERS,
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
            worker_count = DEFAULT_KG_INGESTION_WORKERS
            try:
                import os

                import psutil

                # Calculate based on available memory (assume 3GB RAM per heavy worker)
                mem = psutil.virtual_memory()
                available_gb = mem.available / (1024**3)
                max_mem_workers = max(1, int(available_gb / 3.0))

                # Calculate based on CPU cores (target 36% max utilization)
                cores = os.cpu_count() or 4
                max_cpu_workers = max(1, int(cores * 0.36))

                # Cap workers between 2 and 36% CPU max, constrained by available memory
                dynamic_workers = max(2, min(max_cpu_workers, max_mem_workers))

                # Use the dynamic scale directly to maximize parallelization!
                worker_count = dynamic_workers
            except Exception as e:
                if worker_count is None:
                    worker_count = 4
                logger.debug(
                    f"Dynamic worker scaling failed, falling back to {worker_count}: {e}"
                )

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

        logger.info(f"Starting {worker_count} TaskManager workers...")
        for i in range(worker_count):
            t = threading.Thread(
                target=self._task_worker_loop, name=f"KGTaskWorker-{i}", daemon=True
            )
            t.start()

    def _task_worker_loop(self):
        """Distributed polling loop that picks up pending tasks natively."""
        while True:
            try:
                # Use a thread lock to prevent multiple workers from claiming the same task simultaneously
                job_id = None
                target_path = None
                is_codebase = False
                task_type = "document"

                if not hasattr(self, "_claim_lock"):
                    self._claim_lock = threading.Lock()

                with self._claim_lock:
                    # Priority-aware poll: claim ``priority='high'`` pending tasks
                    # first, then any pending. (The L1 interpreter strips ORDER BY,
                    # so we tier via two queries instead of ORDER BY priority.)
                    # (CONCEPT:KG-2.8 queue control)
                    results = self.query_cypher(
                        "MATCH (t:Task {status: 'pending', priority: 'high'}) RETURN t.id as id, t.metadata as meta LIMIT 1"
                    )
                    if not results:
                        results = self.query_cypher(
                            "MATCH (t:Task {status: 'pending'}) RETURN t.id as id, t.metadata as meta LIMIT 1"
                        )

                    if results:
                        job_id = results[0]["id"]
                        meta = _decode_metadata(results[0].get("meta"))
                        if meta:
                            if "target" in meta:
                                target_path = Path(meta["target"])
                            task_type = meta.get("type", "document")
                            is_codebase = task_type == "codebase"
                            meta["started_at"] = datetime.now(UTC).isoformat()
                            # Ownership stamp for the zombie reaper: the live host's
                            # unique token + a unix claim time. The singleton host
                            # lock guarantees exactly one host runs workers, so any
                            # 'running' task NOT stamped with the live token is an
                            # orphan from a dead host and is safe to requeue.
                            # (CONCEPT:KG-2.8 ingestion durability)
                            meta["claimed_by"] = self._get_host_token()
                            meta["claim_unix"] = time.time()
                            encoded_meta = _encode_metadata(meta)

                        # Immediately claim it while holding the lock
                        self.backend.execute(
                            "MATCH (t:Task {id: $id, status: 'pending'}) SET t.status = 'running', t.metadata = $meta",
                            {"id": job_id, "meta": encoded_meta if meta else ""},
                        )

                if not job_id:
                    time.sleep(2.0)
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
                    time.sleep(2.0)
                    continue

                # Execute the task asynchronously inside this thread (lock is
                # released). Heavy task types (parse storms / background LLM /
                # analysis) run through the shared background throttle so they
                # yield to interactive (foreground) work and stay within the
                # global concurrency cap — a bulk ingest can no longer consume the
                # engine's whole in-flight budget and starve live queries
                # (CONCEPT:KG-2.7 read/ingest plane isolation). Lightweight types
                # (diff/conversation/…) run unthrottled.
                _HEAVY_TASK_TYPES = {
                    "codebase",
                    "document",
                    "deep_analysis",
                    "synthesize",
                    "deep_extract",
                    "background_research",
                    "relevance_sweep",
                }
                if task_type in _HEAVY_TASK_TYPES:
                    from agent_utilities.core.background_throttle import get_throttle

                    with get_throttle().background_slot():
                        asyncio.run(
                            self._run_background_task(
                                job_id, target_path, is_codebase, task_type
                            )
                        )
                else:
                    asyncio.run(
                        self._run_background_task(
                            job_id, target_path, is_codebase, task_type
                        )
                    )

                # Post-ingestion: auto-build HNSW indexes when queue drains
                self._maybe_build_vector_indexes()

            except Exception as e:
                logger.error(f"TaskManager worker error: {e}")
                if job_id:
                    try:
                        self._update_task_status(job_id, "failed", {"error": str(e)})
                    except Exception as inner_e:
                        logger.error(
                            f"Failed to update task status to failed for {job_id}: {inner_e}"
                        )
                time.sleep(5)

    async def _run_background_task(
        self, job_id: str, target: Path, is_codebase: bool, task_type: str = "document"
    ):
        """Execute the ingestion logic."""
        try:
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

                props = {
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
                res = self.query_cypher(
                    "MATCH (t:Task {id: $id}) RETURN t", {"id": job_id}
                )
                t_props = res[0]["t"] if res else {}
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
                ing = IngestionEngine(kg_engine=self)
                cb_res = await ing.ingest(
                    IngestionManifest(
                        content_type=ContentType.CODEBASE,
                        source_uri=str(target),
                        metadata={"features": True},
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
            elif task_type in ("synthesize", "deep_extract", "background_research"):
                from agent_utilities.analysis.analyzer import GraphAnalyzer

                analyzer = GraphAnalyzer(self)
                query = str(target)

                # Fetch metadata to track top_k if provided
                res = self.query_cypher(
                    "MATCH (t:Task {id: $id}) RETURN t", {"id": job_id}
                )
                t_props = res[0]["t"] if res else {}
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
                    self._update_task_status(
                        job_id, "failed", {"error": str(e), "type": task_type}
                    )
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
            self._update_task_status(
                job_id,
                "failed",
                {
                    "error": error_msg,
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
        import numpy as np

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
        remaining = self.query_cypher(
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
        """Update a task's status and metadata using base64-encoded JSON."""
        if not self.backend:
            return

        # Preserve existing metadata timestamps
        existing = self.query_cypher(
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
        self.backend.execute(
            "MATCH (t:Task {id: $id}) SET t.status = $status, t.metadata = $meta",
            {"id": job_id, "status": status, "meta": encoded},
        )
        self._checkpoint_db()

    def aggregate_ingest_metrics(self, window_sec: int = 86400) -> dict[str, Any]:
        """Per-category ingest metrics from completed Task nodes (CONCEPT:KG-2.8).

        Powers the MCP ``graph_ingest`` jobs/job_status breakdown so polling shows
        time/nodes/edges/failures per content type — the same view the harness
        writes to ``progress.json``.
        """
        try:
            rows = self.query_cypher(
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
        results = self.query_cypher(
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
        results = self.query_cypher(
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
        res = self.query_cypher(
            "MATCH (t:Task {id: $id}) RETURN t.id as id", {"id": job_id}
        )
        if not res:
            return False

        self.backend.execute("MATCH (t:Task {id: $id}) DETACH DELETE t", {"id": job_id})
        return True

    def clear_completed_tasks(self) -> dict:
        """Clear all completed or failed tasks from the queue."""
        results = self.query_cypher(
            "MATCH (t:Task) WHERE t.status IN ['completed', 'failed'] "
            "RETURN count(t) as count"
        )
        cleared = results[0]["count"] if results else 0

        self.backend.execute(
            "MATCH (t:Task) WHERE t.status IN ['completed', 'failed'] DETACH DELETE t"
        )

        rem_results = self.query_cypher("MATCH (t:Task) RETURN count(t) as count")
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
        rows = self.query_cypher(
            "MATCH (t:Task {id: $id}) RETURN t.status as s", {"id": job_id}
        )
        if not rows:
            return {"status": "error", "error": f"job {job_id} not found"}
        self.backend.execute(
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
            rows = self.query_cypher("MATCH (t:Task) RETURN t.id as id")
            ids = [r["id"] for r in (rows or []) if r.get("id")]
        elif status == "zombie":
            token = self._get_host_token()
            rows = self.query_cypher(
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
            rows = self.query_cypher(
                "MATCH (t:Task {status: $s}) RETURN t.id as id", {"s": status}
            )
            ids = [r["id"] for r in (rows or []) if r.get("id")]

        for tid in ids:
            self.backend.execute(
                "MATCH (t:Task {id: $id}) DETACH DELETE t", {"id": tid}
            )

        rem = self.query_cypher("MATCH (t:Task) RETURN count(t) as count")
        remaining = rem[0]["count"] if rem else 0
        return {
            "status": "success",
            "cleared": len(ids),
            "filter": status,
            "remaining": remaining,
        }

    def prioritize_task(self, job_id: str, priority: str = "high") -> dict:
        """Set a pending task's queue priority ('high' jumps ahead of normal).

        The worker poll claims ``priority='high'`` pending tasks before any other
        pending task, so a bumped job runs next. (CONCEPT:KG-2.8 queue control)
        """
        priority = (priority or "high").strip().lower()
        if priority not in {"high", "normal"}:
            return {"status": "error", "error": "priority must be 'high' or 'normal'"}
        rows = self.query_cypher(
            "MATCH (t:Task {id: $id}) RETURN t.status as s", {"id": job_id}
        )
        if not rows:
            return {"status": "error", "error": f"job {job_id} not found"}
        self.backend.execute(
            "MATCH (t:Task {id: $id}) SET t.priority = $p",
            {"id": job_id, "p": priority},
        )
        return {
            "status": "success",
            "job_id": job_id,
            "priority": priority,
            "task_status": rows[0].get("s"),
        }
