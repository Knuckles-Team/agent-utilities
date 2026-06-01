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

from agent_utilities.core.config import (
    DEFAULT_KG_INGESTION_WORKERS,
    DEFAULT_KNOWLEDGE_GRAPH_SYNC_BACKGROUND,
)

logger = logging.getLogger(__name__)

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


class SQLiteTaskQueue(QueueBackend):
    """Thread-safe, persistent SQLite-backed queue for tasks to prevent memory loss on restarts."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.Lock()
        with self.lock:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            try:
                with conn:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS queue (id INTEGER PRIMARY KEY, data TEXT)"
                    )
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS staging (id INTEGER PRIMARY KEY, job_id TEXT, graph_data TEXT)"
                    )
            finally:
                conn.close()

    def put(self, item: dict):
        with self.lock:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            try:
                with conn:
                    conn.execute(
                        "INSERT INTO queue (data) VALUES (?)", (json.dumps(item),)
                    )
            finally:
                conn.close()

    def get(self) -> tuple[int, dict] | None:
        with self.lock:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
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
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            try:
                with conn:
                    conn.execute("DELETE FROM queue WHERE id = ?", (item_id,))
            finally:
                conn.close()

    def get_queue_size(self) -> int:
        with self.lock:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
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
            conn = sqlite3.connect(self.db_path, timeout=30.0)
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
            conn = sqlite3.connect(self.db_path, timeout=30.0)
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
            conn = sqlite3.connect(self.db_path, timeout=30.0)
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

        if os.environ.get("AGENT_UTILITIES_TESTING") or "--stage-to-queue" in sys.argv:
            # In test mode, skip all background daemon threads to prevent
            # pytest-xdist worker hangs from orphaned threads on closed backends.
            return

        if "--stage-to-queue" not in sys.argv:
            # Start the dedicated queue writer daemon thread
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

            from agent_utilities.core.config import DEFAULT_KG_MODEL_ID

            if DEFAULT_KG_MODEL_ID:
                self._kg_analysis_thread = threading.Thread(
                    target=self._kg_analysis_loop,
                    daemon=True,
                    name="KG-Analysis-Daemon",
                )
                self._kg_analysis_thread.start()

            # Start conversation compaction daemon (LCM)
            self._compaction_thread = threading.Thread(
                target=self._conversation_compaction_loop,
                daemon=True,
                name="KG-Compaction-Daemon",
            )
            self._compaction_thread.start()

            # Start evolution cycle daemon (research-driven development)
            self._evolution_thread = threading.Thread(
                target=self._evolution_cycle_loop,
                daemon=True,
                name="KG-Evolution-Daemon",
            )
            self._evolution_thread.start()

            # Start SDD plan/tasks watcher daemon natively
            self.start_sdd_watcher()

    def start_sdd_watcher(self):
        """Start the background plan/task watcher thread natively.

        CONCEPT:KG-2.6 — Implementation Plan & Tasks versioning and KG lineage.
        """
        import os
        import sys

        if os.environ.get("AGENT_UTILITIES_TESTING") or "--stage-to-queue" in sys.argv:
            logger.debug("Skipping plan watcher in test mode / staging.")
            return

        from agent_utilities.core.config import config

        if not config.enable_sdd_watcher:
            logger.info(
                "Plan watcher is disabled via config.json / environment variables."
            )
            return

        if getattr(self, "_watcher_thread_running", False):
            logger.debug("Plan watcher thread is already running.")
            return

        try:
            from agent_utilities.sdd.watcher import (
                get_workspace_path,
                run_plan_watcher_loop,
            )

            workspace_path = get_workspace_path()
            self._watcher_thread_running = True
            self._plan_watcher_thread = threading.Thread(
                target=run_plan_watcher_loop,
                args=(self, workspace_path),
                daemon=True,
                name="KGPlanWatcherThread",
            )
            self._plan_watcher_thread.start()
            logger.info(
                f"Successfully launched background KGPlanWatcherThread for {workspace_path}"
            )
        except Exception as e:
            self._watcher_thread_running = False
            logger.error(f"Failed to start background KGPlanWatcherThread: {e}")

    def _kg_analysis_loop(self):
        """Background daemon thread to autonomously synthesize relationships and concepts using LLMs.

        Also schedules periodic relevance sweeps every 60 minutes.
        CONCEPT:KG-2.4 — Autonomous Background Analysis
        """
        import time

        last_relevance_sweep = 0.0
        RELEVANCE_SWEEP_INTERVAL = 3600.0  # 60 minutes

        while True:
            try:
                if not getattr(self, "backend", None):
                    time.sleep(10.0)
                    continue

                # ── Periodic Relevance Sweep (every 60 min) ──
                now = time.time()
                if now - last_relevance_sweep >= RELEVANCE_SWEEP_INTERVAL:
                    try:
                        # Find the primary codebase (most Code nodes)
                        primary = self._detect_primary_codebase()
                        if primary:
                            logger.info(
                                f"KGAnalysisDaemon: scheduling relevance sweep for '{primary}'"
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
                        last_relevance_sweep = now
                    except Exception as e:
                        logger.error(f"Relevance sweep scheduling error: {e}")

                # ── Deep Analysis (existing) ──
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
                    time.sleep(300.0)
                    continue

                node_id = results[0]["id"]
                node_name = results[0].get("name") or node_id

                logger.info(
                    f"KGAnalysisDaemon autonomously selected '{node_name}' ({node_id}) for background deep analysis."
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

                time.sleep(120.0)
            except Exception as e:
                logger.error(f"KGAnalysisDaemon error: {e}")
                time.sleep(60.0)

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

    def _conversation_compaction_loop(self):
        """Background daemon to compact large conversation threads.

        CONCEPT:KG-2.1 — LCM Compaction Daemon

        Runs every 30 minutes. Finds Thread nodes with more than
        COMPACTION_THRESHOLD uncompacted messages and delegates to the
        unified ElasticContextManager.compact_thread() for the actual work.

        Same pattern as _kg_analysis_loop — no new daemon infrastructure.
        """
        import time

        COMPACTION_THRESHOLD = 30
        COMPACTION_INTERVAL = 1800.0  # 30 minutes

        # Wait for backend to be ready
        time.sleep(30.0)

        while True:
            try:
                if not getattr(self, "backend", None):
                    time.sleep(30.0)
                    continue

                # Find threads with many uncompacted messages
                threads = self.query_cypher(
                    "MATCH (t:Thread)-[:CONTAINS]->(m:Message) "
                    "WITH t, count(m) AS msg_count "
                    "WHERE msg_count > $threshold "
                    "AND (t.last_compacted IS NULL) "
                    "RETURN t.id AS id, msg_count "
                    "ORDER BY msg_count DESC LIMIT 3",
                    {"threshold": COMPACTION_THRESHOLD},
                )

                if threads:
                    from agent_utilities.knowledge_graph.memory import (
                        ElasticContextManager,
                    )

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
                                "CompactionDaemon: compacted thread %s (%d msgs) → %s",
                                thread_id,
                                msg_count,
                                result.get("status", "unknown"),
                            )
                        except Exception as e:
                            logger.warning(
                                f"CompactionDaemon: failed to compact {thread_id}: {e}"
                            )

                time.sleep(COMPACTION_INTERVAL)
            except Exception as e:
                logger.error(f"CompactionDaemon error: {e}")
                time.sleep(120.0)

    def _evolution_cycle_loop(self):
        """Background daemon to run autonomous research evolution cycles.

        CONCEPT:KG-2.5 — Evolution Cycle Daemon

        Runs every 60 minutes (configurable via KG_EVOLUTION_INTERVAL).
        Queries the KG for unresolved research topics, triggers relevance
        sweeps against the primary codebase, and logs evolution cycle
        metrics as EvolutionCycle nodes.

        Same daemon pattern as _kg_analysis_loop and _conversation_compaction_loop.
        """
        import os
        import time
        from datetime import datetime

        EVOLUTION_INTERVAL = float(os.getenv("KG_EVOLUTION_INTERVAL", "3600"))

        # Wait for backend initialization
        time.sleep(60.0)

        while True:
            try:
                if not getattr(self, "backend", None):
                    time.sleep(30.0)
                    continue

                cycle_start = datetime.now(UTC)
                cycle_id = f"evo_cycle_{cycle_start.strftime('%Y%m%d_%H%M%S')}"
                logger.info("EvolutionDaemon: starting cycle %s", cycle_id)

                # 1. Detect unresolved research topics
                topics = self.query_cypher(
                    "MATCH (c:Concept) OPTIONAL MATCH (c)-[:ADDRESSED_BY]->(p) "
                    "WHERE p IS NULL RETURN c.id AS id, c.name AS name ORDER BY c.name LIMIT 15"
                )
                topic_count = len(topics) if topics else 0
                logger.info("EvolutionDaemon: found %d unresolved topics", topic_count)

                # 2. Detect primary codebase
                primary_codebase = self._detect_primary_codebase()

                # 3. Run relevance sweep if we have a codebase target
                papers_scored = 0
                if primary_codebase and topic_count > 0:
                    try:
                        # Count total papers/codebases available for scoring
                        count_result = self.query_cypher(
                            "MATCH (n) WHERE n:Document OR n:Codebase "
                            "RETURN count(n) AS total",
                        )
                        papers_scored = (
                            count_result[0].get("total", 0) if count_result else 0
                        )

                        logger.info(
                            "EvolutionDaemon: %d items available for relevance sweep against '%s'",
                            papers_scored,
                            primary_codebase,
                        )
                    except Exception as e:
                        logger.warning(f"EvolutionDaemon: relevance count failed: {e}")

                # 4. Log evolution cycle as a KG node
                try:
                    from agent_utilities.knowledge_graph.core.engine import (
                        IntelligenceGraphEngine,
                    )

                    # 4.5. Log OptimizationTrajectoryNode throughput
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
                            throughput_query[0].get("throughput", 0)
                            if throughput_query
                            else 0
                        )
                        logger.info(
                            "EvolutionDaemon: OptimizationTrajectoryNode throughput = %d",
                            throughput,
                        )
                    except Exception as e:
                        logger.warning(
                            f"EvolutionDaemon: failed to get throughput: {e}"
                        )

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
                            "EvolutionDaemon: logged cycle %s (topics=%d, scored=%d)",
                            cycle_id,
                            topic_count,
                            papers_scored,
                        )
                except Exception as e:
                    logger.warning(f"EvolutionDaemon: failed to log cycle node: {e}")

                # 5. Telemetry Ingestion Sweep
                try:
                    logger.info(
                        "EvolutionDaemon: triggering telemetry_ingestion workflow sweep"
                    )

                    def _run_telemetry():
                        try:
                            from agent_utilities.workflows.runner import WorkflowRunner

                            runner = WorkflowRunner()
                            asyncio.run(
                                runner.execute_by_name(
                                    "telemetry_ingestion",
                                    engine=self,  # type: ignore[arg-type]
                                )
                            )
                        except Exception as e:
                            logger.error(f"EvolutionDaemon telemetry sweep failed: {e}")

                    threading.Thread(
                        target=_run_telemetry, daemon=True, name="KG-Telemetry-Worker"
                    ).start()
                except Exception as e:
                    logger.warning(
                        f"EvolutionDaemon: failed to trigger telemetry sweep: {e}"
                    )

                time.sleep(EVOLUTION_INTERVAL)
            except Exception as e:
                logger.error(f"EvolutionDaemon error: {e}")
                time.sleep(300.0)

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

                # Execute the task asynchronously inside this thread (lock is released)
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
                import os
                import re
                import sys

                # To prevent uv from intercepting the subprocess and using the target directory's .venv,
                # we construct the absolute path to the python binary and strip uv environment variables.
                python_bin = os.path.join(sys.prefix, "bin", "python")
                env = os.environ.copy()
                env.pop("UV_PROJECT_ENVIRONMENT", None)
                env.pop("UV_RUN_TARGET", None)
                env["UV_NO_SYNC"] = "1"

                process = await asyncio.create_subprocess_exec(
                    python_bin,
                    "-m",
                    "agent_utilities.knowledge_graph",
                    "--maintain",
                    "--stage-to-queue",
                    job_id,
                    cwd=str(target),
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await process.communicate()

                if process.returncode != 0:
                    raise Exception(f"Ingestion subprocess failed: {stderr.decode()}")

                out_str = stdout.decode()
                nodes_added = 0
                edges_added = 0
                match = re.search(
                    r"Intelligence Graph Updated: (\d+) nodes, (\d+) edges", out_str
                )
                if match:
                    nodes_added = int(match.group(1))
                    edges_added = int(match.group(2))

                self._update_task_status(
                    job_id,
                    "completed",
                    {
                        "nodes_added": nodes_added,
                        "edges_added": edges_added,
                        "target": str(target),
                        "type": "codebase",
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
                if target.is_dir():
                    docs = SimpleDirectoryReader(
                        input_dir=str(target), recursive=True
                    ).load_data()
                else:
                    docs = SimpleDirectoryReader(input_files=[str(target)]).load_data()

                created = []
                skipped = 0
                ingestion_timestamp = datetime.now(UTC).isoformat()
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

                    embedding = embed_model.get_text_embedding(chunk_text)
                    props = {
                        "content": chunk_text,
                        "embedding": embedding,
                        "metadata": json.dumps(doc.metadata),
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

            # Create RELEVANCE_SCORED edge
            self.link_nodes(
                item_id,
                target_id,
                "RELEVANCE_SCORED",
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

        if status in ("completed", "failed") and "completed_at" not in metadata:
            metadata["completed_at"] = datetime.now(UTC).isoformat()

        encoded = _encode_metadata(metadata)
        self.backend.execute(
            "MATCH (t:Task {id: $id}) SET t.status = $status, t.metadata = $meta",
            {"id": job_id, "status": status, "meta": encoded},
        )
        self._checkpoint_db()

    def _checkpoint_db(self) -> None:
        """Force a WAL checkpoint to ensure data persists across server restarts."""
        if not self.backend:
            return
        backend_name = self.backend.__class__.__name__
        if backend_name in (
            "LadybugBackend",
            "Neo4jBackend",
            "FalkorDBBackend",
            "EpistemicGraphBackend",
        ):
            logger.debug(f"WAL checkpoint skipped for non-SQL backend: {backend_name}")
            return
        try:
            # Use native wal_checkpoint if available on the backend
            if hasattr(self.backend, "wal_checkpoint"):
                if self.backend.wal_checkpoint():
                    logger.debug("WAL checkpoint completed (native).")
                    return

            # Fallback to direct PRAGMA if it's a raw DB handle
            self.backend.execute("CHECKPOINT;")
            logger.debug("WAL checkpoint completed (PRAGMA).")
        except Exception as e:
            logger.debug(f"WAL checkpoint not supported or skipped: {e}")

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
