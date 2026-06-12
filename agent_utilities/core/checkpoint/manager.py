#!/usr/bin/python
from __future__ import annotations

"""State Checkpoint Manager Module.

CONCEPT:ORCH-1.1, CONCEPT:ORCH-1.3, CONCEPT:KG-2.6

Provides a unified interface for persisting graph execution state.
Supports multiple backends (File, Postgres, Redis, and KG).
"""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

from pydantic import TypeAdapter

from agent_utilities.core.config import setting

from ...models.knowledge_graph import RegistryNodeType

try:
    from pydantic_graph.persistence import (
        BaseStatePersistence,
        EndSnapshot,
        NodeSnapshot,
    )
except ImportError:
    BaseStatePersistence = Any  # type: ignore
    NodeSnapshot = Any  # type: ignore
    EndSnapshot = Any  # type: ignore

try:
    from pydantic_graph.persistence.file import FileStatePersistence
except ImportError:
    FileStatePersistence = None  # type: ignore

if TYPE_CHECKING:
    from ...knowledge_graph.core.engine import IntelligenceGraphEngine
    from ...knowledge_graph.core.graph_compute import GraphComputeEngine

logger = logging.getLogger(__name__)

StateT = TypeVar("StateT")


# The installed pydantic-graph BaseStatePersistence interface (the backends below implement it):
#   snapshot_node(state, next_node) / snapshot_node_if_new(snapshot_id, state, next_node) /
#   snapshot_end(state, end) / load_next() / load_all() / record_run(snapshot_id)
# (Earlier pydantic-graph passed whole NodeSnapshot/EndSnapshot objects and a run_id; this layer was
# written for that old API. Migrated to the current signatures below.)


def _node_identifier(next_node: Any) -> str:
    """Best-effort stable id for a graph node across pydantic-graph versions."""
    return str(getattr(next_node, "id", None) or type(next_node).__name__)


def _state_json(state: Any) -> str:
    try:
        return TypeAdapter(type(state)).dump_json(state).decode()
    except Exception:  # noqa: BLE001 - fall back to a best-effort repr for opaque states
        return json.dumps(str(state))


class FileBackend(BaseStatePersistence[StateT]):
    """JSON-based file persistence — forwards to pydantic-graph ``FileStatePersistence``."""

    def __init__(self, json_file: str | Path):
        if FileStatePersistence is None:
            raise ImportError("pydantic-graph file persistence is not available.")
        self.path = Path(json_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._internal = FileStatePersistence(json_file=self.path)

    async def snapshot_node(self, state: StateT, next_node: Any) -> None:
        await self._internal.snapshot_node(state, next_node)

    async def snapshot_node_if_new(
        self, snapshot_id: str, state: StateT, next_node: Any
    ) -> None:
        await self._internal.snapshot_node_if_new(snapshot_id, state, next_node)

    async def snapshot_end(self, state: StateT, end: Any) -> None:
        await self._internal.snapshot_end(state, end)

    def record_run(self, snapshot_id: str) -> Any:
        return self._internal.record_run(snapshot_id)

    async def load_next(self) -> NodeSnapshot[StateT] | None:
        return await self._internal.load_next()

    async def load_all(self) -> list[Any]:
        return await self._internal.load_all()

    def set_graph_types(self, graph: Any) -> None:
        self._internal.set_graph_types(graph)

    def should_set_types(self) -> bool:
        return self._internal.should_set_types()


class PostgresBackend(BaseStatePersistence[StateT]):
    """PostgreSQL-based state persistence using asyncpg."""

    def __init__(self, dsn: str, table_name: str = "graph_snapshots"):
        self.dsn = dsn
        self.table_name = table_name
        self._pool = None
        self._run_id = ""

    async def _get_pool(self):
        try:
            import asyncpg
        except ImportError:
            raise ImportError("asyncpg is required for PostgresBackend") from None

        if self._pool is None:
            self._pool = await asyncpg.create_pool(self.dsn)
        if self._pool is None:
            raise RuntimeError("Failed to create postgres pool")
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        run_id TEXT,
                        timestamp TIMESTAMPTZ,
                        snapshot_id TEXT PRIMARY KEY,
                        node_id TEXT,
                        data JSONB,
                        state JSONB,
                        is_end BOOLEAN DEFAULT FALSE
                    );
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_run_id ON {self.table_name}(run_id);
                """
            )
        return self._pool

    @asynccontextmanager
    async def record_run(self, snapshot_id: str) -> Any:
        self._run_id = snapshot_id
        yield

    async def _insert_node(
        self, snapshot_id: str, state: StateT, next_node: Any
    ) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.table_name} (run_id, timestamp, snapshot_id, node_id, data, state, is_end)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (snapshot_id) DO NOTHING
                """,  # nosec B608
                self._run_id,
                datetime.now(UTC),
                snapshot_id,
                _node_identifier(next_node),
                json.dumps(
                    asdict(next_node)
                    if hasattr(next_node, "__dataclass_fields__")
                    else {}
                ),
                _state_json(state),
                False,
            )

    async def snapshot_node(self, state: StateT, next_node: Any) -> None:
        await self._insert_node(f"{self._run_id}:{time.time()}", state, next_node)

    async def snapshot_node_if_new(
        self, snapshot_id: str, state: StateT, next_node: Any
    ) -> None:
        await self._insert_node(snapshot_id, state, next_node)

    async def snapshot_end(self, state: StateT, end: Any) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.table_name} (run_id, timestamp, snapshot_id, data, state, is_end)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,  # nosec B608
                self._run_id,
                datetime.now(UTC),
                f"{self._run_id}_end",
                json.dumps(getattr(end, "data", str(end)), default=str),
                _state_json(state),
                True,
            )

    async def load_next(self) -> NodeSnapshot[StateT] | None:
        return None

    async def load_all(self) -> list[Any]:
        return []


class RedisBackend(BaseStatePersistence[StateT]):
    """Redis-based state persistence using redis-py (asyncio)."""

    def __init__(self, url: str, prefix: str = "graph:"):
        self.url = url
        self.prefix = prefix
        self._redis: Any = None
        self._run_id = ""

    async def _get_redis(self):
        try:
            import redis.asyncio as redis
        except ImportError:
            raise ImportError("redis is required for RedisBackend") from None

        if self._redis is None:
            self._redis = redis.from_url(self.url, decode_responses=True)
        return self._redis

    @asynccontextmanager
    async def record_run(self, snapshot_id: str) -> Any:
        self._run_id = snapshot_id
        yield

    async def _write_node(
        self, member: str, state: StateT, next_node: Any, *, if_new: bool
    ) -> None:
        r = await self._get_redis()
        key = f"{self.prefix}{self._run_id}:snapshots"
        data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "node_id": _node_identifier(next_node),
            "state": _state_json(state),
            "is_end": False,
        }
        if if_new:
            await r.hsetnx(key, member, json.dumps(data))
        else:
            await r.hset(key, member, json.dumps(data))

    async def snapshot_node(self, state: StateT, next_node: Any) -> None:
        await self._write_node(str(time.time()), state, next_node, if_new=False)

    async def snapshot_node_if_new(
        self, snapshot_id: str, state: StateT, next_node: Any
    ) -> None:
        await self._write_node(snapshot_id, state, next_node, if_new=True)

    async def snapshot_end(self, state: StateT, end: Any) -> None:
        r = await self._get_redis()
        key = f"{self.prefix}{self._run_id}:snapshots"
        data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "output": getattr(end, "data", str(end)),
            "state": _state_json(state),
            "is_end": True,
        }
        await r.hset(key, "end", json.dumps(data, default=str))

    async def load_next(self) -> NodeSnapshot[StateT] | None:
        return None

    async def load_all(self) -> list[Any]:
        return []


class KGBackend:
    """Knowledge Graph based state persistence."""

    def __init__(self, engine: IntelligenceGraphEngine | None = None):
        self.engine = engine

    def checkpoint(
        self,
        state: Any,
        session_id: str | None = None,
        status: str = "active",
    ) -> str:
        if session_id is None:
            session_id = f"sess:{uuid.uuid4().hex[:12]}"

        checkpoint_id = f"ckpt:{session_id}:{int(time.time())}"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        query = getattr(state, "query", "") or ""
        plan = getattr(state, "plan", "") or ""
        node_history = list(getattr(state, "node_history", []) or [])
        current_node = str(node_history[-1]) if node_history else ""

        specialist_results: dict[str, str] = {}
        raw_results = getattr(state, "specialist_results", None)
        if isinstance(raw_results, dict):
            specialist_results = {k: str(v)[:500] for k, v in raw_results.items()}
        elif isinstance(raw_results, list):
            for i, r in enumerate(raw_results):
                specialist_results[f"result_{i}"] = str(r)[:500]

        total_tokens = 0
        usage = getattr(state, "usage", None)
        if usage:
            total_tokens = getattr(usage, "total_tokens", 0) or 0

        state_data: dict[str, Any] = {}
        for attr in ("routed_domain", "routed_specialist", "active_topology"):
            val = getattr(state, attr, None)
            if val is not None:
                try:
                    state_data[attr] = str(val)
                except Exception:
                    pass

        topo_id = ""
        active_topo = getattr(state, "active_topology", None)
        if active_topo:
            topo_id = getattr(active_topo, "id", str(active_topo))

        node_data = {
            "id": checkpoint_id,
            "name": f"Checkpoint: {query[:50]}..."
            if len(query) > 50
            else f"Checkpoint: {query}",
            "type": RegistryNodeType.SESSION_CHECKPOINT.value,
            "session_id": session_id,
            "query": query[:1000],
            "plan": plan[:2000],
            "specialist_results": json.dumps(specialist_results),
            "node_history": node_history,
            "current_node": current_node,
            "total_usage_tokens": total_tokens,
            "state_data": json.dumps(state_data),
            "status": status,
            "topology_template_id": topo_id,
            "timestamp": timestamp,
        }

        if self.engine:
            if (
                hasattr(self.engine, "backend_type")
                and self.engine.backend_type == "rust"
            ):
                cast("GraphComputeEngine", self.engine).add_node(
                    checkpoint_id, properties=node_data
                )
            else:
                if getattr(self.engine, "backend", None):
                    try:
                        self.engine._upsert_node(
                            "SessionCheckpoint", checkpoint_id, node_data
                        )
                    except Exception as e:
                        logger.warning("Failed to checkpoint to backend: %s", e)

                if hasattr(self.engine, "graph") and hasattr(
                    self.engine.graph, "add_node"
                ):
                    self.engine.graph.add_node(checkpoint_id, **node_data)

        return checkpoint_id

    def restore(self, session_id: str) -> dict[str, Any] | None:
        if not self.engine:
            return None

        checkpoint = None

        if self.engine.backend:
            try:
                results = self.engine.backend.execute(
                    "MATCH (c:SessionCheckpoint) WHERE c.session_id = $sid RETURN c ORDER BY c.timestamp DESC LIMIT 1",
                    {"sid": session_id},
                )
                if results:
                    checkpoint = results[0]
                    if isinstance(checkpoint, dict) and "c" in checkpoint:
                        checkpoint = checkpoint["c"]

                    if not isinstance(checkpoint, dict):
                        checkpoint = None
            except Exception as e:
                logger.debug("Backend checkpoint restore failed: %s", e)

        if checkpoint is None and self.engine:
            checkpoint_id = f"SessionCheckpoint_{session_id}"
            if (
                hasattr(self.engine, "backend_type")
                and self.engine.backend_type == "rust"
            ):
                rust_engine = cast("GraphComputeEngine", self.engine)
                if rust_engine.has_node(checkpoint_id):
                    checkpoint = rust_engine[checkpoint_id]
            elif hasattr(self.engine, "graph"):
                if checkpoint_id in self.engine.graph:
                    checkpoint = dict(self.engine.graph.nodes[checkpoint_id])
                else:
                    for nid, data in self.engine.graph.nodes(data=True):
                        if (
                            data.get("type")
                            == RegistryNodeType.SESSION_CHECKPOINT.value
                            and data.get("session_id") == session_id
                        ):
                            checkpoint = dict(data)
                            break

        if checkpoint is None:
            return None

        try:
            state_dict: dict[str, Any] = {
                "session_id": checkpoint.get("session_id", session_id),
                "query": checkpoint.get("query", ""),
                "plan": checkpoint.get("plan", ""),
                "node_history": checkpoint.get("node_history", []),
                "current_node": checkpoint.get("current_node", ""),
                "total_usage_tokens": checkpoint.get("total_usage_tokens", 0),
                "status": checkpoint.get("status", "active"),
                "topology_template_id": checkpoint.get("topology_template_id", ""),
            }

            sr = checkpoint.get("specialist_results", "{}")
            if isinstance(sr, str):
                try:
                    state_dict["specialist_results"] = json.loads(sr)
                except ValueError:
                    state_dict["specialist_results"] = {}
            else:
                state_dict["specialist_results"] = sr

            sd = checkpoint.get("state_data", "{}")
            if isinstance(sd, str):
                try:
                    state_dict["state_data"] = json.loads(sd)
                except ValueError:
                    state_dict["state_data"] = {}
            else:
                state_dict["state_data"] = sd

            return state_dict
        except Exception:
            return None

    def list_sessions(
        self, status: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        if self.engine and self.engine.backend:
            try:
                where = "WHERE c.type = 'session_checkpoint'"
                if status:
                    where += " AND c.status = $status"
                results = self.engine.backend.execute(
                    f"MATCH (c:SessionCheckpoint) {where} RETURN c.session_id AS sid, c.query AS query, c.status AS status, c.timestamp AS ts ORDER BY c.timestamp DESC LIMIT $limit",
                    {"status": status or "", "limit": limit},
                )
                for r in results:
                    sessions.append(
                        {
                            "session_id": r.get("sid", ""),
                            "query": (r.get("query", "") or "")[:100],
                            "status": r.get("status", ""),
                            "timestamp": r.get("ts", ""),
                        }
                    )
            except Exception:
                pass
        return sessions

    def mark_completed(self, session_id: str, success: bool = True) -> None:
        status = "completed" if success else "failed"
        if self.engine and self.engine.backend:
            try:
                self.engine.backend.execute(
                    "MATCH (c:SessionCheckpoint) WHERE c.session_id = $sid SET c.status = $status",
                    {"sid": session_id, "status": status},
                )
            except Exception:
                pass


class CheckpointManager:
    """Consolidated manager for checkpoints and state."""

    def __init__(self, backend: Any = None):
        self.backend = backend

    def save(self, state: Any, **kwargs: Any) -> Any:
        if hasattr(self.backend, "checkpoint"):
            return self.backend.checkpoint(state, **kwargs)
        elif hasattr(self.backend, "snapshot_node_if_new"):
            # Minimal mapping for Pydantic Graph BaseStatePersistence
            return None
        return None

    def restore(self, checkpoint_id: str) -> Any:
        if hasattr(self.backend, "restore"):
            return self.backend.restore(checkpoint_id)
        elif hasattr(self.backend, "load_next"):
            return None
        return None

    @classmethod
    def create(
        cls, persistence_type: str = "file", run_id: str | None = None, **kwargs: Any
    ) -> CheckpointManager:
        """Factory to return a CheckpointManager with configured backend."""
        ptype = persistence_type.lower()
        backend: Any = None

        if ptype == "file":
            path = kwargs.get("path", "agent_data/graph_state")
            filename = kwargs.get("filename", f"{run_id or 'default'}.json")
            backend = FileBackend(json_file=Path(path) / filename)
        elif ptype == "postgres":
            dsn = kwargs.get("dsn") or setting("POSTGRES_DSN")
            if dsn:
                backend = PostgresBackend(dsn=dsn)
        elif ptype == "redis":
            url = kwargs.get("url") or setting("REDIS_URL")
            if url:
                backend = RedisBackend(url=url)
        elif ptype == "kg":
            engine = kwargs.get("engine")
            backend = KGBackend(engine=engine)

        return cls(backend=backend)
