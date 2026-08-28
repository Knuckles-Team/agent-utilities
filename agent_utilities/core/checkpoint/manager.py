#!/usr/bin/python
from __future__ import annotations

"""State Checkpoint Manager Module.

CONCEPT:AU-ORCH.planning.recursion-nesting-depth, CONCEPT:AU-ORCH.execution.execution-budget-caps, CONCEPT:AU-KG.research.research-pipeline-runner

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
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from pydantic import TypeAdapter

from agent_utilities.core.config import setting
from agent_utilities.security.identifiers import validate_sql_identifier

from ...models.knowledge_graph import RegistryNodeType

# pydantic-ai v2 removed the ``pydantic_graph.persistence`` package, and graph
# execution no longer consumes a pluggable persistence object (``Graph.run()``
# dropped the ``persistence=`` parameter). The checkpoint backends below
# implement our OWN minimal persistence interface (``BaseStatePersistence``) and
# are write-only snapshot stores (``load_next``/``load_all`` are no-ops returning
# ``Any | None`` / ``list[Any]``). The live agent checkpointing path uses the
# hook-based capability in ``agent_utilities.capabilities.checkpointing``.

if TYPE_CHECKING:
    from ...knowledge_graph.core.engine import IntelligenceGraphEngine
    from ...knowledge_graph.core.graph_compute import GraphComputeEngine

logger = logging.getLogger(__name__)

StateT = TypeVar("StateT")


class BaseStatePersistence(Generic[StateT]):
    """Minimal checkpoint-persistence interface implemented by the backends below.

    v2 graph execution no longer resumes from persistence, so the snapshot
    writers default to no-ops and ``load_next``/``load_all`` to empty; concrete
    backends (file/postgres/redis) override the writers to record snapshots.
    """

    async def snapshot_node(self, state: StateT, next_node: Any) -> None:
        return None

    async def snapshot_node_if_new(
        self, snapshot_id: str, state: StateT, next_node: Any
    ) -> None:
        return None

    async def snapshot_end(self, state: StateT, end: Any) -> None:
        return None

    async def load_next(self) -> Any | None:
        return None

    async def load_all(self) -> list[Any]:
        return []


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
    """JSON-file snapshot store.

    v2 removed pydantic-graph's ``FileStatePersistence``, so this writes
    snapshots to a JSON array itself. Write-only (graph resume from persistence
    is gone in v2), so ``load_next``/``load_all`` inherit the empty defaults.
    """

    def __init__(self, json_file: str | Path):
        self.path = Path(json_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._run_id = ""

    @asynccontextmanager
    async def record_run(self, snapshot_id: str) -> Any:
        self._run_id = snapshot_id
        yield

    def _append(self, record: dict[str, Any]) -> None:
        existing: list[Any] = []
        if self.path.exists():
            try:
                existing = json.loads(self.path.read_text())
            except (OSError, ValueError):
                existing = []
        existing.append(record)
        self.path.write_text(json.dumps(existing, default=str))

    async def snapshot_node(self, state: StateT, next_node: Any) -> None:
        self._append(
            {
                "run_id": self._run_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "node_id": _node_identifier(next_node),
                "state": _state_json(state),
                "is_end": False,
            }
        )

    async def snapshot_node_if_new(
        self, snapshot_id: str, state: StateT, next_node: Any
    ) -> None:
        await self.snapshot_node(state, next_node)

    async def snapshot_end(self, state: StateT, end: Any) -> None:
        self._append(
            {
                "run_id": self._run_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "output": getattr(end, "data", str(end)),
                "state": _state_json(state),
                "is_end": True,
            }
        )

    def set_graph_types(self, graph: Any) -> None:
        return None

    def should_set_types(self) -> bool:
        return False


class PostgresBackend(BaseStatePersistence[StateT]):
    """PostgreSQL-based state persistence using asyncpg."""

    def __init__(self, dsn: str, table_name: str = "graph_snapshots"):
        self.dsn = dsn
        # ``table_name`` is spliced directly into CREATE TABLE/INDEX/INSERT
        # DDL below (no bound-parameter position exists for a table name) —
        # validate it once here so every later interpolation is already safe.
        self.table_name = validate_sql_identifier(table_name, kind="table")
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
        # Re-validated at the point of use (already validated once in
        # __init__): keeps this DDL site self-evidently safe on its own,
        # independent of the constructor.
        table = validate_sql_identifier(self.table_name, kind="table")
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        run_id TEXT,
                        timestamp TIMESTAMPTZ,
                        snapshot_id TEXT PRIMARY KEY,
                        node_id TEXT,
                        data JSONB,
                        state JSONB,
                        is_end BOOLEAN DEFAULT FALSE
                    );
                    CREATE INDEX IF NOT EXISTS idx_{table}_run_id ON {table}(run_id);
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
        table = validate_sql_identifier(self.table_name, kind="table")
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {table} (run_id, timestamp, snapshot_id, node_id, data, state, is_end)
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
        table = validate_sql_identifier(self.table_name, kind="table")
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {table} (run_id, timestamp, snapshot_id, data, state, is_end)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,  # nosec B608
                self._run_id,
                datetime.now(UTC),
                f"{self._run_id}_end",
                json.dumps(getattr(end, "data", str(end)), default=str),
                _state_json(state),
                True,
            )

    async def load_next(self) -> Any | None:
        return None

    async def load_all(self) -> list[Any]:
        return []


class RedisBackend(BaseStatePersistence[StateT]):
    """Redis-based state persistence using redis-py (asyncio)."""

    def __init__(
        self,
        url: str,
        prefix: str = "graph:",
        *,
        tls_profile: str | None = None,
        tls_profile_ref: str | None = None,
    ):
        from urllib.parse import urlparse

        if urlparse(url).scheme.casefold() != "rediss":
            raise ValueError("Redis checkpoint transport requires rediss://")
        self.url = url
        self.prefix = prefix
        self._redis: Any = None
        self._tls_profile = tls_profile
        self._tls_profile_ref = tls_profile_ref
        self._tls_trust: Any = None
        self._run_id = ""

    async def _get_redis(self):
        try:
            import redis.asyncio as redis
        except ImportError:
            raise ImportError("redis is required for RedisBackend") from None

        if self._redis is None:
            from agent_utilities.core.transport_security import (
                resolve_configured_tls_profile,
            )

            self._tls_trust = resolve_configured_tls_profile(
                "redis",
                profile_name=self._tls_profile,
                profile_ref=self._tls_profile_ref,
            )
            try:
                self._redis = redis.from_url(
                    self.url,
                    decode_responses=True,
                    **self._tls_trust.redis_kwargs(),
                )
            except Exception:
                self._tls_trust.cleanup()
                self._tls_trust = None
                raise
        return self._redis

    async def close(self) -> None:
        """Close the Redis pool and remove runtime TLS material."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
        if self._tls_trust is not None:
            self._tls_trust.cleanup()
            self._tls_trust = None

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

    async def load_next(self) -> Any | None:
        return None

    async def load_all(self) -> list[Any]:
        return []


class KGBackend:
    """Knowledge Graph based state persistence."""

    def __init__(self, engine: IntelligenceGraphEngine | None = None):
        self.engine = engine

    @staticmethod
    def _resolve_checkpoint_plan(state: Any) -> str:
        raw_plan = getattr(state, "plan", "") or ""
        if hasattr(raw_plan, "model_dump_json"):
            return raw_plan.model_dump_json()
        if isinstance(raw_plan, str):
            return raw_plan
        return json.dumps(raw_plan, default=str)

    @staticmethod
    def _resolve_checkpoint_specialist_results(state: Any) -> dict[str, str]:
        specialist_results: dict[str, str] = {}
        raw_results = getattr(state, "specialist_results", None)
        if isinstance(raw_results, dict):
            specialist_results = {k: str(v)[:500] for k, v in raw_results.items()}
        elif isinstance(raw_results, list):
            for i, r in enumerate(raw_results):
                specialist_results[f"result_{i}"] = str(r)[:500]
        return specialist_results

    @staticmethod
    def _resolve_checkpoint_state_data(state: Any) -> dict[str, Any]:
        state_data: dict[str, Any] = {}
        for attr in ("routed_domain", "routed_specialist", "active_topology"):
            val = getattr(state, attr, None)
            if val is not None:
                try:
                    state_data[attr] = str(val)
                except Exception:  # noqa: BLE001 — opaque optional state is best-effort
                    pass
        return state_data

    @staticmethod
    def _resolve_checkpoint_topo_id(state: Any) -> str:
        active_topo = getattr(state, "active_topology", None)
        if active_topo:
            return str(getattr(active_topo, "id", str(active_topo)))
        return ""

    @staticmethod
    def _resolve_checkpoint_graph_fields(state: Any) -> dict[str, Any]:
        return {
            "graph_topology_digest": str(
                getattr(state, "graph_topology_digest", "") or ""
            ),
            "graph_version_digest": str(
                getattr(state, "graph_version_digest", "") or ""
            ),
            "graph_runtime_version": str(
                getattr(state, "graph_runtime_version", "") or ""
            ),
            "graph_node_sequence": list(
                getattr(state, "graph_node_sequence", []) or []
            ),
            "graph_transition_sequence": json.dumps(
                list(getattr(state, "graph_transition_sequence", []) or []),
                sort_keys=True,
                separators=(",", ":"),
            ),
        }

    def _build_checkpoint_node_data(
        self,
        state: Any,
        checkpoint_id: str,
        session_id: str,
        status: str,
        timestamp: str,
    ) -> dict[str, Any]:
        query = getattr(state, "query", "") or ""
        plan = self._resolve_checkpoint_plan(state)
        node_history = list(getattr(state, "node_history", []) or [])
        current_node = str(node_history[-1]) if node_history else ""
        specialist_results = self._resolve_checkpoint_specialist_results(state)
        total_tokens = 0
        usage = getattr(state, "usage", None)
        if usage:
            total_tokens = getattr(usage, "total_tokens", 0) or 0
        state_data = self._resolve_checkpoint_state_data(state)
        topo_id = self._resolve_checkpoint_topo_id(state)
        graph_fields = self._resolve_checkpoint_graph_fields(state)

        return {
            "id": checkpoint_id,
            "name": f"Checkpoint: {query[:50]}..."
            if len(query) > 50
            else f"Checkpoint: {query}",
            "node_type": RegistryNodeType.SESSION_CHECKPOINT.value,
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
            **graph_fields,
            "timestamp": timestamp,
        }

    @staticmethod
    def _write_checkpoint_node(
        engine: IntelligenceGraphEngine, checkpoint_id: str, node_data: dict[str, Any]
    ) -> str | None:
        if hasattr(engine, "backend_type") and engine.backend_type == "rust":
            try:
                cast("GraphComputeEngine", engine).add_node(
                    checkpoint_id, properties=node_data
                )
            except Exception as exc:
                logger.warning("Failed to checkpoint to rust backend: %s", exc)
                return None
            return checkpoint_id

        if getattr(engine, "backend", None):
            try:
                engine._upsert_node("SessionCheckpoint", checkpoint_id, node_data)
            except Exception as exc:
                logger.warning("Failed to checkpoint to backend: %s", exc)
                return None
            return checkpoint_id

        if hasattr(engine, "graph") and hasattr(engine.graph, "add_node"):
            try:
                engine.graph.add_node(checkpoint_id, **node_data)
            except Exception as exc:
                logger.warning("Failed to checkpoint to graph mirror: %s", exc)
                return None
            return checkpoint_id
        return None

    def checkpoint(
        self,
        state: Any,
        session_id: str | None = None,
        status: str = "active",
    ) -> str | None:
        """Persist a snapshot and return its identifier only after a confirmed write."""

        if self.engine is None:
            return None
        engine = self.engine

        if session_id is None:
            session_id = f"sess:{uuid.uuid4().hex}"

        checkpoint_id = f"ckpt:{session_id}:{time.time_ns()}"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        node_data = self._build_checkpoint_node_data(
            state, checkpoint_id, session_id, status, timestamp
        )
        return self._write_checkpoint_node(engine, checkpoint_id, node_data)

    @staticmethod
    def _restore_from_backend(
        engine: IntelligenceGraphEngine, session_id: str
    ) -> dict[str, Any] | None:
        if not engine.backend:
            return None
        try:
            results = engine.backend.execute(
                "MATCH (c:SessionCheckpoint) WHERE c.session_id = $sid RETURN c ORDER BY c.timestamp DESC LIMIT 1",
                {"sid": session_id},
            )
        except Exception as e:  # noqa: BLE001 — one candidate lookup path; the graph/rust-engine mirror lookup is tried next before returning None
            logger.debug("Backend checkpoint restore failed: %s", e)
            return None
        if not results:
            return None
        checkpoint = results[0]
        if isinstance(checkpoint, dict) and "c" in checkpoint:
            checkpoint = checkpoint["c"]
        return checkpoint if isinstance(checkpoint, dict) else None

    @staticmethod
    def _restore_from_graph_mirror(
        engine: IntelligenceGraphEngine, session_id: str
    ) -> dict[str, Any] | None:
        checkpoint_id = f"SessionCheckpoint_{session_id}"
        if hasattr(engine, "backend_type") and engine.backend_type == "rust":
            rust_engine = cast("GraphComputeEngine", engine)
            if rust_engine.has_node(checkpoint_id):
                return cast("dict[str, Any]", rust_engine[checkpoint_id])
            return None
        if hasattr(engine, "graph"):
            if checkpoint_id in engine.graph:
                return dict(engine.graph.nodes[checkpoint_id])
            for _nid, data in engine.graph.nodes(data=True):
                if (
                    data.get("node_type") == RegistryNodeType.SESSION_CHECKPOINT.value
                    and data.get("session_id") == session_id
                ):
                    return dict(data)
        return None

    @staticmethod
    def _decode_checkpoint_json_field(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except ValueError:
            return {}

    def _decode_checkpoint_dict(
        self, checkpoint: dict[str, Any], session_id: str
    ) -> dict[str, Any] | None:
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
            state_dict["specialist_results"] = self._decode_checkpoint_json_field(
                checkpoint.get("specialist_results", "{}")
            )
            state_dict["state_data"] = self._decode_checkpoint_json_field(
                checkpoint.get("state_data", "{}")
            )
            return state_dict
        except Exception as exc:  # noqa: BLE001 — a malformed prior checkpoint degrades to "no checkpoint" (resume starts fresh) rather than crashing resumption
            # "no checkpoint" and "checkpoint unreadable" are different
            # operator questions — log which one this is.
            logger.debug("Checkpoint state could not be decoded: %s", exc)
            return None

    def restore(self, session_id: str) -> dict[str, Any] | None:
        if not self.engine:
            return None
        engine = self.engine

        checkpoint = self._restore_from_backend(engine, session_id)
        if checkpoint is None:
            checkpoint = self._restore_from_graph_mirror(engine, session_id)
        if checkpoint is None:
            return None
        return self._decode_checkpoint_dict(checkpoint, session_id)

    def list_sessions(
        self, status: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        if self.engine and self.engine.backend:
            try:
                where = "WHERE c.node_type = 'session_checkpoint'"
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
            except Exception as exc:
                # Best-effort listing — an empty result is preferable to a
                # crashed UI, but log the real cause so a broken backend
                # doesn't masquerade as "no sessions".
                logger.warning("Failed to list checkpoint sessions: %s", exc)
        return sessions

    def mark_completed(self, session_id: str, success: bool = True) -> None:
        status = "completed" if success else "failed"
        if self.engine and self.engine.backend:
            try:
                self.engine.backend.execute(
                    "MATCH (c:SessionCheckpoint) WHERE c.session_id = $sid SET c.status = $status",
                    {"sid": session_id, "status": status},
                )
            except Exception as exc:
                # Best-effort (matches checkpoint()/restore() above in this
                # class) — callers treat this as fire-and-forget and must not
                # raise, but a silently-stuck "active" session is exactly the
                # kind of bug this log line exists to catch.
                logger.warning("Failed to mark checkpoint session completed: %s", exc)


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

    @staticmethod
    def _create_file_backend(run_id: str | None, kwargs: dict[str, Any]) -> Any:
        path = kwargs.get("path", "agent_data/graph_state")
        filename = kwargs.get("filename", f"{run_id or 'default'}.json")
        return FileBackend(json_file=Path(path) / filename)

    @staticmethod
    def _create_postgres_backend(kwargs: dict[str, Any]) -> Any:
        dsn = kwargs.get("dsn") or setting("POSTGRES_DSN")
        if dsn:
            return PostgresBackend(dsn=dsn)
        return None

    @staticmethod
    def _resolve_redis_connection_profile(
        connection_ref: Any, profile_name: Any, profile_ref: Any
    ) -> tuple[str, Any, Any]:
        """Resolve a stored ``connection_profile_ref`` secret into
        ``(url, tls_profile, tls_profile_ref)``. Fail-closed: a decoded JSON
        object carrying any key outside the allowed set is rejected outright
        rather than silently ignored, and a non-dict payload is treated as a
        bare URL string (unchanged from the caller's own tls args)."""
        from agent_utilities.security.secrets_client import create_secrets_client

        raw = create_secrets_client().resolve_ref(str(connection_ref))
        rendered = (
            raw.decode("utf-8") if isinstance(raw, bytes) else str(raw or "")
        ).strip()
        try:
            profile = json.loads(rendered)
        except (TypeError, ValueError):
            profile = None
        if not isinstance(profile, dict):
            return rendered, profile_name, profile_ref
        if set(profile).difference({"url", "tls_profile", "tls_profile_ref"}):
            raise ValueError("Redis connection profile is invalid")
        url = str(profile.get("url") or "").strip()
        profile_name = profile.get("tls_profile") or profile_name
        profile_ref = profile.get("tls_profile_ref") or profile_ref
        return url, profile_name, profile_ref

    @staticmethod
    def _create_redis_backend(kwargs: dict[str, Any]) -> Any:
        from agent_utilities.core.config import config

        profile_name = kwargs.get("tls_profile")
        profile_ref = kwargs.get("tls_profile_ref")
        url = kwargs.get("url")
        connection_ref = (
            kwargs.get("connection_profile_ref") or config.redis_connection_profile_ref
        )
        if connection_ref:
            url, profile_name, profile_ref = (
                CheckpointManager._resolve_redis_connection_profile(
                    connection_ref, profile_name, profile_ref
                )
            )
        if not url:
            return None
        return RedisBackend(
            url=str(url),
            tls_profile=(str(profile_name) if profile_name else None),
            tls_profile_ref=(str(profile_ref) if profile_ref else None),
        )

    @staticmethod
    def _create_kg_backend(kwargs: dict[str, Any]) -> Any:
        return KGBackend(engine=kwargs.get("engine"))

    @classmethod
    def create(
        cls, persistence_type: str = "file", run_id: str | None = None, **kwargs: Any
    ) -> CheckpointManager:
        """Factory to return a CheckpointManager with configured backend."""
        ptype = persistence_type.lower()
        backend: Any = None

        if ptype == "file":
            backend = cls._create_file_backend(run_id, kwargs)
        elif ptype == "postgres":
            backend = cls._create_postgres_backend(kwargs)
        elif ptype == "redis":
            backend = cls._create_redis_backend(kwargs)
        elif ptype == "kg":
            backend = cls._create_kg_backend(kwargs)

        return cls(backend=backend)
