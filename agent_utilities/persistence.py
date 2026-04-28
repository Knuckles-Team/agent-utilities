#!/usr/bin/python
"""State Persistence Module.

This module provides various backends for persisting graph execution state,
including file-based JSON storage, PostgreSQL via asyncpg, and Redis. It ensures
that long-running agentic workflows can be checkpointed and resumed across
process restarts.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

from pydantic import TypeAdapter

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

logger = logging.getLogger(__name__)

StateT = TypeVar("StateT")


class EnhancedFileStatePersistence(BaseStatePersistence[StateT]):
    """JSON-based file persistence with automatic directory management."""

    def __init__(self, json_file: str | Path):
        if FileStatePersistence is None:
            raise ImportError("pydantic-graph file persistence is not available.")
        self.path = Path(json_file)
        self._internal = FileStatePersistence(json_file=self.path)

    async def snapshot_node_if_new(self, snapshot: NodeSnapshot[StateT]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return await self._internal.snapshot_node_if_new(snapshot)

    async def snapshot_end(self, snapshot: EndSnapshot[StateT]) -> None:
        return await self._internal.snapshot_end(snapshot)

    async def load_next(self, run_id: str) -> NodeSnapshot[StateT] | None:
        return await self._internal.load_next(run_id)

    async def load_all(
        self, run_id: str
    ) -> list[NodeSnapshot[StateT] | EndSnapshot[StateT]]:
        return await self._internal.load_all(run_id)


class PostgresStatePersistence(BaseStatePersistence[StateT]):
    """PostgreSQL-based state persistence using asyncpg."""

    def __init__(self, dsn: str, table_name: str = "graph_snapshots"):
        self.dsn = dsn
        self.table_name = table_name
        self._pool = None

    async def _get_pool(self):
        try:
            import asyncpg
        except ImportError:
            raise ImportError(
                "asyncpg is required for PostgresStatePersistence"
            ) from None

        if self._pool is None:
            self._pool = await asyncpg.create_pool(self.dsn)
        if self._pool is None:
            raise RuntimeError("Failed to create postgres pool")
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
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
                """)
        return self._pool

    async def snapshot_node_if_new(self, snapshot: NodeSnapshot[StateT]) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.table_name} (run_id, timestamp, snapshot_id, node_id, data, state, is_end)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (snapshot_id) DO NOTHING
                """,  # nosec B608
                snapshot.run_id,
                snapshot.timestamp,
                getattr(
                    snapshot,
                    "snapshot_id",
                    f"{snapshot.run_id}_{snapshot.timestamp.timestamp()}",
                ),
                (
                    snapshot.node.node_id
                    if hasattr(snapshot.node, "node_id")
                    else str(snapshot.node)
                ),
                json.dumps(
                    asdict(snapshot.node)
                    if hasattr(snapshot.node, "__dataclass_fields__")
                    else {}
                ),
                TypeAdapter(type(snapshot.state)).dump_json(snapshot.state).decode(),
                False,
            )

    async def snapshot_end(self, snapshot: EndSnapshot[StateT]) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.table_name} (run_id, timestamp, snapshot_id, data, state, is_end)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,  # nosec B608
                snapshot.run_id,
                snapshot.timestamp,
                f"{snapshot.run_id}_end",
                json.dumps(snapshot.output),
                TypeAdapter(type(snapshot.state)).dump_json(snapshot.state).decode(),
                True,
            )

    async def load_next(self, run_id: str) -> NodeSnapshot[StateT] | None:
        return None

    async def load_all(
        self, run_id: str
    ) -> list[NodeSnapshot[StateT] | EndSnapshot[StateT]]:
        return []


class RedisStatePersistence(BaseStatePersistence[StateT]):
    """Redis-based state persistence using redis-py (asyncio)."""

    def __init__(self, url: str, prefix: str = "graph:"):
        self.url = url
        self.prefix = prefix
        self._redis = None

    async def _get_redis(self):
        try:
            import redis.asyncio as redis
        except ImportError:
            raise ImportError("redis is required for RedisStatePersistence") from None

        if self._redis is None:
            self._redis = redis.from_url(self.url, decode_responses=True)
        return self._redis

    async def snapshot_node_if_new(self, snapshot: NodeSnapshot[StateT]) -> None:
        r = await self._get_redis()
        key = f"{self.prefix}{snapshot.run_id}:snapshots"
        member = getattr(snapshot, "snapshot_id", f"{snapshot.timestamp.timestamp()}")

        data = {
            "timestamp": snapshot.timestamp.isoformat(),
            "node_id": (
                snapshot.node.node_id
                if hasattr(snapshot.node, "node_id")
                else str(snapshot.node)
            ),
            "state": TypeAdapter(type(snapshot.state))
            .dump_json(snapshot.state)
            .decode(),
            "is_end": False,
        }
        await r.hsetnx(key, member, json.dumps(data))

    async def snapshot_end(self, snapshot: EndSnapshot[StateT]) -> None:
        r = await self._get_redis()
        key = f"{self.prefix}{snapshot.run_id}:snapshots"
        data = {
            "timestamp": snapshot.timestamp.isoformat(),
            "output": snapshot.output,
            "state": TypeAdapter(type(snapshot.state))
            .dump_json(snapshot.state)
            .decode(),
            "is_end": True,
        }
        await r.hset(key, "end", json.dumps(data))

    async def load_next(self, run_id: str) -> NodeSnapshot[StateT] | None:
        return None

    async def load_all(
        self, run_id: str
    ) -> list[NodeSnapshot[StateT] | EndSnapshot[StateT]]:
        return []


def persistence_factory(
    persistence_type: str = "file", run_id: str | None = None, **kwargs
) -> BaseStatePersistence | None:
    """Factory to return a pydantic-graph persistence backend."""
    ptype = persistence_type.lower()

    if ptype == "file":
        path = kwargs.get("path", "agent_data/graph_state")
        filename = kwargs.get("filename", f"{run_id or 'default'}.json")
        return EnhancedFileStatePersistence(json_file=Path(path) / filename)

    elif ptype == "postgres":
        dsn = kwargs.get("dsn") or os.getenv("POSTGRES_DSN")
        if not dsn:
            return None
        return PostgresStatePersistence(dsn=dsn)

    elif ptype == "redis":
        url = kwargs.get("url") or os.getenv("REDIS_URL")
        if not url:
            return None
        return RedisStatePersistence(url=url)

    return None
