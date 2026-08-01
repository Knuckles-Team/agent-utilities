"""CONCEPT:AU-KG.query.object-graph-mapper"""

from unittest.mock import MagicMock

import pytest
from pydantic_ai import RunContext

from agent_utilities.models import AgentDeps
from agent_utilities.tools.knowledge_tools import (
    create_client,
    create_user,
    log_cron_execution,
    log_heartbeat,
    save_chat_message,
)


class DummyBackend:
    def __init__(self):
        self.queries = []

    def execute(self, query: str, props: dict | None = None):
        self.queries.append({"query": query, "props": props})
        return []


@pytest.mark.asyncio
async def test_log_heartbeat():
    # ``knowledge_tools`` writes via ``engine.add_node``/``engine.link_nodes``
    # directly (not ``engine.backend.execute(...)``); ``backend`` is only
    # truthiness-checked (``if engine.backend:``) to gate the write path.
    backend = DummyBackend()
    engine = MagicMock()
    engine.backend = backend
    deps = AgentDeps(knowledge_engine=engine)
    ctx = MagicMock(spec=RunContext)
    ctx.deps = deps

    res = await log_heartbeat(ctx, "test_agent", "OK")
    assert "Heartbeat logged" in res
    assert engine.add_node.call_count == 2
    assert engine.add_node.call_args_list[0].args[1] == "Heartbeat"
    assert engine.link_nodes.call_count == 1


@pytest.mark.asyncio
async def test_create_client():
    backend = DummyBackend()
    engine = MagicMock()
    engine.backend = backend
    deps = AgentDeps(knowledge_engine=engine)
    ctx = MagicMock(spec=RunContext)
    ctx.deps = deps

    res = await create_client(ctx, "TestClient")
    assert "Client created" in res
    assert engine.add_node.call_count == 1


@pytest.mark.asyncio
async def test_create_user():
    backend = DummyBackend()
    engine = MagicMock()
    engine.backend = backend
    deps = AgentDeps(knowledge_engine=engine)
    ctx = MagicMock(spec=RunContext)
    ctx.deps = deps

    res = await create_user(ctx, "TestUser", "admin", "client_123")
    assert "User created" in res
    assert engine.add_node.call_count == 1
    assert engine.link_nodes.call_count == 1


@pytest.mark.asyncio
async def test_save_chat_message():
    backend = DummyBackend()
    engine = MagicMock()
    engine.backend = backend
    deps = AgentDeps(knowledge_engine=engine)
    ctx = MagicMock(spec=RunContext)
    ctx.deps = deps

    res = await save_chat_message(ctx, "thread_123", "user", "hello")
    assert "Message saved" in res
    assert engine.add_node.call_count == 2
    assert engine.link_nodes.call_count == 1


@pytest.mark.asyncio
async def test_log_cron_execution():
    backend = DummyBackend()
    engine = MagicMock()
    engine.backend = backend
    deps = AgentDeps(knowledge_engine=engine)
    ctx = MagicMock(spec=RunContext)
    ctx.deps = deps

    res = await log_cron_execution(ctx, "job_123", "SUCCESS", "done")
    assert "Cron execution logged" in res
    assert engine.add_node.call_count == 2
    assert engine.link_nodes.call_count == 1
