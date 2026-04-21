import pytest
from unittest.mock import MagicMock

from pydantic_ai import RunContext
from agent_utilities.tools.knowledge_tools import (
    log_heartbeat,
    create_client,
    create_user,
    save_preference,
    save_chat_message,
    log_cron_execution
)
from agent_utilities.models import AgentDeps

class DummyBackend:
    def __init__(self):
        self.queries = []

    def execute(self, query: str, props: dict | None = None):
        self.queries.append({"query": query, "props": props})
        return []

@pytest.mark.asyncio
async def test_log_heartbeat():
    backend = DummyBackend()
    engine = MagicMock()
    engine.backend = backend
    deps = AgentDeps(knowledge_engine=engine)
    ctx = MagicMock(spec=RunContext)
    ctx.deps = deps

    res = await log_heartbeat(ctx, "test_agent", "OK")
    assert "Heartbeat logged" in res
    assert len(backend.queries) == 2
    assert "Heartbeat" in backend.queries[0]["query"]

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
    assert len(backend.queries) == 1

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
    assert len(backend.queries) == 2

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
    assert len(backend.queries) == 2

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
    assert len(backend.queries) == 2
