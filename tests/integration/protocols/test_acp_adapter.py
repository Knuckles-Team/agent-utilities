"""CONCEPT:AU-ECO.messaging.native-backend-abstraction"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from agent_utilities.core.contextual_model import create_context_agent
from agent_utilities.protocols.acp_adapter import (
    _ACP_INSTALLED,
    FileAcpSessionStore,
    build_acp_config,
    create_acp_agent,
)
from tests.integration.protocols._acp_wire import WireAgent, WireClient

pytestmark = pytest.mark.skipif(
    not _ACP_INSTALLED,
    reason="pydantic-ai-harness[acp] not installed",
)


def test_build_acp_config_uses_durable_harness_store(tmp_path: Path) -> None:
    config = build_acp_config(
        tmp_path / "sessions",
        models=["openai:gpt-5.2", "ollama:qwen3"],
        name="graph-os",
    )

    assert isinstance(config.session_store, FileAcpSessionStore)
    assert config.session_store.root == (tmp_path / "sessions").resolve()
    assert config.models == ("openai:gpt-5.2", "ollama:qwen3")
    assert config.name == "graph-os"


@pytest.mark.asyncio
async def test_file_session_store_round_trip_and_safe_filename(tmp_path: Path) -> None:
    from pydantic_ai_harness.experimental.acp import StoredSession

    store = FileAcpSessionStore(tmp_path)
    original = StoredSession(messages=[], updates=[], model="openai:gpt-5.2")

    await store.save("../../not-a-path", original)
    restored = await store.load("../../not-a-path")

    assert restored == original
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert files[0].parent == tmp_path
    assert files[0].name != "not-a-path.json"
    assert files[0].stat().st_mode & 0o077 == 0


@pytest.mark.asyncio
async def test_harness_adapter_negotiates_load_session(tmp_path: Path) -> None:
    agent = create_context_agent(
        TestModel(custom_output_text="hello"),
        default_capabilities=False,
    )
    adapter = create_acp_agent(agent, build_acp_config(tmp_path))

    response = await adapter.initialize(protocol_version=1)

    assert response.agent_capabilities is not None
    assert response.agent_capabilities.load_session is True
    assert response.agent_info is not None


@pytest.mark.asyncio
async def test_harness_adapter_round_trips_over_json_rpc_wire(tmp_path: Path) -> None:
    import acp

    from agent_utilities.core.contextual_model import use_grounding_policy

    agent = create_context_agent(
        TestModel(custom_output_text="wire response"),
        default_capabilities=False,
    )
    adapter = create_acp_agent(agent, build_acp_config(tmp_path))
    client = WireClient()

    # This test exercises the ACP/JSON-RPC wire protocol round-trip (session
    # init -> prompt -> close) against a hermetic TestModel, not real
    # retrieval/grounding. ContextualModel's grounding policy defaults to
    # "required" and fails closed with no configured ContextCompiler engine
    # (CONCEPT:AU-KG.retrieval.fail-closed-grounding-contract, deliberately
    # not opt-out-able except through this documented scope) — opt into the
    # policy's own sanctioned "none" escape hatch for this hermetic call.
    with use_grounding_policy("none"):
        async with WireAgent(adapter, client) as (connection, _client):
            initialized = await connection.initialize(protocol_version=1)
            session = await connection.new_session(cwd="/workspace", mcp_servers=[])
            response = await connection.prompt(
                session_id=session.session_id,
                prompt=[acp.text_block("hello")],
            )
            closed = await connection.close_session(session_id=session.session_id)

    assert initialized.protocol_version == 1
    assert response.stop_reason == "end_turn"
    assert closed is not None
    assert any(
        getattr(update, "session_update", "") == "agent_message_chunk"
        for update in client.updates
    )
