"""CONCEPT:AU-ECO.messaging.native-backend-abstraction"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from acp import RequestError, schema
from pydantic_ai.models.test import TestModel

from agent_utilities.core.contextual_model import create_context_agent
from agent_utilities.protocols.acp_adapter import (
    _ACP_INSTALLED,
    build_acp_config,
    create_graph_acp_agent,
)

pytestmark = pytest.mark.skipif(
    not _ACP_INSTALLED,
    reason="pydantic-ai-harness[acp] not installed",
)


class RecordingClient:
    """Minimal in-memory client for driving the Harness adapter."""

    def __init__(self) -> None:
        self.updates: list[Any] = []

    def on_connect(self, connection: Any) -> None:
        return None

    async def session_update(
        self,
        session_id: str,
        update: Any,
        **kwargs: Any,
    ) -> None:
        self.updates.append(update)


@pytest.mark.asyncio
async def test_graph_adapter_dispatches_through_graph_authority(tmp_path) -> None:
    import acp

    base_agent = create_context_agent(
        TestModel(call_tools=["execute_graph"]),
        default_capabilities=False,
    )
    graph = MagicMock()
    graph_config: dict[str, Any] = {"mcp_toolsets": []}
    adapter = create_graph_acp_agent(
        base_agent,
        build_acp_config(tmp_path),
        graph_bundle=(graph, graph_config),
    )
    client = RecordingClient()
    adapter.on_connect(client)
    await adapter.initialize(protocol_version=1)
    session = await adapter.new_session(cwd="/workspace")

    execute = AsyncMock(return_value={"results": {"output": "graph result"}})
    with patch(
        "agent_utilities.graph.protocol_agnostic_execution.execute_graph",
        execute,
    ):
        response = await adapter.prompt(
            session_id=session.session_id,
            prompt=[acp.text_block("delegate this")],
        )

    assert response.stop_reason == "end_turn"
    execute.assert_awaited_once()
    call = execute.await_args.kwargs
    assert call["graph"] is graph
    # TestModel deterministically synthesizes "a" for a required string tool
    # argument. The assertion proves that the wrapper forwards the model's tool
    # argument unchanged to the one graph execution authority.
    assert call["query"] == "a"
    assert call["mode"] == "ask"
    assert call["requested_model_id"] == "test"
    assert client.updates


@pytest.mark.asyncio
async def test_graph_adapter_persists_and_loads_session(tmp_path) -> None:
    base_agent = create_context_agent(
        TestModel(custom_output_text="hello"),
        default_capabilities=False,
    )
    adapter = create_graph_acp_agent(
        base_agent,
        build_acp_config(tmp_path),
    )
    client = RecordingClient()
    adapter.on_connect(client)
    await adapter.initialize(protocol_version=1)
    session = await adapter.new_session(cwd="/workspace")

    replacement = create_graph_acp_agent(
        base_agent,
        build_acp_config(tmp_path),
    )
    replacement.on_connect(RecordingClient())
    await replacement.initialize(protocol_version=1)
    result = await replacement.load_session(
        cwd="/workspace",
        session_id=session.session_id,
    )

    assert result is not None


@pytest.mark.asyncio
async def test_graph_adapter_rejects_untrusted_client_mcp(tmp_path) -> None:
    base_agent = create_context_agent(
        TestModel(custom_output_text="hello"),
        default_capabilities=False,
    )
    adapter = create_graph_acp_agent(
        base_agent,
        build_acp_config(tmp_path),
        graph_bundle=(MagicMock(), {"mcp_toolsets": []}),
    )
    adapter.on_connect(RecordingClient())
    await adapter.initialize(protocol_version=1)

    with pytest.raises(RequestError) as exc:
        await adapter.new_session(
            cwd="/workspace",
            mcp_servers=[
                schema.McpServerStdio(
                    name="untrusted",
                    command="arbitrary-command",
                    args=[],
                    env=[],
                )
            ],
        )

    assert exc.value.code == RequestError.invalid_params({}).code
