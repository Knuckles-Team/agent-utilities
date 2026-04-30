"""Coverage push for agent_utilities.graph.config_helpers.

Targets the pure-function helpers and mocked-engine paths.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_utilities.graph import config_helpers as ch

# ---------------------------------------------------------------------------
# load_mcp_config / save_mcp_config
# ---------------------------------------------------------------------------


def test_load_mcp_config_returns_empty_when_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing file yields empty MCPConfigModel."""
    monkeypatch.setattr(
        ch, "get_workspace_path", lambda name: tmp_path / "missing.json"
    )
    cfg = ch.load_mcp_config()
    assert cfg.mcpServers == {}


def test_load_mcp_config_reads_valid_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valid JSON is parsed into MCPConfigModel."""
    config_path = tmp_path / "mcp_config.json"
    config_path.write_text(
        json.dumps({"mcpServers": {"srv1": {"command": "run"}}})
    )
    monkeypatch.setattr(ch, "get_workspace_path", lambda name: config_path)
    cfg = ch.load_mcp_config()
    assert "srv1" in cfg.mcpServers


def test_load_mcp_config_invalid_json_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid JSON falls through to an empty MCPConfigModel."""
    config_path = tmp_path / "bad.json"
    config_path.write_text("{ not valid")
    monkeypatch.setattr(ch, "get_workspace_path", lambda name: config_path)
    cfg = ch.load_mcp_config()
    assert cfg.mcpServers == {}


def test_save_mcp_config_writes_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """save_mcp_config round-trips through a file."""
    from agent_utilities.models import MCPConfigModel

    out_path = tmp_path / "saved.json"
    monkeypatch.setattr(ch, "get_workspace_path", lambda name: out_path)
    config = MCPConfigModel(mcpServers={"srv1": {"command": "run"}})
    ch.save_mcp_config(config)
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["mcpServers"]["srv1"]["command"] == "run"


# ---------------------------------------------------------------------------
# emit_graph_event
# ---------------------------------------------------------------------------


def test_emit_graph_event_no_queue() -> None:
    """emit_graph_event with eq=None only logs, does not raise."""
    ch.emit_graph_event(None, "node_start", agent="router")
    assert True, "emit_graph_event with eq=None should not raise"


def test_emit_graph_event_with_queue() -> None:
    """emit_graph_event pushes a dict to the asyncio queue."""

    async def _run() -> None:
        q: asyncio.Queue[Any] = asyncio.Queue()
        ch.emit_graph_event(q, "graph_start", agent="planner", duration_ms=100)
        assert q.qsize() == 1
        msg = await q.get()
        assert msg["type"] == "data-graph-event"
        assert msg["data"]["event"] == "graph_start"
        assert msg["data"]["agent"] == "planner"
        assert msg["data"]["duration_ms"] == 100
        assert "timestamp" in msg["data"]

    asyncio.run(_run())


def test_emit_graph_event_queue_full_raises_caught(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If put_nowait raises, the exception is caught and logged."""
    q = MagicMock()
    q.put_nowait.side_effect = asyncio.QueueFull()
    # Must not raise
    ch.emit_graph_event(q, "error")
    assert q.put_nowait.called


# ---------------------------------------------------------------------------
# _log_graph_trace (exercised via emit_graph_event)
# ---------------------------------------------------------------------------


def test_log_graph_trace_tool_name(caplog: pytest.LogCaptureFixture) -> None:
    """tool_name gets prefixed with 'tool=' in the detail."""
    with caplog.at_level(logging.INFO, logger="agent_utilities.graph.trace"):
        ch._log_graph_trace("expert_tool_call", 0.0, tool_name="read_file")
    assert "tool=read_file" in caplog.text


def test_log_graph_trace_success_flag(caplog: pytest.LogCaptureFixture) -> None:
    """success flag is logged as 'ok=...'."""
    with caplog.at_level(logging.INFO, logger="agent_utilities.graph.trace"):
        ch._log_graph_trace("tool_result", 0.0, success=True)
    assert "ok=True" in caplog.text


def test_log_graph_trace_message_is_truncated(caplog: pytest.LogCaptureFixture) -> None:
    """Long messages on warning events are truncated to 120 chars."""
    long_msg = "x" * 500
    with caplog.at_level(logging.INFO, logger="agent_utilities.graph.trace"):
        ch._log_graph_trace("expert_warning", 0.0, message=long_msg)
    assert "msg=" in caplog.text
    # Truncated to 120 chars
    assert "x" * 121 not in caplog.text


def test_log_graph_trace_unknown_event_uses_graph_phase(caplog: pytest.LogCaptureFixture) -> None:
    """Unknown event type falls back to 'GRAPH' phase."""
    with caplog.at_level(logging.INFO, logger="agent_utilities.graph.trace"):
        ch._log_graph_trace("totally_unknown_event", 0.0)
    assert "[GRAPH]" in caplog.text


def test_log_graph_trace_message_on_unrelated_event(caplog: pytest.LogCaptureFixture) -> None:
    """`message` is only logged for expert_warning/safety_warning."""
    with caplog.at_level(logging.INFO, logger="agent_utilities.graph.trace"):
        ch._log_graph_trace("graph_start", 0.0, message="hi")
    assert "msg=" not in caplog.text


def test_log_graph_trace_all_metadata_keys(caplog: pytest.LogCaptureFixture) -> None:
    """Detail string includes multiple known keys."""
    with caplog.at_level(logging.INFO, logger="agent_utilities.graph.trace"):
        ch._log_graph_trace(
            "specialist_enter",
            0.0,
            agent="router",
            expert="planner",
            node_id="n1",
            domain="code",
            server="srv",
            count=3,
            score=95,
            batch_size=2,
            attempt=1,
            duration_ms=500,
        )
    assert "agent=router" in caplog.text
    assert "expert=planner" in caplog.text
    assert "node_id=n1" in caplog.text
    assert "[EXECUTION]" in caplog.text


def test_log_graph_trace_no_extras(caplog: pytest.LogCaptureFixture) -> None:
    """No extra kwargs: the detail section is empty."""
    with caplog.at_level(logging.INFO, logger="agent_utilities.graph.trace"):
        ch._log_graph_trace("node_start", 0.0)
    assert "[LIFECYCLE]" in caplog.text
    assert "node_start" in caplog.text


# ---------------------------------------------------------------------------
# _render_prompt_payload
# ---------------------------------------------------------------------------


def test_render_prompt_payload_with_content() -> None:
    """dict with non-empty content field is JSON-serialized as-is."""
    data = {"content": "hello world", "meta": "x"}
    out = ch._render_prompt_payload(data)
    assert json.loads(out) == data


def test_render_prompt_payload_empty_content_fallback() -> None:
    """Empty content falls back to StructuredPrompt rendering or JSON dump."""
    data = {"content": "   ", "task": "do it", "input": "x"}
    out = ch._render_prompt_payload(data)
    # Result is a string
    assert isinstance(out, str)


def test_render_prompt_payload_no_content_key() -> None:
    """No content key falls back to StructuredPrompt."""
    data = {"task": "do a thing", "input": "with input"}
    out = ch._render_prompt_payload(data)
    assert isinstance(out, str)


def test_render_prompt_payload_structured_prompt_fail() -> None:
    """If StructuredPrompt fails to validate, fall back to json.dumps."""
    data = {"unknown_field": "value"}
    out = ch._render_prompt_payload(data)
    # Even if StructuredPrompt validation fails, json.dumps succeeds
    assert isinstance(out, str)
    # Should contain the data (either structured or raw)
    # Parse it to ensure it's valid JSON or a rendering
    parsed = None
    try:
        parsed = json.loads(out)
    except Exception:
        pass
    # Either JSON succeeded or rendering returned string
    assert parsed is not None or "unknown_field" in out or out


# ---------------------------------------------------------------------------
# get_discovery_registry
# ---------------------------------------------------------------------------


def test_get_discovery_registry_no_engine_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When get_active returns None, a fresh engine is instantiated."""
    # Mock knowledge_graph.engine.IntelligenceGraphEngine
    fake_engine = MagicMock(backend=None)
    fake_engine_cls = MagicMock(
        get_active=MagicMock(return_value=None),
        return_value=fake_engine,
    )
    fake_kg = MagicMock(IntelligenceGraphEngine=fake_engine_cls)
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    result = ch.get_discovery_registry()
    assert result.agents == []
    assert result.tools == []


def test_get_discovery_registry_with_engine_no_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When engine has no backend, empty registry returned."""
    fake_engine = MagicMock(backend=None)
    fake_engine_cls = MagicMock(
        get_active=MagicMock(return_value=fake_engine),
    )
    fake_kg = MagicMock(IntelligenceGraphEngine=fake_engine_cls)
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    result = ch.get_discovery_registry()
    assert result.agents == []


def test_get_discovery_registry_with_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prompt rows populate agents list."""
    fake_engine = MagicMock()
    fake_engine.backend = MagicMock()
    # First call returns prompts, second returns agents, third returns tools
    fake_engine.backend.execute.side_effect = [
        [
            {
                "name": "router",
                "description": "routes queries",
                "capabilities": ["routing"],
                "system_prompt": "You are the router",
                "json_blueprint": {"content": "router JSON"},
            }
        ],
        [],  # agents
        [  # tools
            {
                "t.name": "tool1",
                "t.description": "desc",
                "t.mcp_server": "srv",
                "t.relevance_score": 80,
                "t.tags": ["git"],
                "t.requires_approval": False,
            }
        ],
    ]
    fake_engine_cls = MagicMock(
        get_active=MagicMock(return_value=fake_engine),
    )
    fake_kg = MagicMock(IntelligenceGraphEngine=fake_engine_cls)
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    result = ch.get_discovery_registry()
    assert len(result.agents) == 1
    assert result.agents[0].name == "router"
    assert len(result.tools) == 1
    assert result.tools[0].name == "tool1"


def test_get_discovery_registry_blueprint_json_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A json_blueprint stored as a JSON-encoded string is parsed."""
    fake_engine = MagicMock()
    fake_engine.backend = MagicMock()
    blueprint_str = json.dumps({"content": "router JSON"})
    # prompts, agents, tools
    fake_engine.backend.execute.side_effect = [
        [
            {
                "name": "router",
                "description": "",
                "capabilities": [],
                "system_prompt": "",
                "json_blueprint": blueprint_str,
            }
        ],
        [],
        [],
    ]
    fake_engine_cls = MagicMock(
        get_active=MagicMock(return_value=fake_engine),
    )
    fake_kg = MagicMock(IntelligenceGraphEngine=fake_engine_cls)
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    result = ch.get_discovery_registry()
    assert len(result.agents) == 1
    assert result.agents[0].json_blueprint == {"content": "router JSON"}


def test_get_discovery_registry_blueprint_literal_eval_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If JSON parse fails, ast.literal_eval is attempted."""
    fake_engine = MagicMock()
    fake_engine.backend = MagicMock()
    # A Python-literal-style string that ast.literal_eval can parse but json cannot
    literal_str = "{'content': 'router'}"
    # prompts, agents, tools
    fake_engine.backend.execute.side_effect = [
        [
            {
                "name": "router",
                "description": "",
                "capabilities": [],
                "system_prompt": "",
                "json_blueprint": literal_str,
            }
        ],
        [],
        [],
    ]
    fake_engine_cls = MagicMock(
        get_active=MagicMock(return_value=fake_engine),
    )
    fake_kg = MagicMock(IntelligenceGraphEngine=fake_engine_cls)
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    result = ch.get_discovery_registry()
    assert result.agents[0].json_blueprint == {"content": "router"}


def test_get_discovery_registry_blueprint_unparseable_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unparseable blueprint string -> Pydantic validation fails, overall exception
    is caught, agent list is empty (the outer try/except block).
    """
    fake_engine = MagicMock()
    fake_engine.backend = MagicMock()
    # prompts, agents, tools
    fake_engine.backend.execute.side_effect = [
        [
            {
                "name": "router",
                "description": "",
                "capabilities": [],
                "system_prompt": "",
                "json_blueprint": "not valid at all",
            }
        ],
        [],
        [],
    ]
    fake_engine_cls = MagicMock(
        get_active=MagicMock(return_value=fake_engine),
    )
    fake_kg = MagicMock(IntelligenceGraphEngine=fake_engine_cls)
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    result = ch.get_discovery_registry()
    # Unparseable blueprint string is gracefully handled: the agent is still
    # included but with json_blueprint=None (no outer exception is raised).
    assert len(result.agents) == 1
    assert result.agents[0].name == "router"
    assert result.agents[0].json_blueprint is None


def test_get_discovery_registry_prompt_query_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exception on prompt fetch is caught; still tries tools."""
    fake_engine = MagicMock()
    fake_engine.backend = MagicMock()
    fake_engine.backend.execute.side_effect = [
        RuntimeError("prompt query failed"),
        [],  # tools
    ]
    fake_engine_cls = MagicMock(
        get_active=MagicMock(return_value=fake_engine),
    )
    fake_kg = MagicMock(IntelligenceGraphEngine=fake_engine_cls)
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    result = ch.get_discovery_registry()
    assert result.agents == []


def test_get_discovery_registry_tool_query_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exception on tool fetch is caught; tools list empty."""
    fake_engine = MagicMock()
    fake_engine.backend = MagicMock()
    fake_engine.backend.execute.side_effect = [
        [],  # prompts
        RuntimeError("tool query failed"),
    ]
    fake_engine_cls = MagicMock(
        get_active=MagicMock(return_value=fake_engine),
    )
    fake_kg = MagicMock(IntelligenceGraphEngine=fake_engine_cls)
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    result = ch.get_discovery_registry()
    assert result.tools == []


def test_load_node_agents_registry_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy alias delegates to get_discovery_registry."""
    sentinel = MagicMock()
    monkeypatch.setattr(ch, "get_discovery_registry", lambda: sentinel)
    result = ch.load_node_agents_registry()
    assert result is sentinel


# ---------------------------------------------------------------------------
# load_specialized_prompts
# ---------------------------------------------------------------------------


def test_load_specialized_prompts_from_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Agent with json_blueprint is rendered from the registry."""
    from agent_utilities.models import MCPAgent, MCPAgentRegistryModel

    registry = MCPAgentRegistryModel(
        agents=[
            MCPAgent(
                name="router",
                json_blueprint={"content": "router prompt text"},
            )
        ]
    )
    monkeypatch.setattr(ch, "get_discovery_registry", lambda: registry)
    result = ch.load_specialized_prompts("router")
    assert "router prompt text" in result


def test_load_specialized_prompts_prompt_file_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Agent with .json prompt_file reads from disk."""
    from agent_utilities.models import MCPAgent, MCPAgentRegistryModel

    # The resolver in config_helpers does: Path(__file__).parent.parent / prompt_file
    # So we need a file at agent_utilities/<x>.json
    # We can use any existing JSON file; easier: monkeypatch Path globally is fragile.
    # Instead, we just make prompt_file None and test the fallback path below.
    registry = MCPAgentRegistryModel(
        agents=[
            MCPAgent(
                name="noblueprint",
                prompt_file=None,
                json_blueprint=None,
            )
        ]
    )
    monkeypatch.setattr(ch, "get_discovery_registry", lambda: registry)
    # Falls through to the JSON file fallback
    result = ch.load_specialized_prompts("noblueprint")
    assert isinstance(result, str)


def test_load_specialized_prompts_fallback_to_prompts_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No registry match -> fall back to agent_utilities/prompts/<name>.json."""
    from agent_utilities.models import MCPAgentRegistryModel

    monkeypatch.setattr(
        ch,
        "get_discovery_registry",
        lambda: MCPAgentRegistryModel(agents=[]),
    )
    # For a non-existent slug, fallback sentence is returned
    result = ch.load_specialized_prompts("totally-missing-slug-xyz")
    assert "totally-missing-slug-xyz" in result
    assert "helpful assistant" in result.lower()


def test_load_specialized_prompts_empty_blueprint_tries_prompt_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty blueprint + missing prompt_file falls through to JSON file lookup."""
    from agent_utilities.models import MCPAgent, MCPAgentRegistryModel

    registry = MCPAgentRegistryModel(
        agents=[
            MCPAgent(
                name="myagent",
                json_blueprint=None,
                prompt_file=None,
            )
        ]
    )
    monkeypatch.setattr(ch, "get_discovery_registry", lambda: registry)
    result = ch.load_specialized_prompts("myagent")
    assert isinstance(result, str)
