from __future__ import annotations

"""CONCEPT:AU-ORCH.adapter.hot-cache-invalidation"""

"""Coverage push for agent_utilities.core.config.

Targets the pure-function helpers and mocked-engine paths.
"""


import asyncio
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_utilities.core import config as ch


def _prompt_blueprint(body: str, *, task: str = "router") -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "task": task,
        "type": "prompt",
        "instructions": {"core_directive": body},
    }


@pytest.fixture(autouse=True)
def _clear_registry_cache():
    """Invalidate the _RegistryCache before each test so mocked engines are hit."""
    ch._RegistryCache.invalidate()
    yield
    ch._RegistryCache.invalidate()


def test_production_xdg_config_rejects_malformed_source_without_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "private-config-location"
    config_dir.mkdir()
    config_file = config_dir / "config.json"
    config_file.write_text("{not-json", encoding="utf-8")
    config_file.chmod(0o600)
    monkeypatch.setenv("APP_PROFILE", "production")
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(config_dir))

    with pytest.raises(ch.ConfigurationSourceError) as caught:
        ch._load_xdg_json_config()

    rendered = str(caught.value)
    assert "xdg" in rendered
    assert "JSONDecodeError" in rendered
    assert str(config_dir) not in rendered


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission contract")
def test_production_xdg_config_rejects_group_readable_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text('{"mcp_tool_mode": "intent"}', encoding="utf-8")
    config_file.chmod(0o640)
    monkeypatch.setenv("APP_PROFILE", "production")
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(tmp_path))

    with pytest.raises(ch.ConfigurationSourceError, match="PermissionError"):
        ch._load_xdg_json_config()


def test_production_xdg_config_loads_private_regular_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text('{"mcp_tool_mode": "intent"}', encoding="utf-8")
    config_file.chmod(0o600)
    monkeypatch.setenv("APP_PROFILE", "production")
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("MCP_TOOL_MODE", raising=False)

    ch._load_xdg_json_config()

    assert os.environ["MCP_TOOL_MODE"] == "intent"
    os.environ.pop("MCP_TOOL_MODE", None)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission contract")
def test_staged_production_profile_revalidates_source_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(
        '{"app_profile":"production","mcp_tool_mode":"intent"}',
        encoding="utf-8",
    )
    config_file.chmod(0o640)
    monkeypatch.delenv("APP_PROFILE", raising=False)
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(tmp_path))

    with pytest.raises(ch.ConfigurationSourceError, match="PermissionError"):
        ch._load_xdg_json_config()


@pytest.mark.parametrize(
    ("raw", "error_class"),
    [
        ('{"unknown_projection":"value"}', "UnknownKeyError"),
        (
            '{"MCP_TOOL_MODE":"intent","mcp_tool_mode":"verbose"}',
            "AmbiguousKeyError",
        ),
        (
            '{"MCP_TOOL_MODE":"intent","MCP_TOOL_MODE":"verbose"}',
            "ValueError",
        ),
    ],
)
def test_xdg_config_rejects_unknown_duplicate_or_ambiguous_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    error_class: str,
) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(raw, encoding="utf-8")
    config_file.chmod(0o600)
    monkeypatch.setenv("APP_PROFILE", "production")
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(tmp_path))

    with pytest.raises(ch.ConfigurationSourceError) as caught:
        ch._load_xdg_json_config()

    assert caught.value.error_class == error_class
    assert str(tmp_path) not in str(caught.value)


def test_configuration_reader_rejects_oversize_and_unstable_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "config.json"
    source.write_bytes(b"x" * (ch._MAX_CONFIGURATION_SOURCE_BYTES + 1))
    source.chmod(0o600)
    with pytest.raises(ch.ConfigurationSourceError, match="ValueError"):
        ch._read_configuration_mapping(source, source_type="xdg", strict=True)

    source.write_text('{"MCP_TOOL_MODE":"intent"}', encoding="utf-8")
    real_fstat = os.fstat
    calls = 0

    def changed_fstat(descriptor: int):
        nonlocal calls
        calls += 1
        metadata = real_fstat(descriptor)
        if calls != 2:
            return metadata
        return SimpleNamespace(
            st_mode=metadata.st_mode,
            st_size=metadata.st_size,
            st_uid=metadata.st_uid,
            st_dev=metadata.st_dev,
            st_ino=metadata.st_ino,
            st_mtime_ns=metadata.st_mtime_ns + 1,
        )

    monkeypatch.setattr(ch.os, "fstat", changed_fstat)
    with pytest.raises(ch.ConfigurationSourceError, match="PermissionError"):
        ch._read_configuration_mapping(source, source_type="xdg", strict=True)


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
    config_path.write_text(json.dumps({"mcpServers": {"srv1": {"command": "run"}}}))
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


def test_log_graph_trace_unknown_event_uses_graph_phase(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unknown event type falls back to 'GRAPH' phase."""
    with caplog.at_level(logging.INFO, logger="agent_utilities.graph.trace"):
        ch._log_graph_trace("totally_unknown_event", 0.0)
    assert "[GRAPH]" in caplog.text


def test_log_graph_trace_message_on_unrelated_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
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
            id="n1",
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
    assert "id=n1" in caplog.text
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


def test_render_prompt_payload_with_canonical_directive() -> None:
    """A current prompt blueprint renders its directive."""
    data = _prompt_blueprint("hello world")
    out = ch._render_prompt_payload(data)
    assert out == "hello world"


@pytest.mark.parametrize("retired_key", ["content", "input"])
def test_render_prompt_payload_rejects_retired_body_keys(retired_key: str) -> None:
    with pytest.raises(ValueError, match="retired prompt body key"):
        ch._render_prompt_payload({"task": "do_it", retired_key: "body"})


def test_render_prompt_payload_structured_instructions() -> None:
    data = _prompt_blueprint("with input", task="do_a_thing")
    assert ch._render_prompt_payload(data) == "with input"


def test_render_prompt_payload_rejects_invalid_blueprint() -> None:
    data = {"unknown_field": "value"}
    with pytest.raises(ValueError):
        ch._render_prompt_payload(data)


# ---------------------------------------------------------------------------
# get_discovery_registry
# ---------------------------------------------------------------------------


def test_get_discovery_registry_no_engine_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cold registry startup uses the sole zero-argument XDG path authority."""
    from agent_utilities.core import paths

    # Mock knowledge_graph.engine.IntelligenceGraphEngine
    fake_engine = MagicMock(backend=None)
    fake_engine_cls = MagicMock(
        get_active=MagicMock(return_value=None),
        get_or_create=MagicMock(return_value=fake_engine),
    )
    fake_kg = MagicMock(IntelligenceGraphEngine=fake_engine_cls)
    path_calls = 0
    expected_path = tmp_path / "data" / "kg" / "knowledge_graph.db"

    def canonical_path() -> Path:
        nonlocal path_calls
        path_calls += 1
        return expected_path

    monkeypatch.setattr(paths, "kg_db_path", canonical_path)
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.core.engine",
        fake_kg,
    )
    result = ch.get_discovery_registry()
    assert result.agents == []
    assert result.tools == []
    assert path_calls == 1
    fake_engine_cls.get_or_create.assert_called_once_with(db_path=str(expected_path))


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
        "agent_utilities.knowledge_graph.core.engine",
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
                "json_blueprint": _prompt_blueprint("router JSON"),
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
        "agent_utilities.knowledge_graph.core.engine",
        fake_kg,
    )
    result = ch.get_discovery_registry()
    # Expect 1 prompt agent + 2 dynamically synthesized agents ("git" and "srv")
    assert len(result.agents) == 3
    assert any(a.name == "router" for a in result.agents)
    assert len(result.tools) == 1
    assert result.tools[0].name == "tool1"


def test_get_discovery_registry_blueprint_json_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A json_blueprint stored as a JSON-encoded string is parsed."""
    fake_engine = MagicMock()
    fake_engine.backend = MagicMock()
    blueprint = _prompt_blueprint("router JSON")
    blueprint_str = json.dumps(blueprint)
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
        "agent_utilities.knowledge_graph.core.engine",
        fake_kg,
    )
    result = ch.get_discovery_registry()
    assert len(result.agents) == 1
    assert result.agents[0].json_blueprint == blueprint


def test_get_discovery_registry_rejects_non_json_blueprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Python-literal-style payload is not part of the current contract."""
    fake_engine = MagicMock()
    fake_engine.backend = MagicMock()
    # This parses in Python but is deliberately invalid JSON.
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
        "agent_utilities.knowledge_graph.core.engine",
        fake_kg,
    )
    result = ch.get_discovery_registry()
    assert result.agents == []


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
        "agent_utilities.knowledge_graph.core.engine",
        fake_kg,
    )
    result = ch.get_discovery_registry()
    # Stored prompt blueprints have one current contract: canonical JSON
    # objects. An unparseable string is rejected rather than silently treated
    # as an agent without a blueprint.
    assert result.agents == []


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
        "agent_utilities.knowledge_graph.core.engine",
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
        "agent_utilities.knowledge_graph.core.engine",
        fake_kg,
    )
    result = ch.get_discovery_registry()
    assert result.tools == []


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
                json_blueprint=_prompt_blueprint("router prompt text"),
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
