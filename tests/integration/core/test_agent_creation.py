"""
Concept: creating-an-agent

Tests the agent creation bootstrap pattern defined in docs/creating-an-agent.md
"""

import json
import os
import pytest
from unittest.mock import patch

from agent_utilities import (
    create_graph_agent_server,
    initialize_workspace,
    load_identity,
)


@pytest.fixture
def temp_workspace(tmp_path, monkeypatch):
    """Create a temporary workspace with required files."""
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()

    # main_agent.json
    main_agent = {
        "name": "Test Creation Agent",
        "description": "A test agent",
        "content": "You are a test agent."
    }
    (workspace_dir / "main_agent.json").write_text(json.dumps(main_agent))

    # mcp_config.json
    mcp_config = {
        "mcpServers": {
            "test-server": {
                "command": "echo",
                "args": ["hello"]
            }
        }
    }
    (workspace_dir / "mcp_config.json").write_text(json.dumps(mcp_config))

    # Use monkeypatch to ensure isolation
    monkeypatch.setattr("agent_utilities.workspace.WORKSPACE_DIR", None)
    monkeypatch.setenv("AGENT_WORKSPACE", str(workspace_dir))

    yield workspace_dir


@pytest.mark.concept("creating-an-agent")
@patch("uvicorn.run")
def test_create_agent_bootstrap_pattern(mock_run, temp_workspace):
    """Test the canonical agent bootstrap pattern."""
    # 1. Initialize workspace
    initialize_workspace()

    # 2. Load identity and build prompt
    with patch("agent_utilities.prompt_builder.load_identity") as mock_load:
        mock_load.return_value = {"name": "Test Creation Agent", "content": ""}
        meta = load_identity()

    from agent_utilities import build_system_prompt_from_workspace

    agent_name = os.getenv("DEFAULT_AGENT_NAME", meta.get("name", "My Agent"))
    system_prompt = os.getenv(
        "AGENT_SYSTEM_PROMPT", meta.get("content") or build_system_prompt_from_workspace()
    )

    # 3. Create server (uvicorn.run is mocked so it won't block)
    create_graph_agent_server(
        name=agent_name,
        system_prompt=system_prompt,
        mcp_config="mcp_config.json",
        workspace=str(temp_workspace),
        enable_web_ui=False,
    )

    # 4. Verify uvicorn.run was called
    mock_run.assert_called_once()

    # Verify the app passed to uvicorn is valid
    app = mock_run.call_args[0][0]
    assert app is not None
