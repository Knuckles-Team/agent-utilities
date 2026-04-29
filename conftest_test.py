import json
import os

import pytest

from agent_utilities import workspace


@pytest.fixture
def temp_workspace(tmp_path):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()

    # main_agent.json
    main_agent = {
        "name": "Test Creation Agent",
        "description": "A test agent",
        "content": "You are a test agent.",
    }
    (workspace_dir / "main_agent.json").write_text(json.dumps(main_agent))

    workspace.WORKSPACE_DIR = None
    os.environ["AGENT_WORKSPACE"] = str(workspace_dir)

    yield workspace_dir

    if "AGENT_WORKSPACE" in os.environ:
        del os.environ["AGENT_WORKSPACE"]
    workspace.WORKSPACE_DIR = None
