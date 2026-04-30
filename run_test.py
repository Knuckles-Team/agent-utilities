import json
import os
from pathlib import Path

from agent_utilities import build_system_prompt_from_workspace
from agent_utilities.core.workspace import (
    get_agent_workspace,
    load_workspace_file,
)

os.environ["AGENT_WORKSPACE"] = "/tmp/test_workspace"
os.makedirs("/tmp/test_workspace", exist_ok=True)
Path("/tmp/test_workspace/main_agent.json").write_text(
    json.dumps(
        {
            "name": "Test Creation Agent",
            "description": "A test agent",
            "content": "You are a test agent.",
        }
    )
)

print("get_agent_workspace:", get_agent_workspace())
print("load_workspace_file:", load_workspace_file("main_agent.json"))
print("build_system_prompt_from_workspace:", build_system_prompt_from_workspace())
