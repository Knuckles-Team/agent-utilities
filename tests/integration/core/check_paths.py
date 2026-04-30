import os

from agent_utilities.core.workspace import CORE_FILES, get_workspace_path

print(f"Current Directory: {os.getcwd()}")
print(f"Workspace Path for MCP_CONFIG: {get_workspace_path(CORE_FILES['MCP_CONFIG'])}")
print(f"Exists: {get_workspace_path(CORE_FILES['MCP_CONFIG']).exists()}")
