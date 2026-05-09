import os
from pathlib import Path

# Add agents to path
WORKSPACE_ROOT = Path("/home/apps/workspace")
AGENTS_DIR = WORKSPACE_ROOT / "agent-packages" / "agents"


def test_instrumentation_presence():
    """Verify that key agents have the expected instrumentation in their source code."""

    # We'll check source code directly to avoid dependency hell
    test_cases = [
        ("portainer-agent", "docker_stop_container_tool", "ctx_confirm_destructive"),
        ("microsoft-agent", "login", "ctx_set_state"),
        ("container-manager-mcp", "pull_image", "ctx_progress"),
        ("microsoft-agent", "list_mail_messages", "ctx_sample"),
        ("postiz-agent", "postiz_get_missing_content", "ctx_sample"),
    ]

    for agent, tool, helper in test_cases:
        # Find the file
        agent_path = AGENTS_DIR / agent
        server_files = []
        for root, dirs, files in os.walk(agent_path):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            if "mcp_server.py" in files:
                server_files.append(Path(root) / "mcp_server.py")

        assert len(server_files) > 0, f"Could not find mcp_server.py for {agent}"

        content = ""
        for server_file in server_files:
            with open(server_file) as f:
                content += f.read() + "\n"

        # Check if the tool function contains the helper
        assert tool in content, f"Tool {tool} not found in {agent}"
        assert helper in content, f"Helper {helper} not found in {agent}"

        # Check for 'async def' if it uses an async helper
        if helper in [
            "ctx_confirm_destructive",
            "ctx_progress",
            "ctx_sample",
            "ctx_set_state",
        ]:
            lines = content.splitlines()
            tool_def_line = next(
                (line for line in lines if f"def {tool}" in line), None
            )
            assert tool_def_line is not None, (
                f"Could not find def for {tool} in {agent}"
            )
            assert "async def" in tool_def_line, (
                f"Tool {tool} in {agent} is not async despite using {helper}"
            )


if __name__ == "__main__":
    test_instrumentation_presence()
