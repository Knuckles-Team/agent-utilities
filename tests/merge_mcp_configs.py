import json
import os
import subprocess


def merge_mcp_configs(target_path):
    master_config = {"mcpServers": {}}

    # Files to ignore (e.g. mock data or temp files if any)
    ignore_paths = ["mock_agent_data", "tests"]

    # Discover all mcp_config.json files
    try:
        result = subprocess.run(
            [
                "find",
                "/home/genius/Workspace/agent-packages",
                "-name",
                "mcp_config.json",
            ],
            capture_output=True,
            text=True,
        )
        paths = result.stdout.strip().split("\n")
    except Exception as e:
        print(f"Error finding configurations: {e}")
        return

    for path in paths:
        if not path or any(ignore in path for ignore in ignore_paths):
            continue

        try:
            with open(path, "r") as f:
                data = json.load(f)
                servers = data.get("mcpServers", {})
                for name, cfg in servers.items():
                    # If duplicate, we just keep the first one found or merge if they match
                    # User requested consolidation, so we just build the master list
                    if name in master_config["mcpServers"]:
                        continue
                    master_config["mcpServers"][name] = cfg
        except Exception as e:
            print(f"Error reading {path}: {e}")

    # Ensure directory exists
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    with open(target_path, "w") as f:
        json.dump(master_config, f, indent=2)

    print(
        f"✅ Consolidated {len(master_config['mcpServers'])} servers into {target_path}"
    )


if __name__ == "__main__":
    target = "/home/genius/Workspace/agent-packages/agents/genius-agent/genius_agent/agent_data/mcp_config.json"
    merge_mcp_configs(target)
