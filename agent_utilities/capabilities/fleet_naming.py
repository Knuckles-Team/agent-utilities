"""Stable public names for fleet tools used by agent-side clients."""

from __future__ import annotations


def clean_tool_name(prefix: str, server_name: str, original_tool_name: str) -> str:
    """Strip redundant server prefixes and respect the public name budget."""
    if server_name.startswith("systems-manager-mcp-"):
        base_server = "systems-manager-mcp"
    elif server_name.startswith("container-manager-mcp-"):
        base_server = "container-manager-mcp"
    else:
        base_server = server_name

    clean_server = base_server.replace("-", "_").lower()
    cleaned = original_tool_name
    strips = [
        f"{clean_server}_mcp_",
        f"{clean_server}_",
        f"{prefix}_mcp_",
        f"{prefix}_",
    ]
    if base_server.endswith("-mcp"):
        mod_server = base_server[:-4].replace("-", "_").lower()
        strips.extend((f"{mod_server}_mcp_", f"{mod_server}_"))

    for redundant in strips:
        if cleaned.startswith(redundant):
            cleaned = cleaned[len(redundant) :]
            break

    candidate = f"{prefix}__{cleaned}"
    if len(candidate) > 44:
        budget = 44 - len(prefix) - 2
        candidate = f"{prefix}__{cleaned[:budget].strip('_')}"
    return candidate
