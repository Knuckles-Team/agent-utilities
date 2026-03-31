#!/usr/bin/python

from __future__ import annotations

import logging
import asyncio


from typing import Dict, List, Optional, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    pass





from .config import *
from .workspace import *


from .models import PeriodicTask

tasks: List[PeriodicTask] = []
lock = asyncio.Lock()




logger = logging.getLogger(__name__)


def load_a2a_peers() -> List[Dict[str, str]]:
    """Parse A2A_AGENTS.md table into list of dicts."""
    content = load_workspace_file(CORE_FILES["AGENTS"])
    if not content:
        return []

    peers = []
    lines = content.splitlines()
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("| Name") or stripped.startswith("| ID"):
            in_table = True
            continue
        if (
            in_table
            and stripped.startswith("|")
            and "|" in stripped
            and not (
                stripped.startswith("|---")
                or stripped.startswith("| ID")
                or stripped.startswith("| Name")
            )
        ):
            parts = [p.strip() for p in stripped.strip("| ").split("|")]
            if len(parts) >= 5:
                peers.append(
                    {
                        "name": parts[0],
                        "url": parts[1],
                        "description": parts[2],
                        "capabilities": parts[3],
                        "auth": parts[4] if len(parts) > 4 else "none",
                        "notes": parts[5] if len(parts) > 5 else "",
                    }
                )
    return peers


def register_a2a_peer(
    name: str,
    url: str,
    description: str = "",
    capabilities: str = "",
    auth: str = "none",
    notes: str = "",
) -> str:
    """Add or update a peer in A2A_AGENTS.md table."""
    path = get_workspace_path(CORE_FILES["AGENTS"])
    if not path.exists():
        initialize_workspace()

    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()

    table_start = -1
    for i, line in enumerate(lines):
        if "| Name" in line or "| ID" in line:
            table_start = i
            break

    if table_start == -1:
        new_table = (
            "\n## Registered A2A Peers\n\n"
            "| Name | Endpoint URL | Description | Capabilities | Auth | Notes / Last Connected |\n"
            "|------|--------------|-------------|--------------|------|------------------------|\n"
        )
        content += new_table
        lines = content.splitlines()
        table_start = len(lines) - 3

    new_row = f"| {name} | {url} | {description} | {capabilities} | {auth} | {notes or datetime.now().strftime('%Y-%m-%d')} |"

    updated = False
    for i in range(table_start + 2, len(lines)):
        if lines[i].strip().startswith(f"| {name} ") or f"| {name} |" in lines[i]:
            lines[i] = new_row
            updated = True
            break

    if not updated:
        lines.insert(table_start + 3, new_row)

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return f"✅ Registered/updated A2A peer '{name}' at {url}"


def get_a2a_peer(name: str) -> Optional[Dict[str, str]]:
    """Return single peer by name (case-insensitive)."""
    peers = load_a2a_peers()
    name_lower = name.lower()
    for p in peers:
        if p.get("name", "").lower() == name_lower:
            return p
    return None


def list_a2a_peers() -> str:
    """List all registered A2A peers formatted for the LLM."""
    peers = load_a2a_peers()
    if not peers:
        return "No A2A peers registered yet."
    lines = ["## Known A2A Peers"]
    for p in peers:
        lines.append(f"- **{p['name']}** → {p['url']}  ({p['capabilities']})")
    return "\n".join(lines)


def delete_a2a_peer(name: str) -> str:
    """Remove a peer from A2A_AGENTS.md registry."""
    path = get_workspace_path(CORE_FILES["AGENTS"])
    if not path.exists():
        return f"❌ {CORE_FILES['AGENTS']} not found."

    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    new_lines = []
    found = False

    for line in lines:

        if line.strip().startswith(f"| {name} ") or f"| {name} |" in line:
            found = True
            continue
        new_lines.append(line)

    if found:
        path.write_text("\n".join(new_lines).strip() + "\n", encoding="utf-8")
        return f"✅ Removed A2A peer '{name}' from registry."
    return f"ℹ️ A2A peer '{name}' not found in registry."


def discover_agents(
    include_packages: list[str] | None = None,
    exclude_packages: list[str] | None = None,
    keywords: list[str] | None = None,
) -> dict[str, str]:
    """Discovers available agent packages using installed package metadata.

    Args:
        include_packages: Optional list of package names to explicitly include (and install if missing).
        exclude_packages: Optional list of package names to ignore.
        keywords: Optional list of keywords to identify agent-like packages.

    Returns:
        dict: {tag: package_name}
    """
    import importlib.metadata
    import importlib.util
    import warnings

    agent_descriptions = {}
    processed_packages = set()

    _keywords = keywords or [
        "agent",
        "api",
        "mcp",
        "manager",
        "transcriber",
        "downloader",
    ]
    skip_packages = exclude_packages or [
        "agent-utilities",
        "universal-skills",
        "genius-agent",
    ]

    if include_packages:
        for pkg in include_packages:
            if ensure_package_installed(pkg):
                package_name = pkg.replace("-", "_")
                tag = pkg.replace("-agent", "").replace("-api", "").lower()
                agent_descriptions[tag] = package_name
                processed_packages.add(pkg)

    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name in processed_packages or any(s in name.lower() for s in skip_packages):
            continue
        processed_packages.add(name)

        if any(keyword in name.lower() for keyword in _keywords):
            package_name = name.replace("-", "_")
            try:

                spec = importlib.util.find_spec(f"{package_name}.agent_server")
                if spec:

                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=DeprecationWarning)
                        module = importlib.import_module(f"{package_name}.agent_server")

                    if hasattr(module, "agent_template"):

                        tag = name.replace("-agent", "").replace("-api", "").lower()
                        agent_descriptions[tag] = package_name
            except (ImportError, Exception):
                pass

    return agent_descriptions
