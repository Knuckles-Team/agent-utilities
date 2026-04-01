#!/usr/bin/python

from __future__ import annotations

import logging
import asyncio


from typing import List, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    pass


from .config import *  # noqa: F403
from .workspace import (
    CORE_FILES,
    parse_a2a_registry,
    serialize_a2a_registry,
)


from .models import PeriodicTask, A2ARegistryModel

tasks: List[PeriodicTask] = []
lock = asyncio.Lock()


logger = logging.getLogger(__name__)


def load_a2a_peers() -> A2ARegistryModel:
    """Parse AGENTS.md table into A2ARegistryModel."""
    content = load_workspace_file(CORE_FILES["AGENTS"])
    if not content:
        return A2ARegistryModel()
    return parse_a2a_registry(content)


def register_a2a_peer(
    name: str,
    url: str,
    description: str = "",
    capabilities: str = "",
    auth: str = "none",
    notes: str = "",
) -> str:
    """Add or update a peer in AGENTS.md table."""
    registry = load_a2a_peers()

    updated = False
    for p in registry.peers:
        if p.name.lower() == name.lower():
            p.url = url
            p.description = description
            p.capabilities = capabilities
            p.auth = auth
            p.notes = notes or datetime.now().strftime("%Y-%m-%d")
            updated = True
            break

    if not updated:
        registry.peers.append(
            A2APeerModel(
                name=name,
                url=url,
                description=description,
                capabilities=capabilities,
                auth=auth,
                notes=notes or datetime.now().strftime("%Y-%m-%d"),
            )
        )

    content = serialize_a2a_registry(registry)
    path = get_workspace_path(CORE_FILES["AGENTS"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"✅ Registered/updated A2A peer '{name}' at {url}"


def list_a2a_peers() -> A2ARegistryModel:
    """List all registered A2A peers."""
    return load_a2a_peers()


def delete_a2a_peer(name: str) -> str:
    """Remove a peer from AGENTS.md registry."""
    registry = load_a2a_peers()
    original_count = len(registry.peers)
    registry.peers = [p for p in registry.peers if p.name.lower() != name.lower()]

    if len(registry.peers) < original_count:
        content = serialize_a2a_registry(registry)
        path = get_workspace_path(CORE_FILES["AGENTS"])
        path.write_text(content, encoding="utf-8")
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
            from .agent_utilities import ensure_package_installed

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
                if spec and spec.origin:
                    import ast

                    with open(spec.origin, "r", encoding="utf-8") as f:
                        tree = ast.parse(f.read())
                        if any(
                            isinstance(node, ast.FunctionDef)
                            and node.name == "agent_template"
                            for node in tree.body
                        ):

                            tag = name.replace("-agent", "").replace("-api", "").lower()
                            agent_descriptions[tag] = package_name
            except Exception as e:
                logger.debug(f"Discovery skipped for {package_name}: {e}")
                pass

    return agent_descriptions
