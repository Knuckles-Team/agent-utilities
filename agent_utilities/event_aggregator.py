#!/usr/bin/python
"""Event Aggregator Module.

This module provides utilities for aggregating scheduling (CRON.md) and
health check (HEARTBEAT.md) data from multiple orchestrated agents. It handles
the discovery of installed agent data directories and the merging of their
diagnostic requirements into a unified workspace.
"""

import importlib.util
import logging
from importlib.resources import as_file, files
from pathlib import Path

logger = logging.getLogger(__name__)


def find_package_data_dir(package_name: str) -> Path | None:
    """Locate the data or configuration directory for an installed package.

    Args:
        package_name: The Python package name to search for.

    Returns:
        The absolute Path to the data directory if found, otherwise None.

    """
    try:
        spec = importlib.util.find_spec(package_name)
        if not spec or not spec.origin:
            return None

        origin_path = Path(spec.origin).resolve()
        # Common patterns:
        # 1. package/agent_data
        # 2. package/agent
        # 3. agents/name-agent/name_agent/agent_data

        candidates = [
            origin_path.parent / "agent_data",
            origin_path.parent / "agent",
            origin_path.parent.parent / "agent_data",
            origin_path.parent.parent / "agent",
        ]

        for candidate in candidates:
            if candidate.is_dir():
                return candidate

        # Try importlib.resources as fallback
        for sub in ["agent_data", "agent"]:
            try:
                pkg_resource_dir = files(package_name) / sub
                if pkg_resource_dir.is_dir():
                    with as_file(pkg_resource_dir) as path:
                        return path.resolve()
            except Exception:
                pass

    except Exception as e:
        logger.debug(f"Failed to find data dir for {package_name}: {e}")

    return None
