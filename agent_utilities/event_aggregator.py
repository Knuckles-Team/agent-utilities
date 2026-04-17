#!/usr/bin/python
# coding: utf-8
"""Event Aggregator Module.

This module provides utilities for aggregating scheduling (CRON.md) and
health check (HEARTBEAT.md) data from multiple orchestrated agents. It handles
the discovery of installed agent data directories and the merging of their
diagnostic requirements into a unified workspace.
"""

import logging
from pathlib import Path
from typing import List, Optional
import importlib.util
from importlib.resources import files, as_file

from .workspace import (
    parse_cron_registry,
    serialize_cron_registry,
    CronRegistryModel,
    get_agent_workspace,
)
from .discovery import discover_agents

logger = logging.getLogger(__name__)


def find_package_data_dir(package_name: str) -> Optional[Path]:
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


def extract_heartbeat_checks(heartbeat_path: Path) -> List[str]:
    """Extract individual check items from a HEARTBEAT.md file.

    Args:
        heartbeat_path: Path to the HEARTBEAT.md markdown file.

    Returns:
        A list of check strings (bullet points) extracted from the file.

    """
    if not heartbeat_path.exists():
        return []

    content = heartbeat_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    checks = []
    in_checks = False

    for line in lines:
        s_line = line.strip()
        if s_line.startswith("## ") and any(
            x in s_line for x in ["Checks", "Pulse", "Self-Check"]
        ):
            in_checks = True
            continue
        if in_checks:
            if s_line.startswith("##") and not any(
                x in s_line for x in ["Checks", "Pulse"]
            ):
                in_checks = False
                continue
            if s_line.startswith(
                ("- ", "* ", "1. ", "2. ", "3. ", "4. ", "5. ", "6. ")
            ):
                checks.append(s_line)
    return checks


def aggregate_orchestrated_data(target_package: str, skip_heartbeats: bool = True):
    """Aggregate diagnostic data from all discovered agents.

    Merges CRON.md and HEARTBEAT.md files from sub-agents into the target
    package's workspace. This ensures the master orchestrator can manage
    the lifecycle and health of the entire ecosystem.

    Args:
        target_package: The name of the master agent package.
        skip_heartbeats: Whether to exclude heartbeat-specific tasks from CRON.md.

    """
    logger.info(f"Starting runtime event aggregation for {target_package}")

    target_data_dir = get_agent_workspace()
    if not target_data_dir.exists():
        logger.error(f"Target data directory not found: {target_data_dir}")
        return

    # Discover installed agents
    agents = discover_agents()
    logger.info(f"Discovered {len(agents)} agents for aggregation.")

    all_tasks = []
    all_heartbeats = {}

    # 1. Load target agent's own tasks first to preserve them
    cron_path = target_data_dir / "CRON.md"
    if cron_path.exists():
        try:
            content = cron_path.read_text(encoding="utf-8")
            reg = parse_cron_registry(content)
            all_tasks.extend(reg.tasks)
        except Exception as e:
            logger.error(f"Failed to parse target CRON.md: {e}")

    # 2. Load target agent's own heartbeats
    heartbeat_path = target_data_dir / "HEARTBEAT.md"
    if heartbeat_path.exists():
        all_heartbeats[target_package] = extract_heartbeat_checks(heartbeat_path)

    for tag, package_name in agents.items():
        # Avoid aggregating self
        if package_name == target_package.replace("-", "_"):
            continue

        data_dir = find_package_data_dir(package_name)
        if not data_dir:
            logger.debug(f"Could not find data dir for agent: {package_name}")
            continue

        # Handle CRON.md
        sub_cron_path = data_dir / "CRON.md"
        if sub_cron_path.exists():
            try:
                content = sub_cron_path.read_text(encoding="utf-8")
                reg = parse_cron_registry(content)
                for task in reg.tasks:
                    # Filtering: skip heartbeat tasks from sub-agents if requested
                    if skip_heartbeats and "heartbeat" in task.id.lower():
                        continue

                    new_id = f"{tag}_{task.id}".replace("-", "_")
                    new_name = f"[{tag}] {task.name}"

                    if not any(t.id == new_id for t in all_tasks):
                        task.id = new_id
                        task.name = new_name
                        all_tasks.append(task)
            except Exception as e:
                logger.warning(f"Failed to parse CRON.md for {package_name}: {e}")

        # Handle HEARTBEAT.md
        sub_heartbeat_path = data_dir / "HEARTBEAT.md"
        if sub_heartbeat_path.exists():
            checks = extract_heartbeat_checks(sub_heartbeat_path)
            if checks:
                all_heartbeats[tag] = checks

    # Write merged CRON.md
    if all_tasks:
        # Keep base tasks (those without tags) first, then sort others
        base_tasks = [
            t
            for t in all_tasks
            if "_" not in t.id or not any(t.id.startswith(f"{tg}_") for tg in agents)
        ]
        other_tasks = [t for t in all_tasks if t not in base_tasks]
        other_tasks.sort(key=lambda t: t.id)

        final_reg = CronRegistryModel(tasks=base_tasks + other_tasks)
        cron_path.write_text(serialize_cron_registry(final_reg), encoding="utf-8")
        logger.info(f"Updated CRON.md with {len(all_tasks)} tasks.")

    # Write merged HEARTBEAT.md
    if all_heartbeats:
        heartbeat_lines = [
            "# Heartbeat — Periodic Self-Check",
            "",
            "You are running a scheduled heartbeat. Perform these checks and report results concisely.",
            "",
            "## Core Checks",
            "",
        ]

        if target_package in all_heartbeats:
            for check in all_heartbeats[target_package]:
                heartbeat_lines.append(check)
            del all_heartbeats[target_package]

        heartbeat_lines.append("")
        heartbeat_lines.append("## Orchestrated Agent Checks")
        heartbeat_lines.append("")

        sorted_tags = sorted(all_heartbeats.keys())
        for tag in sorted_tags:
            checks = all_heartbeats[tag]
            heartbeat_lines.append(f"### {tag.replace('-', ' ').title()}")
            for check in checks:
                heartbeat_lines.append(check)
            heartbeat_lines.append("")

        heartbeat_lines.append("## Response Format")
        heartbeat_lines.append("")
        heartbeat_lines.append("### If everything is healthy:")
        heartbeat_lines.append("```")
        heartbeat_lines.append(
            "HEARTBEAT_OK — All systems nominal. Orchestrated sub-agents verified."
        )
        heartbeat_lines.append("```")
        heartbeat_lines.append("")
        heartbeat_lines.append("### If issues found:")
        heartbeat_lines.append("```")
        heartbeat_lines.append("HEARTBEAT_ALERT — [summary of issues found]")
        heartbeat_lines.append("- Agent name: [issue description]")
        heartbeat_lines.append("```")

        heartbeat_path.write_text("\n".join(heartbeat_lines), encoding="utf-8")
        logger.info(
            f"Updated HEARTBEAT.md with checks from {len(all_heartbeats)} sub-agents."
        )
