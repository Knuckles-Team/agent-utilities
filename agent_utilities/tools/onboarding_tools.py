#!/usr/bin/python
"""Onboarding Tools Module.

This module provides utilities for bootstrapping new agentic workspaces,
detecting project technology stacks, and initializing core configuration files
like ``main_agent.json``.
"""

import logging
from pathlib import Path
from typing import Any

from pydantic_ai import RunContext

from ..models import (
    AgentDeps,
)

logger = logging.getLogger(__name__)


def detect_tech_stack(root: Path) -> dict[str, Any]:
    """Identify languages, frameworks, and build tools in the repository.

    Args:
        root: The Path to the project root directory.

    Returns:
        A dictionary containing lists of detected languages, frameworks, and tools.

    """
    stack: dict[str, list[str]] = {"languages": [], "frameworks": [], "tools": []}

    # Check for specific markers
    markers = {
        "package.json": ("JavaScript/TypeScript", "Node.js"),
        "pyproject.toml": ("Python", "Poetry/Base"),
        "requirements.txt": ("Python", "Pip"),
        "go.mod": ("Go", "Go Modules"),
        "Cargo.toml": ("Rust", "Cargo"),
        "composer.json": ("PHP", "Composer"),
        "Gemfile": ("Ruby", "Bundler"),
        "pom.xml": ("Java", "Maven"),
        "build.gradle": ("Java/Kotlin", "Gradle"),
        "Dockerfile": (None, "Docker"),
        "compose.yaml": (None, "Docker Compose"),
        "docker-compose.yml": (None, "Docker Compose"),
        "next.config.js": (None, "Next.js"),
        "vite.config.ts": (None, "Vite"),
        "tailwind.config.js": (None, "Tailwind CSS"),
    }

    found_files = [f.name for f in root.iterdir() if f.is_file()]

    for filename, (lang, tool) in markers.items():
        if filename in found_files:
            if lang and lang not in stack["languages"]:
                stack["languages"].append(lang)
            if tool and tool not in stack["tools"]:
                stack["tools"].append(tool)

    return stack


def scan_for_entry_points(root: Path) -> list[str]:
    """Identify potential main interaction points and entry scripts.

    Args:
        root: The Path to the project root directory.

    Returns:
        A list of relative paths to detected entry point files.

    """
    entry_patterns = [
        "main.py",
        "app.py",
        "server.py",
        "index.ts",
        "index.js",
        "server.js",
        "main.go",
        "src/main.rs",
        "manage.py",
    ]

    found = []
    for pattern in entry_patterns:
        # Check root and one level deep
        if (root / pattern).exists():
            found.append(pattern)
        else:
            # Check for pattern in subdirs (e.g. src/index.ts)
            for item in root.iterdir():
                if (
                    item.is_dir()
                    and not item.name.startswith(".")
                    and (item / pattern).exists()
                ):
                    found.append(f"{item.name}/{pattern}")

    return found


async def bootstrap_project(ctx: RunContext[AgentDeps]) -> str:
    """Identify the project's technical stack and entry points.

    Args:
        ctx: The agent run context.

    Returns:
        A summary message describing the detected stack.

    """
    root = ctx.deps.workspace_path
    logger.info(f"Analyzing project at {root}...")

    stack = detect_tech_stack(root)
    entry_points = scan_for_entry_points(root)

    return (
        f"Project analysis complete.\n"
        f"Detected Languages/Tools: {', '.join(stack['languages'] + stack['tools'])}\n"
        f"Potential Entry Points: {', '.join(entry_points)}"
    )


# Registration list
onboarding_tools = [bootstrap_project]
