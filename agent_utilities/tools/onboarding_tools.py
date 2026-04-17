#!/usr/bin/python
# coding: utf-8
"""Onboarding Tools Module.

This module provides utilities for bootstrapping new agentic workspaces,
detecting project technology stacks, and initializing core metadata files
like IDENTITY.md and MEMORY.md.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from pydantic_ai import RunContext

from ..models import (
    AgentDeps,
    IdentityModel,
    UserModel,
    MCPAgentRegistryModel,
)
from ..workspace import (
    CORE_FILES,
    serialize_identity,
    serialize_user_info,
    serialize_node_registry,
)

logger = logging.getLogger(__name__)


def detect_tech_stack(root: Path) -> Dict[str, Any]:
    """Identify languages, frameworks, and build tools in the repository.

    Args:
        root: The Path to the project root directory.

    Returns:
        A dictionary containing lists of detected languages, frameworks, and tools.

    """
    stack = {"languages": [], "frameworks": [], "tools": []}

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


def scan_for_entry_points(root: Path) -> List[str]:
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
    """Initialize core metadata files by scanning the workspace.

    This tool identifies the project's technical stack and creates missing
    files such as IDENTITY.md, USER.md, and MEMORY.md with initial context.

    Args:
        ctx: The agent run context.

    Returns:
        A summary message describing the bootstrapped files and detected stack.

    """
    root = ctx.deps.workspace_path
    logger.info(f"Bootstrapping project at {root}...")

    stack = detect_tech_stack(root)
    entries = scan_for_entry_points(root)

    written = []

    # 1. IDENTITY.md
    identity_path = root / CORE_FILES["IDENTITY"]
    if not identity_path.exists():
        id_model = IdentityModel(
            name="New Agent",
            role="Technical Specialist",
            emoji="🤖",
            vibe="Professional and efficient",
            system_prompt="You are a helpful coding assistant for this project.",
        )
        identity_path.write_text(serialize_identity(id_model), encoding="utf-8")
        written.append(CORE_FILES["IDENTITY"])

    # 2. USER.md
    user_path = root / CORE_FILES["USER"]
    if not user_path.exists():
        u_model = UserModel(name="The Human", emoji="👤")
        user_path.write_text(serialize_user_info(u_model), encoding="utf-8")
        written.append(CORE_FILES["USER"])

    # 4. NODE_AGENTS.md
    mcp_path = root / CORE_FILES["NODE_AGENTS"]
    if not mcp_path.exists():
        mcp_path.write_text(
            serialize_node_registry(MCPAgentRegistryModel(agents=[], tools=[])),
            encoding="utf-8",
        )
        written.append(CORE_FILES["NODE_AGENTS"])

    # 5. MEMORY.md
    memory_path = root / CORE_FILES["MEMORY"]
    if not memory_path.exists():
        memory_content = "# Project Memory (MEMORY.md)\n\n"
        memory_content += "## Tech Stack Metadata\n"
        memory_content += json.dumps(stack, indent=2) + "\n\n"
        memory_content += "## Architectural Context\n"
        memory_content += f"Initial scan performed on {datetime.now().strftime('%Y-%m-%d')}. Detected {len(entries)} entry points: {', '.join(entries)}.\n"
        memory_path.write_text(memory_content, encoding="utf-8")
        written.append(CORE_FILES["MEMORY"])

    if written:
        return f"Successfully bootstrapped {', '.join(written)} based on detected tech stack: {stack}"
    else:
        return f"Project already has metadata files. Detected tech stack: {stack}"


# Registration list
onboarding_tools = [bootstrap_project]
