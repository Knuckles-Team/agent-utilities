#!/usr/bin/python
"""Memory Management Tools.

This module provides tools for managing project memory via AGENTS.md,
replacing the traditional CLAUDE.md.
"""

from pathlib import Path

from pydantic_ai import RunContext

from ..models import AgentDeps


async def read_agents_md(ctx: RunContext[AgentDeps]) -> str:
    """Read the content of AGENTS.md from the workspace root.

    AGENTS.md contains project-specific instructions, build commands,
    test commands, and style guidelines.
    """
    root = Path(ctx.deps.workspace_path)
    paths = [
        root / "AGENTS.md",
        root / ".agents" / "AGENTS.md",
        root / "AGENTS.local.md",
    ]

    content = []
    for p in paths:
        if p.exists():
            content.append(f"--- Content from {p.name} ---\n{p.read_text()}")

    if not content:
        return "No AGENTS.md found in workspace root."

    return "\n\n".join(content)


async def update_agents_md(
    ctx: RunContext[AgentDeps], content: str, filename: str = "AGENTS.md"
) -> str:
    """Update the content of AGENTS.md or AGENTS.local.md.

    Use this to persist important project rules, discovered commands, or style notes.
    """
    root = Path(ctx.deps.workspace_path)
    if filename not in ["AGENTS.md", "AGENTS.local.md"]:
        return "Error: Can only update AGENTS.md or AGENTS.local.md"

    path = root / filename
    path.write_text(content)
    return f"Successfully updated {filename}"


async def init_agents_md(ctx: RunContext[AgentDeps]) -> str:
    """Initialize a new AGENTS.md file with standard templates.

    This replaces the '/init' functionality from Claude Code.
    """
    root = Path(ctx.deps.workspace_path)
    path = root / "AGENTS.md"

    if path.exists():
        return "AGENTS.md already exists."

    template = """# Project: [Name]

## Build Commands
- Build: `npm run build` or `make`

## Test Commands
- Run all tests: `pytest` or `npm test`
- Run specific test: `pytest path/to/test.py`

## Style Guidelines
- Language: Python / TypeScript
- Linting: ruff / eslint
- Patterns: Use structured prompting and Knowledge Graph integration.

## Useful Commands
- Local dev: `npm run dev`
"""
    path.write_text(template)
    return f"Initialized AGENTS.md at {path}"


memory_tools = [
    read_agents_md,
    update_agents_md,
    init_agents_md,
]
