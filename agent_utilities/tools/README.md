# Agentic Tools

This directory contains the atomic tools used by agents across the ecosystem.

## Overview

Tools are grouped by domain and are designed to be easily discovered and bound to specialized agents. Most tools are Pydantic-native and leverage `AgentDeps` for state and context.

## Tool Categories

- **Developer Tools (`developer_tools.py`)**: File operations, terminal execution, and codebase manipulation.
- **Knowledge Tools (`knowledge_tools.py`)**: Deep integration with the Knowledge Graph (LadybugDB/Neo4j).
- **SDD Tools (`sdd_tools.py`)**: Tools for managing Specs, Tasks, and Implementation Plans.
- **Git Tools (`git_tools.py`)**: Branch management, commits, and diffing.
- **Pattern Tools (`pattern_tools.py`)**: Entry points for high-level patterns like TDD and manual testing.
- **Team Tools (`team_tools.py`)**: Coordination and messaging between agents.
- **Self-Improvement Tools (`self_improvement_tools.py`)**: Feedback capture and prompt evolution.

## Browser Tools

The `browser/` subdirectory contains specialized tools for web automation using Playwright/Rodney.

## How to Add a Tool

1. Create a new file or add to an existing domain file.
2. Define the tool using `@tool` (if using pydantic-ai directly) or add it to the `AgentDeps` tool registry.
3. Ensure the tool is self-documenting with clear docstrings and type hints.
4. Add unit tests in `tests/unit/test_tools_logic.py` or a specific domain test file.

## Maintenance

- **Security**: Always use `TOOL_GUARD_MODE` for destructive or sensitive operations.
- **Performance**: Tools should be non-blocking where possible. Use `asyncio` for I/O bound tasks.
- **Schema**: Ensure tool inputs and outputs remain stable or follow semantic versioning.
