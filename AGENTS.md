# AGENTS.md

## Tech Stack & Architecture
- **Language**: Python 3.10+
- **Core Framework**: [Pydantic AI](https://ai.pydantic.dev)
- **Tooling**: `requests`, `pydantic`, `pyyaml`, `python-dotenv`
- **Architecture**: Centered around the `create_agent` factory, which automates workspace initialization, skill discovery via `SkillsToolset`, and MCP server integration.
- **Key Principles**:
    - Functional and modular utility design.
    - Lazy loading for heavy dependencies (FastAPI, LlamaIndex).
    - Standardized workspace management (`IDENTITY.md`, `MEMORY.md`).

## Core Architecture Diagram
```mermaid
graph TD
    User([User Request]) --> Agent[Pydantic AI Agent]
    subgraph AgentUtilities [agent-utilities]
        Agent -- uses --> CA[create_agent]
        CA -- initializes --> ST[SkillsToolset]
        CA -- configures --> MCP[MCP Clients]
        ST -- discovers --> SkillDir[Skill Directories]
        MCP -- connects --> MCPServer[MCP Servers]
    end
    Agent -- executes --> Tool[Agent Tools]
    Tool -- calls --> Skill[Universal Skill]
```

## Commands (run these exactly)
# Development & Quality
ruff check --fix .
ruff format .
pytest

# Installation
pip install -e "."      # Install in editable mode
pip install -e ".[all]" # Install with all optional extras

## Project Structure Quick Reference
- `agent_utilities/agent/` → Agent templates and `IDENTITY.md` definitions.
- `agent_utilities/agent_utilities.py` → Main entry point for `create_agent` and `create_agent_server`.
- `agent_utilities/mcp_utilities.py` → Utilities for FastMCP and MCP tool registration.
- `agent_utilities/base_utilities.py` → Generic helpers for file handling, type conversions, and CLI flags.
- `agent_utilities/tools.py` → Core "OS" tools for agents (read/write, search, list files).
- `agent_utilities/embedding_utilities.py` → Vector DB and embedding integration (LlamaIndex based).

## File Tree
```text
.
├── agent_utilities/
│   ├── agent/                 # Agent templates and definitions
│   ├── agent_utilities.py      # Main entry point factory
│   ├── mcp_utilities.py       # MCP integration helpers
│   ├── base_utilities.py      # Core shared helpers
│   ├── tools.py               # Built-in agent tools
│   ├── embedding_utilities.py # Vector/Embedding utilities
│   ├── api_utilities.py       # Generic API helpers
│   └── models.py              # Shared Pydantic models
├── pyproject.toml
└── README.md
```

## Code Style & Conventions
**Always:**
- Use the `try/except ImportError` guardrail pattern for optional dependencies.
- Use `agent_utilities.base_utilities.to_boolean` for parsing environment variables and CLI flags.
- Support `SSL_VERIFY` environment variable and `--insecure` CLI flag for all network operations.
- Prefer `pathlib.Path` for file path manipulations.

**Good example (Guardrail):**
```python
try:
    from some_external_lib import feature
except ImportError:
    print("Error: Missing 'some_external_lib'. Please install with extras.")
    sys.exit(1)
```

## Dos and Don'ts
**Do:**
- Use `create_agent` for all new agent instances to ensure consistent workspace setup.
- Register tools with descriptive docstrings as they are parsed by the LLM.
- Keep `base_utilities` free of heavy dependencies.

**Don't:**
- Import `fastapi` or `llama_index` at the top level (use lazy imports inside functions or classes).
- Hardcode file paths; use relative paths from the workspace root or environment variables.

## Safety & Boundaries
**Always do:**
- Validate user-provided file paths to prevent traversal attacks.
- Run `ruff` and `pytest` before submitting PRs.

**Ask first:**
- Introducing new top-level dependencies.
- Changes to the `IDENTITY.md` or `MEMORY.md` management logic.

**Never do:**
- Commit API keys or hardcoded secrets.
- Run tests that require external API access without proper mocks or environment configuration.

## When Stuck
- Refer to `agent_utilities.py` for the implementation details of `create_agent`.
- Review `mcp_utilities.py` for how tools are being registered and exposed to MCP.
- Ask for clarification if the multi-agent supervisor logic is unclear.
