# Creating an Agent with Python

This guide walks you through creating a production-ready AI agent using `agent-utilities`. The canonical reference implementation is [`genius-agent/agent_server.py`](../../agents/genius-agent/genius_agent/agent_server.py).

## Prerequisites

- Python 3.11+
- `agent-utilities[agent]` installed (includes `pydantic-ai`, `fastmcp`, `universal-skills`, `skill-graphs`)
- An LLM endpoint (local LM Studio, OpenAI, Anthropic, etc.)

## Project Structure

Every agent in the ecosystem follows the same layout:

```
my-agent/
├── my_agent/
│   ├── __init__.py
│   ├── __main__.py          # Entry point: from .agent_server import agent_server; agent_server()
│   ├── agent_server.py      # Agent server bootstrap (see below)
│   ├── main_agent.json      # Agent identity (name, description, system prompt)
│   └── mcp_config.json      # MCP server configuration
├── pyproject.toml
├── .env.example
└── README.md
```

## Step 1: Create `main_agent.json`

This JSON file defines your agent's identity. It is loaded by `load_identity()`:

```json
{
  "name": "My Agent",
  "type": "prompt",
  "description": "My custom AI agent for X operations.",
  "capabilities": ["task-automation", "data-analysis"],
  "tags": ["custom"],
  "content": "You are My Agent. You specialize in X operations and can help users with Y and Z."
}
```

## Step 2: Create `agent_server.py`

This is the heart of your agent. Here is the reference pattern used by all ecosystem agents:

```python
#!/usr/bin/python
import logging
import os
import sys
import warnings

# Filter RequestsDependencyWarning early to prevent log spam
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from requests.exceptions import RequestsDependencyWarning
        warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
    except ImportError:
        pass

warnings.filterwarnings("ignore", message=".*urllib3.*or chardet.*")
warnings.filterwarnings("ignore", message=".*urllib3.*or charset_normalizer.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="fastmcp")

from agent_utilities import (
    build_system_prompt_from_workspace,
    create_agent_parser,
    create_graph_agent_server,
    initialize_workspace,
    load_identity,
)

__version__ = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Initialize workspace and load identity
initialize_workspace()
meta = load_identity()

DEFAULT_AGENT_NAME = os.getenv("DEFAULT_AGENT_NAME", meta.get("name", "My Agent"))
DEFAULT_AGENT_DESCRIPTION = os.getenv(
    "AGENT_DESCRIPTION",
    meta.get("description", "My custom AI agent."),
)
DEFAULT_AGENT_SYSTEM_PROMPT = os.getenv(
    "AGENT_SYSTEM_PROMPT",
    meta.get("content") or build_system_prompt_from_workspace(),
)


def agent_server():
    print(f"{DEFAULT_AGENT_NAME} v{__version__}", file=sys.stderr)
    logger.info("Application startup complete")

    parser = create_agent_parser()
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    create_graph_agent_server(
        mcp_url=args.mcp_url,
        mcp_config=args.mcp_config or "mcp_config.json",
        host=args.host,
        port=args.port,
        provider=args.provider,
        model_id=args.model_id,
        router_model=args.model_id,
        agent_model=args.model_id,
        base_url=args.base_url,
        api_key=args.api_key,
        custom_skills_directory=args.custom_skills_directory,
        enable_web_ui=args.web,
        enable_otel=args.otel,
        otel_endpoint=args.otel_endpoint,
        otel_headers=args.otel_headers,
        otel_public_key=args.otel_public_key,
        otel_secret_key=args.otel_secret_key,
        otel_protocol=args.otel_protocol,
        debug=args.debug,
    )


if __name__ == "__main__":
    agent_server()
```

### Key Functions Explained

| Function | Purpose |
|---|---|
| `initialize_workspace()` | Discovers and sets the workspace root directory |
| `load_identity()` | Loads `main_agent.json` from the package directory |
| `build_system_prompt_from_workspace()` | Constructs a system prompt from `AGENTS.md`, README, etc. |
| `create_agent_parser()` | Creates a standardized CLI argument parser with provider, model, host, port, MCP config, etc. |
| `create_graph_agent_server()` | Bootstraps the full graph agent server with ACP, A2A, AG-UI endpoints, knowledge graph, and MCP tool loading |

## Step 3: Configure `mcp_config.json`

List the MCP servers your agent should connect to. Environment variables are automatically expanded:

```json
{
  "mcpServers": {
    "my-service": {
      "command": "uv",
      "args": ["run", "my-service-mcp", "-t", "stdio"],
      "env": {
        "SERVICE_URL": "${SERVICE_URL}",
        "SERVICE_TOKEN": "${SERVICE_TOKEN}"
      }
    }
  }
}
```

## Step 4: Set Up `pyproject.toml`

```toml
[project]
name = "my-agent"
version = "1.0.0"
requires-python = ">=3.11,<3.14"
dependencies = [
    "agent-utilities[agent]>=0.2.40",
]

[project.scripts]
my-agent = "my_agent.agent_server:agent_server"
my-agent-mcp = "my_agent.mcp_server:mcp_server"
```

## Step 5: Run Your Agent

```bash
# Development
uv run my-agent --provider openai --model-id llama-3.2-3b-instruct --base-url http://localhost:1234/v1 --debug

# With web UI
uv run my-agent --web --port 8080

# Using environment variables
export PROVIDER=openai
export MODEL_ID=gpt-4o-mini
export LLM_API_KEY=sk-...
uv run my-agent
```

### CLI Flags Reference

All agents inherit these flags from `create_agent_parser()`:

| Flag | Description | Default |
|---|---|---|
| `--provider` | LLM provider | `openai` |
| `--model-id` | Model identifier | `llama-3.2-3b-instruct` |
| `--base-url` | LLM API base URL | `http://host.docker.internal:1234/v1` |
| `--api-key` | LLM API key | `llama` |
| `--host` | Server bind address | `0.0.0.0` |
| `--port` | Server port | `8000` |
| `--mcp-config` | MCP config file path | `mcp_config.json` |
| `--mcp-url` | Direct MCP server URL | None |
| `--web` | Enable embedded web UI | `False` |
| `--debug` | Enable debug logging | `False` |
| `--otel` | Enable OpenTelemetry | `False` |
| `--custom-skills-directory` | Custom skills directory | None |

## Next Steps

- Add MCP tools to your agent: see [Building MCP Servers](building-mcp-servers.md)
- Add custom skills: place `.md` files in a directory and pass `--custom-skills-directory`
- Deploy with Docker: see your agent's `Dockerfile` and `compose.yaml`
- Learn about the graph orchestration pipeline: see [Architecture](architecture.md)
