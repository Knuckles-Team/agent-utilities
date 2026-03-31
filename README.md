# Agent Utilities - Pydantic AI Utilities

![PyPI - Version](https://img.shields.io/pypi/v/agent-utilities)
![MCP Server](https://badge.mcpx.dev?type=server 'MCP Server')
![PyPI - Downloads](https://img.shields.io/pypi/dd/agent-utilities)
![GitHub Repo stars](https://img.shields.io/github/stars/Knuckles-Team/agent-utilities)
![GitHub forks](https://img.shields.io/github/forks/Knuckles-Team/agent-utilities)
![GitHub contributors](https://img.shields.io/github/contributors/Knuckles-Team/agent-utilities)
![PyPI - License](https://img.shields.io/pypi/l/agent-utilities)
![GitHub](https://img.shields.io/github/license/Knuckles-Team/agent-utilities)

![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/Knuckles-Team/agent-utilities)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Knuckles-Team/agent-utilities)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/Knuckles-Team/agent-utilities)
![GitHub issues](https://img.shields.io/github/issues/Knuckles-Team/agent-utilities)

![GitHub top language](https://img.shields.io/github/languages/top/Knuckles-Team/agent-utilities)
![GitHub language count](https://img.shields.io/github/languages/count/Knuckles-Team/agent-utilities)
![GitHub repo size](https://img.shields.io/github/repo-size/Knuckles-Team/agent-utilities)
![GitHub repo file count (file type)](https://img.shields.io/github/directory-file-count/Knuckles-Team/agent-utilities)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/agent-utilities)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/agent-utilities)

*Version: 0.2.36*

## Overview

Agent Utilities provides a robust foundation for building production-ready Pydantic AI Agents. Recently refactored into a high-performance **modular architecture**, it simplifies agent creation, adds advanced **Graph Orchestration**, and provides essential "operating system" tools including state persistence, resilience, and high-fidelity streaming.

## Key Features

- **Agent Creation**: Streamlined `create_agent` function that handles MCP servers, skills, and model configuration automatically.
- **Advanced Graph Orchestration**: Multi-domain routing with `HybridRouterNode` (Rule-based + LLM) and **parallel execution** with `ParallelDomainNode`.
- **Resilience & Accuracy**: Native support for **Error Recovery** (exponential backoff), **State Persistence** (checkpointing), and **Result Validation**.
- **Observability**: Real-time **Graph Streaming** (SSE) and **Lifecycle Hooks** (`on_tool_start`, `on_tool_end`) for distributed telemetry.
- **Typed Foundation**: Zero-config dependency injection using `AgentDeps`.
- **Multi-Agent Support**: Native support for the supervisor pattern, allowing complex tasks to be delegated to specialized child agents.
- **Agent Server**: Built-in FastAPI server with standardized `/mcp`, `/a2a`, `/ag-ui`, and `/stream` (SSE) endpoints.
- **Workspace Management**: Automated management of agent state through standard markdown files (`IDENTITY.md`, `MEMORY.md`, `USER.md`).
- **Lightweight & Lazy**: Core utilities are lightweight. Heavy dependencies are lazy-loaded only when requested via optional extras.

## Architecture & Orchestration

| `adguard-home-agent` | Graph |
| `agent-utilities` | Library | Production-grade Orchestration. Supports Parallel execution, Real-time sub-agent streaming, High-fidelity observability, and Session Resumability |
| `agent-webui` | Library | Cinematic Graph Activity Visualization. |

`agent-utilities` implements a multi-stage execution pipeline using `pydantic-graph` for maximum precision and resilience.

### C4 Container Diagram
```mermaid
C4Container
    title Container diagram for Agent Orchestration System

    Person(user, "User", "Interacts via Web UI")

    Container_Boundary(c1, "Agent Ecosystem") {
        Container(webui, "Agent WebUI", "React, Tailwind", "Renders streaming responses and graph activity visualization")
        Container(gateway, "Agent Gateway (FastAPI)", "Python, Pydantic-AI", "Handles SSE streams, merges graph events into chat annotations")
        Container(orchestrator, "Graph Orchestrator", "Pydantic-Graph", "Routes queries, executes parallel domains, validates results")
        Container(subagent, "Domain Sub-Agents", "Pydantic-AI", "Specialized agents for Git, Web, Cloud, etc.")
    }

    System_Ext(mcp, "MCP Servers", "Contextual tools (GitHub, Slack, etc.)")
    System_Ext(otel, "OpenTelemetry Collector", "Tracing and monitoring")

    Rel(user, webui, "Uses", "HTTPS/WSS")
    Rel(webui, gateway, "Queries", "POST /stream (SSE)")
    Rel(gateway, orchestrator, "Dispatches", "Async Python")
    Rel(orchestrator, subagent, "Delegates", "Parallel Execution")
    Rel(subagent, mcp, "Invokes Tools", "JSON-RPC (stdio/SSE)")
    Rel(orchestrator, otel, "Exports Spans", "OTLP")
```

### Execution Flow
```mermaid
graph TD
  User([User Query]) --> HybridRouter{Hybrid Router}

  subgraph Routing Layer
    HybridRouter -- "Cache/Regex" --> DomainNode
    HybridRouter -- "LLM Classify" --> DomainNode
    HybridRouter -- "Critical Failure" --> Recover[ErrorRecoveryNode]
  end

  subgraph Execution Layer
    DomainNode -- "Isolated Context" --> SubAgent[Sub-Agent Execution]
    SubAgent -- "Live Events" --> SSE((SSE Stream))
  end

  subgraph Validation Layer
    SubAgent --> Validator{Validator}
    Validator -- "Pass" --> EndNode([End with Result])
    Validator -- "Fail / Too Short" --> DomainNode
  end

  subgraph Resilience Layer
    Recover -- "Backoff/Retry" --> HybridRouter
    Checkpoint[(Persistence Store)] -.-> Resume[ResumeNode]
    Resume --> HybridRouter
  end
```


## Installation

```bash
# Core utilities only
pip install agent-utilities

# With full agent support (recommended)
pip install agent-utilities[agent]

# With MCP server support
pip install agent-utilities[mcp]

# With embedding/vector support
pip install agent-utilities[embeddings]
```

## Quick Start

```python
from agent_utilities import create_agent

# Create a simple agent with workspace tools
agent = create_agent(name="MyAgent")

# Or create a multi-agent supervisor
agent = create_agent(
    name="Supervisor",
    agent_definitions=[{"name": "Researcher", "description": "Search the web"}]
)
```
