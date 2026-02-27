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

*Version: 0.2.6*

## Overview

Agent Utilities provides a robust foundation for building production-ready Pydantic AI Agents. It simplifies agent creation, adds multi-agent supervisor patterns, and provides essential "operating system" tools for agents, including workspace management, scheduling, and discovery.

## Key Features

- **Agent Creation**: Streamlined `create_agent` function that handles MCP servers, skills, and model configuration automatically.
- **Multi-Agent Support**: Native support for the supervisor pattern, allowing complex tasks to be delegated to specialized child agents.
- **Agent Server**: Built-in FastAPI server (`create_agent_server`) with SSE support for easy integration into web UIs and A2A networks.
- **Workspace Management**: Automated management of agent state through standard markdown files (`IDENTITY.md`, `MEMORY.md`, `USER.md`).
- **A2A Integration**: Seamless discovery and communication between agents in a distributed network.
- **Periodic Scheduler**: In-memory task scheduler for running background agent jobs.
- **Lightweight & Lazy**: Core utilities are lightweight. Heavy dependencies like FastAPI or LlamaIndex are lazy-loaded only when requested via optional extras.

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
