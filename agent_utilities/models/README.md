# Agentic Models

This directory contains the Pydantic models that define the data structures for the entire ecosystem.

## Overview

Models are the source of truth for communication between agents, protocols, and frontends. They ensure type safety and schema consistency.

## Key Models

- **AgentDeps (`agent.py`)**: The primary dependency injection container for agents.
- **Spec / Task / ImplementationPlan (`sdd.py`)**: Models for Spec-Driven Development.
- **GraphState (`graph.py`)**: Shared state for `pydantic-graph` orchestration.
- **MCP / A2A Models (`mcp.py`, `a2a.py`)**: Protocol-specific message and registry models.
- **KG Models (`knowledge_graph.py`)**: Node and edge definitions for the Knowledge Graph.

## Design Principles

1. **Immutability**: Prefer immutable models where possible.
2. **Serialization**: Ensure all models are JSON-serializable for transport over ACP/SSE.
3. **Validation**: Use Pydantic's built-in validation to enforce data integrity at the boundaries.
4. **Composition**: Build complex models by composing smaller, atomic schemas.

## Maintenance

- **Breaking Changes**: Since these models define the protocol, avoid breaking changes or provide explicit migration paths.
- **Schema Export**: Models are used to generate OpenAPI specs for the FastAPI server. Ensure docstrings are descriptive.
