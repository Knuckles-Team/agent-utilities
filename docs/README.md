# Agent Utilities Documentation

Welcome to the `agent-utilities` documentation. This index organizes the guides logically to help you navigate the complex ecosystem of features and advanced concepts.

## 🧭 Getting Started

- [**Overview Map**](overview.md) — Start here! The Conceptual Map connecting the 27 core concepts (CONCEPT:ORCH-1.0 to CONCEPT:ECO-4.2).
- [**Creating an Agent**](creating-an-agent.md) — Guide to bootstrapping a new Pydantic AI agent.
- [**Building MCP Servers**](building-mcp-servers.md) — Creating FastMCP servers and wrappers.

## 🏛️ Architecture

- [**Architecture Deep Dive**](architecture.md) — Diagrams, HSM state routing, and protocol adapters.
- [**Features**](features.md) — Summary of the "operating system" features provided to agents.
- [**Hierarchical State Machines**](hsm.md) — Orthogonal regions, entry/exit hooks, and static routing.
- [**Design Patterns**](design-patterns-alignment.md) — AHE and SDD design pattern alignment.

## 🚀 Advanced Concepts

- [**Knowledge Graph**](knowledge-graph.md) — The MAGMA-inspired persistence layer bridging network topology and Cypher.
- [**Spec-Driven Development (SDD)**](sdd.md) — Transform `.specify` definitions into parallel execution plans.
- [**RLM / REPL**](rlm.md) — Recursive Language Model execution patterns and autonomous loops.
- [**Agentic Harness Engineering (AHE)**](AHE_ARCHITECTURE.md) — Deep traceability and prompt evolution mechanisms.
- [**Emergent Architecture**](emergent-architecture.md) — OGM, Swarm, Variant Selection, Self-Model, and Global Workspace Attention (CONCEPT:KG-2.0 to CONCEPT:ORCH-1.2).

## 🔬 First Principles Architecture

- [**First Principles Overview**](first-principles.md) — Registry Hot Cache, TeamConfig Promotion, AgentCapability System, and A2A PlannerGraphSkill (CONCEPT:ORCH-1.2 to CONCEPT:ECO-4.2).
- [**Registry Cache Deep-Dive**](registry-cache.md) — Session-scoped O(1) specialist lookups with event-driven invalidation.
- [**Process Lifecycle Management**](process-lifecycle.md) — Sidecar cleanup, signal handling, and child process management.

## 📖 Reference Guides

- [**Configuration**](configuration.md) — Unified reference for all environment variables, config files, and CLI flags.
- [**Secrets & Authentication**](secrets-auth.md) — Authlib JWT integration and Vault backends.
- [**Models & Routing**](models.md) — Multi-model registries, routing tiers, and `MODELS_CONFIG`.
- [**Tools Registry**](tools.md) — Catalog of the 18 tool modules available to specialists.
- [**Structured Prompts**](structured-prompts.md) — JSON schema blueprints replacing unstructured Markdown prompts.
- [**Capabilities**](capabilities.md) — Self-healing, circuit breakers, checkpointing, capability auto-activation, and team dispatch.
- [**Agents & Orchestration**](agents.md) — Specialist registry, MCP loading, event system, governance.
- [**Development Guide**](development.md) — Contribution standards and troubleshooting.
