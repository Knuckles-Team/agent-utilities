# Guides & Recipes

Task-oriented how-tos across the **79 guides** in this repository — narrower and more hands-on than the [architecture reference](../architecture/index.md), which explains *why* a subsystem is built the way it is. If you're trying to *do* something (deploy, configure, integrate, migrate), start here; if you're trying to *understand* something, start there.

The most load-bearing guides — [Quick Start](quick-start.md), [Installation](installation.md), [Consumption Models](consumption-models.md), and the [Pre-bundled Workflow Skill Suite](kg-skill-suite.md) — stay directly in the site nav; everything else is grouped here.

## Onboarding & first run

Getting from zero to a running agent.

- [Quick Start](quick-start.md)
- [Installation](installation.md)
- [Consumption Models](consumption-models.md)
- [Creating an Agent with Python](creating-an-agent.md)
- [Features](features.md)
- [First Principles Architecture](first-principles.md)
- [Tools Registry](tools.md)

## Deployment shapes & runbooks

Every way to stand this up — profiles, private-repo CI, sovereign/air-gapped, and the runbooks for keeping it running.

- [Deploying agent-utilities](deployment.md)
- [Supported Configurations (the ladder)](deployment-configurations.md)
- [Day-0 Deployment](day0.md)
- [Self-Setup (config-complete)](self-setup.md)
- [Standard Private Repos + CI (per profile)](genesis-private-repos.md)
- [Sovereign / Self-Hosted (+ air-gap path)](sovereign-self-hosted.md)
- [Company Bootstrap Deployment Guide](company-bootstrap.md)
- [Enterprise enablement runbook](enterprise-enablement-runbook.md)
- [Graph Database Deployment & Multi-Backend Guide](graph-db-deployment.md)
- [Safe Redeploy Runbook](redeploy_kg_server.md)
- [Backend parity & deployment-profile testing](backend-parity-and-profile-testing.md)
- [Process Lifecycle Management](process-lifecycle.md)
- [Scalable Frontends — one shared backend, many thin instances](scalable-frontends.md)
- [Single-GPU LLM serving — tuning for extraction throughput](single-gpu-llm-serving.md)

## Configuration, secrets & identity

Wiring the config surface, credentials, and SSO.

- [Configuration](configuration.md)
- [Secrets & Authentication](secrets-auth.md)
- [OAuth 2.0 / OIDC SSO Authentication Guide](oauth_sso.md)
- [Permissions Kernel](permissions-kernel.md)
- [Workspace Manifest (workspace.yml)](workspace-config.md)
- [Hierarchical State Machine (HSM) Infrastructure](hsm.md)

## Knowledge graph & memory

Using and extending the graph itself — schema, retrieval scoring, long-context management, and the native compute engine.

- [Knowledge Graph](knowledge-graph.md)
- [Graph Engine (Authority + Mirrors)](graph_engine.md)
- [KG-Native Orchestration](kg_native_orchestration.md)
- [KG Schema Extensions: Research Assimilation](kg_schema_extensions.md)
- [Universal Knowledge Assimilation Engine](knowledge-assimilation.md)
- [Scoring Methodology & Retrieval Semantics](scoring_methodology.md)
- [Registry Hot Cache](registry-cache.md)
- [Lossless Context Management (LCM) Guide](lcm_memory.md)
- [Recursive Language Models (RLM)](rlm.md)
- [Native Numeric Kernel](numeric-kernel.md)
- [AU native numeric call-site gap report](numeric-kernel-callsite-gaps.md)
- [KV-Cache Layering (vLLM → LMCache → engine)](kvcache-vllm-lmcache.md)

## Orchestration & agent runtime

How agents, skills, and workflows execute — routing, evolution, and the conductor/loop mechanics underneath.

- [Agents & Orchestration](agents.md)
- [Agent Registry](agent-registry.md)
- [Agent OS Architecture Reference](agent-os-architecture.md)
- [Enabling Autonomous Evolution](autonomous-evolution.md)
- [AGI→ASI Implementation Guide](agi-to-asi-implementation.md)
- [Conductor Orchestration](conductor-orchestration.md)
- [Cognitive Scheduler](cognitive-scheduler.md)
- [Centralized Dynamic Tool Selection & Visibility](dynamic-tool-selection.md)
- [Loop Engine — running the self-improvement / research / goal loop](loop-engine.md)
- [Graph-Native Durable Execution](durable-execution.md)
- [Squeeze Evolve: Confidence-Gated Routing & Evolutionary Aggregation](squeeze-evolve-routing.md)
- [Structured Prompts](structured-prompts.md)
- [Spec-Driven Development (SDD) Orchestrator](sdd.md)

## Skill graphs

Building, migrating, and hardening skill graphs.

- [Pre-bundled Workflow Skill Suite](kg-skill-suite.md)
- [Skill-Graph Migration Runbook (legacy → unified KG-2.7 contract)](skill-graph-migration.md)
- [Skill-Graph Migration Plan — updating all existing graphs to the KG-driven format](skill-graph-migration-plan.md)
- [Skill-Graph Acquisition — Robustness Ledger](skill-graph-robustness-ledger.md)

## MCP fleet & tool surface

Building, exposing, and tuning MCP servers and their tool modes.

- [Building MCP Servers & API Wrappers](building-mcp-servers.md)
- [Building Fleet API Clients](building-fleet-api-clients.md)
- [MCP Tool Modes (condensed/verbose/both)](mcp-tool-modes.md)
- [MCP Fleet Auth & Monitoring Runbook](mcp-fleet-auth-and-monitoring-runbook.md)

## Enterprise & external-system integration

Connecting this platform to the enterprise tool landscape.

- [LeanIX EA Integration (mirror/delta/backfeed)](leanix-integration.md)
- [CMDB/ERP Bidirectional (ServiceNow + ERPNext)](cmdb-bidirectional-integration.md)
- [Universal Enterprise Entities](enterprise_entities.md)
- [Enterprise Ingestion Architecture (Hub-and-Spoke)](enterprise_ingestion.md)
- [Enterprise Trust — Epistemic Audit & Compliance](epistemic-audit-compliance.md)
- [X Personal Assistant & Social Content Ingestion Guide](x-assistant.md)

## Observability & operations

Watching the system run and keeping it safe while it does.

- [Langfuse, Usage & Tracing](observability-usage-tracking.md)
- [Secure Jupyter Sandbox](secure-sandbox.md)

## Architecture & design-pattern primers

Lighter-weight companions to the full architecture reference — good starting points before the deep dives.

- [Architecture](architecture.md)
- [Agentic Harness Engineering (AHE) — Architecture](AHE_ARCHITECTURE.md)
- [Capability-Based Architecture Guide](capability_architecture.md)
- [Capabilities (Self-Healing Patterns)](capabilities.md)
- [Agentic Design Patterns — Architecture Alignment](design-patterns-alignment.md)
- [Emergent Architecture](emergent-architecture.md)
- [System Integration Architecture](system_integration.md)
- [Mathematical Foundations & Financial Engineering Reference](mathematical_foundations.md)
- [Multi-Model Registry & Configuration](models.md)
- [AgentSpecs Catalog](agentspec-catalog.md)

## Contributor workflow

Working on agent-utilities itself.

- [Development Guide](development.md)
