# Architecture Reference

This is the curated entry point to the **141 architecture documents** in this repository — deep dives on one subsystem, one seam, or one hardening incident each. Start with **North-Star Architecture** for the whole-program view, then drop into whichever group matches what you're touching. Every file here is also reachable through MkDocs Material's site search if you already know its name.

> New to the codebase? [`docs/pillars/index.md`](../pillars/index.md) is the 5-pillar concept map these documents implement; this page is the implementation detail underneath it.

## Start here — whole-program view

The north star, the operating model, and the standards every other document here assumes you've read.

- [North-Star Architecture (the whole program)](north-star-architecture.md)
- [Delegation-First Operating Model (orchestrate + resolve)](delegation-first-operating-model.md)
- [agent-utilities-expert (KG-bound delegate)](agent-utilities-expert.md)
- [Epistemic Operations Protocol](https://knuckles-team.github.io/epistemic-graph/architecture/epistemic-operations-protocol/)
- [Empirical Development Standards — the incidents behind the rules](empirical-development-standards.md)
- [Troubleshooting (cross-layer diagnose)](troubleshooting.md)

## Identity, session & authority

Who is acting, on whose behalf, with what authority — from the first request to a delegated sub-agent.

- [Graph Authority Convergence (session, client, work state)](graph-authority-convergence.md)
- [Per-Agent On-Behalf-Of Identity (delegation chain, ceiling, revocation)](per-agent-delegation.md)
- [Verified Identity Carrier Contract (GOC-15)](verified-identity-carrier-contract.md)
- [IdP-Agnostic Role Inheritance & Identity-Scoped Auto-Load](https://knuckles-team.github.io/graph-os/architecture/identity-inheritance/)
- [MCP Fleet Authentication (JWT + Eunomia)](https://knuckles-team.github.io/graph-os/architecture/mcp-auth/)
- [Engine-authoritative cluster discovery and session continuity](cluster-discovery-session-continuity.md)
- [Concept hierarchy](concept-hierarchy.md)
- [Cross-host concept reservation authority](concept-reservation-authority.md)
- [ACL Registration Convergence (write-time + read-time defence in depth)](acl_registration_convergence.md)

## Control-plane suite

The remaining runtime authority and resource-management seams.
- [Relational Authority & Registry Read Model](relational-authority.md)
- [Resource-pool capability authority](resource-pool-authority.md)
- [Native WorkItem resource-reservation boundary](native-resource-reservation.md)
- [Fleet scale authority](https://knuckles-team.github.io/graph-os/architecture/fleet-scale-authority/)
- [Durable ScaleIntent actuation](scaling-intent-actuation.md)
- [Service-surface scale units](service-scale-units.md)

## Ingestion & data preparation

Getting external and internal signal into the graph — classification, commit-history tails, Arrow-based prep, and stability under load.

- [Intelligent Ingestion (classify/commit-history/tail)](intelligent-ingestion.md)
- [Content-Aware Ingestion (ArchiveBox/crawl4ai/scholarx)](https://knuckles-team.github.io/agent-connector-sdk/architecture/content-aware-ingestion/)
- [Real Data-Preparation Acceptance (NE-115)](data-prep-acceptance.md)
- [Arrow Data Preparation & Profiling](data-prep-arrow-kernel.md)
- [Optional Operator Data-Quality Certification](data-quality-certification.md)
- [Ingestion Throughput (lanes, tick collapse, bulk)](ingestion_throughput.md)
- [KG Connectors, Ingestors & Enrichers](https://knuckles-team.github.io/agent-connector-sdk/architecture/connectors-and-ingestion/)
- [KG as Bidirectional ETL Hub](https://knuckles-team.github.io/agent-connector-sdk/architecture/etl-hub/)
- [Document → KG Fact Extraction](document_fact_extraction.md)
- [Knowledge Graph Ingestion Stability & Locking Architecture](knowledge_graph_ingestion_stability.md)
- [Skill-Workflow → Knowledge-Graph Ingestion](skill_workflow_ingestion.md)
- [Knowledge Graph Ingestion — Concept Extraction Standards](concept_extraction_standards.md)

## Ontology & semantic layer

OWL/RDF, SHACL, argumentation, and object-centric process semantics — the reasoning layer over the raw graph.

- [OWL/RDF Layer (local, always-on)](owl_rdf_layer.md)
- [Ontology System](ontology_system.md)
- [Ontology Library (catalog + anti-drift)](https://knuckles-team.github.io/epistemic-graph/architecture/ontology-library/)
- [Ontology Federation & Package Migration (domain ttls in owning repos)](https://knuckles-team.github.io/epistemic-graph/architecture/ontology-federation/)
- [Ontology-Guided Ingestion & Entity Resolution (exceeds sift-kg)](ontology-guided-ingestion.md)
- [Ontology integrity-policy activation](ontology-integrity-activation.md)
- [Ontology-native classification — full handoff (Phase A → checkpoint → Phase B)](ontology-native-classification.md)
- [Vendor-Neutral Enterprise Ontology](vendor_neutral_enterprise_ontology.md)
- [AIF Argumentation (I-nodes/S-nodes → Dung acceptability)](https://knuckles-team.github.io/epistemic-graph/architecture/aif-argumentation/)
- [Governed JSON-OCEL exchange](governed_ocel.md)
- [Incremental Object-Centric Derivation + Conformance](object_centric_derivation_and_conformance.md)
- [Dynamic graph construction](dynamic-graph-construction.md)
- [Governed retrieval](governed_retrieval.md)
- [Shortcut-Resistant Search-Task Synthesis](shortcut_resistant_search_synthesis.md)

## Memory, reasoning & cognition

How the system remembers, attends, and reasons across turns and agents.

- [Latent-Native Memory](latent_native_memory.md)
- [Self-Improving Reasoning Substrate](self_improving_reasoning_substrate.md)
- [Reasoning Algorithms as Versioned Graph Topologies (CoT/ToT/GoT/ReAct/RAP)](reasoning-graph-topologies.md)
- [Global Workspace Attention](global_workspace_attention.md)
- [Perspectival Inquiry (STORM, native)](perspectival_inquiry.md)
- [Multi-Agent Social System](multi_agent_social_system.md)
- [Runtime Org Dynamics (recruiter, work-item DAG, self-grown staff)](org-runtime.md)

## Evolution, evaluation & learning loops

The self-improvement flywheel: propose, evaluate, distill, and the evidence trail that grounds it.

- [Self-Evolution Flywheel (transparent + steerable)](self-evolution-flywheel.md)
- [The Evolvable Surface (native programs)](evolvable_surface.md)
- [Failure-Driven Evolution](failure_driven_evolution.md)
- [Knowledge Distillation → Skill-Graphs](knowledge_distillation_skill_graphs.md)
- [Multi-Source Assimilation Program](multi_source_assimilation.md)
- [Graph-Native Assimilation Engine](assimilation_engine.md)
- [Model Registry as Graph Resources + Routing Provenance](model_registry_graph_resources.md)
- [Harness Foundry (surpass HarnessX)](harness_foundry.md)
- [Agent-Operator Program (closing the loops)](agent-operator-program.md)
- [Evidence Spine (Artifact → addressable Fragment)](evidence-spine.md)
- [Evidence-Spine Convergence (Seam 2)](evidence_spine_convergence.md)
- [Epistemic-columns currency (Seam 1) — consuming epistemic-graph's `KnowledgeBatch`](epistemic-columns-currency.md)

## Orchestration, execution & routing

How a request becomes a plan, a plan becomes tool calls, and results come back through one entrypoint.

- [Orchestration Execution Seam (ingested capability → executed)](orchestration-execution-seam.md)
- [Non-Blocking Hierarchical Execution](non-blocking-execution.md)
- [Unified Agent Entrypoint (verified routing map)](unified-agent-entrypoint.md)
- [Entrypoint Unification (one orchestrator)](entrypoint-unification.md)
- [Intent Surface (Seam 8 — condensed tool-surface collapse)](intent-surface.md)
- [Skills-over-MCP (unified capability space)](skills_over_mcp.md)
- [Edit-Application Engine](edit_application_engine.md)
- [Chunked Async Drain (full re-ingest, non-blocking)](https://knuckles-team.github.io/agent-connector-sdk/architecture/chunked-async-drain/)
- [Governed Warm-Fork Sandboxes](warm-fork-sandboxes.md)
- [Agents-as-Data Activation (dormant rows, worker pool, scale proof)](agents-as-data-activation.md)
- [Queue-Driven Agent Dispatch](agent_dispatch.md)
- [Durable Execution (unified plane, supersedes restate)](durable-execution.md)
- [Event Sourcing and Query Routing Architecture](event_sourcing_and_routing.md)
- [Layered Hybrid Architecture — KG Comparative Analysis Pipeline](layered_analysis_architecture.md)

## Enterprise system integrations

Bridges into the enterprise tool landscape — BPM, GRC, and the runtime that carries the organization's own intelligence.

- [Camunda + ARIS ↔ Knowledge Graph](https://knuckles-team.github.io/agent-connector-sdk/architecture/camunda-aris-integration/)
- [CISO Assistant ↔ Knowledge Graph](https://knuckles-team.github.io/agent-connector-sdk/architecture/ciso-assistant-integration/)
- [Company Brain Runtime](company_brain_runtime.md)
- [Enterprise Parity, Supervisory Plane & Durable Execution](enterprise_supervisory_and_parity.md)

## Ecosystem, communication & UI surface

How the platform talks to the outside world — cross-session messaging, reactions, frontend contribution points, and code intelligence.

- [Agent Communication Bus (cross-session/host/provider)](https://knuckles-team.github.io/graph-os/architecture/agent-bus/)
- [Messaging Reach (Telegram + agents)](messaging_reach.md)
- [Secure Messaging Ingress (zero open ports)](messaging_security.md)
- [Reactions / Emotes (system-wide, renderer contract)](reactions.md)
- [FrontendContribution.v1 (package-authored WebUI descriptors)](frontend-contributions.md)
- [Modular Prompt & Skill Contribution (fleet entry-points)](modular-prompt-skill-contribution.md)
- [Agentic Resource Discovery (ARD) interop (publish + consume + federate)](ard-interop.md)
- [Codebase Context via the KG (query, don't grep)](codebase-context.md)
- [Code Intelligence (type/scope-resolved calls)](https://knuckles-team.github.io/epistemic-graph/architecture/code-intelligence/)

## Scaling, sharding & the engine surface

Everything about running more than one of something — engine shards, GPUs, gateways, caches, tenants, and the wire protocols between them.

- [Authoritative Engine Placement & Sharding](https://knuckles-team.github.io/graph-os/architecture/engine-sharding/)
- graph-os Horizontal Scaling (the HPA blocker, precisely) — described the superseded
  `out-of-process-shared` (Shape 2) topology; condensed into the
  ["History" section](https://knuckles-team.github.io/graph-os/architecture/graphos-self-hosting/#history-the-horizontal-scaling-shape-this-cutover-replaced)
  of graph-os's self-hosting cutover doc, not carried forward as a standalone page.
- [Distributed Multi-GPU Concurrency](distributed_gpu_concurrency.md)
- [Adaptive Model Concurrency (vLLM auto-scale)](adaptive_model_concurrency.md)
- [Scaling the Gateway](https://knuckles-team.github.io/graph-os/architecture/gateway-scaling/)
- [Resource-Priority Edict (interactive over ingestion, end-to-end)](resource-priority-edict.md)
- [Scaling Authority Contract (NE-164)](scaling-authority.md)
- [Durable-State Externalization](state_externalization.md)
- [Event Backbone (Kafka)](event_backbone_architecture.md)
- [Multi-Tenant graph-os over Streamable-HTTP](https://knuckles-team.github.io/graph-os/architecture/multi-tenant-streamable-http/)
- [HNSW Vector Index Lifecycle](https://knuckles-team.github.io/epistemic-graph/architecture/vector-index-lifecycle/)
- [Task-Aware Sampling Profiles](sampling_profiles.md)
- [LLM/Embedding Server-Capacity Guard (never OOM the model host)](llm-server-capacity-guard.md)
- [KV-Cache-Layering Policy (per-execution cache-worthiness)](kv-cache-layering-policy.md)
- [KV-Checkpoint Intelligence (when to freeze a context, RAM vs disk)](kv-checkpoint-intelligence.md)
- [Graph Backend Architecture](https://knuckles-team.github.io/epistemic-graph/architecture/graph-backends/)
- [Epistemic Graph Service Layer Architecture](https://knuckles-team.github.io/epistemic-graph/architecture/service-layer/)
- [GraphOS Embedded Fleet Gateway](https://knuckles-team.github.io/graph-os/architecture/fleet-gateway/)
- [MCP 2026-07-28 Native Protocol Surface](https://knuckles-team.github.io/graph-os/architecture/mcp-2026-protocol-surface/)
- [fastmcp 4 as the Default (MCP SDK v2 protocol bridge)](fastmcp4-default.md)
- [Staged httpx to httpx2 Migration (transport-factory strangler)](httpx_httpx2_migration.md)
- [Gateway Daemon (all runtime components)](https://knuckles-team.github.io/graph-os/architecture/gateway-daemon/)

## Observability, governance & safety

Metrics/logs/traces, autonomous governance, and the reliability loop that watches all of the above.

- [Observability (Metrics/Logs/Traces/Alerts)](observability.md)
- [Runtime-Reliability Loop (detect→signal→gap→heal)](runtime-reliability-loop.md)
- [Autonomous Governance & Zero-Trust Consensus](autonomous_governance_and_zero_trust.md)
- [Fleet Autonomy Control Plane](https://knuckles-team.github.io/graph-os/architecture/fleet-autonomy/)

## Deployment, release & repo operations

Shipping the platform itself — containerization, orchestrator migrations, versioning, the merge queue, and the shared dev environment.

- [Containerized Deployment (microservices)](containerized-deployment.md)
- [Orchestrator Migration & Cutover (Swarm→k8s hardening)](orchestrator-migration-cutover.md)
- [Genesis k8s Deployment Inputs + Named Environment Profiles](genesis-environment-profiles.md)
- [Drift-Proof Release & Versioning](drift_proof_release.md)
- [Lane Concurrency (four arbitration classes)](lane-concurrency.md)
- [Merge Queue (continuous merge, tiered gate)](merge-queue.md)
- [Repository-development WorkItem authority](repository-workitem-authority.md)
- [Configuration Reference & Flag Audit](configuration.md)
- [Universal External Graph Connectors](https://knuckles-team.github.io/agent-connector-sdk/architecture/universal-graph-connectors/)
- [Privacy-safe External Graph Ingestion](https://knuckles-team.github.io/agent-connector-sdk/architecture/privacy-safe-ingestion/)
- [Mandatory ContextCompiler Model Boundary](mandatory-context-compiler.md)
- [The shared workspace `.venv` — sync, flip-on-merge, drift, upgrade](shared-venv-lifecycle.md)
- [Phased Dependency Release Architecture](phased_release_architecture.md)
- [GOC-44 dependency/runtime compatibility — baseline revalidation (2026-08-16)](goc-44-dependency-runtime-compatibility-baseline.md)
- [graph-os Self-Hosting Cutover (design)](https://knuckles-team.github.io/graph-os/architecture/graphos-self-hosting/)
- [Unattended Claude Code Harness](unattended_claude_harness.md)
- [Pydantic AI v2 Migration](pydantic-ai-v2-migration.md)
- [In-House Training Substrate](in_house_training_substrate.md)

## Trace, provenance & incident history

The canonical record of what ran, why, and — for closed incidents — what broke and how it was found.

- [Canonical Trace and Outcome Ontology](trace_outcome_ontology.md)
- RCA: graph-os fleet-mount state desync (D-OB-3) — closed and merged
  (`fix/mcp-tool-state-desync`, `fe3c1f519`); the generalized "favorable-restatement"
  invariant it named lives on in
  [graph-os's fleet gateway architecture](https://knuckles-team.github.io/graph-os/architecture/fleet-gateway/#history-the-favorable-restatement-invariant-d-ob-3).
  The full RCA text is preserved in git history (`07dcbac23`), not carried forward as a
  standalone page.
- Optimization Campaign Checkpoint — a dated (2026-06-19) session resume-point note, not
  an architecture description; superseded by everything merged since. Preserved in git
  history (last touched `274d4c375`), not carried forward as a standalone page.
- [Historical Hardening Audit (non-authoritative)](epistemic-os-hardening.md)
