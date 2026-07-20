---
name: agent-utilities-deployment
description: >-
  Plan, provision, configure, preflight, verify, migrate, upgrade, or recover an
  agent-utilities installation from a minimal local profile through a multi-node
  production profile. Use when deployment profile, topology, infrastructure,
  identity, secrets, observability, connectors, or installation state may change.
  For incidents in an already-running Graph-OS runtime that need no deployment
  change, use graph-runtime-and-governance.
---

# Agent Utilities deployment

Resolve the deployment profile from requirements, produce a reviewable plan,
apply only authorized steps, and verify the user-visible Graph-OS path.

Keep post-deployment health, trace, audit, and policy diagnosis that does not
change the manifest or topology in `graph-runtime-and-governance`.

## Workflow

### 1. Gather requirements

Confirm:

- expected users, tenants, workload, durability, and recovery objectives;
- local, single-node, or multi-node topology;
- existing database, identity, secret-store, ingress, and observability services;
- allowed deployment mechanism and change window;
- which connectors and user entry points are in scope.

Do not infer permission to create external infrastructure or rotate credentials.

Use the skill directly for a bounded plan, preflight, or health check. Delegate
dependency-ordered phases for a multi-node rollout or recovery, keeping every
external change behind the same approval boundary.

### 2. Select a profile

| Profile | Intended shape |
|---|---|
| `tiny` | One process, minimal external infrastructure, development or evaluation |
| `single-node-prod` | Durable engine plus optional mirrors and core services on one node |
| `enterprise` | Multi-node services, durable queues, identity, policy, and observability |

Use the repository's `genesis.yaml` as the deployment manifest. Treat live
inventory and secret values as operator-owned inputs, never as skill content.

### 3. Preflight and plan

- Run the deployment preflight for the selected profile.
- Generate configuration from the canonical schema instead of hand-authoring a
  partial file.
- Resolve each dependency as deploy, reuse, or skip.
- Present the plan, destructive steps, rollback, and verification before apply.
- Keep credentials in the configured secret store; pass references, not values.

### 4. Deploy in dependency order

1. engine authority and durable storage;
2. Graph-OS gateway and workers;
3. identity, policy, and secret integration;
4. observability and health reporting;
5. multiplexer and connector fleet;
6. optional user interfaces and scheduled ingestion.

Use canary rollout and health gates for multi-node or unfamiliar components.
Stop a dependent stage when its prerequisite is unhealthy.

### 5. Verify

- Run `agent-utilities-doctor` and resolve all in-scope failures.
- Verify configuration validation and engine reachability.
- Execute a synthetic write/read round trip in an isolated graph when mutation
  is authorized; otherwise use read-only health checks.
- Verify REST and MCP reach the same core behavior.
- Confirm authentication, authorization, audit, metrics, and connector freshness.
- Record profile, component status, evidence, and remaining operator actions.

### 6. Migrate, recover, or upgrade

Classify the change before applying it:

- A release upgrade replaces current binaries or packages without changing stored
  data.
- A configuration migration renders the current typed schema from operator-owned
  references; it does not retain retired keys or aliases in runtime code.
- A persisted-format migration transforms durable state once at the deployment
  boundary. Use the engine's supported migration hook, checkpoint progress, and do
  not add a permanent read-old/write-new path.
- An ontology or object-schema migration belongs to
  `graph-modeling-and-mutation`; coordinate its deployment ordering here.

Capture current health and version state, back up durable data, validate the backup,
stage the change, and preserve a rollback. Quiesce writers when the migration cannot
provide transactional or resumable semantics. A binary rollback is insufficient
after a stored-format change; prove that the data restore path is usable before
apply. After migration or upgrade, repeat the same user-visible checks and inspect
migration completion evidence; process health alone is insufficient.

Use an economy model for inventory classification, configuration comparison,
and checklist execution. Escalate architecture, security, and recovery decisions
when the evidence is ambiguous or the blast radius is high.

## Guardrails

- Never store or print credentials, tokens, private keys, or recovery material.
- Never embed hostnames, addresses, inventories, or machine paths in the skill.
- Do not weaken authentication or policy to pass a health check.
- Do not keep compatibility shims, retired configuration names, or dual-format
  readers after the one-time migration boundary.
- Do not claim deployment success while a required doctor check is failing.
- Require explicit approval for destructive data, identity, network, or external
  service changes.
