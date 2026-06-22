---
name: agent-utilities-deployment
aliases:
  - self-setup
  - deploy-agent-utilities
description: >
  One guided, config-complete path to deploy agent-utilities — the single entry
  point Claude follows to set ITSELF up or stand up a server. Profile-driven
  (tiny → single-node-prod → enterprise): installs, generates a COMPLETE config.json
  covering every option (setup-config), resolves secrets from OpenBao/Vault or .env,
  provisions databases (Stardog + pg-age), launches the graph-os gateway + MCP
  multiplexer, wires auth/observability, and verifies the whole deployment with a
  config doctor + MCP reachability + KG smoke test. Delegates the enterprise
  multi-node swarm tier to the agent-os-genesis (day0) skill rather than duplicating
  it. Use when the user says "deploy agent-utilities", "set up agent-utilities",
  "Claude set yourself up", "full install", "generate my config", "stand up the
  knowledge graph + gateway". Do NOT use for bare-host swarm bootstrap (use
  agent-os-genesis/day0) or databases-only (use database-environment-setup).
domain: infrastructure
tags:
  - deployment
  - self-setup
  - agent-utilities
  - config
  - knowledge-graph
  - gateway
requires:
  - graph-os
---

# Agent-Utilities Deployment (self-setup)

The one runbook to take agent-utilities from nothing to running, with **every config
option considered**. It composes existing skills/commands rather than re-implementing
them — `setup-config` (complete config generation + validation), the
`database-environment-setup` skill, `secret-vault-manager`, `docker-compose-operator`,
and — for the enterprise multi-node tier — the **`agent-os-genesis`** (alias `day0`)
swarm bootstrap.

## Prerequisites
- **graph-os** (agent-utilities) installed: `pip install agent-utilities[all]` (or a
  narrower extra set — see Step 1). Provides the `setup-config` / `setup-databases`
  console scripts and the `graph_configure` MCP tool.

## Profiles (rungs of docs/guides/deployment-configurations.md)
| Profile | For | Externals |
|---|---|---|
| **tiny** | Claude's own laptop self-setup; edge | none (in-process L1 + LadybugDB L2) |
| **single-node-prod** | one durable host | Postgres/pg-age, optional OpenBao/Langfuse |
| **enterprise** | multi-node fleet | swarm, Postgres, Kafka, Keycloak, observability |

## Data source

The repo's **`genesis.yaml`** (root of agent-utilities) is the machine-readable
manifest for this whole flow: the profiles below, the host preflight, the MCP
`servers` fleet (with per-profile membership), the optional UI `components`, and the
`ide_targets` for skill/MCP wiring. Loop it rather than hard-coding lists — it is
generated from `deploy/mcp-fleet.registry.yml` + the config profiles, so it never
drifts.

## Steps

### Step 0 — Preflight the host (before installing anything)
Confirm the host has the runtimes/tools for the chosen profile + any UI components:
```
agent-utilities-doctor --preflight --profile <tiny|single-node-prod|enterprise> [--component agent-webui|geniusbot|agent-terminal-ui]
# or, remotely over MCP:  graph_configure(action="preflight", config_key="<profile>")
```
It returns ok/warn/fail per dependency with a remediation. Key facts: **no Rust is
needed** (the epistemic-graph engine ships as a prebuilt wheel — Rust is only a
fallback); Docker is only required above `tiny`; Node+pnpm only for `agent-webui`; a
Qt display only for `geniusbot`. The one-command `scripts/install.sh` /
`scripts/install.ps1` runs this step for you.

### Step 1 — Choose profile & install
Ask the user (or infer from context) which profile. Install the matching extras:
- tiny: `pip install agent-utilities[all]` or run `scripts/bootstrap.sh`.
- single-node-prod / enterprise: `pip install agent-utilities[all]` plus `[owl,postgres,stardog]` as needed.

### Step 2 — Generate the COMPLETE config (all options)
Don't hand-author config.json. Generate a full, profile-seeded one covering every
AgentConfig field:
```
setup-config generate --profile <tiny|single-node-prod|enterprise>   # → XDG config.json
# or MCP: graph_configure(action="generate_config", config_key="<profile>")
```
Review/adjust the handful of deployment-varying values it seeds (DSNs, endpoints).
To see every option grouped by subsystem: `setup-config reference`.

### Step 3 — Resolve secrets (OpenBao/Vault, .env fallback)
For single-node/enterprise, store credentials in OpenBao/Vault (reuse
**secret-vault-manager** to unseal/seed) and reference them with `vault://` in config;
otherwise a local `.env`. Never commit real secrets — generated config blanks them.
Once secrets are seeded, register them with the **`automated-credential-rotation`**
skill (same OpenBao paths) so they rotate on policy (6-month baseline) — that skill is
the rotation counterpart to this provisioning step, and agent-os-genesis arms it as
Step 14b.

### Step 4 — Databases (single-node-prod / enterprise)
Run the **database-environment-setup** skill: provisions Stardog (prod) or local
SPARQL (dev) + a Postgres with AGE + pgvector + pg_search, wires the durable backend,
and backfills the graph into AGE. (tiny skips this — LadybugDB L2 is built in.)

### Step 5 — Launch the runtime
- KG MCP server / gateway: `graph-os` (and `graph-os-daemon` for the REST gateway).
- Tool multiplexer: `mcp-multiplexer`.
- Containerized: use **docker-compose-operator** with `docker/mcp.compose.yml`
  (+ `docker/pg-age-full.compose.yml` for the durable tier).

### Step 6 — Auth & observability (enterprise)
- Identity: set `KG_AUTH_REQUIRED=1` + `AUTH_JWT_JWKS_URI` (Keycloak via
  **keycloak-client-onboarder**); policy via **eunomia-policy-manager**.
- Observability: **service-observability-provisioner** wires `/metrics` + LGTM;
  `OTEL_EXPORTER_OTLP_ENDPOINT` is already in the generated enterprise config.

### Step 7 — Enterprise multi-node → delegate to agent-os-genesis (day0)
For a multi-host swarm (SSH mesh, placement, overlay networks, ingress, GitOps,
fleet deploy), hand off to the **`agent-os-genesis`** skill (alias `day0`). This skill
does NOT reimplement swarm bootstrap — it generates the config and validates the end
state around it.

### Step 8 — Verify the deployment
1. **Holistic doctor (run this first):** `agent-utilities-doctor` (or
   `graph_configure(action="system_doctor")`) — one sweep across config, engine,
   backend, secrets, auth, MCP fleet, hooks, and observability; each finding carries
   a remediation + the skill that fixes it. `--live` also probes MCP endpoints;
   `--fix` runs safe auto-remediations. This composes the focused checks below.
2. **Config health (focused):** `setup-config doctor --profile <profile>` — required
   keys, durability rules, secret-ref resolvability.
3. **MCP reachability:** `python scripts/validate_mcp_config.py --live` (catches 502s).
4. **KG smoke test:** a `graph_write` + `graph_query` round-trip, and (if databases
   were set up) confirm backfill consistency via the database skill's report.

Report the final state: profile, config path, SPARQL URL, gateway URL, and any doctor
findings the operator still needs to resolve.

### Step 9 — External data sources (optional) → delegate to `agent-utilities-source-integration`
After the runtime is verified, connect any external sources (LeanIX, Camunda, ARIS,
ServiceNow, Egeria, …) by running the **`agent-utilities-source-integration`** skill —
one standardized path that discovers (where a metamodel exists), mirrors, delta-syncs
(`source_sync`), and optionally backfeeds, per source. It needs only the source's
credentials + a reachable graph-os. Skip when no external sources are in use.

## Notes
- No new env flags — `setup-config` operates over the existing AgentConfig schema.
- Single-purpose delegation: this skill orchestrates; databases live in
  `database-environment-setup`, swarm in `agent-os-genesis` (`day0`), secrets in
  `secret-vault-manager`, and external data-source onboarding (LeanIX/Camunda/ARIS/
  ServiceNow/…) in `agent-utilities-source-integration`.
  Full narrative: agent-utilities `docs/guides/self-setup.md`.
