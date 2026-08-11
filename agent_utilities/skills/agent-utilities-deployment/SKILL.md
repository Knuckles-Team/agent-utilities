---
name: agent-utilities-deployment
skill_type: skill
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

## Genesis handoff

When `agent-os-genesis` supplies an infrastructure handoff, validate and consume it
instead of re-discovering or reprovisioning the substrate. The handoff includes the
plan digest, target/permission boundary, namespace or runtime, provider references,
component selection, resource constraints, and rendered artifact locations.

Set `substrate_resolved=true` for this path. If a required infrastructure capability
is absent, call `agent-os-genesis` in bounded `substrate-only` mode and resume with
its updated handoff; do not recursively restart a full Genesis/application workflow.

If the user requests an application deployment onto a suitable existing substrate,
run this skill directly. Use `agent-os-genesis` first only when host, cluster,
namespace, platform-provider, or orchestrator state must be created or changed.

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
- An orchestrator migration delegates target substrate and traffic-foundation work
  to `agent-os-genesis` in `migrate-substrate` mode. Consume the returned
  infrastructure handoff, then migrate each dependency-closed application unit using
  supported logical data migration, canary traffic, user-visible verification, and
  an exact rollback.

Capture current health and version state, back up durable data, validate the backup,
stage the change, and preserve a rollback. Quiesce writers when the migration cannot
provide transactional or resumable semantics. A binary rollback is insufficient
after a stored-format change; prove that the data restore path is usable before
apply. After migration or upgrade, repeat the same user-visible checks and inspect
migration completion evidence; process health alone is insufficient.

Use an economy model for inventory classification, configuration comparison,
and checklist execution. Escalate architecture, security, and recovery decisions
when the evidence is ambiguous or the blast radius is high.

## Identity & authorization — graph-os + agent-webui (READ before wiring identity)

`agent-os-genesis` resolves *whether* an OIDC provider is deployed or reused
(the `identity_ref` in its Phase 7 handoff). It does not resolve *this*
application's own role/scope contract — that is step 4.3 ("identity, policy,
and secret integration") here, and it is the single most common source of a
Day-1 "deployment succeeded, application is broken" outage: not a code defect,
but the wrong role granted to the wrong identity. Verify this contract before
declaring Verify (step 5) complete.

### Two OIDC clients — conflating them is the trap

Two separate identities call graph-os, not one:

- **`agent-webui`** — the browser/user-facing client (env
  `WEBUI_OIDC_CLIENT_ID`). Standard authorization-code (+ PKCE) flow; its
  tokens carry the signed-in human's identity.
- **`agent-webui-svc`** — the webui **backend's own service identity** (env
  `OIDC_CLIENT_ID`, `serviceAccountsEnabled=true`). This is what the webui
  backend presents when *it* calls graph-os. In Keycloak its principal is the
  synthetic user `service-account-agent-webui-svc`.

A role granted to the browser client does not flow to the backend's outbound
calls, and vice versa — they are separate credentials with separate role
assignments. Always name which of the two you are granting a role to.

### Realm roles

Roles are read from `realm_access.roles` / `resource_access.*.roles` / `scope`
and normalized IdP-agnostically (an Okta group maps the same way a Keycloak
realm role does), so this table holds regardless of IdP:

| Role | Grants | Notes |
|---|---|---|
| `kg:read` | read-only graph access | base scope |
| `kg:write` | graph mutation | **hierarchical** — expands to include `kg:read` |
| `kg:admin` | graph administration | **hierarchical** — expands to `kg:read` + `kg:write` |
| `admin:cluster-read` | the engine's `PlacementRoute` capability | required by every placement resolution; missing it fails as `ACCESS_DENIED: verified request context lacks required scope 'admin:cluster-read'` |
| `webui:admin` | UI-level admin surfaces in agent-webui | **not equivalent to `kg:admin`.** The code is explicit: "a generic application role named `admin` is not equivalent" to the graph capability. A user with only `webui:admin` gets into the UI and then every KG-backed panel fails — empty graph, no MCP tools, 503s. |

(Hierarchy source: `agent_utilities/security/request_identity.py`,
`_GRAPH_AUTH_SCOPES`.)

### Required assignments

| Identity | Roles |
|---|---|
| Human admin user | `kg:admin` **+** `webui:admin` (the second grants nothing graph-side; it only opens UI admin surfaces) |
| `service-account-agent-webui-svc` | `kg:write` (or `kg:read`-only for a read-only deployment) **and** `admin:cluster-read` |

Provision with `kcadm.sh` (`-r <realm>` — the reference deployment uses
`homelab`; use whichever realm this installation's IdP wiring selected):

```bash
# authenticate once against the admin realm
kcadm.sh config credentials --server <keycloak-base-url> --realm master \
  --user admin --password <admin-password>

# create the graph-capability + cluster + UI roles (idempotent — skip any
# that already exist)
kcadm.sh create roles -r <realm> -s name=kg:read
kcadm.sh create roles -r <realm> -s name=kg:write
kcadm.sh create roles -r <realm> -s name=kg:admin
kcadm.sh create roles -r <realm> -s name=admin:cluster-read
kcadm.sh create roles -r <realm> -s name=webui:admin

# grant the human admin
kcadm.sh add-roles -r <realm> --uusername <admin-user> \
  --rolename kg:admin --rolename webui:admin

# grant the webui BACKEND service account — note --uusername targets the
# synthetic service-account user, not the client id itself
kcadm.sh add-roles -r <realm> --uusername service-account-agent-webui-svc \
  --rolename kg:write --rolename admin:cluster-read
```

`service-account-agent-webui-svc` only exists once the `agent-webui-svc`
client has `serviceAccountsEnabled=true`; if `add-roles` reports "user not
found," check that flag on the client before assuming the role is missing —
the synthetic user is Keycloak-generated from it.

### After any role change: re-authenticate

A token minted before a grant does not carry it — the IdP never retroactively
enriches a live token. Log out via the webui's `/auth/logout` route (it
exists and returns a 302 redirect) and log back in so a fresh token is minted
carrying the new roles. This is the standing explanation for "it worked
yesterday, it's broken today" immediately after any role edit — check it
before treating the report as a regression.

### Diagnosis: symptom → cause

| Symptom | Cause | Fix |
|---|---|---|
| UI loads, but the graph shows 0 nodes/0 edges, no MCP tools, or panels return 503 — even though the engine provably has data | The signed-in user holds only `webui:admin` (a UI role), no `kg:*` scope | Grant `kg:read`/`kg:write`/`kg:admin` to the user (see `kcadm.sh` above), then re-login |
| webui backend logs `ACCESS_DENIED: verified request context lacks required scope 'admin:cluster-read'` | `service-account-agent-webui-svc` lacks `admin:cluster-read` — every placement resolution fails closed | Add `admin:cluster-read` to the service account; restart/redeploy the backend if it caches a client-credentials token |
| A permission that was just granted still doesn't take effect | Stale token minted before the grant | Log out (`/auth/logout`) and back in — role changes are never retroactive |

## Guardrails

- Never store or print credentials, tokens, private keys, or recovery material.
- Never embed hostnames, addresses, inventories, or machine paths in the skill.
- Do not weaken authentication or policy to pass a health check.
- Do not keep compatibility shims, retired configuration names, or dual-format
  readers after the one-time migration boundary.
- Do not claim deployment success while a required doctor check is failing.
- Require explicit approval for destructive data, identity, network, or external
  service changes.
