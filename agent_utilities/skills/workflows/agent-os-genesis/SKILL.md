---
name: agent-os-genesis
description: >-
  Day-0, idempotent Agent OS substrate provisioning for a laptop, bare-metal
  host, Docker Compose, Docker Swarm, Podman, an existing Kubernetes namespace,
  an existing cluster, or a newly provisioned multi-node cluster. Use for
  environment discovery, topology selection, cluster or container substrate,
  identity/secrets/PKI boundaries, Helm/GitOps foundations, workspace bootstrap,
  and handoff to agent-utilities-deployment for the application layer.
---

# Agent OS Genesis

Create the smallest production-suitable substrate for the requested
`agent-packages` deployment, prove it is ready, and then invoke
`agent-utilities-deployment` to install and verify the application layer.

This skill is environment-neutral. Never copy host names, addresses, storage paths,
registries, credentials, cluster names, or DNS suffixes from examples. Discover them
or require operator-owned references at run time.

## Responsibility boundary

`agent-os-genesis` owns Day-0 infrastructure:

- inventory and constraint discovery;
- deploy/use-existing/skip decisions per infrastructure capability;
- orchestrator selection and provisioning;
- namespaces, service accounts, RBAC, CNI/network boundaries, storage classes,
  ingress, certificates, secret-store integration, and GitOps substrate;
- multi-node placement, capacity, backup, restore, and failure-domain planning;
- a sanitized `genesis-plan.yaml` and deployment handoff.

`agent-utilities-deployment` owns the application:

- agent-utilities profile and engine topology;
- `config.json`, runtime environment, secret references, and connector selection;
- graph-os, epistemic-graph, MCP fleet, skills, prompts, ontologies, UIs, messaging
  entrypoints, migrations, application health checks, and release verification.

**Mandatory delegation rule:** after Day-0 preflight succeeds, invoke
`agent-utilities-deployment` with the resolved handoff. If a suitable substrate
already exists and no Day-0 change is required, skip directly to that skill. Do not
duplicate its deployment wizard or application configuration here.

Read [deployment-contract.md](references/deployment-contract.md) before changing
infrastructure.

## Safety invariants

1. Default every capability to `use-existing` when a compatible managed service is
   reachable; otherwise `deploy`; use `skip` only when its dependants are also
   disabled.
2. Read and inventory before mutation. Render and validate before apply.
3. Never persist plaintext secrets in Git, Helm values, Compose files, generated
   plans, logs, the Knowledge Graph, or agent context. Store only secret references.
4. Pin production images by digest and verify signatures/SBOMs where the registry
   supports them. Floating tags are development-only.
5. Never grant `cluster-admin` to an application or agent. Existing-namespace mode
   must remain namespaced.
6. Preserve rollback until health, data, identity, ingress, and delegated tool calls
   pass from the user-facing route.
7. A plan is not a deployment. Report `planned`, `rendered`, `applied`, and
   `verified` independently.
8. Make every action idempotent and record an idempotency key plus before/after
   evidence. Destructive replacement, data migration, DNS cutover, or trust-root
   rotation requires explicit operator approval.

## Phase 0 — Resolve intent and scope

Collect or infer:

- mode: `development`, `evaluation`, `small-production`, or
  `production-at-scale`;
- target: `bare-metal`, `compose`, `swarm`, `podman`, or `kubernetes`;
- substrate authority:
  - Kubernetes: `namespace-only`, `existing-cluster`, `provision-cluster`, or
    `provision-multi-node`;
  - containers: local engine, existing engine, or new Swarm;
  - bare metal: existing OS/service manager or newly prepared host;
- selected components: `core`, named packages, manifest label/filter, or `all`;
- engine topology: `unified-in-process` or `out-of-process-shared`;
- identity, secrets, PKI, DNS/ingress, storage, observability, backup, and GitOps
  providers, each as `deploy`, `use-existing`, or `skip`;
- **optional data-plane substrate** — object storage, a table/catalog service, a SQL
  query engine, batch compute, event streaming, and a triple store — each named
  service independently `deploy`, `use-existing`, or `skip`. These are never
  required: default every one to `skip` unless the scope explicitly selects it, and
  a laptop/minimal profile must resolve with none selected. Read
  [data-plane-substrate.md](references/data-plane-substrate.md) before selecting or
  connecting any of them;
- tenancy, availability, recovery objectives, resource ceilings, egress policy,
  change window, and approval boundaries.

Do not assume that “enterprise” means self-host every dependency. A namespace in a
managed cluster using existing OIDC, Vault-compatible secrets, ingress, storage, and
observability is a first-class production target.

Genesis resolves *which* IdP is deployed or reused; it does not resolve the
*application's* own role/scope contract on top of it. For agent-utilities
(graph-os + agent-webui) that contract has a specific, easy-to-conflate
shape — two distinct OIDC clients (a browser client and a separate backend
service-account client), a hierarchical graph-capability scope set, a
cluster-placement capability role, and a UI-level admin role that is
deliberately **not** equivalent to graph administration. Provisioning and
diagnosing it is `agent-utilities-deployment`'s job (its *Identity &
authorization* section) — confirm it there before treating a Day-0 identity
handoff as verified; a missing scope on the service account fails every
downstream placement call closed, and looks like a data or wiring bug rather
than an authz gap.

## Phase 1 — Discover and preflight

Inspect without changing state:

- CPU architecture, cores, RAM, accelerators, disk capacity/IOPS, filesystem,
  network interfaces, MTU, time synchronization, kernel/cgroup support, and open
  ports;
- container runtime and orchestrator versions;
- cluster version, API capabilities, namespace quotas, LimitRanges, Pod Security,
  RBAC, CNI, CSI, ingress/Gateway API, cert-manager, External Secrets, metrics,
  topology labels, and autoscaling APIs;
- existing OIDC discovery document, secret-store auth method, trust bundle, DNS
  authority, registry access, observability endpoints, and backup target;
- required architectures and image availability for each selected component;
- overlap among host, service, pod, and VPN CIDRs;
- current `workspace.yml`, selected packages, dependency order, and deployment
  artifacts.

Classify every prerequisite as `ready`, `degraded`, `missing`, or `incompatible`,
with evidence and a remediation. Never mutate merely to perform discovery.

## Phase 2 — Produce the deployment contract

Write a sanitized, operator-owned `genesis-plan.yaml` conforming to
[deployment-contract.md](references/deployment-contract.md). It must include:

- target and authority level;
- selected component names and source revision;
- per-capability `deploy|use-existing|skip` action;
- resource requests, limits, placement, failure domains, and concurrency ceilings;
- network, ingress, storage, identity, secrets, PKI, observability, and backup
  provider references;
- immutable image references;
- ordered phases, health gates, rollback, and idempotency keys;
- the exact handoff for `agent-utilities-deployment`;
- redacted evidence locations.

Resolve conflicts before execution. Examples: a skipped secret store cannot satisfy
an ExternalSecret; a namespace-only service account cannot install cluster-scoped
CRDs; a `ReadWriteOnce` engine volume cannot be mounted by multiple writers.

## Phase 3 — Select the minimum substrate

| Situation | Target | Reference |
|---|---|---|
| Developer changing many repositories | bare metal or Compose/Podman | [development-workspace.md](references/development-workspace.md) |
| One host, durable production | Compose, rootful Podman, or systemd | [container-orchestrators.md](references/container-orchestrators.md), [bare-metal.md](references/bare-metal.md) |
| Several existing container hosts | Swarm when Kubernetes is not desired | [container-orchestrators.md](references/container-orchestrators.md) |
| Existing namespace with no cluster administration | Kubernetes `namespace-only` | [kubernetes-and-helm.md](references/kubernetes-and-helm.md) |
| Existing cluster with platform administration | Kubernetes `existing-cluster` | [kubernetes-and-helm.md](references/kubernetes-and-helm.md) |
| New single-node or edge cluster | Kubernetes `provision-cluster` | [kubernetes-and-helm.md](references/kubernetes-and-helm.md) |
| Multi-node HA, multiple failure domains | Kubernetes `provision-multi-node` | [kubernetes-and-helm.md](references/kubernetes-and-helm.md) |
| Workload needs object storage, an Iceberg-style catalog, SQL federation, batch compute, streaming, or a triple store | optional data-plane substrate, Kubernetes only | [data-plane-substrate.md](references/data-plane-substrate.md) |

Do not provision Kubernetes merely because it is available. Choose it when its
scheduling, policy, availability, GitOps, or tenancy benefits justify the operational
cost.

## Phase 4 — Provision or attach

### Kubernetes

Follow [kubernetes-and-helm.md](references/kubernetes-and-helm.md). The shipped
`assets/helm/agent-os` chart is the native application substrate recipe and supports:

- use of the release namespace without creating or owning it;
- namespaced service accounts/RBAC;
- unified or shared engine topology;
- persistent storage, probes, resources, PDB, HPA, ingress, and NetworkPolicy;
- selected MCP connectors and additional components from values;
- existing Secret references only.

Cluster bootstrap is provider-pluggable. The plan may select a managed service,
Cluster API, kubeadm, RKE2, k3s, Talos, or an operator-approved equivalent, but it
must record the provider/version and prove the same postconditions. Do not hardcode
one distribution.

#### Optional data-plane substrate

Each selected data-plane service (object storage, catalog, query engine, compute,
streaming, triple store) is a **sibling to the chart, not a chart template** — it
ships as its own checked-in, environment-specific config-as-code manifest
(`services/<name>/k8s/manifests.yaml` in this workspace) rather than a Helm values
list, because these are independently versioned platform services, not
`agent-utilities` application pods. Discover before deploying, apply the same
render→validate→apply→verify gates as the chart, and connect selected
`agent-utilities`/`epistemic-graph` consumers only through resolved endpoint and
auth references in the handoff. Full discover/deploy/connect procedure, the
component catalog, and every known gotcha:
[data-plane-substrate.md](references/data-plane-substrate.md).

### Compose, Swarm, and Podman

Follow [container-orchestrators.md](references/container-orchestrators.md). Generate
one environment-neutral model and render it to the chosen runtime. Keep data,
configuration, and secret references separate; preserve healthchecks, restart
policy, resource bounds, and placement intent.

### Bare metal

Follow [bare-metal.md](references/bare-metal.md). Install into a dedicated service
account and Python environment, emit hardened systemd units, use the platform secret
store, and keep persistent state outside the checkout.

## Phase 5 — Establish cross-cutting services

Follow [security-and-operations.md](references/security-and-operations.md):

1. workload identity and least-privilege authorization;
2. secret reference resolution and rotation;
3. root/intermediate trust and service certificates;
4. ingress/Gateway API and internal service discovery;
5. persistent storage, snapshots, backup, and restore drill;
6. logs, metrics, traces, alert routing, and cost/resource attribution;
7. egress allow-list and tenant/network isolation;
8. GitOps ownership and drift detection.

Connect to existing providers through declared endpoints and references. Provision a
new provider only when the plan selected `deploy`.

au's own background daemons (the unified scheduler chief among them) need a distinct
admission credential to authenticate to the engine and be granted access to its
control graph — without it every scheduler tick fails closed, invisibly, with no
application-level symptom until the trace is inspected. This is not covered by the
IdP/workload-identity provisioning above: it is a separate, engine-side shared-secret
mechanism with its own trust properties, including some the operator should treat as
known risks rather than incidental detail. Provision it as an explicit step here —
dedicated signer, engine-side signer-map entry, and the matching provisioner secret —
and verify it per the named observable, never merely "the pod started." Read
[engine-identity-admission.md](references/engine-identity-admission.md) end to end
before doing this for a new environment; it documents the exact chain, the four
design properties an operator should be uneasy about and their compensating controls,
and the concrete rotation/revocation sequence.

## Phase 6 — Bootstrap a development workspace

When `mode: development`, follow
[development-workspace.md](references/development-workspace.md):

1. clone the supplied repository URL;
2. locate the canonical root `workspace.yml`;
3. install `repository-manager`;
4. run its manifest-driven parallel clone/setup against an explicit workspace root;
5. validate and mirror the manifest into the Graph-OS XDG runtime location and the
   packaged repository-manager seed;
6. install selected repositories in editable mode in dependency order;
7. start only the selected services and enable source mounts only in the dev profile;
8. delta-ingest the checked-out sources after Graph-OS is healthy.

Never guess repository URLs from package names. The manifest is authoritative.

## Phase 7 — Mandatory application handoff

Invoke `agent-utilities-deployment` with:

```yaml
deployment_profile: <evaluation|development|small-production|production-at-scale>
run_target: <bare-metal|compose|swarm|podman|kubernetes>
substrate_resolved: true
namespace: <namespace-or-null>
engine_topology: <unified-in-process|out-of-process-shared>
components: [<resolved component names>]
providers:
  identity_ref: <reference-or-null>
  secrets_ref: <reference>
  trust_bundle_ref: <reference-or-null>
  ingress_class: <name-or-null>
  storage_class: <name-or-null>
  observability_ref: <reference-or-null>
  data_plane_ref: <reference-or-null>  # optional; see data-plane-substrate.md
artifacts:
  helm_values: <path-or-null>
  compose_model: <path-or-null>
  inventory: <path>
constraints:
  namespace_scoped: <true|false>
  immutable_images: <true|false>
```

The deployment skill must render/install the application configuration, deploy the
selected graph-os/engine/fleet/UI entrypoints, run migrations where needed, and
return verification evidence. If it reports a missing substrate capability, resume
genesis only for that bounded prerequisite and then re-invoke it.

## Phase 8 — End-to-end exit gates

Do not declare success until all applicable gates pass:

1. rendered artifacts contain no unresolved placeholders or plaintext secrets;
2. policy/schema validation and dry-run/server-side apply pass;
3. workloads satisfy readiness, resource, placement, and restart checks;
4. engine persistence survives a controlled restart and a backup restore is tested;
5. OIDC/static auth, tenant separation, secret resolution, TLS, and egress policy
   behave as declared;
6. Graph-OS exposes its health and MCP discovery surface;
7. selected skills, prompts, ontologies, and MCP tools are discoverable after
   ingestion;
8. one read-only delegated local-model run traverses the actual Pydantic execution
   layer, calls an allow-listed tool, and records model/tool/graph spans;
9. every enabled user entrypoint reaches the same backend execution contract;
10. rollback is executable and evidence is stored without credentials;
11. au's own engine-identity admission chain is provisioned and proven, not merely
    deployed — a fresh environment must not be declared complete while the
    scheduler cannot read its own control graph. Confirm the named observable from
    [engine-identity-admission.md](references/engine-identity-admission.md)
    ("Verifying it works"): admission ran and reported success, the identity is
    registered engine-side with the expected role, and a control-graph read that
    previously failed with `CypherEngineError(PermissionError)` now succeeds. A
    healthy pod with an unprovisioned admission chain fails this gate.

For graph execution, distinguish these claims:

- `Agent.run` tool loop validated;
- multi-node `pydantic_graph` DAG validated;
- harness `DynamicWorkflow` validated.

One does not prove the others. Record node transitions, state/checkpoint IDs, tool
arguments after redaction, model/provider, termination reason, and trace linkage for
each claim.

## Output

Return:

- resolved plan and component matrix;
- what was used, provisioned, skipped, or blocked;
- immutable artifact/revision identifiers;
- `agent-utilities-deployment` handoff and result;
- per-gate evidence links;
- rollback instructions;
- deferred work with owner and reason.

Never report an unexecuted recipe as live or a partial trace as end-to-end
validation.
