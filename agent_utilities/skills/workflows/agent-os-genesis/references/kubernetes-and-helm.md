# Kubernetes and Helm

Use one postcondition contract across managed Kubernetes, an existing cluster, or a
new cluster. Distribution-specific steps are adapters, not workflow semantics.

## Authority modes

### Existing namespace

- Use the supplied namespace and service account.
- Create only namespaced resources allowed by RBAC.
- Do not install CRDs, ClusterRoles, admission controllers, CNI, CSI, ingress
  controllers, cluster issuers, or cluster-wide operators.
- Preflight those capabilities and emit a precise administrator requirement when
  missing.

### Existing cluster with platform authority

- Create a dedicated namespace and service accounts.
- Install shared operators only when selected in the plan and absent/incompatible.
- Keep application workloads namespaced and least-privileged.

### Provision a cluster

Select an operator-approved managed service, Cluster API provider, kubeadm, RKE2,
k3s, Talos, or equivalent. Pin versions. Inventory nodes and choose non-overlapping
pod/service CIDRs, CNI, CSI, control-plane endpoint, ingress, load balancing,
registry trust, and failure domains. A multi-node production control plane normally
uses an odd number of voters across failure domains.

After bootstrap, run the same existing-cluster preflight. Provider installation does
not prove substrate readiness.

## Helm chart

The chart is at `assets/helm/agent-os` relative to the skill. Render before apply:

```bash
helm lint <chart-path> --values <operator-values>
helm template <release> <chart-path> \
  --namespace <namespace> \
  --values <operator-values> > <rendered-output>
kubectl apply --dry-run=server --namespace <namespace> -f <rendered-output>
```

Use `--create-namespace` only when the contract grants namespace creation. The chart
does not own namespaces by default.

The chart deliberately accepts only references to an existing Secret. Create or
synchronize that Secret through the selected secret provider before installation.
Production values must pin image digests.

### Topology — `unified-in-process` is the standard

`topology=unified-in-process` (the chart default) creates **one** durable graph-os
workload with the engine in-process and no separate engine workload. This is the
standard deployment. graph-os is a single MCP + API + webui server; the dashboard
already runs as an in-process co-service under `ENABLE_WEB_UI`, so a second workload
buys nothing and costs a config contract that must be kept identical on both sides.

That contract is not theoretical. Splitting graph-os into a client and a `KG_DAEMON_ROLE=host`
peer is what allowed the engine signer identity to be provisioned on one side and not the
other, which surfaces as `503 engine_admission_unavailable` across every panel of the
webui rather than as anything resembling a topology problem (see
`engine-identity-admission.md`). Env keys that are meaningful on exactly one of the two —
`MESSAGING_INTAKE_ENABLED` is the canonical example — silently do nothing when set on the
wrong one.

Collapsing the split means summing both containers' resource requests and limits onto the
single one, and moving any host-only settings (`KG_DAEMON_ROLE=host`, the loop-engine keys,
`MESSAGING_INTAKE_ENABLED`) onto it. Never scale the unified workload above one writer.

`topology=out-of-process-shared` creates a durable engine StatefulSet and stateless
graph-os clients that can use an HPA. Choose it only when those replicas are genuinely
required.

**Metrics.** Export from graph-os natively by binding the engine's metrics listener to the
pod address (`--metrics-addr 0.0.0.0:<port>`). Do not add a sidecar to forward a
loopback-bound listener to the pod address — that is a second container solving a
one-argument problem.

Connectors and optional components are data-driven lists in values. Dependency
closure and application-specific configuration are produced by
`agent-utilities-deployment`, not inferred by Helm.

## Required preflight

- Kubernetes/API versions supported by the chart.
- Namespaced RBAC for get/list/watch/create/update/patch/delete of selected kinds.
- Pod Security and admission policies.
- default or selected StorageClass and access modes.
- CNI NetworkPolicy support.
- ingress class or GatewayClass and certificate mechanism.
- metrics API before enabling HPA.
- node architectures, topology labels, taints/tolerations, accelerators, quotas, and
  LimitRanges.
- registry pull and signature policy.
- DNS, MTU, NTP, and outbound access to declared providers.

## Production gates

Run chart schema validation, lint, template, policy checks, server dry-run, rollout
status, probes, PDB/HPA checks, NetworkPolicy tests, persistence restart, backup and
restore, and an external user-route test. Re-run `helm upgrade --install` with the
same values and require no unintended drift.
