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

`topology=unified-in-process` creates one durable graph-os workload and no separate
engine. `topology=out-of-process-shared` creates a durable engine StatefulSet and
stateless graph-os clients that can use an HPA. Never scale the unified workload
above one writer.

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
