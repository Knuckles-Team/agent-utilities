# Portable deployment contract

Genesis resolves intent into a sanitized contract before it changes infrastructure.
The contract is operator-owned and contains references, never credential values.

## Minimal schema

```yaml
schema_version: 1
operation: plan # plan | apply | verify | migrate | recover
source:
  repository_url: <operator-supplied URL>
  revision: <immutable commit or release>
  workspace_manifest: workspace.yml
scope:
  mode: runtime # runtime | development | both
  selectors: [core]
  include: []
  exclude: []
target:
  runtime: kubernetes # bare-metal | compose | swarm | podman | kubernetes
  ownership: use-existing # use-existing | provision
  authority: namespace-only # namespace-only | cluster-admin | host-admin
capabilities:
  identity: {action: use-existing, provider_ref: <reference>}
  secrets: {action: use-existing, provider_ref: <reference>}
  pki: {action: use-existing, provider_ref: <reference>}
  ingress: {action: use-existing, provider_ref: <reference>}
  storage: {action: use-existing, provider_ref: <reference>}
  observability: {action: use-existing, provider_ref: <reference>}
  gitops: {action: skip, provider_ref: null}
  data_plane: [] # optional named services — see below and data-plane-substrate.md
topology:
  engine: unified-in-process
  high_availability: false
  failure_domains: []
artifacts:
  images: {}
  inventory_ref: <path-or-object-reference>
approvals:
  destructive: false
  cutover: false
```

Each selected component expands into a dependency-closed record with `action`,
`provider`, immutable artifact, resources, placement, health gate, rollback, and
idempotency key. Record a stable digest of the resolved plan.

`capabilities.data_plane` is a list, not a single provider reference, because it
names independently optional services rather than one substrate axis — each entry
takes the same `action: deploy | use-existing | skip` plus a `provider_ref`:

```yaml
capabilities:
  data_plane:
    - {name: object-store, action: skip, provider_ref: null}
    - {name: catalog, action: skip, provider_ref: null}
    - {name: query-engine, action: skip, provider_ref: null}
    - {name: compute, action: skip, provider_ref: null}
    - {name: streaming, action: use-existing, provider_ref: <reference>}
    - {name: triple-store, action: use-existing, provider_ref: <reference>}
```

Default every entry to `skip`; a minimal/laptop profile must resolve with an empty
or all-`skip` list. See
[data-plane-substrate.md](data-plane-substrate.md) for the concrete service
catalog, discovery, deploy, and connect procedure.

## Infrastructure handoff

The handoff to `agent-utilities-deployment` contains:

- plan digest and source revision;
- runtime and permission boundary;
- namespace or service account;
- registry, storage class, ingress class, DNS zone, and trust-bundle references;
- identity issuer/client references and secret-store references;
- resolved data-plane service endpoint/auth references, when any are selected;
- selected component names;
- resource ceilings and topology;
- rendered artifact locations;
- preflight evidence and unresolved administrator requirements.

Set `substrate_resolved: true` on the application call. If application deployment
requests Genesis for a missing prerequisite, Genesis runs `substrate-only` and returns
a new handoff; it never recursively deploys the application.

## State labels

- `planned`: a validated contract exists.
- `rendered`: runtime artifacts validate offline.
- `applied`: the target accepted the artifacts.
- `verified`: the user-visible route and persistence gates passed.
- `blocked`: a named missing capability or approval prevents the next transition.

Never collapse these labels into a single “done” state.
