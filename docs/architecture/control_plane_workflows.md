# Immutable workflow control plane

The workflow control-plane core owns versioned definitions and release
selection. It does not execute work or become a second queue. Every step pins
one exact approved agent, skill, tool, delegation, and/or policy reference by
concrete semantic version and SHA-256 digest. Definitions contain only opaque
artifact/schema references and bounded summaries; prompts, credentials,
arguments, bodies, and results remain outside this boundary.

```mermaid
flowchart LR
    T[Immutable template identity] --> D[Workflow definition version]
    D --> V[Cycle + bound + exact-binding validation]
    V --> C[Immutable version catalog]
    C --> P[CAS channel pointer]
    P --> R[Resolution digest + safe summary]
    R -. submit definition-derived work .-> W[Native WorkItem]
    W --> X[Claim / lease / result authority]
```

## Invariants

- A `(workflow_id, version)` can be written once. Reusing it with a different
  definition digest is rejected.
- A channel update compares the complete prior pointer and increments its
  generation. Stale writers fail closed.
- Resolution verifies that the pointer, immutable definition, exact binding
  registry, graph bounds, and resolution digest still agree.
- Versions and capability bindings use concrete semantic versions. `latest`,
  ranges, floating aliases, and unresolved capabilities are rejected.
- Requested graph and execution budgets cannot exceed the policy-attested
  budget or the hard process ceilings.
- Rollback is a CAS update to an already-published lower version; it does not
  mutate or delete the newer version.
- The catalog has no claim/lease/result methods. Native `WorkItem` remains the
  sole execution authority after a caller materializes a resolved definition.
