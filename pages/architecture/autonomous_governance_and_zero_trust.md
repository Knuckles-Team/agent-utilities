# Autonomous Governance & Zero-Trust Consensus

The ecosystem enforces **Zero-Trust** security across all operations utilizing the `PermissionsKernel` alongside a specialized background actor, the `GraphGovernanceAgent`.

## 1. Zero-Trust C4 Diagram

This illustrates how agent identities and multisig mutations flow securely to the Rust `epistemic-graph` service.

<div class="admonition architecture" markdown>
<p class="admonition-title">C4 context: zero-trust multi-sig mutations</p>

Agent 1 (Orchestrator) initiates a mutation requiring quorum by submitting
a signed proposal to `PermissionsKernel` (Python, in `agent-utilities`),
which manages identity, sandbox restrictions, and collects BFT signatures.
Agent 2 (Peer) validates and submits a cryptographic signature to the same
`PermissionsKernel`.

`PermissionsKernel` talks to two Rust components in `epistemic-graph`:
it calls `IsolationLayer` via `RegisterIdentity(id, role, signature)`
(HMAC / TCP) — which maintains cryptographic identity keys and role
definitions — and calls the Graph Compute Service via
`ApplyMultisigMutation(payload, signatures)` (UDS / TCP), the definitive
authority for data mutations and state. `IsolationLayer` authorizes each
request to the Graph Compute Service based on quorum and roles.
</div>

### Shared Architecture via IntelligenceGraphEngine

Graph workflows and background daemon tasks such as consolidation and governance
share the single native `IntelligenceGraphEngine` gateway.

When `agent_utilities` starts via `app.py`, the `FastAPI` lifespan boots a singleton `IntelligenceGraphEngine`. This engine establishes a pool of connections (UDS/TCP) to the persistent backends and the transient `epistemic-graph` service.

By injecting this exact `engine` into the `GraphGovernanceAgent` at startup, the governance daemon:
1. Reuses the exact same network pools, reducing socket pressure.
2. Sees identical state as the standard agent workloads.
3. Automatically respects the Zero-Trust policies enforced within the engine's `SyncEpistemicGraphClient`.

## 2. Governance Workflow Diagram

<div class="admonition architecture" markdown>
<p class="admonition-title">C4 container: GraphGovernanceAgent event loop</p>

Inside the Gateway API (`app.py`), `IntelligenceGraphEngine` (the shared
engine) is injected into `GraphGovernanceAgent` (a daemon) on boot.
`GraphGovernanceAgent` triggers `GovernanceWorkflow.run_audit_cycle()`,
which calls `ConfigStalenessAuditor` to find legacy objects to remove,
checks agent roles and fetches active proposals through the shared
`IntelligenceGraphEngine`, and persists decisions (`gov_decision:*`) to
the Knowledge Graph (`EpistemicGraph`, Rust) — the durable store of
rules, decisions, and ontology.
</div>

### Workflow Execution
1. **Audit Cycle**: Periodically, the daemon audits the graph for stale data or newly proposed `AGENTS.md` reflectors.
2. **Scoring**: It computes a `risk_score` for each ecosystem mutation proposal.
3. **Approval Gates**:
   - Low-risk (score < 0.4) are Auto-Approved by the Daemon.
   - High-risk (score > 0.4) are persisted as `PENDING` nodes in the Knowledge Graph for a human or a Multi-Sig threshold of administrative agents to approve.
