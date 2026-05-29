# Autonomous Governance & Zero-Trust Consensus

The ecosystem enforces **Zero-Trust** security across all operations utilizing the `PermissionsKernel` alongside a specialized background actor, the `GraphGovernanceAgent`.

## 1. Zero-Trust C4 Diagram

This illustrates how agent identities and multisig mutations flow securely to the Rust `epistemic-graph` service.

```mermaid
C4Context
    title C4 Context: Zero-Trust Multi-Sig Mutations

    Person(Agent1, "Agent 1 (Orchestrator)", "Initiates a mutation requiring quorum")
    Person(Agent2, "Agent 2 (Peer)", "Validates and cryptographically signs")

    System_Boundary(b0, "agent-utilities (Python)") {
        Component(PermKernel, "PermissionsKernel", "Python", "Manages identity, sandbox restrictions, and collects BFT signatures")
    }

    System_Boundary(b1, "epistemic-graph (Rust)") {
        Component(Isolation, "IsolationLayer", "Rust", "Maintains Cryptographic Identity Keys and Role definitions")
        Component(GraphService, "Graph Compute Service", "Rust", "Definitive authority for data mutations and state")
    }

    Rel(Agent1, PermKernel, "Submits signed proposal")
    Rel(Agent2, PermKernel, "Submits cryptographic signature")

    Rel(PermKernel, Isolation, "RPC: RegisterIdentity(id, role, signature)", "HMAC / TCP")
    Rel(PermKernel, GraphService, "RPC: ApplyMultisigMutation(payload, signatures)", "UDS / TCP")
    Rel(Isolation, GraphService, "Authorizes request based on quorum and roles")
```

### Shared Architecture via IntelligenceGraphEngine

Both the legacy Graph workflows and the new background daemon tasks (like Consolidation and Governance) share a single, native gateway layer interface known as the `IntelligenceGraphEngine`.

When `agent_utilities` starts via `app.py`, the `FastAPI` lifespan boots a singleton `IntelligenceGraphEngine`. This engine establishes a pool of connections (UDS/TCP) to the persistent backends and the transient `epistemic-graph` service.

By injecting this exact `engine` into the `GraphGovernanceAgent` at startup, the governance daemon:
1. Reuses the exact same network pools, reducing socket pressure.
2. Sees identical state as the standard agent workloads.
3. Automatically respects the Zero-Trust policies enforced within the engine's `SyncEpistemicGraphClient`.

## 2. Governance Workflow Diagram

```mermaid
C4Container
    title C4 Container: GraphGovernanceAgent Event Loop

    Container_Boundary(b_app, "Gateway API (app.py)") {
        Component(Engine, "IntelligenceGraphEngine", "Python", "Shared Engine")

        Component(GovAgent, "GraphGovernanceAgent", "Daemon", "Runs periodic background tasks for audit and review")
        Component(GovWorkflow, "GovernanceWorkflow", "Policy Pipeline", "Calculates risk scores, persists to KG")

        Component(Staleness, "ConfigStalenessAuditor", "Python", "Finds stale configs")
    }

    ContainerDb(KG, "Knowledge Graph (Rust)", "EpistemicGraph", "Stores rules, decisions, ontology")

    Rel(GovAgent, Engine, "Injected on boot")
    Rel(GovAgent, GovWorkflow, "Triggers run_audit_cycle()")
    Rel(GovWorkflow, Staleness, "Finds legacy objects to remove")

    Rel(GovWorkflow, Engine, "Check agent roles and fetch active proposals")
    Rel(GovWorkflow, KG, "Persist decisions (gov_decision:*)")
```

### Workflow Execution
1. **Audit Cycle**: Periodically, the daemon audits the graph for stale data or newly proposed `AGENTS.md` reflectors.
2. **Scoring**: It computes a `risk_score` for each ecosystem mutation proposal.
3. **Approval Gates**:
   - Low-risk (score < 0.4) are Auto-Approved by the Daemon.
   - High-risk (score > 0.4) are persisted as `PENDING` nodes in the Knowledge Graph for a human or a Multi-Sig threshold of administrative agents to approve.
