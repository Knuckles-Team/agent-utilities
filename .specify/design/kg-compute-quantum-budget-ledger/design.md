# Design Document: Quantum control-plane quota/cost-budget gate (Q8)

CONCEPT:AU-KG.compute.quantum-budget-ledger

> `agent_utilities/knowledge_graph/quantum/budget.py`

## Decision — fail-closed budget reservation for any non-exempt quantum backend

`plans/au-eg-program/program/quantum-external-providers.md` §1.5 (Q8/Q9 —
reachability and governance) states that closing the "nothing in this program
is reachable from graph-os" gap requires Q8 (quota/cost budgets) alongside Q9
(provenance, `CONCEPT:AU-KG.temporal.quantum-run-provenance`). §1.3 is explicit
that this is not optional: the IBM Open Plan free tier is ~10 minutes of QPU
runtime per 28-day rolling window — a hard, shared resource a handful of agent
calls can trivially exhaust.

This module owns the ENFORCEMENT surface any hardware/cloud provider adapter
(wired by the sibling lane `w6-quantum-q10-providers`) must clear before a
request with a non-default backend is sent over the wire. Design, deliberately
conservative: the engine's own two registered backends today (`sv-cpu`/
`stabilizer`, `eg-quantum-sim`) are always free and unlimited
(`QUOTA_EXEMPT_BACKEND_IDS`) — the default `graph_quantum` call path never
touches this module. Any OTHER `backend_id` is treated as "might be a paid/
quota-limited provider" and is fail-closed by default: `reserve_quantum_budget`
denies the request unless a `:QuantumProviderQuota` node already exists for
that backend's provider family. Usage is tracked per `(tenant,
provider_family)` in daily buckets (`:QuantumUsageDay`), summed over the
trailing 28 days for the rolling-window check.

**The rejected alternative** was an allowlist of known-expensive backend ids.
That inverts the safety direction: a NEW backend id the engine registers later
would default to unlimited/free until someone remembered to add it to the
list. Fail-closed-by-default (deny unless a quota grant exists) means a new
backend is safe by construction the day it ships, with zero action required
from this module.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/quantum/budget.py` only;
  consumed by the Q10 provider-adapter lane, not yet wired to a live provider.
- **Backward Compatible**: Yes — the default (exempt, in-process) backends are
  never gated.
- **Known weak point**: usage recording (`record_quantum_usage`) is
  best-effort and never raises on a KG write failure — a cold/absent KG
  degrades the reservation check itself to fail-closed (correct), but a
  post-hoc usage true-up can silently under-count if the KG write fails after
  a reservation succeeds.
