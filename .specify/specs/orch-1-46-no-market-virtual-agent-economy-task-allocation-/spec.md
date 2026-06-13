# Spec: Agent-Market Task Allocation (Virtual Agent Economy) (ORCH-1.46)

> Status: **proposed**.
> **Wire-First:** add a market allocation backend **behind the existing dispatch seam**
> (`orchestration/agent_dispatch.py` + ORCH-1.45 `AgentTurnEnvelope` queue) as an alternative to the
> default uniform queue-pull — a sealed-bid / second-price auction where workers bid a cost for
> capability-matched tasks. Reuses KG-2.27 calibration as the bid-quality prior and the OS-5.24
> ActionPolicy as the spend gate; clearing prices persist as graph nodes fed back into the OS-5.29
> autoscaler as scarcity signals. Queue-pull stays the default; market mode is one flag.

## Pre-Flight Checklist
- [x] **Extension target identified** — allocation today is centrally pushed by static role strings
  (`DistributedCoordinator.route_task → agent.tasks.<role>`, OS-5.5) or keyword matching
  (`adaptive_agent_router` `RuleBasedPolicy`); ORCH-1.45 dispatch is queue-pull but **uniform**
  (session-key partition, no price). `agent_step_po.py` (AHE-3.15) has RL step-credit but no budget,
  bid, currency, or clearing. There is no mechanism where agents hold/spend a budget and a market clears.
- [x] **New CONCEPT:ORCH-1.46 justified** — a price-coordinated allocation regime is distinct from
  rule/role routing (ORCH-1.x) and RL credit (AHE-3.15); it is the §5.4 "Virtual Agent Economy"
  decentralized self-organization mechanism, and surfaces scarcity/value signals routing cannot.
- [x] **Wire-First confirmed** — 1 allocation backend behind the ORCH-1.45 dispatch seam; bids priced
  via token-budget + KG-2.27 confidence; spend gated by OS-5.24; clearing prices → OS-5.29 signals.
- [x] **Success metric defined** — under market mode, capability-matched tasks are assigned to the
  clearing-winning bidder, an audit-able clearing price is recorded per task, and disabling the flag
  reproduces today's uniform queue-pull byte-for-byte.

## User Stories

### US-1 — Workers bid; the market clears by price
**As** a large agent collective, **I want** tasks allocated by a clearing auction, **so that** the
collective self-organizes allocation and surfaces scarcity/value instead of round-robin.
- **AC1**: a `MarketAllocator` runs a sealed-bid second-price (or continuous double) auction over the
  ORCH-1.45 `AgentTurnEnvelope`; eligible bidders are capability-matched (reuse the existing router's
  capability match), bids = token-budget cost weighted by KG-2.27 calibration confidence.
- **AC2**: the winning bidder is assigned the task and the **clearing price** is persisted as a graph
  node (task id, winner, price, losers' bids) for audit and feedback.
- **AC3**: a bidder with insufficient budget or a task whose spend exceeds the OS-5.24 ActionPolicy cap
  is excluded — no allocation bypasses the spend gate.

### US-2 — Opt-in; scarcity feeds autoscaling
**As** an operator, **I want** the market opt-in and its prices fed back, **so that** scaling reacts to
value/scarcity, not just queue depth.
- **AC4**: market mode is a single dispatch-backend flag; default remains uniform queue-pull
  (`AGENT_DISPATCH_BACKEND=queue`), fully unchanged when off.
- **AC5**: clearing prices are exposed as an OS-5.29 `scaling_signal` so persistently high prices for a
  capability raise that capability's replica target (scarcity → capacity).

## Non-Functional Requirements
- `tests/unit/orchestration/test_orch_1_46_agent_market.py` (`@pytest.mark.concept(id="ORCH-1.46")`),
  ≤60s, no live engine: stub 3 bidders; assert second-price winner + clearing price recorded, an
  over-budget bidder excluded, the ActionPolicy cap enforced, and flag-off reproduces queue-pull.
- `pre-commit run --all-files` green; `scripts/build_concepts_yaml.py` re-run so ORCH-1.46 lands in
  `docs/concepts.yaml`; `scripts/check_concepts.py` passes.
- Per-concept doc under `docs/architecture/` (extend `agent_dispatch.md`), naming KG-2.27 as the
  bid-quality prior and OS-5.24 as the spend gate; relate to ORCH-1.49 (scaling-law measurement of
  whether the market helps).
