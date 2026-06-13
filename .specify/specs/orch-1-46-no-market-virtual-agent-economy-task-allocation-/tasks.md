# Tasks: Agent-Market Task Allocation (ORCH-1.46)

Wire-first; compose existing primitives behind the dispatch seam.

1. **Read the seams.** `orchestration/agent_dispatch.py` + `agent_dispatch_worker.py` (ORCH-1.45
   queue-pull, `AgentTurnEnvelope`), `orchestration/adaptive_agent_router.py` (capability match),
   `knowledge_graph/retrieval/capability_index.py` + KG-2.27 calibration, `orchestration/action_policy.py`
   (OS-5.24 spend gate), `orchestration/fleet_autoscaler.py` + `scaling_signals.py` (OS-5.29).
2. **Allocator.** Add `orchestration/agent_market.py::MarketAllocator` — sealed-bid second-price auction
   over capability-matched bidders; bid = token-budget cost × KG-2.27 confidence.
3. **Clearing nodes.** Persist a clearing-price graph node per task (winner, price, losing bids).
4. **Spend gate.** Enforce the OS-5.24 ActionPolicy cap and per-agent budget in bid eligibility.
5. **Wire the seam.** Add a `market` dispatch-backend mode selecting `MarketAllocator` instead of uniform
   queue-pull; default stays `queue` (no behavior change when off).
6. **Scarcity feedback.** Emit clearing prices as an OS-5.29 `scaling_signal`.
7. **Test** `tests/unit/orchestration/test_orch_1_46_agent_market.py` per the spec ACs.
8. **Gates.** `pre-commit run --all-files`; regenerate `docs/concepts.yaml`; `scripts/check_concepts.py`;
   extend `docs/architecture/agent_dispatch.md`.
