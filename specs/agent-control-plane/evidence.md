# Evidence: agent control plane

**Delivery:** PARTIAL. **Acceptance:** NOT VERIFIED. Existing AU routing, topology and training modules are source evidence only. The target `agent_utilities/layers/` and `agent_utilities/decide/` implementation directories were absent in the checkout inspected for this spec; related work may exist on unmerged branches and must be re-evaluated by exact merged head.

| Requirement | Current source observation | Acceptance evidence required |
|---|---|---|
| AC-01–03 | `graph/builder.py`, `graph/routing/` and `graph/subagent_patterns.py` exist | served EG decision/commit fixture and exact commit/CI URL |
| AC-04 | `rlm/sandboxes/` and external harness integration surfaces exist | five-adapter conformance receipts, launch inspection, negative tests |
| AC-05 | `graph/topology_engine.py` has `ElasticTopologyAdmission` and `record_outcome`; `graph/team_composer.py` selects by success rate | source deletion gate plus topology/capacity/fence integration |
| AC-06–07 | run/trace and evaluation modules exist in AU | durable event replay, independent outcome and promoted-head proof |

Update this file when a PR lands with its merged commit SHA, test commands, CI URL, contract version and artifacts. `PARTIAL` does not mean accepted; do not close this spec from a branch-only result.
