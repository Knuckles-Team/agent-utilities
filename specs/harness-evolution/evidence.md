# Evidence: harness evolution

**Delivery:** PARTIAL. **Acceptance:** NOT VERIFIED. **EH-350:** DEFERRED by design; no authorization to implement a generative model or autograd in EG.

| Requirement | Source observation | Required acceptance evidence |
|---|---|---|
| HE-01–03 | `harness/substrate_trainer.py` emits GRPO job specs through an injected dispatcher; this is not KLPO capture/training | provider capability negatives, exact capture replay and external job receipts |
| HE-04 | `knowledge_graph/research/gaps.py` has canonical Gap helpers, direct Cypher and local sorting; `loops.py` has native WorkItem claim integration | typed atomic Gap/WorkItem and served Decide→commit→claim fixture |
| HE-05 | AU evolution publication path still has local Git and publication shapes | source deletion scan and approved materialization receipt fixture |
| HE-06 | no accepted held-out policy promotion receipt identified | frozen set, independent evaluator, CAS canary and rollback evidence |
| HE-07 | deferred requirement | absence of in-engine generative/autograd code and a separate approved Gap before reconsideration |

When work lands, record merged commit SHA, generated-client version, exact test command and CI URL, receipt digests, measured capture/training resource use and limitations. A branch-only or fixture-only result does not change acceptance status.
