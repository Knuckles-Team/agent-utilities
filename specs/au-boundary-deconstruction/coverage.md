# AU program coverage and disposition

This map uses stable item IDs so delivery can be audited per obligation. A mention here is a routing decision, not acceptance. The sibling specs contain the complete AU behavior, design and test contract. The same ID can have a separate owner implementation in another public repository; AU completion still requires its own caller/deletion proof.

| AU capability or disposition | Items | Native spec or proof |
|---|---|---|
| Agent execution, routing, swarm, task claims | RF-029, EH-035–039, EH-048, EH-206, EH-453–464 (AU) | [`agent-control-plane`](../agent-control-plane/spec.md) |
| Harness capture, training, work market and governed change | EH-346–349 | [`harness-evolution`](../harness-evolution/spec.md) |
| Security, identity, guardrails and cache invalidation | EH-097, EH-379, EH-401–403, EH-405, EH-407, EH-541 (AU) | [`agent-security-policy`](../agent-security-policy/spec.md) |
| Portable tests, liveness and type coverage | EH-103, EH-176, EH-187, EH-209–210, EH-380, EH-385–386, EH-391, EH-467 | [`au-boundary-quality`](../au-boundary-quality/spec.md), with historical accepted IDs to be audited before status change |
| Semantic schema and generated EG client | EH-204–205, EH-367, EH-377, EH-389 (AU), EH-431, EH-470–474, EH-493–510, EH-657 | [`au-semantic-client`](../au-semantic-client/spec.md) |
| Served, connector, graph and state authority cuts | EH-368, EH-476–516 except EH-513 | [`au-boundary-deconstruction`](spec.md) |
| Retrieval/context and finance agent roles | EH-394, EH-397–399, EH-419, EH-423, EH-513, EH-704 (AU) | [`agent-context-and-finance`](../agent-context-and-finance/spec.md) |
| Public contributor environment and skill lifecycle | EH-518 (AU), EH-630 (AU), EH-654 | [`au-developer-environment`](../au-developer-environment/spec.md) |
| Historical release and named AU workstreams | EH-202–203, EH-207–208, EH-321, EH-344, EH-365, EH-651; AU final composition, AU ontology/SHACL/OWL clean cut, AU public application control plane, AU public docs | [`au-integration-reliability`](../au-integration-reliability/spec.md) |

For multi-owner IDs, the native AU spec describes its own boundary and explicit input/output contract; it does not make another repository's implementation state an AU acceptance claim. Each contributor must confirm the current source tree and evidence at the exact head before moving a task state.
