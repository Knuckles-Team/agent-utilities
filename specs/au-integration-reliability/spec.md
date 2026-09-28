# AU integration and release reliability

**ID:** AU-INTEGRATION-001 · **Owner:** agent-utilities · **Delivery:** SPECIFIED; item-level historical state requires an exact-head audit.
**Items:** EH-103, EH-187, EH-202–205, EH-207–210, EH-321 (cross-owner gate), EH-344, EH-365, EH-467, EH-651, EH-657. Additional named workstreams: AU final composition, AU ontology/SHACL/OWL clean cut, AU public application control plane, and AU public docs. These workstream labels need stable IDs assigned through review; no synthetic ID is treated as authoritative.

## Outcome

AU's final composition has no silent fallback, stale provider contract, red baseline test, abandoned worktree state, or undocumented authority path. Public contributors can reproduce the release checks and understand exactly which AU obligations are implemented, verified or waiting.

## Requirements

1. **Control state:** the reserved `__control__` graph is created or a typed error is returned before first work item; no content-graph fallback. `messaging_intake_enabled` reaches the backend that applies it. Clustered mode refuses direct Agent Library writes deliberately, through a tested policy path (EH-103, EH-187).
2. **Dependencies and client contract:** remove unintended heavy `nltk` dependency; keep EG as a declared core dependency; generate its client contract gate from a pinned public artifact rather than a sibling checkout (EH-203–205).
3. **Test baseline:** preserve regression tests for lazy config resolution, resource lease fencing, session boundary, model factory routing, gateway and deployment backend behavior. Resolve liveness deferrals with evidence and keep an auditable expiry workflow (EH-209–210).
4. **Documentation and privacy:** a moved external graph contract check retains content markers and privacy scan in its new owner. AU's author identity scanner accepts only exact canonical automation identities while still rejecting credential and private endpoint leakage (EH-365, EH-467). GitHub Pages is the detailed public documentation home; a stale project `docs/` deployment check cannot block unrelated PRs.
5. **Legacy removal:** retire the exposed Fuseki/Stardog publisher, schedule token, config, health/widget and docs paths when no live caller exists. If publication is needed, EG must expose one atomic named-graph method; AU does not revive a local publisher (EH-657).
6. **Research gate:** AI predicate grouping, prompt/KV reuse and GPU fairness are measured against the typed EG/AU serving path with tenant/purpose constraints before integration. A benchmark proposal is not an accepted implementation (EH-651).
7. **Lane integrity:** before pruning a branch/worktree, prove its commits and uncommitted diff are represented in an accepted public branch or an explicit recovery artifact. AU final composition, AU ontology/SHACL/OWL clean cut, AU public application control plane, and AU public docs record exact merged commits and tests per item. A zero-ahead branch alone does not prove it is disposable (EH-344).
8. **Disposition:** EH-207 is superseded only when its replacement has independent ground-truth proof; EH-208's rejected path stays rejected. EH-202 is a historical composition claim that must be checked against exact main history. EH-321's EG-owned Kafka gate is reported as dependency evidence, never misrepresented as AU code proof.

## Acceptance

Every item has an explicit disposition and exact source/test evidence, mandatory gates are reproducible in public CI, and final AU composition passes source, contract, security, quality and portable integration checks at the same merged head.
