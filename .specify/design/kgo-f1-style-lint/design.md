# Design Document: A naming-style linter closes the one CI gap the existing validity/connectivity/integrity gates don't cover — pure, read-only, advisory

CONCEPT:AU-KG.ontology.style-lint

> `agent_utilities/knowledge_graph/ontology/style_lint.py`.

## Decision — lint the live `InterfaceRegistry` for naming-convention + typo violations, returning structured issues rather than blocking anything itself

`style_lint.py:1-21` states the gap this closes, analysed against Microsoft's
Ontology-Playground: its `scripts/style-validator.ts` enforces PascalCase
class labels / snake_case property labels plus common-typo detection as part
of its CI gate. This platform already gates the canonical ontology library's
**validity/connectivity** (`check_ontology.py`, see
`.specify/design/kgo-d1-anti-sprawl-gate/design.md`) and **supply-chain
integrity** (`ontology_integrity.py`) — "but nothing checks naming *style* or
flags typos in interface/property names and descriptions. This module is that
missing check," scoped to the live `InterfaceRegistry` (the same
always-populated registry `ontology_interface` already serves).

**The rejected alternative is a blocking style gate** — reject a registration
that violates PascalCase/snake_case convention or contains a likely typo,
mirroring the strictness of the validity gate. The module is explicit about
NOT doing that: "Pure and read-only: never mutates the registry, has no
engine/network dependency, and returns structured issues rather than raising
— callers (the `ontology_interface` MCP tool's `lint` action, a future
pre-commit hook) decide whether a warning/error blocks anything"
(`style_lint.py:18-21`). Style violations are advisory because a naming
convention, unlike a dangling import or a broken OWL-RL closure, doesn't break
correctness — blocking on it would be a much stronger stance than the platform
takes on style anywhere else. The typo table is deliberately conservative:
"a small, curated set of common English typos ... grown deliberately (a real
second false-positive earns an entry), not as a general spell-checker"
(`style_lint.py:37-39`) — the rejected alternative there is a general
spell-checker, which would produce far more false positives against
domain-specific vocabulary than a curated table.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/ontology/style_lint.py`,
  `ontology_interface` MCP tool's `lint` action.
- **Backward Compatible**: Yes — advisory only; no registration is rejected
  by this module.
- **Known weak point**: because lint results never block anything by
  themselves, a naming-convention violation persists indefinitely unless a
  caller (a human, a pre-commit hook that doesn't yet exist) actually acts on
  the `lint` action's output — there is no enforcement backstop.
