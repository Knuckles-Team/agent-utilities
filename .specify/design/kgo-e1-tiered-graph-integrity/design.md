# Design Document: Graph integrity validation is FOUR tiers of decreasing severity, not one pass/fail check

CONCEPT:AU-KG.ontology.graph-integrity-validator

> `agent_utilities/knowledge_graph/security/graph_validator.py`.

## Decision — auto-fix silently, log integrity violations, flag quality concerns, and raise ONLY on catastrophic failure

`graph_validator.py:4-18` states the tiering directly, "Inspired by
Understand-Anything's `graph-reviewer` agent": **Tier 1 (Auto-fix)** silently
corrects recoverable issues (nulls → defaults, LLM type aliases,
weight/score clamping); **Tier 2 (Integrity)** detects referential integrity
violations (dangling edges, invalid node references, duplicate IDs) and logs
them as warnings; **Tier 3 (Quality)** flags softer quality concerns (orphan
nodes, generic summaries, self-referencing edges, missing descriptions);
**Tier 4 (Fatal)** raises, and ONLY on catastrophic failures (zero valid
nodes, missing critical schema, broken graph structure). `ValidationReport.is_healthy`
(`graph_validator.py:179-182`) is defined narrowly — "True if no tier-2
violations or tier-4 fatals" — deliberately excluding tier-1 (already fixed,
nothing to report as unhealthy) and tier-3 (quality concerns worth surfacing
but not health-affecting) from the health determination.

**The rejected alternative is a single pass/fail validator** — the obvious
simpler design, and the one that would force an operator to choose between
"reject the graph for a null field that has an obvious default" (too strict,
blocks routine operation) or "never reject anything" (too permissive, misses
real referential-integrity breakage). Four tiers let each severity get the
RIGHT response: silent repair for the recoverable, a logged warning for the
structurally-broken-but-non-fatal, a review flag for cosmetic quality, and a
hard stop reserved for the truly catastrophic — a graph with zero valid nodes
or a broken schema is not safely usable regardless of how the caller wants to
handle lesser issues.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/security/graph_validator.py`
  (`GraphValidator`, `ValidationReport`, `ValidationIssue`), every caller of
  `GraphValidator(engine).validate()`.
- **Backward Compatible**: Yes — a new non-blocking validation layer; it does
  not change write paths, only reports on/auto-fixes graph state.
- **Known weak point**: Tier-1 auto-fixes are applied SILENTLY — a caller
  that only inspects `is_healthy` (which ignores tier-1 by design) would never
  learn that data was actually mutated (a null coerced to a default, a score
  clamped) unless it explicitly reads `tier1_fixes` off the report.
