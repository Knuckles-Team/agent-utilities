# Design Document: A detected repetition loop is persisted as a cross-session ExperienceNode, not just denied in the moment

CONCEPT:AU-AHE.harness.experience-node-architecture

> `agent_utilities/security/execution_stability_engine.py:31-38,246-265`.

## Decision — on DENY, the repetition pattern is converted into a KG-persistable tactical rule, not just blocked and forgotten

`RepetitionGuard` (adapted from Goose's `RepetitionInspector`) detects a tool
called N times in a row with identical arguments and returns
ALLOW/WARN/DENY. `create_experience_node` (`execution_stability_engine.py:246-265`)
is invoked specifically on a DENY verdict: it builds a KG-persistable
`ExperienceNode` — a `condition` (e.g. "Tool 'X' called N times consecutively
with identical arguments") paired with an `action` — so "the agent avoids the
same pattern in future sessions" (`execution_stability_engine.py:35-38`).

**The rejected alternative is denying the repeated call in the moment and
letting the pattern be forgotten once the session ends** — the guard's core
detection (argument hashing, consecutive-count tracking, warn-before-deny)
would still prevent the immediate loop, but a *new* session hitting the same
condition would have to rediscover it from scratch. Persisting the
condition/action pair as a graph node instead makes the loop pattern a
durable, queryable fact: a future session (or a future planning step) can
check against accumulated `ExperienceNode`s before attempting an action that
has already been flagged as a loop, rather than relying on the same
in-session guard to catch it again after it's already repeated.

## Risk Assessment

- **Blast Radius**: `agent_utilities/security/execution_stability_engine.py`
  only — `create_experience_node` returns a plain dict; the KG-persistence
  step itself is the caller's responsibility (not shown in this module).
- **Backward Compatible**: Yes — `create_experience_node` returns `None` for
  any non-DENY verdict, so nothing changes for ALLOW/WARN behavior.
- **Known weak point**: the `condition` string is templated directly from the
  tool name and consecutive count with no normalization beyond that — two
  semantically identical loops phrased with slightly different argument
  shapes would produce different `ExperienceNode` conditions and not be
  recognized as the same pattern later.
