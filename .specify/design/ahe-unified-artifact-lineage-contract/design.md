# Design Document: One lifecycle contract for every evolvable artifact, not one per artifact kind

CONCEPT:AU-AHE.harness.unified-artifact-lineage

> `agent_utilities/models/knowledge_graph.py:1751-1763` (`ArtifactVersionNode`),
> generalizing `SkillVersionNode`/`PromptVersionNode`;
> `agent_utilities/orchestration/artifact_promotion.py` (the promotion gate it
> feeds).

## Decision — `ArtifactVersionNode` is the ONE lifecycle contract every content-addressed artifact version implements

`ArtifactVersionNode` (`knowledge_graph.py:1751`) generalizes the propose-only
lifecycle contract `SkillVersionNode` already had — `status`:
`proposal | active | rejected`, a version is ALWAYS persisted, only a
promotion gate flips it to `active` — to every vector on the unified
promotion gate: skill markdown, system prompt, MCP tool description, and
native `eg-program` revisions. `SkillVersionNode`/`PromptVersionNode` become
thin subclasses of it, with additive fields carrying safe defaults and EXACT
existing field names preserved on both.

**The rejected alternative is what preceded this: a separate version-node
type per artifact kind**, each independently implementing its own
propose/active/reject lifecycle. The docstring names the migration
discipline explicitly: "the strangler-then-delete pattern, not a parallel v2
type" (citing the repo's No Legacy rule) — the existing per-kind types are
narrowed into subclasses of the general contract in place, rather than a new
`ArtifactVersionNodeV2` being introduced alongside the old types while
callers slowly migrate. That choice means every existing query or
constructor call against `SkillVersionNode`/`PromptVersionNode` keeps working
unmodified, because the field names and defaults are preserved exactly.

## Risk Assessment

- **Blast Radius**: `agent_utilities/models/knowledge_graph.py`,
  `agent_utilities/orchestration/artifact_promotion.py`, every caller of
  `SkillVersionNode`/`PromptVersionNode`.
- **Backward Compatible**: Yes by construction — the subclass migration is the
  entire point.
- **Known weak point**: a new artifact kind (`artifact_kind` is a free-text
  field, not a closed enum) can be introduced by any caller without a
  corresponding review of whether the generalized contract's fields
  (`origin`, `status`) actually make sense for it.
