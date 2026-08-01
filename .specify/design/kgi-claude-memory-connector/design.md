# Design Document: The harness's own cross-session memory becomes a KG-native `:AgentMemory` node, not flat markdown

> `agent_utilities/knowledge_graph/core/source_sync.py:4357` (`_sync_claude_memory`).

CONCEPT:AU-KG.ingest.claude-memory-connector

## Decision — ingest the Claude Code `MEMORY.md` topic files as typed, searchable KG nodes

`source_sync.py:4357-4380`.

**The problem, named directly in the docstring**: "The harness keeps its
cross-session memory as flat markdown outside the graph." Every `MEMORY.md`
topic file the harness writes for cross-session continuity lived purely as
disk-resident prose — not queryable, not linkable to any other KG entity, and
invisible to `graph_search`/`graph_query`.

**The rejected alternative is exactly that status quo**: leaving harness
memory as flat markdown, permanently outside the KG the rest of the system
reasons over. It is explicitly framed as a gap, not a neutral choice — the
docstring calls the fix "dogfood[ing] our OWN memory substrate": the same
graph that ingests every other source (Jira, GitLab, Confluence, ...) had no
path for its own operator's memory.

**The design chosen**: each topic file becomes a semantically-searchable
`:AgentMemory` node — name/type/description/body embedded, findable via
`graph_search` — and its `[[other-slug]]` wiki-links become `RELATED_TO`
edges, so a session's accumulated knowledge is connected to the rest of the
ecosystem graph instead of stranded on disk. The connector is deliberately
**zero-infra and offline**: it reads local markdown only, no network call,
scanning `CLAUDE_MEMORY_DIR` when set, else sweeping every
`~/.claude/projects/*/memory` directory — so ingesting the harness's own
memory never depends on a live service being reachable, unlike most other
`source_sync` handlers.

This connector's writes route through the same envelope-atomic path as other
migrated sources — see
`.specify/design/kgi-change-envelope-atomic/design.md` — so a re-ingest of
an unchanged memory file is a content-hash no-op, not a duplicate node.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/core/source_sync.py`
  (`_sync_claude_memory` and its registration in the source-sync dispatch
  table).
- **Backward Compatible**: Yes — purely additive; a host with no
  `~/.claude/projects/*/memory` directories simply ingests zero nodes.
- **Breaking Changes**: None.
- **Known weak point**: the `[[other-slug]]` wiki-link parser assumes the
  harness's own link syntax stays stable; a change to how `MEMORY.md` files
  encode cross-references would silently stop producing `RELATED_TO` edges
  rather than erroring, since a missing link is indistinguishable from a
  topic file that legitimately has none.
