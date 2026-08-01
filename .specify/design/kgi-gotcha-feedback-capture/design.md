# Design Document: A hard-won trap is pinned to the code it bit you on, not left to be relearned

> `agent_utilities/knowledge_graph/adaptation/feedback.py:576-590` (capture),
> `agent_utilities/knowledge_graph/retrieval/code_context.py:313-325` (surfacing).

CONCEPT:AU-KG.ingest.gotcha-feedback-capture

## Decision — a `:Gotcha` node keyed by path+note, surfaced automatically when an agent next touches that file

`feedback.py:576-583`.

**The problem, named directly as "the dogfood fix"**: traps like "gen
scripts import the canonical copy, not the worktree" or "`_get_engine()`
hangs in a one-off host process" were rediscovered every session — painful
lessons that lived nowhere durable, so the same mistake cost the same
debugging time repeatedly.

**The rejected alternative**: leaving such traps in ephemeral session
context, a wiki page nobody re-reads before touching the code, or a code
comment that only a reader of that exact file/line sees. None of those
surface the warning AT THE MOMENT an agent is about to repeat the mistake.

**The design chosen**: `record_gotcha` (`feedback.py`) pins a `:Gotcha`
node keyed by a NORMALIZED path + note. The pairing with `code_context`'s
`_gotchas` (`code_context.py:313-325`) is what makes this more than a log
entry: any code-context lookup for a file (the mechanism an agent uses when
it "touches" an area) runs `MATCH (g:Gotcha) WHERE g.path = $fp` and returns
pinned notes+severity for that exact file, so the trap is IN the KG attached
to the code and surfaced ON TOUCH — the correction becomes future behavior
without the agent needing to remember to check a separate gotcha registry.
This concept id is referenced from the MCP `graph_feedback` tool's own
`correction_type` documentation (`write_ingest_tools.py:366`) alongside
sibling correction types (`outcome`, `rule`, `eval`, `action_outcome`,
`selective_erasure`) — `gotcha` is one governed feedback shape among several,
distinguished by being keyed on a code LOCATION rather than an action/agent.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/adaptation/feedback.py`
  (`record_gotcha`), `agent_utilities/knowledge_graph/retrieval/code_context.py`
  (`_gotchas`), `agent_utilities/mcp/tools/write_ingest_tools.py`
  (`graph_feedback` tool's `gotcha` correction type).
- **Backward Compatible**: Yes — additive node type; existing code-context
  lookups without any pinned gotchas simply return an empty list.
- **Breaking Changes**: None.
- **Known weak point**: matching is on NORMALIZED PATH EQUALITY
  (`g.path = $fp`), not a directory/module-scope match — a gotcha pinned to
  one file is invisible when an agent touches a sibling file in the same
  module that shares the same underlying trap, even though the trap likely
  applies there too.
