# Design Document: A package skill authored to drive a fleet server's tools is upgraded to run against them, not left prompt-only

CONCEPT:AU-ORCH.execution.skill-bound-server-tools

> `agent_utilities/orchestration/agent_runner.py:1954-2010`
> (`_bind_skill_to_owning_server`) and its call site at `agent_runner.py:2133`.
> Introduced by commit `debb3c59` ("skills bind their owning MCP server's
> tools (F7)").

## Decision — resolve a fleet-provider skill as a single-server AGENT bound to its own server's tools, keeping the skill's SOP as the system prompt

A skill contributed by a fleet MCP provider (e.g. a `github-mcp`-authored
skill) is written assuming it can drive that provider's tools — its
instructions describe steps that call specific tools by name. Resolved as a
bare `AGENT_SKILL` resource, a skill runs **prompt-only**: `meta["type"] =
"skill"`, no tools attached, so it can only *describe* what should happen,
never execute it (`agent_runner.py:2124-2132`, `_hydrate_skill_runnable`).

`_bind_skill_to_owning_server` closes that gap. Given the skill's
`provider_ref` (a privacy-safe `provider://<name>` identity persisted at
ingestion time — filesystem paths are deliberately not retained), it tries
`<provider>-mcp` then `<provider>` as candidate server names against the KG
backend. On a hit, it sets `meta["type"] = "server"` and populates the
server's tool list, so the run routes through the single-server focused-tools
path — **while keeping the skill's own instructions as the system prompt**,
rather than replacing them with a generic server-agent prompt. The skill's
authored SOP now drives real tool calls against the server it was written
for.

**The rejected alternative is the status quo it replaces: run every skill
prompt-only, regardless of what it was authored to do.** That is safe by
default (a skill can never accidentally call tools it wasn't vetted for) but
makes any skill that names concrete actions ("open this ticket", "search this
index") strictly descriptive — it produces a plan, not a result, for exactly
the class of skill this mechanism targets: one contributed by the same fleet
provider whose server it's meant to drive. The fix is scoped narrowly
(explicitly excludes `provider` values of `"agent-utilities"`,
`"configured-overlay"`, `"xdg-local"` — first-party/local skills are never
auto-upgraded to a server binding) and is **best-effort**: a provider whose
server can't be resolved simply leaves the skill prompt-only, the pre-existing
safe behavior — there is no failure mode where a miss breaks the run.

## Risk Assessment

- **Blast Radius**: `agent_utilities/orchestration/agent_runner.py`,
  `tests/unit/test_skill_tool_binding.py`.
- **Backward Compatible**: Yes — a skill whose provider doesn't resolve to a
  registered server behaves exactly as before (prompt-only).
- **Known weak point**: the `<provider>-mcp`/`<provider>` naming convention is
  a heuristic, not a declared mapping — a fleet server registered under a
  name that doesn't match either pattern silently leaves its skills
  unupgraded, with no signal that a binding was attempted and missed.
