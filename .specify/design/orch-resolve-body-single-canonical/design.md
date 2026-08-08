# Design Document: One canonical `resolve_body()` replaces three divergent prompt-body readers that each silently missed a different field

CONCEPT:AU-ORCH.routing.resolve-body-single-canonical

> Realised by `agent_utilities/prompting/structured.py:737-776`
> (`resolve_body`) and `:778-812` (`validate_canonical`); adopted at
> `agent_utilities/agent/registry_builder.py:110`,
> `agent_utilities/orchestration/action_policy.py:100`, and the CI schema check
> `scripts/gen_prompt_schema.py`. Introduced by commit `beb1cf8a`.

## Decision — a single resolver with a stated resolution order, because three independent readers is not a style problem but a correctness one

Three functions read a prompt blueprint's body text independently:
`builder._extract_prompt_content`, `builder.extract_agent_metadata`, and
`registry_builder._resolve_fields`. Each had grown its own idea of where the
body lives.

The commit describes what that produced: the three readers *"each read a
different subset and silently missed `instructions.core_directive` — the bug
that left StructuredPrompt-shaped files (incl. the packaged `main_agent.json`)
with an empty body on the workspace path."* The packaged main agent prompt
resolved to an empty string. Nothing raised: each reader looked in the places
it knew about, found nothing, and returned empty, which is a legal result.

`resolve_body(data)` (`structured.py:737-776`) is now the single source of
truth, with an explicit ordered fallback: `instructions.core_directive`, then
the rendered structured sections, then `""`. The ordering is the substance —
it is the shared answer to "which field wins when several could supply a body",
a question the three readers had each answered differently and none had
written down.

**The rejected alternative is not "three readers" as a stylistic accident — it
is the natural, incremental thing that happens without a canonical
resolver**, and it was rejected because its failure mode is silence. Any
number of independent readers will drift as the blueprint schema gains fields,
and each drift presents as empty output rather than an error, so it survives
until someone notices an agent behaving as though it had no instructions.

`validate_canonical` (`:778-812`) exists to keep the fix from decaying: it is
wired into the CI schema check via `scripts/gen_prompt_schema.py`, so a
blueprint whose body cannot be resolved fails the build rather than shipping
empty.

## Risk Assessment

- **Blast Radius**: `agent_utilities/prompting/structured.py`,
  `agent_utilities/prompting/builder.py`,
  `agent_utilities/agent/registry_builder.py`,
  `agent_utilities/orchestration/action_policy.py`,
  `scripts/gen_prompt_schema.py`.
- **Backward Compatible**: Yes for well-formed blueprints, and strictly better
  for StructuredPrompt-shaped ones that previously resolved empty.
- **Known weak point**: the guarantee holds only for callers that actually go
  through `resolve_body`. Nothing structurally prevents a fourth reader from
  being added — this is a convention plus a CI check on the *data*, not an
  enforced chokepoint on the *code*. A new call site that re-implements body
  extraction would reintroduce exactly the original bug and `validate_canonical`
  would not catch it, since the blueprint itself would still be valid.
