# Design Document: A prompt hardening cycle is propose-only by default; the winner is never applied silently

CONCEPT:AU-AHE.harness.hardening-transparency-surface

> `agent_utilities/harness/evolve_agent.py:489-540,569-580` (primary — gated
> apply + audit trail), `agent_utilities/harness/memorydata/adapter.py:4-19`
> and `agent_utilities/harness/memorydata/client.py:4-14` (pointer — the same
> never-silent discipline applied to transport selection).

## Decision — a hardened prompt candidate is committed only when it beat baseline AND the auto-apply gate is on; otherwise it is written as an audit record, never silently discarded or silently applied

`EvolveAgent.apply_edits` (`evolve_agent.py:489-540`) states the rule
directly: for a system-prompt edit carrying reference-only native compiled
state, the state is attached and committed **only when (a)** it beat baseline
(`edit.metadata['promote']`) **and (b)** the `KG_AGENT_AUTO_APPLY` gate is on
(`_auto_apply_enabled`, default OFF/shadow, `evolve_agent.py:540-547`).
Otherwise the cycle is propose-only: a queryable `ProposedPromptChange` audit
record is written — before/after metric, the decision, and the
rejected/held candidate — and the live prompt is left untouched. The
docstring is explicit about why: "a prompt rewrite is high-impact, so it is
never silent."

**The rejected alternative is applying whichever candidate wins without a
persisted decision trail** — either always auto-applying the winner (no
human visibility into what changed and why) or silently discarding a losing
candidate with no record it was ever tried. Both lose the same thing: an
auditable "why is the prompt what it is today" trail. The chosen shape keeps
that trail regardless of which branch fires — `dry_run` additionally forces
propose-only even when auto-apply is otherwise enabled, so a caller
verifying behavior never accidentally commits a live change.

### Pointer — the same never-silent discipline applied to memory-transport selection

`adapter.py:4-19`, `client.py:4-14`. `GraphOSMemoryMethod` implements the
MemoryData `send_message` contract over a pluggable `MemoryBackendClient`;
`client.py` names two transports explicitly — `MockBackendClient`
(deterministic, dependency-free, `transport="mock"` default) and
`GraphOSRestClient` (the live engine). The decision this shares with the
prompt-hardening cycle: which transport is actually in effect is always an
explicit, inspectable setting (`transport=`), never inferred or silently
substituted — a caller running the offline test suite or a dry run knows
unambiguously whether it exercised a live engine or a deterministic
stand-in, the same discipline that keeps a prompt-hardening decision always
visible rather than assumed.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/evolve_agent.py`,
  `agent_utilities/harness/memorydata/adapter.py`,
  `agent_utilities/harness/memorydata/client.py`.
- **Backward Compatible**: Yes — `KG_AGENT_AUTO_APPLY` defaults OFF; existing
  callers get propose-only behavior unless they explicitly opt in.
- **Known weak point**: `_auto_apply_enabled` fails closed (returns `False`)
  on any config-read exception — safe by default, but a misconfigured
  environment that silently can't read `kg_agent_auto_apply` looks
  indistinguishable from one that deliberately has it off, with only a
  swallowed exception (`noqa: BLE001`) as the difference.
