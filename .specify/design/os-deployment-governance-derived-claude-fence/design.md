# Design Document: Generate Claude Code's `settings.json` permission fence FROM the live `ActionPolicy` + a hard-coded safety floor, instead of hand-writing/hand-maintaining it

CONCEPT:AU-OS.deployment.governance-derived-claude-code

> `agent_utilities/claude_harness/claude_fence.py:1-29` (module docstring);
> CLI entry point `agent_utilities/deployment/cli.py:48-53` (`harness-fence`).

## Decision — `claude_fence.py` derives the unattended-Claude-Code `settings.json` (`allow`/`ask`/`deny` + `defaultMode`) and `.claudeignore` from two sources of truth the platform already owns — a hard-coded irreversible-command/secret-file safety floor, and the LIVE `ActionPolicy` ruleset (every `forbidden` rule → static `deny`, every `approval_required` rule → `ask`) — regenerated on every run, rather than maintaining the fence as a hand-written file

Running Claude Code unattended needs a permission fence so the CLI can act "while
you sleep" without a vague prompt force-pushing `main` or a cleanup command nuking
a directory (`claude_fence.py:8-9`). The common approach elsewhere is a
hand-written `settings.json`. This module instead DERIVES it: layer one is a
hard-coded baseline (irreversible commands, secret-file paths); layer two reads the
live `ActionPolicy` and maps its rule tiers onto Claude Code's own permission
vocabulary (`forbidden` → `deny`, `approval_required` → `ask`, where the rule's
fleet *kind* maps to a shell/MCP surface the CLI can emit). Because the deny list
is regenerated from the live policy EVERY run, "adding a `forbidden` governance
rule... propagates into the IDE fence automatically — it surpasses a static
hand-edited file" (`claude_fence.py:22-24`). `defaultMode` is hard-pinned to
`acceptEdits` and `bypassPermissions` is never emitted — not derivable, not
overridable by policy content. The companion DYNAMIC gate
(`AU-OS.deployment.dynamic-two-fail-closed`) consults the same live policy at
decision time; together "they are the fence the article draws once, made
self-updating" (`claude_fence.py:25-26`).

## Rejected alternative — a hand-written, manually maintained `settings.json`

The module's own opening sentence names the alternative it replaces: "A
hand-written file is the common approach; this module **derives** the fence
instead" (`claude_fence.py:11-12`). A hand-written fence has to be manually kept in
sync with every governance-rule change — a new `forbidden` rule added to
`ActionPolicy` (in `deploy/action-policy.default.yml` or a KG `governance_rule`
node) has NO effect on a hand-written `settings.json` until someone remembers to
edit it too, which is exactly the drift risk a governance system exists to close.
Deriving the fence instead makes the IDE-level static floor track the SAME source
of truth the dynamic gate already consults, so the two can never silently diverge
from each other the way a hand-maintained file and a live policy inevitably would.

## Risk Assessment

- **Blast Radius**: `agent_utilities/claude_harness/claude_fence.py`,
  `agent_utilities/deployment/cli.py` (`harness-fence` subcommand); the target
  `~/.claude/settings.json` + `.claudeignore`.
- **Backward Compatible**: Yes — an explicit CLI-invoked generation step; nothing
  regenerates the fence automatically without the operator running `harness-fence`.
- **Known weak point**: "regenerated on every run" means the fence is only as
  current as the LAST time `harness-fence` was invoked — unlike the dynamic gate
  (`AU-OS.deployment.dynamic-two-fail-closed`), which consults the live policy on
  every tool call, this static file can still lag a freshly added governance rule
  until the CLI is re-run.
