# Design Document: Claude Code's PreToolUse gate is TWO layers — a static, daemon-independent deny floor plus a live `ActionPolicy` consult — instead of one layer alone

CONCEPT:AU-OS.deployment.dynamic-two-fail-closed

> `agent_utilities/claude_harness/pretooluse_gate.py:1-28` (module docstring,
> the two-layer description); invoked via `agent_utilities/cli/__init__.py:85`
> (`agent-utilities harness-gate`).

## Decision — every gated Claude Code tool call passes through a static secret-path/irreversible-command deny (built from `claude_fence`, enforced by the `ECO-4.13` `PermissionPolicyEngine`, works even when graph-os is unreachable) FIRST, then a dynamic consult of the LIVE `ActionPolicy` (OS-5.24) classifying the mapped `ActionRequest` into allow/ask/deny — with any exception, unparseable input, or import failure resolving to `deny`, never a silent allow

This is the hook body Claude Code invokes before every gated tool call. Layer 1 is
static and daemon-independent: a secret-path/irreversible-command deny list
generated from `claude_fence`'s governance-derived fence (see
`AU-OS.deployment.governance-derived-claude-code`), enforced without needing the
graph-os engine to be reachable at all — "so secrets stay protected with the daemon
down" (`pretooluse_gate.py:17-18`). Layer 2 is dynamic and governed: the Bash
verb/file-op is mapped to an `ActionRequest` and classified by the LIVE
`ActionPolicy.classify` against whatever rules are currently loaded (a new
`forbidden` YAML rule, or a KG `governance_rule` node), so a governance change takes
effect on the next tool call with no fence regeneration or CLI redeploy needed. The
whole gate is fail-closed end to end: "any exception, unparseable stdin, or import
failure returns a `deny` — never a silent allow" (`pretooluse_gate.py:26-27`),
mirroring `ActionPolicy.decide`'s own contract.

## Rejected alternative — a single layer: either the static fence alone, or the dynamic governed gate alone

Both single-layer shapes are named and rejected by what the two-layer design is
built to cover. Static-only (just the `claude_fence`-derived deny list, no live
`ActionPolicy` consult) is what the SEPARATE static settings.json fence already
provides on its own (`AU-OS.deployment.governance-derived-claude-code`) — it cannot
react to a governance rule change without regenerating and rewriting that file, and
"the static half of the fence" is explicitly described as only "the floor"
(`pretooluse_gate.py:9-10`), not the whole gate. Dynamic-only (consult
`ActionPolicy` live, skip the static floor) was rejected because it makes tool-call
safety depend entirely on the graph-os engine being reachable at decision time —
exactly the daemon-down case the docstring calls out: "this consults the live
`ActionPolicy`... so governance rules... take effect without re-writing the fence,"
but if the daemon that serves `ActionPolicy` is down, a dynamic-only gate has
nothing to fail closed AGAINST except its own exception path, leaving secrets
protected only by chance rather than by a dedicated, daemon-independent floor.
Layering both means secrets stay protected even with the daemon down (static), and
governance changes still propagate without a redeploy (dynamic) — each layer covers
exactly the case the other cannot.

## Risk Assessment

- **Blast Radius**: `agent_utilities/claude_harness/pretooluse_gate.py`,
  `agent_utilities/claude_harness/claude_fence.py`,
  `agent_utilities/orchestration/action_policy.py`.
- **Backward Compatible**: Yes — a hook body invoked by Claude Code's own
  PreToolUse mechanism; no change to gated tools themselves.
- **Known weak point**: the static layer's deny list is only as current as the last
  fence regeneration (`claude_fence` run) — a `forbidden` rule added to
  `ActionPolicy` after the fence was last written is enforced by the DYNAMIC layer
  correctly, but is invisible to layer 1 until the fence is regenerated, so the
  daemon-down fallback protection for a brand-new rule lags behind the governed one.
