# Design Document: The focused-tools fleet-URL gate honors the SAME private-host allowlist as the multiplexer's egress gate

CONCEPT:AU-ORCH.execution.focused-tools-fleet-egress

> `agent_utilities/orchestration/agent_runner.py:3165-3200`. Introduced by
> commit `238fd3c8` ("focused-tools fleet gate must honor
> MCP_HTTP_ALLOWED_PRIVATE_HOSTS").

## Decision — a fleet MCP endpoint reached over plain HTTP is allowed exactly when it's on the operator-declared private-host allowlist, not only when it's loopback

The focused-tools dispatch path validates a candidate fleet server URL before
using it: scheme must be `http`/`https`, no embedded credentials, no
query/fragment, under 8192 chars. For a plain-`http` URL specifically, it
additionally requires the hostname to be in an allowed set. Before this fix,
that set was **hardcoded to loopback only** (`localhost`, `127.0.0.1`,
`::1`), so a fleet server reached over HTTP from behind a TLS-terminating
ingress — a legitimate, already-supported deployment shape declared via the
operator config `MCP_HTTP_ALLOWED_PRIVATE_HOSTS` — still hard-failed here,
even though the codebase's OTHER egress gate (the multiplexer's own child-MCP
egress check, "Remote MCP child requires HTTPS outside loopback") already
honored that same allowlist correctly.

The result was a confusing failure mode: an operator who had correctly
configured `MCP_HTTP_ALLOWED_PRIVATE_HOSTS` still saw the focused-tools path
degrade with "couldn't reach github-mcp" (the ORCH-1.74 focused-tools
degrade), because THIS gate — a twin of the multiplexer's, not the same
code — hadn't been told about the allowlist. The fix reads
`_agent_config.mcp_http_allowed_private_hosts` and unions it into the allowed
set here too, so the two independent HTTPS-enforcement gates agree.

**The rejected alternative is the two gates staying independently
maintained**, each hardcoding its own notion of "safe to reach over plain
HTTP." That is exactly what produced the bug: two code paths enforcing what
is conceptually the same policy (loopback, or an operator-declared private
host, requires TLS everywhere else) drifted the moment one of them was
extended and the other wasn't. Making both gates read from the SAME
config-declared allowlist is the fix, not adding more hardcoded exceptions to
this one gate specifically.

## Risk Assessment

- **Blast Radius**: `agent_utilities/orchestration/agent_runner.py` (the
  focused-tools fleet-URL validation only).
- **Backward Compatible**: Yes — a deployment that never set
  `MCP_HTTP_ALLOWED_PRIVATE_HOSTS` sees identical (loopback-only) behavior;
  only deployments that already relied on the allowlist for the multiplexer
  gain matching behavior here.
- **Known weak point**: the two egress gates (multiplexer + focused-tools)
  are still two separate code paths that happen to read the same config value
  — nothing structurally prevents a third egress-adjacent check from being
  added later with its own hardcoded loopback-only default, reintroducing the
  same class of drift this fix closed for these two.
