# Design: Messaging-Orchestration Transparency

> Every messaging reply (Telegram and every other backend — they all share the universal reply
> path) surfaces the agent-graph's routing + failures, TRANSLATED into clean, actionable chat
> text, so a failure is a troubleshooting entry point, never a black box. Extends ORCH-1.21
> (KG-to-LLM Execution Bridge / `run_agent`) and CONCEPT:AU-ECO.messaging.universal-graph-agent
> (the messaging router's universal reply path); composes with the existing `RunTrace`/
> `:ToolCall` provenance surface (KG-2.296) instead of inventing a parallel tracing system.

## Investigation findings (current state, before this change)

A user messaged the Telegram bot "does my github org have issues/PRs". The agent graph
correctly routed to `github-mcp` via the FOCUSED-TOOLS altitude
(`agent_runner.run_agent`'s `elif getattr(shape, "tool_servers", ())` branch, ORCH-1.74) and
failed the fleet HTTPS gate (`_fleet_server_url` raising `RuntimeError: fleet MCP endpoint
requires HTTPS outside loopback`). `_fleet_server_failed_result()` (~L1712) DID compose a
truthful failure string including the error. The user nonetheless received only a generic
"some sort of failure" — traced to **four independent swallowed-cause sites** on the same
path, not one:

1. **The reply-budget wall.** `messaging/router.py`'s `_graph_agent_reply` wraps the whole
   `Orchestrator.execute_agent` call in `asyncio.wait_for(..., timeout=reply_timeout)`. When
   the run (KG resolution + focused-tools connect attempt + failure composition) takes longer
   than the budget, the coroutine is CANCELLED before it can return the already-composed
   truthful string — the `except TimeoutError:` branch returned a fixed generic message with
   zero information about what was attempted or why.
2. **A hardcoded RunTrace error string.** `run_agent`'s Step 5 recorded
   `error="delegation produced no usable data (degraded)"` on EVERY degraded outcome,
   discarding the real cause even when one was sitting right in `result.results.output`
   (`_fleet_server_failed_result`'s own composed message) — so even a `get_run_trace` lookup
   after the fact could not recover the real reason.
3. **A raw dict-repr fallback.** `run_agent`'s final render fell back to `str(result)` — the
   whole Python dict, not the message — whenever `results.output` was empty but a real cause
   sat in `result["error"]` (the `execute_graph` critical-failure branch), or whenever
   `execute_graph`'s terminal `error_recovery_step` `End({"error": ..., "results": {...}})`
   was (incorrectly) stringified into `results.output` under `status="completed"`.
4. **The plain-chat fallback fired on ANY unusable/errored reply** (`except Exception as e:`
   logs `e` then calls `_plain_chat_reply`, which drafts an UNRELATED conversational answer)
   — the real cause reached the log line and was then dropped from the user-facing text.

What already exists and is reused (Extend-Before-Invent):
- **`RunTrace`/`:ToolCall` provenance** (`observability/trace_ontology.py`,
  `agent_runner._record_execution_trace`/`_persist_tool_calls`) — the durable
  "troubleshooting entry point" this feature links to via `trace_ref`, not a new store.
- **`_delegation_degraded`/`_fleet_server_failed_result`** (ORCH-1.74,
  AU-ORCH.execution.no-silent-hallucination) — the existing truthful-failure composition;
  this feature makes their output actually REACH the chat user.
- **`_flatten_exception_group`** — the existing BaseExceptionGroup→leaf-message unwrap this
  feature's translation table consumes.
- **`core/log_privacy.sanitize_log_text`** — the existing endpoint/path/email redactor,
  reused (not reimplemented) to keep a failure detail privacy-safe once it rides out to an
  external chat surface (a MORE exposed boundary than the internal logs it was built for).

## Proposed design

### A. `run_summary` — a structured, opt-in field on every terminal `run_agent` outcome
`{route: {agents, servers, why}, outcome: ok|degraded|failed|timeout, stage_reached, trace_ref,
failure?: {raw, translated, category, hint}}`. `route`/`stage_reached` are tracked ALONGSIDE
the existing dispatch `if/elif` chain (bound-template / focused-tools / single-server / full
graph) — the SAME branch that already runs, not re-derived after the fact. Populated at the
four existing terminal points: `_fleet_server_failed_result`'s degraded exit, the generic
`_delegation_degraded` exit, the outer failure handler, and — best-effort — a `CancelledError`
before it re-raises (see §C). Opt-in via `include_run_summary=True` (new `run_agent`/
`Orchestrator.execute_agent` param, additive, default off): forces the existing envelope
(`_render_agent_result`, the same mechanism `return_mermaid`/`channel_id` already use) to
include `run_summary` — the bare-string contract is untouched for every existing caller.

### B. `failure_translation.py` — a data-driven error → {translated, category, hint} registry
A new leaf module: an ordered tuple of `{category, markers (case-insensitive substrings),
translated, hint}`, first match wins, most-specific markers first. Covers the fleet HTTPS gate,
`delegated_child_tool_failed`, the retrieval-quality gate, the reply-budget timeout,
engine-unreachable, access-denied/RBAC, tool wall-clock timeout, toolset-bind failure, and a
security-block, plus a generic fallback that ALWAYS keeps a sanitized raw error tail — never a
bare "failure". `build_failure_detail(raw)` is the one function both `agent_runner.py` and
`router.py` call to get the ready-to-store `{raw, translated, category, hint}` dict.

### C. The reply-budget-timeout path — the one exit `run_agent` cannot describe itself
A caller-side `asyncio.wait_for` cancellation tears the coroutine down before it can return
anything — no result, no envelope. Two changes close this gap without a new tracing system:
- The router **pre-mints** the run handle (`run_identity.new_run_id()`) BEFORE the call and
  passes it through (`run_id=`, new optional `run_agent` param — `None` still mints fresh,
  unchanged for every other caller), so a `trace_ref` is fixed regardless of how the call ends.
- `run_agent`'s `CancelledError` branch best-effort writes a `status="timeout"` `RunTrace` for
  that SAME id (reusing `_record_execution_trace`, a synchronous call — safe post-cancellation)
  before re-raising, so the `trace_ref` resolves to a REAL node.
- The router itself synthesizes the best-KNOWN `run_summary` from what it already knew BEFORE
  the call (the planned route, from the per-job `shape.tool_servers` the caller already
  computed) — `_timeout_run_summary`. Never a bare generic message again.

### D. The router renders it — one footer, four exit points
`_transparency_footer(run_summary)` → `""` on `outcome == "ok"` (or the
`MESSAGING_TRANSPARENCY_FOOTER` opt-out), else `⚠️ {translated} — {hint} (trace: {trace_ref})`.
`_with_transparency(text, run_summary)` appends it. Applied at EVERY exit of
`_graph_agent_reply`: the normal success/degraded return, the backend-timeout graceful message,
the caller-side reply-budget timeout (§C), and — new — the plain-chat fallback (both the
"no usable reply" and the genuine-exception branches), which previously discarded the cause
entirely in favor of an unrelated conversational answer.

## Concept & wiring
- **New CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency** — sub-concept of
  ORCH-1.21 (`run_agent`, the KG-to-LLM execution bridge) and
  CONCEPT:AU-ECO.messaging.universal-graph-agent (the messaging router's universal reply
  path); composes with ORCH-1.74/AU-ORCH.execution.no-silent-hallucination's existing
  truthful-failure composition and the KG-2.296 `RunTrace`/`:ToolCall` provenance surface. No
  new concept was extendable in place of this one: the nearest neighbors (ORCH-1.21 itself,
  AU-ORCH.execution.no-silent-hallucination) describe HOW a failure is detected/composed
  server-side, not how it is TRANSLATED + surfaced to an external chat user with a
  troubleshooting handle — a genuinely new cross-cutting concern (a chat-facing rendering
  layer over an existing detection mechanism), hence `new` rather than `augment`/`specialize`.
  Wire-First: ≤2 hops from `messaging/router.py`'s `_graph_agent_reply` →
  `Orchestrator.execute_agent(include_run_summary=True)` → `run_agent`'s existing dispatch.
- Marker sites: `agent_utilities/messaging/router.py`, `agent_utilities/orchestration/
  agent_runner.py`, `agent_utilities/orchestration/engine.py`, `agent_utilities/orchestration/
  failure_translation.py`, `agent_utilities/orchestration/manager.py`.

## Wire-First path
`Backend.listen()` → `InboundRouter.planner_handler` → `_reply_to_burst` → `_graph_agent_reply`
→ `Orchestrator.execute_agent(include_run_summary=True)` → `run_agent` (builds `run_summary` at
its 4 terminal points) → JSON envelope → router unwraps `output` + `run_summary` → footer
appended → `backend.send_message`. ≤2 hops from the messaging entrypoint to `run_agent`'s
existing dispatch; no new server, store, or transport.

## Risk assessment
- **Blast radius**: `agent_runner.run_agent` (additive params + a durable-trace fix on the
  degraded path), `orchestration/manager.py` (param forwarding), `orchestration/engine.py`
  (one new failure-shaped branch in `execute_graph`'s terminal-result handling), `messaging/
  router.py` (`_graph_agent_reply`'s footer wiring). One new leaf module
  (`failure_translation.py`) with zero inbound dependents to break.
- **Backward compatible**: yes — `include_run_summary`/`run_id` are opt-in additive params;
  `_render_agent_result`'s bare-string contract is unchanged unless a caller opts in; the new
  `execute_graph` branch only fires for a shape (`dict` with a truthy `error` key) the OLD code
  silently mishandled (stringified into a fake "completed" answer), so no existing caller could
  have depended on that behavior.
- **Breaking changes**: none.

## Critical files
- `agent_utilities/orchestration/failure_translation.py` (new — the translation registry)
- `agent_utilities/orchestration/agent_runner.py` (`run_summary` construction at the 4 terminal
  points; `route`/`stage_reached` tracking; the `CancelledError` best-effort trace write; the
  two swallowed-cause fixes — the hardcoded degraded-error string and the dict-repr fallback)
- `agent_utilities/orchestration/manager.py` (`Orchestrator.execute_agent` param forwarding)
- `agent_utilities/orchestration/engine.py` (`execute_graph`'s terminal-error-dict branch)
- `agent_utilities/messaging/router.py` (`_graph_agent_reply`'s footer wiring; the
  `_timeout_run_summary`/`_exception_run_summary` router-side synthesis)
- `agent_utilities/observability/trace_ontology.py` (`trace_id` — the `trace_ref` this all
  keys off; unmodified, reused as-is)
- `agent_utilities/core/log_privacy.py` (`sanitize_log_text` — reused for the failure detail's
  raw-text redaction)
