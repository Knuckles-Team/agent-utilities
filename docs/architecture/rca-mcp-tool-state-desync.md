# RCA: graph-os fleet-mount bookkeeping disagreed with the callable tool surface (D-OB-3)

CONCEPT:AU-OS.governance.truthful-state-invariant.

**Status:** root-caused and fixed on `fix/mcp-tool-state-desync` (worktree
`${WORKTREES_ROOT}/au-mcp-tool-desync`, commit `5f235f52`), **awaiting merge to
`main`**. This document is the formal RCA the operator asked for; it also
generalizes the finding into a named invariant and surveys the codebase for
other places the same class of bug can hide — two of which are still open (see
[Where else it hides](#where-else-it-hides) and
[Known gaps this RCA surfaces](#known-gaps-this-rca-surfaces-not-fixed-here)).

## Symptom

`load_tools(servers=["github-mcp"])` **succeeded**. `multiplexer_status`,
`list_catalog`, and `find_tools` all reported the `gith__*` tools `mounted:
true` with server `state: "up"`. But **every direct `gith__*` invocation
failed with `"No such tool available"`**, and pinning one through the
`ask`/`act` intent verbs was rejected with `"Pinned capability is not allowed
for this intent verb."` — a second, independent symptom traced to the same
investigation (see [Bug 4](#bug-4-a-mounted-fleet-tool-was-invisible-to-the-intent-verb-surface)).

Why it mattered: graph-os is the control plane every agent in this workspace
is told to query before doing anything else (`AGENTS.md`'s *"Delegate to the KG
+ graph-os"*). A control plane that reports a capability as usable when it is
not forces agents back onto raw CLIs — defeating the MCP-driven operating
model the whole platform is built around.

## Root cause: three independent truthfulness bugs, one shared mechanism

All three live in `agent_utilities/mcp/multiplexer.py`. None of them is a
logic error in the ordinary sense — each function was internally consistent
and passed its own unit tests. The defect is architectural: **each status
surface re-derived "is this tool usable" against a different piece of
bookkeeping than the one the dispatch gate actually enforces**, and the two
were allowed to drift apart.

### Bug 1 — `list_catalog`'s `"mounted"` measured the wrong layer

Two independent facts share one name:

- **Process-level**: `server in self.children` — is the child process spawned
  at all. Set once, by `mount_child`.
- **Session-level**: whether `SessionVisibilityMiddleware` will actually let
  *this calling session* dispatch the tool — gated by `_exposed` (a live
  forwarder was registered) **and** `_session_loaded` (this session ran
  `load_tools` for it).

Before the fix, `list_catalog` reported the **process-level** fact under the
key `"mounted"` in three places:

- the single-server drilldown, `agent_utilities/mcp/multiplexer.py:2465`
  (pre-fix) — `"mounted": server in self.children`
- the same drilldown's `"mounted"` per tool (no session-scoped equivalent
  existed at all)
- the all-servers summary per-entry field, `multiplexer.py:2505` (pre-fix) —
  `"mounted": name in self.children`
- the all-servers summary's top-level field, `multiplexer.py:2519` (pre-fix) —
  `"mounted": sorted(self.children.keys())`

`find_tools` (`discover_tools`) was already session-correct — it threads a
`loaded: set[str]` through and only `list_catalog` lied. So the *same fleet
state*, asked two different ways, gave two different answers: `find_tools`
correctly said "not yet callable, run `load_tools`"; `list_catalog` said
"mounted: true". A caller that reasonably reads `list_catalog` as the
authoritative browse surface — which is literally its stated purpose — was
told a tool was ready when a **second/subagent session that had never run its
own `load_tools`** would get `"No such tool available"` from the exact same
name.

### Bug 2 — `resolve_and_mount` silently dropped unresolvable requested tools

`resolve_and_mount` (`multiplexer.py:2524`, pre-fix) computed
`(mounted_servers, to_expose, failed)`. An explicitly requested tool name that
did not resolve — an unknown name, or one disabled by config/runtime policy
*after* its owning server mounted successfully — was left out of **both**
`to_expose` and `failed`. It vanished with **no signal at all**: not exposed,
not reported as failed, just absent. A caller diffing what it asked for
against `newly_exposed` had to notice a name was missing by omission — nothing
told it *why*, or even *that* something had gone wrong for that specific tool.

### Bug 3 — `_notify_tools_changed` swallowed the client notification failure

`_notify_tools_changed` (`multiplexer.py:2767`, pre-fix) sends
`notifications/tools/list_changed` so an MCP client refreshes its own tool
list after a dynamic mount. On failure it logged a `logger.warning` — visible
only to the **server operator** — and returned nothing. `load_session_tools`
(the `load_tools` core) reported `newly_exposed` as an unconditional success
whenever the mount itself succeeded, regardless of whether the client was
ever actually told. A client whose notification silently failed to arrive
kept its stale tool list and would hit `"no such tool"` on the very name
`load_tools` had just reported as newly available — a THIRD way the same
symptom could occur, indistinguishable from Bug 1 at the calling agent's
vantage point.

### Bug 4 — a mounted fleet tool was invisible to the intent-verb surface

A second, independently-discovered symptom from the same investigation:
pinning a dynamically-mounted fleet tool through `ask`/`act`/`write`/`manage`
(`hints_json={"tool": ...}`) failed with `"Pinned capability is not allowed
for this intent verb."` — phrased as a verb-specific policy denial. Root
cause: `intent_tools._build_candidates()` builds its candidate table
exclusively from graph-os's own CPD-backed `REGISTERED_TOOLS`; a dynamically
mounted fleet tool was **never a candidate at all**, independent of which verb
was requested. The error message named the wrong cause. Fixed alongside Bugs
1–3: `dispatch_intent` now distinguishes "known capability, wrong verb" from
"not part of the intent-verb surface at all" and names the tool + points at
calling it directly after `load_tools`.

## The fix

`MCPMultiplexer.tool_dispatchable(prefixed_name, session_key=None) -> bool` is
now the **single source of truth** for "can THIS session dispatch this tool
right now" — the exact predicate `SessionVisibilityMiddleware` enforces at the
`tools/call` gate. Every status-reporting surface now derives its claim from
that one function instead of re-deriving the same logic against a second,
parallel structure:

- `list_catalog`'s per-tool field is renamed to reflect what it actually
  means: the process-level fact is now `process_running` /
  `servers_running`, and `"mounted"` (drilldown) / `"dispatchable_tools"`
  (summary) are computed by calling `tool_dispatchable()` per tool — so they
  **cannot** claim a tool is usable when a call would actually be rejected,
  because they run the identical check the rejection path runs.
- `SessionVisibilityMiddleware._visible()` and `on_call_tool()` now delegate to
  `tool_dispatchable()` too, deleting the middleware's own duplicated
  `_gated`/`_visible` logic — there is structurally one computation for "is
  this callable", not two that can drift.
- `resolve_and_mount` now always reports an unresolved requested tool in
  `failed` with a specific reason (`"tool is not present in the fleet
  catalog"` or `"tool is not registered by its owning server (disabled by
  config or rejected by its runtime policy)"`) — never silently absent from
  both `to_expose` and `failed`.
- `_notify_tools_changed` now returns whether the push actually reached the
  client (`False` for "no active request context" — an expected, silent
  no-op — vs. `False` for a live client that rejected/dropped it, both
  distinguished from `True`). `load_tools`/`unload_tools` surface this as a
  `"notified"` field, so a caller can tell its OWN client's tool list may be
  stale instead of discovering it later via a dispatch failure.

Regression tests prove: `tool_dispatchable` is session-scoped and false for a
catalogued-but-never-mounted tool; `list_catalog`'s per-tool `"mounted"` /
`"dispatchable_tools"` match the real dispatch outcome across two live FastMCP
`Client` sessions (reproducing the reported parent-vs-subagent asymmetry at
the state layer); `resolve_and_mount` reports an unknown/disabled requested
tool in `failed`, never silently; `_notify_tools_changed`'s return value and
`load_tools`'s `"notified"` field are honest about context-absence vs. send
failure; the two intent-verb-pinning messages are distinguished by tool
origin. 183 tests passed on the fix branch.

## The generalized invariant

This is not the first time this exact shape of bug has appeared. Naming it:

> **The favorable-restatement anti-pattern.** A status field is *derived* by a
> second layer instead of *read through* from the one component that actually
> performed (or can verify) the operation. The restatement drifts toward
> "looks more done than it is" because the second layer encodes what the
> operation was *supposed* to mean, not what actually happened — and nothing
> forces the two to be re-checked against each other when either side changes.

**The invariant this fix restores, stated as a rule:** *Reported state must be
derived from the authoritative source at the moment of reporting — never
restated, cached, or inferred by a second layer that only witnessed the
operation's outcome secondhand.* Concretely: if component A is the one that
actually gates/performs/persists an operation, every surface that reports on
that operation must call into A's own predicate (or A's own confirmed result)
— never re-implement an equivalent-looking check against a different signal
(a sibling flag, a process-level fact, an input parameter, a "should have
succeeded" assumption).

Two structural tells that a piece of code is *about* to violate this
invariant, both present in every instance found so far:

1. **Two names for what reads like one concept** (`mounted` process-level vs.
   `mounted` session-level; `active` requested vs. `active` confirmed;
   `status` claim vs. `status` fact) — when a second field is added that
   *could* answer the same question a first field already answers, the two
   are now free to disagree, and nothing enforces that they don't.
2. **A success value assigned before, or independent of, the fallible step it
   describes** — set from an input parameter or an early branch, then never
   revisited even when a later step (that the field is supposed to describe)
   fails.

## Other confirmed instances of the same class

This is at minimum the **fifth** time this shape of bug has been found in this
codebase — cited here so this RCA generalizes rather than narrates one fix:

1. **N1 — the provenance layer recorded `succeeded` for a failed
   operation.** A failed `graph_query` returned `claims[0].status="failed"`
   while the *same response*'s `routing.decision_trace.result_provenance.status`
   read `"succeeded"`. Root cause: `intent_tools.py`'s
   `_execution_succeeded()` (`agent_utilities/mcp/tools/intent_tools.py:1006`)
   only recognized `dict`/`str` results; `graph_query`/`graph_ask`/`nl_query`
   return a typed `EvidenceBundle`, which matched neither branch and fell
   through to `return result is not None` — true for *any* non-empty bundle
   regardless of its content. **Fixed** (`51182953`): dump the model first so
   `EvidenceBundle.error` is honored as the one source of truth. The
   poisoned path was `_record_dispatch_outcome(..., success=True)`, which fed
   the capability router's learned reward for that tool/task-class — a
   restated status corrupting a *learning signal*, not just a display field.
2. **The catalog's specific-sounding "budget exceeded" attribution may not be
   true of the specific server it names.** `probe_fleet_catalog`
   (`agent_utilities/mcp/multiplexer.py:2186`) bounds a fleet-wide probe pass
   with `asyncio.wait(tasks, timeout=budget)`; every task still `pending`
   when the aggregate wait returns is stamped
   `f"catalog discovery budget exceeded after {budget:g}s"`. The *aggregate*
   claim is true — the overall pass did exceed budget — but probes are
   admitted through a bounded `asyncio.Semaphore(16)`, so a "pending" task
   may never have started running at all; it could have been queued behind
   15 others the entire time. Every still-pending server is given the
   identical, specific-sounding reason regardless of which is true. This is a
   close cousin of the main invariant (a plausible-sounding cause substituted
   for an unverified one, rather than a favorable status substituted for an
   unfavorable one) — flagged here as **still open**, not fixed by this
   branch; see [Known gaps](#known-gaps-this-rca-surfaces-not-fixed-here).
3. **`OntologyLifecycle.set_active()`/`load()` can report `active=True`
   independent of whether the engine actually loaded the axioms.**
   `set_active()` (`agent_utilities/knowledge_graph/ontology/lifecycle.py:579-582`)
   writes `record["active"] = bool(active)` **before** calling
   `self._load_axioms(...)`, and never revisits it against the result.
   `load()` (`lifecycle.py:326`) writes `"active": bool(activate)` — the
   caller's *requested* intent — while `_load_axioms()`
   (`lifecycle.py:249-258`) swallows an `add_triples` failure into
   `{"loaded_to_engine": False, "reason": str(exc)}` and returns normally
   (never raises). So a caller can read `active: true` on a record whose
   nested `engine.loaded_to_engine` is `false` in the very same response.
   D-OB-14 records this as fixed ("Now `active` reflects what the engine
   actually confirmed") — **verified against current `main` while writing
   this RCA, that fix is not present here**; it most likely lives on a
   still-unmerged sibling branch, the same "fixed but not yet landed" state
   this RCA's own fix is in. Recorded as a still-open, independently-verified
   instance in [Known gaps](#known-gaps-this-rca-surfaces-not-fixed-here)
   rather than silently assumed closed.

## Where else it hides

A grep-based checklist for auditing the rest of the codebase against the
invariant — anywhere one of these shapes appears, check whether the two sides
can independently drift:

- **Any field that mirrors a boolean input parameter into a persisted/returned
  "outcome" field** (`"active": bool(activate)`, `"success": requested`,
  `"enabled": flag`) instead of the actual result of the fallible operation
  the field claims to describe.
- **Any place a process-/resource-level fact (`X in self.children`, "is the
  connection open", "is the worker alive") is surfaced under a name that
  reads like a capability/dispatch-level claim** ("mounted", "ready",
  "available") — the two are almost always independently gated (a process can
  be up with nothing dispatchable through it, e.g. mid-restart, or gated by a
  session/tenant/ACL layer the process-level check never consults).
  `_server_level_fallback()` (`multiplexer.py:2202-2219`, used by
  `discover_tools`'s no-match branch at `multiplexer.py:2441`) still returns
  `"mounted": server in self.children` — the exact Bug 1 shape, **not touched
  by this fix** because it is a separate code path from `list_catalog`. See
  [Known gaps](#known-gaps-this-rca-surfaces-not-fixed-here).
- **Any `except Exception` that returns a "handled" dict instead of
  re-raising or setting a caller-visible failure flag**, when the surrounding
  success/status field was already decided before entering the `try` (Bug 3's
  shape, and `_load_axioms`'s shape above).
- **Any two independent code paths that compute what should be the same
  fact** (a middleware's own gating check vs. a status tool's separate
  recomputation of "is this visible") — the fix's core lesson is to collapse
  these into one function, not keep them in sync by convention.
- **Any error-message template applied uniformly across a batch** ("budget
  exceeded", "timeout", "not found") to the whole batch after only the
  aggregate condition was checked, when per-item causes could differ (this
  RCA's item 2).
- **Provenance/telemetry/reward-feeding status fields specifically** — per
  N1, a favorable restatement here doesn't just mislead a display, it can
  poison a *learned* signal (`calibrated_outcome_reward`, capability routing).
  Any new evidence/outcome-shaped field added to the AHE reward pipeline
  should be checked against this invariant before it ships.

## Known gaps this RCA surfaces (not fixed here)

Recorded honestly rather than left implicit, per this RCA's own invariant:

- **D-OBC-1** — `_server_level_fallback()` (`multiplexer.py:2202-2219`) still
  reports process-level `"mounted"` for its `find_tools`-fallback results —
  the exact Bug 1 shape, outside the branch this RCA documents.
- **D-OBC-2** — `OntologyLifecycle.set_active()`/`load()`
  (`lifecycle.py:326`, `579-582`) still write `active` from the requested
  flag, not the engine's confirmed result, on current `main` — despite
  D-OB-14 recording this as fixed elsewhere.
- **D-OBC-3** — the catalog's per-server "budget exceeded" attribution
  (`multiplexer.py:2186`) does not distinguish "this server's probe actually
  timed out" from "this server's probe was never dequeued from the semaphore
  in time."

These are filed in `reports/deferred/lane-ob-closures.md` for a
follow-up lane — deliberately not fixed inline here to keep this change
surgical to the RCA the operator asked for.

## Verification

- Fix branch: `fix/mcp-tool-state-desync` @ `5f235f52`, worktree
  `${WORKTREES_ROOT}/au-mcp-tool-desync`. 183 tests passed. Awaiting merge.
- 11 pre-existing, unrelated `ChildRuntime`/circuit-breaker test failures are
  recorded separately as `D-DESYNC-1` (`reports/deferred/lane-mcp-desync.md`)
  — reproduced identically on the unmodified base commit, not introduced by
  this fix.
- This RCA's own citations against current `main` (`64a41727`) were verified
  by direct file read while writing this document, not copied from the fix
  branch's commit message — which is how the two still-open gaps above were
  found.
