# Lane w7-mcp-catalog-unify — D-DEL-1 (2026-08-07)

Status: **code fix implemented and reviewed in-worktree; NOT YET committed,
tested, or landed** — blocked mid-lane by host disk exhaustion (see
"Blocker" below). This report documents the root cause (confirmed live,
beyond what D-DEL-1 already stated), the chosen fix, and exactly what is
left to finish once the blocker clears.

## The chosen writer contract, and why

**Decision: `source_sync._sync_fleet`'s `:MCPServer`/`SERVES`/`Tool`|`Skill`
schema is the canonical fleet capability catalog. The reader
(`agent_runner.py`) is made schema-tolerant — it now resolves a server
through EITHER schema, through one shared helper, rather than only the
`:Server`/`PROVIDES`/`CallableResource` shape it exclusively read before.**

Considered and rejected:
- **Rewrite `source_sync` to emit `:Server`/`PROVIDES`/`CallableResource`.**
  Rejected: that schema is what a handful of consumers read
  (`dynamic_tool_orchestrator.py`, `engine_mcp_discovery.py`,
  `workflow_compiler.py`), but `:MCPServer`/`Tool`/`Skill`/`SERVES` is what
  ~10 other modules already key off (`semantic.py` enrichment,
  `connector_manifest_gate.py`, `graph_compute.py`, `resource_priority.py`,
  the ranking/tag/synonym pipeline). Rewriting the actively-populated,
  richly-tagged fleet writer to match the thinner, largely-vestigial reader
  schema would be the higher-blast-radius change for no benefit — it is
  the schema doing the real work today.
- **Wire `engine_ingestion.ingest_mcp_server` into the periodic fleet sync
  too**, so both schemas get written. Rejected: that is TWO writers running
  forever for the same data — exactly the "ONE writer contract" mandate
  forbids. It would also double ingestion cost and require the exact
  ACL-classification fix (`_classify_mcp_node`) that `source_sync` writes
  never call today (see "Related finding" below) to even be readable.
- **Chosen: teach the reader to resolve identity through whichever schema
  actually has the server**, treating `:MCPServer`/`SERVES` as the primary,
  actually-populated source and `:Server`/`PROVIDES` as a secondary
  identity-only source (still written by the self-registration heartbeat in
  `mcp/server_factory.py` and the on-demand refresh in
  `tools/dynamic_tool_orchestrator.py`), with a final fallback to the live
  multiplexer configuration name (no KG tool data, but the real transport
  still binds — see below). This is the minimal-blast-radius fix that
  converges on one CONTRACT for capability binding without deleting a
  schema other code depends on.

## Additional root-cause layers found live (D-DEL-1 named the schema mismatch;
these compound it and were found investigating the fix, not re-deriving it)

1. **Duplicate `:MCPServer` nodes per server, one with a plain id
   (`mcp_server_ansible-tower-mcp`) and one with a persistence-privacy
   pseudonymized id (`mcp_server:pref_mcp_server_<hash>`)** — confirmed live
   for `ansible-tower-mcp`. A lookup keyed by a synthesized id
   (`f"mcp_server_{name}"`, what the old reader effectively assumed) finds
   the wrong/empty duplicate. Fix: **never construct a server id** — resolve
   by `{name: $name}` and let Cypher walk every node matching the pattern,
   which finds real edges regardless of which physical duplicate holds them.
2. **`WHERE prop = $a OR prop = $b` silently returns zero rows on this KG's
   native engine even when a row matches**, verified by direct comparison
   against the equivalent single-predicate and inline-map-pattern queries
   against the live pod (`kubectl -n platform exec ... python3 -c ...`
   against a real `:Server` node with a known, existing name — the OR form
   returned `[]`, the single-predicate and inline-map forms both returned the
   real row). This is exactly the form both cited reader sites used
   (`agent_runner.py:2038` and the old `:2126`). Fixed by never combining a
   name/id disjunction into one `WHERE ... OR ...` — two single-predicate
   queries (or one inline-map pattern) instead.
3. **The candidate-name derivation in `_bind_skill_to_owning_server` could
   never have matched ANY real fleet server**, independent of the schema
   mismatch. `fleet_skill_harvest.py` persists a runnable skill's
   `provider_ref` as `provider=f"mcp:{server_name}"`, then
   `skill_workflow_ingest._slug()` lower-cases and collapses every run of
   non-alnum characters — including both `:` and `-` — to `_`. So
   `servicenow-mcp` is persisted as `provider://mcp_servicenow_mcp` (verified
   live: the real `servicenow-incident-management` CallableResource's
   `provider_ref` is exactly `provider://mcp_servicenow_mcp`). The old code
   tried `<provider>` and `<provider>-mcp` — neither is `servicenow-mcp`.
   Slugging is lossy/one-directional, so it cannot be reverse-transformed;
   fixed by re-deriving the SAME slug the writer computed for every server
   name the KG or the live fleet config knows, and matching on equality
   (`_fleet_server_candidates`).

Any one of #2 or #3 alone would have kept auto-resolve delegation broken
even after fixing the schema mismatch in isolation — worth flagging because
a narrower fix (just adding a second `MATCH` clause for `:MCPServer`, which
the lane brief explicitly warned against doing reflexively) would still have
silently failed.

## The fix (file: `agent_utilities/orchestration/agent_runner.py`)

- **New `_lookup_server_identity(engine, name)`** — the one place that
  resolves a server's identity + tool catalog across both schemas, always
  name-anchored, never `WHERE ... OR ...`. Returns `None` only when the name
  is unknown to both schemas; a server that exists with zero populated tools
  (e.g. a fleet child the probe couldn't reach — confirmed this is exactly
  `servicenow-mcp`'s live state right now, see "Live findings" below) still
  returns a real, empty-tools identity, since `_toolset_for_id` never reads
  tool data from the KG for transport anyway (`_build_execution_config`'s own
  documented contract: "KG server nodes carry capability identity and opaque
  provenance, never executable transport").
- **New `_known_kg_server_names` / `_fleet_server_candidates`** — invert a
  skill's slugged `provider_ref` back to a real server name by recomputing
  the writer's own `_slug()` over every name the KG or the live fleet config
  knows, instead of guessing a reverse suffix transform.
- **`_bind_skill_to_owning_server`** — rewritten to use the two helpers
  above, in order: legacy `:Server` identity → fleet `:MCPServer` catalog →
  live fleet-config name pin (no tool data). ★ **Fail loud**: when a skill
  declares a genuine external provider and none of the three resolves it,
  sets `meta["binding_error"]` instead of silently returning with the skill
  left prompt-only.
- **`_resolve_agent_from_kg` Search 1** — now calls the same
  `_lookup_server_identity`, so a direct `agent_name == <server name>`
  resolution (and therefore `_catalog_toolset_binding`, the explicit
  `tool_server=` path) benefits from the same schema-tolerance and gets real
  tool data instead of only the empty-tools live-config pin.
- **`run_agent`** — right after skill resolution, on the auto-resolve path
  (`not tool_server`), raises `LookupError(agent_meta["binding_error"])`
  when set, instead of continuing into a prompt-only run that still reports
  success. The explicit `tool_server=` branch is untouched — it already
  fails loud on its own terms via `_catalog_toolset_binding`'s
  `LookupError` on an unresolved server, and never carries a stale
  `binding_error` (it rebuilds `agent_meta` from a fresh `server_meta`).

## Live findings (proof of "before" state — via `kubectl -n platform exec`
into `graph-os-7587f9d77d-5kfv6`, `graph-os-host` container, matching how
D-EXIT-1 was itself verified; the graph-os MCP tools were unusable this
session — client-side token expired mid-lane, `multiplexer_status` /
`graph_query` both returned "requires re-authorization")

- `servicenow-mcp` IS a real `:MCPServer` node (`MATCH (s:MCPServer)` full
  scan finds it, name-matched), but has **zero `SERVES` edges** — the probe
  never successfully harvested its tools. `:Server`/`PROVIDES` has nothing
  for it either (0 of 14 live `:Server` nodes are named `servicenow-mcp`;
  all 14 are `external-<hash>` identities from the on-demand refresh path,
  not the self-registration heartbeat).
- `_resolve_agent_from_kg(engine, "servicenow-incident-management")` on the
  **pre-fix** code returns `type='skill', server_id='', toolset_id='',
  tools=0` — confirmed live, i.e. the exact silent prompt-only degrade
  D-DEL-1 describes.
- `scripts/delegation_probe.py --skill servicenow-incident-management`
  (no `--server`) currently fails at **stage 6 (skill)**, not stage 7
  (toolset) as in D-EXIT-1's run — `LookupError: skill
  'servicenow-incident-management' does not resolve in the graph`. Root
  cause of THAT is unrelated to D-DEL-1: the probe's own fresh
  `IntelligenceGraphEngine.get_or_create(defer_background_start=True)`
  reproducibly attaches to a near-empty view (`count(n)`≈0-1) in a fresh
  script process, vs. 45,715 real nodes reachable the same way without
  `defer_background_start=True`. Flagging as a separate, not-yet-filed
  environment/engine-bootstrap issue — did not chase further; it blocks
  using the probe script as-is for the "after" proof and should be re-run
  once fixed (or the probe adjusted not to defer background start).
  Direct-function reproduction (`_resolve_agent_from_kg` above, from a
  non-deferred engine) is the evidence of record for this lane instead.

## Related finding, not fixed here (flagging, in scope of "watch for a third
writer")

- `engine_ingestion.py`'s `_classify_mcp_node` stamps a PUBLIC ACL on every
  `CallableResource`/`ToolMetadata` node it writes, specifically because,
  per its own docstring, an unclassified node is invisible to
  `query_cypher`'s governed read path ("no ACL defined — default deny").
  `source_sync._write_fleet_nodes` (the writer actually populating
  `:MCPServer`/`Tool`/`Skill`/`SERVES`) never calls the equivalent
  classification. `engine.backend.execute` (used throughout this fix, and by
  the pre-existing reader) is a lower-level path than `query_cypher` and was
  not observed to be ACL-filtered in live testing, so this did not block
  today's fix — but if any caller reads the fleet catalog through
  `query_cypher` instead, the ACL gap would reproduce the exact "capability
  invisible" failure shape one layer up. Left unfixed (out of this lane's
  scope); worth a follow-up register item if not already tracked.

## Register

Opening `D-DEL-1` update / re-verification once Bash is available again
(`scripts/deferred_registry.py` requires a shell). This report is the
detail that update will point to.

## Coordination — lane `w7-webui-surfaces` (MCP Servers count)

Not yet contacted (SendMessage also depends on the harness being healthy;
retrying). What to tell them: the writer contract is now "the reader
accepts `:MCPServer`/`SERVES` (source_sync's fleet catalog) as well as the
legacy `:Server`/`PROVIDES` shape" — if their "MCP Servers 0" count queries
`:Server` only, it should be pointed at `:MCPServer` (90 live nodes as of
this session) instead/also, name-matched, never via a constructed id (see
"duplicate node ids" finding above). I did not locate the webui's count
query in this repo (it's presumably in the separate `agent-webui` repo) —
did not go looking further, per the brief's "coordinate, don't duplicate."

## Messaging / AgentConfig question, for the operator

**Config: unified. Deployment: deliberately split.**

- `agent_utilities/messaging/daemon.py` (`main()`, `_validate_fleet_auth()`,
  `mint_process_identity()`) loads config through the exact same
  `agent_utilities.core.config.config` / `load_config()` module graph-os
  uses — same XDG `config.json`, same `AgentConfig` schema, no parallel
  config path. Every process "consumes the same XDG AgentConfig
  declaration" (the module's own comment).
- The codebase already has a **combined-process mode**:
  `agent_utilities/mcp/co_service_supervisor.py` starts graph-os PLUS the
  messaging inbound router in the same process, gated on whether
  `messaging.daemon.configured_platforms()` finds a configured platform.
- **The live k8s deployment does not use that mode.** The `graph-os-host`
  container (`graph-os-7587f9d77d-5kfv6`, namespace `platform`) runs
  `python3 -m agent_utilities.gateway.daemon`, whose own code comment states
  the inbound messaging router deliberately runs in ITS OWN process so the
  host's CPU-bound maintenance (ingestion/relevance sweeps) can never starve
  the inbound reply loop. Messaging is meant to run as the separate
  `agent-utilities-messaging` systemd unit.
- Checked live on this host: `systemctl status agent-utilities-messaging` →
  **`loaded ... disabled; ... Active: inactive (dead)`**. So on this host,
  messaging is not currently running at all via either path — worth the
  operator's attention as a separate, likely-unrelated finding (not touched
  here per "do not re-architect the deployment in this lane").
- **Answer to "bundled with graph-os and configured via same agent config?"**
  — same config: yes. Same process/bundle: no, by design, and the design
  choice (isolation from host CPU contention) is documented in code. A
  combined-process option already exists in the codebase if the operator
  wants it; adopting it is a deployment decision, not something this lane
  changed.

## Blocker — host disk exhaustion mid-lane

Partway through live verification, the harness's own `/tmp/claude-1000/...`
partition hit `ENOSPC` and every `Bash` tool call started failing before
even running the command (`open '.../tasks/<id>.output': ENOSPC`). This
took out: further `kubectl exec` verification, `git diff`/`git add`/`git
commit`, `scripts/deferred_registry.py`, `scripts/safe_precommit_all_files.py`,
and `merge-queue enqueue`. `Read`/`Edit`/`Write` against the actual worktree
kept working throughout (different filesystem), which is how the code fix
above was completed and this report was written.

**Left to do once the harness/disk recovers** (in order):
1. `git -C /home/apps/worktrees/agent-utilities/w7-mcp-catalog-unify status`
   / `diff` — confirm only `agent_utilities/orchestration/agent_runner.py`
   changed, then commit.
2. Re-run the live "after" proof: re-exec
   `_resolve_agent_from_kg(engine, "servicenow-incident-management")` (the
   same direct-function probe used for "before" above) and confirm
   `type=='server'`, `toolset_id` set, and a real `run_agent`/
   `Orchestrator.execute_capability(skill_name="servicenow-incident-management",
   task=...)` call **without** `tool_server` actually calls
   `servicenow_get_incidents` and returns real PDI data (D-EXIT-1's own
   incidents, INC0000055/INC0000053, are the reference values already
   proven reachable via the explicit-`tool_server` path).
3. Loud-failure proof: call the same path with a skill whose provider names
   a server that exists in neither KG schema nor the live fleet config, and
   confirm a `LookupError` surfaces instead of a prompt-only 200.
4. `python3 scripts/safe_precommit_all_files.py`, then
   `scripts/deferred_registry.py open --lane w7-mcp-catalog-unify --id
   D-DEL-1 ...` (or a re-verification `close`, if the wave's other lanes
   agree the schema-tolerant reader is the accepted contract) with
   `--evidence-file` pointing at this report.
5. `SendMessage` to `w7-webui-surfaces` with the "MCP Servers count" note
   above.
6. Land via `agent-utilities merge-queue enqueue` from the worktree — never
   a hand merge.
