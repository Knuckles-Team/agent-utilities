# Design Document: Ingest agent chat/session history from the CLIENT side (parse local logs) rather than only from server-side traces

CONCEPT:AU-ECO.mcp.client-side-chat-session

> `agent_utilities/cli/__init__.py:155-176` (`agent-utilities ingest-sessions`) and
> `agent_utilities/ingestion/collector.py:173`. REST twin:
> `agent_utilities/gateway/usage_api.py:7,327`. Both dispatch through the same
> `graph-os` `ingest_sessions` upload action so the CLI and REST paths never drift.

## Decision — a CLI command parses THIS host's local agent chat logs and either sinks them into a local engine or uploads them to a remote one, instead of relying only on the server-side `RunTrace`/`ToolCall` provenance already recorded for delegated work

`agent-utilities` already records rich provenance for everything it orchestrates
(`RunTrace`/`ToolCall` nodes, KG-2.296). But a large share of real agent activity
happens in **other clients this platform did not launch** — Claude Code, Antigravity,
and other IDE/CLI agents running directly on an operator's machine, with their own
local session-log formats and no relationship to `graph-os`'s own orchestration path.
That activity is invisible to the KG unless something parses those logs.

`agent-utilities ingest-sessions` (`cli/__init__.py:155`) is that something: it
auto-detects every supported agent's local log format on the current host, parses
changed files only (`--all` re-parses everything), and ingests the parsed
session/turn/tool-call records into the KG's usage-cost/observability store
(`gateway/usage_api.py`) via `ingestion/collector.py`. Two paths, one core:

* **`collect` (default)** — sinks directly into a local engine. The natural mode
  when the engine already runs on the same host as the CLI client being harvested.
* **`--upload`** — pushes the parsed rows to a REMOTE engine via the `graph-os`
  MCP `ingest_sessions` upload action (`--server`/`--url` select the target). This
  is the mode a laptop running Claude Code uses when the actual KG engine lives on
  a different host in the homelab.

The parsing happens **client-side, on the host that owns the logs** — not by
having the engine reach out and scrape a client's local filesystem itself.

## Rejected alternative — a server-side/engine-side puller that scans known client log-directory conventions on hosts it can reach

The engine already has a privileged, authenticated position to pull data (this is
exactly how connector ingestion — GitLab, Jira, ServiceNow — works elsewhere in the
platform via `source_sync`). The same shape was available here: teach the engine
`~/.claude/projects/**` and the equivalent Antigravity paths and have it scan them
directly, the same way it scans a repository. That was rejected because client chat
logs are **local, host-scoped, and not a networked source with an API** the way a
connector's upstream system is — the engine may not even run on the same host as the
IDE client (the `--upload` mode is the common case in this homelab, per the CLI
worktree's own `AGENTS.md`/CLAUDE.md network topology). Pulling would require either
running the engine's ingestion logic on every client host (duplicating the engine) or
giving the engine remote filesystem access to arbitrary developer machines — a much
larger trust boundary than a CLI command the operator runs locally and explicitly
pushes from. Parsing client-side and pushing (or sinking locally) keeps the log
format's knowledge colocated with the logs themselves and needs no new engine-side
privilege.

## Risk Assessment

- **Blast Radius**: `agent_utilities/cli/__init__.py`, `agent_utilities/ingestion/collector.py`,
  `agent_utilities/gateway/usage_api.py`, `agent_utilities/mcp/tools/write_ingest_tools.py`.
- **Backward Compatible**: Yes — a new CLI subcommand; no existing ingestion path changes.
- **Known weak point**: log-format detection is per-agent and hand-maintained; a new
  agent client (or a breaking log-format change in an existing one) needs an explicit
  parser addition here, there is no generic/self-describing fallback.
