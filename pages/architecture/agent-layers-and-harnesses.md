# Agent layers and harnesses

`agent_utilities.layers` implements the Agent Utilities side of RF-ADR-010:
one harness port for every agent runtime, one sandbox port for local
isolation, fail-closed capability negotiation, a conformance kit, the single
L5 outcome writer and typed clients for layers L0-L5.

<div class="admonition architecture" markdown>
<p class="admonition-title">Harness execution path</p>

`HarnessAgentExecutor` hands a `RunSpec` to `negotiate()`, which returns a
`NegotiatedRunSpec` for one `HarnessPort`: pydantic-ai (the in-process AU
runtime), the Claude Code CLI, the Codex CLI, the Grok Build CLI, or the Devin
API v3. The port's `RunResult` and trace go to `RunOutcomeWriter`, which
records them in the Epistemic Graph through `commit_result(outcome_extension)`.
</div>

## RunSpec and negotiation

A `RunSpec` binds the task, agent reference, required and optional
capabilities, the run toolset (EG context endpoint, MCP servers, tool
allowlist, required tools, digest-pinned skills), the minimum trace fidelity,
the allowed execution environments, the budget, the account mode and a
credential reference. Its SHA-256 digest is stamped on every run event.

`negotiate(spec, descriptor, policy)` returns the one `NegotiatedRunSpec` that
may execute, or raises `HarnessRefused` naming every unmet requirement. Each
rule is one table row. A run is refused when a required capability, fidelity,
environment, account mode, vendor term, strict budget unit, MCP client, tool
proof, allowlist, skill proof or runtime option is not available. By default
the policy also requires the EG context MCP endpoint.

## Adapters

| Harness | Invocation | Fidelity | Usage | Environment | Tool/skill proof |
|---|---|---|---|---|---|
| `pydantic-ai` | `Orchestrator.execute_agent` with a progress sink | tool-calls | unavailable | caller-managed-host | runtime contract |
| `claude-code` | `claude -p --output-format stream-json`, strict per-run MCP config, project fence settings, `dontAsk` | tool-calls | measured (tokens, USD) | caller-managed-host | `system/init` inventory |
| `codex` | `codex exec --json --ephemeral`, MCP by `-c` override | tool-calls | measured (tokens) | caller-managed-host | none |
| `grok` | `grok --prompt-file ... --output-format streaming-json` | tool-calls | unavailable | caller-managed-host | none |
| `devin` | Devin API v3 sessions | final-output | unavailable (ACU recorded) | provider-managed-remote | none |

Every adapter supports `api_key` and `subscription` accounts. Credentials are
references resolved at launch through `SecretsClient`; MCP bearer tokens reach
a CLI only as environment variables named in the generated config. The child
environment is an allowlist. A missing binary raises `HarnessNotInstalled`, a
missing credential raises `HarnessNotConfigured`.

A failure after a tool call may have had effects. Unless the RunSpec declares
`side_effects="none"`, such a failure, a timeout or a cancellation ends the run
as `outcome_uncertain`, and nothing retries it automatically.

## Sandbox port

`RouterSandboxPort` leases the cheapest RLM sandbox backend that meets the
job's requirements and the pin/deny policy. It skips unavailable and
non-isolating backends and records every exclusion on the lease. CLI harnesses
run in a private per-run workspace (`host_workspace_lease`). That workspace is
not a containment boundary.

## Conformance kit

`agent_utilities.layers.conformance` checks descriptor honesty, session
lifecycle, streaming events, tool-call surfacing, cancellation, sandbox
boundary and error typing. Each target labels its evidence as
`real-harness`, `recorded-live-transcript` or `synthetic-transcript`.
`TranscriptLauncher` replays captured JSONL through an adapter's real parsing
path.

## L5 writer and layer clients

`RunOutcomeWriter.commit` commits a run's terminal outcome, its RunTrace,
ToolCall and OutcomeEvaluation receipts and one RunEvent atomically, through
EG's `work_items.commit_result(outcome_extension=...)`, under the worker's
lease fence. An uncertain run is committed as a non-retryable, degraded failure.
A transport failure is reconciled by reading the outcome back.

`LayerClients.for_session(eg_client, session)` provides one typed client per
layer: L0 provenance, L1 components, L2 agents, L3 `AgentAssemble` and
`DecisionCommit`, and L5 committed outcomes. L4 is the `HarnessRegistry`.
