# agent-utilities — Ubiquitous Language (CONTEXT)

> Domain glossary (ubiquitous language) for agent-utilities, complementing the machine-generated
> concept registry (`docs/concepts.yaml`). Records domain language only — single source of truth per
> term, with `Avoid:` guidance to prevent terminology drift across code, PRs, and docs.
> CONCEPT:AU-ECO.toolkit.self-documenting-plugin-bundle — self-documenting. Assimilated from open-design's `CONTEXT.md` discipline.

## Language

**Adapter** — A declarative description (`AdapterDefinition`, CONCEPT:AU-ORCH.adapter.multi-cli-adapter-dispatch) of an external
agent-CLI or local-LLM runtime backend that the execution engine can spawn.
_Avoid_: driver, plugin, connector.

**Provider Proxy** — The BYOK normalizing layer (CONCEPT:AU-ORCH.adapter.byok-provider-proxy) that turns any LLM provider's
stream into the canonical event union behind a DNS-resolved SSRF guard.
_Avoid_: gateway, relay, passthrough.

**Canonical Event** — A provider/CLI-agnostic execution event (`ExecEvent`: start/text_delta/tool_use/
error/end) every adapter and proxy normalizes to.
_Avoid_: chunk, token, message.

**Held Turn** — A run paused mid-turn awaiting a tool result (CONCEPT:AU-ORCH.execution.held-turn-registry-mid), resumed via
`/api/runs/{id}/tool-result`.
_Avoid_: blocked run, suspended task, pending approval.

**Run-Scoped Token** — A short-lived, HMAC-signed capability token (CONCEPT:AU-OS.observability.run-wide-correlation-id) bound to a run,
project, endpoint allowlist, and expiry; minted by the CLI `run` command.
_Avoid_: API key, session token, bearer (unqualified).

**Pre-Emit Gate** — The layered quality pipeline (CONCEPT:AU-AHE.harness.pre-emit-quality-gate): preflight checklist →
multi-dimensional critique, run before output is emitted.
_Avoid_: validator, linter, post-check.

**Live Artifact** — A refreshable output (CONCEPT:AU-KG.memory.live-refreshable-artifact-models) = template + bounded data + provenance that
re-derives its data from the KG on refresh, preserving the prior render on failure.
_Avoid_: report, document, static artifact.

**Process Stamp** — The 5-field identity (app/mode/namespace/ipc/source) of a sidecar process that
resolves to an isolated UDS path per run.
_Avoid_: pid, handle, slot.

**Scenario** — The semantic grouping of a skill (design/marketing/operation/engineering/finance/hr/
sales/personal) used by the eval-scored skill picker.
_Avoid_: category (category is the secondary axis), tag, mode.

## Relationships
- An **Adapter** produces **Canonical Events** when run by the execution engine.
- The **Provider Proxy** also produces **Canonical Events** from upstream LLM streams.
- A **Held Turn** is resumed by injecting a tool result; a **Run-Scoped Token** authorizes that call.
- A **Live Artifact** records, in its provenance, the **Adapter**/model that produced its data.
- A **Pre-Emit Gate** decision can block emission of an output in `block` mode.
- A **Scenario** filters candidates before the skill picker scores them.
