# Design Document: `vault_sync` reconciles a service's secrets as read-existing-then-seed-missing-only, so genesis/deployment never re-prompts for (or overwrites) a secret that already exists

CONCEPT:AU-OS.deployment.vault-first-routine-genesis (covers the cluster:
`AU-OS.deployment.vault-seed-service` is the pointer for the MCP-tool call site
that invokes this same routine)

> `agent_utilities/security/secrets_client.py:776-800` (`SecretsClient.vault_sync`);
> REST twin `agent_utilities/mcp/kg_server.py:2034`
> (`graph_configure(action="vault_sync")`); caller
> `agent_utilities/mcp/tools/analysis_tools.py:2655`.

## Decision — `vault_sync(service, env_keys, values, overwrite=False)` is the ONE genesis/deployment routine for getting a service's secrets into the store: for each env-var key a service needs, it READS the existing value at `<service>/<KEY>` first (already-present keys are kept and reported, never re-prompted), then WRITES only keys that are missing (or every supplied key when `overwrite=True` is explicitly requested), and always emits a resolvable `vault://<service>/<KEY>` reference — backend-agnostic (Vault or the encrypted engine store, same `get`/`set` contract)

Provisioning a new service's secrets, or re-running genesis against an
already-provisioned deployment, needs to be safe to run repeatedly without
clobbering secrets an operator (or Claude) already set up. `vault_sync`'s decision
is "read-existing + seed" as ONE reconciliation call: "an operator (or Claude)
never re-supplies a secret that already exists" (`secrets_client.py:785-786`).
Because it always returns `vault://<service>/<KEY>` references for every key
(existing or newly seeded), the caller can drop the resolvable refs straight into
`config.json`, and those refs round-trip through `resolve_ref` regardless of which
backend (Vault vs. encrypted engine store) actually holds the value — the
"vault-first" framing means callers program against ONE reference scheme, not
against whichever concrete backend happens to be mounted.

## Rejected alternative — always prompt for / overwrite every secret on each genesis run, or require the caller to check existence itself before writing

Two shapes are rejected by "read-existing + seed" as the DEFAULT (`overwrite=False`)
behaviour. First: a genesis routine that always prompts for (or blindly writes) every
configured secret on every run — that was rejected because re-running genesis
against an already-provisioned deployment is a normal operational action (idempotent
redeploys, config drift reconciliation), and forcing a full re-supply of every
secret every time is exactly the "never re-prompted" cost this design avoids.
Second: pushing the read-then-write ordering onto every CALLER (each genesis step
independently checking existence before writing) — that was rejected in favour of
making it the ONE routine's own default contract, so every caller (the MCP tool at
`analysis_tools.py:2655`, the REST twin, and any future genesis step) gets the
non-destructive behaviour automatically rather than needing to remember to
implement the check-then-write pattern itself; `overwrite=True` remains available as
an explicit, opt-in escape hatch for the deliberate-rotation case.

## Risk Assessment

- **Blast Radius**: `agent_utilities/security/secrets_client.py`,
  `agent_utilities/mcp/kg_server.py` (`vault_sync` REST action),
  `agent_utilities/mcp/tools/analysis_tools.py`.
- **Backward Compatible**: Yes — `overwrite` defaults to `False`, so an existing
  call site that never asked for overwrite behaves identically before and after.
- **Known weak point**: "already-present keys are kept... never re-prompted" means
  a secret that was seeded with a WRONG value early in a deployment's life stays
  wrong on every subsequent `vault_sync` unless someone explicitly passes
  `overwrite=True` (or uses the dedicated rotation flow) — the routine has no way
  to distinguish "correct and stable" from "wrong and stale" for an existing value.
