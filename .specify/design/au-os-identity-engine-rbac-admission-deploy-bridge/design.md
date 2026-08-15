# Design Document: Engine-RBAC admission deployment bridge (BUG-068/BUG-038)

CONCEPT:AU-OS.identity.engine-rbac-admission-deploy-bridge

> `agent_utilities/security/tier2_admission_cli.py`
> (`run_tier2_admission`, `resolve_provisioner_authority`, `main`), driving
> `agent_utilities.security.engine_rbac_admission.provision_tier2_admission`.

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| `agent_utilities.knowledge_graph.maintenance.graph_ownership_apply` / `scripts/apply_graph_ownership_grants.py` | the existing DEFAULT-IS-DRY-RUN, `--apply`-required convention this module reuses | high | OS |
| `AdmissionPolicy.decide` (Wire-First's own worked example of a policy engine nobody called) | the failure mode this bridge exists to close | high | OS |

### Extension Analysis

- **Primary Extension Point**: `provision_tier2_admission` itself — fully
  implemented, unit-tested, and (before this bridge) never called from
  anywhere real.
- **Extension Strategy**: augment — add the missing caller, not a new
  admission mechanism.
- **New Concept Required?**: Yes — the deployment-tooling bridge that
  actually invokes it, resolving real credentials, is itself the decision.

## Problem

`provision_tier2_admission` (engine-RBAC admission for a new service
identity) existed, fully implemented and unit-tested, with zero live
callers — BUG-068/BUG-038's "nothing calls it" gap, the exact Wire-First
failure mode this repo's own `AGENTS.md` names explicitly (importable,
tested, never invoked).

## Decision

`tier2_admission_cli.py` is the deployment-tooling bridge that actually RUNS
`provision_tier2_admission` against a resolved manifest, resolving the
provisioner's signer credentials from the configured `SecretsClient` — never
minting, hard-coding, or persisting one itself (this repo's "Secrets &
credential retrieval" doctrine).

- **DEFAULT-IS-DRY-RUN**: mirrors `graph_ownership_apply`/
  `apply_graph_ownership_grants.py`'s convention for the same reason that
  pair uses it — this is another live, mutating, security-sensitive
  engine-admin RPC, and the codebase's existing precedent for "same shape of
  danger" is exactly that split, not a freshly invented one. `--apply` is
  required to actually mutate.
- **Two callers, by design**: `python3 -m
  agent_utilities.security.tier2_admission_cli` (a manifest JSON file/stdin,
  dry-run unless `--apply`) for standalone/human use, and
  `run_tier2_admission` called **in-process** by a Tier-1 provisioner
  (`agent-webui/scripts/provision_identity.py`'s `tier2-admission` stage)
  immediately after it resolves a service's Keycloak identity — so the
  manifest's `agent_id` is always the JUST-RESOLVED value that will appear as
  that service's own `VerifiedRequestContext.agent_id`, never a value guessed
  ahead of time.
- **Injectable client/secrets**: the `client`/`secrets_client` parameters let
  a caller (or a test) inject a `FixtureEngineAdmissionClient` and a fake
  secrets source, exercising the real credential-resolution + admission code
  path without constructing a `LiveEngineAdmissionClient` or touching a live
  secrets backend.

## Wire-First

Both callers are real: the CLI entrypoint's own tests exercise the
CLI-wiring proof, and the Tier-1 provisioner call site closes BUG-068/
BUG-038 for the deployment path. See the module's own header for exactly
what "fresh-store deploy" evidence this represents vs. what remains
unexecuted.
