# Engine identity admission: au's own system-principal credential

This is the credential that lets au's own background daemons — the unified scheduler
chief among them — authenticate to the engine as themselves and be granted a role on
the engine's own independent RBAC store. It is unrelated to a human/OIDC login and
unrelated to artifact/release signing (see the note at the end of this document). Read
this before seeding, rotating, or reasoning about `engine-admission/provisioner`.

## What this credential is, and is not

- It is **not** an application login. No human ever authenticates with it.
- It is **not** tenant content access. It grants nothing on any `tenant__*` graph.
- It **is** a shared HMAC key that lets one caller register an engine identity
  (`RegisterIdentity`) and mint/grant a narrow RBAC role
  (`agent_utilities.security.system_rbac_admission.CONTROL_ROLE_NAME`,
  `"control:system"`) with exactly Read + Write on the engine's isolated control
  graph (`CONTROL_GRAPH_NAME`, `"__control__"`, defined in
  `agent_utilities/knowledge_graph/core/shard_topology.py`).
- **The important admission an operator must not miss:** possessing this credential is
  possessing the ability to register *any* `agent_id` with *any* `roles` on the engine
  — not merely the narrow role this repo's own calling code happens to ask for. See
  "Design problem 1" below before treating this as a scoped, low-stakes credential.

If you are asking "does creating this make me a trust root," the honest answer is:
**yes, for engine identity as a whole**, not just for the scheduler. Provision it with
that in mind.

## The full chain

1. **au side, credential resolution.**
   `agent_utilities/security/system_rbac_admission.py`'s `resolve_provisioner_authority()`
   reads secret key **`engine-admission/provisioner`** via
   `agent_utilities.security.secrets_client.create_secrets_client()` (the deployment's
   configured `SecretsClient` backend — `engine` or `vault`, see `SECRETS_BACKEND`
   below). It expects the value to be a JSON object:

   ```json
   {"agent_id": "<provisioner-agent-id>", "signer_id": "<signer-id>", "signer_key": "<hex-key>"}
   ```

   `agent_id`, `signer_id`, and `signer_key` above are placeholders — never a real
   value. A missing key raises `SystemAdmissionError` naming the exact secret key and
   the CLI command to seed it (see "Failure modes").

2. **au side, the RPC.** `provision_system_principal_access()` calls the engine's
   `consensus.register_identity(agent_id, role, teams, roles, signer_id=…,
   signer_key=…)` for the principal being admitted (by default, au's own scheduler
   process identity), then `rbac.add_role("control:system")` and two
   `rbac.add_grant("control:system", {"Graph": "__control__"}, <Read|Write>,
   "Allow")` calls. Both the `client.register_identity` call's `signer_id`/`signer_key`
   and the `agent_id` of the principal being registered are distinct: `signer_id`/
   `signer_key` authenticate *who is making the call*; `agent_id` names *who is being
   registered*. They do not have to be the same identity, and for this module they
   normally are not (the provisioner registers the scheduler, not itself).

3. **Engine side, signature verification.** `epistemic-graph/src/server/auth.rs`
   (`SignerKeyRegistry`, ~lines 1576-1660) verifies a signature of the form
   `signer_id:hex_hmac` using **HMAC-SHA256**, keyed from the trusted-signer map the
   engine loads once at process start from the env var
   **`EPISTEMIC_GRAPH_SIGNER_KEYS_JSON`** (a JSON object, `{signer_id: key}`). If the
   `signer_id` in the signature is not a key in that map, or the HMAC does not verify,
   the `RegisterIdentity` call is rejected before it ever reaches the RBAC or identity
   store (`epistemic-graph/src/server/dispatch.rs`, the `Method::RegisterIdentity`
   handler, ~lines 3652-3691). Once the signature verifies, the handler applies the
   caller-supplied `agent_id`/`role`/`teams`/`roles` **as given** — see "Design problem
   1" for why that matters.

4. **Delivery of the engine-side key.** `EPISTEMIC_GRAPH_SIGNER_KEYS_JSON` is delivered
   to the engine process as an environment variable sourced from a Kubernetes Secret
   (conventionally named `epistemic-graph-secrets` in this deployment's manifests),
   which an ExternalSecret keeps in sync from an OpenBao-compatible ClusterSecretStore,
   reading key **`agent-utilities/deployment`** (property
   `EPISTEMIC_GRAPH_SIGNER_KEYS_JSON` within that KV entry). Because the env var is
   read once at process start (`OnceLock` in the Rust source), a changed value never
   takes effect without a pod restart — see "Rotation and revocation."

5. **What the grant buys.** Once the identity is registered and holds
   `control:system`, `RbacPolicy::evaluate` on the engine allows Read + Write on
   `Graph("__control__")` for that identity, which is exactly what
   `core/schedule_engine.py`'s `run_scheduler_tick` needs to run its
   `MATCH (s:Schedule) …` reads and its run-state upserts on every due tick. Before
   admission, every tick fails identically with `CypherEngineError(PermissionError)` —
   this is the literal shape of the outage this credential exists to end (see the
   `system_rbac_admission.py` module docstring, BUG-295).

## Provisioning a fresh environment

Do these in order. Nothing here touches a live cluster by itself — treat every command
below as something an operator runs deliberately, not something this skill or any
agent runs unattended.

1. **Decide on a dedicated signer, distinct from any cluster-admin or other-purpose
   signer.** The mechanism gives no per-signer role scoping (Design problem 1), so the
   only compensating control available is separation: mint one `signer_id` used for
   nothing except au system-principal admission, never reused for a human's Vault
   login, a CI signer, or any other engine-admin bridge (`tenant_admission_cli.py`,
   `tier2_admission_cli.py` — check whether those already have their own dedicated
   signers before reusing one). Choose a `signer_id` that says what it is, e.g.
   `<environment>-au-system-admission` — never a generic name like `admin`.

2. **Generate the key.** A 256-bit random value, hex-encoded, generated on a trusted
   workstation and never typed into chat, a ticket, or a log:

   ```bash
   openssl rand -hex 32
   ```

   Treat the output as the credential itself. Do not echo it again after generating
   it; pipe it directly into the next two writes.

3. **Add it to the engine's trusted-signer map.** Read the current
   `EPISTEMIC_GRAPH_SIGNER_KEYS_JSON` value at OpenBao key `agent-utilities/deployment`
   (property `EPISTEMIC_GRAPH_SIGNER_KEYS_JSON`), merge in
   `{"<signer-id>": "<hex-key>"}` — **merge, never overwrite**, since this map may
   already carry other trusted signers — and write the merged JSON back to the same
   key/property. This requires a write-capable token on that OpenBao path (see "Live
   state on this deployment" below for why that is currently not available here).

4. **Write the matching provisioner secret.** The admission caller reads
   `engine-admission/provisioner` from the deployment's *configured*
   `SecretsClient` backend — check `SECRETS_BACKEND` (default `"engine"`,
   `agent_utilities/core/config.py`) before assuming which store this is; see "Design
   problem 3" for why that default is the wrong choice for this specific secret. Write:

   ```json
   {"agent_id": "<provisioner-agent-id>", "signer_id": "<signer-id>", "signer_key": "<hex-key>"}
   ```

   using the *same* `<signer-id>`/`<hex-key>` written in step 3. `agent_id` here
   identifies the provisioner for audit purposes; it is not the identity that ends up
   registered (that is the principal passed to `provision_system_principal_access`,
   normally au's own process identity, e.g. `graph-os-scheduler`).

   - If the backend is `vault`: `python3 -m agent_utilities.security.cli set
     engine-admission/provisioner --value-ref <vault://path/to/the/json>`.
   - If the backend is `engine`: see Design problem 3 before writing it this way —
     provisioning the store that adjudicates its own admission credential is a real
     ordering hazard, not a hypothetical one.

5. **Restart the engine pods that read `EPISTEMIC_GRAPH_SIGNER_KEYS_JSON`.** The map is
   loaded once at process start; a live pod never observes the OpenBao/Secret update.
   Confirm the ExternalSecret has actually refreshed (its `refreshInterval`, typically
   on the order of an hour, not immediate) before restarting, or the restart will just
   reload the stale value.

6. **Run admission.** Either let it happen automatically at the next au daemon boot
   (`ensure_system_principal_access`, called once from `kg_server.py`'s daemon-role
   bootstrap path), or run it explicitly with the operator-gated CLI:

   ```bash
   echo '{"role": "control:system",
          "principals": [{"agent_id": "graph-os-scheduler", "role": "Agent",
                           "teams": [], "existing_roles": []}]}' \
     | python3 -m agent_utilities.security.system_admission_cli --apply
   ```

   Run it once without `--apply` first — the default is a dry run against an
   in-memory fixture and never touches a live engine or resolves a real credential, so
   it is a safe way to confirm the manifest is well-formed before trusting it to the
   live path.

## Verifying it works

"The pod started" proves nothing. Prove all three of the following, in order:

1. **Admission ran and reported success.** Either the `system_admission_cli.py
   --apply` run printed `all_admitted=True`, or the au daemon's own log shows the
   `ensure_system_principal_access` info line: `system-principal admission: <agent
   reference> admitted into control:system`. If neither appears, admission has not run
   — do not proceed to steps 2-3 assuming it has.

2. **The identity is registered with the expected role, engine-side.** The engine log
   shows `RegisterIdentity committed` for the expected `agent_id`
   (`epistemic-graph/src/server/dispatch.rs`'s `RegisterIdentity` handler logs this on
   every successful commit — bootstrap or not). If you can query RBAC state directly,
   confirm the identity's `roles` includes `control:system` and that a grant exists
   for `("control:system", Graph("__control__"), Read, Allow)` and the `Write` sibling.

3. **The observable that actually matters: a previously-failing control-graph read now
   succeeds.** Before provisioning, au's scheduler logs a `CypherEngineError`
   wrapping a `PermissionError` on essentially every tick (`core/schedule_engine.py`'s
   `run_scheduler_tick`, its `MATCH (s:Schedule) …` read against `__control__`). After
   provisioning and a scheduler tick has run, that error must stop appearing, and
   entries in `deploy/schedules.yml` that are due must actually fire (visible as the
   `_upsert` run-state advance touching the `:Schedule` node, and downstream jobs
   getting enqueued). **This is the named, load-bearing observable** — not pod
   readiness, not "the secret exists," not "the CLI printed success." A run that
   stops at step 1 or 2 has not proven the outage is over.

## The four design problems

Documented here plainly because this mechanism was flagged as something the operator
is uneasy with, and the unease is warranted. Each is a property of the current
design, not a hypothetical.

### 1. A signer key is unconstrained authority over identity

`register_identity` takes `roles` from the caller, and nothing engine-side restricts
*which* roles a given `signer_id` may grant (confirmed by reading the
`Method::RegisterIdentity` handler in `dispatch.rs`: once the signature verifies, the
handler applies the caller-supplied `agent_id`/`role`/`teams`/`roles` unchanged to
`try_register_agent`/`try_bootstrap_system_identity`). Any holder of any trusted
signer key can register any `agent_id` with any `roles` — up to and including
`System`, which bypasses RBAC entirely. The narrow `control:system` role this repo's
Python code asks for is a **convention of the calling code**, never an
engine-enforced ceiling. A trusted signer key is, in practice, identity-plane admin.

- **Blast radius:** full — a leaked or over-shared signer key can mint an identity
  with `role="System"` (unconditional RBAC bypass on every graph) or any other role
  combination, not just the narrow one this module happens to request.
- **Compensating control:** a dedicated signer per purpose (see "Provisioning a fresh
  environment," step 1) so a compromise of the au system-admission signer does not
  also compromise whatever else shares a signer key; a restrictive OpenBao policy on
  the `agent-utilities/deployment` path scoped to only the identities that legitimately
  need to read it; and an audit trail (log/alert) on every `RegisterIdentity` call,
  since the engine itself will not stop an out-of-scope one.

### 2. It is a shared symmetric secret

The same HMAC key must exist in the engine's `EPISTEMIC_GRAPH_SIGNER_KEYS_JSON`
registry *and* in whatever store the admission caller resolves via
`resolve_provisioner_authority()`. There is no asymmetric signing (no keypair, no
signature that only the holder of a private half can produce) — possession of the
key is indistinguishable from authorization to use it, and every additional place
the key is copied to is a full, independent compromise path with no way to tell one
holder's calls from another's after the fact.

- **Blast radius:** every reader of the OpenBao path and every reader of the
  `engine-admission/provisioner` secret is a full holder, not a scoped delegate.
- **Compensating control:** minimize the number of places the key is written to
  (ideally exactly two: the engine's signer map and the one provisioner secret); treat
  a third copy anywhere as a rotation trigger, not routine hygiene.

### 3. Bootstrap circularity

`SECRETS_BACKEND` defaults to `"engine"` (`agent_utilities/core/config.py`), meaning
the credential that authorizes writing identities to the Knowledge Graph can itself be
stored *in that Knowledge Graph* — `InEpistemicGraphBackend`
(`agent_utilities/security/secrets_client.py`) stores secret values as `:Secret` nodes
on a dedicated `__secrets__` graph, read and written through the same
`GraphComputeEngine` the admission RPC itself depends on. If `engine-admission/
provisioner` is stored this way, reading it at boot requires the calling process to
already be able to read the `__secrets__` graph — which is exactly the kind of
engine-side authorization this whole mechanism exists to bootstrap in the first
place. In practice this has not yet caused a deadlock only because the engine's
default/bootstrap identity historically has broader implicit access than the narrow
`control:system` role this module grants; that is not a property to depend on.

- **Blast radius:** an environment that stores this specific secret in the `engine`
  backend has coupled "can I read my own admission credential" to "is the engine
  already in a state where secret reads succeed" — a genuine bootstrap-ordering
  hazard on any change to engine-side default access.
- **Compensating control:** store `engine-admission/provisioner` specifically in a
  Vault-compatible backend (`SECRETS_BACKEND=vault`, or a per-secret override if one
  exists), even in an otherwise engine-backend deployment. This is the one secret in
  the whole admission chain where "prefer workload identity, store in the engine" is
  the wrong default — it must live somewhere that does not itself require engine RBAC
  to read.

### 4. No rotation path

The engine reads `EPISTEMIC_GRAPH_SIGNER_KEYS_JSON` once at process start
(`OnceLock`, `epistemic-graph/src/server/auth.rs`). Changing a signer key therefore
requires: re-writing the OpenBao entry, waiting for the ExternalSecret's refresh
interval (not immediate), and **restarting every engine pod that verifies signatures**
— there is no live-reload. Because it is a flat `{signer_id: key}` map, there is no
overlap window unless the operator deliberately keeps the old signer's entry present
alongside the new one during the transition; removing the old entry and adding the new
one in the same write is a hard cutover with a restart-shaped outage window for any
caller still using the old key.

- **Blast radius:** a rotation performed as a single overwrite briefly breaks every
  in-flight admission call signed with the old key, with no automatic recovery until
  the caller retries after the restart completes.
- **Compensating control:** always rotate as two writes — add the new
  `{signer_id: key}` entry *alongside* the old one, restart, confirm the new signer
  works end-to-end (repeat "Verifying it works" above with the new signer), *then*
  remove the old entry in a second write and restart again. See "Rotation and
  revocation" for the concrete sequence.

## Rotation and revocation

Rotation is not a single-command operation; it is the sequence below, because of
Design problem 4.

1. Generate a new key for a **new** `signer_id` (do not reuse the old `signer_id` with
   a new key — that makes it impossible to distinguish old-key and new-key calls
   during the overlap window). `openssl rand -hex 32`.
2. Read the current `EPISTEMIC_GRAPH_SIGNER_KEYS_JSON` map at OpenBao key
   `agent-utilities/deployment`, add the new `{signer_id: key}` entry, **keep the old
   entry**, write the merged map back.
3. Restart every engine pod that verifies admission signatures. Confirm via "Verifying
   it works" that the *new* signer can successfully register an identity end-to-end
   before touching the old entry.
4. Update `engine-admission/provisioner` to the new `{"agent_id", "signer_id",
   "signer_key"}` triple (new signer). Confirm the au side picks it up: either wait
   out `resolve_provisioner_authority`'s process-local cache/backoff, or restart the au
   daemon.
5. Once every caller is confirmed on the new signer, remove the **old** entry from
   `EPISTEMIC_GRAPH_SIGNER_KEYS_JSON` and restart the engine pods again. Until this
   step, the old key remains fully live — rotation is not complete at step 4.

**Revocation** (the old key is compromised, not merely due for routine rotation) skips
the overlap window: remove the old `signer_id` entry immediately and restart, accepting
that any caller still holding only the old key is locked out until re-provisioned with
the new signer. A compromised signer key is unconstrained identity-plane authority
(Design problem 1) — treat revocation as urgent, not as a maintenance-window item.

## Failure modes

| Symptom | Root cause | What to check |
|---|---|---|
| Scheduler logs `CypherEngineError(PermissionError)` on every tick, 0 successes | The scheduler's own process identity has never been registered/granted `control:system` on the engine — admission has not run, or ran and failed silently before this doc's "Verifying it works" was applied | Run "Verifying it works" steps 1-3 in order; do not assume step 1 succeeding implies step 3 does |
| `SystemAdmissionError`/`SystemAdmissionCliError` naming a missing secret key | `engine-admission/provisioner` does not exist in the configured `SecretsClient` backend (the expected state on an unprovisioned environment — see "Live state on this deployment") | Seed it per "Provisioning a fresh environment," step 4, in the *same* backend `SECRETS_BACKEND` currently resolves to |
| Admission RPC succeeds (no exception) but the scheduler still fails with the same `PermissionError` | A grant was applied against the wrong resource selector — the single most expensive mistake seen on this mechanism so far | A `Pattern("tenant__homelab__*")`-style grant (or any `Pattern(...)` selector) can **never** match `Graph("__control__")`; `IsolationLayer::provision_tenant_graph_access` only ever matches `tenant__<slug>__{__commons__\|default}` graph names. The correct selector is a plain `{"Graph": "__control__"}`, exactly what `provision_system_principal_access` sends — if a grant was applied by hand instead of through that function, re-check the resource shape before assuming the role name or teams were wrong |
| `register_identity` RPC fails with `"signature uses untrusted signer '<signer_id>'"` | The `signer_id` in the provisioner secret is not present in the engine's current `EPISTEMIC_GRAPH_SIGNER_KEYS_JSON` — either it was never added, or the engine pod has not restarted since it was added | Confirm the OpenBao entry, confirm the ExternalSecret refreshed, confirm the engine pod restarted *after* that refresh |
| Admission worked once, then started failing again after an engine-side key rotation | A signer key was removed from `EPISTEMIC_GRAPH_SIGNER_KEYS_JSON` (Design problem 4's hard-cutover risk) while a caller was still resolving the old `signer_id`/`signer_key` pair from its own secret | Complete "Rotation and revocation" as a full sequence, including updating `engine-admission/provisioner` to the new signer before removing the old engine-side entry |
| Everything above looks correct, but `ensure_system_principal_access` still raises | Its process-local negative-outcome cache (`_FAILURE_BACKOFF_SECONDS`, 30s) is still holding a stale failure from before the fix was applied | Wait out the backoff window, or restart the au daemon process to clear in-memory state — `reset_admission_cache_for_tests()` is test-only and not available at runtime |

## Live state on this deployment

As of this writing, `engine-admission/provisioner` **does not exist** in either
configured secrets backend on this deployment, and the only OpenBao token available in
this session is read-only (a policy scoped to reads, not writes). **The chain above is
therefore documented as what correct provisioning would be, not as something currently
live here.** Do not treat any command in this document as having been run against the
real cluster; every write step requires an operator with write access to both the
OpenBao entry and the provisioner secret, executing deliberately, outside of this
skill's own automated path.

## Not to be confused with: build/artifact signing

`security-and-operations.md`'s "Signing keys specifically" section describes a
*different* signer concept — a key that signs release artifacts, resolved from a
versioned store reference at build time, deliberately refusing an environment-variable
reference. That signer proves *what was built matches what was reviewed*. The signer
described in this document proves *who is allowed to register an engine identity*.
They share the word "signer" and nothing else — never assume a control appropriate for
one applies to the other.
