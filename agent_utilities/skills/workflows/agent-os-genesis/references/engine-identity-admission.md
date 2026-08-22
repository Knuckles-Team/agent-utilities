# Engine identity admission: a process's own signer key

This is the credential that lets au's own processes — graph-os and the unified
scheduler chief among them — authenticate to the engine as themselves and be granted a
role on the engine's own independent RBAC store. It is unrelated to a human/OIDC login
and unrelated to artifact/release signing (see the note at the end of this document).
Read this before seeding, rotating, or reasoning about
`EPISTEMIC_GRAPH_SIGNER_KEYS_JSON`.

## The one rule that shapes everything else

The engine requires **the signer to be the calling principal**. From
`verify_register_identity_signature` (`epistemic-graph/src/server/auth.rs`):

```rust
let signer = registry.verify(signature, &digest)?;
if signer != context.principal() {
    return Err(SIGNER_TRUST_DENIED.to_string());
}
```

So there is no such thing as a separate "provisioner" identity that admits on another
process's behalf. A process admits using its own verified principal, signing as itself,
or it does not admit at all. `agent_utilities.security.admission_authority.
resolve_admission_authority()` is the single resolver that expresses this: it reads the
bound verified actor, uses that principal as both `agent_id` and `signer_id`, and finds
that principal's key in this process's own signer registry. It never reads a secrets
backend, so admission never depends on the engine authorization it exists to establish.

Every admission bridge — tenant (`tenant_admission_cli`), Tier-2
(`tier2_admission_cli`), and system/control-plane (`system_admission_cli`,
`system_rbac_admission`) — calls that one resolver.

## What this credential is, and is not

- It is **not** an application login. No human ever authenticates with it.
- It is **not** tenant content access. It grants nothing on any `tenant__*` graph.
- It **is** a shared HMAC key that lets a process register an engine identity
  (`RegisterIdentity`) and mint/grant a narrow RBAC role
  (`agent_utilities.security.system_rbac_admission.CONTROL_ROLE_NAME`,
  `"control:system"`) with exactly Read + Write on the engine's isolated control
  graph (`CONTROL_GRAPH_NAME`, `"__control__"`, defined in
  `agent_utilities/knowledge_graph/core/shard_topology.py`).
- **The important admission an operator must not miss:** possessing this credential is
  possessing the ability to register *any* `agent_id` with *any* `roles` on the engine
  (see "Design problem 1"). The narrow role au's own code requests is a convention of
  the calling code, not an engine-enforced ceiling.

## Who holds one

**graph-os.** It is the process that talks to the engine on the platform's behalf, and
it serves the agent-webui dashboard in-process as a supervised co-service
(`agent_utilities.server.webui_co_service`, wired by
`agent_utilities.mcp.co_service_supervisor`, enabled by `ENABLE_WEB_UI`). A co-service
runs on `_authorized_background_thread`, inheriting graph-os's verified actor and
`GraphSession` for its whole lifetime — so a browser sign-in that must enrol a new user
principal in `tenant:<slug>` is admitted by graph-os, signing as graph-os.

This is why the dashboard is served by graph-os rather than deployed beside it: a
separate frontend deployment would need its own signer entry, which means a second
identity holding unconstrained identity-plane authority (Design problem 1) for no gain.
One process, one principal, one key.

A process that self-hosts its engine as a child (the packaged single-node shape)
generates its own bootstrap key and injects it into that child, so a laptop or
single-host install needs no operator step at all. Remote engines are
operator-provisioned, per the next section.

## Provisioning a fresh environment

The key exists in exactly two places, and they must agree: the admitting process's
environment and the engine's. Both read the same variable name.

1. **Choose the principal.** This is the `agent_id` the admitting process already
   authenticates as — for the platform, graph-os's process identity. It is not a new
   name invented for admission; using a separate one is exactly what the engine
   rejects.

2. **Generate a dedicated key.** `openssl rand -hex 32`. Use a key dedicated to this
   principal so a compromise elsewhere does not become identity-plane authority here.

3. **Write it once, to the shared store.** Merge `{"<principal>": "<hex-key>"}` into
   the `EPISTEMIC_GRAPH_SIGNER_KEYS_JSON` map at the OpenBao key
   `agent-utilities/deployment` (property `EPISTEMIC_GRAPH_SIGNER_KEYS_JSON`). Merge —
   do not overwrite — or you will revoke every other signer in the map.

4. **Deliver it to both sides.** The engine reads
   `EPISTEMIC_GRAPH_SIGNER_KEYS_JSON` to verify signatures; the admitting process reads
   the same variable to produce them. Both are delivered by ExternalSecret from that
   one OpenBao entry, so there is one authoritative copy and no drift between verifier
   and signer.

5. **Restart both.** The engine reads its registry once at process start
   (`OnceLock`, `epistemic-graph/src/server/auth.rs`) — there is no live reload.

There is no separate credential to seed, and nothing in this chain is stored in the
graph. That is deliberate: a credential that authorizes writing identities must not
live behind the authorization it grants.

## Verifying it works

Run these in order, and do not assume an earlier step passing implies a later one does.

1. **The registry reached the process.** Confirm the admitting process's environment
   carries `EPISTEMIC_GRAPH_SIGNER_KEYS_JSON` and that its own principal is a key in
   that map. A process holding a registry that does not name *itself* cannot sign
   anything, and this is the single most common misconfiguration.
2. **The registry reached the engine**, and the engine restarted *after* it did.
3. **An end-to-end admission succeeds** — run the relevant bridge with `--apply` and
   confirm the identity and its grant exist on the engine afterwards, not merely that
   the call returned without raising.

## The three design problems

Documented plainly because the unease about this mechanism is warranted. Each is a
property of the design, not a hypothetical.

### 1. A signer key is unconstrained authority over identity

`register_identity` takes `roles` from the caller, and nothing engine-side restricts
*which* roles a given signer may grant (confirmed by reading the
`Method::RegisterIdentity` handler in `dispatch.rs`: once the signature verifies, the
handler applies the caller-supplied `agent_id`/`role`/`teams`/`roles` unchanged to
`try_register_agent`/`try_bootstrap_system_identity`). Any holder of any trusted signer
key can register any `agent_id` with any `roles` — up to and including `System`, which
bypasses RBAC entirely. The narrow `control:system` role au's Python code asks for is a
**convention of the calling code**, never an engine-enforced ceiling. A trusted signer
key is, in practice, identity-plane admin.

- **Blast radius:** full — a leaked key can mint an identity with `role="System"`
  (unconditional RBAC bypass on every graph).
- **Compensating control:** one signer per principal, and as few principals as the
  platform actually needs — this is the concrete reason the dashboard is served by
  graph-os rather than given a signer of its own. Plus a restrictive OpenBao policy on
  the `agent-utilities/deployment` path, and an audit trail on every
  `RegisterIdentity`, since the engine itself will not stop an out-of-scope one.

### 2. It is a shared symmetric secret

The same HMAC key must exist in the engine's registry *and* in the admitting process's.
There is no asymmetric signing — possession of the key is indistinguishable from
authorization to use it, and every additional copy is a full, independent compromise
path with no way to tell one holder's calls from another's after the fact.

- **Blast radius:** every reader of the OpenBao path is a full holder, not a scoped
  delegate.
- **Compensating control:** the signer-is-the-principal rule bounds this structurally —
  a key is only useful to the one principal it names, so it cannot be handed to a
  second component to use "on behalf of" the first. Keep the copies to the two the
  design requires (engine + that principal's process); treat a third as a rotation
  trigger, not routine hygiene.

### 3. No rotation path

The engine reads `EPISTEMIC_GRAPH_SIGNER_KEYS_JSON` once at process start
(`OnceLock`), so changing a key requires re-writing the OpenBao entry, waiting for the
ExternalSecret refresh (not immediate), and **restarting every engine pod that verifies
signatures**. Because it is a flat `{signer_id: key}` map, there is no overlap window
unless the operator deliberately keeps the old entry alongside the new one during the
transition.

## Rotation and revocation

Rotation is not a single-command operation; it is the sequence below, because of Design
problem 3. Note that rotating a key does **not** mean introducing a new `signer_id`:
the signer id is the principal, and changing it would change who the process *is*.
Rotation replaces the key material for the same principal, so the overlap window is
achieved by staging the restart, not by running two identities.

1. Generate a new key: `openssl rand -hex 32`.
2. Write the new key for the principal into the OpenBao entry.
3. Restart the **engine** pods first, so verification accepts the new key, then restart
   the admitting process so it signs with it. Between those two restarts, admission
   calls fail closed and retry — a brief, self-healing window, not silent breakage.
4. Confirm end-to-end per "Verifying it works" before considering rotation complete.

**Revocation** (the key is compromised, not merely due for rotation): overwrite the
entry and restart the engine immediately, accepting that the principal is locked out
until its own process restarts with the new key. A compromised signer key is
unconstrained identity-plane authority (Design problem 1) — treat revocation as urgent,
not as a maintenance-window item.

## Failure modes

| Symptom | Root cause | What to check |
|---|---|---|
| `AdmissionAuthorityError: this process holds no signer key for verified principal '<id>'` | The process authenticates fine but its registry does not name its own principal — so it cannot sign | Confirm `EPISTEMIC_GRAPH_SIGNER_KEYS_JSON` reached this process AND contains a `<id>` key. A process serving a UI it did not expect to admit for is the usual cause |
| `register_identity` fails with `SIGNER_TRUST_DENIED` | The signature verified, but against a signer that is not the calling principal | Never pair a signer with another identity. If a caller supplies `signer_id` explicitly, it must equal the verified principal — `AdmissionAuthority` refuses the mismatch locally so this never reaches the wire |
| `register_identity` fails with `"signature uses untrusted signer '<id>'"` | `<id>` is absent from the engine's current registry — never added, or the engine has not restarted since | Confirm the OpenBao entry, that the ExternalSecret refreshed, and that the engine pod restarted *after* that refresh |
| Scheduler logs `CypherEngineError(PermissionError)` on every tick, 0 successes | The scheduler's principal has never been registered/granted `control:system` — admission has not run, or failed before "Verifying it works" was applied | Run "Verifying it works" steps 1-3 in order |
| Admission RPC succeeds but the caller still fails with the same `PermissionError` | A grant was applied against the wrong resource selector — the single most expensive mistake seen on this mechanism | A `Pattern("tenant__homelab__*")`-style selector can **never** match `Graph("__control__")`; `IsolationLayer::provision_tenant_graph_access` only matches `tenant__<slug>__{__commons__\|default}`. The correct selector is a plain `{"Graph": "__control__"}`, exactly what `provision_system_principal_access` sends |
| Everything looks correct, but `ensure_system_principal_access` still raises | Its process-local negative-outcome cache (`_FAILURE_BACKOFF_SECONDS`, 30s) is holding a stale failure | Wait out the backoff, or restart the process — `reset_admission_cache_for_tests()` is test-only |

## Live state on this deployment

The platform runs graph-os with the engine self-hosted as an in-pod child, so its
signer key is the per-install bootstrap key that process generates and injects into
that child (`_load_or_create_engine_bootstrap_signer_key`), not an operator-provisioned
registry entry: `EPISTEMIC_GRAPH_SIGNER_KEYS_JSON` **does not exist** in the graph-os
pod environment here, and it does not need to. The agent-webui dashboard runs as a
co-service inside that same process and therefore admits under that same identity.

The operator-provisioned chain in "Provisioning a fresh environment" applies to a
**remote** engine — one this process did not start. It is documented as what correct
provisioning would be, **not as something currently live here**. Do not treat any write
command in this document as having been run against the real cluster; the OpenBao token
available in an ordinary session is read-only, and every write step requires an operator
acting deliberately, outside this skill's automated path.

## Not to be confused with: build/artifact signing

`security-and-operations.md`'s "Signing keys specifically" section describes a
*different* signer concept — a key that signs release artifacts, resolved from a
versioned store reference at build time, deliberately refusing an environment-variable
reference. That signer proves *what was built matches what was reviewed*. The signer
described in this document proves *who is allowed to register an engine identity*.
They share the word "signer" and nothing else — never assume a control appropriate for
one applies to the other.
