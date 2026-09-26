# Drift-proof release & versioning

> **Design principle (operator):** *"We prefer to design systems that cannot drift at all."*
> Prevention over detection, detection over repair.

Every value that appears in two places will eventually appear differently in those two
places. This page records, for each release value that has actually drifted, which
mechanism now prevents it and what class that mechanism reaches.

| Class | Guarantee | Where the failure surfaces |
|:--|:--|:--|
| **A — cannot drift** | The value is *derived* from one authority, not duplicated | Nowhere; there is no second copy to disagree |
| **B — cannot be committed** | A gate fails closed at generation/commit time | Before the wrong value enters the tree |
| **C — caught at promotion** | Detected during release assembly | Before publication |
| **D — detected after the fact** | A report | After the damage |

---

## 1. Attestation compatibility band — Class A

`connector_source_attestation.SOURCE_COMPATIBILITY` used to be a literal that
`.bumpversion.cfg` rewrote on every bump, compared for *exact string equality* by the
admission gate. On 2026-07-28 the commit "Bump 2.1.0 → 2.1.1" rewrote the floor to
`>=2.1.1,<3` and invalidated all 68 provider attestations — 22 minutes after the only
key that could re-sign them was believed lost. **Every patch release did this.**

Now:

* `source_compatibility()` **derives** a *minor-floor band* (`>=MAJOR.MINOR.0,<MAJOR+1`)
  from `agent_utilities._version.__version__` and from the engine version pinned in
  `deploy/release/compatibility-matrix.yml`. A patch bump recomputes the identical
  string — there is nothing to rewrite.
* Validation is **containment**, not equality: a declared band is admissible when it is
  bounded on both sides and still contains the released version. An attestation signed
  under 2.1.0 stays valid at 2.1.1 and 2.9.9, and is rejected only at the major
  boundary it already declared.
* `.bumpversion.cfg` no longer targets the module, and
  `scripts/check_version_consistency.py` emits `bumpversion-attestation-coupling:<path>`
  if any bump section reaches an attestation input again (the Class-B backstop).

**Test:** `tests/unit/knowledge_graph/integrations/test_connector_source_attestation_drift.py`
— simulates a patch bump against a freshly signed attestation and asserts the bytes and
the signature both survive, plus the structural assertion that no bump section can reach
the band.

## 2. Connector MCP server alias — Class A (+ Class B for the source tree)

`deploy/mcp-fleet.registry.yml` is the one authority for what a fleet server is called.
Every provider *restated* that name in `mcp_source_presets.json`, and
`generate_connector_manifests.py` copied it verbatim: 27 providers named their
distribution (`github-agent`) where the fleet runs the service (`github-mcp`), and 9
more named services the committed registry did not contain at all.

Now:

* The generator **derives** `server` from the registry by distribution name
  (`registry_server_alias`) and normalises the preserved `raw` preset with it, so a
  wrong alias cannot enter a signed manifest. A provider absent from the registry has
  no derivable alias and **fails closed before any artifact is projected**.
* A preset that restates a *different* alias is a hard error, not a warning.
* `scripts/check_connector_source_presets.py` reports (and with `--fix` repairs) the
  same disagreement across the fleet, so the source tree cannot hold it silently.

### Why the registry had rotted

Host ports were assigned by list index, so adding one provider renumbered the published
port of every service after it — regeneration was a deployment-breaking change, so the
registry was left stale instead. `gen_mcp_fleet_registry.py` now **preserves already
allocated ports** and gives only genuinely new services the next free port, which makes
regeneration a no-op for unchanged services and lets the freshness expectation be
enforced at all.

**Test:** `tests/unit/scripts/test_connector_server_alias_drift.py`.

## 3. Cross-project version cascade — Class A

`scripts/release/version_cascade.py` models the release train as three tiers:

```
epistemic-graph X.Y.Z
  -> agent-utilities pyproject: epistemic-graph[full]>=X.Y.Z,<{X+1}.0.0   (+ the matrix)
     -> ~68 providers: agent-utilities[extras]>={A}.0.0,<{A+1}.0.0
```

Provider pins are a **major band derived from the agent-utilities major**, so a patch or
minor bump produces *zero* provider edits by construction. Only a major bump moves them —
and it moves *all* of them, which is exactly the fastmcp-4 case (au 2.x → 3.0.0
invalidates every `agent-utilities>=2.0.0,<3.0.0` ceiling). The planner reports that as
`severity: major` with an explicit `breaking_notes` entry.

Dry-run is the default; `--apply` writes files and verifies each replacement target still
matches before writing (fail closed on concurrent modification). It never runs git — each
provider is its own repository and committing/pushing is a separate, deliberate act.

**Test:** `tests/unit/release/test_version_cascade.py`.

## 4. Release-signing key custody — Class A (custody) + Class B (agreement)

The 2026-07-28 incident had two distinct causes.

**Custody.** The signing seed's source of truth was `env://` backed by a local
`~/.config/agent-utilities/runtime-secrets.json`. That file was overwritten in place with
a different seed; it has no version history and no backup, so the original looked
destroyed. The *same seed* had been written to OpenBao KV v2 at
`apps/agent-utilities` minutes earlier and survived untouched — because KV v2 keeps every
version. A store that can be silently overwritten beyond recovery is not custody.

`release_signer_for_publication()` is now the only signer a publication path may use, and
it **refuses any reference that is not `vault://` or `secret://`**. The required
configuration is:

```
SECRETS_BACKEND=vault
SECRETS_VAULT_URL=<openbao url>
SECRETS_VAULT_MOUNT=apps
ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF=vault://agent-utilities#ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY@2
```

(the `vault` extra — `hvac` — must be installed on the release host).

**Which token authenticates this read (D-AR-4).** `VaultBackend` (`security/secrets_client.py`)
authenticates with whatever static token is on `VAULT_TOKEN` (or `token=` passed explicitly) when
no OIDC/AppRole/Kubernetes auth is configured. That token **must carry the `agent-apps-rw`
policy** — the same policy every service's own `OPENBAO_TOKEN` carries, scoped to `create/read/
update/delete` on `apps/data/*` and `list/read/delete` on `apps/metadata/*`. Two credentials look
plausible here and only one is correct:

- ✅ **`OPENBAO_TOKEN`** (`apps/<service>`, e.g. `apps/openbao-mcp`, policy `agent-apps-rw`) — reads
  `apps/data/agent-utilities` and `apps/metadata/agent-utilities` directly. This is what
  `VAULT_TOKEN` should resolve to for release signing. A freshly-minted short-TTL `agent-apps-rw`
  token (see below) also works and is the more auditable choice for a one-shot release run.
- ❌ **`OPENBAO_ADMIN_TOKEN`** (`apps/openbao-mcp`, policy `agent-apps-token-minter`) — **do not**
  set `VAULT_TOKEN` to this value. Despite the name, this token is deliberately scoped to
  `create`/`update`/`sudo` on `auth/token/create` **only** (minting fresh `agent-apps-rw` tokens
  for others to use) and holds zero capability on `apps/data/*` or `apps/metadata/*` — it 403s on
  both `apps/data/agent-utilities` and `apps/metadata/agent-utilities` by design, not by bug (see
  `services/openbao/k8s/bootstrap-policies.sh` and the openbao-mcp `secret-vault-manager` skill's
  `rotation-operational-facts.md` fact 8). If you need a fresh scoped credential instead of the
  shared `OPENBAO_TOKEN`, mint one FROM `OPENBAO_ADMIN_TOKEN` first
  (`POST auth/token/create {"policies":"agent-apps-rw","ttl":"..."}` — `policies` must be a plain
  string, not a JSON list) and use *that* minted token as `VAULT_TOKEN`; never grant
  `OPENBAO_ADMIN_TOKEN` itself direct data-plane access.

**Pin the KV version (D-AR-5).** The bare form above (no `@<version>`) resolves whatever KV
v2 version is *current* at signing time — reproducible only as long as nobody else writes to
`apps/agent-utilities` in the meantime. `resolve_ref()` (`security/secrets_client.py`) supports
an explicit `#field@<version>` pin (CONCEPT:AU-KG.ontology.release-key-rotation, D-OC-1) so the
reference names an immutable version instead of "latest"; the release host's actual
`ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF` should always carry an explicit `@<version>` suffix
(bump it deliberately, as a recorded step, whenever `deploy/release/signing-key-rotation.yml`
gains a new entry) rather than track "latest" implicitly.

**Agreement.** The mismatch was only discovered at *admission* time, long after artifacts
had been produced. `assert_signing_key_matches_locks()` performs the same comparison at
**signing** time, before a single byte is written, and refuses when the seed that would
sign is not a seed `ontology.lock` pins. `scripts/update_ontology_lock.py` no longer adds
the ambient runtime key to its trusted set — trust comes from declared sources only.

**Rotation is a recorded transaction.** `deploy/release/signing-key-rotation.yml` is an
ordered ledger: each entry names the key rotated *from*, the key rotated *to*, the
authorisation, and the artifacts that moved. `release_trusted_public_keys()` includes the
key the newest entry designates, so a new trust anchor becomes trusted by being
**recorded**, never by merely being present.

**Test:** `tests/unit/knowledge_graph/ontology/test_release_key_custody.py`.

**The controlled release job (GOC-16/BUG-234).** This section documents the *read path*
mechanism; it does not by itself give any real pipeline a way to reach OpenBao with the
right identity. `docs/release/connector-manifest-signing-custody.md` is the operator
runbook that closes that gap for connector-manifest signing specifically: the exact
OpenBao policy/Vault-Kubernetes-auth-role commands, the Kubernetes Job
(`deploy/release/connector-manifest-signing-job.yaml`) that holds the key via workload
identity only (never a materialized k8s `Secret`, never a GitHub Actions secret), and the
keyless diff/freeze report (`connector-manifest-diff` in `.github/workflows/advisory.yml`)
that gives the operator the exact frozen commit + dependency-lock digest to sign against.
