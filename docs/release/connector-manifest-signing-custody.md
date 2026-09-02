# Connector-manifest signing custody path (GOC-16 / BUG-234 / GOC-84)

This page is the operator runbook for the ONE remaining step BUG-234 named: making the
already-built release-signing custody mechanism (`docs/architecture/drift_proof_release.md`
§4, D-DP-2/D-OC-1) reachable from a **controlled release job**, instead of nowhere. It is
written for the person who will actually run this, not as design narrative.

## What already exists (do not rebuild)

* The real signing key already lives in OpenBao: mount `apps`, secret `agent-utilities`,
  field `ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY`, KV v2 version `2` — confirmed against the
  live cluster on 2026-07-31 (`plans/au-eg-program/deferred/lane-openbao-custody.md`,
  D-OC-1/D-OC-2), derived public key `QkigdPNpcUU7x7NcSkwCsUXIQpaFncMQbYSoiSgI2SY`, matching
  `agent_utilities/knowledge_graph/ontology.lock` exactly. **The key is not missing. No
  environment that can reach OpenBao with the right identity has ever been wired up.**
* `agent_utilities/knowledge_graph/ontology/ontology_integrity.py`
  (`release_signer_for_publication`, `ReleaseSigner`) is the ONLY signer any generator may
  use, refuses anything but a versioned `vault://`/`secret://` reference, and fails closed
  (`ReleaseSigningError`) rather than falling back to an ephemeral key. Proven against a
  fake OpenBao transport in `tests/unit/security/test_openbao_signing_custody.py`.
* All four manifest generators (`scripts/generate_connector_manifests.py`,
  `scripts/generate_native_connector_manifest.py`,
  `scripts/generate_connector_capability_bundles.py`, and the fingerprint step in
  `scripts/certify_connector_tool_schemas.py`) already call this signer internally — this
  page's job is never to re-implement signing, only to give it a place with the real key.
* `agent_utilities/knowledge_graph/ontology/connector_manifest_gate.py` already fails
  closed on an unsigned manifest, a tampered manifest, or a wrong signer — this is
  `precheck_source`, the gate every `sync_source` call goes through **before any tool is
  exposed**.

## What GOC-16 built (this lane)

| Piece | Path |
|---|---|
| The controlled-release orchestrator (freeze → regenerate → sign → verify) | `scripts/release/regenerate_and_sign_connector_manifests.py` |
| The Kubernetes Job template that holds the real key via OpenBao workload identity | `deploy/release/connector-manifest-signing-job.yaml` |
| The keyless diff/freeze report job (GitHub Actions, `workflow_dispatch`-only) | `.github/workflows/advisory.yml` → `connector-manifest-diff` |
| Source-only input/output custody contract (synthetic fixture + static gate) | `tests/fixtures/release/connector-manifest-signing-inputs.yml`, `tests/unit/release/test_connector_manifest_signing_job_contract.py` |
| Known-bad proofs for the four named adversarial cases | `tests/unit/knowledge_graph/ontology/test_connector_manifest_signing_known_bad.py` |

(GOC-16's original `dependency_lock_digest` signed pin was removed: it made every
`pyproject.toml` dependency addition a release-Job-gated operation for no benefit in an
internal deployment. Provenance now records `ProvenanceSpec.source_commit` — the git
commit SHA a manifest was generated from — nothing more.)

**Not built here, and deliberately not attempted:** regenerating and signing the REAL
bundled fleet manifests. That is GOC-84's own work, hard-blocked on GOC-83's lock freeze
landing first (`plans/graph-os-completion-program/lanes/GOC-84-connector-manifest-
implementation-parity.md`) — this lane builds the mechanism GOC-84 will run, it does not
run GOC-84's regeneration itself, and it never requests, generates, or handles the actual
key.

## Exactly what the operator must supply, and where

Nothing new needs to be typed into a `.env` file, a GitHub secret, or an agent session.
Everything below is infrastructure the operator applies directly against the live cluster.

### 0. The signing key itself — mint it and store it under a version

**Verify first; this step is not always needed.** As of 2026-08-20 it *is*: a cluster-wide
sweep found `ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY` in no `ExternalSecret` and no `Secret`,
so the reference `vault://agent-utilities#ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY@2` that
`deploy/release/connector-manifest-signing-job.yaml` resolves does not exist yet and the
Job would fail closed at startup. Re-check before assuming:

```bash
kubectl get externalsecrets,secrets -A -o json \
  | grep -c ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY   # 0 => this step is required
```

This step is called out separately, and deliberately left to a human, because it mints a
**root of trust**. Everything else in this document is recoverable by re-running it; a
signing key is not. `ontology_integrity.py` refuses an `env://` reference by design
(D-OB-5), so the key must land in versioned custody — the `@2` suffix pins the exact
version the Job reads, which is what makes a rotation observable instead of silent.

The private key is ed25519, stored as base64 of the 32-byte seed. Generate it **off the
cluster**, on a host you trust, and never let it transit an agent session or a shell
history file:

```bash
umask 077
python3 - <<'EOF' > /dev/shm/ontology-signing-key.json
import base64, json
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
k = Ed25519PrivateKey.generate()
priv = k.private_bytes(serialization.Encoding.Raw,
                       serialization.PrivateFormat.Raw,
                       serialization.NoEncryption())
pub = k.public_key().public_bytes(serialization.Encoding.Raw,
                                  serialization.PublicFormat.Raw)
print(json.dumps({"private": base64.b64encode(priv).decode(),
                  "public":  base64.b64encode(pub).decode()}))
EOF
```

Write it to the `apps/agent-utilities` secret **without disturbing the other keys already
there** (`bao kv patch`, never `put` — `put` replaces the whole secret and would silently
drop every co-resident value):

```bash
P=$(kubectl -n platform get pod -l app=openbao -o name | head -1)
jq -r .private /dev/shm/ontology-signing-key.json \
  | kubectl -n platform exec -i "$P" -- env BAO_ADDR=http://127.0.0.1:8200 \
      BAO_TOKEN="$BAO_ROOT_TOKEN" \
      bao kv patch apps/agent-utilities ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY=-
kubectl -n platform exec "$P" -- env BAO_ADDR=http://127.0.0.1:8200 \
      BAO_TOKEN="$BAO_ROOT_TOKEN" bao kv metadata get -format=json apps/agent-utilities \
  | jq .data.current_version    # must equal the @N pinned in the Job manifest
shred -u /dev/shm/ontology-signing-key.json
```

If `current_version` is not `2`, update the `@N` in
`deploy/release/connector-manifest-signing-job.yaml` to the printed value rather than
writing the key again to force the number — a rewrite to reach a version number produces
two live keys and no way to tell which signed a given manifest.

Finally, publish the **public** half: add it to `DEFAULT_TRUSTED_SIGNERS` in
`agent_utilities/knowledge_graph/ontology/ontology_integrity.py` under the signer id the
Job passes. Until that lands, `verify_release_signature()` returns `False` for everything
this key signs — which is the intended fail-closed default, not a bug, and is exactly what
`UNSIGNED-PREVIEW` relies on to stay unverifiable.

### 1. An OpenBao read-only policy, scoped to exactly one secret

Add to `services/openbao/k8s/bootstrap-policies.sh` (or run once by hand, same
`kubectl exec` pattern that file already uses — never over the network, never with a
static token typed anywhere):

```bash
P=$(kubectl -n platform get pod -l app=openbao -o name | head -1)
run() { kubectl -n platform exec -i "$P" -- env BAO_ADDR=http://127.0.0.1:8200 BAO_TOKEN="$BAO_ROOT_TOKEN" "$@"; }

printf 'path "apps/data/agent-utilities"     { capabilities = ["read"] }\npath "apps/metadata/agent-utilities" { capabilities = ["read","list"] }\n' \
  | run bao policy write agent-utilities-connector-manifest-signer-ro -
```

This is deliberately narrower than the existing `eso-read` policy (`apps/data/*`) — the
signing job only ever needs `apps/data/agent-utilities`, nothing else in the `apps` mount.

### 2. A Vault Kubernetes-auth role bound to the Job's own ServiceAccount

Kubernetes auth is already enabled at `auth/kubernetes/`; the deployed `openbao`
`ClusterSecretStore` proves that path. Create the role
`agent-utilities-connector-manifest-signer` with a ten-minute TTL and only these
bindings: ServiceAccount `connector-manifest-signer`, namespace `release-tooling`, token
audience `openbao`, and policy `agent-utilities-connector-manifest-signer-ro`. The Job
manifest already projects that audience, so the role and workload contract agree.

### 3. Prepare the digest-attested input and public-output boundaries

The signing Job does not read a live checkout, a mutable host mount, or a floating
image. Prepare one operator-owned `connector-manifest-signing-input` PVC containing a
reviewed release input bundle. It contains `release-inputs.sha256` with relative digest
entries, the exact built wheel under `wheels/`, the frozen provider fleet under
`agents/`, and the repository-manager workspace manifest at its normal package path.

The Job verifies the attestation file digest, then verifies every listed input before
copying the fleet into its bounded disposable `/work` staging volume. The operator
substitutes the attestation digest and wheel digest in the Job template; the image is
also required to be pinned as `image@sha256:<digest>`. Create a separate durable
`connector-manifest-signing-public-output` PVC for reviewed public output. The Job
writes only three run-scoped directories there: `connector-bundles/` for public bundles,
`manifests/` for fleet projections, and `native/` for the native projection.

Neither PVC contains signing authority material. The input is read-only to the Job;
all provider writes stay in disposable staging, and a unique run directory prevents a
later attempt from overwriting an earlier output.

### 4. Apply the namespace/ServiceAccount and run the Job

Apply the reviewed namespace and ServiceAccount setup with
`kubectl apply -f deploy/release/connector-manifest-signing-job.yaml`.

Then, for an actual signing run: take the `frozen_sha` the keyless
`connector-manifest-diff` GitHub Actions job reports (`workflow_dispatch` it from the
Actions tab, read the job summary), substitute it plus the frozen commit's built
image digest, input attestation/wheel digests, release timestamp, and unique output run
ID into a COPY of the `Job` in `deploy/release/connector-manifest-signing-job.yaml`
(never re-apply the same Job name twice — give each run a unique name). Apply that copy,
wait up to 30 minutes for its `complete` condition in `release-tooling`, then read the
run-scoped Job logs. Those logs are the JSON report and never contain key material.

The Job's `restartPolicy: Never` / `backoffLimit: 0` mean a failure never silently retries
with stale state — read the printed JSON report for the exact `[freeze|regenerate|verify]`
stage and reason.

### 5. Review before applying

The Job signs. It does not review. Before ever running the signing Job, review the diff
report the keyless `connector-manifest-diff` job (or a local
`regenerate_and_sign_connector_manifests.py` run without `--sign`) produced — GOC-84's own
acceptance gates require a field-by-field reviewed diff, not merely "the gate didn't
complain."

## Why this shape (workload identity, not a materialized Secret)

`release_signer_for_publication()` explicitly refuses an `env://` reference for release
signing (`DURABLE_SECRET_SCHEMES = ("vault://", "secret://")`) — so the design deliberately
does **not** follow the common `ExternalSecret` → k8s `Secret` → `envFrom` pattern this
workspace uses elsewhere (e.g. `inventory/k8s-migration/cutover/apptier/github-runner.yaml`).
Materializing the key into a k8s `Secret` object would (a) require a second custody surface
(the `Secret`) that this workspace's own documented trap warns is silently reverted if
hand-patched via `kubectl patch` rather than written through OpenBao, and (b) persist the
key at rest in etcd for the Secret's lifetime instead of only holding it in memory for the
duration of one `sign()` call. The Job instead authenticates to OpenBao directly with its
own Kubernetes ServiceAccount token (`VAULT_AUTH_METHOD=kubernetes`,
`VaultBackend._try_kubernetes()`), reads the key live via the KV v2 API at sign time, and
never writes it anywhere — matching this lane's own invariant ("KMS/HSM access is
workload-identity-only") and BUG-234's explicit instruction ("via the existing `openbao`
`ClusterSecretStore`" — same OpenBao instance and the same custody discipline, reached here
by direct API call under a scoped role rather than through the ESO CRD, because ESO's job is
materializing a k8s Secret, which is exactly the persistence this design avoids).

## Fail-closed behavior this mechanism enforces (proven, not asserted)

`tests/unit/knowledge_graph/ontology/test_connector_manifest_signing_known_bad.py` proves,
against real signed manifests built with the same generator code this job runs, that each of
the following is refused with a bounded diagnostic — never a silent pass, never a bare crash:

| Adversarial case | Caught by | Diagnostic prefix |
|---|---|---|
| One-bit source change (native connector code diverges post-signing) | `_native_provider_violations` | `[tool-schema] ...differs from its signed code fingerprint` |
| Schema change (a field mapping changes post-signing) | `_check_manifest_bytes` integrity/signature check | `[integrity]` / `[signature]` |
| Alias change (a sync preset's `server` changes post-signing) | `_signature_violations` | `[signature]` |

`scripts/release/regenerate_and_sign_connector_manifests.py` additionally refuses to
proceed (`verify_freeze`) on a dirty working tree, a commit SHA mismatch, or (with
`--require-built-artifact`) an editable/source-tree-only install — the "built-wheel vs
source-tree mismatch" case GOC-84 names explicitly.

## What remains unverified by this lane

* The OpenBao policy and Vault Kubernetes-auth role above are **documented, not applied** —
  this lane never touches the live OpenBao instance. An operator must run the commands in
  §1–§2 before the Job in §3 can authenticate at all.
* `deploy/release/connector-manifest-signing-job.yaml` has not been `kubectl apply`'d or
  run against the live cluster — it is reviewed infrastructure-as-code, not a proven
  deployment.
* GOC-84's own regeneration of the REAL 11 bundled connector manifests has not happened —
  this page only removes the "no custody path exists" blocker; GOC-84's own W01–W05 work
  breakdown (frozen-commit confirmation, field-by-field diff review, the full ten-case
  adversarial matrix against the real bundle) is unstarted and remains that lane's own work,
  gated on GOC-83.
