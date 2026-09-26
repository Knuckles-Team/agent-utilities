# Live workload certification contract

`graphos-certification-load` is the only load command accepted by the production
certification campaign. The campaign requires it to invoke
`scripts.scale.loadgen` with `--engine live`, `--scale 1.0`, the exact campaign
duration, the campaign report path, and the release digest supplied by the
signed release manifest.

The load generator validates the remaining runtime boundary before it opens an
engine client. A live run fails closed unless all of the following are present:

- `KG_DAEMON_ROLE=client` and a non-empty `GRAPH_SERVICE_ENDPOINTS` topology;
- exactly one engine process-identity source (`KG_AUTH_TOKEN_REF`,
  `KG_IDENTITY_OAUTH2`, or `GRAPH_SERVICE_AUTH_SECRET`) and either an authenticated
  engine TLS profile (`ENGINE_TLS_PROFILE_REF` or `ENGINE_TLS_PROFILE`) or the
  complete injected bundle (`ENGINE_CA_BUNDLE` plus `ENGINE_TLS_SERVER_NAME`, with
  `ENGINE_CLIENT_CERT` and `ENGINE_CLIENT_KEY` supplied together when mTLS is used);
- `LOADGEN_TENANT`, `LOADGEN_PRINCIPAL`, and `LOADGEN_AUDIENCE`;
- immutable `LOADGEN_RELEASE_DIGEST`, `LOADGEN_TOPOLOGY_DIGEST`,
  `LOADGEN_IMAGE_DIGEST`, and `LOADGEN_WORKLOAD_CONTRACT_DIGEST` values.
- a tracked source authority: `LOADGEN_SOURCE_REPOSITORY`, its 40-character
  `LOADGEN_SOURCE_REVISION`, the exact
  `LOADGEN_SOURCE_MANIFEST_DIGEST`, and the matching
  `LOADGEN_SOURCE_AUTHORITY_DIGEST`.

The workload-contract digest must equal the SHA-256 digest of the exact
`scripts/scale/workload_contract.yml` bytes loaded by the process. Reports carry
only the release, topology, image, contract, identity, and engine-authority
digests plus the opaque source-authority digest. The certification campaign
preserves those fields in the normalized load-report digest before signing the
existing AU `OperationalEvidence` boundary.

Before any deployment consumes a loadgen manifest, run the packaged source gate
against the canonical checkout. Run it once for every definition that the
deployment can consume (currently `compose.yml` and `k8s/manifests.yaml`),
pairing each `--manifest` with that file's exact digest and using the same
repository/revision pins:

```bash
python -m scripts.scale.loadgen_source_authority \
  --manifest /absolute/path/to/canonical/loadgen/compose.yml \
  --workspace-manifest /absolute/path/to/workspace.yml \
  --repository https://gitlab.example.invalid/homelab/containers/services/loadgen.git \
  --revision <full-40-character-commit> \
  --manifest-digest sha256:<exact-manifest-digest>
```

The gate requires a regular, tracked file, the exact origin URL and commit,
matching bytes, and a `services.items` registration for `loadgen` in the
workspace manifest. It never accepts a workspace snapshot with no Git root or
an unregistered copy. Its JSON output supplies the source-authority digest for
that exact manifest; a digest produced for `compose.yml` must never be reused
for `k8s/manifests.yaml`. The deployment adapter must reject the deployment if
either definition's gate fails. Missing or failed output is a hard preflight failure
and must not be replaced by hand-written environment values.

The CI path is explicit and separate:

```bash
graphos-certification-load --engine mock --scale 0.001 --duration-s 5 \
  --report-json ./mock-report.json
```

Mock reports are marked `runtime.mode=mock`; the production campaign rejects
them regardless of their SLO result. A mock run therefore provides regression
coverage but can never qualify live certification.

The gateway k6 scenario and the AU certification load are separate workloads:
the k6 image must be pinned by digest, while `CERT_LOAD_COMMAND` must run the
AU image carrying `graphos-certification-load`, also pinned by digest. The AU
certification workload must provide the four artifact digests and identity
references through its secret/config boundary; the k6 stack must retain its own
explicit endpoint/token boundary and bind any combined evidence to the same
release/topology pins. A mutable tag, missing live authority, absent credentials,
or a mock/live mode mismatch is a hard failure. This source repository does not
contain cluster credentials or apply deployment manifests; the operator must
complete the real topology rehearsal separately.

## Current GitOps-source disposition

The checked workspace currently has no `services.items` entry for `loadgen`, and
`services/loadgen/` has no `.git` boundary or tracked parent repository. Those
files are therefore an orphan migration snapshot, not an authority. Do not
deploy them, initialize a Git repository in place, or treat a copied digest as
proof.

Remediation is deliberately explicit: create or identify the canonical GitOps
repository, add its exact URL as the `loadgen` service registration in
`workspace.yml`, move the compose/Kubernetes definitions into that repository,
commit the immutable image and command bindings, and run the source gate above
from that checkout. Only after the gate succeeds may its authority digest and
revision be injected into the AU live-certification environment. The k6 gateway
stack must pin its own image; the AU certification image must separately carry
`graphos-certification-load`. Until those steps are complete, production
certification remains fail-closed.

The AU-owned replacement source is `deploy/loadgen/`. It follows the existing
release-template/Kustomize convention and renders both production Compose and
Kubernetes output, plus a separate non-certifying mock overlay, only when the
operator supplies a digest-pinned `repo@sha256:<64hex>` image and the complete
release/topology/contract/source/workload-identity binding. Generate and check
the bundle with `scripts/release/render_loadgen_assets.py` and
`scripts/release/check_loadgen_assets.py`; the resulting
`source-registration.json` is deterministic metadata, not a substitute for the
canonical Git source gate.

Against the current `<workspace-root>/services/loadgen` snapshot the gate
must stop at `source_git_authority_missing`; that is the expected safe result,
not a reason to initialize a repository beside the snapshot.
