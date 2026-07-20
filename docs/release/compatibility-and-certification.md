# Exact release train and production certification

Production promotion binds code, configuration, catalogs and operational proof into one
exact signed release. `compatibility-matrix.yml` pins one canonical spelling of every
component version; minimum ranges, normalized local-version aliases, and different
application/protocol versions are rejected from the serving plane.

Matrix version 2 also binds every OCI component to the same digest-pinned Python 3.12
slim base and names its offline, hash-locked wheelhouse target. The image targets reject
source distributions and install only from staged wheels plus a `--require-hashes`
requirements file; ordinary published-package targets remain separate and cannot be
substituted into exact release assembly.

## Exact release contents

The manifest contains exactly eight components:

1. Epistemic Operations protocol catalog and all twelve schema versions;
2. Epistemic Graph OCI image with verified context, MutationBatch, ChangeEnvelope,
   WorkItem, MultiRaft, RLS, analytics worker RPC and queue metrics;
3. Agent Utilities OCI image with GraphSession, mandatory ContextCompiler,
   intent-mode GraphOS, governed connectors, the standalone analytics worker,
   native Langfuse MCP discovery and metadata-only opaque trace export;
4. Langfuse Agent OCI image with provider-profile transport, verified private-CA
   TLS, metadata-only trace query, and parent-authority graph ingestion;
5. the complete exact 65-package connector capability catalog;
6. the exact ten consolidated pre-bundled skills;
7. the canonical ontology lock;
8. the one-time persisted-state migration catalog.

The three runtime components are independently materialized OCI subjects: Epistemic
Graph uses its `release-local` target, Agent Utilities uses its `agent-local` target,
and Langfuse Agent uses its MCP-only `mcp-local` target. Their OCI root descriptor digests
must be pairwise distinct. Reusing one unified image under three component names fails
closed, even when that image happens to contain all three Python distributions. Each
image is built from the minimal closed wheelhouse reachable from that component's named
root and selected extras.

Every component supplies an exact version, artifact kind, digest, source digest, SBOM
digest, provenance digest, capability set and external signature bundle reference.
All eight component source documents must bind the snapshot and evidence digests
recomputed from one retained canonical source-freeze record. The assembly and manifest
retain that record exactly once as `sourceFreezeEvidence`, bind its byte digest as
`sourceFreezeEvidenceDigest`, validate its ordered G-01 through G-39 results against the
packaged gate manifest, and compare every component to that independent authority.
Repeated component claims without the record, or components produced from a different
freeze, are rejected before the release manifest can be signed.
OCI artifacts use digest references; catalogs use opaque `catalog:<name>@sha256:<digest>`
references. A verifier command is injected by environment as JSON argv and must bind the
verified subject to the declared digest. Inline signing keys are forbidden.

The manifest retains only release-relative evidence references. The gate opens each
bounded, unaliased regular file without following symlinks, recomputes every digest,
semantically validates the component source, CycloneDX 1.5/1.6 SBOM, exact SLSA
provenance/v1 statement, and external signature-bundle schemas, and enforces their
cross-document bindings. For an OCI component, the artifact digest is the verified OCI
root descriptor identity and the SBOM wheel inventory must be non-empty. The opened
signature bundle is passed to the external component verifier with the exact canonical
composite subject bytes used by the signer; a verifier response must repeat both that
subject digest and the artifact digest. Absolute paths, traversal (including `.` and
`..` segments), URLs, aliases, hardlinks, and missing or oversized evidence are
rejected. The complete manifest is itself externally signed; its signature binds the
canonical unsigned manifest and is independently verified before certification.

An SBOM is accepted only when `metadata.component` identifies the exact release
component. Its name and version must match the release declaration, its `bom-ref` and
PURL must be the canonical package URL (`pkg:pypi` for OCI application components and
`pkg:generic` for catalogs), and its sole SHA-256 root hash must equal the immutable
artifact digest. Empty or unrelated CycloneDX documents therefore cannot satisfy the
evidence gate even when their file digest is declared correctly.

The release manifest also carries an exact `certificationDigests` map for the
signed connector live-certification ledger, the 20-case pre-bundled skill
validation matrix, its exact `SkillValidationDeployment`, the signed
skill-validation lifecycle evidence, the signed exact-artifact closure, and the
aggregate-only OCI vulnerability-scan evidence. All six non-sentinel digests are
copied into operational evidence. The deployment
document binds the verifier selector and local model-registry policy used by the
campaign. The lifecycle document is a separate required authority: it binds the
exact validation-evidence digest and proves the candidate process lifecycle for
the same release. The closure independently binds the
exact engine to the fault/restart, protocol authorization, multimodal,
KnowledgeBatch, reasoning/repair, performance, and exact-local campaigns, so neither
a component catalog digest nor prose can stand in for live acceptance.
The OCI scan document binds the independently verified archive and root digests for
all three runtime images to signed Trivy 0.72.0 and fresh vulnerability/Java database
attestations; HIGH or CRITICAL findings, online fallback, or retained raw output fail
closed.

`exactGateEvidence` is the exhaustive machine-readable authority map for every
source-freeze row classified as `exact-artifact`. Each entry contains one
or more `{authority, digest}` records pointing directly to an exact component or one
of the six certification documents. Assembly derives this map from the opened
component and certification bytes; callers cannot supply or normalize it. The
compatibility gate reconstructs all 30 entries and rejects a missing gate, added
authority, reordered authority, or digest substitution before release signing.
G-22 and G-38 include the OCI vulnerability-scan certification as an explicit
authority in addition to their component authorities.

Before closure binding, run the six non-local engine campaigns with the
canonical `scripts/certification/run_exact_engine_campaigns.py` producer. It
takes only explicit `AgentConfig`-resolved and exact-release inputs, verifies
one full engine digest, one release-Python digest, and the source-frozen
Epistemic Graph producer tree, then runs G-37, G-02/G-05, G-01/G-04, G-14, G-15,
and G-17 serially. The new private output directory contains exactly
`performance.json`, `fault-restart.json`, `protocol-authorization.json`,
`multimodal.json`, `knowledge-batch.json`, and `reasoning-repair.json`; G-14 is
bound to the digest of those exact G-37 bytes. A failed or mutated campaign
publishes nothing. The producer retains no child output, local paths,
environment values, or identifying data. See
[exact-artifact closure evidence](exact-artifact-closure.md) for the complete
command and binder inputs.

## Deterministic assembly

Generate the connector catalog only after all 65 workspace-authoritative provider
bundles pass their signed manifest, certification, artifact-ledger, SHACL, mapping,
fixture, ontology-lock, and persistence-privacy checks. The retained catalog contains
only neutral connector names, aggregate counts, and content digests; endpoints,
credentials, signer identities, and filesystem locations are excluded.

```sh
python scripts/release/generate_connector_bundle_catalog.py --check
```

Generate the prebundled-skill catalog from the exact ten installed skill trees. Runtime
skill-validation evidence uses the digest of these same canonical catalog bytes, so a
skill-body change cannot reuse evidence from another release.

```sh
python scripts/release/generate_prebundled_skill_catalog.py --check
```

Generate and check the executable one-time migration catalog from the runtime registry:

```sh
python scripts/release/generate_index_migration_catalog.py \
  --output deploy/release/index-migrations.catalog.json --check
```

After every connector has a current externally live signed record, assemble and sign
the aggregate ledger. The signer and verifier variables contain JSON argv arrays; key
material never enters arguments, output, or source control. The current release ledger
is closed-world: it contains exactly the same 65 ordered, unique provider connectors as
the compatibility matrix. A structurally valid but partial ledger cannot be signed into
a release.

```sh
python scripts/release/connector_ledger.py assemble \
  --agents-root "$AGENTS_ROOT" --records-root "$CERT_RECORDS" \
  --output "$RELEASE_ROOT/evidence/connector-ledger.json" \
  --signer-env CONNECTOR_LEDGER_SIGNER_COMMAND \
  --verifier-env CONNECTOR_LEDGER_VERIFIER_COMMAND

python scripts/release/connector_ledger.py verify \
  --agents-root "$AGENTS_ROOT" --records-root "$CERT_RECORDS" \
  --ledger "$RELEASE_ROOT/evidence/connector-ledger.json"
```

`ReleaseAssembly` is generated deterministically from the opaque release id, the exact
eight component declarations, and release-relative references for the canonical source
freeze, configuration, migration, and certification evidence. The generator opens every referenced artifact,
requires the component set from the compatibility matrix, and computes every digest;
operators do not hand-author the assembly or supply SBOM, provenance, source,
certification, or signature-bundle digests.

First generate the exact typed configuration and one-time migration plan. These
documents bind the release id, current matrix, enabled evolution features, mandatory
security posture, and exact index-migration catalog; an empty or hand-normalized JSON
object cannot satisfy assembly.

```sh
generate-graphos-release-input configuration \
  --release-id "$RELEASE_ID" \
  --matrix deploy/release/compatibility-matrix.yml \
  --output "$RELEASE_ROOT/evidence/configuration.json"

generate-graphos-release-input migration-plan \
  --release-id "$RELEASE_ID" \
  --matrix deploy/release/compatibility-matrix.yml \
  --index-migration-catalog deploy/release/index-migrations.catalog.json \
  --output "$RELEASE_ROOT/evidence/migration-plan.json"
```

```sh
generate-graphos-release-assembly \
  --release-id "$RELEASE_ID" \
  --matrix deploy/release/compatibility-matrix.yml \
  --source-freeze-evidence "evidence/source-freeze.json" \
  --configuration "evidence/configuration.json" \
  --migration-plan "evidence/migration-plan.json" \
  --connector-ledger "evidence/connector-ledger.json" \
  --skill-validation-matrix "evidence/skill-validation.json" \
  --skill-validation-deployment "evidence/skill-deployment.json" \
  --skill-validation-lifecycle-evidence "evidence/skill-lifecycle.json" \
  --exact-artifact-closure "evidence/exact-artifact-closure.json" \
  --oci-vulnerability-scan "evidence/oci-vulnerability-scan.json" \
  --component "epistemic-graph=$RELEASE_ROOT/evidence/epistemic-graph.json" \
  --component "agent-utilities=$RELEASE_ROOT/evidence/agent-utilities.json" \
  --component "langfuse-agent=$RELEASE_ROOT/evidence/langfuse-agent.json" \
  --component "epistemic-operations-protocol=$RELEASE_ROOT/evidence/epistemic-operations-protocol.json" \
  --component "connector-bundles=$RELEASE_ROOT/evidence/connector-bundles.json" \
  --component "prebundled-skills=$RELEASE_ROOT/evidence/prebundled-skills.json" \
  --component "ontology-lock=$RELEASE_ROOT/evidence/ontology-lock.json" \
  --component "index-migrations=$RELEASE_ROOT/evidence/index-migrations.json" \
  --output "$RELEASE_ROOT/assembly.json"

assemble-graphos-release assemble \
  --assembly "$RELEASE_ROOT/assembly.json" \
  --matrix deploy/release/compatibility-matrix.yml \
  --output "$RELEASE_ROOT/release.unsigned.json"

assemble-graphos-release sign \
  --input "$RELEASE_ROOT/release.unsigned.json" \
  --matrix deploy/release/compatibility-matrix.yml \
  --output "$RELEASE_ROOT/release.json" \
  --signer-env RELEASE_MANIFEST_SIGNER_COMMAND \
  --verifier-env RELEASE_MANIFEST_VERIFIER_COMMAND
```

The unsigned and signed files share one evidence root so their relative references are
stable. Assembly emits a schema-conformant `unsigned-local-binder`; this lets an
operator validate the complete local topology and evidence bindings before any release
signature or live connector ledger is available. An unsigned or placeholder connector
ledger is authoring input only. After external live certification, replace it with the
signed ledger and re-run assembly so its digest is exact. The sign operation rejects a
missing, unsigned, malformed, unbound, or non-65-entry connector ledger, changes
`manifestState` to `signed-release`, and signs that exact state. Only the signed state
can pass promotion.

The structural gate can be run without signature verification for authoring, but its
output is never deployable evidence:

```sh
check-graphos-compatibility \
  --manifest "$RELEASE_ROOT/release.json" --structure-only
```

Structure-only still opens and validates every evidence file and signature binding; it
only suppresses invocation of external verifiers. Rendering and certification always
invoke those verifiers.

Connector bundle promotion additionally requires the complete signed
[connector live-certification](connector-live-certification.md) ledger. Offline fixture
records cannot satisfy this gate: every connector with a signed source preset must have
an `external-live` record for the current manifest, fixture, and SHACL hashes, with all
lifecycle/governance/schema/count checks passed.

## 24–72 hour campaign

`certification-campaign.yml` fixes scale at 1.0 and the default duration at 24 hours
(operators may extend it to 72). It drives the live engine and captures aggregate raw
metrics every 15 seconds with at least 95% coverage. There is no mock or skip branch.

The scenario set first proves verified GraphOS identity, workload mTLS, stale-policy
rejection and an opaque content-free Langfuse trace. It then applies faults at all
commit phases plus worker, Raft leader, broker leader, node and zone loss; broker
rebalance; online reshard; an atomic exact-release cutover; one-time index and ontology
migrations;
backup/restore; regional recovery; and policy/deletion propagation.
Each hook must attest `faultApplied=true`, return a non-sentinel action digest, and prove
every scenario invariant with a separate observation digest, measured RPO and measured
recovery time.

`graphos-certification-fault` is the installed standard runtime adapter. Its only
configuration authority is the typed `AgentConfig` contract. For a scenario named
`node-loss`, it selects that key from `CERT_FAULT_ACTION_COMMANDS` and
`CERT_FAULT_PROBE_COMMANDS`; both settings are JSON objects that map every exact
campaign scenario identifier to a JSON argv array. The action consumes the request on
stdin and returns `applied=true` plus its required invariant names. The probe is polled
until all aggregate invariant booleans are true or the timeout expires. Raw
action/probe output is discarded after hashing.

Configure the campaign through `AgentConfig` (normally its XDG configuration and
runtime-secret sources). The current-only fields are:

| Purpose | `AgentConfig` alias and value contract |
|---|---|
| activation and inputs | `CERTIFICATION_MODE=production`, absolute `CERT_RELEASE_MANIFEST`, absolute `CERT_ARTIFACTS_DIR`, and a non-identifying `CERT_HARDWARE_CLASS` such as `capacity-standard` or `tier-large` |
| workload and telemetry | `CERT_LOAD_COMMAND` and `CERT_METRICS_COMMAND`, each a bounded JSON argv array with an absolute executable; the load command must include `{report_file}` |
| scenario orchestration | `CERT_HOOK_COMMANDS`, a JSON object mapping the exact 15 scenario identifiers to bounded JSON argv arrays |
| real fault implementation | `CERT_FAULT_ACTION_COMMANDS` and `CERT_FAULT_PROBE_COMMANDS`, each an exact 15-entry scenario-to-argv JSON object |
| evidence authority | `CERT_EVIDENCE_SIGNER_COMMAND` and `CERT_EVIDENCE_VERIFIER_COMMAND`, each a bounded JSON argv array with an absolute executable |
| aggregate metrics | HTTPS `CERT_PROMETHEUS_URL` and the dedicated TLS selector fields `CERT_PROMETHEUS_TLS_PROFILE` and/or runtime reference `CERT_PROMETHEUS_TLS_PROFILE_REF` |
| optional Prometheus authentication | `CERT_PROMETHEUS_BEARER_TOKEN_REF`, an `env://`, `secret://`, or `vault://` runtime secret reference |

The bearer credential itself is never an `AgentConfig` value. A raw token, token-file
setting, or filesystem path to token material is not supported; only the runtime secret
reference is persisted. All command settings are argv, never shell text, and all three
scenario maps must contain the exact canonical scenario set.

Preflight the complete local authority without starting the 24-hour run:

```sh
agent-utilities-doctor --only production_certification
```

The doctor does not treat a readable JSON file as release proof. It verifies the
configured manifest as a `signed-release` against the packaged compatibility matrix
with independent signature verification enabled. Invalid, unsigned, differently
matrix-bound, or unverifiable manifests fail with redacted diagnostics before TLS or
campaign execution is attempted.

For a direct campaign invocation, `--release` and `--artifacts-dir` must name the same
files selected by `CERT_RELEASE_MANIFEST` and `CERT_ARTIFACTS_DIR`. The canonical
campaign and compatibility matrix are package resources when their optional flags are
omitted.

```sh
graphos-certification-campaign \
  --release <signed-release.json> \
  --artifacts-dir <empty-private-artifact-directory>
```

An explicit `--campaign` may only extend `durationSeconds` within the supported
24–72 hour range. Every other field is compared to the packaged policy byte-for-byte
after canonicalization: scenario order, fault phases, invariants, metric interval,
coverage threshold and SLO/RPO/RTO bounds cannot be overridden.

The opt-in pytest entry is
`tests/scale/soak/test_production_certification.py`. Selecting it without the complete
typed configuration fails immediately; it never reports a skip.

## Evidence and promotion

The deterministic evaluator normalizes the load report to fixed counts, rates,
latency percentiles, SLO booleans and invariant-violation counts; entity-level findings
are discarded and the on-disk report is replaced with that aggregate form. It combines
the normalized report digest, raw aggregate metric digest,
sample coverage, scenario action/observation/metric digests, invariants and observed
RPO/RTO. A pass additionally requires actual monotonic elapsed time and the normalized
load report's real duration to reach the configured duration; 95% metric coverage is
not accepted as a substitute for a full 24-hour run. Gateway error ratio and recent pod
restarts are required metrics with canonical bounds, alongside latency, queue, lag,
membership, WAL-drop and checkpoint-age signals. It signs failure evidence as well as passing evidence, then immediately invokes
an independent verifier.

Every certification adapter uses a shared combined stdout/stderr byte ceiling and a
fixed timeout; overflow and timeout failures expose only stable error categories.
Prometheus responses are streamed through a separate byte ceiling before JSON decoding,
and response bodies, adapter output and endpoint details never enter evidence or errors.

Operational evidence rejects endpoints, filesystem locations, host/user/email fields,
direct identifiers and unsigned/sentinel digests. It retains only release/component/
configuration/certification digests, controlled scenario labels, a non-identifying capacity class,
aggregate counts and external signature material. Passing evidence is impossible if any fault was not applied,
any invariant failed, the SLO evaluator failed, or RPO/RTO exceeded target.

Promotion requires a passing signed evidence document for the exact release digest.
Evidence from another build, configuration, ontology, skill/connector catalog or index
migration is not transferable.

The skill catalog also requires fresh, signed JSON runtime validation evidence for the
exact release. The release gate reconstructs the current test and case catalogs from
the installed runtime-validation contract and requires their exact digests. It then
requires the exact ordered set of all 20 checked-in cases (direct and
GraphOS-delegated for each of the ten skills), the exact skill/mode/model-class/case
digest for each case, 20/20 passing cases, and 10/10 fully passing skills. Every check
must pass (with delegation marked not-applicable only for direct cases), references
must be opaque, Langfuse lookup must be exact-name and metadata-only with one match,
parent-KG readback must have one match, and every privacy flag must be false. Its
signature subject binds the canonical unsigned evidence; release specification,
promotion, GraphOS, and engine digests; runtime configuration, profile, and local
model-registry digests; and the prebundled-skill component digest. A prose matrix or an otherwise valid
signature over different totals cannot satisfy promotion. Generate the evidence with
`agent-utilities-validate-skills --mode all` against the deployed
candidate, retain its content-safe digest in the release evidence, and reject reports
from another skill catalog or runtime configuration. Delegated cases invoke the
execute-only `graph_orchestrate` contract without an `action` argument and poll the
returned run handle exclusively through
`graph_jobs(action="status", job_id=run_id)`. The handle uses the sole current
128-bit opaque format. Trace proof is an exact-name Langfuse MCP lookup whose
metadata must bind that run to the exact configured model, model class, skill, and
skill-body digest; unfiltered project-wide trace baselines are not certification
evidence. Each matched trace must also resolve to exactly one parent-mediated KG
`Trace` node under verified `kg:write` authority.

The signed matrix alone is insufficient for promotion. The mandatory
`skillValidationDeployment` and `skillValidationLifecycleEvidence`
certifications must bind that exact 20-case document digest, the current release
specification and promotion evidence, and the exact configuration, profile,
GraphOS, engine, and local model-registry authorities. The deployment document
selects the external verifier by environment-reference name; release verification
loads that exact manifest-bound document and never substitutes a hardcoded
verifier. The lifecycle must independently prove the global GraphOS,
candidate-GraphOS, and candidate-engine process counts of `0 → 1 → 0`, including
the exact engine executable digest and successful reaping. The release verifier
checks both closed v2 schemas, the matrix and lifecycle signatures,
cross-document digests, passing result, and privacy assertions before release
signing. G-07, G-12, G-18, G-27, G-29, G-30, G-36, and G-38 all carry the exact
lifecycle digest, so omitting either required input or substituting material from
another validation run invalidates release assembly.
