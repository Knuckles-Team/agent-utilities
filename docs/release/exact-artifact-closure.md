# Exact-artifact closure evidence

The local release has one current certification chain for gates G-01, G-02,
G-04, G-05, G-08, G-09, G-14, G-15, G-17, G-26, G-30, G-32, G-34, G-35,
and G-37.
The chain accepts no editable install, artifact discovery, legacy manifest,
fallback binary, skipped case, or mock live evidence.

## Generate the campaign manifest

First promote the exact local release and independently verify its signed
promotion evidence. Set `EXACT_LOCAL_EVIDENCE_VERIFIER_COMMAND` to a JSON argv
array naming the deployment-owned verifier. Then generate the campaign input:

```bash
generate-exact-local-gates-manifest \
  --release-id "${RELEASE_ID}" \
  --spec "${RELEASE_SPEC}" \
  --promotion-evidence "${PROMOTION_EVIDENCE}" \
  --source-root "${AGENT_UTILITIES_SOURCE_ROOT}" \
  --output "${PRIVATE_EVIDENCE_ROOT}/exact-local-gates-manifest.json"
```

The source root is used only to hash the current certification harness and its
fixed test catalog. It is never copied into the manifest. The generator verifies
the promotion signature and release specification, requires successful
zero-process pre/post gates plus passing canary and doctor checks, cross-checks the engine
identity against the native artifact digest, and emits the strict
`deploy/release/exact-local-gates-manifest.schema.json` shape. The output binds
the promotion evidence, release specification, installed Agent Utilities tree,
installed distribution closure, release Python, GraphOS launcher, full engine,
campaign harness, and test catalog.

Run `scripts/certification/exact_local_gates.py` with that manifest and its
independently calculated SHA-256. The campaign evidence now carries the
promotion-evidence and release-specification bindings in addition to the exact
runtime artifacts.

## Produce the exact engine campaign set

Deployment resolves the following explicit values from `AgentConfig` and the
exact release inputs. The producer does not inspect ambient configuration,
discover a binary, build a binary, or fall back to another Python or source
tree.

```bash
python scripts/certification/run_exact_engine_campaigns.py \
  --release-id "${RELEASE_ID}" \
  --engine-binary "${ENGINE_BINARY}" \
  --engine-sha256 "${ENGINE_SHA256}" \
  --campaign-python "${RELEASE_PYTHON}" \
  --campaign-python-sha256 "${RELEASE_PYTHON_SHA256}" \
  --epistemic-graph-root "${EPISTEMIC_GRAPH_SOURCE_ROOT}" \
  --source-freeze-evidence "${SOURCE_FREEZE_EVIDENCE}" \
  --source-freeze-sha256 "${SOURCE_FREEZE_EVIDENCE_SHA256}" \
  --authority-config "${G37_AUTHORITY_CONFIG}" \
  --work-root "${PRIVATE_LINUX_WORK_ROOT}" \
  --output-dir "${PRIVATE_EVIDENCE_ROOT}/engine-campaigns"
```

The work root, authority file, and output parent must be caller-owned private
Linux locations. The output directory must not exist. The producer verifies and
stages the one digest-pinned full engine once, verifies the release Python and
source-frozen Epistemic Graph producer tree, and invokes every child with argv
and a minimal isolated environment. It runs G-37 performance first, followed by
G-02/G-05 fault and restart, G-01/G-04 protocol authorization, G-14 multimodal,
G-15 KnowledgeBatch, and G-17 reasoning and repair. G-14 receives the SHA-256
of the exact `performance.json` bytes produced by G-37.

A successful final directory contains exactly these closure-ready documents:

- `performance.json`
- `fault-restart.json`
- `protocol-authorization.json`
- `multimodal.json`
- `knowledge-batch.json`
- `reasoning-repair.json`

Each document is semantically validated by the closure validators before the
directory is atomically published. Timeouts, non-zero exits, output overflow,
artifact or source mutation, malformed evidence, extra output, and any failed
campaign delete the unpublished staging directory and fail closed. Child output
is drained but not retained. Temporary engine, environment, and Markdown
artifacts are removed; the six aggregate JSON documents contain no local paths,
environment values, payloads, or personal identifiers.

Run the manifest-bound exact-local campaign separately, using the same exact
release engine identity, before binding the complete closure. Then set these two
environment variables to deployment-owned JSON argv arrays:

- `EXACT_ARTIFACT_CLOSURE_SIGNER_COMMAND`
- `EXACT_ARTIFACT_CLOSURE_VERIFIER_COMMAND`

## Bind every current live campaign

The signer and verifier command variables are the only environment-provided
inputs to the binder.

The signer receives bounded canonical unsigned JSON on standard input and
returns `algorithm`, opaque `keyId`, encoded `signature`, and `subjectDigest`.
The verifier receives the complete signed document and must return exactly
`verified=true`, the same `subjectDigest`, and the same opaque `keyId`. Commands,
keys, and credentials never enter arguments or evidence.

```bash
bind-exact-local-release-evidence \
  --release-id "${RELEASE_ID}" \
  --spec "${RELEASE_SPEC}" \
  --promotion-evidence "${PROMOTION_EVIDENCE}" \
  --source-root "${AGENT_UTILITIES_SOURCE_ROOT}" \
  --campaign-manifest "${EXACT_LOCAL_GATES_MANIFEST}" \
  --fault-restart-evidence "${PRIVATE_EVIDENCE_ROOT}/engine-campaigns/fault-restart.json" \
  --protocol-authorization-evidence "${PRIVATE_EVIDENCE_ROOT}/engine-campaigns/protocol-authorization.json" \
  --performance-evidence "${PRIVATE_EVIDENCE_ROOT}/engine-campaigns/performance.json" \
  --multimodal-evidence "${PRIVATE_EVIDENCE_ROOT}/engine-campaigns/multimodal.json" \
  --knowledge-batch-evidence "${PRIVATE_EVIDENCE_ROOT}/engine-campaigns/knowledge-batch.json" \
  --reasoning-repair-evidence "${PRIVATE_EVIDENCE_ROOT}/engine-campaigns/reasoning-repair.json" \
  --exact-local-evidence "${EXACT_LOCAL_EVIDENCE}" \
  --output "${PRIVATE_EVIDENCE_ROOT}/exact-artifact-closure.json"
```

The binder independently re-verifies promotion evidence and regenerates the
expected campaign manifest. It rejects any release, specification, harness,
catalog, Agent Utilities, GraphOS, engine, or campaign digest mismatch. It also
requires the complete 60-case G-02/G-05 crash matrix and both G-05 restart
proofs; the exact G-01/G-04 identity and authorization slice, including denial
across all 14 served data paths and ten enabled wire protocols; 30 G-37
scenario families and all 54 ledger rows; four G-14
modalities, 12 behavior dimensions, 16 fault cases, and the exact G-37 digest;
all seven G-15 KnowledgeBatch families, requirements, and snapshot cases; all
nine G-17 restart, retraction, causal, and fenced-repair cases; and passing
G-08 WorkItem lifecycle, local G-09 inbox-to-WorkItem crash replay, G-26/G-30/
G-32/G-34, and G-35 permission-governance details from cryptographically
verified exact-local evidence. The closure exposes those shared exact-local
bytes as the separately digest-bound `workItemAgentBus`, `exactLocal`, and
`permissionGovernance` campaign summaries.

The resulting document conforms to
`deploy/release/exact-artifact-closure-evidence.schema.json`. It contains only
opaque release identifiers, SHA-256 bindings, fixed aggregate counts, gate
status, and external signature material. It contains no paths, endpoints,
credentials, personal identifiers, payloads, prompts, traces, logs, or raw
campaign results. Both output destinations must be new absolute files under an
existing caller-owned private directory.

Run the source-only guard independently:

```bash
python scripts/check_exact_artifact_closure.py
```

This source guard validates the schemas, exact gate and test inventories,
console entry points, documentation, and current manifest agreement. It does
not execute or replace any exact-artifact campaign.
