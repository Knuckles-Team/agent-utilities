# Exact installed local certification

The exact-local campaign is the release gate for the local GraphOS and
Epistemic Graph integration. It certifies the artifacts that will be deployed;
it does not discover, install, build, or replace them. A passing source test or
a passing editable checkout is not release evidence.

The campaign covers seven program gates:

- **G-08 — native WorkItem lifecycle:** an eight-case native WorkItem lifecycle
  runs against the supplied engine. It proves fairness-group-scoped claim,
  renewable leases, checkpoint fencing, retry backoff, atomic dependency
  release, dead-letter exhaustion, stale-worker rejection after lease reclaim,
  and idempotent terminal replay.
- **G-09 — transactional AgentBus:** a two-case transactional AgentBus campaign
  commits `BusInbox`, the canonical `WorkItem`, `BusDeliveryOutcome`, and
  `MutationOutbox` together. Redelivery after the commit-before-broker-ack crash
  window must resolve to the same opaque identifiers without a second
  transaction or duplicate WorkItem.

- **G-26 — intent surface:** the initial stdio surface is exactly six intent
  verbs and five control tools. The campaign also exercises dynamic expert-tool
  loading and unloading, bound previews, impact and cost metadata, idempotency,
  destructive-operation approval, structured routing rationale, ambiguity,
  prompt-injection denial, and poisoned-feedback denial.
- **G-30 — native optimization:** exactly 13 semantic optimizer families with
  one current spelling each are exercised across all 14 modalities. Avatar
  `tool_policy` artifacts and `compare_tool_use` plans prove comparator-driven
  tool-policy optimization rather than an optimizer alias. Budget enforcement, typed
  modality fixtures, governed executor plans, materialized optimizer artifacts,
  candidate cardinality, evaluated promotion, and per-modality semantic
  non-regression are mandatory. The complete installed distribution closure
  and loaded-module set must contain no duplicate DSPy, DSRs, or LiteLLM
  runtime.
- **G-32 — provider materialization:** clean external XDG roots are used to
  certify distribution ownership, content-addressed generations, atomic
  concurrent activation, deactivation and pruning, tamper detection, link and
  special-file rejection, duplicate-owner rejection, and path-free doctor
  output.
- **G-34 — native A2A:** the exact six-scenario suite certifies atomic creation
  and reconciliation, renewable fenced leases, stale delivery rejection,
  poison and tamper bounds, governed references, crash recovery, durable
  cancellation, and late-completion rejection. Its crash case SIGKILLs a real
  separate executor process without broker cleanup, restarts the exact engine,
  starts a new executor process, and rejects the old executor's late commit.
- **G-35 — permission governance:** an eight-case permission-governance campaign
  bootstraps one signed identity exclusively through referenced signing
  material, then proves closed-world denial across functions, MCP tools,
  ontology markings, governed actions, agent constructors, and graph
  delegation. A foreign signing authority and an invalid configured policy are
  both rejected. The corresponding exact-installed negative source matrix is
  mandatory and may not skip a case.

## Artifact contract

The caller supplies all three exact artifacts and their expected SHA-256
identities, plus a digest-pinned release manifest and an external Ed25519
signing authority:

- a non-editable, materialized `agent-utilities` release interpreter and its
  path-independent installed-tree digest from the release manifest;
- the GraphOS console launcher and its file digest;
- the full Epistemic Graph server executable and its file digest.

The GraphOS launcher shebang must bind to the supplied release interpreter. The
server executable must be the interpreter's packaged sibling, so GraphOS cannot
silently start a different engine. Symlinks, missing executables, digest drift,
editable installs, direct-URL installs, and source-checkout imports fail closed.
There is no artifact discovery or fallback path.

The installed-tree digest is defined by the campaign's
`agent-utilities-installed-release-v2` algorithm. It hashes the normalized
distribution-relative name, byte length, and content SHA-256 of every installed
`agent_utilities` and distribution-metadata file. The campaign is
RECORD-verified: every installed distribution must have a complete hashed
`RECORD`, every recorded file is rehashed, ownership collisions fail, and any
unlisted site-package file fails. A second path-free SBOM digest binds the full
installed distribution closure. The expected values must come from the release
pipeline; calculating both expected and actual identity from the machine under
test would not establish artifact provenance.

The closed release manifest contains only its schema versions, `release_id`, and
SHA-256 bindings for the signed promotion evidence, release specification,
agent-utilities tree, full distribution closure, release interpreter, GraphOS
launcher, Epistemic Graph server, certification harness, and test catalog. Generate
it from verified promotion evidence with the current-only public CLI:

```bash
generate-exact-local-gates-manifest \
  --release-id "${RELEASE_ID}" \
  --spec "${RELEASE_SPEC}" \
  --promotion-evidence "${PROMOTION_EVIDENCE}" \
  --source-root "${AGENT_UTILITIES_SOURCE_ROOT}" \
  --output "${RELEASE_MANIFEST}"
```

`EXACT_LOCAL_EVIDENCE_VERIFIER_COMMAND` must name the deployment-owned verifier
as a JSON argv array. The source root is used only to digest the current harness
and fixed test catalog and never enters the output. The manifest's file digest is
supplied independently to the campaign. The runner stages the
digest-pinned harness and native executables, then rehashes the original
launcher, packaged engine sibling, harness, test catalog, and installed closure
after the campaign. Passing evidence is signed over canonical JSON with a
32-byte Ed25519 private key supplied through the named environment variable;
only the opaque signer ID, public key, and signature enter evidence.

## Run the campaign

Run the gates serially to keep native-engine resource use bounded:

```bash
python scripts/certification/exact_local_gates.py \
  --release-python "${RELEASE_PYTHON}" \
  --release-sha256 "${RELEASE_TREE_SHA256}" \
  --release-id "${RELEASE_ID}" \
  --release-manifest "${RELEASE_MANIFEST}" \
  --release-manifest-sha256 "${RELEASE_MANIFEST_SHA256}" \
  --graphos "${GRAPHOS_LAUNCHER}" \
  --graphos-sha256 "${GRAPHOS_SHA256}" \
  --engine-binary "${EPISTEMIC_GRAPH_SERVER}" \
  --engine-sha256 "${EPISTEMIC_GRAPH_SHA256}" \
  --signer-id "${OPAQUE_SIGNER_ID}" \
  --signing-key-env "EXACT_LOCAL_ED25519_KEY" \
  --output "${NEW_PRIVATE_EVIDENCE_PATH}"
```

This native certification runner supports Linux or WSL. Run its temporary and
evidence directories on a Linux-native filesystem rather than a mounted Windows
filesystem so Unix sockets, modes, owners, hard links, and SIGKILL semantics are
authoritative. The output path must be absolute, have an existing parent owned
by the current user with no group/world permissions, use a bounded portable file
name, and not already exist. Evidence is written through a no-follow directory
descriptor, atomically hard-linked, synced, and retained with owner-only
permissions. It
contains only artifact digests, bounded versions/counts, controlled gate names,
and pass status. Runtime paths, endpoints, opaque record identifiers, authority
material, test data, logs, and personal identifiers are never retained.

The runner creates isolated external configuration, data, cache, state,
workspace, runtime, and temporary roots. It executes copied gate cases with the
explicit release interpreter under isolated-import mode, disables core dumps,
reaps process groups, and deletes runtime state before emitting evidence. A
failed, skipped, missing, or timed-out case fails the campaign.

G-08 and G-09 share one serial engine process so the campaign proves their
transaction boundary while keeping native resource use bounded. G-35 receives
its randomly generated authority only through the environment reference named
in its isolated `AgentConfig` fixture; neither the reference value, resolved
material, runtime paths, nor generated identities enter evidence.

Run the lightweight source guard separately:

```bash
python scripts/check_exact_local_gates_harness.py
```

That guard verifies the fixed gate cardinalities, artifact-binding controls,
privacy controls, documentation, and navigation. It does not execute GraphOS or
the native engine and cannot close a runtime gate. G-08, G-09, G-26, G-30,
G-32, G-34, and G-35 remain open until the exact installed campaign above
produces passing evidence.
