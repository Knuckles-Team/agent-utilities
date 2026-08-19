# NE-115 real data-preparation acceptance

NE-115 is the bounded cross-repository acceptance contract for the governed
external-data path. It is intentionally opt-in and must run against one real,
isolated epistemic-graph endpoint and one empty tenant. It is not a substitute
for unit coverage and it never uses an in-memory graph, a fake client, a mock
server, or a connector-specific write shortcut.

The acceptance scenario is:

```text
dirty GitLab fixture
  -> ACL admission
  -> bounded Arrow profile
  -> typed CleanPipeline + strict Pydantic model
  -> privacy-safe profile/clean evidence
  -> deterministic Artifact + lineage mapping
  -> native ingest_envelope / SHACL-ICV admission
  -> durable checkpoint
  -> current + historical-LSN native profile
  -> exact zero-delta replay
  -> transport crash/reconnect and durable replay re-read
```

## Explicit prerequisites

The test refuses defaults and checks each source checkout's exact `HEAD` before
opening the graph client. Values must be supplied by the operator or secret
manager; the auth secret is never committed or printed.

```text
NE115_RUN_LIVE_ACCEPTANCE=1
NE115_AU_ROOT=/absolute/path/to/agent-utilities
NE115_AU_REVISION=<exact 40-hex AU HEAD>
NE115_EG_ROOT=/absolute/path/to/epistemic-graph
NE115_EG_REVISION=<exact 40-hex EG HEAD>
NE115_GITLAB_API_ROOT=/absolute/path/to/gitlab-api
NE115_GITLAB_API_REVISION=<exact 40-hex connector HEAD>
NE115_ENGINE_ENDPOINT=unix:///absolute/path/to/isolated.sock
NE115_ENGINE_ISOLATED=true
NE115_ENGINE_TENANT=ne115:<unique-run-id>
NE115_ENGINE_GRAPH=ne115:<unique-run-id>
NE115_ENGINE_AUTH_SECRET=<injected secret>
NE115_ENGINE_PROFILE_TARGET=<native engine row-set/table reference>
NE115_ENGINE_PROFILE_SCHEMA_DIGEST=sha256:<64 lowercase hex>
```

The endpoint must be a single absolute Unix socket and must not contain a
mock/emulator/test-server marker. `NE115_ENGINE_TENANT` and
`NE115_ENGINE_GRAPH` must match the `ne115:` form. The harness reads the graph
before mutation and refuses a non-empty tenant, so a reused or shared tenant is
an unavailable acceptance environment rather than a pass.

The engine must expose the typed `analytics.profile` operation from NE-112's
`ProfileRequest`/`ProfileResult` seam. The harness does not approximate an
engine profile with local Arrow data. A missing operation, wrong result type,
stale schema/LSN, over-limit result, below-threshold top-k value, or async
profile response is a fail-closed unavailable/error outcome.

## Bounded and privacy invariants

- IPC bytes are bounded before decompression; rows, columns, decompressed bytes,
  profile cardinality, top-k, quantiles, warnings, and deadlines are finite.
- Cancellation, malformed IPC, oversized input, lossy casts, missing ACL, secret
  fields, omitted/wrong tenant, and wrong governance-shape digest are rejected
  before native admission.
- Only content-free profile and preparation evidence is attached to the envelope.
  Raw rows, URLs, titles, bearer material, and rejected validation payloads never
  appear in profile/evidence/lineage assertions.
- `ingest_envelope` is the sole durable path. Native SHACL/ICV rejection must
  report no checkpoint advance; successful commit advances once; an exact replay
  returns `status=skipped`, `receipt.replayed=true`, and zero watermark delta.
- The trace records crash-before-replay and crash-after-commit-before-checkpoint
  transport loss. A fresh native client must recover the durable receipt before
  the source checkpoint is advanced.

## Root-owned execution

The delegated agent must not execute this harness. Root runs it once, after
pinning AU/EG/gitlab-api revisions and provisioning the real isolated endpoint:

```bash
pytest -q -m 'live and engine' \
  tests/integration/knowledge_graph/test_data_prep_acceptance.py
```

The normal pytest expression excludes `live`; explicit selection is required.
An unavailable prerequisite is not a successful certification and must be
reported as such. No credentials belong in shell history, source, fixtures, or
logs.
