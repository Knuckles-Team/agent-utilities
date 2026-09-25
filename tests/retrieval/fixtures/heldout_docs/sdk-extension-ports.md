# Extension ports

Five typed protocols keep the SDK open. An implementation ships in its own
distribution and declares an entry point; nothing in the SDK changes.

| Port | Module | Entry-point group | Reference implementation |
|---|---|---|---|
| `SourceAdapter` | `ports.source_adapter` | `agent_connector_sdk.source_adapters` | `mcp_tool` |
| `ArtifactKind` | `ports.artifact_kind` | `agent_connector_sdk.artifact_kinds` | `tools`, `skills`, `prompts`, `resources` |
| `Transport` | `ports.transport` | `agent_connector_sdk.transports` | `mcp` |
| `Sink` | `ports.sink` | `agent_connector_sdk.sinks` | `epistemic_graph` |
| `WriteBackPort` | `ports.writeback` | generated connector binding | `GovernedWriteBack` + fixture transport |

## SourceAdapter

| Method | Contract |
|---|---|
| `describe()` | capabilities, without I/O |
| `discover(session)` | verify the live source contract; required before extraction |
| `extract(session, checkpoint)` | one page and the provider checkpoint that resumes after it |
| `reconcile(session, known_ids)` | ids missing from the source and unknown to the sink |

The `mcp_tool` adapter extracts through a connector's MCP tool as a preset
describes it (pagination, a since-watermark), requires the pinned
`tool_schema_sha256`, and rejects records that do not match the preset. The
watermark advances only when a sweep is exhausted.

| `pagination` | Parameters | Next page |
|---|---|---|
| `none` | | none |
| `cursor` | `cursor_param`; `cursor_path` or `cursor_record_field`; optional `more_path` | the token, until it is absent or repeats |
| `page` | `page_param`, `page_size_param`, `page_size`, `start_page`; `page_kind` `number` (the spelling `page` is rejected) | the next page index, until a page is shorter than `page_size` |
| `offset` | `page_param` (the offset), `page_size_param`, `page_size`; `page_kind` `offset` | the offset plus the records returned, until a page is shorter than `page_size` |

A `page_kind` that does not apply to the mode is rejected. A preset with an
empty `id_field` is rejected: a sweep without record identity, such as a SQL
table sweep, belongs to a data-platform source adapter.

The package validator requires the current fingerprint algorithm and rejects a
`tool_schema_sha256` derived from an empty input schema. The pin binds both
input and output schemas. An action selected by a preset must appear in the
action argument's JSON Schema `enum` (or its single-value `const` form).
Certify from the server's `tools/list` with
[`connector-certify`](connector-certify.md).

## ArtifactKind

Prompt packs read both `prompts/list` and `prompts/get` through the same MCP
session. Each prompt entry retains its listing definition, the argument contract,
and the complete ordered MCP result, including roles, typed multimodal content,
resource references and response metadata. The capture records that it is a
rendered prompt with no supplied arguments; it does not claim to be a template
or a system prompt. Changed message content changes the pack digest.

Pack provisioning has no configured prompt-argument input. A prompt with required
arguments fails closed before retrieval; no values are invented. Optional
arguments are omitted so the server may apply its defaults. Empty, incomplete or
malformed results are rejected. Aggregate prompt capture uses the existing
16 MiB response-byte limit and fails before a pack can be acknowledged.

## Sink

| Method | Contract |
|---|---|
| `submit(batch)` | commit a record batch; the receipt is returned only after commit |
| `source_status(connector, stream)` | read EG's sole durable checkpoint and live-set authority |
| `import_pack(pack)` | import a content pack keyed by its digest |
| `readiness()` | whether this sink can commit right now, without side effects -- returns a `SinkReadiness(ready, reason)`; `reason` is set whenever `ready` is `False` and must never carry a credential or other secret value |

The connector-sync runner's `/health/ready` (see [Connector sync](connector-sync.md)
"Health") calls `readiness()` directly, bounded by a short timeout, so a sink
implementation must answer it without a side effect and should not assume it
is ever skipped. `epistemic_graph` requires a verified client and a live
ConnectorPack authority resolver at construction; the testing kit's
`InMemorySink` always reports ready.

`SourceIngestionRequest` is imported from
`epistemic_graph.generated.source_ingestion`; the SDK does not define an alias,
parallel DTO, or digest. Its generated `canonical_digest()` binds raw records,
provenance, exact per-record mapping references, provider checkpoint, and
expected previous checkpoint. The runner sends it through generated
`send_source_ingest` with a stable idempotency key and advances the checkpoint
only from the matching generated receipt. EG compare-and-swaps the expected
position atomically with the commit.

A manifest mapping reference names one exact mapping:
`manifest:<connector>#schema_mappings/<key>`. The shorter
`manifest:<connector>` form is a convenience only for a manifest containing
exactly one `schema_mappings` entry. Package validation fails closed when the
short form is ambiguous or an explicit fragment names no declared key. Other
reference schemes are rejected at the EG boundary.

Lifecycle mode is explicit on every page. `full` is a non-authoritative load,
`delta` may carry provider-declared withdrawals, and `reconcile` carries the
complete authoritative live-id and relationship sets so EG derives removals
atomically. Empty authoritative full/reconcile commits require a non-empty
descriptor approval and the verified server's `source:reconcile-empty` scope;
the SDK cannot grant that capability. Provider content hashes are optional and
are echoed, never invented. An unchanged empty delta is not submitted as a
fake checkpoint advance.

`SourceIngest` is the one durable operation; SQL batches and pre-mapped change
envelopes are not substitutes. EG resolves the manifest mapping, admits raw
evidence, deduplicates, writes provenance/outbox state, and advances the
checkpoint before returning `SourceIngestionReceipt`. The SDK verifies that the
receipt's batch digest, mode, and accepted checkpoint bind the submitted
request.

## Runner ports

The connector-sync runner adds two runtime ports, described in
[Connector sync](connector-sync.md): `ConnectorRegistry`
(`ports.connector_registry`) and `ChangeSource` (`ports.change_source`). Source
checkpoint and live-set state are read directly from EG's generated
`SourceIngestStatus` contract and are never persisted by the SDK.

## WriteBackPort

`WriteBackPort` is the D18 source-I/O boundary. It reads the current source
version, produces a side-effect-free field diff, rejects an optimistic conflict
before mutation, verifies an exact durable authorization decision, applies under
an idempotency key, and reconciles every possible effect before retry. The only
authorization modes are `proposal_approval`, `standing_policy`, and
`manual_trigger`; a mode or reference supplied by a caller grants nothing by
itself.

```mermaid
sequenceDiagram
    participant EG as EG change set
    participant SDK as GovernedWriteBack
    participant Auth as AuthorizationVerifier
    participant Source as WriteBackTransport
    SDK->>Source: read_current
    SDK->>SDK: compare base version and field scope
    SDK->>Auth: verify exact digest/mode/ref/policy
    SDK->>Source: compare-and-apply(idempotency key)
    alt acknowledgement certain
        Source-->>SDK: source observation
    else possible effect
        Source--xSDK: outcome uncertain
        SDK->>Source: reconcile key + source version
        Source-->>SDK: applied / no effect / still uncertain
    end
    SDK-->>EG: source observations for durable receipts
```

The canonical models come from `epistemic_graph.generated.write_back`.
`SourceChangeSet.canonical_digest()` and `.patch_digest()` implement EG's framed
MessagePack digest contract; the SDK neither reconstructs those schemas nor
implements a parallel digest. EG alone creates and persists `WriteBackReceipt`
and `ReconciliationReceipt`. No live vendor write-back is enabled by the
reference in-memory transport.

## Activation

`load_extension(group, name, policy=...)` loads an extension only when the
activation policy certifies its exact group, name, distribution and version.
There is no permissive default. `sdk_reference_extensions()` certifies the
implementations this SDK version ships. Two distributions declaring the same
name in a group is an error.

## Contract ownership

Source ingestion, ConnectorPack, and WriteBack request/result models come
directly from `epistemic_graph.generated`. The SDK owns only connector authoring,
capture, transport, and extraction shapes; it defines no parallel EG DTO or
digest.
