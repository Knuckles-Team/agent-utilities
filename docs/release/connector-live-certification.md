# Connector live certification

Connector capability bundles are release inputs, not proof that a connector and
engine work together. The live-certification harness closes that gap without
persisting source records or deployment details.

Each provider's `ontology/certification.json` is an exact schema-v2
`ConnectorSourceAttestation`. It is signed, marked `mode=offline-source`,
`status=source-validated`, and `live_certified=false`. Its checks cover declared
tool fingerprints, generated synthetic-fixture contracts, SHACL parsing, privacy,
and manifest integrity only. It contains no `live_tool_schema` pass state. The
capability-bundle gate rejects the former shape that inferred live or executed
fixture success from a nonempty manifest.

Source-bundle generation derives its exact provider set from the explicit
repository-manager workspace, resolves one signer for the whole run, and stages and
validates every selected provider before publication. An ordinary staging or write
failure leaves provider-owned and bundled artifacts unchanged; successful publication
uses the same validated bytes for both copies.

```sh
python scripts/generate_connector_capability_bundles.py \
  --agents-root "$AGENTS_ROOT" \
  --bundled-output "$BUNDLED_MANIFEST_ROOT" \
  --workspace "${XDG_CONFIG_HOME:-$HOME/.config}/agent-utilities/workspace.yml" \
  --now "$RELEASE_TIMESTAMP" \
  --apply
```

The source signature and external production certification are different evidence
classes. `ConnectorSourceAttestation` proves only offline repository structure.
Production acceptance requires a separate `ConnectorLiveCertification` record with
`mode=external-live`, `status=certified`, `live_certified=true`, all required checks
passed, and a current bundle binding. An ephemeral source key is suitable only for a
disposable local source cycle: the live-certification gate trusts the public key in the
signed manifest, so a releasable bundle and its later live evidence must use the
operator-managed release key rather than a discarded ephemeral key.

For each signed source preset, the harness verifies the real connector's MCP tool
name and structural schema, then sends the connector-owned synthetic fixture through
the real `ChangeEnvelope` ingestion driver in an isolated scope. It proves:

- bounded initial ingest and exact live-count reconciliation;
- identical-envelope replay is an idempotent no-op;
- a new source version updates without changing the live count;
- delete and delete replay preserve the expected live count;
- tenant, ACL, classification, retention, and legal-hold state survive updates and
  remain attached to the tombstone;
- the materialized synthetic objects satisfy the connector's SHACL shapes; and
- cleanup returns the isolated live count to zero.

The signed record contains only aggregate counts, pass/fail states, bundle hashes,
evidence hashes, controlled labels, and release signature material. Endpoint,
credential, runtime-reference, host, user, local-path, source-record, and object-id
values are never written. Failures are signed with an exception class only.

## Offline fixture validation

Offline mode uses the same lifecycle sequence against a bounded in-memory reference
driver. It is useful while authoring a bundle, but the record is deliberately marked
`offline-validated` and `live_certified=false`; the release gate rejects it.

```sh
graph-os-certify-connector \
  --mode offline-fixture \
  --connector-root "$CONNECTOR_ROOT" \
  --output "$CERT_RECORDS/connector.json"
```

Signing still requires the normal external release-key contract:
`ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF` is preferred. A source checkout can invoke
the equivalent `scripts/certify_connectors_live.py` wrapper.

## External live validation

Live mode accepts one argument containing a secret reference. Literal URLs,
credentials, engine addresses, TLS-disable flags, and command lines are not accepted.
The referenced JSON object has this exact deployment-neutral shape:

```json
{
  "driver_command_ref": "secret://connector-certification/driver",
  "connector_runtime_ref": "secret://connector-certification/source",
  "engine_runtime_ref": "secret://connector-certification/engine",
  "tenant": "connector-certification",
  "retention": "certification-ephemeral",
  "tls_profile_ref": "secret://connector-certification/tls"
}
```

The command reference resolves to a JSON argv array. The driver starts or connects to
the real MCP connector, lists tools, and applies/inspects/counts synthetic envelopes on
the configured engine. It resolves the source, engine, and TLS references at runtime;
the harness never resolves or records their contents. The child receives a minimal
environment plus only variables explicitly named by `env://` references. TLS
verification remains enabled and is controlled by the referenced TLS profile.

```sh
graph-os-certify-connector \
  --mode external-live \
  --profile-ref secret://connector-certification/profile \
  --connector-root "$CONNECTOR_ROOT" \
  --output "$CERT_RECORDS/connector.json"
```

The driver is a bounded JSON-stdin/JSON-stdout protocol with four actions:
`list_tools`, `apply`, `inspect`, and `count` are called by the harness (cleanup uses
`apply` with delete envelopes). A non-zero exit, timeout, oversized response, malformed
JSON, schema drift, governance mismatch, replay mismatch, SHACL failure, count mismatch,
or incomplete cleanup produces a signed failed record and a non-zero CLI exit.

## Release gate

Promotion must validate the complete fleet with live evidence:

```sh
python scripts/check_connector_live_certification.py \
  --agents-root "$AGENTS_ROOT" \
  --records-root "$CERT_RECORDS" \
  --require-live
```

The gate derives the certifiable provider set from signed manifests. It rejects a
missing record, offline record, failed/not-run check, invalid signature, altered
record, or evidence bound to older manifest/fixture/shape bytes. Providers without a
source-sync preset are not falsely reported as live-certified.

For the current exact release train, the aggregate signed ledger contains exactly 65
ordered, unique connector entries. The schema enforces the closed-world count and the
release verifier additionally rejects duplicate identities, ordering drift, sentinel
digests, and count mismatches.
