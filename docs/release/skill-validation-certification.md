# Exact skill-validation certification

The release-grade skill campaign is one current, four-stage contract:

1. `graph-os-generate-skill-runtime-profile` deterministically derives the closed
   runtime profile from the AgentConfig file and output path named by deployment
   references.
2. `graph-os-generate-skill-certification` derives a strict deployment document
   from an exact release specification, externally signed promotion evidence, the
   active AgentConfig document, and an external runtime-profile document.
3. `graph-os-certify-skills` creates one verified, in-process HTTPS loopback OIDC
   authority, starts the selected installed `graph-os`, runs the packaged
   readiness probe from the same release, executes all twenty direct/delegated
   cases, and publishes signed validation and lifecycle evidence.
4. `graph-os-verify-skill-certification` independently reopens the referenced
   inputs, verifies both signatures, and cross-binds every release, runtime,
   catalog, process, engine, and validation digest.

No command assumes a release installation directory. The GraphOS executable is
resolved from the deployment-owned JSON argv reference. The running local engine
is discovered only by its unguessable inherited campaign marker, and its open
`/proc` executable is hashed during the campaign. This works with the current
`current/bin` release layout without storing that or any other filesystem location
in durable evidence.

## External input contract

The eight deployment references and two authority controls are first-class,
current-only `AgentConfig` fields. Configure the references in the active XDG
`config.json`; the authority controls have safe self-contained defaults. There
are no alternate field names or untyped call-site-only certification settings:

| AgentConfig key | Typed value | Purpose |
|---|---|---|
| `SKILL_CERT_RUNTIME_CONFIGURATION` | absolute path string | Active XDG AgentConfig input |
| `SKILL_CERT_RUNTIME_PROFILE` | absolute path string | Generated closed runtime profile |
| `SKILL_CERT_RELEASE_SPEC` | absolute path string | Exact-local release specification |
| `SKILL_CERT_PROMOTION_EVIDENCE` | absolute path string | Signed successful promotion evidence |
| `SKILL_CERT_GRAPHOS_ENDPOINT` | loopback HTTP(S) URL | Active GraphOS MCP endpoint |
| `SKILL_CERT_GRAPHOS_COMMAND` | JSON argv array | Exact installed `graph-os` start command |
| `SKILL_VALIDATION_EVIDENCE_SIGNER_COMMAND` | JSON argv array | External evidence signer command |
| `SKILL_VALIDATION_EVIDENCE_VERIFIER_COMMAND` | JSON argv array | External evidence verifier command |
| `SKILL_CERT_IDENTITY_AUTHORITY_MODE` | `ephemeral-https-loopback` | Lifecycle-owned certification authority |
| `SKILL_CERT_IDENTITY_TOKEN_TTL_SECONDS` | integer, 180–3600 | Renewable token lifetime; default `300`, subject to the computed campaign-window floor |

The numeric `180–3600` range is only the authority's structural bound. A generated
deployment also fails closed unless the lifetime covers its maximum case timeout,
the current trace precheck/export/ingestion windows, shutdown grace, and the lease
expiry safety margin. With the current default `120`-second case timeout,
`30`-second trace window, and `30`-second shutdown grace, the computed minimum is
`245` seconds; the default `300` therefore remains valid. The floor is recomputed
when campaign timeouts change rather than being treated as a second fixed setting.

AgentConfig performs structural validation only: paths must be bounded and
absolute, the endpoint must use HTTP or HTTPS with a literal loopback address or
`localhost`, and commands must be bounded JSON argv arrays whose executable is an
absolute path. It does not touch deployment material while parsing configuration.
The `skill_certification` doctor check owns availability proof: each path must
resolve to a bounded, single-link regular file, the configuration and endpoint
must be the active ones, the start executable must resolve to exact `graph-os`,
and all three commands must resolve to executable, non-symlink, non-shell regular
files. The check never returns a path, endpoint, command, credential, or file
content.

Run the isolated readiness check before generating or executing a campaign:

```bash
agent-utilities-doctor --only skill_certification
```

It skips only when all eight references are unset, fails closed for partial, invalid,
or unavailable material, and passes only when the complete deployment boundary is
ready.

The generator consumes four bounded absolute regular files:

- the exact-local release specification;
- its signed successful promotion evidence;
- the exact active XDG `config.json`; and
- a deployment-owned `SkillValidationRuntimeProfile` document.

The profile document is deliberately closed and contains no environment values:

```json
{
  "apiVersion": "graphos.io/v2",
  "kind": "SkillValidationRuntimeProfile",
  "configurationDigest": "sha256:<digest>",
  "modelRegistryDigest": "sha256:<digest>",
  "identityAuthority": {
    "mode": "ephemeral-https-loopback",
    "tokenTtlSeconds": 300,
    "tlsVerificationRequired": true,
    "lifecycleOwned": true,
    "renewableCredentialsRequired": true
  },
  "engineTopology": "local-autostart",
  "observability": "metadata-only",
  "sequential": true
}
```

Do not hand-author or commit this document, or commit a generated deployment for a
particular environment. Keep both in deployment-owned configuration. Generate the
profile through reference names already present in AgentConfig:

```bash
graph-os-generate-skill-runtime-profile \
  --configuration-reference SKILL_CERT_RUNTIME_CONFIGURATION \
  --profile-reference SKILL_CERT_RUNTIME_PROFILE
```

The configuration reference must resolve to the bounded absolute AgentConfig JSON
file; the profile reference resolves to its external JSON destination. The command
publishes deterministic, mode-`0600` JSON and reports no paths or configuration
values. The generated
`SkillValidationDeployment` contains only digests, bounded numeric controls,
booleans, and environment-variable names. It contains no command argv, endpoint,
credential, identity, profile content, certificate material, or filesystem path.

The generator verifies the release evidence signature before producing output. It
also recomputes the release-specification, promotion-evidence, GraphOS,
configuration, and profile digests. It independently attests the installed release
closure and writes these mandatory fields inside the deployment's top-level
`release` object:

```json
{
  "agentUtilitiesSha256": "sha256:<digest>",
  "agentUtilitiesFileCount": 10,
  "distributionClosureSha256": "sha256:<digest>",
  "releasePythonSha256": "sha256:<digest>"
}
```

The count shown is the schema minimum, not a fixed release count. The generator
derives all four values from the verified promotion evidence and installed release;
they are not operator-supplied CLI values. The configuration must describe exactly
one `light` and one `normal` chat model. Both models must:

- use either a literal loopback/private address or an exact private DNS name
  present in `MODEL_HTTP_ALLOWED_PRIVATE_HOSTS`;
- use only HTTP or HTTPS without embedded credentials, query, or fragment; and
- have reference-backed API-key, OAuth2-secret, or header material.

Before any service starts, private DNS must resolve to exactly one loopback,
RFC1918, or IPv6 ULA address. The governed request transport independently
resolves and pins every call, retains logical Host/SNI for hostname verification,
verifies the connected peer, and rejects public or ambiguous rebinding. Public,
unreferenced, missing, duplicate-tier, or ambiguous model entries are rejected.
Only a canonical model-registry digest and aggregate booleans/counts are retained;
model identifiers, DNS names, and addresses are not.

## Generation

All `*Reference` options below are environment-variable names. At certification
time the material references resolve to absolute regular files, the start/signer/
verifier references resolve to bounded JSON argv arrays, and the endpoint reference
resolves to the active loopback GraphOS URL.

```bash
graph-os-generate-skill-certification \
  --release-id release-<id> \
  --release-specification <external-spec> \
  --promotion-evidence <external-promotion-evidence> \
  --runtime-configuration <external-config-json> \
  --runtime-profile <external-profile-json> \
  --output <external-deployment-json> \
  --specification-reference SKILL_CERT_RELEASE_SPEC \
  --promotion-evidence-reference SKILL_CERT_PROMOTION_EVIDENCE \
  --configuration-reference SKILL_CERT_RUNTIME_CONFIGURATION \
  --profile-reference SKILL_CERT_RUNTIME_PROFILE \
  --endpoint-reference SKILL_CERT_GRAPHOS_ENDPOINT \
  --start-command-reference SKILL_CERT_GRAPHOS_COMMAND \
  --signer-command-reference SKILL_VALIDATION_EVIDENCE_SIGNER_COMMAND \
  --verifier-command-reference SKILL_VALIDATION_EVIDENCE_VERIFIER_COMMAND
```

The normative deployment schema is
`deploy/release/skill-validation-deployment.schema.json`.

The campaign does not require an external identity provider. The orchestrator
generates a private CA, loopback server certificate, opaque client credentials,
and asymmetric JWT key for that run. It projects only `env://` secret/profile
references into the exact GraphOS, readiness, and validator child environments;
it neither rewrites AgentConfig nor places runtime endpoints, credentials,
certificate material, or temporary locations in evidence. TLS verification has
no disable switch.

## Readiness and lifecycle execution

`graph-os-skill-readiness` is the packaged probe invoked as a short-lived process
from the exact release by the orchestrator. Do not launch it separately during an
exact campaign: only `graph-os-certify-skills` owns the authority and complete
zero-to-one-to-zero lifecycle. The probe proves:

- the referenced configuration is the active XDG AgentConfig file and still has
  the signed private/local model-registry digest;
- every private-DNS model resolves uniquely inside the private boundary enforced
  by the pinned request transport;
- the endpoint is the active loopback GraphOS endpoint;
- the selected named TLS profile resolves with peer and hostname verification;
- the configured outbound MCP identity mode is complete;
- authenticated GraphOS discovery exposes `graph_orchestrate`, `graph_query`, and
  `graph_jobs`;
- a governed zero-row query succeeds; and
- engine resolution is local, never a configured remote topology.

Run the campaign with adjacent output destinations:

```bash
graph-os-certify-skills \
  --deployment <external-deployment-json> \
  --report <external-report.md> \
  --validation-evidence <external-validation.json> \
  --lifecycle-evidence <external-lifecycle.json>
```

Before service startup, the orchestrator re-verifies the signed promotion evidence
and recomputes the specification, promotion, GraphOS, configuration, profile,
model-registry, Agent Utilities tree, distribution closure, and release-Python
digests plus the Agent Utilities file count. A passing lifecycle proves all of the
following:

- the in-process HTTPS OIDC authority: `0 → 1 → 0`;
- its private CA and server key existed only in private ephemeral work and were
  removed before evidence signing;
- peer and hostname verification succeeded and at least two renewable credentials
  were minted without retaining tokens, identities, endpoints, or certificates;
- all private-DNS models resolved uniquely to private addresses and the request
  transport's public-rebinding guard was active;
- global GraphOS processes: `0 → 1 → 0`;
- marker-qualified candidate GraphOS processes: `0 → 1 → 0`;
- marker-inherited local engine processes: `0 → 1 → 0`;
- terminal native Langfuse MCP child process count: `0`;
- terminal loopback OIDC fixture process count: `0`;
- the running engine executable digest equals the promoted engine digest;
- the candidate and marked engine were stopped and reaped;
- the validator exited zero; and
- `processGate.installedReleaseAttested` is exactly `true`;
- the signed validation evidence contains exactly twenty passing cases and is
  bound by its exact file digest.

The normative lifecycle schema is
`deploy/release/skill-validation-deployment-evidence.schema.json`.

## Validation evidence

The validator receives release, GraphOS, engine, configuration, profile, and model
registry digests only from the already verified deployment. Its signed subject
binds the exact ten-skill catalog, canonical test catalog, all twenty case digests,
direct/delegated execution, configured model class, exact skill body, one
metadata-only Langfuse lookup, and one governed parent-graph trace readback per
case. Passing evidence requires twenty cases and ten complete direct/delegated
pairs.

The normative validation schema is
`deploy/release/prebundled-skill-validation-evidence.schema.json`.

## Independent verification

```bash
graph-os-verify-skill-certification \
  --deployment <external-deployment-json> \
  --validation-evidence <external-validation.json> \
  --lifecycle-evidence <external-lifecycle.json>
```

The verifier does not trust the producer's earlier verification result. It resolves
the verifier JSON argv through
`deployment.validation.verifierCommandReference`, sends canonical signed JSON on
standard input, and requires exactly:

```json
{
  "verified": true,
  "subjectDigest": "sha256:<canonical-unsigned-subject-digest>",
  "keyId": "key:<opaque-key-digest>"
}
```

It then independently checks both JSON schemas, current catalog digests, all
deployment bindings, the four installed-release attestations and file count, the
validation file digest, and the passing lifecycle invariants. A fixed verifier
command, shell command, literal key, unsigned subject, retired v1 document, or
alternate response shape is rejected.

All durable evidence is content-free and path-free. It stores no prompt, model
output, source content, endpoint, credential, identity, raw trace identifier,
profile value, command, or filesystem location.
