# Deployment and configuration

GraphOS ships a local MCP server, an authenticated HTTP transport, configuration
and diagnostic tools, and guarded production operations. Choose a profile,
validate it, and run the release canary before changing live traffic.

## Install

GraphOS supports Python 3.12 through 3.14.

```bash
python -m pip install graph-os
```

Add the optional Agent WebUI host integration when the same process will serve
the UI:

```bash
python -m pip install "graph-os[webui]"
```

## Generate configuration

The configuration tool supports three deployment profiles:

| Profile | Intended use |
|---|---|
| `tiny` | Local evaluation and a minimal single-process environment |
| `single-node-prod` | A production host with externalized persistence and identity |
| `enterprise` | A managed multi-service deployment with production policy controls |

Generate and validate a profile:

```bash
setup-config generate --profile tiny
setup-config doctor --profile tiny
```

The default output follows the XDG configuration directories. Generated
configuration must contain secret references such as `vault://` or
`engine://__secrets__`; do not place plaintext credentials in configuration,
examples, command history, or source control.

Inspect the complete option reference without writing configuration:

```bash
setup-config reference
```

## Local MCP transport

Use `stdio` when an MCP client launches GraphOS as a child process:

```bash
graph-os --transport stdio
```

The helper can register that portable launcher with Codex:

```bash
setup-config codex
```

For other clients, configure `graph-os --transport stdio` using the client's
standard MCP server configuration format.

## Network transport

Serve the streamable HTTP transport on a private listener with a real token
verifier. This example uses the JWT boundary:

```bash
AUTH_JWT_AUDIENCE=graph-os KG_POLICY_VERSION=baseline-v1 \
  graph-os --transport streamable-http \
  --host 127.0.0.1 --port 8000 \
  --auth-type jwt \
  --token-jwks-uri https://identity.example/.well-known/jwks.json \
  --token-issuer https://identity.example/ \
  --token-audience graph-os
```

Network serving is a security boundary. Configure validated identity, tenant
isolation, TLS, and the deployment's authorization policy before exposing the
listener beyond loopback. GraphOS refuses network serving without a constructed
authentication provider and rejects required authority that is absent or
invalid.

## Validate a candidate

Run diagnostics against the resolved deployment configuration, then execute the
bounded release canary in the candidate environment:

```bash
agent-utilities-doctor
graph-os-release-canary
```

The canary reports aggregate readiness checks and does not print paths,
credentials, identities, or graph contents. A failed canary is a release
failure, not a warning to suppress.

The host daemon can be inspected independently:

```bash
graph-os-daemon --status
```

## Production operations

`graph-os-production-ops` provides guarded backup and restore-validation
commands. They require the deployment's workload identity, graph coordinator,
encryption, and mounted storage policy. Review the command help in the exact
candidate environment before use:

```bash
graph-os-production-ops --help
```

Production operations emit opaque digests and aggregate counts rather than
endpoints, principals, storage paths, credentials, or graph contents. Backup
and restore validation are operational changes; run them through the normal
change-control and recovery procedure for the target deployment.

## Release model

The release workflow builds the wheel after the quality and scanner jobs pass.
PyPI publication runs only for an explicit version tag and uses the protected
`pypi-publish` environment. A push to `main` updates source and documentation;
it does not publish a package.

See [Capability status](status.md) before deployment. Capability-gated paths
remain unavailable whenever their required authority is absent or unverified.
