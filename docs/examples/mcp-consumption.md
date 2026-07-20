# Worked Example: Consuming GraphOS over MCP

GraphOS is the sole MCP process for both graph operations and progressive
discovery of configured `*-mcp` connector tools. Clients either launch the
installed command over stdio or connect to one deployed streamable-HTTP URL.

## Local stdio

```json
{
  "mcpServers": {
    "graph-os": {
      "command": "graph-os",
      "args": ["--transport", "stdio"]
    }
  }
}
```

The installed command must already be present in the environment. Runtime
launch does not resolve dependencies or rebuild the package.
For the zero-infrastructure form, generate `DEPLOYMENT_PROFILE=tiny` and leave
`GRAPH_SERVICE_ENDPOINTS`, `KG_AUTH_TOKEN_REF`, and `KG_IDENTITY_OAUTH2` unset.
Run `agent-utilities-doctor --only graph_identity auth`, then launch the entry
above. GraphOS validates a neutral, short-lived bootstrap JWT signed with an
in-memory key and discards both key and token before returning the session; it
persists no personal, host, endpoint, or filesystem identity.

Every network transport, non-tiny profile, explicit engine endpoint, or configured
external identity instead requires exactly one of `KG_AUTH_TOKEN_REF` and
`KG_IDENTITY_OAUTH2`, its JWT validation policy, and successful secret resolution.
An invalid configured source never falls back locally. The launcher stays
machine-neutral and contains no endpoint, path, certificate, or credential.

## Remote streamable HTTP

```json
{
  "mcpServers": {
    "graph-os": {
      "url": "https://graph-os.example.test/mcp"
    }
  }
}
```

Network listeners require the configured JWT/OIDC policy. Keep credentials in
the secrets backend and reference them through AgentConfig; never place them in
the client file.

## Fleet discovery

GraphOS resolves `MCP_CONFIG` from AgentConfig/runtime setup as a separate
child-server fleet catalog. Do not point it at this client launcher or persist
its machine location in the launcher or a report. The fleet catalog contains
only child `*-mcp` definitions; it never contains GraphOS itself. The resident
surface stays bounded: use `find_tools` or `list_catalog`, mount the selected tool
with `load_tools`, call it, and optionally release it with `unload_tools`.
Per-child timeouts, concurrency, session pools, restart supervision, and circuit
breakers are enforced inside GraphOS.

```json
{"tool":"find_tools","arguments":{"query":"inspect a repository"}}
```

No second aggregation service is launched or deployed.
