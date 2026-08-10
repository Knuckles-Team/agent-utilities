# Design Document: One service identity for the multiplexer's outbound child auth, three interchangeable schemes

CONCEPT:AU-OS.identity.so-jwt-protected-children
CONCEPT:AU-OS.identity.rotating-file-bearer

> `agent_utilities/mcp/client_credentials.py`; `docs/architecture/mcp_auth.md`
> "Multiplexer outbound authentication"

## Decision — `MCP_CLIENT_AUTH` selects ONE of three outbound credential schemes, applied uniformly to every child that doesn't declare its own header

The MCP multiplexer aggregates many child MCP servers. When a child enforces
auth (JWT bearer, or a reverse proxy demanding HTTP Basic) the multiplexer
must present a valid credential or its calls are rejected (401). Children are
configured per-entry in `mcp_config.json` and historically carried no
`Authorization` header, so flipping a child to enforce auth made it
unreachable through the multiplexer. `client_credentials.py` gives the
multiplexer ONE service identity, in one of three schemes, selected by
`MCP_CLIENT_AUTH`:

- `oidc-client-credentials` — mint an OIDC provider service token via the
  OAuth2 `client_credentials` grant, cache it, refresh it before expiry.
- `basic` — attach a static `Authorization: Basic <base64(user:pass)>`.
- `rotating-file-bearer` (`CONCEPT:AU-OS.identity.rotating-file-bearer`,
  BUG-051) — re-read `Authorization: Bearer <token>` from a local file on
  EVERY outbound request, instead of once at connect time. For a remote MCP
  server whose bearer is minted by an OUT-OF-PROCESS refresh daemon (e.g.
  `services/graphos-token-refresh/refresh-graphos-token.sh`, a cron-driven
  client-credentials mint this process has no control over), a header baked
  in once at connect/construction time goes stale the moment the daemon
  rotates the file — the connection keeps presenting the OLD token until
  something reconnects it. Reading the file fresh on every request instead
  renews the credential IN-BAND, inside the same long-lived session, with no
  reconnect: the daemon's next scheduled write is picked up by the very next
  call. This generalizes the mechanism `graphos-codex-bridge.py` already
  proved for Codex's stdio bridge (`RotatingBearerAuth`) into a first-class,
  tested primitive any agent-utilities-built MCP client can select via
  config. The file must hold exactly the bearer token (mode 0600); anything
  else fails closed rather than falling back to an unauthenticated request.

Either way, the credential is never applied over a child's explicit
`Authorization` header. Once a service-auth mode is selected, incomplete
configuration or a mint failure aborts the outbound request — it never
silently becomes anonymous. Unset (`MCP_CLIENT_AUTH=none`, the default) makes
every helper an inert no-op.

**The rejected alternative** for `rotating-file-bearer` specifically was
reconnecting the child session whenever the daemon rotates the token (e.g. a
file-watch that tears down and rebuilds the connection). That would interrupt
any in-flight call on that child session for every rotation — reading the
file fresh per-request achieves the same freshness with zero connection
churn, at the cost of one extra file read per outbound call.

## Risk Assessment

- **Blast Radius**: `agent_utilities/mcp/client_credentials.py`; every
  outbound multiplexer child connection when `MCP_CLIENT_AUTH` is set.
- **Backward Compatible**: Yes — `MCP_CLIENT_AUTH=none` (default) is an inert
  no-op; existing unauthenticated children are unaffected.
- **Known weak point**: `rotating-file-bearer` adds a file read to every
  outbound request — acceptable overhead for the multiplexer's call volume,
  but a design that would not scale to a much higher-QPS caller without
  adding an in-process TTL cache in front of the file read.
