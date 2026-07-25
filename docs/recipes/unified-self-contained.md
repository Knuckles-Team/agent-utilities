# Recipe — Unified self-contained graph-os (both MCP transports)

> **The unified-binary program's W-D workstream:** one `graph-os` process that IS
> everything — MCP tool surface + the engine authority — with **no separate engine
> service to deploy**, served over **either** MCP transport. Today the engine is a
> supervised **local child process** (`epistemic-graph-server`, packaged with the
> wheel); the in-process PyO3 embedding (W-A of the unified-binary program) is the
> same self-contained shape with the socket round-trip removed later, with **no
> change to anything on this page** — the config surface below is transport- and
> embedding-shape-agnostic. Concepts: `CONCEPT:AU-OS.deployment.engine-resolver-auto-provision`
> (the one resolver every entrypoint shares), `CONCEPT:AU-P0-1` (session currency).
> Companion pages: [Tiny recipe](tiny.md) (stdio only), [Supported Configurations
> ladder](../guides/deployment-configurations.md), [Configuration Reference §A.1
> Engine resolution](../architecture/configuration.md#a1-engine-resolution-one-resolver-every-entrypoint-conceptau-osdeploymentengine-resolver-auto-provision).

## TL;DR — one config surface, two `--transport` values

**Self-contained vs. remote is decided ONLY by `GRAPH_SERVICE_ENDPOINTS`.** The
`--transport` flag (`stdio` vs. `streamable-http`) never affects this decision —
it only picks how `graph-os` itself is reached and which process-identity
bootstrap it uses. Verified this session by reading `mcp_server()`
(`agent_utilities/mcp/kg_server.py`) end to end: it calls
`_get_engine()` → `create_backend()` → `GraphComputeEngine.get_or_create()` →
`engine_resolver.resolve_engine()` **identically for both transports**, inside the
same `with use_actor(...), use_session(...):` block, with no transport-conditional
branch anywhere in between. A dedicated test proves this:
`tests/unit/mcp/test_graphos_bootstrap_isolation.py::test_mcp_server_selects_local_engine_path_for_both_transports`.

| | stdio (packaged-local / CLI-embedded) | streamable-http (served / one self-contained pod) |
|---|---|---|
| Invocation | `graph-os --transport stdio` | `graph-os --transport streamable-http --host 0.0.0.0 --port 8004` |
| Engine locality | **Local** iff `GRAPH_SERVICE_ENDPOINTS` is unset (same rule as the other column) | **Local** iff `GRAPH_SERVICE_ENDPOINTS` is unset (same rule as the other column) |
| Process identity | May use the zero-config in-memory bootstrap (`DEPLOYMENT_PROFILE=tiny`, no endpoints, no external identity source) | **Always** an external source: `KG_AUTH_TOKEN_REF` or `KG_IDENTITY_OAUTH2` (the tiny bootstrap is stdio-only) |
| Inbound MCP auth | N/A (no separate clients; stdio has no request-level Authorization header) | `--auth-type`/`AUTH_TYPE` must resolve to a real scheme (`static` is the simplest self-contained choice — no external IdP) |
| Typical shape | One agent tool (Claude Code, opencode, IDE) spawns its own instance | One pod behind a Service/Ingress; scale by adding more self-contained pods (see [Scaling caveat](#scaling-caveat-each-pod-is-independent)) |

## Exact config for the unified/self-contained path

Both invocations below share the SAME engine-locality config. Only the identity
block differs (stdio can use the zero-config bootstrap; streamable-http cannot).

```jsonc
// ~/.config/agent-utilities/config.json (or AGENT_UTILITIES_CONFIG_DIR)
{
  // Absence — not an empty list, not an empty string — is what selects the
  // local engine. If this key is present in config.json AT ALL (even null/"")
  // it still participates in config-projection each load; DELETE the key.
  // "GRAPH_SERVICE_ENDPOINTS": "...",              // <- must not be present

  "DEPLOYMENT_PROFILE": "tiny",                      // required for stdio's zero-config identity bootstrap;
                                                      // irrelevant to engine locality itself (that's GRAPH_SERVICE_ENDPOINTS alone)
  "GRAPH_SERVICE_PERSIST_DIR": "/var/lib/agent-utilities/engine",  // pin the durable snapshot dir (optional; defaults to the XDG data dir)
  "ENGINE_LIFECYCLE": "refcounted",                  // or "persistent" for a served pod that should stay warm
  "ENGINE_IDLE_SHUTDOWN_SECS": 60
}
```

> ⚠️ **Config-precedence trap (validated this session).** `config.json` is
> re-projected into the process environment on every config load
> (`core/config.py` `_commit_xdg_environment_projection`, `_load_xdg_json_config_locked`),
> and it fills any key that ISN'T already an **explicit** process env var. If a
> prior setup ever wrote `graph_service_endpoints` into `config.json` (e.g. this
> host previously pointed at a remote/shared engine), **removing or blanking the
> `GRAPH_SERVICE_ENDPOINTS` environment variable is not enough** — an absent env
> var doesn't block the config.json projection, so the old remote endpoint keeps
> winning. You must delete the key from `config.json` itself (`setup-config` /
> `graph_configure action=set` / hand-editing the file) to actually reach the
> local-engine path.

> ⛔ **Do not set `ENGINE_MODE`, `ENGINE_ENDPOINT`, or `EPISTEMIC_GRAPH_AUTOSTART`.**
> These are **retired** durable configuration keys (`core/config.py`
> `_RETIRED_CONFIGURATION_KEYS`) — if present in `config.json` or the environment
> they raise `ValueError` and **hard-fail boot** before graph-os does anything
> else. `GRAPH_SERVICE_ENDPOINTS` unset/set is the entire, current mechanism.

### stdio — packaged-local, CLI-embedded

```jsonc
// mcp_config.json (Claude Code / any MCP-aware IDE)
{
  "mcpServers": {
    "graph-os": {
      "command": "graph-os",
      "args": ["--transport", "stdio"]
    }
  }
}
```

With `DEPLOYMENT_PROFILE=tiny`, `GRAPH_SERVICE_ENDPOINTS` unset, and neither
`KG_AUTH_TOKEN_REF` nor `KG_IDENTITY_OAUTH2` set, `_mint_process_session("stdio")`
takes the zero-config leg (`local_process_authority_enabled()` →
`mint_local_process_session()`): an asymmetric key signs a short-lived JWT
in-memory as a one-time proof, the key and token are destroyed immediately, and
the resulting process-lifetime `GraphSession` carries no user, host, endpoint, or
proof material. This is the **sole** local bootstrap in the whole codebase — it
requires stdio, `tiny`, and both identity refs unset simultaneously; any one of
those conditions failing falls through to the external-identity path below, never
silently.

### streamable-http — served, one self-contained pod

```bash
graph-os --transport streamable-http --host 0.0.0.0 --port 8004 \
  --auth-type static --static-tokens-ref env://GRAPHOS_STATIC_TOKENS
```

`streamable-http` is a `SERVED_TRANSPORT` (`security/request_identity.py`), so TWO
identity requirements apply that stdio's zero-config path never touches — **neither
one is about engine locality; both are required even though the engine stays
local**:

1. **The process's own bootstrap session** — `_mint_process_session()` never takes
   the local leg for a network transport (`transport == "stdio"` is a hard AND
   condition, not just profile-gated). Configure exactly one of `KG_AUTH_TOKEN_REF`
   (a secret reference resolving to a provisioned JWT) or `KG_IDENTITY_OAUTH2` (a
   client-credentials block). For a genuinely zero-external-IdP pod, the simplest
   choice is a self-issued static token stored in your secret backend and
   referenced by `KG_AUTH_TOKEN_REF` — no Keycloak/OIDC required.
2. **Inbound per-request identity** — `apply_served_security_profile()` refuses to
   start unless `--auth-type`/`AUTH_TYPE` resolves to a real scheme (`static` is the
   simplest self-contained option: `--static-tokens-ref` names a secret holding a
   `{token: {client_id, scopes}}` JSON map; `jwt`/`oidc-proxy`/`oauth-proxy`/
   `remote-oauth` are for an external IdP). `AUTH_JWT_AUDIENCE` (or
   `MCP_JWT_AUDIENCE`) and `KG_POLICY_VERSION` must also be set. Full detail:
   [Secured single node (rung b)](../guides/deployment-configurations.md#rung-b-secured-single-node),
   [identity-jwt example](../examples/identity-jwt.md).

Everything else — `GRAPH_SERVICE_ENDPOINTS` unset, `GRAPH_SERVICE_PERSIST_DIR`,
`ENGINE_LIFECYCLE` — is identical to the stdio column above. `DEPLOYMENT_PROFILE`
does not have to be `tiny` here (it only gates stdio's zero-config identity path);
`single-node-prod` is equally valid for a served self-contained pod. If you do set
`DEPLOYMENT_PROFILE` to anything other than `tiny` (or `APP_PROFILE=production`),
also set `EPISTEMIC_GRAPH_ENCRYPTION_KEY_REF` — `_resolve_engine_encryption_key()`
requires an explicit reference outside the tiny/dev combination rather than
silently persisting a local key file.

## The engine auth `EPISTEMIC_GRAPH_*` inheritance

Whichever transport is serving, when `graph-os` autostarts its local engine child
(`GraphComputeEngine._autostart_engine`) it does NOT blindly forward its own
environment. `_engine_child_environment()` passes through only OS essentials, proxy
settings, and anything already prefixed `EG_`/`EPISTEMIC_GRAPH_`/`GRAPH_SERVICE_`/
`LC_` (stripping the raw encryption key/ref and the auth secret, which are handled
explicitly below) — so unrelated provider API keys never reach the Rust child.
`_autostart_engine()` then:

- **Derives** `EPISTEMIC_GRAPH_AUDIENCE`, `EPISTEMIC_GRAPH_TENANT`, and
  `EPISTEMIC_GRAPH_POLICY_VERSION` from the verified process `GraphSession`'s
  `engine_verified_context()` — the SAME session `_mint_process_session()` produced
  for whichever transport is running. If any of these three happens to already be
  set in the passed-through environment and disagrees with the session, boot fails
  loud (`"local engine authority policy does not match the verified process"`)
  rather than silently picking one.
- **Reuses or mints** `EPISTEMIC_GRAPH_SIGNER_KEYS_JSON`: if already present
  (passed through) it must contain a ≥32-byte key for the CURRENT verified actor
  or boot fails closed; if absent, a per-install signer key is generated once
  (`_load_or_create_engine_bootstrap_signer_key()`) and injected for this one
  bootstrap.
- **Sets** `GRAPH_SERVICE_AUTH_SECRET` to `resolve_engine_auth(config)` — the
  configured secret, or a per-install HMAC secret auto-generated at
  `data_dir()/engine_secret` (mode `0600`) the first time no secret is configured.
- **Sets** `EPISTEMIC_GRAPH_ENCRYPTION_KEY` (never `_REF`) — resolved once, at
  spawn time, from `EPISTEMIC_GRAPH_ENCRYPTION_KEY_REF` if configured, else (tiny +
  non-production only) an auto-generated local private key. The raw key never
  round-trips through `config.json`.

None of this differs by `--transport` — the child sees the exact same
`EPISTEMIC_GRAPH_*` projection whether its parent is a stdio or a streamable-http
`graph-os`, because both mint their bootstrap session through the same
`_mint_process_session()` chokepoint before `_autostart_engine()` ever runs.

## Scaling caveat: each pod is independent

The unified-binary program's target is "the image runs one `graph-os` per pod;
multiple pods for scale." For the **self-contained** shape (this page), that is
literally true and nothing more: each pod's `graph-os` autostarts (or shares, if
already running in that pod) its OWN local engine with its OWN
`GRAPH_SERVICE_PERSIST_DIR`. Two self-contained pods do **not** share a knowledge
graph — there is no coordination between them. If you need N replicas serving the
**same** graph, that is the horizontal-scale-out shape: point every replica's
`GRAPH_SERVICE_ENDPOINTS` at one shared out-of-process engine instead (see
[Split-storage engine](split-storage-engine.md) / [Enterprise](enterprise.md)) —
`GRAPH_SERVICE_ENDPOINTS` is still the ONLY switch between the two shapes.

## Verify

```bash
# Zero-config identity boundary (stdio only)
agent-utilities-doctor --only graph_identity auth

# Confirm the resolved engine target for THIS process/profile:
agent-utilities-doctor --preflight --profile tiny

# stdio, ad hoc:
graph-os --transport stdio &
# streamable-http, ad hoc:
graph-os --transport streamable-http --host 127.0.0.1 --port 8004 \
  --auth-type static --static-tokens-ref env://GRAPHOS_STATIC_TOKENS &
curl -s localhost:8004/health   # liveness always 200; body reports real engine reachability
curl -s localhost:8004/health/ready   # 200/503 mapped onto the SAME report
```

`/health` and `/health/ready` both call the shared
`observability.runtime_health.collect_health()` core, so they report the SAME
truthful engine-reachability detail the REST gateway's `/health` uses — a
self-contained pod that failed to bring its local engine up shows unhealthy in the
body even though the liveness endpoint itself still answers (kubelet must not
crash-loop the pod for a dependency a restart can't fix; `/health/ready` is what
should gate Service routing).

## Wiring proof (this session)

`GraphComputeEngine.get_or_create()` is reached from `mcp_server()` through
`_start_engine_bootstrap()` → `_get_engine()` → `create_backend()` (always the
operational `EpistemicGraphBackend()` — no branch on transport) →
`GraphComputeEngine.__init__` → `engine_resolver.resolve_engine(config, graph_name)`,
which reads only `config.graph_service_endpoints` — never `args.transport`. A new
test exercises the real resolver from inside `mcp_server()`'s actual control flow
(only the deepest socket connect is mocked) and asserts a non-`"remote"`
resolution for **both** `--transport stdio` and `--transport streamable-http` when
`GRAPH_SERVICE_ENDPOINTS` is unset:

```bash
pytest tests/unit/mcp/test_graphos_bootstrap_isolation.py::test_mcp_server_selects_local_engine_path_for_both_transports -q
```
