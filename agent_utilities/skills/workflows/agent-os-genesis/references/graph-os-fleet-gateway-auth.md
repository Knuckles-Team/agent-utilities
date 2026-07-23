# graph-os unified fleet gateway — auth config (DEFAULT reference)

> The standalone `mcp-multiplexer` is **absorbed into graph-os** via the in-process fleet
> loader (`attach_fleet_loader`, `agent_utilities/mcp/multiplexer.py`). There is **one** MCP
> endpoint — `graph-os` — that serves its own KG/engine tools **and** lazily fronts the whole
> authenticated `*-mcp` fleet (`find_tools` / `list_catalog` / `load_tools`). This is the
> canonical auth wiring; multiplexer-absorb validated live 2026-07-05, config contract
> modernized 2026-07-20.

## Architecture

```
client ──▶ graph-os (graph-os.arpa, streamable-http :8000)   ← inbound JWT validated
            ├─ always-on: graph-os KG/engine tools
            ├─ engine process identity ▶ epistemic-graph engine (GRAPH_SERVICE_ENDPOINTS)
            │        ← graph-os authenticates ITSELF (KG_IDENTITY_OAUTH2 | KG_AUTH_TOKEN_REF)
            └─ fleet loader (attach_fleet_loader), reads MCP_CONFIG
                 └─ lazy ▶ <name>-mcp.arpa   ← outbound client-credentials bearer attached
                          (each child is 401 without it; Eunomia authorizes per principal)
```

**Two distinct OAuth2 identities** (both client-credentials to Keycloak — do not conflate):
1. **Outbound fleet-child minter** — the bearer graph-os attaches to each `*-mcp` child
   (`MCP_CLIENT_AUTH` + `OIDC_*`).
2. **Engine process identity** — how graph-os authenticates ITSELF to the epistemic-graph
   engine reached over `GRAPH_SERVICE_ENDPOINTS` (`KG_IDENTITY_OAUTH2` **xor**
   `KG_AUTH_TOKEN_REF`). A failure here **blocks boot** with `Graph process identity
   acquisition failed`.

## Complete env (set on the `graph-os` service; secret **values** live in OpenBao `apps/graph-os`, referenced by `*_REF`)

> **Durable-secret policy** (`config.py` `_validate_durable_xdg_secret_policy`). Under the
> XDG/production posture every credential-suffix key must be a **reference** (`*_REF` →
> `vault://…` / `env://…`), never an inline value — a bare `OIDC_CLIENT_SECRET` fails
> validation. Use `OIDC_CLIENT_SECRET_REF`.

| Var | Value / source | Purpose |
|-----|----------------|---------|
| `MCP_CONFIG` | `/root/.config/agent-utilities/mcp_config.json` (the central fleet list — the file `mcp_config_central.json` the multiplexer used, mounted via the XDG config volume) | **The fleet server list.** ⚠️ **Gotcha:** the fleet loader's `_resolve_config_path` checks `~/.gemini/antigravity/mcp_config.json` **first**, so you MUST set `MCP_CONFIG` explicitly or graph-os silently loads a stale default (symptom: `list_catalog` shows a handful of servers, not the fleet; `github-mcp` "not in catalog"). |
| `MCP_CLIENT_AUTH` | `oidc-client-credentials` | Turn on the **outbound fleet-child minter**. |
| `OIDC_ISSUER` | `https://keycloak.arpa/realms/homelab` | Token endpoint auto-discovered (OS-5.46 — no `OIDC_TOKEN_URL` needed **on a flat network**). **HTTPS**: the OAuth2 client rejects a plaintext `http` token endpoint, so the host must trust the issuer CA — see **Host CA trust** below. ⚠️ **On Kubernetes, pin `OIDC_TOKEN_URL` explicitly** (next row) — auto-discovery routes the mint through the edge, which can 502 mid-migration. |
| `OIDC_TOKEN_URL` | *(flat network: omit)* — **on k8s: `https://<idp-service>.<idp-ns>.svc:8443/realms/homelab/protocol/openid-connect/token`** | **The fleet-wide-401 fix.** Auto-discovery dials the *public* issuer host, which routes through the **edge** — during a progressive migration the edge can **502 the token endpoint**, the mint **silently returns no header**, and **every child 401s** (`Session terminated` / `no such host`). Pin the mint to the **in-cluster IdP Service** to take the edge out of the auth path — same resilience trick as pinning JWKS for *inbound* validation. The IdP stamps the external issuer claim regardless of which endpoint minted the token, so child issuer-validation still matches. |
| `OIDC_CLIENT_ID` | `mcp-multiplexer` (reuses the multiplexer's Keycloak client; a dedicated `graph-os` client is optional) | Client-credentials principal. |
| `OIDC_CLIENT_SECRET_REF` | `vault://apps/graph-os/OIDC_CLIENT_SECRET` (or `env://<VAR>`) — **never the inline secret** | Client-secret **reference** (durable-secret policy). |
| `OIDC_AUDIENCE` | `agent-services` | Token audience the children validate. |
| `EUNOMIA_TYPE` | `embedded` | In-process PDP over the fleet tool surface (per-principal authorization). |
| `EUNOMIA_POLICY_FILE` | `/eunomia_policy.json` (mounted) | The policy. |
| `AUTH_JWT_*` / `FASTMCP_SERVER_AUTH_JWT_*` | realm `homelab`, audience `agent-services` | **Inbound** — validate the caller's Keycloak JWT (unchanged from before). |
| `GRAPH_SERVICE_ENDPOINTS` | `tcp://<engine-host>:9100` (e.g. `tcp://10.0.0.10:9100`); `unix://…` for a local socket | **The engine connection.** Replaces the **retired** `ENGINE_MODE` / `ENGINE_ENDPOINT` / `GRAPH_SERVICE_TCP_ADDR` / `EPISTEMIC_GRAPH_AUTOSTART` (all four now hard-fail boot as retired keys). Connect-only: a configured endpoint is dialed, never autostarted (the split-storage default — engine on the fast-NVMe node). Omit entirely for a co-located autostart. |
| `KG_IDENTITY_OAUTH2` **xor** `KG_AUTH_TOKEN_REF` | OAuth2 form: `{"token_url":"https://keycloak.arpa/realms/homelab/protocol/openid-connect/token","client_id":"mcp-multiplexer","client_secret":"vault://apps/graph-os/OIDC_CLIENT_SECRET","audience":"agent-services"}` — **or** a static bearer `KG_AUTH_TOKEN_REF=vault://…` | **The engine _process_ identity** (how graph-os authenticates itself to the engine over `GRAPH_SERVICE_ENDPOINTS`, distinct from the outbound minter). **Configure exactly ONE** — `request_identity.py` raises *"Configure exactly one graph process identity source: KG_AUTH_TOKEN_REF or KG_IDENTITY_OAUTH2"* if both or neither is set. The OAuth2 form does a client-credentials grant to Keycloak, so it **also** needs host CA trust (below). |

## Host CA trust (REQUIRED when the IdP is HTTPS behind a private CA) — the #1 boot blocker

Both OAuth2 mints (outbound minter **and** engine process identity) POST to
`https://keycloak.arpa/.../token`. On the homelab that endpoint serves a **cert-manager cert
issued by the private `homelab-arpa-ca` ClusterIssuer**, so the **graph-os host must trust
`homelab-arpa-ca`** — otherwise every mint fails and graph-os **never finishes booting**:

```
RuntimeError: Graph process identity acquisition failed        # engine identity mint
# …or a fleet-wide child 401 when only the outbound minter is affected
```

- **Proper fix — bake the CA into the host trust store (Step 1b `ca-trust-provisioner`).** This
  is what makes the default TLS context (and `certifi`) trust it; it is a **root** step that
  Step 0f/1b automate for every host:
  ```bash
  sudo cp homelab-arpa-ca.crt /usr/local/share/ca-certificates/homelab-arpa-ca.crt
  sudo update-ca-certificates   # → folds into /etc/ssl/certs/ca-certificates.crt (what certifi resolves to on Debian)
  ```
  Verified this session: with `homelab-arpa-ca` in the bundle the mint returns **HTTP 200** and
  graph-os boots; without it, the mint raises.
- **Env-var CA injection does NOT reach the mint.** `SSL_CERT_FILE` / `TLS_CA_BUNDLE` /
  `CA_BUNDLE` / `REQUESTS_CA_BUNDLE` do **not** apply to the resolved `oauth2-token` TLS
  profile's `ssl_context`. And `certifi.where()` here resolves to the **root-owned**
  `/etc/ssl/certs/ca-certificates.crt`, so appending to certifi as a non-root user fails.
  Host baking (or the config-native profile below) is the only reliable path.
- **No-root alternative — a config-native TLS profile.** Point the `oauth2-token` profile at a
  bundle that includes the CA: set `oauth2_token_tls_profile` (or a named `TLS_PROFILES`
  catalog entry) with `ca_bundle_path: <certifi+CA combined bundle>` and `system_trust: false`
  so the mint verifies against it exclusively. Prefer host baking; use this only where root is
  unavailable.
- **The IdP must also serve a REAL cert.** ingress-nginx's *default* backend answers unmatched
  `.arpa` TLS with a **fake self-signed cert**, so a host that already trusts `homelab-arpa-ca`
  STILL fails. Every `.arpa` ingress that terminates TLS needs its own cert-manager
  `Certificate` + `tls:` block (ClusterIssuer `homelab-arpa-ca`) — see
  [`keycloak-realm-consolidation.md`](keycloak-realm-consolidation.md).

## Secrets & durability

- **Source of truth:** OpenBao `apps/graph-os` (KV v2), mirroring `apps/mcp-multiplexer`. Seed with
  the root token (`services/openbao/.env` → `BAO_ROOT_TOKEN`; the scoped `openbao-mcp` token is
  `apps/data/<its-own>` only and 403s cross-app):
  `curl -H "X-Vault-Token: $BAO_ROOT_TOKEN" -d '{"data":{…}}' $OPENBAO_URL/v1/apps/data/graph-os`.
- **Deploy env:** `services/graph-os/.env` (committed) carries the **non-secret** config; the
  deploy sources it, and references the client secret as
  `OIDC_CLIENT_SECRET_REF=vault://apps/graph-os/OIDC_CLIENT_SECRET` (the durable-secret policy
  forbids an inline `OIDC_CLIENT_SECRET`). `compose.dev.yml` / k8s manifests reference everything
  as `${VAR}` so the stack is reproducible.

## Validate (live)

```bash
# From inside the graph-os container:
# 1) Engine process identity mints (the boot-blocker path) — prints a token length, does NOT raise:
python3 -c 'from agent_utilities.core.config import config; from agent_utilities.security.oauth_client_credentials import build_provider_from_config; print(len(build_provider_from_config(config.kg_identity_oauth2).get_token()))'
# 2) Outbound fleet-child bearer is attached (200/400 from a protected child, NOT 401):
python3 -c 'from agent_utilities.mcp.client_credentials import child_auth_header; print(bool(child_auth_header({}).get("Authorization")))'
# Session/client side: the fleet is visible and github loads
list_catalog            # → ~58 servers incl github-mcp
load_tools(servers=["github-mcp"])   # → mounted, callable
```

Retire the standalone `mcp-multiplexer` service once graph-os is durable — it is redundant.

## Migrating this onto a new orchestrator

When moving the gateway + fleet to another orchestrator (**e.g.** k8s), the outbound-mint
`OIDC_TOKEN_URL` pinning above is one of several cutover-hardening rules — see the
migrate-mode runbook [`orchestrator-migration-cutover.md`](orchestrator-migration-cutover.md)
(and `docs/architecture/orchestrator-migration-cutover.md` for the full rationale).
