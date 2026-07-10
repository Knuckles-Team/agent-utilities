# graph-os unified fleet gateway — auth config (DEFAULT reference)

> The standalone `mcp-multiplexer` is **absorbed into graph-os** via the in-process fleet
> loader (`attach_fleet_loader`, `agent_utilities/mcp/multiplexer.py`). There is **one** MCP
> endpoint — `graph-os` — that serves its own KG/engine tools **and** lazily fronts the whole
> authenticated `*-mcp` fleet (`find_tools` / `list_catalog` / `load_tools`). This is the
> canonical auth wiring; validated live 2026-07-05.

## Architecture

```
client ──▶ graph-os (graph-os.arpa, streamable-http :8000)   ← inbound JWT validated
            ├─ always-on: graph-os KG/engine tools
            └─ fleet loader (attach_fleet_loader), reads MCP_CONFIG
                 └─ lazy ▶ <name>-mcp.arpa   ← outbound client-credentials bearer attached
                          (each child is 401 without it; Eunomia authorizes per principal)
```

## Complete env (set on the `graph-os` service; secrets from OpenBao `apps/graph-os`)

| Var | Value / source | Purpose |
|-----|----------------|---------|
| `MCP_CONFIG` | `/root/.config/agent-utilities/mcp_config.json` (the central fleet list — the file `mcp_config_central.json` the multiplexer used, mounted via the XDG config volume) | **The fleet server list.** ⚠️ **Gotcha:** the fleet loader's `_resolve_config_path` checks `~/.gemini/antigravity/mcp_config.json` **first**, so you MUST set `MCP_CONFIG` explicitly or graph-os silently loads a stale default (symptom: `list_catalog` shows a handful of servers, not the fleet; `github-mcp` "not in catalog"). |
| `MCP_CLIENT_AUTH` | `oidc-client-credentials` | Turn on the outbound minter. |
| `OIDC_ISSUER` | `http://keycloak.arpa/realms/homelab` | Token endpoint auto-discovered (OS-5.46 — no `OIDC_TOKEN_URL` needed). |
| `OIDC_CLIENT_ID` | `mcp-multiplexer` (reuses the multiplexer's Keycloak client; a dedicated `graph-os` client is optional) | Client-credentials principal. |
| `OIDC_CLIENT_SECRET` | **OpenBao `apps/graph-os`** (`bao kv get apps/graph-os`) — never committed | Client secret. |
| `OIDC_AUDIENCE` | `agent-services` | Token audience the children validate. |
| `EUNOMIA_TYPE` | `embedded` | In-process PDP over the fleet tool surface (per-principal authorization). |
| `EUNOMIA_POLICY_FILE` | `/eunomia_policy.json` (mounted) | The policy. |
| `AUTH_JWT_*` / `FASTMCP_SERVER_AUTH_JWT_*` | realm `homelab`, audience `agent-services` | **Inbound** — validate the caller's Keycloak JWT (unchanged from before). |
| `ENGINE_MODE` / `ENGINE_ENDPOINT` / `EPISTEMIC_GRAPH_AUTOSTART` | `remote` / `tcp://<engine-host>:9100` / `0` | Split-storage: dial the engine on the fast-NVMe node instead of autostarting a local one. Omit for the co-located default. |

## Secrets & durability

- **Source of truth:** OpenBao `apps/graph-os` (KV v2), mirroring `apps/mcp-multiplexer`. Seed with
  the root token (`services/openbao/.env` → `BAO_ROOT_TOKEN`; the scoped `openbao-mcp` token is
  `apps/data/<its-own>` only and 403s cross-app):
  `curl -H "X-Vault-Token: $BAO_ROOT_TOKEN" -d '{"data":{…}}' $OPENBAO_URL/v1/apps/data/graph-os`.
- **Deploy env:** `services/graph-os/.env` (committed) carries the **non-secret** config; the deploy
  sources it before `docker stack deploy -c compose.dev.yml graph-os`, and injects
  `OIDC_CLIENT_SECRET` from OpenBao `apps/graph-os` (same pattern as the mux). `compose.dev.yml`
  references everything as `${VAR}` so the stack is reproducible.

## Validate (live)

```bash
# From inside the graph-os container: mint a token and hit a protected child — 200/400 (NOT 401)
python3 -c 'from agent_utilities.mcp.client_credentials import child_auth_header; print(bool(child_auth_header({}).get("Authorization")))'
# Session/client side: the fleet is visible and github loads
list_catalog            # → ~58 servers incl github-mcp
load_tools(servers=["github-mcp"])   # → mounted, callable
```

Retire the standalone `mcp-multiplexer` service once graph-os is durable — it is redundant.
