# Fleet census — AU-BOUNDARY-R005..R009 — 2026-10-09

Method: for every connector under `agent-packages/agents/*` plus `repository-manager`,
`git fetch origin main`, then `git ls-tree -r --name-only origin/main` to enumerate
PRODUCTION files (tests/, conftest.py excluded), grepped for imports of the banned
modules `agent_utilities.mcp`, `.core.config`, `.core.exceptions`,
`.core.transport_security`, `.base_utilities`. `agent_server.py` (the sanctioned A2A
host exception) is excluded. Hits that are comment/docstring prose (not executable
import statements), or that live only inside code-generator template strings whose
actual generated output is already clean (verified for arr-mcp, jellyfin-mcp,
mattermost-mcp), are not counted as violations. Imports with an inline "SDK gap" /
"intentionally still agent_utilities" rationale comment are sanctioned SDK-gap
imports per the task's exclusion and are not counted as violations either.

Pending-merge PRs noted by the operator (vector-mcp #10, freshrss-agent #3,
genius-agent #6, git-host-api #15) had all already merged into `origin/main` by the
time of this census (2026-10-09, between 20:43–20:44 UTC), so `origin/main` already
reflects their intended post-merge state for those four repos.

## R005 (connector packages A–E) — 0 unsanctioned violators → LANDED

| Connector | Migration PR | Remaining sanctioned SDK-gap imports |
|---|---|---|
| ansible-tower-mcp | n/a | none found |
| archimate-mcp | n/a | none found |
| archivebox-api | n/a | none found |
| aris-mcp | n/a | none found |
| arr-mcp | n/a | `scripts/generate_api.py` template string only (dev codegen tool; generated `*_api.py` output is clean) |
| atlassian-agent | n/a | none found |
| audiobookshelf-mcp | n/a | `audiobookshelf_mcp/auth.py` docstring prose mentioning `agent_utilities.mcp.delegated_auth` / `.core.transport_security` (not an import statement) |
| audio-transcriber | n/a | none found |
| caddy-mcp | n/a | none found |
| camunda-mcp | n/a | none found |
| ciso-assistant-api | n/a | none found |
| clarity-api | n/a | none found |
| container-manager-mcp | n/a | none found |
| data-science-mcp | n/a | `data_science_mcp/mcp_server.py:38` `agent_utilities.mcp.verbose_tools.register_tool_surface` — annotated SDK gap (SDK-CONNECTOR-CONTROL-R009: SDK's `register_tool_surface` drops the per-method verbose tool surface this connector depends on) |
| dockerhub-api | n/a | none found |
| documentdb-mcp | n/a | none found |
| egeria-mcp | n/a | none found |
| emerald-exchange | n/a | none found |
| erpnext-agent | n/a | none found |

## R006 (connector packages F–J) — unsanctioned violators remain → NOT landed

| Connector | Migration PR | Remaining violators |
|---|---|---|
| fan-manager | none open | `fan_manager/kg_control.py:432` `from agent_utilities.core.config import config, setting` — bare, no SDK-gap rationale comment |
| firefly-iii-mcp | n/a | none found |
| freshrss-agent | #3 (merged 2026-10-09 20:43 UTC) | none found post-merge |
| genius-agent | #6 (merged 2026-10-09 20:44 UTC) | none found post-merge |
| github-agent | n/a | none found |
| git-host-api | #15 (merged, scope: kg_ingest.py/auth.py only) | `gitlab_api/instances.py:42` `from agent_utilities.core.config import config`; `gitlab_api/mcp_server.py:1247` `from agent_utilities.mcp.context_helpers import (...)` — both bare, no SDK-gap rationale, not covered by #15's scope |
| gramps-mcp | n/a | none found |
| hdhomerun-mcp | n/a | none found |
| home-assistant-agent | n/a | none found |
| jellyfin-mcp | n/a | `scripts/generate_code.py` template string only (dev codegen tool; generated output is clean) |
| jena-mcp | n/a | none found |

## R007 (connector packages K–O) — 0 unsanctioned violators → LANDED

| Connector | Migration PR | Remaining sanctioned SDK-gap imports |
|---|---|---|
| kafka-mcp | n/a | none found |
| keycloak-agent | n/a | `keycloak_agent/auth.py:16` `agent_utilities.mcp.client_credentials.ClientCredentialsTokenProvider` — annotated SDK gap (SDK's counterpart needs a credential ref + mandatory httpx.Client; out of scope mechanical rename) |
| lakekeeper-mcp | n/a | none found |
| langfuse-agent | n/a | `langfuse_agent/auth.py:18` `agent_utilities.core.config.resolve_langfuse_host` — annotated SDK gap (no SDK equivalent for Langfuse-specific host/transport helpers) |
| leanix-agent | n/a | none found |
| legal-peripherals-mcp | n/a | none found |
| lgtm-mcp | n/a | none found |
| listmonk-api | n/a | none found |
| market-data-mcp | n/a | none found |
| mattermost-mcp | n/a | `generate_mattermost_mcp.py` template string only (dev codegen tool; actual `mattermost_mcp/mcp_server.py` imports only `agent_connector_sdk.*`) |
| mealie-mcp | n/a | none found |
| media-downloader | n/a | none found |
| microsoft-agent | n/a | none found |
| nextcloud-agent | n/a | none found |
| objectstore-mcp | n/a | none found |
| okta-agent | n/a | none found |
| onetrust-api | n/a | none found |
| openbao-mcp | n/a | none found |
| opensearch-mcp | n/a | `opensearch_mcp/auth.py:79` `agent_utilities.mcp.delegated_auth` — annotated SDK gap (same rationale as `agent_server.py`'s sanctioned exception: SDK's `create_mcp_server` deliberately omits delegation middleware; no fleet connector has proven that wiring yet) |
| owncast-agent | n/a | none found |

## R008 (connector packages P–S) — 0 unsanctioned violators → LANDED

| Connector | Migration PR | Remaining sanctioned SDK-gap imports |
|---|---|---|
| paperless-ngx-mcp | n/a | none found |
| plane-agent | n/a | none found |
| portainer-agent | n/a | none found |
| postiz-agent | n/a | none found |
| pulselink-mcp | n/a | none found |
| qbittorrent-agent | n/a | none found |
| repository-manager | n/a | `repository_manager/repository_manager.py:51` `agent_utilities.base_utilities.get_library_file_path` — annotated SDK gap ("no agent-connector-sdk equivalent") |
| rom-manager | n/a | `rom_manager/romm/auth.py:60` `agent_utilities.mcp.delegated_auth` — annotated SDK gap (same delegation-middleware rationale as opensearch-mcp/agent_server.py) |
| salesforce-agent | n/a | none found |
| scholarx | n/a | none found |
| searxng-mcp | n/a | `searxng_mcp/apps.py` docstring prose mentioning `agent_utilities.mcp.tools.mcp_apps` (not an import statement) |
| servicenow-api | n/a | none found |
| spark-mcp | n/a | none found |
| sql-mcp | n/a | none found |
| stirlingpdf-agent | n/a | none found |
| systems-manager | n/a | none found |

## R009 (connector packages T–Z) — unsanctioned violators remain → NOT landed

| Connector | Migration PR | Remaining violators |
|---|---|---|
| technitium-dns-mcp | n/a | none found |
| tunnel-manager | n/a | `scripts/check_env_var_drift.py:29` `from agent_utilities.mcp import check_env_var_drift` — dev-tooling CI gate script (sys.path-hacks to a sibling checkout at dev time), not shipped package code; no production-package violation |
| twenty-mcp | n/a | `twenty_mcp/mcp/mcp_graphql.py:75` `from agent_utilities.mcp.context_helpers import (...)` — bare, no SDK-gap rationale comment |
| uptime-kuma-agent | n/a | none found |
| vaultwarden-mcp | n/a | `vaultwarden_mcp/api/api_client_base.py:27` `from agent_utilities.core.transport_security import ResolvedTLSProfile` — bare, no SDK-gap rationale comment |
| vector-mcp | #10 (merged 2026-10-09 20:43 UTC, agent_server.py off core.config) | none found post-merge |
| wger-agent | n/a | none found |
| world-reference-mcp | n/a | none found |

## Summary

- **R005 (A–E): LANDED** — 0 unsanctioned violators.
- **R006 (F–J): not landed** — fan-manager, git-host-api carry bare, unannotated `agent_utilities.core.config` / `.mcp.context_helpers` imports in production modules.
- **R007 (K–O): LANDED** — 0 unsanctioned violators (keycloak-agent, langfuse-agent, opensearch-mcp carry annotated SDK-gap imports only).
- **R008 (P–S): LANDED** — 0 unsanctioned violators (repository-manager, rom-manager carry annotated SDK-gap imports only).
- **R009 (T–Z): not landed** — twenty-mcp, vaultwarden-mcp carry bare, unannotated `agent_utilities.mcp.context_helpers` / `.core.transport_security` imports in production modules.
