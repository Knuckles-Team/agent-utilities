# API Reference

> **GENERATED — do not edit by hand.** Run `python scripts/generate_openapi.py --write`. Source: the served app's own `app.openapi()`, projected from `graph_os_webui.server.create_agent_web_app` exactly as production builds it — not hand-copied.

This is the generated reference for **GraphOS** (OpenAPI 3.1.0) — 172 paths, 200 operations, 200 with a summary. The running server's own interactive `/docs` (Swagger UI) and `/redoc` pages reflect the identical document live; this page is the static, diffable snapshot checked into docs so it cannot silently drift from the code that generates it.

## Coverage note

**This reference does not yet cover the whole live API.** 8 route(s) across 7 mounted path prefix(es) below are served by the app but carry no OpenAPI schema, so `app.openapi()` — and therefore this page — cannot see them:

`/api/chats`, `/api/configure`, `/api/health`, `/api/healthz`, `/health`, `/health/ready`, `/healthz`

Those surfaces (the canonical Knowledge Graph REST/graph-execution routes, among others) are mounted as raw Starlette routes rather than typed FastAPI routers, so they carry no request/response schema for FastAPI to introspect — not an omission from this generator. This count is recomputed from the live app every time `python scripts/generate_openapi.py --write` runs (see `schemaless_routes()` in `scripts/generate_openapi.py`), so as typed schemas land for those routes this note shrinks and eventually disappears on its own, with no route list to hand-maintain here.

The full spec (not just this rendering of it) is published at [`openapi.json`](openapi.json).

## Operations by path prefix

| Path prefix | Operations |
|---|---|
| `/api/apps` | 9 |
| `/api/chats` | 5 |
| `/api/contact` | 1 |
| `/api/enhanced` | 184 |
| `/api/rum` | 1 |

## All operations

| Method | Path | Summary |
|---|---|---|
| GET | `/api/apps` | App Catalog |
| GET | `/api/apps/markets/chart` | Chart |
| GET | `/api/apps/markets/listings` | List Listings |
| GET | `/api/apps/markets/macro-events` | Macro |
| GET | `/api/apps/markets/scanner` | Scanner |
| POST | `/api/apps/markets/shares` | Share |
| GET | `/api/apps/markets/shares/{lease_id}` | Shared |
| POST | `/api/apps/markets/shares/{lease_id}/revoke` | Revoke |
| GET | `/api/apps/markets/status` | Status |
| GET | `/api/chats` | List Chats |
| POST | `/api/chats` | Save Chat |
| DELETE | `/api/chats/{chat_id}` | Delete Chat |
| GET | `/api/chats/{chat_id}` | Get Chat |
| PUT | `/api/chats/{chat_id}` | Update Chat |
| POST | `/api/contact` | Submit Contact |
| GET | `/api/enhanced/agent-icon` | Get Agent Icon |
| POST | `/api/enhanced/agent-library/a2a` | Register A2A Agent |
| GET | `/api/enhanced/agent-library/agents` | List Library Agents |
| POST | `/api/enhanced/agent-library/agents` | Create Library Agent |
| DELETE | `/api/enhanced/agent-library/agents/{agent_id}` | Archive Library Agent |
| GET | `/api/enhanced/agent-library/agents/{agent_id}` | Get Library Agent |
| PUT | `/api/enhanced/agent-library/agents/{agent_id}` | Update Library Agent |
| GET | `/api/enhanced/agent-library/config-summary` | Agent Config Summary |
| GET | `/api/enhanced/agent-library/suggestions` | Suggest Library Agents |
| GET | `/api/enhanced/agent-library/tools` | List Library Tools |
| GET | `/api/enhanced/agents` | List Agents |
| GET | `/api/enhanced/atlas/sources` | Atlas Source Catalog |
| GET | `/api/enhanced/atlas/sources/connections` | Atlas Source Connections |
| POST | `/api/enhanced/atlas/sources/connections` | Atlas Source Connection Create |
| DELETE | `/api/enhanced/atlas/sources/connections/{connection_id}` | Atlas Source Connection Delete |
| POST | `/api/enhanced/atlas/sources/connections/{connection_id}/discover` | Atlas Source Connection Discover |
| GET | `/api/enhanced/atlas/sources/connections/{connection_id}/mapping` | Atlas Source Connection Mapping |
| POST | `/api/enhanced/atlas/sources/connections/{connection_id}/mapping/approve` | Atlas Source Connection Mapping Approve |
| POST | `/api/enhanced/atlas/sources/connections/{connection_id}/mapping/propose` | Atlas Source Connection Mapping Propose |
| GET | `/api/enhanced/atlas/sources/connections/{connection_id}/status` | Atlas Source Connection Status |
| GET | `/api/enhanced/atlas/sources/providers` | Atlas Source Providers |
| GET | `/api/enhanced/atlas/sources/runs/{run_id}` | Atlas Source Run Status |
| POST | `/api/enhanced/atlas/sources/runs/{run_id}/cancel` | Atlas Source Run Cancel |
| POST | `/api/enhanced/atlas/sources/sync` | Atlas Source Sync |
| POST | `/api/enhanced/atlas/sources/sync/preview` | Atlas Source Sync Preview |
| GET | `/api/enhanced/atlas/sources/{source_id}/connection` | Atlas Source Connection Status Ui |
| POST | `/api/enhanced/atlas/sources/{source_id}/connection` | Atlas Source Connection Connect |
| POST | `/api/enhanced/atlas/sync/preview` | Atlas Source Sync Preview Ui |
| GET | `/api/enhanced/atlas/sync/runs` | Atlas Source Sync Runs |
| POST | `/api/enhanced/atlas/sync/runs` | Atlas Source Sync Start |
| POST | `/api/enhanced/atlas/sync/runs/{run_id}/cancel` | Atlas Source Sync Cancel |
| GET | `/api/enhanced/code/instances` | Code Instances |
| POST | `/api/enhanced/code/nav` | Code Nav |
| GET | `/api/enhanced/commands/autocomplete` | Autocomplete Slash Command |
| POST | `/api/enhanced/commands/execute` | Execute Slash Command |
| GET | `/api/enhanced/config` | Get Config File |
| PUT | `/api/enhanced/config` | Update Config File |
| GET | `/api/enhanced/config-files` | List Config Files |
| GET | `/api/enhanced/config/backend` | Get Backend Config |
| PUT | `/api/enhanced/config/backend` | Update Backend Config |
| GET | `/api/enhanced/config/groups` | Get Config Field Groups |
| GET | `/api/enhanced/config/schema` | Get Agent Config Schema |
| GET | `/api/enhanced/config/secret-status` | Get Config Secret Status |
| GET | `/api/enhanced/container-manager/containers` | List Docker Containers |
| GET | `/api/enhanced/cron/calendar` | Get Cron Calendar |
| GET | `/api/enhanced/cron/logs` | Get Cron Logs |
| GET | `/api/enhanced/decisions` | List Decisions |
| GET | `/api/enhanced/decisions/aggregate` | Get Decision Aggregate |
| GET | `/api/enhanced/decisions/{record_id}` | Get Decision |
| GET | `/api/enhanced/decisions/{record_id}/provenance` | Get Decision Provenance |
| GET | `/api/enhanced/download/{filename}` | Download File |
| GET | `/api/enhanced/ecosystem/atlassian/kanban` | Get Atlassian Kanban |
| GET | `/api/enhanced/ecosystem/datascience/training` | Get Datascience Training |
| GET | `/api/enhanced/ecosystem/github/prs` | Get Github Prs |
| GET | `/api/enhanced/ecosystem/gitlab/mrs` | Get Gitlab Mrs |
| GET | `/api/enhanced/ecosystem/homeassistant/devices` | Get Homeassistant Devices |
| GET | `/api/enhanced/ecosystem/mediadownloader/downloads` | Get Mediadownloader Downloads |
| GET | `/api/enhanced/ecosystem/microsoft/emails` | Get Microsoft Emails |
| GET | `/api/enhanced/ecosystem/nextcloud/events` | Get Nextcloud Events |
| GET | `/api/enhanced/ecosystem/portainer/stacks` | Get Portainer Stacks |
| GET | `/api/enhanced/ecosystem/qbittorrent/torrents` | Get Qbittorrent Torrents |
| GET | `/api/enhanced/ecosystem/scholarx/papers` | Get Scholarx Papers |
| GET | `/api/enhanced/ecosystem/searxng/search` | Get Searxng Search |
| GET | `/api/enhanced/ecosystem/services` | List Ecosystem Services |
| GET | `/api/enhanced/ecosystem/stirlingpdf/jobs` | Get Stirlingpdf Jobs |
| GET | `/api/enhanced/ecosystem/uptime/status` | Get Uptime Status |
| GET | `/api/enhanced/editor-context` | Get Editor Context |
| POST | `/api/enhanced/editor-context` | Publish Editor Context |
| GET | `/api/enhanced/files` | List Files |
| DELETE | `/api/enhanced/files/{filename}` | Delete Workspace File |
| GET | `/api/enhanced/files/{filename}` | Get File |
| PUT | `/api/enhanced/files/{filename}` | Update File |
| GET | `/api/enhanced/goals` | List Goals |
| POST | `/api/enhanced/goals` | Create Goal |
| POST | `/api/enhanced/goals/{goal_id}/cancel` | Cancel Goal |
| GET | `/api/enhanced/goals/{goal_id}/iterations` | Get Goal Iterations |
| GET | `/api/enhanced/graph/graph3d` | Get Graph 3D |
| GET | `/api/enhanced/graph/graph3d/clusters` | Get Graph 3D Lod Clusters |
| GET | `/api/enhanced/graph/graph3d/expand` | Get Graph 3D Lod Expand |
| GET | `/api/enhanced/graph/graph3d/refresh` | Get Graph 3D Lod Refresh |
| GET | `/api/enhanced/graph/graph3d/scope` | Get Graph 3D Lod Scope |
| GET | `/api/enhanced/graph/impact/{symbol}` | Get Impact |
| POST | `/api/enhanced/graph/link` | Link Nodes |
| POST | `/api/enhanced/graph/magma` | Magma Retrieve |
| POST | `/api/enhanced/graph/memory` | Add Memory |
| DELETE | `/api/enhanced/graph/memory/{memory_id}` | Delete Memory |
| GET | `/api/enhanced/graph/memory/{memory_id}` | Get Memory |
| PUT | `/api/enhanced/graph/memory/{memory_id}` | Update Memory |
| GET | `/api/enhanced/graph/node-types` | Get Graph Node Types |
| GET | `/api/enhanced/graph/nodes` | Get Graph Nodes |
| POST | `/api/enhanced/graph/query` | Execute Cypher |
| GET | `/api/enhanced/graph/relationships` | Get Graph Relationships |
| GET | `/api/enhanced/graph/search` | Hybrid Search |
| GET | `/api/enhanced/graph/stats` | Get Graph Stats |
| GET | `/api/enhanced/graph/viz/capabilities` | Get Viz Capabilities |
| POST | `/api/enhanced/graph/viz/render` | Render Viz |
| GET | `/api/enhanced/info` | Get Info |
| GET | `/api/enhanced/kb/article/{article_id}` | Get Kb Article |
| POST | `/api/enhanced/kb/health` | Kb Health Check |
| POST | `/api/enhanced/kb/ingest` | Ingest Kb |
| GET | `/api/enhanced/kb/list` | List Kbs |
| GET | `/api/enhanced/kb/search` | Search Kb |
| POST | `/api/enhanced/kb/update` | Update Kb |
| GET | `/api/enhanced/llm/embedding-models` | List Embedding Models |
| PUT | `/api/enhanced/llm/embedding-models` | Update Embedding Models |
| GET | `/api/enhanced/llm/model-detail` | Get Llm Model Detail |
| GET | `/api/enhanced/llm/model-schema` | Get Llm Model Schema |
| GET | `/api/enhanced/llm/models` | List Llm Models |
| PUT | `/api/enhanced/llm/models` | Update Llm Models |
| GET | `/api/enhanced/maintenance/status` | Get Maintenance Status |
| POST | `/api/enhanced/maintenance/trigger` | Trigger Maintenance |
| POST | `/api/enhanced/mcp/apps/resource` | Read Mcp App Resource Route |
| GET | `/api/enhanced/mcp/server-schema` | Get Mcp Server Schema |
| POST | `/api/enhanced/mcp/servers` | Create Mcp Server |
| DELETE | `/api/enhanced/mcp/servers/{server_name}` | Delete Mcp Server |
| PUT | `/api/enhanced/mcp/servers/{server_name}` | Update Mcp Server |
| GET | `/api/enhanced/mcp/servers/{server_name}/config` | Get Mcp Server Config |
| GET | `/api/enhanced/mcp/servers/{server_name}/tools` | List Mcp Server Tools |
| POST | `/api/enhanced/mcp/tools/call` | Call Mcp Tool Route |
| GET | `/api/enhanced/models` | List Configured Models |
| GET | `/api/enhanced/ontology/actions` | Ontology Actions |
| POST | `/api/enhanced/ontology/derive` | Derive Ontology Property |
| POST | `/api/enhanced/ontology/document/process` | Process Ontology Document |
| POST | `/api/enhanced/ontology/function/invoke` | Invoke Ontology Function |
| GET | `/api/enhanced/ontology/interfaces` | List Ontology Interfaces |
| GET | `/api/enhanced/ontology/interfaces/{name}/implementers` | Get Interface Implementers |
| POST | `/api/enhanced/ontology/object-set/action` | Ontology Object Set Action |
| POST | `/api/enhanced/ontology/object-set/aggregate` | Ontology Object Set Aggregate |
| GET | `/api/enhanced/ontology/object-set/list` | Ontology Object Set List |
| POST | `/api/enhanced/ontology/object-set/pivot` | Ontology Object Set Pivot |
| POST | `/api/enhanced/ontology/object-set/save` | Ontology Object Set Save |
| POST | `/api/enhanced/ontology/object-set/search` | Ontology Object Set Search |
| POST | `/api/enhanced/ontology/object-set/search-around` | Ontology Object Set Search Around |
| GET | `/api/enhanced/ontology/object-types` | List Object Types |
| GET | `/api/enhanced/ontology/object-view/{object_type}` | Get Ontology Object View |
| POST | `/api/enhanced/ontology/object-view/{object_type}` | Save Ontology Object View |
| GET | `/api/enhanced/ontology/object/{object_id}` | Get Ontology Object |
| POST | `/api/enhanced/ontology/object/{object_id}/edit` | Edit Ontology Object |
| POST | `/api/enhanced/ontology/object/{object_id}/revert` | Revert Ontology Edit |
| GET | `/api/enhanced/ontology/property-types` | List Ontology Property Types |
| GET | `/api/enhanced/pipeline/status` | Get Pipeline Status |
| POST | `/api/enhanced/pipeline/trigger` | Trigger Pipeline |
| GET | `/api/enhanced/prompts` | List Prompts |
| GET | `/api/enhanced/prompts/graph` | List Graph Prompts |
| POST | `/api/enhanced/prompts/graph` | Create Graph Prompt |
| GET | `/api/enhanced/prompts/graph/{prompt_id}` | Get Graph Prompt |
| PUT | `/api/enhanced/prompts/graph/{prompt_id}` | Update Graph Prompt |
| GET | `/api/enhanced/prompts/graph/{prompt_id}/diff/{version_a}/{version_b}` | Diff Graph Prompt Versions |
| POST | `/api/enhanced/prompts/graph/{prompt_id}/rollback/{version_id}` | Rollback Graph Prompt |
| GET | `/api/enhanced/prompts/graph/{prompt_id}/versions` | Get Graph Prompt Versions |
| GET | `/api/enhanced/prompts/{name}` | Get Prompt By Name |
| PUT | `/api/enhanced/prompts/{name}` | Update Prompt By Name |
| POST | `/api/enhanced/reload` | Reload Agent |
| POST | `/api/enhanced/repository-manager/bulk` | Trigger Workspace Bulk Actions |
| GET | `/api/enhanced/repository-manager/repos` | List Workspace Repos |
| GET | `/api/enhanced/resources` | List Resources |
| POST | `/api/enhanced/resources/spawn` | Spawn Agent |
| GET | `/api/enhanced/sdd/constitution` | Get Constitution |
| POST | `/api/enhanced/sdd/constitution` | Save Constitution |
| GET | `/api/enhanced/sdd/plans` | List Plans |
| POST | `/api/enhanced/sdd/spec` | Create Spec |
| GET | `/api/enhanced/sdd/specs` | List Specs |
| POST | `/api/enhanced/sdd/sync` | Sync Sdd To Memory |
| GET | `/api/enhanced/sdd/tasks` | Get Tasks |
| GET | `/api/enhanced/security/doctor` | Security Doctor |
| GET | `/api/enhanced/sessions` | Get All Sessions |
| DELETE | `/api/enhanced/sessions/{session_id}` | Delete Session |
| GET | `/api/enhanced/sessions/{session_id}` | Get Session Details |
| POST | `/api/enhanced/sessions/{session_id}/cancel` | Cancel Session Run |
| POST | `/api/enhanced/sessions/{session_id}/reply` | Submit Session Reply |
| POST | `/api/enhanced/skills/{skill_id}/toggle` | Toggle Skill |
| GET | `/api/enhanced/system` | Get System Prompt |
| GET | `/api/enhanced/systems-manager/processes` | List System Processes |
| GET | `/api/enhanced/systems-manager/resources` | Get System Resources |
| GET | `/api/enhanced/tools/graph` | List Graph Tools |
| POST | `/api/enhanced/tools/graph/{tool_id}/toggle` | Toggle Graph Tool |
| POST | `/api/enhanced/tools/toggle` | Toggle Tool Status |
| GET | `/api/enhanced/tunnel-manager/hosts` | Get Tunnel Hosts |
| POST | `/api/enhanced/tunnel-manager/hosts` | Add Tunnel Host |
| POST | `/api/enhanced/upload` | Upload File |
| POST | `/api/enhanced/voice/transcribe` | Transcribe Voice Chunk |
| GET | `/api/enhanced/workflows` | List Workflows |
| POST | `/api/enhanced/workflows` | Save Workflow |
| POST | `/api/enhanced/workflows/{wid}/run` | Run Workflow |
| POST | `/api/rum` | Rum |
