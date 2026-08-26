# API Reference

> **GENERATED — do not edit by hand.** Run `python scripts/generate_openapi.py --write`. Source: the served app's own `app.openapi()`, projected from `agent_webui.server.create_agent_web_app` exactly as production builds it — not hand-copied.

This is the generated reference for **Agent Web Dashboard** (OpenAPI 3.1.0) — 192 paths, 215 operations, 215 with a summary. The running server's own interactive `/docs` (Swagger UI) and `/redoc` pages reflect the identical document live; this page is the static, diffable snapshot checked into docs so it cannot silently drift from the code that generates it.

## Coverage note

**This reference does not yet cover the whole live API.** 238 route(s) across 31 mounted path prefix(es) below are served by the app but carry no OpenAPI schema, so `app.openapi()` — and therefore this page — cannot see them:

`/api/audit`, `/api/chat`, `/api/compliance`, `/api/concept`, `/api/configure`, `/api/connector`, `/api/data`, `/api/document`, `/api/engine`, `/api/epistemic`, `/api/fleet`, `/api/goals`, `/api/graph`, `/api/graphlearn`, `/api/health`, `/api/incident`, `/api/intent`, `/api/media`, `/api/mining`, `/api/object`, `/api/ontology`, `/api/ops`, `/api/pipeline`, `/api/quant`, `/api/research`, `/api/sessions`, `/api/source`, `/api/sparql`, `/api/spec`, `/api/tools`, `/api/usage`

Those surfaces (the canonical Knowledge Graph REST/graph-execution routes, among others) are mounted as raw Starlette routes rather than typed FastAPI routers, so they carry no request/response schema for FastAPI to introspect — not an omission from this generator. This count is recomputed from the live app every time `python scripts/generate_openapi.py --write` runs (see `schemaless_routes()` in `scripts/generate_openapi.py`), so as typed schemas land for those routes this note shrinks and eventually disappears on its own, with no route list to hand-maintain here.

The full spec (not just this rendering of it) is published at [`openapi.json`](openapi.json).

## Operations by path prefix

| Path prefix | Operations |
|---|---|
| `/api/dashboard` | 15 |
| `/api/enhanced` | 149 |
| `/api/graph` | 2 |
| `/api/oauth` | 1 |
| `/api/objects` | 4 |
| `/api/ontology` | 23 |
| `/api/providers` | 2 |
| `/api/registry` | 12 |
| `/api/research` | 7 |

## All operations

| Method | Path | Summary |
|---|---|---|
| GET | `/api/dashboard/daemon/shards` | Daemon Shards |
| POST | `/api/dashboard/daemon/start` | Daemon Start |
| GET | `/api/dashboard/daemon/status` | Daemon Status |
| GET | `/api/dashboard/data` | Get All Data |
| GET | `/api/dashboard/data-subset` | Get Dashboard Subset |
| GET | `/api/dashboard/data/{service_id}` | Get Service Data |
| GET | `/api/dashboard/discover` | Discover Services |
| GET | `/api/dashboard/full` | Get Full Dashboard |
| GET | `/api/dashboard/health` | Health Check |
| POST | `/api/dashboard/hydrate` | Trigger All Hydration |
| POST | `/api/dashboard/hydrate/{source}` | Trigger Hydration |
| GET | `/api/dashboard/hydration-status` | Get Hydration Status |
| GET | `/api/dashboard/layout` | Get Layout |
| PUT | `/api/dashboard/layout` | Save Layout |
| GET | `/api/dashboard/widgets` | List Available Widgets |
| GET | `/api/enhanced/agent-icon` | Get Agent Icon |
| POST | `/api/enhanced/agent-library/a2a` | Register A2A Agent |
| GET | `/api/enhanced/agent-library/agents` | List Library Agents |
| POST | `/api/enhanced/agent-library/agents` | Create Library Agent |
| DELETE | `/api/enhanced/agent-library/agents/{agent_id}` | Archive Library Agent |
| GET | `/api/enhanced/agent-library/agents/{agent_id}` | Get Library Agent |
| GET | `/api/enhanced/agent-library/config-summary` | Agent Config Summary |
| GET | `/api/enhanced/agent-library/suggestions` | Suggest Library Agents |
| GET | `/api/enhanced/agent-library/tools` | List Library Tools |
| GET | `/api/enhanced/agents` | List Agents |
| GET | `/api/enhanced/chats` | List Chats |
| POST | `/api/enhanced/chats` | Save Chat |
| DELETE | `/api/enhanced/chats/{chat_id}` | Delete Chat |
| GET | `/api/enhanced/chats/{chat_id}` | Get Chat |
| PUT | `/api/enhanced/chats/{chat_id}/title` | Update Chat Title |
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
| GET | `/api/enhanced/container-manager/containers` | List Docker Containers |
| GET | `/api/enhanced/cron/calendar` | Get Cron Calendar |
| GET | `/api/enhanced/cron/logs` | Get Cron Logs |
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
| GET | `/api/enhanced/graph/impact/{symbol}` | Get Impact |
| POST | `/api/enhanced/graph/link` | Link Nodes |
| POST | `/api/enhanced/graph/magma` | Magma Retrieve |
| POST | `/api/enhanced/graph/memory` | Add Memory |
| DELETE | `/api/enhanced/graph/memory/{memory_id}` | Delete Memory |
| GET | `/api/enhanced/graph/memory/{memory_id}` | Get Memory |
| PUT | `/api/enhanced/graph/memory/{memory_id}` | Update Memory |
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
| GET | `/api/enhanced/llm/models` | List Llm Models |
| GET | `/api/enhanced/maintenance/status` | Get Maintenance Status |
| POST | `/api/enhanced/maintenance/trigger` | Trigger Maintenance |
| POST | `/api/enhanced/mcp/apps/resource` | Read Mcp App Resource Route |
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
| GET | `/api/enhanced/skills` | List Skills |
| POST | `/api/enhanced/skills/{skill_id}/toggle` | Toggle Skill |
| GET | `/api/enhanced/system` | Get System Prompt |
| GET | `/api/enhanced/systems-manager/processes` | List System Processes |
| GET | `/api/enhanced/systems-manager/resources` | Get System Resources |
| GET | `/api/enhanced/tools` | List All Tools |
| GET | `/api/enhanced/tools/graph` | List Graph Tools |
| POST | `/api/enhanced/tools/graph/{tool_id}/toggle` | Toggle Graph Tool |
| POST | `/api/enhanced/tools/toggle` | Toggle Tool Status |
| GET | `/api/enhanced/tunnel-manager/hosts` | Get Tunnel Hosts |
| POST | `/api/enhanced/tunnel-manager/hosts` | Add Tunnel Host |
| POST | `/api/enhanced/upload` | Upload File |
| POST | `/api/enhanced/voice/transcribe` | Transcribe Voice Chunk |
| GET | `/api/enhanced/workflows` | List Workflows |
| POST | `/api/enhanced/workflows` | Save Workflow |
| GET | `/api/enhanced/workflows/capabilities` | Workflow Capabilities |
| POST | `/api/enhanced/workflows/{wid}/run` | Run Workflow |
| DELETE | `/api/graph/write` | Delete an edge (graph_write action=delete_edge) |
| POST | `/api/graph/write` | Write a node/edge or run another graph_write action |
| GET | `/api/oauth/callback` | Oauth Callback |
| GET | `/api/objects/{object_id}` | Get Object |
| GET | `/api/objects/{object_id}/as-of` | Get Object As Of |
| GET | `/api/objects/{object_id}/history` | Get Object History |
| GET | `/api/objects/{source_id}/path/{target_id}` | Get Object Path |
| GET | `/api/ontology/catalogue` | Get Ontology Catalogue |
| GET | `/api/ontology/export` | Export Ontology |
| GET | `/api/ontology/functions` | List Functions |
| GET | `/api/ontology/functions/{name}` | Get Function |
| GET | `/api/ontology/generate` | Generate Ontology Get |
| POST | `/api/ontology/generate` | Generate Ontology Post |
| GET | `/api/ontology/interfaces` | List Interfaces |
| GET | `/api/ontology/interfaces/{name}` | Get Interface Implementers |
| POST | `/api/ontology/leanix/sync` | Sync Leanix Ontology Route |
| GET | `/api/ontology/lint` | Get Ontology Lint |
| POST | `/api/ontology/load` | Load Ontology |
| GET | `/api/ontology/model-profiles` | List Model Profiles |
| POST | `/api/ontology/model-profiles/sync` | Sync Model Profiles |
| GET | `/api/ontology/model-profiles/{model_id}` | Get Model Profile |
| GET | `/api/ontology/property-types` | List Property Types |
| GET | `/api/ontology/property-types/{type_ref}` | Describe Property Type |
| GET | `/api/ontology/sampling-profiles` | List Sampling Profiles |
| GET | `/api/ontology/sampling-profiles/{task_class}` | Describe Sampling Profile |
| GET | `/api/ontology/schema-graph` | Get Ontology Schema Graph |
| GET | `/api/ontology/schema-summary` | Get Ontology Schema Summary |
| POST | `/api/ontology/validate` | Validate Ontology Candidate |
| GET | `/api/ontology/value-types` | List Value Types |
| GET | `/api/ontology/value-types/{name}` | Get Value Type |
| POST | `/api/providers/{provider_id}/authorize` | Authorize |
| POST | `/api/providers/{provider_id}/revoke` | Revoke |
| GET | `/api/registry/discoveries` | List Registry Discoveries |
| GET | `/api/registry/discoveries/{item_id}` | Get Registry Discoveries |
| GET | `/api/registry/prompts` | List Registry Prompts |
| GET | `/api/registry/prompts/{item_id}` | Get Registry Prompts |
| GET | `/api/registry/resources` | List Registry Resources |
| GET | `/api/registry/resources/{item_id}` | Get Registry Resources |
| GET | `/api/registry/servers` | List Registry Servers |
| GET | `/api/registry/servers/{item_id}` | Get Registry Servers |
| GET | `/api/registry/skills` | List Registry Skills |
| GET | `/api/registry/skills/{item_id}` | Get Registry Skills |
| GET | `/api/registry/tools` | List Registry Tools |
| GET | `/api/registry/tools/{item_id}` | Get Registry Tools |
| GET | `/api/research/artifact/{article_id}` | Get Artifact |
| GET | `/api/research/artifacts` | List Artifacts |
| POST | `/api/research/capture` | Capture Event |
| POST | `/api/research/compile` | Compile Artifact |
| POST | `/api/research/inquire` | Inquire |
| POST | `/api/research/reason` | Reason |
| POST | `/api/research/review` | Review Artifact |
