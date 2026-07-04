# Connector ingestion & OWL mapping (day-0 reference)

> The genesis-side companion to `connector-catalog.md` (which is install/deploy/secrets).
> This doc answers, for every connector worth ingesting: **its `source_sync` source key,
> the entities it pulls into the Knowledge Graph, the OWL classes they map to, and the
> creds/env it needs.** The KG is OWL-native — connectors emit *typed* entities promoted
> to OWL classes (`core/owl_bridge.py` `PROMOTABLE_NODE_TYPES` → a class in the canonical
> ontology library `ontology*.ttl`), not just generic Documents.
>
> Authoritative behavioral reference: the `knowledge-graph-ingest` skill §8b. The code of
> record is `core/source_sync.py` (`_DELTA_HANDLERS`) + `connectors/mcp_tool.py`
> (`MCP_TOOL_PRESETS`). Keep all three in lockstep.

## How a connector is ingested

Two shapes (both fan out from `source_sync(source="all", mode="full")` as laned
`connector_sync` tasks; trigger one directly with `source_sync(source="<key>")`):

- **Typed entity rebuild** — a `_DELTA_HANDLERS` handler drains records and rebuilds
  `ingest_external_batch` entities with `type=<owl-class>` + relationships. First-class OWL
  classes the reasoner acts on. (jira, plane, gitlab, leanix, and the ops/platform set below.)
- **Document** — `PACKAGE_PRESETS` (the fleet sweep) or `MCP_TOOL_PRESETS` turn a record into
  a `:Document`+`:Chunk` (`doc_type` retained). Searchable, not a domain class.

**MCP-configured signal.** The ops/platform connectors below ingest only when their `*-mcp`
server is registered in the connector `mcp_config.json` (`_MCP_TRACKER_SERVERS`), reached via
a host-minted fleet bearer (see `plane-provisioning-and-connector-auth.md` §2). The sweep keeps
each as a candidate when reachable, drops it (skipped, never errored) otherwise.

## Day-0 connector ingestion map

| `source` key | Server (mcp_config) | Entities → OWL classes | Path | Creds / env (OpenBao `apps/<svc>`) |
|---|---|---|---|---|
| `jira` | `atlassian-mcp` | issues→`:Issue`, assignees→`:Person`, epics→`:Goal` | typed (AU-KG.compute.jira-first-class-delta) | Atlassian token in `apps/atlassian-mcp`; `JIRA_PROJECT_KEYS` |
| `confluence` | `atlassian-mcp` | pages→`:ConfluencePage` (`:Document`) | doc (AU-KG.compute.confluence-first-class-delta) | as jira; `CONFLUENCE_SPACE_IDS` |
| `plane` | `plane-mcp` | work items→`:Issue`, projects→`:SoftwareProject` | typed (AU-KG.compute.plane-first-class-delta) | `apps/plane-mcp` (`PLANE_API_KEY`, `PLANE_WORKSPACE_SLUG`, `PLANE_BASE_URL`); `PLANE_PROJECT_IDS` |
| `gitlab` | `gitlab-mcp` / REST | projects→`:Repository`, files→`:File`, symbols→`:Code`, MRs→`:MergeRequest` | typed (AU-KG.backend.declared-columns-so-schema) | `GITLAB_URL`/`GITLAB_TOKEN` or `GITLAB_INSTANCES` |
| `leanix` | `leanix-agent` | fact sheets→`:Application`/`:ITComponent`/`:BusinessCapability` | typed | LeanIX token in `apps/leanix-mcp` |
| `egeria` | `egeria-mcp` | metadata/governance/lineage→`:ProcessModel`/`:GovernanceRule` | materialize | Egeria platform URL |
| `camunda` | `camunda-mcp` | processes→`:BusinessProcess`, steps→`:ProcessStep` | materialize | `apps/camunda-mcp` (Camunda 7/8 URL+creds) |
| `aris` | `aris-mcp` | EPC models→`:ProcessModel`/`:ArchimateElement` | materialize | `apps/aris-mcp` (OAuth) |
| `rss` | native + scholarx | feed items→`:Document`/`:ResearchInquiry` (gated) | feed (KG-2.121) | `KG_RSS_FEEDS` (none for scholarx) |
| `freshrss` | `freshrss-mcp` | news/research→`:Document` (world-model gated) | feed (AU-KG.compute.homelab-rss-reader-as) | `FRESHRSS_URL` or freshrss-mcp GReader creds |
| `archivebox` | `archivebox-api` | snapshots→`:Document` | typed (KG-2.7) | `ARCHIVEBOX_URL` |
| **`dockerhub`** | `dockerhub-mcp` | repos→`:Repository`, images→`:ContainerImage` (`contains`) | typed (**AU-KG.compute.dockerhub-repositories**) | `DOCKERHUB_NAMESPACE(S)`; Docker Hub PAT in `apps/dockerhub-api` |
| **`langfuse`** | `langfuse-mcp` | traces→`:Trace`, observations→`:Observation`, LLM-calls→`:Generation` (`part_of`) | typed (**AU-KG.compute.langfuse-traces-observations**) | Langfuse public/secret keys in `apps/langfuse-mcp` |
| **`technitium`** | `technitium-dns-mcp` | zones→`:DnsZone`, records→`:DnsRecord` (`part_of`) | typed (**AU-KG.compute.technitium-dns-zones-records**) | Technitium API token in `apps/technitium-dns-mcp` |
| **`tunnel_manager`** | `tunnel-manager-mcp` | hosts→`:Host`, tunnels→`:Tunnel` (`connects_via`) | typed (**AU-KG.compute.tunnel-manager-hosts**) | SSH inventory (no extra secret; reads tunnel-manager state) |
| **`uptime_kuma`** | `uptime-mcp` (alias `uptime-kuma-mcp`) | monitors→`:UptimeMonitor`, heartbeats→`:HeartbeatStat` (`part_of`) | typed (**AU-KG.compute.uptime-kuma-monitors**) | Uptime-Kuma URL+creds in `apps/uptime-mcp` |
| **`home_assistant`** | `home-assistant-mcp` | entities→`:Entity`, device groups→`:Device` (`part_of`) | typed (**AU-KG.compute.home-assistant-states**) | HA long-lived token in `apps/home-assistant-mcp` |
| **`twenty`** | `twenty-mcp` | people→`:Person`, companies→`:Company`, opportunities→`:Opportunity` (`member_of`/`part_of`) | typed (**AU-KG.compute.twenty-crm-people-companies**) | Twenty API key in `apps/twenty-mcp` |
| **`audiobookshelf`** | `audiobookshelf-mcp` | libraries→`:Library`, books→`:Book`, authors→`:Author` (`part_of`/`authored_by`) | typed (**AU-KG.compute.audiobookshelf-libraries-books-authors**) | `AUDIOBOOKSHELF_URL`+`AUDIOBOOKSHELF_TOKEN` (Bearer JWT) in `apps/audiobookshelf-mcp` |
| **`firefly_iii`** | `firefly-iii-mcp` | accounts→`:Account`, transactions→`:Transaction`, budgets→`:Budget` (`part_of`/`member_of`) | typed (**AU-KG.compute.firefly-iii-accounts-transactions**) | `FIREFLY_III_URL`+`FIREFLY_III_TOKEN` (Passport PAT) in `apps/firefly-iii-mcp` |
| **`paperless_ngx`** | `paperless-ngx-mcp` | documents→`:Document`, correspondents→`:Correspondent`, tags→`:Tag` (`member_of`/`tagged_with`) | typed (**AU-KG.compute.paperless-ngx-documents-correspondents**) | `PAPERLESS_NGX_URL`+`PAPERLESS_NGX_TOKEN` (DRF token) in `apps/paperless-ngx-mcp` |
| **`gramps`** | `gramps-mcp` | people→`:Person`, families→`:Family`, events→`:Event` (`member_of`/`part_of`) | typed (**AU-KG.compute.gramps-web-people-families**) | `GRAMPS_WEB_URL`+`GRAMPS_WEB_USERNAME`/`PASSWORD` (or `GRAMPS_WEB_TOKEN`) in `apps/gramps-mcp` |

## Available via MCP, ingestion-optional

Configured/fixed today but with **no dedicated ingestion handler** — ingest only when there is
clear knowledge value (else reach them live as tools). They surface through the declarative
fleet sweep / `MCP_TOOL_PRESETS` as `:Document`s, not typed domain classes:
`owncast`, `mealie` (`mealie-recipes` preset), `searxng` (`searxng-search` preset), `lgtm`,
`nextcloud` (`nextcloud-files` preset), `arr`.

## Wiring a new connector (the recipe)

1. `MCP_TOOL_PRESET` in `connectors/mcp_tool.py` (server + tool + records_path + field map +
   pagination) — the drain half.
2. A `_DELTA_HANDLER` in `core/source_sync.py` that rebuilds typed entities from
   `metadata.record` and `ingest_external_batch`-es them (for typed/OWL connectors), plus the
   `_MCP_TRACKER_SERVERS` row so the sweep gates it on `mcp_config`.
3. The node `type`(s) in `PROMOTABLE_NODE_TYPES` (`core/owl_bridge.py`) and any new edge type
   in `PROMOTABLE_EDGE_TYPES`.
4. The OWL class in the **canonical** ontology library (extend `ontology_<domain>.ttl`, never a
   new per-connector `.ttl`); run `python3 scripts/check_ontology.py`.
5. The §8b row in the `knowledge-graph-ingest` skill + this table.
