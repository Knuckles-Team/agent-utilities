---
name: kg-ingest
skill_type: skill
description: >-
  Bulk ingests the workspace projects, ScholarX documents, and conversation logs into the Knowledge Graph,
  AND fans a single "full ingest" trigger across every ingestion path at once — workspace codebase +
  documents PLUS the native 'rss' and 'freshrss' feed connectors AND every agent-packages/agents/*
  connector (the fleet sweep). Use when the user wants to "ingest the workspace", "bulk ingest",
  "full ingest", "ingest everything", "ingest all sources", "ingest these git urls",
  "ingest conversations", "backup the kg", or "wipe the kg".
  Automatically handles finding all workspace paths natively via the repository-manager MCP,
  cloning parallel git URLs if requested, and firing off the ingestion pipeline across all sources/lanes.
  Also documents the fleet's NATIVE package-side push (every agents/* connector ingests its own data
  into epistemic-graph in three modalities — typed OWL nodes, documents, and raw BLOBS/attachments —
  via the shared native_ingest primitive), so the KG stores the data itself, not just metadata.
license: MIT
tags: [knowledge-graph, ingestion, workspace, bulk, git, conversations, backup, rss, freshrss, connectors, fleet, native, blob]
tier: core
wraps: [graph_ingest, source_sync, source_drain, source_connector, document_process]
metadata:
  author: Genius
  version: '2.2.0'
---

# Knowledge Graph Ingestion

This skill coordinates bulk data ingestion into the unified Knowledge Graph. It handles retrieving workspace configuration, cloning ad-hoc repositories, ingesting conversation logs from multiple IDEs/agents, and triggering the ingestion pipeline.

## 0. Full Ingest — one trigger, every path at once (CONCEPT:AU-KG.ingest.enterprise-source-extractor)

> Triggers: "full ingest", "ingest everything", "ingest all sources", "exercise every
> ingestion path", or any "ingest the workspace" where the user wants the complete picture.

A FULL ingest run exercises **all four ingestion families in one fan-out** so the KG is
hydrated from every reachable source simultaneously. Each family lands on its OWN task
**lane** (see `agent_utilities/knowledge_graph/core/task_lanes.py`), so heavy codebase
indexing in the `ingestion` lane can never head-of-line-block the connector/feed syncs in
the `connectors` / `worldview` lanes — they all drain in parallel below the shared queue.

| # | Family | Tool (native go__*) | Lane | What it covers |
|---|--------|---------------------|------|----------------|
| 1 | **Codebase + documents** | `graph_ingest` (alias `kg_ingest`) with `target_path` = the workspace/doc paths (Sections 1, 3, 7) | `ingestion` | every repo, ScholarX papers, conversations, ontologies, configs, skills |
| 2 | **Native RSS feeds** | `source_sync(source="rss", mode="full")` | `connectors`/`worldview` | `KG_RSS_FEEDS` + the runtime `:FeedSource` registry + ScholarX arXiv, world-model gated (`_sync_rss`) |
| 3 | **FreshRSS** | `source_sync(source="freshrss", mode="full")` | `connectors`/`worldview` | the FreshRSS GReader API → world-model gated news/research (`_sync_freshrss`) |
| 4 | **Every `agents/*` connector** | `source_sync(source="all", mode="full")` | `connectors` | the fleet sweep — fans out one laned `connector_sync` task per configured connector (`sweep_all_sources`), reaching each agent-package via the MCP fleet adapter (`PACKAGE_PRESETS`) plus the gitlab/leanix/jira/confluence/plane/materialize sources |

### The single declarative trigger

`source_sync(source="all", mode="full")` is THE one connector trigger: it calls
`sweep_all_sources`, which enqueues a `connector_sync` task **per** source in the candidate
union — the delta-capable handlers (which **include `rss` and `freshrss`**), every
capability-registry source that env-detects as configured, the `agents/*` MCP-package
presets (Section 10), and the materialize extractor sources. So families 2, 3 and 4 above
are all driven by this ONE call; you only fire 1 and 4:

```text
# 1) codebase + documents (heavy file-ingestion lane) — see Sections 1/3/7 for the path list
graph_ingest(target_path="<JSON array of workspace + doc + ontology + config + skill paths>")

# 2-4) every connector + both feed sources, fanned across the connectors/worldview lanes
source_sync(source="all", mode="full")
```

Use `mode="full"` for a complete (re-)hydrate; `mode="delta"` for an incremental top-up
(the write-layer content-hash delta makes unchanged entities a no-op either way). Then
follow Section 5 to monitor every lane drain.

> Declarative, not procedural: you do **not** enumerate connectors by hand. The candidate
> set is computed by `sweep_all_sources` from the registries; adding a connector to
> `_DELTA_HANDLERS`, the capability registry, or `PACKAGE_PRESETS` makes the next full
> ingest pick it up with **no change to this skill**.

## Capabilities

### 1. Default Workspace & ScholarX Ingestion
When the user asks to ingest the workspace (without specifying explicit targets), you MUST:
1. Get the local workspace paths by executing the `mcp_repository-manager_rm_workspace` tool with `action: 'paths'`. This natively returns a list of all absolute paths for projects defined in the ecosystem.
2. Append the default ScholarX document directory to the list: `~/.local/share/scholarx/papers`
3. Convert the combined list of paths into a JSON-formatted array.
4. Execute the `mcp_agent-utilities-kg_kg_ingest` tool, passing the JSON array to the `target_path` parameter.
5. **Prompt for Chat Ingestion**: Explicitly prompt the user to confirm whether they would like to ingest all conversation/chat logs from active IDE platforms (e.g. Antigravity or Claude Code) to capture development history and context.
6. **Tool/Skill Configuration Hydration**: Incorporate the IDE's/global active `mcp_config.json` (e.g., at `~/.config/agent-utilities/mcp_config.json`) and the agent skills directories (defaulting to `/home/apps/workspace/agent-packages/skills/universal-skills` and `/home/apps/workspace/agent-packages/skills/skill-graphs`) as ingestion targets to ensure the Knowledge Graph is fully hydrated with active tool, schema, and capability definitions.

### 2. Parallel Git URL Cloning
If the user specifies explicit comma-separated Git URLs to ingest:
1. You MUST clone them locally in parallel before ingestion.
2. Use your `run_command` tool to execute a bash script that clones all URLs simultaneously into `/home/apps/workspace/open-source-libraries/` (or another appropriate directory).
   - **Example:** `git clone <url1> & git clone <url2> & wait`
3. After the clones complete, compile the local absolute paths of the cloned directories into a JSON array.
4. Execute `mcp_agent-utilities-kg_kg_ingest` with the JSON array.

### 3. Conversation Log Ingestion
Ingest conversation logs from supported IDE/agent platforms:
- **Antigravity**: `~/.gemini/antigravity/brain/*/overview.txt`
- **Windsurf**: `~/.codeium/windsurf/memories/` or `~/.windsurf/memories/`
- **Claude Code**: `~/.claude/projects/` or `~/.config/claude/`
- **Codex**: `~/.codex/sessions/`

When the user consents to or requests conversation ingestion, you MUST identify all existing logs from these directories, compile them into a target list, and call `mcp_agent-utilities-kg_kg_ingest` with the log directories/files.
Conversation logs are ingested as `Conversation` nodes with `DISCUSSED_IN` edges linking to relevant Concept nodes.

### 4. DB Backup & Wipe
- **Backup**: `mcp_agent-utilities-kg_kg_inspect` with `view: 'backup'` — creates a timestamped backup of the database.
- **Wipe**: `mcp_agent-utilities-kg_kg_inspect` with `view: 'wipe'` — clears all nodes and edges for a fresh start.

### 5. Progress Monitoring
After triggering the ingestion, you should:
1. Call `mcp_agent-utilities-kg_kg_jobs` with `action: 'list'` to monitor the ingestion queue.
2. Report the completion percentage and job status to the user.

### 6. ScholarX Paper Downloads & Ingestion
When the user asks to download or ingest a research paper using the ScholarX MCP tools, and they provide only a raw numerical or alphanumeric ID (e.g., `2605.12975`):
1. You MUST explicitly prompt the user to confirm the paper's source (e.g., "Is this from arXiv, PMC, bioRxiv, etc.?").
2. Once the user confirms the source, you MUST prepend the source prefix to the ID (e.g., `arxiv:2605.12975`) before executing the `sx_search` or `sx_storage` tools.
3. After the paper is downloaded, you can ingest it by executing `mcp_agent-utilities-kg_kg_ingest` with the local downloaded file path as the `target_path`.

### 7. Infrastructure Topology Ingestion (CONCEPT:AU-OS.governance.reactive-multi-axis-budget)
When ingesting the workspace, you MUST also ingest infrastructure state to fully
hydrate the Knowledge Graph with the physical and virtual topology:

#### 7a. Inventory File
Ingest `~/.config/agent-utilities/inventory.yaml` as the canonical hardware inventory.
For each host entry, create or update a `HardwareNode` KG node with:
- `hostname`, `ip_address` (from `ansible_host`), `group`, `status`
- `ssh_user` (from `ansible_user`), `ssh_key` (from `ansible_ssh_private_key_file`)
- Any extended metadata (`hardware.*`, `os.*`, `roles`, `containers.*`, `networking.*`)

Create `HAS_INTERFACE` edges from `HardwareNode` → `NetworkInterface` nodes for each
network interface defined, and `CONNECTS_VIA` edges for any VPN tunnel entries.

#### 7b. Ontology Files
Include all OWL ontology files as ingestion targets:
- `agent_utilities/knowledge_graph/ontology.ttl` — base ontology
- `agent_utilities/knowledge_graph/ontology_infrastructure.ttl` — infrastructure module

These provide the formal BFO-aligned class hierarchy for all infrastructure nodes.

#### 7c. Workflow Catalog
Ingest `agent_utilities/workflows/catalog.yaml` to create `WorkflowDefinition` nodes
for each workflow, with `HAS_STEP` edges to individual `WorkflowStep` nodes and
`REQUIRES_TOOL` edges to the MCP server `CallableResource` nodes they reference.

#### 7d. Topology Snapshots
If topology maps exist at `~/.local/share/agent-utilities/topology/`, ingest them:
- `topology.json` — full infrastructure graph snapshot
- `service_map.json` — service dependency chains
- `network_map.json` — network topology

These create/update `Container`, `ContainerStack`, `NetworkSubnet`, `DNSRewrite`,
`ReverseProxy`, and `ObservabilityStack` nodes with their respective edges
(`RUNS_ON`, `BELONGS_TO_STACK`, `DEPLOYED_ON`, `ROUTES_TO`, `RESOLVES_DNS_FOR`, etc.).

#### 7e. DNS Rewrites
Query `technitium-dns-mcp` → `list_records` and ingest each record as a `DNSRecord`
node with `RESOLVES_DNS_FOR` edges linking to the corresponding `PlatformService` node.

#### 7f. Container State (Live)
If `container-manager-mcp` or `portainer-mcp` are available, query live container
and stack state and ingest as `Container` and `ContainerStack` nodes with `RUNS_ON`
and `BELONGS_TO_STACK` edges to the appropriate `HardwareNode` nodes.

#### Default Ingestion Target List
When performing a full workspace ingestion, the following infrastructure paths
MUST be appended to the ingestion target list:
```
~/.config/agent-utilities/inventory.yaml
~/.config/agent-utilities/mcp_config.json
~/.config/agent-utilities/config.json
~/.local/share/agent-utilities/topology/
agent_utilities/knowledge_graph/ontology.ttl
agent_utilities/knowledge_graph/ontology_infrastructure.ttl
agent_utilities/workflows/catalog.yaml
```

---

## 8. Unified Ingestion Engine — category → `graph_ingest` matrix (KG-2.7/2.8)

All content enters through ONE `IngestionEngine` (`graph_ingest` is a thin MCP
wrapper). Set `content_type` to route a path/sentinel synchronously; otherwise
it auto-classifies. Delta-skip (durable manifest) means re-ingesting unchanged
sources is a no-op.

| Category | content_type | target_path | Produces |
|---|---|---|---|
| LLM/embedding config | `config` | `config.json` | `LanguageModel`/`EmbeddingModel`/`SystemConfig` |
| Prompts | `prompt` | `agent_utilities/prompts/*.json` | `Prompt` + `Concept` (MENTIONS) |
| MCP servers | `mcp_server` | `mcp_config.json` | `Server` + `NativeTool` (PROVIDES) w/ tool descriptions |
| Skills | `skill` | a skill dir (`SKILL.md` frontmatter) | `Skill` |
| Documents | `document` | a file, dir, or URL (md/pdf/txt) | `Document{content}` + verbatim `IdeaBlock` chunks (`PART_OF`) + `Concept` (MENTIONS) — same shape regardless of submission form (KG-2.7) |
| Specs | (auto in codebase) | `**/.specify/**` | `Spec`/`ImplementationPlan` |
| Chats | `conversation` | `"chats"` sentinel | `Thread`/`Message` + per-thread `Concept` |
| Codebases | `codebase` | a repo path | `Code`/`Test`/`Feature` (CALLS/IMPLEMENTS/COVERS) |

## 8a. Connector fan-out — source → `source_sync` matrix (KG-2.9 / AU-KG.compute.gitlab-api-gitlab-atlassian)

The connector side of a full ingest mirrors the document matrix above, but the
entrypoint is `source_sync` and the fan-out is laned. The candidate set the
`source="all"` sweep dispatches is **computed declaratively** (`sweep_all_sources`)
from three registries — you never list connectors by hand:

| Group | Registry (data, not code in this skill) | `source` value | Lane |
|---|---|---|---|
| Native feeds | `_DELTA_HANDLERS` (`rss`, `freshrss`) | `rss`, `freshrss` | `connectors` → `worldview` (world-model gated) |
| Enterprise / tracker / IaC | `_DELTA_HANDLERS` + capability registry (`gitlab`, `leanix`, `jira`, `confluence`, `plane`, `archivebox`, …) | each source id | `connectors` |
| Ops / platform typed connectors | `_DELTA_HANDLERS` (`dockerhub`, `langfuse`, `technitium`, `tunnel_manager`, `uptime_kuma`, `home_assistant`, `twenty`) — typed OWL entities, MCP-configured (AU-KG.compute.dockerhub-repositories–2.161) | each source id | `connectors` |
| Media / finance / doc / genealogy connectors | `_DELTA_HANDLERS` (`audiobookshelf`, `firefly_iii`, `paperless_ngx`, `gramps`) — typed OWL entities, MCP-configured (AU-KG.compute.audiobookshelf-libraries-books-authors–2.166) | each source id | `connectors` |
| **Every `agents/*` connector** | `package_manifest.PACKAGE_PRESETS`, drained by `_sync_fleet_connectors` via the generic `mcp` connector | `fleet_connectors` | `connectors` |
| Materialize extractors | `enrichment.materialize.MATERIALIZE_SOURCES` (`camunda`, `aris`, `egeria`) | each source id | `connectors` |
| Fleet capability elevation | `_sync_fleet` (slow MCP re-probe; boot/explicit only, NOT the */20m sweep) | `fleet` | `connectors` |

`source_sync(source="all")` enqueues one laned `connector_sync` task **per**
candidate, so every connector (both feeds + the whole `agents/*` fleet) drains in
parallel. The `fleet_connectors` source iterates `PACKAGE_PRESETS`, attempts only
packages whose MCP server is registered in `mcp_config.json`, and reports
unconfigured packages as *skipped* (never errored). Each yielded record ingests
through the same `DocumentProcessor` (chunk + concept-link) as documents. Add a new
package to `PACKAGE_PRESETS` and the next full ingest picks it up with **no change
to this skill** — that is the declarative contract.

See the companion declarative manifest `ingest_manifest.yaml` (next to this file)
for the machine-readable family→tool→lane mapping that a driver can consume.

## 8b. Connector → OWL entity reference (what gets ingested + how it's modeled)

This is the **authoritative map of every configured connector**: its `source_sync`
source key, the entities it ingests, and the OWL ontology classes they map to. The KG
is OWL-native — a connector's records are not generic Documents but **typed entities**
whose `type` is promoted to its OWL class (`core/owl_bridge.py` `PROMOTABLE_NODE_TYPES`
→ a class declared in the canonical ontology library). **Three ingestion shapes** (a
"maximum-ingestion" connector uses every one that applies — see §8c):

- **Typed entity rebuild** (`_DELTA_HANDLERS` in `core/source_sync.py`): the handler
  drains records and rebuilds `ingest_external_batch` entities with `type=<owl-class>`
  + relationships — first-class OWL classes the reasoner acts on.
- **Document** (`PACKAGE_PRESETS` via `fleet_connectors`, or `MCP_TOOL_PRESETS`): the
  record becomes a `:Document`+`:Chunk` (with `doc_type`), searchable but not a domain
  class. Most `agents/*` connectors land here through the fleet sweep.
- **Blob (raw bytes)** (`memory/media_store.py` `MediaStore` / `memory/native_ingest.py`
  `media_store`): the raw file/attachment/scan/media → a content-addressed `:Blob` +
  `:MediaAsset` node (deduped), with extracted text/OCR/transcript flowing into the
  Document shape above. This is how the KG stores the data *itself*, not just metadata.

| `source` key | Connector / server | Entities ingested | OWL classes (`ontology*.ttl`) | Path |
|---|---|---|---|---|
| `jira` | atlassian-mcp | issues, assignees, epics | `:Issue` / `:Person` / `:Goal`(epic) | typed `_sync_jira` (AU-KG.compute.jira-first-class-delta) |
| `confluence` | atlassian-mcp | wiki pages | `:ConfluencePage` (`:Document`) | doc `_sync_confluence` (AU-KG.compute.confluence-first-class-delta) |
| `plane` | plane-mcp | work items, projects | `:Issue` / `:SoftwareProject` | typed `_sync_plane` (AU-KG.compute.plane-first-class-delta) |
| `gitlab` | gitlab-mcp / REST | projects, files, symbols, MRs | `:Repository` / `:File` / `:Code` / `:MergeRequest` | typed `_sync_gitlab` (AU-KG.backend.declared-columns-so-schema) |
| `egeria` | egeria-mcp | metadata, governance, lineage | `:ProcessModel` / `:GovernanceRule` / lineage | materialize `MATERIALIZE_SOURCES` |
| `camunda` | camunda-mcp | processes, deployments | `:BusinessProcess` / `:ProcessStep` | materialize `MATERIALIZE_SOURCES` |
| `aris` | aris-mcp | EPC process models | `:ProcessModel` / `:ArchimateElement` | materialize `MATERIALIZE_SOURCES` |
| `leanix` | leanix-agent | fact sheets (apps, IT components) | `:Application` / `:ITComponent` / `:BusinessCapability` | typed `_sync_leanix` |
| `rss` | native + scholarx | news/research feed items | `:Document` / `:ResearchInquiry` (gated) | feed `_sync_rss` (KG-2.121) |
| `freshrss` | freshrss-mcp | curated news/research | `:Document` (world-model gated) | feed `_sync_freshrss` (AU-KG.compute.homelab-rss-reader-as) |
| (scholarx) | scholarx-mcp | research papers | `:Document` + `:Concept` | via `rss` feed + `graph_ingest` |
| `archivebox` | archivebox-api | preserved web snapshots | `:Document` | typed `_sync_archivebox` (KG-2.7) |
| **`dockerhub`** | dockerhub-mcp | registry images, repos | **`:Repository` / `:ContainerImage`** (`contains`) | typed `_sync_dockerhub` (**AU-KG.compute.dockerhub-repositories**) |
| **`langfuse`** | langfuse-mcp | LLM traces, observations, generations | **`:Trace` / `:Observation` / `:Generation`** (`part_of`) | typed `_sync_langfuse` (**AU-KG.compute.langfuse-traces-observations**) |
| **`technitium`** | technitium-dns-mcp | DNS zones + records | **`:DnsZone` / `:DnsRecord`** (`part_of`) | typed `_sync_technitium` (**AU-KG.compute.technitium-dns-zones-records**) |
| **`tunnel_manager`** | tunnel-manager-mcp | host inventory, tunnels | **`:Host` / `:Tunnel`** (`connects_via`) | typed `_sync_tunnel_manager` (**AU-KG.compute.tunnel-manager-hosts**) |
| **`uptime_kuma`** | uptime-(kuma-)mcp | monitors + heartbeat stats | **`:UptimeMonitor` / `:HeartbeatStat`** (`part_of`) | typed `_sync_uptime_kuma` (**AU-KG.compute.uptime-kuma-monitors**) |
| **`home_assistant`** | home-assistant-mcp | devices, entities/states | **`:Device` / `:Entity`** (`part_of`) | typed `_sync_home_assistant` (**AU-KG.compute.home-assistant-states**) |
| **`twenty`** | twenty-mcp | CRM people, companies, opportunities | **`:Person` / `:Company` / `:Opportunity`** (`member_of`/`part_of`) | typed `_sync_twenty` (**AU-KG.compute.twenty-crm-people-companies**) |
| **`audiobookshelf`** | audiobookshelf-mcp | libraries, books/audiobooks, authors | **`:Library` / `:Book` / `:Author`** (`part_of`/`authored_by`) | typed `_sync_audiobookshelf` (**AU-KG.compute.audiobookshelf-libraries-books-authors**) |
| **`firefly_iii`** | firefly-iii-mcp | accounts, transactions, budgets | **`:Account` / `:Transaction` / `:Budget`** (`part_of`/`member_of`) | typed `_sync_firefly_iii` (**AU-KG.compute.firefly-iii-accounts-transactions**) |
| **`paperless_ngx`** | paperless-ngx-mcp | documents, correspondents, tags | **`:Document` / `:Correspondent` / `:Tag`** (`member_of`/`tagged_with`) | typed `_sync_paperless_ngx` (**AU-KG.compute.paperless-ngx-documents-correspondents**) |
| **`gramps`** | gramps-mcp | people, families, events | **`:Person` / `:Family` / `:Event`** (`member_of`/`part_of`) | typed `_sync_gramps` (**AU-KG.compute.gramps-web-people-families**) |

The new ops/platform connectors (bold) are **MCP-configured**: each ingests only when its
`*-mcp` server is registered in `mcp_config.json` (`_MCP_TRACKER_SERVERS`), so the
`source="all"` sweep keeps it as a candidate when reachable and drops it (skipped, never
errored) otherwise. Trigger one directly with `source_sync(source="<key>", mode="delta")`.

**Available via MCP but ingestion-optional** (their `*-mcp` is fixed/configured today, no
dedicated handler — ingest only when there is clear knowledge value, else reach live):
`owncast`, `mealie` (recipe docs preset `mealie-recipes`), `searxng` (search preset
`searxng-search`), `lgtm`, `nextcloud` (folder preset `nextcloud-files`), `arr`. These
surface through the declarative fleet sweep / `MCP_TOOL_PRESETS` as `:Document`s, not
typed domain classes.

> Still declarative: adding a typed connector = a `_DELTA_HANDLER` + its `MCP_TOOL_PRESET`
> + the `PROMOTABLE_NODE_TYPES` entries + the OWL class in the canonical ontology — then the
> next `source="all"` sweep picks it up. Keep this table in lockstep with `_DELTA_HANDLERS`.

## 8c. Native connector push — package-side ingestion (nodes + documents + blobs)

Complementing the hub-side **pull** above (`source_sync` drains a connector from the hub),
every `agents/*` connector also ships **native push**: its OWN code writes its data into the
ONE engine as it works, via the shared primitive
`agent_utilities/knowledge_graph/memory/native_ingest.py`
(CONCEPT:AU-KG.ingest.enterprise-source-extractor). This is the "maximum ingestion" bar — a
connector pushes in **every modality that applies**, not just one:

| Primitive | Modality | Produces | Package module |
|---|---|---|---|
| `native_ingest.ingest_entities(entities, rels, source, domain)` | typed nodes | OWL `:Class` nodes + links | `<pkg>/kg_ingest.py` (thin mapper) |
| `native_ingest.ingest_documents(docs, source, domain)` | documents | `:Document` (text + `source_uri`; hub chunks/embeds) | `<pkg>/kg_ingest.py` |
| `native_ingest.media_store().store_media(bytes, …)` | blob | `:Blob` + `:MediaAsset` (content-addressed, deduped) | `<pkg>/kg_media.py` |

All three ride the **lightweight** `GraphComputeEngine()._client` (the heavy
`IntelligenceGraphEngine` is not constructible in a connector). Every entry point is
dependency-/engine-guarded — it **no-ops** with no reachable engine, so a connector runs with
zero KG infra. Wired default-on into the package's fetch/download flow + surfaced on an MCP
tool (Wire-First). Node ids: `<domain>:<class>:<externalId>`; `type` matches the package's
`ontology_providers` `.ttl`. Reach the engine via `GRAPH_SERVICE_ENDPOINTS=tcp://<host>:9100`.

**Reference implementations (LIVE-verified):** `media-downloader/kg_media.py` (a downloaded
video → `:MediaAsset` blob, fetch-back byte-identical) and `gitlab-api/kg_ingest.py` (projects
→ `:Project`/`:GitLabGroup` typed nodes).

### Per-package native ingestion — the "maximum ingestion" matrix

The fleet enrichment gives each connector native push in every applicable modality. Enterprise
record-sources do **all three** (typed nodes + KB/notes documents + attachment blobs); file/media
packages are document+blob heavy:

| Connector | Typed nodes | Documents | Blobs |
|---|---|---|---|
| `servicenow-api` | `:Incident`/`:Change`/`:ConfigurationItem`/`:Person` | KB articles | ticket attachments |
| `erpnext-agent` | `:Customer`/`:SalesOrder`/`:Item`/`:Invoice`/`:Supplier`/`:Employee` | notes/descriptions | print-format PDFs |
| `atlassian-agent` (jira) | `:Issue`/`:Epic`/`:Sprint`/`:Person` | Confluence pages | issue attachments |
| `nextcloud-agent` | share/folder structure | file text (pdf/office via `read_any`), image OCR | the files themselves |
| `paperless-ngx-mcp` | `:Correspondent`/`:Tag` | OCR text | scanned PDFs |
| `mattermost-mcp` | `:Channel`/`:Person`/`:Team` | messages | attachments |
| `gitlab-api` / `github-agent` | `:Project`/`:MergeRequest`/`:Issue` | — | (release/CI artifacts) |
| `salesforce-agent` / `twenty-mcp` | `:Account`/`:Contact`/`:Opportunity` | — | — |
| `media-downloader` | — | subtitles/metadata | **video/audio** (proven) |
| `jellyfin-mcp` / `audiobookshelf-mcp` | `:MediaAsset`/`:Book`/`:Author` | — | posters, audio→transcript |
| `gramps-mcp` | `:Person`/`:Family`/`:Event` | — | photos/records (OCR) |
| `langfuse-agent` / `lgtm-mcp` | `:Trace`/`:Dashboard` | — | — (+ time-series) |

> Declarative contract holds: the shared `native_ingest` primitive is the ONE write path;
> a connector adds only its mapper (`kg_ingest.py`/`kg_media.py`). Keep this matrix in step
> with the packages' `kg_ingest`/`kg_media` modules as the fleet enrichment fans out.

## 9. Skill-graph packages — distill OUT / import back (KG-2.7 / AHE-3.9)

> **Auto-ingested by default.** When the KG is reachable, the `knowledge_base`
> pipeline phase auto-ingests **all** packaged skill-graphs (`get_skill_graphs_path(
> default_enabled=True)`) and the universal-skills workflow corpus — delta-skipped, so
> only the first run is heavy. Disable on constrained installs via
> `KG_AUTO_INGEST_SKILLS=false`. The actions below are for explicit, on-demand control.

The KG is the source of truth; a **skill-graph is a versioned, round-trippable
projection of a KG subgraph**. Two symmetric `graph_ingest` actions:

- **Distill (export):** `graph_ingest(action="distill", target_path="<out dir>",
  corpus_name="<seed node id>"  OR  description="<semantic query>", max_depth=2)`.
  Walks a coherent subgraph → a `reference/` markdown tree + `kg_manifest.json`
  (node ids, edges, ontology, snapshot). The output dir is consumable verbatim by
  `skill-graph-builder` (`generate_skill.py --from-kg "<seed-or-query>"`), which
  also surfaces the manifest in the SKILL.md frontmatter (`kg_manifest`,
  `kg_snapshot`, `kg_anchors`). Community detection → folders; edges → TOC nesting
  + inline cross-links; a `Document` and its chunks are de-duplicated to one file.
  Pass `content_type="workflow"` to instead distill a **graph-native
  skill-workflow** — a procedure step-DAG where `PRECEDES` edges become
  `depends_on` ordering (validatable by `skill-workflow-builder`).
- **Import (round-trip):** `graph_ingest(action="import_pack",
  target_path="<skill-graph dir>", corpus_name="dedup")`. Reads `kg_manifest.json`
  and reconstructs the subgraph here — preserving original node ids + edges — so a
  curated package merges into another brain. `corpus_name="dedup"` runs the
  IdeaBlock dedup-merge so two packages on the same topic converge instead of
  duplicating.

Route a fresh crawl straight in with `crawl.py --ingest-kg` or
`generate_skill.py --ingest-kg` (standardized document ingestion), then distill a
clean, curated, shareable package from the KG.

## Verification queries (read-only; supported shapes only)

```cypher
-- per-category node counts
MATCH (n:Concept) RETURN count(n)
-- cross-category interweaving (chats/docs share Concept nodes)
MATCH (t:Thread)-[r:MENTIONS]->(c:Concept) RETURN count(r)
MATCH (s:Document)-[:MENTIONS]->(c:Concept) RETURN count(DISTINCT c)
-- OWL edges present: PROVIDES, IMPLEMENTS, CALLS, CONTAINS, MENTIONS, ADDRESSES
```

NOTE: the engine serves relationship traversal natively from its own compute over
its durable store; under `fanout`, a Postgres/pg-age mirror (`kg_edges`) keeps a
queryable copy via the `FanOutBackend`. **Negation (`WHERE NOT (c)-[:R]->()`) is
NOT transpiled** — compute set-differences in code (see
`topic_resolver.unresolved_topics`).

## OWL cross-category relationship expectations

`MENTIONS` (chats/docs→Concept), `RELATES_TO`/`REALIZES` (Concept→Code, via the
embedding-backed `link_concepts_to_code` once embeddings are backfilled),
`PROVIDES` (Server→NativeTool), `CONTAINS` (Thread→Message), `IMPLEMENTS`/`CALLS`
/`DEPENDS_ON` (Code), `ADDRESSES`/`ADDRESSED_BY` (research source→topic).

## Orchestration + self-evolution recipe

```text
# regular prompt through the pydantic-ai graph (dynamic model+prompt selection)
graph_orchestrate(action="execute_agent", agent_name="<agent>", task="...")
# skill-workflow
graph_orchestrate(action="execute_workflow", agent_name="<workflow>", task="...")
# propose-only self-evolution golden loop (intake→acquire→ADDRESSES→synthesize)
graph_orchestrate(action="golden_loop", max_fan_out=5)
```

See the companion **`autonomous-research-loop`** skill for the golden loop.

## Performance & robustness (KG-2.8 optimization pass)

Bulk ingest is bounded/throttled via `KG_BULK_INGEST=1` (keeps queue drainers,
skips analytical daemons). Hardened hot paths: O(1) id-keyed upserts + `count(n)`
fast-path + single-round-trip full-scan in the epistemic backend; `os.walk`
skip-dir pruning for the delta-hash; bounded LLM `timeout`/retries; concurrent
chat concept extraction (`KG_CHAT_CONCURRENCY`); a dedicated embedding-backfill
daemon thread (`KG_EMBED_BACKFILL`) so vector features have substrate. The KG
runs ONE consolidated daemon (`KG_DAEMON_ROLE` host/client/auto) hosted by the
gateway / `graph-os-daemon`.
