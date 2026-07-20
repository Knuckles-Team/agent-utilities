# Ingestion connector reference

Deep reference for `graph-ingestion-and-integration`: the full category → tool
matrix, the connector fan-out registries, the connector → OWL-entity map, the
per-package native-push matrix, and the skill-graph distill/import recipe. This
folds in the former `kg-ingest` skill's detailed sections; the parent
[`SKILL.md`](../SKILL.md) covers the day-to-day workflow and the "Full ingest"
one-call recipe.

## Unified ingestion engine — category → `graph_ingest` matrix (KG-2.7/2.8)

All content enters through ONE `IngestionEngine` (`graph_ingest` is a thin MCP
wrapper). Set `content_type` to route a path/sentinel synchronously; otherwise it
auto-classifies. Delta-skip (durable manifest) means re-ingesting unchanged sources
is a no-op.

| Category | content_type | target_path | Produces |
|---|---|---|---|
| LLM/embedding config | `config` | `config.json` | `LanguageModel`/`EmbeddingModel`/`SystemConfig` |
| Prompts | `prompt` | `agent_utilities/prompts/*.json` | `Prompt` + `Concept` (MENTIONS) |
| MCP servers | `mcp_server` | `mcp_config.json` | `Server` + `NativeTool` (PROVIDES) w/ tool descriptions |
| Skills | `skill` | a skill dir (`SKILL.md` frontmatter) | `Skill` |
| Documents | `document` | a file, dir, or URL (md/pdf/txt) | `Document{content}` + verbatim `IdeaBlock` chunks (`PART_OF`) + `Concept` (MENTIONS) — same shape regardless of submission form |
| Specs | (auto in codebase) | `**/.specify/**` | `Spec`/`ImplementationPlan` |
| Chats | `conversation` | `"chats"` sentinel | `Thread`/`Message` + per-thread `Concept` |
| Codebases | `codebase` | a repo path | `Code`/`Test`/`Feature` (CALLS/IMPLEMENTS/COVERS) |

## Connector fan-out — source → `source_sync` matrix (KG-2.9)

The connector side of a full ingest mirrors the document matrix above, but the
entrypoint is `source_sync` and the fan-out is laned. The candidate set the
`source="all"` sweep dispatches is **computed declaratively** (`sweep_all_sources`)
from three registries — connectors are never listed by hand:

| Group | Registry (data, not code) | `source` value | Lane |
|---|---|---|---|
| Native feeds | `_DELTA_HANDLERS` (`rss`, `freshrss`) | `rss`, `freshrss` | `connectors` → `worldview` (world-model gated) |
| Enterprise / tracker / IaC | `_DELTA_HANDLERS` + capability registry (`gitlab`, `leanix`, `jira`, `confluence`, `plane`, `archivebox`, …) | each source id | `connectors` |
| Ops / platform typed connectors | `_DELTA_HANDLERS` (`dockerhub`, `langfuse`, `technitium`, `tunnel_manager`, `uptime_kuma`, `home_assistant`, `twenty`) | each source id | `connectors` |
| Media / finance / doc / genealogy connectors | `_DELTA_HANDLERS` (`audiobookshelf`, `firefly_iii`, `paperless_ngx`, `gramps`) | each source id | `connectors` |
| Every `agents/*` connector | `package_manifest.PACKAGE_PRESETS`, drained by `_sync_fleet_connectors` via the generic `mcp` connector | `fleet_connectors` | `connectors` |
| Materialize extractors | `enrichment.materialize.MATERIALIZE_SOURCES` (`camunda`, `aris`, `egeria`) | each source id | `connectors` |
| Fleet capability elevation | `_sync_fleet` (slow MCP re-probe; boot/explicit only, NOT the routine sweep) | `fleet` | `connectors` |

`source_sync(source="all")` enqueues one laned `connector_sync` task **per**
candidate, so every connector (both feeds + the whole `agents/*` fleet) drains in
parallel. `fleet_connectors` iterates `PACKAGE_PRESETS`, attempts only packages
whose MCP server is registered in `mcp_config.json`, and reports unconfigured
packages as *skipped* (never errored). Each yielded record ingests through the same
`DocumentProcessor` (chunk + concept-link) as documents. Add a package to
`PACKAGE_PRESETS` and the next full ingest picks it up with no change to this
skill or reference — that is the declarative contract.

## Connector → OWL entity reference (what gets ingested + how it's modeled)

The authoritative map of every configured connector: its `source_sync` source key,
the entities it ingests, and the OWL ontology classes they map to. The KG is
OWL-native — a connector's records are not generic Documents but **typed entities**
whose `type` is promoted to its OWL class (`core/owl_bridge.py`
`PROMOTABLE_NODE_TYPES` → a class in the canonical ontology library). Three
ingestion shapes (a "maximum-ingestion" connector uses every one that applies):

- **Typed entity rebuild** (`_DELTA_HANDLERS` in `core/source_sync.py`): drains
  records and rebuilds `ingest_external_batch` entities with `type=<owl-class>` +
  relationships — first-class OWL classes the reasoner acts on.
- **Document** (`PACKAGE_PRESETS` via `fleet_connectors`, or `MCP_TOOL_PRESETS`):
  the record becomes a `:Document`+`:Chunk` (with `doc_type`), searchable but not a
  domain class. Most `agents/*` connectors land here through the fleet sweep.
- **Blob (raw bytes)** (`memory/media_store.py` `MediaStore` /
  `memory/native_ingest.py` `media_store`): the raw file/attachment/scan/media
  becomes a content-addressed `:Blob` + `:MediaAsset` node (deduped), with
  extracted text/OCR/transcript flowing into the Document shape above.

| `source` key | Connector / server | Entities ingested | OWL classes | Path |
|---|---|---|---|---|
| `jira` | atlassian-mcp | issues, assignees, epics | `:Issue` / `:Person` / `:Goal`(epic) | typed `_sync_jira` |
| `confluence` | atlassian-mcp | wiki pages | `:ConfluencePage` (`:Document`) | doc `_sync_confluence` |
| `plane` | plane-mcp | work items, projects | `:Issue` / `:SoftwareProject` | typed `_sync_plane` |
| `gitlab` | gitlab-mcp / REST | projects, files, symbols, MRs | `:Repository` / `:File` / `:Code` / `:MergeRequest` | typed `_sync_gitlab` |
| `egeria` | egeria-mcp | metadata, governance, lineage | `:ProcessModel` / `:GovernanceRule` / lineage | materialize `MATERIALIZE_SOURCES` |
| `camunda` | camunda-mcp | processes, deployments | `:BusinessProcess` / `:ProcessStep` | materialize `MATERIALIZE_SOURCES` |
| `aris` | aris-mcp | EPC process models | `:ProcessModel` / `:ArchimateElement` | materialize `MATERIALIZE_SOURCES` |
| `leanix` | leanix-agent | fact sheets (apps, IT components) | `:Application` / `:ITComponent` / `:BusinessCapability` | typed `_sync_leanix` |
| `rss` | native + scholarx | news/research feed items | `:Document` / `:ResearchInquiry` (gated) | feed `_sync_rss` |
| `freshrss` | freshrss-mcp | curated news/research | `:Document` (world-model gated) | feed `_sync_freshrss` |
| (scholarx) | scholarx-mcp | research papers | `:Document` + `:Concept` | via `rss` feed + `graph_ingest` |
| `archivebox` | archivebox-api | preserved web snapshots | `:Document` | typed `_sync_archivebox` |
| `dockerhub` | dockerhub-mcp | registry images, repos | `:Repository` / `:ContainerImage` (`contains`) | typed `_sync_dockerhub` |
| `langfuse` | langfuse-mcp | LLM traces, observations, generations | `:Trace` / `:Observation` / `:Generation` (`part_of`) | typed `_sync_langfuse` |
| `technitium` | technitium-dns-mcp | DNS zones + records | `:DnsZone` / `:DnsRecord` (`part_of`) | typed `_sync_technitium` |
| `tunnel_manager` | tunnel-manager-mcp | host inventory, tunnels | `:Host` / `:Tunnel` (`connects_via`) | typed `_sync_tunnel_manager` |
| `uptime_kuma` | uptime-(kuma-)mcp | monitors + heartbeat stats | `:UptimeMonitor` / `:HeartbeatStat` (`part_of`) | typed `_sync_uptime_kuma` |
| `home_assistant` | home-assistant-mcp | devices, entities/states | `:Device` / `:Entity` (`part_of`) | typed `_sync_home_assistant` |
| `twenty` | twenty-mcp | CRM people, companies, opportunities | `:Person` / `:Company` / `:Opportunity` (`member_of`/`part_of`) | typed `_sync_twenty` |
| `audiobookshelf` | audiobookshelf-mcp | libraries, books/audiobooks, authors | `:Library` / `:Book` / `:Author` (`part_of`/`authored_by`) | typed `_sync_audiobookshelf` |
| `firefly_iii` | firefly-iii-mcp | accounts, transactions, budgets | `:Account` / `:Transaction` / `:Budget` (`part_of`/`member_of`) | typed `_sync_firefly_iii` |
| `paperless_ngx` | paperless-ngx-mcp | documents, correspondents, tags | `:Document` / `:Correspondent` / `:Tag` (`member_of`/`tagged_with`) | typed `_sync_paperless_ngx` |
| `gramps` | gramps-mcp | people, families, events | `:Person` / `:Family` / `:Event` (`member_of`/`part_of`) | typed `_sync_gramps` |

The ops/platform connectors above are **MCP-configured**: each ingests only when its
`*-mcp` server is registered in `mcp_config.json` (`_MCP_TRACKER_SERVERS`), so the
`source="all"` sweep keeps it as a candidate when reachable and drops it (skipped,
never errored) otherwise. Trigger one directly with
`source_sync(source="<key>", mode="delta")`.

**Available via MCP but ingestion-optional** (a fixed/configured `*-mcp`, no
dedicated handler — ingest only when there is clear knowledge value, else reach
live): `owncast`, `mealie` (recipe docs preset `mealie-recipes`), `searxng` (search
preset `searxng-search`), `lgtm`, `nextcloud` (folder preset `nextcloud-files`),
`arr`. These surface through the declarative fleet sweep / `MCP_TOOL_PRESETS` as
`:Document`s, not typed domain classes.

Still declarative: adding a typed connector = a `_DELTA_HANDLER` + its
`MCP_TOOL_PRESET` + the `PROMOTABLE_NODE_TYPES` entries + the OWL class in the
canonical ontology — then the next `source="all"` sweep picks it up. Keep this
table in lockstep with `_DELTA_HANDLERS`.

## Native connector push — package-side ingestion (nodes + documents + blobs)

Complementing the hub-side **pull** above (`source_sync` drains a connector from the
hub), every `agents/*` connector also ships **native push**: its OWN code writes its
data into the ONE engine as it works, via the shared primitive
`agent_utilities/knowledge_graph/memory/native_ingest.py`. This is the
"maximum ingestion" bar — a connector pushes in **every modality that applies**:

| Primitive | Modality | Produces | Package module |
|---|---|---|---|
| `native_ingest.ingest_entities(entities, rels, source, domain)` | typed nodes | OWL `:Class` nodes + links | `<pkg>/kg_ingest.py` (thin mapper) |
| `native_ingest.ingest_documents(docs, source, domain)` | documents | `:Document` (text + `source_uri`; hub chunks/embeds) | `<pkg>/kg_ingest.py` |
| `native_ingest.media_store().store_media(bytes, …)` | blob | `:Blob` + `:MediaAsset` (content-addressed, deduped) | `<pkg>/kg_media.py` |

All three ride the **lightweight** `GraphComputeEngine()._client` (the heavy
`IntelligenceGraphEngine` is not constructible in a connector). Every entry point
is dependency-/engine-guarded — it **no-ops** with no reachable engine, so a
connector runs with zero KG infra. Wired default-on into the package's
fetch/download flow + surfaced on an MCP tool. Node ids:
`<domain>:<class>:<externalId>`; `type` matches the package's `ontology_providers`
`.ttl`. Reach the engine by setting `GRAPH_SERVICE_ENDPOINTS` to the engine's
TCP address and port (e.g. `<engine-host>:9100`).

**Reference implementations (LIVE-verified):** `media-downloader/kg_media.py` (a
downloaded video → `:MediaAsset` blob, fetch-back byte-identical) and
`gitlab-api/kg_ingest.py` (projects → `:Project`/`:GitLabGroup` typed nodes).

### Per-package native ingestion — the "maximum ingestion" matrix

Enterprise record-sources do all three (typed nodes + KB/notes documents +
attachment blobs); file/media packages are document+blob heavy:

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

## Infrastructure topology ingestion

When ingesting the workspace, also ingest infrastructure state so the KG is
hydrated with the physical and virtual topology:

- **Inventory file** — `$XDG_CONFIG_HOME/agent-utilities/inventory.yaml`
  (default `$HOME/.config/agent-utilities/inventory.yaml`) → a `HardwareNode`
  per host (`hostname`, `ip_address`, `group`, `status`, ssh fields, extended
  metadata) + `HAS_INTERFACE`/`CONNECTS_VIA` edges.
- **Ontology files** — `agent_utilities/knowledge_graph/ontology.ttl` and
  `ontology_infrastructure.ttl` as ingestion targets, providing the formal
  BFO-aligned class hierarchy for infrastructure nodes.
- **Workflow catalog** — `agent_utilities/workflows/catalog.yaml` →
  `WorkflowDefinition` nodes with `HAS_STEP`/`REQUIRES_TOOL` edges.
- **Topology snapshots** — `$XDG_DATA_HOME/agent-utilities/topology/{topology,
  service_map,network_map}.json` (the XDG data directory, per the XDG Base
  Directory spec) → `Container`/`ContainerStack`/`NetworkSubnet`/`DNSRewrite`/
  `ReverseProxy`/`ObservabilityStack` nodes.
- **DNS rewrites** — `technitium-dns-mcp` → `list_records` → `DNSRecord` nodes
  linked to their `PlatformService`.
- **Container state (live)** — `container-manager-mcp`/`portainer-mcp` → live
  `Container`/`ContainerStack` nodes with `RUNS_ON`/`BELONGS_TO_STACK` edges.

Default ingestion target list appended for a full workspace ingestion (XDG
config/data directories, per the XDG Base Directory spec):
```
$XDG_CONFIG_HOME/agent-utilities/inventory.yaml
$XDG_CONFIG_HOME/agent-utilities/mcp_config.json
$XDG_CONFIG_HOME/agent-utilities/config.json
$XDG_DATA_HOME/agent-utilities/topology/
agent_utilities/knowledge_graph/ontology.ttl
agent_utilities/knowledge_graph/ontology_infrastructure.ttl
agent_utilities/workflows/catalog.yaml
```

## Skill-graph packages — distill OUT / import back (KG-2.7 / AHE-3.9)

**Auto-ingested by default.** When the KG is reachable, the `knowledge_base`
pipeline phase auto-ingests all packaged skill-graphs and the universal-skills
workflow corpus — delta-skipped, so only the first run is heavy. Disable on
constrained installs via `KG_AUTO_INGEST_SKILLS=false`.

The KG is the source of truth; a skill-graph is a versioned, round-trippable
projection of a KG subgraph. Two symmetric `graph_ingest` actions:

- **Distill (export):** `graph_ingest(action="distill", target_path="<out dir>",
  corpus_name="<seed node id>" OR description="<semantic query>", max_depth=2)`.
  Walks a coherent subgraph → a `reference/` markdown tree + `kg_manifest.json`
  (node ids, edges, ontology, snapshot), consumable verbatim by
  `skill-graph-builder`. Community detection → folders; edges → TOC nesting +
  inline cross-links. Pass `content_type="workflow"` to instead distill a
  graph-native skill-workflow (a procedure step-DAG, `PRECEDES` → `depends_on`).
- **Import (round-trip):** `graph_ingest(action="import_pack",
  target_path="<skill-graph dir>", corpus_name="dedup")`. Reads
  `kg_manifest.json` and reconstructs the subgraph, preserving original node ids
  + edges, so a curated package merges into another brain.
  `corpus_name="dedup"` runs the IdeaBlock dedup-merge.

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

OWL cross-category relationship expectations: `MENTIONS` (chats/docs→Concept),
`RELATES_TO`/`REALIZES` (Concept→Code, via the embedding-backed
`link_concepts_to_code` once embeddings are backfilled), `PROVIDES`
(Server→NativeTool), `CONTAINS` (Thread→Message), `IMPLEMENTS`/`CALLS`/
`DEPENDS_ON` (Code), `ADDRESSES`/`ADDRESSED_BY` (research source→topic).

## Performance & robustness (KG-2.8 optimization pass)

Bulk ingest is bounded/throttled via `KG_BULK_INGEST=1` (keeps queue drainers,
skips analytical daemons). Hardened hot paths: O(1) id-keyed upserts +
`count(n)` fast-path + single-round-trip full-scan in the epistemic backend;
`os.walk` skip-dir pruning for the delta-hash; bounded LLM `timeout`/retries;
concurrent chat concept extraction (`KG_CHAT_CONCURRENCY`); a dedicated
embedding-backfill daemon thread (`KG_EMBED_BACKFILL`) so vector features have
substrate. The KG runs ONE consolidated daemon (`KG_DAEMON_ROLE` host/client/auto)
hosted by the gateway / `graph-os-daemon`.

## Declarative full-ingest manifest

The machine-readable family → tool → lane mapping (`ingest_manifest.yaml`, kept
alongside this reference) is a companion for a driver to consume: each family
(`codebase_documents`, `native_rss`, `freshrss`, `fleet_connectors`,
`native_push`) plus the single `all_connectors_trigger` (`source_sync(source=
"all", mode="full")`, fanning out via `sweep_all_sources`). See
[`ingest_manifest.yaml`](ingest_manifest.yaml) for the full declarative form.
