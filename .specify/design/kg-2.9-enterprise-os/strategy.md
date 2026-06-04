# KG‑2.9 — Enterprise OS on the Epistemic Graph

> Make the enterprise itself *native* to the graph: one ontology over systems
> (ERPNext, ServiceNow, LeanIX/ArchiMate, Jira/Plane, Grafana, GitLab…),
> infrastructure (servers via tunnel‑manager `inventory.yaml`), services (Docker
> services/containers via container‑manager/Portainer), data (universal SQL/
> GraphQL connectors), people, and code — with epistemic‑graph as the fast
> reasoning/cache layer. Extends **agent‑os** and reuses the **KG‑2.8** enrichment
> framework (extractor → typed nodes → embed → cross‑link → reason).

## What already exists (reuse, don't rebuild)
- **Connectors**: `agents/erpnext-agent`, `leanix-agent`, `plane-agent`,
  `servicenow-api`, `gitlab-api`, `portainer-agent`, `container-manager-mcp`,
  `tunnel-manager`, `technitium-dns-mcp`.
- **Enterprise ontology classes** (in `models/schema_definition.py`):
  BusinessRole, ApplicationComponent, BusinessProcess, ProcessFlow/Step,
  Organization, BusinessDivision, Role, User, Server, BladeServer, HardwareNode,
  PlatformService, DNSService, MCPServer.
- **DataConnectorProtocol + registry** (`protocols/data_connector.py`) — fetch +
  fallback (read‑oriented today).
- **Infra ingestion** — `infrastructure-orchestrator` + `agent-os-deployment`
  skills already discover hardware/containers/DNS and ingest topology to the KG.
- **Compute/cache** — epistemic‑graph (sharded via ShardRouter/ConnectionPool) +
  tiered backend (L1 epistemic‑graph / L3 pggraph) + the single `GraphBackend`
  interface.

## The gap (what KG‑2.9 builds)
1. **One enterprise ontology** unifying the above into a connected graph (not silos):
   `Person/Employee ─ASSIGNED_TO→ Ticket ─AFFECTS→ Service ─RUNS_ON→ Server`,
   `Application ─SUPPORTS→ BusinessProcess`, `Service ─MONITORED_BY→ Dashboard`,
   `Application ─IMPLEMENTED_BY→ Code`, `Change ─MODIFIES→ CI`, etc. Add
   `ontology_enterprise_os.ttl` (ArchiMate + ITSM + DevOps + observability) and
   promote the edges in `owl_bridge` for transitive reasoning.
2. **Per‑system EnrichmentExtractors** (the KG‑2.8 pattern, one per source) that pull
   from the EXISTING connectors → typed nodes + edges, embedded + cross‑linked,
   hash/delta‑incremental:
   - **Infra**: `inventory.yaml` (tunnel‑manager) → Server/HardwareNode;
     container‑manager/Portainer → Service/Container `RUNS_ON` Server; DNS →
     Endpoint. *(Concrete, foundational — good first slice.)*
   - **ITSM**: ServiceNow → Incident/Change/CI; Plane/Jira → Issue/Epic; link to
     Service/Application/Person.
   - **EA**: LeanIX → Application/Capability/ArchiMate → ApplicationComponent/
     BusinessProcess.
   - **ERP**: ERPNext (Frappe REST) → Employee/Customer/Order/Invoice/CostCenter.
   - **Observability**: Grafana → Dashboard/Panel/Alert; Prometheus → Metric →
     `MONITORS` Service.
3. **Universal DataConnector (read/write/update, multi‑DB + GraphQL)** — upgrade
   `DataConnectorProtocol` to a driver‑based connector covering postgres/mysql/
   mssql/oracle/sqlite/mongo + GraphQL endpoints, with **schema introspection
   into the KG** (`DataSource/Table/Column/GraphQLType` + `FOREIGN_KEY`/`HAS_FIELD`
   edges). The KG stores *how to use each source* (schema + relationships +
   sample queries), so agents can plan queries. Writes/updates go through the
   connector to the system of record; reads can be served from cache.
4. **epistemic‑graph as the fast cache/reasoning layer** — source‑of‑truth stays in
   each system; the enterprise graph (entities + relationships + hot/derived data)
   is cached in epistemic‑graph for sub‑ms traversal/reasoning. Enables
   impact/blast‑radius queries ("if Server X fails → which Services, Apps,
   BusinessProcesses, open Changes, and on‑call People are affected?") as a single
   graph traversal (`get_blast_radius` already exists in the engine), and
   read‑through caching of expensive cross‑system joins.
5. **Scale (100K+ employees)** — sharded epistemic‑graph + partitioned pggraph;
   delta‑sync connectors (only changed records by updated‑at/hash); batched
   embeddings; the single `GraphBackend` interface + tiered backend already
   support horizontal scale. Coordination/eventing on `__bus__`; durable knowledge
   in the `kg` tenant; ephemeral sync scratch in per‑job tenants (KG‑2.8 planes).

## How it composes with KG‑2.8
Same machinery, new sources: enterprise entities get capability cards, concepts,
patterns where relevant, and cross‑links (e.g. a ServiceNow Incident `RELATES_TO`
the Code/Service that caused it; a research Concept `REALIZES`‑able into an
Application). `find_related(goal)` and `what_specs_could_we_build()` then span the
whole enterprise — business + infra + code + research.

## Phased plan
- **9.0 (foundational):** `ontology_enterprise_os.ttl` + unify existing schema
  classes; ingest `inventory.yaml` → Server/HardwareNode; Docker services →
  Service `RUNS_ON` Server; blast‑radius query over infra.
- **9.1:** universal DataConnector (multi‑DB SQL/GraphQL read/write/update) +
  schema‑introspection‑to‑KG; epistemic‑graph read‑through cache.
- **9.2:** per‑system extractors (ServiceNow, Plane, LeanIX, ERPNext, Grafana)
  via existing connectors → enterprise graph + cross‑links.
- **9.3:** enterprise reasoning/agents (impact analysis, ITSM automation,
  capability/cost rollups) + delta‑sync at scale.
