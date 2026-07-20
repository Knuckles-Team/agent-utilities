# Ontology Library

> The single, documented catalog of every bundled OWL/RDF ontology in
> `agent_utilities/knowledge_graph/`. This page is the **anti-sprawl entry**: it names
> every `.ttl`, its IRI, its role, and the rules that keep the library *valid and
> connected*. It is enforced by `scripts/check_ontology.py` (CI + pre-commit) — adding
> or removing a `.ttl` without updating this page fails the gate.

## The one canonical ontology

There is exactly **one** root ontology: **`ontology.ttl`** (IRI `http://knuckles.team/kg`).
It is an upper ontology (BFO-aligned, with PROV-O / Schema.org / Dublin Core / FOAF /
SKOS / OWL-Time / BIBO / FIBO mappings) and it **`owl:imports` every domain module** —
one edge per `ontology_<module>.ttl`. The canonical file is the federation index: if a
module exists on disk, it is imported here; if it is imported here, the file exists.

> **History / why this page exists.** There used to be a second, divergent copy at
> `core/ontology.ttl` (a 57-class subset) that the background OWL reasoner
> (`maintenance/owl_closure.py`) loaded **instead** of the real 69-class root, and ~17
> domain modules that no file referenced. Because the owlready2 backend loads ontologies
> by **globbing the sibling `ontology*.ttl` files of whatever path it is given** (see
> below), pointing the reasoner at the `core/` subset silently dropped every domain
> module. That duplicate and an orphaned `ontology_infra.ttl` were deleted; the reasoner
> now loads the one canonical file; every module is imported. The gate keeps it that way.

## How ontologies are loaded (two mechanisms, kept in lockstep)

1. **Glob load (the live reasoning path).** `backends/owl/owlready2_backend.py`
   `_register_local_imports` globs `ontology*.ttl` in the directory of the file it is
   given and pre-loads them all (it strips `owl:imports`, resolving them by file
   instead). `collect_bundled_ontology_graph()` (`core/ontology_publisher.py`, KG-2.52)
   does the same glob for Stardog/Fuseki publishing. ⇒ Every `ontology_*.ttl` in this
   directory is loaded, so each must parse.
2. **`owl:imports` federation (Stardog / remote inheritance).**
   `core/ontology_loader.py` resolves `owl:imports` recursively, mapping
   `http://knuckles.team/kg/<X>` → `ontology_<X>.ttl` (and `http://knuckles.team/kg` →
   `ontology.ttl`). ⇒ Every domain module must declare its `owl:Ontology` IRI and be
   imported by the canonical file, or it is invisible to federation.

The gate enforces that these two views agree: the canonical `owl:imports` set == the
set of domain-module files on disk. No module is loaded-but-unlinked or linked-but-missing.

## Adding a new ontology (the recipe)

1. Create `agent_utilities/knowledge_graph/ontology_<name>.ttl`.
2. Declare its IRI at the top: `<http://knuckles.team/kg/<name>> a owl:Ontology ; rdfs:label "…" .`
3. Add `<http://knuckles.team/kg/<name>>` to `owl:imports` in `ontology.ttl`.
4. Add a row to the catalog below.
5. Run `python3 scripts/check_ontology.py` — it must pass.

## Anti-drift / validity gate

`scripts/check_ontology.py` enforces, for every `.ttl`:
- **Valid** — parses as Turtle; no duplicate `owl:Ontology` IRIs; the merged graph
  survives **OWL-RL closure** without error; every `shapes/*.ttl` is well-formed SHACL
  that **pyshacl** can load and run (catches SHACL syntax/breakage).
- **Connected** — every domain module declares one IRI and is imported by the canonical
  ontology (no unlinked module); every `owl:imports` in our own namespace resolves to a
  present file (no dangling/broken import).
- **Documented** — every `.ttl` on disk appears in this catalog.

## Catalog

### Root

| File | IRI | Role |
|------|-----|------|
| `ontology.ttl` | `http://knuckles.team/kg` | Canonical upper ontology; imports every domain module below. |

### Domain modules

| File | IRI | Role |
|------|-----|------|
| `ontology_a2a.ttl` | `…/kg/a2a` | Agent-to-Agent (A2A) protocol entities. |
| `ontology_concepts.ttl` | `…/kg/concepts` | Generated OKF-CIS governed concepts (`:GovernedConcept`/`:conceptId`/`:partOf` + SKOS taxonomy). Built by `scripts/build_concept_rdf.py`. |
| `ontology_action.ttl` | `…/kg/action` | Ontology action types (KG-2.42). |
| `ontology_calendar.ttl` | `…/kg/calendar` | Calendar / scheduling / OWL-Time bindings. |
| `ontology_capability.ttl` | `…/kg/capability` | Agent/system capabilities. |
| `ontology_company.ttl` | `…/kg/company` | Company / organization entities. |
| `ontology_company_infra.ttl` | `…/kg/company_infra` | Company-internal infrastructure mapping. |
| `ontology_energy_geopolitics.ttl` | `…/kg/energy_geopolitics` | Energy & geopolitics domain. |
| `ontology_enterprise.ttl` | `…/kg/enterprise` | Enterprise EA governance, ADR decision traces. |
| `ontology_government.ttl` | `…/kg/government` | Government domain extension. |
| `ontology_harness.ttl` | `…/kg/harness` | Agentic harness engineering (AHE). |
| `ontology_hr.ttl` | `…/kg/hr` | Human-resources domain. |
| `ontology_identity.ttl` | `…/kg/identity` | Identity / IdP / access entities. |
| `ontology_infrastructure.ttl` | `…/kg/infrastructure` | Runtime-agnostic infra topology (AU-OS.governance.reactive-multi-axis-budget). |
| `ontology_medical.ttl` | `…/kg/medical` | Medical domain extension. |
| `ontology_native_source_connector.ttl` | `…/kg/native_source_connector` | Generated, environment-neutral ontology for the built-in governed source-document connector. Regenerate it from the signed native connector manifest; do not edit it by hand. |
| `ontology_orchestration.ttl` | `…/kg/orchestration` | Orchestration: skill proposals, workflow/process distillation. |
| `ontology_personal.ttl` | `…/kg/personal` | Personal-knowledge domain. |
| `ontology_sdd.ttl` | `…/kg/sdd` | Spec-driven development (Requirements, Features, TestCases). |
| `ontology_sdlc_lifecycle.ttl` | `…/kg/sdlc_lifecycle` | SDLC lifecycle spine: the cross-cutting supertypes (`:Ticket`, `:PipelineRun`, `:CodeChange`, `:Validation`, `:ControlGate`, `:Approval`, `:EscalationRequest`, `:LifecycleStep`) + predicates (`:triggers`/`:specifies`/`:implements`/`:proposes`/`:triggersPipeline`/`:builds`/`:builtFrom`/`:deployedAs`/`:validatedBy`/`:resolves`) that unify the per-package incident/ticket/spec/MR/CI/image/deploy nodes into one enter-anywhere loop (`:Procedure`≡`:WorkflowDefinition`). |
| `ontology_software.ttl` | `…/kg/software` | Software: code, tests, assertions. |
| `ontology_system.ttl` | `…/kg/system` | System: interface link constraints, system-level types. |
| `ontology_trm.ttl` | `…/kg/trm` | Threat & risk-management (TRM) domain. |
| `ontology_worldview.ttl` | `…/kg/worldview` | WorldView subject-domain upper taxonomy (`:WorldViewDomain`/`:Topic`, SKOS `broader`/`narrower`) every topic classification hangs from (CONCEPT:AU-KG.enrichment.worldview-subject-ontology). |

### Federated (package-contributed) ontologies — CONCEPT:AU-KG.ontology.federation-provider-leg

Some ontology modules no longer live in this wheel: they are **contributed by fleet
agent-packages** through the `agent_utilities.ontology_providers` entry-point (the
third federation leg alongside skills and prompts). When the owning package is
installed, its `.ttl` is discovered by
`knowledge_graph/core/ontology_federation.py::resolve_provider_ontologies()` and
treated identically to a bundled module — parsed into the published TBox, pre-loaded
into the live OWL reasoner, and swept by this valid/connected/SHACL gate. The
canonical `ontology.ttl` keeps its `owl:imports` edge; when the provider is absent
the import is a tolerated superset no-op (registered in
`REGISTERED_FEDERATED_IRIS`).

| File (in provider wheel) | IRI | Provider package | Role |
|------|-----|------------------|------|
| `servicenow.ttl` (`servicenow_api/ontology/`) | `…/kg/servicenow` | `servicenow-api` | ServiceNow ITSM integration (incidents, changes, CMDB). |
| `leanix.ttl` (`leanix_agent/ontology/`) | `…/kg/leanix` | `leanix-agent` | LeanIX EAM integration. |
| `erpnext.ttl` (`erpnext_agent/ontology/`) | `…/kg/erpnext` | `erpnext-agent` | ERPNext integration. |
| `archimate.ttl` (`archimate_mcp/ontology/`) | `…/kg/archimate` | `archimate-mcp` | ArchiMate enterprise-architecture vocabulary. |
| `egeria.ttl` (`egeria_mcp/ontology/`) | `…/kg/egeria` | `egeria-mcp` | Egeria open-metadata integration (imports `…/kg/enterprise`). |
| `quant.ttl` (`emerald_exchange/ontology/`) | `…/kg/quant` | `emerald-exchange` | Quantitative finance domain. |
| `trading.ttl` (`emerald_exchange/ontology/`) | `…/kg/trading` | `emerald-exchange` | Trading: microstructure signals, market-knowledge provenance. |
| `banking.ttl` (`emerald_exchange/ontology/`) | `…/kg/banking` | `emerald-exchange` | Banking domain extension (imported by core `ontology_company.ttl`). |
| `legal.ttl` (`legal_peripherals_mcp/ontology/`) | `…/kg/legal` | `legal-peripherals-mcp` | Legal domain extension (imported by core `ontology_company.ttl`). |
| `media.ttl` (`jellyfin_mcp/ontology/`) | `…/kg/media` | `jellyfin-mcp` | Media domain. |
| `microsoft.ttl` (`microsoft_agent/ontology/`) | `…/kg/microsoft` | `microsoft-agent` | Environment-neutral Microsoft service resources and relationships. |
| `grafana.ttl` (`lgtm_mcp/ontology/`) | `…/kg/grafana` | `lgtm-mcp` | Grafana dashboards / observability assets. |
| `observability.ttl` (`lgtm_mcp/ontology/`) | `…/kg/observability` | `lgtm-mcp` | Observability (metrics/logs/traces) entities. |
| `social.ttl` (`postiz_agent/ontology/`) | `…/kg/social` | `postiz-agent` | Social / community domain. |
| `feed.ttl` (`freshrss_agent/ontology/`) | `…/kg/feed` | `freshrss-agent` | Unified RSS/Atom feed sources + items (`:FeedSource`/`:RssFeed`/`:FeedItem`). |
| `wellness.ttl` (`wger_agent/ontology/`) | `…/kg/wellness` | `wger-agent` | Wellness domain. |
| `database.ttl` (`sql_mcp/ontology/`) | `…/kg/database` | `sql-mcp` | Database/schema domain (imports `…/kg/enterprise`). |
| `tunnel.ttl` (`tunnel_manager/ontology/`) | `…/kg/tunnel` | `tunnel-manager` | Governed tunnel configuration and lifecycle resources. |

#### Certified provider source inventory

Source certification discovers providers only from their declared
`agent_utilities.ontology_providers` entry point. The table is exhaustive for the
workspace fleet: every asset is relative to that provider's declared ontology root,
and the ontology gate requires the provider and relative asset to occur together.
No connection profile, endpoint, credential, identity, or machine path belongs in
this catalog.

| Provider | Declared ontology assets |
|----------|----------------------------|
| `ansible-tower-mcp` | `ansible.ttl`, `shapes/connector.shacl.ttl` |
| `archimate-mcp` | `archimate.ttl`, `shapes/connector.shacl.ttl` |
| `archivebox-api` | `archivebox.ttl`, `shapes/connector.shacl.ttl` |
| `arr-mcp` | `arr.ttl`, `shapes/connector.shacl.ttl` |
| `atlassian-agent` | `atlassian.ttl`, `shapes/connector.shacl.ttl` |
| `audio-transcriber` | `audio.ttl`, `shapes/connector.shacl.ttl` |
| `audiobookshelf-mcp` | `audiobookshelf.ttl`, `shapes/connector.shacl.ttl` |
| `caddy-mcp` | `caddy.ttl`, `shapes/connector.shacl.ttl` |
| `camunda-mcp` | `camunda.ttl`, `shapes/connector.shacl.ttl` |
| `ciso-assistant-api` | `ciso.ttl`, `shapes/connector.shacl.ttl` |
| `clarity-api` | `clarity.ttl`, `shapes/connector.shacl.ttl` |
| `container-manager-mcp` | `container.ttl`, `shapes/connector.shacl.ttl` |
| `data-science-mcp` | `datascience.ttl`, `shapes/connector.shacl.ttl` |
| `dockerhub-api` | `dockerhub.ttl`, `shapes/connector.shacl.ttl` |
| `documentdb-mcp` | `documentdb.ttl`, `shapes/connector.shacl.ttl` |
| `egeria-mcp` | `egeria.ttl`, `shapes/connector.shacl.ttl` |
| `emerald-exchange` | `banking.ttl`, `emerald.ttl`, `quant.ttl`, `trading.ttl`, `shapes/connector.shacl.ttl` |
| `erpnext-agent` | `erpnext.ttl`, `shapes/connector.shacl.ttl` |
| `fan-manager` | `fan.ttl`, `shapes/connector.shacl.ttl` |
| `firefly-iii-mcp` | `firefly.ttl`, `shapes/connector.shacl.ttl` |
| `genius-agent` | `genius.ttl`, `shapes/connector.shacl.ttl` |
| `github-agent` | `github.ttl`, `shapes/connector.shacl.ttl` |
| `gitlab-api` | `gitlab.ttl`, `shapes/connector.shacl.ttl` |
| `gramps-mcp` | `gramps.ttl`, `shapes/connector.shacl.ttl` |
| `home-assistant-agent` | `homeassistant.ttl`, `shapes/connector.shacl.ttl` |
| `jellyfin-mcp` | `media.ttl`, `shapes/connector.shacl.ttl` |
| `jena-mcp` | `jena.ttl`, `shapes/connector.shacl.ttl` |
| `kafka-mcp` | `kafka.ttl`, `shapes/connector.shacl.ttl` |
| `keycloak-agent` | `keycloak.ttl`, `shapes/connector.shacl.ttl` |
| `langfuse-agent` | `langfuse.ttl`, `shapes/connector.shacl.ttl` |
| `leanix-agent` | `leanix.ttl`, `shapes/connector.shacl.ttl` |
| `legal-peripherals-mcp` | `compliance.ttl`, `compliance_bsa_aml.ttl`, `compliance_cfpb.ttl`, `compliance_dodd_frank.ttl`, `compliance_flsa.ttl`, `compliance_hipaa.ttl`, `compliance_llc.ttl`, `compliance_occ.ttl`, `compliance_taxation.ttl`, `domain_commercial.ttl`, `domain_corporate.ttl`, `domain_employment.ttl`, `domain_litigation.ttl`, `domain_privacy.ttl`, `domain_regulatory.ttl`, `legal.ttl`, `shapes/connector.shacl.ttl` |
| `lgtm-mcp` | `grafana.ttl`, `observability.ttl`, `shapes/connector.shacl.ttl` |
| `listmonk-api` | `listmonk.ttl`, `shapes/connector.shacl.ttl` |
| `mattermost-mcp` | `mattermost.ttl`, `shapes/connector.shacl.ttl` |
| `mealie-mcp` | `mealie.ttl`, `shapes/connector.shacl.ttl` |
| `media-downloader` | `media.ttl` (`…/kg/media-downloader`), `shapes/connector.shacl.ttl` |
| `microsoft-agent` | `microsoft.ttl`, `shapes/connector.shacl.ttl` |
| `nextcloud-agent` | `nextcloud.ttl`, `shapes/connector.shacl.ttl` |
| `objectstore-mcp` | `objectstore.ttl`, `shapes/connector.shacl.ttl` |
| `okta-agent` | `okta.ttl`, `shapes/connector.shacl.ttl` |
| `onetrust-api` | `onetrust.ttl`, `shapes/connector.shacl.ttl` |
| `openbao-mcp` | `openbao.ttl`, `shapes/connector.shacl.ttl` |
| `owncast-agent` | `owncast.ttl`, `shapes/connector.shacl.ttl` |
| `paperless-ngx-mcp` | `paperless.ttl`, `shapes/connector.shacl.ttl` |
| `plane-agent` | `plane.ttl`, `shapes/connector.shacl.ttl` |
| `portainer-agent` | `portainer.ttl`, `shapes/connector.shacl.ttl` |
| `postiz-agent` | `social.ttl`, `shapes/connector.shacl.ttl` |
| `pulselink-mcp` | `pulselink.ttl`, `shapes/connector.shacl.ttl` |
| `qbittorrent-agent` | `qbittorrent.ttl`, `shapes/connector.shacl.ttl` |
| `repository-manager` | `repository.ttl`, `shapes/connector.shacl.ttl` |
| `rom-manager` | `rom.ttl`, `shapes/connector.shacl.ttl` |
| `salesforce-agent` | `salesforce.ttl`, `shapes/connector.shacl.ttl` |
| `scholarx` | `scholarx.ttl`, `shapes/connector.shacl.ttl` |
| `searxng-mcp` | `searxng.ttl`, `shapes/connector.shacl.ttl` |
| `servicenow-api` | `servicenow.ttl`, `shapes/connector.shacl.ttl` |
| `sql-mcp` | `database.ttl`, `shapes/connector.shacl.ttl` |
| `stirlingpdf-agent` | `stirlingpdf.ttl`, `shapes/connector.shacl.ttl` |
| `systems-manager` | `systems.ttl`, `shapes/connector.shacl.ttl` |
| `technitium-dns-mcp` | `technitium.ttl`, `shapes/connector.shacl.ttl` |
| `tunnel-manager` | `tunnel.ttl`, `shapes/connector.shacl.ttl` |
| `twenty-mcp` | `twenty.ttl`, `shapes/connector.shacl.ttl` |
| `uptime-kuma-agent` | `uptimekuma.ttl`, `shapes/connector.shacl.ttl` |
| `vector-mcp` | `vector.ttl`, `shapes/connector.shacl.ttl` |
| `wger-agent` | `wellness.ttl`, `shapes/connector.shacl.ttl` |

### SHACL shapes

SHACL shapes are validation constraints, not importable ontologies (no `owl:Ontology`
IRI). They are validated for well-formedness by the gate via pyshacl.

| File | Role |
|------|------|
| `shapes/governance.shapes.ttl` | Governance SHACL shapes (the closure/validation gate in `owl_closure.py`). |
| `shapes/sdlc_lifecycle.shapes.ttl` | SDLC lifecycle REQUIRED-shape constraints (design §1.3) — consulted in DIFF mode by the enter-anywhere orchestrator so "find the gaps" is a validation query (a merged `:CodeChange` REQUIRES a `:PipelineRun`, a resolving `:Deployment` REQUIRES `:validatedBy` evidence). |
| `shapes/harness.shapes.ttl` | Harness-engineering SHACL shapes. |
| `shapes/feed.shapes.ttl` | Feed-ingestion SHACL shapes (`:FeedSource` must carry `source_system`). |
| `shapes/temporal.shapes.ttl` | Bi-temporal fact invariants (CONCEPT:AU-KG.domains.ohlcv-gap-fill): well-formed validity window + a superseded fact must have its belief window closed (KG-2.251). |
| `shapes/portfolio_intelligence.shapes.ttl` | Portfolio comparative-intelligence decision shapes (CONCEPT:AU-KG.enrichment.portfolio-intelligence): a `:Recommendation` must carry a valid adopt/reject/consolidate/migrate `:verdict` + non-empty `:rationale`; an `:Assessment` must record its `:assessmentScore`; a `:ComparisonCriterion` must declare its kind + weight. |
| `shapes/connector.shacl.ttl` (provider wheels) | Provider-owned connector resource constraints, discovered and validated through the ontology-provider entry point. |
