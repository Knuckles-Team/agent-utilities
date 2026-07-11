# Capability-Based Architecture Guide

> **CONCEPT:AU-KG.research.research-pipeline-runner — Capability Abstraction Layer**

## Overview

The Knowledge Graph uses a **Capability/Tool Separation** pattern to model
infrastructure and enterprise services.  Instead of encoding specific platform
names (e.g., "ServiceNow", "EAR", "WireGuard") into the ontology as
first-class classes, we define **abstract capability classes** as ontological
anchors and let concrete tools declare which capabilities they provide.

This enables:

| Benefit | Description |
|---------|-------------|
| **Tool-Agnostic Reasoning** | Business rules and policies reference `requiresCapability :DNSCapability` — not a specific tool. Swapping the tool only changes the instance, not the policy. |
| **Swappable Backends** | Adding a new DNS tool (e.g., migrating from Pi-hole to Technitium) requires zero ontology changes — just register a new instance with `providesCapability :DNSCapability`. |
| **Day-Zero DR Bootstrap** | Deployment manifests reference capabilities, enabling automated disaster recovery that selects available tools per capability. |
| **Vendor-Agnostic Governance** | Compliance assertions bind to capabilities, not product names, surviving vendor changes without governance drift. |
| **Cross-Enterprise Portability** | The same KG schema works for a homelab or an enterprise Fortune 500 — only the tool instances change. |

## Architecture

```
┌─────────────────────────────────────────────────┐
│                 ONTOLOGY LAYER                   │
│                                                  │
│   ServiceCapability (abstract root)              │
│   ├── DNSCapability                              │
│   ├── ReverseProxyCapability                     │
│   ├── VPNCapability                              │
│   ├── ContainerOrchestrationCapability           │
│   ├── MonitoringCapability                       │
│   ├── UptimeMonitoringCapability                 │
│   ├── ITSMCapability                             │
│   ├── ERPCapability                              │
│   ├── CRMCapability                              │
│   ├── EnterpriseArchitectureCapability           │
│   ├── AuthenticationCapability                   │
│   ├── SecretManagementCapability                 │
│   ├── CollaborationCapability                    │
│   ├── MailingCapability                          │
│   ├── SocialMediaCapability                      │
│   ├── SourceControlCapability                    │
│   ├── CICDCapability                             │
│   ├── DocumentManagementCapability               │
│   ├── ResearchCapability                         │
│   └── FinancialExchangeCapability                │
│                                                  │
│   VPNPurpose taxonomy:                           │
│     SecurityVPN, CorporateVPN, CustomerVPN,      │
│     PartnerVPN, SiteToSiteVPN, RemoteAccessVPN   │
│                                                  │
│   DevelopmentDomain / DevelopmentStandard         │
│     (runtime-extensible — no hardcoded domains)  │
└────────────────────┬────────────────────────────┘
                     │ owl:imports
┌────────────────────┴────────────────────────────┐
│              HYDRATION PIPELINE                  │
│                                                  │
│   CAPABILITY_REGISTRY (hydration.py)             │
│   ┌─────────────┬────────────────┬─────────────┐ │
│   │ source key  │ category       │ method      │ │
│   ├─────────────┼────────────────┼─────────────┤ │
│   │ gitlab      │ source_control │ _hydrate_…  │ │
│   │ servicenow  │ itsm           │ _hydrate_…  │ │
│   │ caddy       │ reverse_proxy  │ _hydrate_…  │ │
│   │ ...         │ ...            │ ...         │ │
│   └─────────────┴────────────────┴─────────────┘ │
│                                                  │
│   hydrate_source() resolves via registry,        │
│   not hard-coded method dispatch.                │
└────────────────────┬────────────────────────────┘
                     │ providesCapability
┌────────────────────┴────────────────────────────┐
│              OWL BRIDGE (owl_bridge.py)           │
│                                                  │
│   PROMOTABLE_NODE_TYPES includes:                │
│     service_capability, vpn_purpose,             │
│     development_domain, development_standard,    │
│     ea_fact_sheet, process_model                 │
│                                                  │
│   PROMOTABLE_EDGE_TYPES includes:                │
│     provides_capability, requires_capability,    │
│     swappable_with, has_purpose,                 │
│     applies_to_domain, works_on_domain,          │
│     must_follow                                  │
└──────────────────────────────────────────────────┘
```

## Key Properties

### Capability Linking

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `providesCapability` | (any tool instance) | `ServiceCapability` | Declares that a tool fulfills this capability |
| `requiresCapability` | (any service/function) | `ServiceCapability` | Declares a dependency on a capability |
| `swappableWith` | (tool) | (tool) | Symmetric — two tools that can substitute for each other |

### VPN Purpose

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `hasPurpose` | VPN instance | `VPNPurpose` | Classifies the VPN's business purpose |
| `requiresVPNForPurpose` | (any service) | `VPNPurpose` | Service needs VPN access for this purpose |

### Domain & Standards

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `appliesToDomain` | `DevelopmentStandard` | `DevelopmentDomain` | Links a standard to its domains |
| `worksOnDomain` | `Team` | `DevelopmentDomain` | Links a team to its domains |
| `mustFollow` | `Team` | `DevelopmentStandard` | Inferred: team → domain → standard chain |

## Adding a New Source Connector

To add a new hydration source (e.g., a new monitoring tool):

1. **Register in CAPABILITY_REGISTRY** (`hydration.py`):
   ```python
   CAPABILITY_REGISTRY["new_tool"] = {
       "category": "monitoring",  # abstract capability
       "method": "_hydrate_new_tool",
   }
   ```

2. **Add env-var check** in `get_status()`:
   ```python
   "new_tool": {
       "configured": bool(os.environ.get("NEW_TOOL_TOKEN")),
       "url": os.environ.get("NEW_TOOL_URL", ""),
   },
   ```

3. **Implement connector method** on `HydrationManager`:
   ```python
   def _hydrate_new_tool(self, engine):
       # OWL Mapping: NewToolEntity -> platform_service
       ...
   ```

4. **No changes needed** in `hydrate_source()`, `hydrate_all()`, or the
   MCP server `graph_hydrate` tool — they resolve dynamically.

## Universal Relationship Properties (CONCEPT:AU-KG.research.research-pipeline-runner)

Beyond capabilities, the core ontology provides **31 universal relationship
properties** that apply across all domains.  These use BFO root classes as
domain/range to maximize reuse across Person, Organization, SoftwareProject,
Event, and all other entity types.

### Lineage / Ancestry

The same `hasParent`/`hasChild` pattern works for biological genealogy,
software forks, organizational spin-offs, concept derivation, and event
hierarchies.

| Property | Type | Domain | Range | Description |
|----------|------|--------|-------|-------------|
| `hasParent` | ObjectProperty | Entity | Entity | Direct parent/progenitor (literal or figurative) |
| `hasChild` | ObjectProperty | Entity | Entity | Direct child/derivative (`inverseOf hasParent`) |
| `hasAncestor` | **TransitiveProperty** | Entity | Entity | Full lineage chain (reasoner infers) |
| `hasDescendant` | **TransitiveProperty** | Entity | Entity | All descendants (`inverseOf hasAncestor`) |
| `hasSibling` | **SymmetricProperty** | Entity | Entity | Shares a common parent |

**Example queries:**
```sparql
# All ancestors of a person
SELECT ?ancestor WHERE { :JohnDoe :hasAncestor ?ancestor }

# All forks of a software project (same pattern!)
SELECT ?fork WHERE { :OriginalRepo :hasDescendant ?fork . ?fork a :SoftwareProject }
```

### Participation (Entity ↔ Event)

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `participatedIn` | Entity | Event | Any entity's involvement in an event |
| `hadParticipant` | Event | Entity | Inverse of participatedIn |
| `occurredAt` | Event | Place | Where an event happened |
| `occurredDuring` | Event | Phase | When an event took place |

### Membership

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `memberOf` | IC | IC | Person→Org, Agent→Team, Server→Cluster |
| `hasMember` | IC | IC | Inverse of memberOf |

### Ownership

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `owns` | IC | Entity | Person→Asset, Org→System, Team→Project |
| `ownedBy` | Entity | IC | Inverse of owns |

### Authorship / Creation

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `createdBy` | Entity | IC | Aligned to `prov:wasAttributedTo` |
| `authorOf` | IC | Entity | Inverse of createdBy |

### Spatial Containment

| Property | Type | Domain | Range | Description |
|----------|------|--------|-------|-------------|
| `locatedIn` | **TransitiveProperty** | IC | Place | Server→Rack→DC→Region inference |
| `contains` | ObjectProperty | Place | IC | Inverse of locatedIn |

### Temporal Succession

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `succeeds` | Entity | Entity | This entity comes after another |
| `precedes` | Entity | Entity | Inverse of succeeds |

### Influence (PROV-O aligned)

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `influencedBy` | Entity | Entity | Aligned to `prov:wasInfluencedBy` |
| `influenced` | Entity | Entity | Inverse of influencedBy |

### Derivation (PROV-O aligned)

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `derivedFrom` | Entity | Entity | Aligned to `prov:wasDerivedFrom` |
| `hadDerivation` | Entity | Entity | Inverse of derivedFrom |

### Governance / Accountability

| Property | Domain | Range | Description |
|----------|--------|-------|-------------|
| `governedBy` | Entity | Entity | System→Policy, Org→Regulation |
| `approvedBy` | Entity | IC | Decision audit trail |

### Dependency / Composition

| Property | Type | Domain | Range | Description |
|----------|------|--------|-------|-------------|
| `dependsOn` | ObjectProperty | Entity | Entity | Generic dependency |
| `dependencyOf` | ObjectProperty | Entity | Entity | Inverse of dependsOn |
| `partOf` | **TransitiveProperty** | Entity | Entity | BFO-standard mereology |
| `hasPart` | **TransitiveProperty** | Entity | Entity | Inverse of partOf |

### Classification / Alignment

| Property | Type | Domain | Range | Description |
|----------|------|--------|-------|-------------|
| `classifiedAs` | ObjectProperty | Entity | GDC | Classification category |
| `alignedWith` | **SymmetricProperty** | Entity | Entity | Cross-domain same-entity (weaker than `owl:sameAs`) |

## Replaced Platform-Specific Classes

| Old Class (Removed) | Replacement | Notes |
|---------------------|-------------|-------|
| `ProcessModel` | `ProcessModel` | Generic process model from any BPM tool |
| `EAFactSheet` | `EAFactSheet` | Generic fact sheet from any EA tool |

## Ontology-Driven Tool/Agent Routing (X-4)

AU-P1-3 gave capability retrieval an engine-native filtered ANN + a durable
contextual bandit, but candidate matching was flat exact-string equality — a
tool declaring `providesCapability :DNSCapability` was invisible to a request
for the broader `:ServiceCapability`, even though the ontology already models
that as-a relationship. X-4 closes that gap: selection now combines the
engine's filtered ANN **plus ontology subsumption** (a request for capability
type `T` matches any tool whose declared type is `T` or a *narrower*
`rdfs:subClassOf` subtype) **plus** tenant/policy filters, re-ranked by the
same durable bandit, with a WHY-eligible explanation (including the concrete
subsumption path) attached to every candidate.

| Component | File |
|-----------|------|
| Dependency-free `rdfs:subClassOf` reader (no rdflib — safe on every install) | `knowledge_graph/ontology/capability_hierarchy.py` |
| Versioned capability descriptor (typed I/O, side effects, cost/latency/locality, policy/approval class, calibrated reliability) | `knowledge_graph/retrieval/capability_descriptor.py` |
| Subsumption-aware `CapabilityIndex` filtering + shared `compute_eligibility()` | `knowledge_graph/retrieval/capability_index.py` |
| Subsumption-aware engine push-down/post-filter | `knowledge_graph/retrieval/engine_capability_search.py` |
| Top-level routing entry point (`route_capability_request`, `explain_routing_eligibility`) | `graph/routing/enrichers/capability_routing.py` |

Every lower-level primitive (`CapabilityIndex`, `engine_filtered_search`) keeps
subsumption **opt-in** (`capability_hierarchy=None` by default) for exact
backward compatibility; `route_capability_request` — the X-4 entry point — has
it **on by default** via the bundled ontology's singleton
(`ontology/capability_hierarchy.get_default_hierarchy()`).

## Files Modified

| File | Change |
|------|--------|
| `ontology.ttl` | Added 31 universal relationship properties (CONCEPT:AU-KG.research.research-pipeline-runner), `owl:imports` for capability ontology |
| `ontology_capability.ttl` | **[NEW]** Capability classes, VPN taxonomy, domain/standard classes |
| `ontology_enterprise.ttl` | Replaced `ARISProcess` → `ProcessModel`, `LeanIXFactSheet` → `EAFactSheet` |
| `ontology_infrastructure.ttl` | Generalized vendor-specific comments |
| `ontology_company_infra.ttl` | Generalized `dnsRewrite` comment |
| `hydration.py` | Added `CAPABILITY_REGISTRY`, refactored `hydrate_source()` |
| `engine_ingestion.py` | Generalized job names and comments |
| `owl_bridge.py` | Added capability + universal relationship types to promotable sets (178 edge types, 159 node types) |
| `kg_server.py` | Generalized `graph_hydrate` tool description |
