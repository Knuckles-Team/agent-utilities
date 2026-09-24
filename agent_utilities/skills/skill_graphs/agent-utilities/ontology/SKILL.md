---
name: agent-utilities-ontology
skill_type: graph
description: >-
  Where the platform ontology lives and how to extend it. agent-utilities owns no
  ontology: epistemic-graph owns the ontology lifecycle, SHACL, RDF and OWL, and
  connectors ship their own vocabulary as SDK-certified packs. Includes SPARQL
  query patterns for the engine.
tags: [owl, ontology, rdf, sparql, bfo, knowledge-graph]
---
# Ontology — owned by epistemic-graph

agent-utilities does not own, ship or reason over ontologies. The authority is split:

| What | Owner | Where |
|---|---|---|
| Core and domain modules (foundation, capability, archimate, enterprise, company, hr, infrastructure, …) | epistemic-graph | `crates/eg-core/ontology/<module>-v<N>.ttl`, registered in `crates/eg-core/src/graph/schema_sources.rs::core_specs()` as the engine-owned source `core:<module>@<N>` |
| Connector vocabulary and shapes | the connector package | authored in the connector, certified by agent-connector-sdk into a digest-pinned pack, committed by EG `ConnectorPack` |
| Validation, reasoning, schema lifecycle | epistemic-graph | SHACL, OWL tableau + RL/property saturation, `GraphSchema` |

To add a class or shape: edit the EG core source (engine change, EG lane) or the
connector's pack — never a `.ttl` in agent-utilities, and never rdflib, pyshacl,
owlrl or owlready2 in agent-utilities code or tests. Every IRI stays in the
`http://knuckles.team/kg#` namespace, aligned to BFO.

## 🔗 Core Classes (BFO alignment, as served by EG)

```
BFO:IndependentContinuant (things that exist on their own)
├── Person → Employee, Contractor
├── Organization → Company, Department, CorrespondentBank
├── Place → Jurisdiction
└── Agent → SpecialistAgent

BFO:Process (things that happen over time)
├── Event → PerformanceReview, LegalMatter, HiringPipeline
└── Procedure → ComplianceAudit, PayrollRun

BFO:GenericallyDependentContinuant (information entities)
├── Document → Contract, Article, ResearchPaper
├── Concept → KBConcept, StrategicGoal
├── Credential → License, Certification
└── Statute → FederalStatute, StateStatute
```

## 🔍 SPARQL Query Examples

```sparql
# Find all employees in a department
SELECT ?emp ?name WHERE {
  ?emp a :Employee ;
       :assignedToDepartment ?dept ;
       rdfs:label ?name .
  ?dept rdfs:label "Engineering" .
}

# Find OKR cascade chain
SELECT ?okr ?parent WHERE {
  ?okr :cascadesTo+ ?parent .
  ?parent a :OKR .
}

# Cross-domain: Find employees with expired credentials
SELECT ?emp ?cred ?expiry WHERE {
  ?emp :holdsCredential ?cred .
  ?cred :credentialExpiry ?expiry .
  FILTER (?expiry < NOW())
}
```
