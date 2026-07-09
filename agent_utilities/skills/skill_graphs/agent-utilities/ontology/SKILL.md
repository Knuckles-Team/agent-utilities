---
name: agent-utilities-ontology
skill_type: graph
description: >-
  OWL ontology reference for agent-utilities. Covers the core ontology,
  16 domain ontologies, BFO alignment, and SPARQL query patterns.
tags: [owl, ontology, rdf, sparql, bfo, knowledge-graph]
---

# agent-utilities OWL Ontology Guide

## 📋 Ontology Catalog

All ontologies use namespace `@prefix : <http://knuckles.team/kg#>` and are
aligned to [BFO (Basic Formal Ontology)](https://basic-formal-ontology.org/).

| File | Domain | Key Classes |
|------|--------|-------------|
| `ontology.ttl` | **Core** | Person, Organization, Event, Place, Document, Concept |
| `ontology_hr.ttl` | HR/Workforce | Employee, Department, CompensationBand, PerformanceReview, OKR |
| `ontology_legal.ttl` | Legal | LegalMatter, CaseLaw, Statute, Contract, ContractClause |
| `ontology_banking.ttl` | Banking | BankAccount, KYCRecord, PaymentMessage, CreditRiskAssessment |
| `ontology_enterprise.ttl` | Enterprise Architecture | ArchiMateElement, ArchitectureDecisionRecord, Policy |
| `ontology_infrastructure.ttl` | Infrastructure | Host, Container, Network, Volume, Service |
| `ontology_quant.ttl` | Quantitative Finance | Strategy, Signal, Portfolio, BacktestResult |
| `ontology_medical.ttl` | Medical/Health | Patient, Diagnosis, Treatment, MedicalRecord |
| `ontology_wellness.ttl` | Wellness | WellnessGoal, ExerciseSession, NutritionLog |
| `ontology_social.ttl` | Social/Media | SocialPost, Platform, Engagement, Audience |
| `ontology_personal.ttl` | Personal | PersonalGoal, Habit, JournalEntry |
| `ontology_government.ttl` | Government | Agency, Regulation, PublicService |
| `ontology_energy_geopolitics.ttl` | Energy/Geopolitics | EnergySource, GeopoliticalEntity |
| `ontology_media.ttl` | Media/Content | MediaAsset, ContentPipeline, Distribution |
| `ontology_sdd.ttl` | SDD Pipeline | DesignDocument, Specification, Implementation |
| `ontology_company.ttl` | Company Operations | Company, StrategicGoal, KPI, AgentDepartment |
| `ontology_company_infra.ttl` | Company Infrastructure | CompanySoftware, DeploymentBlueprint |

## 🔗 Core Classes (BFO Alignment)

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

## 📝 Writing Ontology Extensions

### Template for new domain ontology
```turtle
@prefix : <http://knuckles.team/kg#> .
@prefix bfo: <http://purl.obolibrary.org/obo/BFO_> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<http://knuckles.team/kg/my_domain> a owl:Ontology ;
    rdfs:label "My Domain Ontology" ;
    rdfs:comment "Description of what this ontology covers." ;
    owl:imports <http://knuckles.team/kg> .

:MyClass a owl:Class ;
    rdfs:label "My Class" ;
    rdfs:comment "Explanation of what this class represents." ;
    rdfs:subClassOf bfo:0000004 .  # IndependentContinuant

:myProperty a owl:ObjectProperty ;
    rdfs:label "my property" ;
    rdfs:domain :MyClass ;
    rdfs:range :OtherClass .
```

### Key Patterns
- **Transitive properties**: Use `owl:TransitiveProperty` for hierarchies
  (e.g., `reportsTo`, `cascadesTo`, `departmentPartOf`)
- **Symmetric properties**: Use `owl:SymmetricProperty` for bidirectional
  (e.g., `conflictsWithParty`)
- **Inverse properties**: Use `owl:inverseOf` for pairs
  (e.g., `nostroOf` / `vostroOf`)

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
