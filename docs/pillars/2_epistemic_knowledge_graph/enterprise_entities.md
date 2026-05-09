# Universal Enterprise Entities (CONCEPT:KG-2.51)

> The entities that **every** 100,000+ employee enterprise needs, regardless of sector. These form the cross-domain foundation of the Company Brain.

---

## Overview

While the Knowledge Graph already models agents, tools, skills, and domain-specific concepts (finance, medical, legal), an enterprise of 100K+ employees requires a shared organizational substrate. These entities are **sector-agnostic** — a bank, hospital, law firm, and government agency all need workforce management, supply chain tracking, governance frameworks, and communication primitives.

All entities align to the **BFO upper ontology (ISO 21838-2)** and map to established industry standards.

---

## 1. Human Capital & Workforce Ontology

### Entities

| Entity | OWL Class | BFO Alignment | Standard Alignment | Description |
|--------|-----------|---------------|-------------------|-------------|
| **Employee** | `:Employee` | `bfo:0000004` (IC) | foaf:Person, HR-XML | A person employed by the organization. Extends `:Person` with employment metadata. |
| **Department** | `:Department` | `bfo:0000004` (IC) | W3C ORG `org:OrganizationalUnit` | An organizational unit within the enterprise hierarchy. |
| **Position** | `:PositionRole` | `bfo:0000020` (SDC) | HR-XML `PositionOpening` | A time-bounded role a person occupies. Specifically dependent on the employee. |
| **Competency** | `:Competency` | `bfo:0000031` (GDC) | SFIA Framework | A skill or knowledge area with proficiency levels (1-5). |
| **Credential** | `:Credential` | `bfo:0000031` (GDC) | Open Badges, HR-XML | A certification, license, or qualification with expiry tracking. |
| **Performance Review** | `:PerformanceReview` | `bfo:0000015` (Process) | — | A periodic evaluation event linking Employee → Reviewer → Ratings. |
| **Compensation Band** | `:CompensationBand` | `bfo:0000031` (GDC) | — | A salary range tied to position levels and geographies. |
| **OKR** | `:OKR` | `bfo:0000020` (SDC) | — | An Objective & Key Result, cascading from org → department → individual. |
| **Hiring Pipeline** | `:HiringPipeline` | `bfo:0000015` (Process) | — | Requisition → sourcing → screening → offer → onboarding lifecycle. |

### Relationships

| Relationship | OWL Type | Domain → Range | Why Critical |
|-------------|----------|---------------|-------------|
| `reportsTo` | Transitive | Employee → Employee | Org hierarchy traversal for escalation routing. If A reportsTo B and B reportsTo C, A transitively reportsTo C. |
| `hasCompetency` | Object | Employee → Competency | Skills matrix for workforce planning and agent routing. |
| `holdsCredential` | Object | Employee → Credential | Authorization tracking — who is certified to do what. |
| `assignedToDepartment` | Object | Employee → Department | Organizational membership for tenant scoping. |
| `reviewedIn` | Object | Employee → PerformanceReview | Links employees to their evaluation history. |
| `compensatedAt` | Object | PositionRole → CompensationBand | Compensation governance. |
| `certifiedFor` | Object | Credential → Procedure | Maps certifications to authorized activities. |
| `cascadesTo` | Transitive | OKR → OKR | Objective hierarchy: org OKR → dept OKR → individual OKR. |

### Enterprise Value

- **Succession planning**: Traverse `reportsTo` chains + `hasCompetency` to identify internal candidates
- **Skills gap analysis**: Compare required `Competency` for open positions vs. existing workforce
- **Compliance**: Ensure employees `holdsCredential` for activities they perform
- **Agent augmentation**: AI agents query the org chart to route tasks to the right human

---

## 2. Supply Chain & Asset Ontology

### Entities

| Entity | OWL Class | BFO Alignment | Standard Alignment | Description |
|--------|-----------|---------------|-------------------|-------------|
| **Vendor** | `:Vendor` | `bfo:0000004` (IC) | schema:Organization | An external supplier or service provider. |
| **Contract** | `:Contract` | `bfo:0000031` (GDC) | LKIF-Core | A legal agreement between the organization and a vendor/partner. |
| **SLA** | `:ServiceLevelAgreement` | `bfo:0000031` (GDC) | ITIL | Performance targets within a contract. |
| **Procurement Order** | `:ProcurementOrder` | `bfo:0000015` (Process) | — | RFP → evaluation → contract → delivery → payment lifecycle. |
| **Asset** | `:Asset` | `bfo:0000004` (IC) | ISO 55000 | A physical or digital asset with lifecycle tracking. |
| **Location** | `:Location` | `bfo:0000004` (IC) | schema:Place | A physical site, warehouse, data center, or office. |

### Relationships

| Relationship | OWL Type | Domain → Range | Why Critical |
|-------------|----------|---------------|-------------|
| `suppliedBy` | Object | Asset → Vendor | Vendor risk assessment and supply chain tracing. |
| `governedByContract` | Object | Vendor → Contract | Contract lifecycle management. |
| `hasSLA` | Object | Contract → SLA | SLA compliance monitoring. |
| `locatedAt` | Object | Asset → Location | Physical asset tracking. |
| `procuredVia` | Object | Asset → ProcurementOrder | Procurement audit trail. |

---

## 3. Governance & Compliance Ontology

### Entities

| Entity | OWL Class | BFO Alignment | Standard Alignment | Description |
|--------|-----------|---------------|-------------------|-------------|
| **Compliance Rule** | `:ComplianceRule` | `bfo:0000031` (GDC) | LKIF-Core, LegalRuleML | An enforceable regulation or internal policy requirement. |
| **Audit Finding** | `:AuditFinding` | `bfo:0000015` (Process) | COBIT, ITIL | A finding from an internal or external audit. |
| **Risk Register Entry** | `:RiskRegisterEntry` | `bfo:0000031` (GDC) | ISO 31000 | An identified risk with likelihood × impact scoring. Extends existing `RiskAssessment`. |
| **Policy Version** | `:PolicyVersion` | `bfo:0000031` (GDC) | — | An immutable snapshot of a policy document with effective dates. |
| **Control Objective** | `:ControlObjective` | `bfo:0000031` (GDC) | COBIT, NIST 800-53 | A measurable control requirement (e.g., "All PII must be encrypted at rest"). |
| **Compliance Certificate** | `:ComplianceCertificate` | `bfo:0000031` (GDC) | — | A formal attestation (SOC2, ISO 27001, HIPAA, FedRAMP). |

### Relationships

| Relationship | OWL Type | Domain → Range | Why Critical |
|-------------|----------|---------------|-------------|
| `obligatedBy` | Object | Organization → ComplianceRule | Maps regulations to the entities they apply to. |
| `supersedes` | Transitive | PolicyVersion → PolicyVersion | Policy version chains — find the current version by following the chain. |
| `implementsControl` | Object | Procedure → ControlObjective | Maps activities to the controls they satisfy. |
| `auditFound` | Object | AuditFinding → ControlObjective | Links findings to the control that was violated. |
| `mitigatesRisk` | Object | ControlObjective → RiskRegisterEntry | Links controls to the risks they reduce. |
| `certifiedBy` | Object | Organization → ComplianceCertificate | Tracks certification status. |

### Compliance Lifecycle

```
Regulation → Control Objective → Control Activity → Evidence Collection → Assessment → Finding → Remediation
```

Each step is a first-class KG entity with temporal validity, jurisdiction scoping, and automated evidence mapping.

---

## 4. Communication & Decision Ontology

### Entities

| Entity | OWL Class | BFO Alignment | Description |
|--------|-----------|---------------|-------------|
| **Meeting** | `:Meeting` | `bfo:0000015` (Process) | A scheduled collaboration event with attendees, agenda, and outcomes. |
| **Decision Record** | `:DecisionRecord` | subClassOf `:Decision` | An Architecture Decision Record (ADR) with context, options, and chosen outcome. |
| **Escalation** | `:Escalation` | `bfo:0000015` (Process) | An escalation event routing an issue up the `reportsTo` chain. |
| **Approval Chain** | `:ApprovalChain` | `bfo:0000015` (Process) | A multi-step approval workflow with sequential or parallel gates. |

### Relationships

| Relationship | OWL Type | Domain → Range | Why Critical |
|-------------|----------|---------------|-------------|
| `decidedIn` | Object | DecisionRecord → Meeting | Links decisions to the meeting where they were made. |
| `escalatesTo` | Transitive | Escalation → Employee | Routes escalations up the org hierarchy. |
| `requiresApprovalFrom` | Object | ApprovalChain → Role | Defines who must approve at each gate. |
| `conflictOfInterest` | Symmetric | Person → Person | Chinese walls in banking/law — bidirectional by construction. |

---

## 5. Integration with Existing Architecture

### Company Brain Alignment

All enterprise entities integrate with the existing Company Brain infrastructure:

| Company Brain Primitive | Enterprise Integration |
|------------------------|----------------------|
| **TenancyManager** | Departments map to tenants; `assignedToDepartment` ↔ tenant membership |
| **ProvenanceTracker** | Every employee action generates provenance (human or AI equally) |
| **DataLevelPermissions** | Compensation data is RESTRICTED; org charts are INTERNAL |
| **ConflictResolver** | HR data from HRIS wins over agent inference (SOURCE_AUTHORITY_WINS) |
| **EventStreamIngester** | HRIS webhooks, Slack messages, Jira updates → graph mutations |

### OWL Reasoning Benefits

| Reasoning Pattern | Enterprise Application |
|------------------|----------------------|
| **Transitive `reportsTo`** | "Find everyone who reports to the VP of Engineering" — automatic chain traversal |
| **Transitive `supersedes`** | "What is the current version of the Data Retention Policy?" — follow chain to head |
| **Symmetric `conflictOfInterest`** | "Can analyst Jane review a trade involving client X?" — bidirectional check |
| **Transitive `propagatesRiskTo`** | "A vendor breach propagates risk through the supply chain" — automatic upstream inference |

---

## See Also

- [Company Brain Index](company_brain/00_index.md) — Multi-tenancy, concurrency, provenance
- [Ontology & Epistemics](KG-2.2-Ontology_&_Epistemics.md) — OWL class definitions
- [Risk Scoring Ontology](KG-2.7-Risk_Scoring_Ontology.md) — Domain-agnostic risk framework
