# OWL Ontology

> **Own the ontology. Own the perspective.** The OWL ontology is what transforms raw data into company-specific knowledge.

---

## What the Ontology Does

The 74KB OWL ontology (`ontology.ttl`) defines the **company-specific perspective** that transforms raw data into organizational knowledge. It is not a static schema — the `OWLBridge` runs active reasoning cycles that discover new facts through logical inference.

### Standards Alignment

| Standard | Purpose | How We Use It |
|:---------|:--------|:--------------|
| **BFO** (Basic Formal Ontology) | Upper ontology for foundational categories | Root classes for Entity, Process, Quality |
| **PROV-O** (W3C Provenance) | Attribution and derivation tracking | `wasGeneratedBy`, `wasDerivedFrom`, `wasAttributedTo` edges |
| **SKOS** (Knowledge Organization) | Taxonomic hierarchies | `broader`, `narrower`, `relatedConcept` for concept navigation |
| **FIBO** (Financial Industry) | Financial domain concepts | `FinancialInstrument`, `Account`, `Transaction` classes |

---

## Reasoning Cycle: Promote → Reason → Downfeed

The `OWLBridge` operates a three-phase cycle:

### 1. Promote (LPG → OWL)

High-importance nodes from the property graph (NetworkX/LadybugDB) are promoted into the OWL ontology as named individuals:

```
NetworkX Node: {"id": "customer:001", "risk_level": "high", "importance_score": 0.9}
    ↓ promote
OWL Individual: :Customer001 rdf:type :Customer ; :hasRiskLevel "high"
```

### 2. Reason (OWL Inferencing)

The OWL reasoner applies:
- **Transitive closure** — If A `dependsOn` B and B `dependsOn` C, then A `dependsOn` C
- **Symmetric properties** — If A `relatedTo` B, then B `relatedTo` A
- **RDFS+ entailment** — Subclass hierarchies propagate properties downward
- **SKOS hierarchy** — `broader`/`narrower` relationships enable category navigation

### 3. Downfeed (OWL → LPG)

Newly inferred facts are written back to the property graph as edges:

```
OWL Inference: :Customer001 :dependsOn :Service002 (via transitive closure)
    ↓ downfeed
NetworkX Edge: ("customer:001", "service:002", "depends_on")
    + {"inferred": true, "inference_type": "transitive_closure"}
```

---

## Ontological Lensing

The same underlying data can be viewed through different ontological perspectives — the "different lenses" concept:

- **Risk Lens** — View the graph through risk propagation edges
- **Compliance Lens** — View through regulatory and audit edges
- **Operational Lens** — View through dependency and process edges
- **Financial Lens** — View through FIBO-aligned financial edges

---

## Key OWL Classes (from ontology.ttl)

| Class | Description | BFO Alignment |
|:------|:------------|:-------------|
| `:Agent` | Any actor (human or AI) | `bfo:Object` |
| `:Process` | A temporal sequence of actions | `bfo:Process` |
| `:Episode` | A bounded work session | `bfo:TemporalRegion` |
| `:Concept` | An abstract knowledge unit | `bfo:GenericallyDependentContinuant` |
| `:Decision` | A recorded decision with rationale | `bfo:Process` |
| `:Policy` | An organizational rule or constraint | `bfo:GenericallyDependentContinuant` |
| `:Skill` | A reusable agent capability | `bfo:Disposition` |
| `:Evidence` | Supporting data for a claim | `bfo:InformationContentEntity` |

---

## Company-Specific Extension

To add your own domain concepts, extend `ontology.ttl`:

```turtle
:CustomerSegment rdfs:subClassOf bfo:GenericallyDependentContinuant ;
    rdfs:label "Customer Segment" ;
    rdfs:comment "A company-specific classification of customers." .

:belongsToSegment rdf:type owl:ObjectProperty ;
    rdfs:domain :Customer ;
    rdfs:range :CustomerSegment ;
    rdfs:label "belongs to segment" .
```

The `OWLBridge` will automatically include new classes and properties in its reasoning cycle without any code changes.
