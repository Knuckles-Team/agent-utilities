# Pillar 2: Epistemic Knowledge Graph

## Overview

The **Epistemic Knowledge Graph** transforms how the agent ecosystem perceives, stores, and retrieves information. It replaces static vector-based Retrieval-Augmented Generation (RAG) with a dynamic, graph-native, and self-organizing memory substrate that leverages formal mathematics and topological reasoning.

## Why We Built This (Rationale)

Standard RAG architectures suffer from three critical flaws that block Agentic General Intelligence (AGI):
1. **Context Fragmentation**: Vector databases retrieve isolated chunks without understanding structural dependencies, leading to "hallucinations" when synthesizing complex concepts.
2. **Retrieval Degradation (O(N) Scanning)**: As memory grows, scanning all vectors becomes computationally expensive and introduces irrelevant noise.
3. **Poisoning and Contradictions**: Agents continually ingesting data can overwrite critical instructions or believe contradictory facts if there is no epistemological verification.

## How It Works (Implementation)

The solution is the `15-phase Unified Intelligence Pipeline` backed by LadybugDB (Cypher) and NetworkX.

### RAG-KG Unification & Spectral Clustering (KG-2.38 & KG-2.34)
We collapsed separate vector indexes directly into the Knowledge Graph. By computing an **Auto-Similarity Memory Graph**, the system pre-computes semantic proximity and creates `SIMILAR_TO` edges. Retrieval is now accelerated to O(degree) complexity via shortest-path traversal. The **Spectral Cluster Navigator** groups these nodes using normalized Laplacian eigengap heuristics, providing hierarchy-aware context scoping.

### Multi-Domain Architecture (KG-2.51)
Transitioned the agent framework into a **Multi-Domain Expert System**, supporting modular expansion into `finance`, `medical`, `law`, and `science`. The architecture relies on Vectorized Topological Memory and the core Knowledge Graph for semantic interoperability. Domain-specific dependencies (e.g., PyTorch, Statsmodels for quantitative finance) are loaded optionally via environment tags (like `agent-utilities[finance]`) to keep the core graph orchestrator lightweight.

### Enterprise Architecture Scaling (Hub-and-Spoke Ingestion)
To support 100,000+ employees and scale to true enterprise size, the architecture has decoupled its localized NetworkX memory from the persistent Backend. NetworkX now serves purely as an **ephemeral compute scratchpad** for localized sub-graph analytics, preventing Out-Of-Memory (OOM) bottlenecks.
Furthermore, raw data ingestion (e.g., Active Directory, Workday, ServiceNow) is externalized to peripheral "spoke" agents. These webhooks utilize high-throughput asynchronous batched `UNWIND` logic directly into the central graph.

### Deterministic Garbage Collection (Mark-and-Sweep)
To maintain 1:1 parity between external file systems and the Knowledge Graph, the system utilizes a **Mark-and-Sweep Synchronization** approach:
- **Mark**: During ingestion, every parsed file (e.g., `:Code` or `:Article` nodes) is tagged with a session-specific `last_seen_timestamp` in the pipeline context.
- **Sweep**: After parsing concludes, a cleanup Cypher query automatically detaches and deletes any nodes in the active workspace scope that have an older timestamp.
- **Handling Duplicates/Updates**: Because upserts are idempotent and keyed on `id`, repeatedly ingesting the same file naturally updates its properties rather than creating duplicates.
- **MD5 Checksums vs Timestamps**: We explicitly opted for temporal mark-and-sweep over md5 checksum tracking. Checksum tracking introduces significant state overhead and collision complexities. In contrast, timestamp-based pruning is an atomic and stateless mechanism that natively drops deleted files while gracefully updating modified ones.

### Graph-Level Access Control (RBAC/ABAC)
Security is implemented natively at the query layer. Cryptographic ABAC middleware injects dynamic `requiresClassification` and `SecurityClearance` filters directly into Cypher statements. This guarantees that agents cannot traverse restricted sub-graphs (like executive compensation data) regardless of the prompt.

### Semantic Subsumption & Inductive Hypergraphs (KG-2.16 & KG-2.4)
When new information is encountered, **OWL-Driven Semantic Subsumption** automatically computes embedding similarities against OWL class prototypes, injecting the new concept into the correct lineage. **Inductive Knowledge Hypergraphs** vectorize relationship intersections via `EncPI` (Positional Interaction Encodings), enabling the graph to perform zero-shot generalization over entirely novel runtime topologies.

### Formal Mathematical Primitives (KG-2.41 — KG-2.49)
We integrated advanced primitives from the MIT Mathematics for Computer Science (MCS) curriculum:
- **Formal Relations (KG-2.47)**: Enforces Reflexive, Symmetric, and Transitive closures for zero-shot entity resolution.
- **State Machine Invariants (KG-2.48)**: Validates deterministic transitions against structural invariants.
- **Markov Transition Forecasting (KG-2.49)**: Predicts statistical failure nodes in execution traces. Extended with:
  - **Chapman-Kolmogorov Multi-Step Forecasting**: N-step transition probabilities via matrix powers for long-horizon prediction.
  - **Markov Regime Detection**: Three-state (Bull/Bear/Sideways) market regime classification from financial time-series with per-asset-class default thresholds (equities, crypto, forex, commodities, fixed income).
  - **Hidden Markov Model Inference**: Gaussian HMM with Baum-Welch estimation and Viterbi decoding for latent regime detection (`hmmlearn` integration).
  - **Walk-Forward Backtesting**: Rolling-window regime model re-estimation with strict no-lookahead bias guarantees.
  - **PreemptiveCacheEngine Integration**: `predict_next_states()` satisfies the forecaster contract for predictive context pre-loading.
- **Structural Causal Reasoning (KG-2.43)**: Utilizes do-calculus and d-separation to perform counterfactual analysis, moving beyond correlation to causal verification.

## Benefits Introduced

- **Provable Safety**: Formal state machines and causal reasoning provide mathematical guarantees against infinite loops and logical paradoxes.
- **Retrieval Precision**: The `Hybrid Search Index` combining semantic+keyword scoring (72%/28%), augmented by `Backlink-Density Boost`, yields unmatched relevance, fetching central hub concepts naturally.
- **Epistemic Integrity**: The two-phase Entity-Claim Extraction extracts assertions and explicitly maps contradictions (`CONTRADICTS` edge), ensuring the system maintains a logically consistent worldview.

## Key Concepts Leveraged
- **KG-2.0**: Active Knowledge Graph
- **KG-2.4**: Inductive Knowledge Hypergraphs
- **KG-2.16**: Semantic Subsumption
- **KG-2.21**: Multi-Timescale Memory
- **KG-2.34**: Spectral Cluster Navigator
- **KG-2.38**: RAG-KG Unification
- **KG-2.41–2.49**: Formal Mathematics, Causal Reasoning, and Optimal Execution
- **KG-2.51**: Multi-Domain Architecture
- **KG-2.7**: Context Graph Architecture (SPARQL, ArchiMate, ADR)

## Enterprise Ontology Alignments

The Knowledge Graph is heavily aligned with the **Basic Formal Ontology (BFO)** and other industry standards, enabling enterprise-scale deployments:
- **BFO (Basic Formal Ontology)**: Provides a mathematically sound upper ontology, allowing transitive reasoning for critical structural analysis (e.g., blast-radius detection via `dependsOn`).
- **PROV-O (Provenance Ontology)**: Ensures every action or fact injected into the graph is completely auditable (`wasDerivedFrom`, `wasAttributedTo`), which is non-negotiable for regulated enterprise environments.
- **SKOS (Simple Knowledge Organization System)**: Enables semantic mapping between internal proprietary terminology and industry-standard vocabularies dynamically (`broader`, `narrower`, `exactMatch`).
- **Dublin Core**: Provides standard metadata tracing for documents, datasets, and codebase artifacts.
- **ArchiMate 3.1**: Enterprise architecture types (`BusinessRole`, `ApplicationComponent`, `BusinessProcess`) are registered in both the OWL ontology (`ontology.ttl`) and the Python graph schema (`schema_definition.py`), enabling cross-repository capability mapping.

## Continuous Ingestion (Git Hook Pipeline)

The Knowledge Graph maintains currency with codebase changes through a **post-commit hook** pipeline:

1. **`scripts/install_git_hooks.py`**: Deploys `.git/hooks/post-commit` to all repositories in the workspace.
2. **`.git/hooks/post-commit`**: On each commit, invokes `scripts/submit_diff.py` with the latest diff.
3. **`scripts/submit_diff.py`**: Bridges the git hook to the KG task queue via `engine.submit_task(task_type="diff")`.
4. **`engine_tasks.py`**: The `TaskManagerMixin` processes diff tasks, creating `DiffEntry` nodes that link the patch content to the originating repository.

This ensures the Knowledge Graph stays synchronized with the active development state without requiring manual re-ingestion.

## Entity Lifecycle Management

All graph nodes follow a converged lifecycle state machine:

```
ACTIVE ──(soft-delete)──▶ ARCHIVED ──(hard-delete)──▶ REMOVED
   ▲                         │
   └──────(restore)──────────┘
```

- **`status: ACTIVE`** — Default state. Node is included in all search and retrieval operations.
- **`status: ARCHIVED`** — Soft-deleted. Excluded from `search_hybrid()`, `_search_keyword()`, and `discover_all_capabilities()`. Can be restored via `DocumentDeletionPipeline.restore_document()`.
- **`status: DEPRECATED`** — Marked for eventual removal but still discoverable for migration purposes.
- **Hard deletion** — Permanently removed from the graph after age-based cleanup (`DocumentCleanup.cleanup_soft_deleted_documents()`).

This lifecycle is enforced uniformly across:
- `QueryMixin` (engine_query.py) — Search-time filtering
- `DocumentDeletionPipeline` (document_deletion.py) — Soft-delete/restore operations
- `DocumentUpdatePipeline` (document_update.py) — Update rejection for archived nodes
- `DocumentCleanup` (document_cleanup.py) — Age-based hard deletion

## Context Graph Architecture (KG-2.7)

The Knowledge Graph implements the **Context Graph Architecture** pattern, formalizing the decision trace, enterprise governance, and semantic interoperability layers that turn a fragmented graph into a unified intelligence substrate.

### Architecture Decision Records (ADR)

`ArchitectureDecisionRecord` is a first-class KG node type that captures the full decision context:
- **Context**: Why the decision was needed
- **Decision**: What was decided
- **Rationale**: Why this option was chosen
- **Alternatives**: Options that were considered
- **Consequences**: Known tradeoffs
- **Authority**: Who/what approved (user, policy, evolution daemon)
- **Impacted Concepts**: Which concept IDs are affected

ADR lifecycle: `proposed → accepted → deprecated → superseded`

The `supersedes` relationship is declared as an OWL `TransitiveProperty`, enabling full decision lineage queries (if ADR-C supersedes ADR-B supersedes ADR-A, then ADR-C transitively supersedes ADR-A).

### ArchiMate EA Governance Layer

The `ArchiMateLayer` module (`core/archimate_layer.py`) maps KG node types to **ArchiMate 3.2** metamodel elements across five layers:

| Layer | KG Types | ArchiMate Types |
|-------|----------|----------------|
| **Business** | Policy, ProcessFlow, Organization, Role, Team | BusinessRule, BusinessProcess, BusinessActor |
| **Application** | Agent, Tool, Skill, SystemPrompt | ApplicationComponent, ApplicationService |
| **Technology** | Server, DataConnector, Pipeline, Repository | TechnologyService, Node, Artifact |
| **Strategy** | Concept, Capability, Experiment, ADR | Capability, CourseOfAction |
| **Motivation** | Goal, Principle, Regulation, EngineeringRule | Goal, Principle, Constraint, Requirement |

This enables enterprise-architecture-level views and governance over the agent ecosystem.

### SPARQL Read-Only Endpoint

The OWL bridge (`core/owl_bridge.py`) provides a SPARQL read-only interface via `rdflib` materialization:

1. **rdflib Materialization**: The LPG is materialized into an in-memory `rdflib.Graph` with typed individuals and property assertions under the `au:` namespace.
2. **Query Execution**: Full SPARQL SELECT, ASK, and CONSTRUCT queries are supported.
3. **Cache**: The RDF graph is cached and invalidated when the LPG changes.
4. **MCP Exposure**: Available via `kg_query(scope="sparql")` for direct SPARQL queries from agents.

This enables Semantic Web interoperability without requiring a native SPARQL triplestore.

### SPARQL HTTP Endpoint

The `SPARQLEndpoint` class (`core/sparql_http.py`) provides a **W3C SPARQL Protocol**-compliant HTTP interface:

- **GET/POST** handlers with `?query=` parameter
- **Content negotiation**: `application/sparql-results+json`, `text/turtle`
- Returns standard **W3C SPARQL Results JSON** format
- Mountable as a Starlette ASGI app onto FastMCP

This allows other agent-utilities deployments to consume the KG as a standard SPARQL endpoint over HTTP.

### SDD Ontology Layer

The Spec-Driven Development ontology (`ontology_sdd.ttl`) formalizes the SDD workflow into OWL classes mapped to ArchiMate 3.2:

| SDD Class | OWL Parent | ArchiMate Layer → Type |
|-----------|-----------|----------------------|
| `Specification` | bfo:GDC | Strategy → Capability |
| `SoftwareFeature` | bfo:GDC | Application → ApplicationFunction |
| `Requirement` | Specification | Motivation → Requirement |
| `UserStory` | Requirement | Motivation → Requirement |
| `AcceptanceCriteria` | Specification | Motivation → Requirement |
| `SoftwareComponent` | bfo:IC | Application → ApplicationComponent |
| `APIContract` | bfo:GDC | Application → ApplicationInterface |
| `TestCase` | bfo:Process | Application → ApplicationFunction |
| `DesignGuideline` | Principle | Motivation → Principle |
| `ComplianceConstraint` | Regulation | Motivation → Constraint |

SDD properties include `realizes`, `specifies`, `testedBy`, `constrainedBy`, `guidelineFor`, `implementedBy`, `exposesAPI`, and `derivedFrom` (transitive).

### Enterprise Core Ontology

The `ontology_enterprise.ttl` module extracts the governance-relevant subset into a standalone importable standard:

- **ArchiMate 3.2 layer hierarchy** (Business, Application, Technology, Strategy, Motivation)
- **ADR decision trace** classes and properties
- **Enterprise governance** properties (`governedBy`, `enforces`, `complianceStatus`)
- **Enterprise integration points**: `LeanIXFactSheet` and `ARISProcess` classes for EA tool interoperability
- `externalToolId` property for linking KG nodes to external EA tools

Domain deployments import via `owl:imports <http://knuckles.team/kg/enterprise>`.

### Modular Ontology Architecture

The ontology is organized into domain modules following the `owl:imports` pattern:

```
ontology.ttl                → Core upper ontology (BFO, PROV-O, SKOS)
├── owl:imports enterprise  → ArchiMate, ADR, governance
├── owl:imports sdd         → Spec-driven development classes
└── Domain modules:
    ├── ontology_banking.ttl     → ISO 20022, KYC/AML, Basel III
    ├── ontology_government.ttl  → Government-specific
    ├── ontology_hr.ttl          → Human resources
    ├── ontology_legal.ttl       → Legal domain
    └── ontology_medical.ttl     → Healthcare/medical
```

The `OntologyLoader` (`core/ontology_loader.py`) resolves `owl:imports` declarations at runtime, fetching remote ontologies via HTTP with TTL-based caching.

### SHACL Governance Validation

The `SHACLValidator` (`core/shacl_validator.py`) validates the materialized RDF graph against SHACL shape constraints using `pyshacl`:

- **Single-file validation**: Validate against one shapes file
- **Layered validation**: Apply global shapes first, then domain-specific overrides
- **KG integration**: `validate_kg(bridge)` materializes the LPG and validates in one call
- **MCP exposure**: Available via `kg_inspect(view="shacl_validate")`

Default governance shapes (`shapes/governance.shapes.ttl`) enforce:
- ADR must have `context`, `decision`, and `authority`
- Agent must have a `name`
- Policy should link to at least one Concept
- Specification must have a `name`
- Requirement should have a `priority`

### Ontology Publisher

The `OntologyPublisher` (`core/ontology_publisher.py`) enables agent-utilities to serve as both ontology author and distributor:

- **Local export**: Serialize RDF to TTL/XML/N3 with version tags
- **Stardog push**: Upload via `pystardog` to centralized Stardog instances
- **Fuseki push**: Upload via REST API to Apache Jena Fuseki
- **MCP exposure**: Available via `kg_inspect(view="export_ontology")`

This completes the "Hub-and-Spoke" ontology distribution pattern where agent-utilities maintains the authoritative source and pushes evolved ontologies to enterprise infrastructure.
