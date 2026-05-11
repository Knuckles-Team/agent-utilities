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
- **Markov Transition Forecasting (KG-2.49)**: Predicts statistical failure nodes in execution traces.
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

## Enterprise Ontology Alignments

The Knowledge Graph is heavily aligned with the **Basic Formal Ontology (BFO)** and other industry standards, enabling enterprise-scale deployments:
- **BFO (Basic Formal Ontology)**: Provides a mathematically sound upper ontology, allowing transitive reasoning for critical structural analysis (e.g., blast-radius detection via `dependsOn`).
- **PROV-O (Provenance Ontology)**: Ensures every action or fact injected into the graph is completely auditable (`wasDerivedFrom`, `wasAttributedTo`), which is non-negotiable for regulated enterprise environments.
- **SKOS (Simple Knowledge Organization System)**: Enables semantic mapping between internal proprietary terminology and industry-standard vocabularies dynamically (`broader`, `narrower`, `exactMatch`).
- **Dublin Core**: Provides standard metadata tracing for documents, datasets, and codebase artifacts.
