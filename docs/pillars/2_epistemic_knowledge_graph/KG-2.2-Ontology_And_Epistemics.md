# OWL-Driven Semantic Subsumption (CONCEPT:KG-2.2)

## Overview
Enables hierarchy-aware zero-shot ontology alignment by comparing new topological embeddings against OWL class prototypes to automatically infer and inject full class lineage.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/core/semantic_subsumption.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Model Display Optimization (CONCEPT:KG-2.6)

## Overview
Display-predict decoupling engine: optimizes model `__str__()` for LLM consumption independently of `predict()` logic. 5 strategies (linear_collapse, piecewise_table, symbolic_equation, coefficient_summary, adaptive/SmartAdditive). Bounded complexity budgets. Based on arXiv:2605.03808.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/core/model_display.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Versioned KG Mutations (CONCEPT:KG-2.3)

## Overview
Git-like transactional mutation semantics for Knowledge Graph evolution: KGTransaction (batched mutations), KGCommit (atomic application with rollback data), KGVersionEngine (commit/rollback/diff), KGDiff (structural diff between graph versions). Derived from Evolving Idea Graphs (arXiv:2605.04922v1).

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/core/kg_versioning.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
