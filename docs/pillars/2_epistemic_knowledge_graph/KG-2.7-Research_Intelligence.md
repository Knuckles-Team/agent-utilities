# Research Intelligence Pipeline (CONCEPT:AU-KG.research.research-pipeline-runner)

## Overview
Automated end-to-end research ingestion: ScholarX Discovery → 9-domain Relevance Scoring → Tiered Ingestion (full for ≥3.0, abstract-only for ≥1.0) → OWL Enrichment → Digest Generation. Supports arXiv, local files, and web URLs.

## Implementation Details
- **Source Code**: ``agent_utilities/automation/research_pipeline.py``
- **Pillar**: KG

Both relevance tiers use one authoritative graph-write contract. The Article,
Source, pseudonymous author references, and their edges are committed as one
engine-native `ChangeEnvelope` graph slice; there is no direct
`engine.graph`/backend upsert or ScholarX bridge fallback. Raw author names and
emails are not durable metadata: only stable non-reversible references cross the
persistence boundary.

When a full paper has a PDF, the article slice commits first. Full text is then
read from the local file without persisting its path, scrubbed of known author
identifiers, and materialized as a Document/Chunk slice through
`DocumentProcessor`'s native `ChangeEnvelope` path. That second slice is an
optional retrieval projection, not a competing graph authority; a projection
failure is logged without undoing the already committed Article.

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# KG Source Resolver (CONCEPT:AU-KG.research.research-pipeline-runner)

## Overview
Bridges the KG indexing layer to the comparative-analysis skill by materializing stored documents to filesystem paths with metadata enrichment. Optional — gracefully returns empty when no KG is available.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/core/source_resolver.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Research Intelligence Sub-Agent (CONCEPT:AU-KG.research.research-pipeline-runner)

## Overview
Isolated research context with citation graph traversal (Semantic Scholar API), doom-loop detection, and KG persistence. Findings become `EvidenceNode` entries with `wasDerivedFrom` provenance chains. Adapted from ml-intern's research_tool.py sub-agent pattern.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/orchestration/research_subagent.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Research Orchestration Integration (CONCEPT:AU-KG.research.research-pipeline-runner)

## Overview
Connects ResearchSubagent (AU-KG.research.zero-llm-pack-link) to ResearchPipelineRunner (KG-2.3) and UnifiedRAGKGRetriever (KG-2.38) for automated daily research cycles. 7-phase pipeline: discovery → subagent session → citation traversal → pipeline ingestion → similarity linking → cluster refresh → KG persistence. MCP-compatible for `run_research_cycle` tool registration.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/orchestration/research_orchestrator.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
