# Research Intelligence Pipeline (CONCEPT:KG-2.6)

## Overview
Automated end-to-end research ingestion: ScholarX Discovery → 9-domain Relevance Scoring → Tiered Ingestion (full for ≥3.0, abstract-only for ≥1.0) → OWL Enrichment → Digest Generation. Supports arXiv, local files, and web URLs.

## Implementation Details
- **Source Code**: ``agent_utilities/automation/research_pipeline.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# KG Source Resolver (CONCEPT:KG-2.6)

## Overview
Bridges the KG indexing layer to the comparative-analysis skill by materializing stored documents to filesystem paths with metadata enrichment. Optional — gracefully returns empty when no KG is available.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/source_resolver.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Research Intelligence Sub-Agent (CONCEPT:KG-2.6)

## Overview
Isolated research context with citation graph traversal (Semantic Scholar API), doom-loop detection, and KG persistence. Findings become `EvidenceNode` entries with `wasDerivedFrom` provenance chains. Adapted from ml-intern's research_tool.py sub-agent pattern.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/orchestration/research_subagent.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Research Orchestration Integration (CONCEPT:KG-2.6)

## Overview
Connects ResearchSubagent (KG-2.33) to ResearchPipelineRunner (KG-2.11) and UnifiedRAGKGRetriever (KG-2.38) for automated daily research cycles. 7-phase pipeline: discovery → subagent session → citation traversal → pipeline ingestion → similarity linking → cluster refresh → KG persistence. MCP-compatible for `run_research_cycle` tool registration.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/orchestration/research_orchestrator.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
