# Query Analysis (CONCEPT:AU-ECO.connector.apply-any-query-analysis)

## Overview
Derives source-type and time-window filters from a natural-language query (LLM when available,
deterministic regex/keyword fallback offline) and attaches citations to results. Wired into
`HybridRetriever.retrieve_hybrid(query_analysis=True)` as an opt-in pre-filter; the
`CitationProcessor` annotates `[n]` markers with source links from existing provenance.

## Implementation Details
- **Source Code**: `agent_utilities/knowledge_graph/retrieval/query_analysis.py`
- **Pillar**: ECO
