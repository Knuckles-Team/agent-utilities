# Contextual-Retrieval Enrichment (CONCEPT:KG-2.50)

## Overview
Before a chunk is embedded, a short *context* string situating it within the whole document is
computed and prepended to the embedding input (Anthropic "Contextual Retrieval"), markedly
improving recall. LLM path uses the lite chat model; a deterministic heuristic (title + nearest
heading + part i/n + opening sentence) keeps it working offline. Default OFF in the KG-2.48
`DocumentProcessor`; the connector ingestion path turns it ON.

## Implementation Details
- **Source Code**: `agent_utilities/knowledge_graph/ontology/contextual_enrichment.py`;
  wired in `ontology/document_processing.py` (`DocumentProcessor(contextual=True)`)
- **Pillar**: KG
