# Token-Aware Context Compaction (CONCEPT:AU-KG.memory.tiered-memory-caching)

## Overview
Intelligent context window management with three strategies (summarize_tools, drop_middle, progressive). Compaction summaries persist as `EpisodeNode` snapshots for cross-session context recall. Adapted from Goose's context_mgmt.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/memory/agent_context.py`` (`ContextCompactor`)
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Cross-Session Chat Recall (CONCEPT:AU-KG.memory.tiered-memory-caching)

## Overview
Keyword-based search across stored chat sessions using the KG Cypher backend. Adapted from Goose.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/retrieval/chat_search.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Project-Aware Context (CONCEPT:AU-KG.memory.tiered-memory-caching)

## Overview
Native support for Claude-style project rules. Backend automatically loads and injects `AGENTS.md` (Project Rules) into the system prompt for high-fidelity codebase awareness.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/core/agents_md.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Elastic Context Operators (CONCEPT:AU-KG.ingest.engineering-rules)

## Overview
5 atomic operators for elastic context orchestration: Skip, Compress, Rollback, Snippet, Delete (`ContextOperator`). Compress is expressively complete while specialized operators reduce hallucination risk. Includes checkpoint/rollback support for speculative context operations. Derived from LongSeeker (arXiv:2605.05191v1).

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/memory/agent_context.py`` (`ContextOperator`, `AgentContextManager`)
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Multi-Timescale Memory Dynamics (CONCEPT:AU-KG.ingest.engineering-rules)

## Overview
Three-tier memory with timescale-aware exponential decay: Working (5min half-life), Episodic (4hr), Semantic (30-day). High-activation memories consolidate from Working→Episodic→Semantic via access-count thresholds. Relevance-weighted retrieval with keyword scoring. Derived from Continual Knowledge Updating (arXiv:2605.05097v1).

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/memory/agent_context.py`` (`TimescaleMemoryStore`, `MemoryTimescale`)
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
