# Token-Aware Context Compaction (CONCEPT:KG-2.1)

## Overview
Intelligent context window management with three strategies (summarize_tools, drop_middle, progressive). Compaction summaries persist as `EpisodeNode` snapshots for cross-session context recall. Adapted from Goose's context_mgmt.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/context_compactor.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Cross-Session Chat Recall (CONCEPT:KG-2.1)

## Overview
Keyword-based search across stored chat sessions using the KG Cypher backend. Adapted from Goose.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/chat_search.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Project-Aware Context (CONCEPT:KG-2.1)

## Overview
Native support for Claude-style project rules. Backend automatically loads and injects `AGENTS.md` (Project Rules) into the system prompt for high-fidelity codebase awareness.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/agents_md.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Elastic Context Operators (CONCEPT:KG-2.2)

## Overview
5 atomic operators for elastic context orchestration: Skip, Compress, Rollback, Snippet, Delete. Compress is expressively complete while specialized operators reduce hallucination risk. Includes checkpoint/rollback support for speculative context operations. Derived from LongSeeker (arXiv:2605.05191v1).

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/context_compactor.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Multi-Timescale Memory Dynamics (CONCEPT:KG-2.2)

## Overview
Three-tier memory with timescale-aware exponential decay: Working (5min half-life), Episodic (4hr), Semantic (30-day). High-activation memories consolidate from Working→Episodic→Semantic via access-count thresholds. Relevance-weighted retrieval with keyword scoring. Derived from Continual Knowledge Updating (arXiv:2605.05097v1).

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/timescale_memory.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
