# Unified Memory Architecture

This document consolidates the memory paradigms used by agent-utilities.

## Core Memory Features

- **Autonomous Memory Architecture (CONCEPT:KG-2.1)**: MAGMA-inspired orthogonal reasoning views (Semantic, Temporal, Causal, Entity) combined with Autonomous Self-Improvement loops. Unifies code awareness, chat memory, and **Research Knowledge Bases** (Medical, Chemistry, etc.) into a singular, schema-enforced graph. Cross-domain relationships emerge automatically through shared concepts. Supports unified ingestion of MCP, A2A, and Skill-based resources with automated importance scoring and temporal decay.
- **Cross-Agent Observational Memory Bridge (CONCEPT:KG-2.1)**: Shared local memory layer across 10 terminal agents (Claude Code, Codex, Grok Build, Devin, Antigravity, Windsurf, OpenCode, agent-terminal-ui, Cowork, Hermes). KG is the source of truth; materialized Markdown files (`observations.md`, `reflections.md`, `profile.md`, `active.md`) provide inspectable, editable views at `~/.local/share/agent-utilities/memory/`. Bidirectional sync ensures user edits flow back to the KG. Includes LLM-powered Observer/Reflector pipeline, budgeted startup context injection via agent hooks (ECO-4.6), and `agent-utilities-memory` CLI.
- **Token-Aware Context Compaction (CONCEPT:KG-2.1)**: Intelligent context window management with three strategies (`summarize_tools`, `drop_middle`, `progressive`). Adapted from Goose's `context_mgmt/mod.rs`. Compaction summaries persist as `EpisodeNode` snapshots for cross-session context recall via `MemoryRetriever`.
- **Multi-Timescale Memory Dynamics (CONCEPT:KG-2.2)**: Three-tier memory with timescale-aware exponential decay (Working 5min, Episodic 4hr, Semantic 30-day). Consolidation promotes high-activation memories. Derived from Continual Knowledge Updating (arXiv:2605.05097v1).
- **Memory-Aware Test-Time Scaling (CONCEPT:AHE-3.4)**: Integrates batch-parallel trajectory generation into the HTN planner. Distills reasoning memory concurrently across multiple parallel attempts (successes and failures) yielding zero-shot hypergraph generalization and structural topological feedback.






## Memento Context Management

Integrated context compression and block-masking architecture to optimize KV cache usage and improve long-context agent performance. This powers "sawtooth" context construction, enabling infinite-horizon agent execution.
