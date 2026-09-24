# KG-2.6: Observational Memory Bridge

> **Concept**: `CONCEPT:AU-KG.memory.tiered-memory-caching`
> **Pillar**: 2 — Epistemic Knowledge Graph
> **Status**: Implemented
> **Research**: Inspired by observational-memory v0.6.3

---

## Overview

The Observational Memory Bridge provides **cross-agent session memory** that survives
switching between Claude Code, Codex, Grok Build, Devin, Antigravity, Windsurf,
OpenCode, and agent-terminal-ui. It extends the Knowledge Graph with a materialized
Markdown view layer and LLM-powered observation/reflection pipeline.

### Core Design Principle

> **Markdown files are materialized views — the KG is the source of truth.**

The KG (`GraphBackend` + OWL) materializes into `observations.md`,
`reflections.md`, `profile.md`, and `active.md`; edits to `profile.md`/
`active.md` are file-watched back into the KG as upserts.

> The KG store is the `GraphBackend` — epistemic-graph is the one authority (system
> of record, durable); Postgres/pg-age and LadybugDB/Neo4j/FalkorDB are opt-in
> write-only mirrors (the latter three under `backends/contrib/`). The local
> SQLite-style `knowledge_graph.db` file is the default opt-in LadybugDB mirror.

---

## Architecture

<div class="admonition architecture" markdown>
<p class="admonition-title">Eight external agent surfaces, one memory bridge, one KG</p>

The Hook Installer writes session hooks into eight external agent
surfaces (Claude Code, Codex, Grok Build, Devin, Antigravity IDE,
Windsurf, OpenCode, `agent-terminal-ui`), which all report through those
hooks to the Observer. The Observer writes `ObservationNode`s to the
Knowledge Graph (`LadybugDB`, with Tiered Memory, `SynthesisEngine`, and
`HybridRetriever`). The Reflector reads and writes back to the KG. The
Materializer reads the KG and writes the four materialized views
(`observations.md`, `reflections.md`, `profile.md`, `active.md`);
`observations.md`, `reflections.md`, and `profile.md` are in turn
file-watched back into the KG. `HybridRetriever` feeds the Startup
Context Builder.
</div>

---

## Components

### 1. Memory Materializer

**Module**: `knowledge_graph/memory/memory_engine.py` (`MemoryMaterializer`)

Renders 4 Markdown files from KG state:

| File | Content | Update Frequency |
|:-----|:--------|:----------------|
| `observations.md` | Recent notes with 🔴/🟡/🟢 priority tags, dated sections | After every observe cycle |
| `reflections.md` | Long-term condensed facts, categorized with confidence | After every reflect cycle |
| `profile.md` | Stable user identity (name, preferences, style) | After reflect, on user edit |
| `active.md` | Current goals, recent sessions, thread snapshot | After every materialize |

**Features**:
- Bidirectional sync: user edits to Markdown are ingested back into the KG
- Cursor-based change detection (MD5 hash comparison)
- XDG-compliant storage at `~/.local/share/agent-utilities/memory/`

### 2. Observer & Memento Context Compressor

**Module**: `knowledge_graph/memory/observer.py` and `knowledge_graph/memory/agent_context.py` (`compress_to_memento`)

LLM-powered transcript compression and Context Management:
- **Observations**: Extracts decisions, preferences, lessons, context from raw conversations using a 🔴/🟡/🟢 priority system.
- **Mementos**: Segments long-running agent action-observation cycles and compresses them into dense `MementoBlock` nodes (preserving precise formulas and state).
- **KV Cache Compaction**: Intercepts the history stream and constructs a sawtooth context pattern (`[Past Mementos] + [Current Active Block]`) to prevent context window explosion and OOM errors during infinite-horizon tasks.
- Cursor-based incremental processing (avoid re-processing seen messages)
- Per-source parsers for Claude, Codex, Grok JSONL formats

### 3. Reflector

**Module**: `knowledge_graph/memory/optimization_engine.py` (`run_reflector`)

Condenses observations into durable long-term memory:
- Wired into existing SynthesisEngine (KG-2.4)
- Merges, promotes, demotes, and archives observations
- Extracts preferences and principles into dedicated KG node types
- Triggers materialization after reflection

### 4. Startup Context Builder

**Module**: `knowledge_graph/memory/memory_engine.py` (`StartupContextBuilder`)

Produces deterministic, budgeted startup payloads:
- Default budget: 24,000 characters
- Priority-scored chunks from profile + active context
- Routing terms from `--cwd` and `--task` boost relevant sections
- Overflow handles for expanding specific sections via `recall`

### 5. Hook Installer (AU-ECO.mcp.toolkit-live-discovery)

**Module**: `ecosystem/hook_installer.py`

Writes startup/checkpoint hooks into external agent configurations:

| Agent | Hook Mechanism |
|:------|:--------------|
| Claude Code | `~/.claude/settings.json` SessionStart/End hooks |
| Codex | `~/.codex/hooks.json` |
| Grok Build | `~/.grok/hooks/agent-utilities-memory.json` |
| Devin | `~/.devin/hooks.json` |
| Antigravity | `~/.gemini/antigravity/hooks.json` |
| Windsurf | `~/.codeium/windsurf/hooks.json` |
| OpenCode | `~/.opencode/hooks.json` |
| agent-terminal-ui | Direct Python API (zero-copy) |
| Cowork | macOS plugin directory |
| Hermes | `$HERMES_HOME/plugins/` |

---

## CLI Commands

```bash
# Produce startup context for an agent
agent-utilities-memory context --for codex --cwd $PWD --task "fix CI" --budget-chars 24000

# Search KG memory
agent-utilities-memory recall --query "what was decided about the database?"

# Expand a specific startup handle
agent-utilities-memory recall --handle startup:profile:preferences

# Process pending transcripts (requires --file pointing at a JSONL transcript)
agent-utilities-memory observe --source claude --file ~/.claude/transcript.jsonl

# Run reflection cycle
agent-utilities-memory reflect

# Install hooks into external agents (comma-separated; empty = all)
agent-utilities-memory install --agents claude,codex,grok

# Verify hook health
agent-utilities-memory doctor
```

---

## Data Flow

```
Agent A finishes session
  → Session hook fires (SessionEnd)
  → Transcript parsed + ingested into KG as Thread/Message nodes
  → LLM Observer extracts decisions, preferences, lessons → ObservationNode in KG
  → SynthesisEngine reflects → updates Semantic Memory
  → KG Materializer renders updated Markdown files

Agent B starts session
  → Startup hook fires (SessionStart)
  → Calls `agent-utilities-memory context --for <agent> --cwd $PWD`
  → Context Builder queries KG HybridRetriever
  → Produces budgeted startup payload
  → Agent B receives startup context
  → Agent B can expand via `agent-utilities-memory recall --query "..."`
```

---

## File Storage

- `~/.local/share/agent-utilities/`
    - `kg/`
        - `knowledge_graph.db` — source of truth (LadybugDB)
    - `memory/` — materialized views (KG-2.6)
        - `observations.md`
        - `reflections.md`
        - `profile.md`
        - `active.md`
        - `.memory_cursor.json` — materialization state
        - `.observer_cursor.json` — observer incremental state
    - `...`

---

## Related Concepts

- **KG-2.1**: Tiered Memory & Context — underlying memory tier lifecycle
- **KG-2.4**: Inductive Knowledge & Hypergraphs — SynthesisEngine hosts reflection rules
- **KG-2.3**: Graph Integrity & Retrieval — HybridRetriever powers startup context
- **KG-2.7**: External Graph Federation — multi-machine KG sync
- **AU-ECO.mcp.toolkit-live-discovery**: Agent Hook Installer — cross-agent integration
- **OS-5.0**: Agent OS Kernel — XDG path resolution for memory directory
