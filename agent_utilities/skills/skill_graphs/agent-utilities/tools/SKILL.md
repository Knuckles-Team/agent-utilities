---
name: agent-utilities-tools
description: >-
  Tool reference for agent-utilities. Covers KG tools (graph-os MCP),
  X search tools, workflow tools, and the tool registration system.
tags: [tools, mcp, kg, x-search, workflows, graph-os]
---

# agent-utilities Tools Reference

## 🔍 Knowledge Graph Tools (graph-os MCP)

The KG is exposed via the `graph-os` MCP server with these actions:

### graph_query — Read-only Cypher queries
```python
# Find all concepts in the ORCH pillar
result = graph_query(cypher="MATCH (n:Concept) WHERE n.pillar = 'ORCH' RETURN n")
```

### graph_search — Semantic + keyword hybrid search
```python
# Modes: hybrid, concept, analogy, memory, discover, dci
result = graph_search(query="multi-agent orchestration patterns", mode="hybrid", top_k=10)

# Look up a specific concept
result = graph_search(query="ORCH-1.2", mode="concept")
```

### graph_write — Mutate the KG
```python
# Add a node
graph_write(action="add_node", node_id="my_agent", node_type="Agent",
            properties='{"name": "My Agent", "description": "..."}')

# Add an edge
graph_write(action="add_edge", source_id="agent_1", target_id="tool_1",
            rel_type="USES_TOOL")

# Store a memory
graph_write(action="store_memory", properties='{"content": "...", "tier": "semantic"}')
```

### graph_analyze — Cross-reference analysis
```python
# Synthesize: cross-reference query across all KG content
graph_analyze(action="synthesize", query="agent evolution patterns")

# Blast radius: find all nodes affected by a change
graph_analyze(action="blast_radius", node_id="ORCH-1.2", depth=3)

# Security scan
graph_analyze(action="security_scan", target="agent_utilities/security/")
```

### graph_ingest — Add data to the KG
```python
# Ingest a codebase
graph_ingest(action="ingest", target_path="/path/to/project")

# Ingest a URL
graph_ingest(action="ingest", target_path="https://example.com/article")

# Ingest knowledge pack (ScholarX papers, etc.)
graph_ingest(action="ingest_knowledge_pack", target_path="/path/to/papers")
```

### graph_orchestrate — Multi-agent workflows
```python
# Dispatch a task
graph_orchestrate(action="dispatch", task="Review PR #42 for security issues")

# Execute a named agent
graph_orchestrate(action="execute_agent", agent_name="legal-compliance-agent")
```

---

## 🐦 X Search Tools

### x_search — Search X posts
```python
from agent_utilities.tools.x_search_tool import x_search, browse_x_post

# Search for posts
results = await x_search("multi-agent systems research")

# Browse a specific post with auto-ingestion to KG
result = await browse_x_post(
    "https://x.com/i/status/2057129225593741768",
    auto_ingest=True  # Automatically classifies and persists to KG
)
```

Auto-ingest pipeline:
1. xAI Grok-4.3 fetches post content (1M context)
2. `UniversalKnowledgeClassifier` scores importance + evolution potential
3. `XIngestionBridge` creates SocialPost + Person + Concept nodes in KG
4. X Articles are fully ingested via `KBIngestionEngine.ingest_url()`
5. High evolution potential triggers `EvolutionCandidateNode` creation

---

## ⚙️ Workflow Tools

### WorkflowStore — KG-native persistence
```python
from agent_utilities.knowledge_graph.workflow_store import WorkflowStore

store = WorkflowStore(engine)

# Save a workflow
workflow_id = store.save_workflow("my_workflow", plan, description="...")

# Load and replay
plan = store.load_workflow("my_workflow")

# List available workflows
workflows = store.list_workflows()
```

### SkillCompiler — SKILL.md → KG registration
```python
from agent_utilities.workflows.skill_compiler import SkillCompiler

# Compile a skill directory into the KG
result = SkillCompiler.compile_skill(skill_dir, engine)
# Returns: {workflow_id, team_config_id, ...}
```

### Pre-built Workflows
- `x_research` — Search X → classify → ingest
- `knowledge_assimilation` — Multi-source → classify → ingest → analyze → plan
- `self_evolution_v2` — Process pending EvolutionCandidates

```python
from agent_utilities.knowledge_graph.kb.x_workflows import get_workflow_plan

plan = get_workflow_plan("knowledge_assimilation")  # Returns GraphPlan
```
