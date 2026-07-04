---
name: kg-memory
description: >-
  Thin verb over the engine's EG-KG.memory.eg-batch-decay-caller memory surface — episodic→semantic consolidation, the
  spatial scene graph, RL trajectories, plus unified store/recall/link memory-CRUD. Use
  for agent memory ops — "consolidate memory", "store/recall a memory", "append a
  trajectory step", "scene transform".
license: MIT
tags: [graph-os, engine, memory]
tier: core
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-memory

`graph_memory` (CONCEPT:AU-KG.coordination.engine-message-broker) fronts the engine's memory surface. Engine methods (action-routed 1:1, dashes→underscores): `create_summary`, `consolidate`, `maintain`, `add_scene_object`, `world_transform`, `start_trajectory`, `append_step`, `discounted_return`, `get_*`. Unified memory-CRUD (AU-KG.memory.unified-memory-crud-core) `store` (`agent_id`+`content`[+`memory_type`,`tags`]), `recall` (`query`[+`memory_type`]), `link` (`source`+`target`[+`rel_type`]) route into the SAME graph_write memory core as the REST `/graph/write/memory` twins. Structured args go via `params_json`.

## Invoke
- **MCP:** `load_tools(tools=["graph_memory"])`, then `graph_memory(action="recall", params_json='{"query":"..."}')`.
- **REST twin:** `POST /graph/memory` with `{"action": "consolidate"}`.

## Example
```
graph_memory(action="store", params_json='{"agent_id":"a1","content":"...","memory_type":"semantic"}')
```
