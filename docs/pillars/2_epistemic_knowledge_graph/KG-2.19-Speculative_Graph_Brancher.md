# Speculative Graph Brancher (CONCEPT:KG-2.19)

## Overview
The **Speculative Graph Brancher** enables concurrent, non-blocking mutations on the Knowledge Graph by spawning isolated transactional branches (`KGTransaction`). Multiple agents can execute reasoning paths in parallel without acquiring global database locks. 

When execution concludes, a semantic diff is calculated and validated for conflicts (such as concurrent deletion of nodes modified by the branch) before being atomically committed back to the main graph state (`KGCommit`).

## Architecture

The brancher builds on the `KGVersionEngine`, which provides git-like transactional semantics for KG evolution. The key architectural layers are:

1. **Mutation Layer** — `KGMutation` captures five atomic operations (`ADD_NODE`, `UPDATE_NODE`, `DELETE_NODE`, `ADD_EDGE`, `DELETE_EDGE`), each recording both forward data and rollback data.
2. **Transaction Layer** — `KGTransaction` batches multiple mutations into a single logical changeset with a timestamp and SHA-256 transaction ID.
3. **Commit Layer** — `KGCommit` records the applied mutation count, parent commit reference, and rollback data, forming an append-only commit log (similar to git's DAG).
4. **Branching Layer** — `SpeculativeGraphBrancher` manages named branches, each a deep-copy of the main graph state at branch creation time.

```
Main State ───────────────────────────────────────►
       │                                     ▲
       ├──► Branch "research-a" (deep copy) ─┤ merge
       │                                     │
       └──► Branch "research-b" (deep copy) ─┘
```

## API Surface

### KGTransaction

```python
tx = KGTransaction(description="Add research findings")
tx.add_node("paper:001", {"title": "New Paper", "type": "paper"})
tx.add_edge("paper:001", "concept:ORCH-1.2", "enhances")
tx.update_node("concept:ORCH-1.2", {"enhanced_by": "paper:001"})
tx.delete_node("stale:old_concept")
tx.delete_edge("paper:001", "stale:old_concept", "cites")
```

### KGVersionEngine

```python
engine = KGVersionEngine()
graph = {"nodes": {}, "edges": []}

# Commit atomically (auto-rollback on exception)
commit = engine.commit(tx, graph)

# Inspect diff between two states
diff = KGVersionEngine.diff(state_before, state_after)
print(diff.total_changes)  # nodes_added + removed + modified + edges

# Full rollback
engine.rollback(commit, graph)

# Browse commit history
for c in engine.history:
    print(c.commit_id, c.mutations_applied)
```

### SpeculativeGraphBrancher

```python
brancher = SpeculativeGraphBrancher(engine, main_state)

# Create isolated branch
branch_state = brancher.create_branch("experiment-alpha")

# Mutate branch independently
branch_tx = KGTransaction(description="Experiment")
branch_tx.add_node("hypothesis:1", {"claim": "..."})
engine.commit(branch_tx, branch_state)

# Merge back to main — raises ValueError on conflict
commit = brancher.merge_branch("experiment-alpha")
```

## Conflict Detection

Merge validation performs three checks:

1. **Concurrent Deletion Check** — If a node modified by the branch was deleted in the main state since branch creation, a `ValueError("Merge Conflict: Node 'X' was deleted in main graph.")` is raised.
2. **Structural Diff** — `KGVersionEngine.diff()` computes the exact delta between main state and branch state to build a minimal merge transaction.
3. **Atomic Application** — The merge transaction is committed through the standard `engine.commit()` path, which itself supports auto-rollback on partial failure.

## Edge Cases

| Scenario | Behavior |
|---|---|
| Branch modifies node deleted from main | `ValueError` — clean merge conflict |
| Two branches add the same node ID | Second merge succeeds with `ADD_NODE` (idempotent — skips if already exists) |
| Empty branch (no mutations) | Returns a no-op `KGCommit` with `mutations_applied=0` |
| Branch references non-existent branch ID | `ValueError("Branch 'X' does not exist.")` |
| Exception during commit | Auto-rollback via `_do_rollback()` — main state unchanged |
| Concurrent branch creation | Safe — each branch is a `deepcopy` snapshot |

## OWL Integration

When the OWL Bridge (KG-2.2) is active, speculative branches can be used to test ontological consistency before committing:

1. Create a branch and apply proposed schema changes
2. Run `OWLBridge.materialize_rdf()` against the branch state
3. Validate with SHACL shapes against the branch's RDF graph
4. If validation passes, merge the branch back to main

This pattern enables **safe schema evolution** without risking production KG integrity.

## Implementation Details
- **Source Code**: [`kg_versioning.py`](file:///home/apps/workspace/agent-packages/agent-utilities/agent_utilities/knowledge_graph/core/kg_versioning.py) (358 lines)
- **Classes**: `KGVersionEngine`, `SpeculativeGraphBrancher`, `KGTransaction`, `KGCommit`, `KGDiff`, `KGMutation`, `MutationType`
- **Tests**: [`test_synergies.py`](file:///home/apps/workspace/agent-packages/agent-utilities/tests/unit/knowledge_graph/test_synergies.py)
- **Pillar**: KG
- **Dependencies**: `pydantic`, `copy.deepcopy`, `hashlib` (SHA-256 IDs)
