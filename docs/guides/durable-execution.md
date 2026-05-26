# Graph-Native Durable Execution (CONCEPT:ECO-4.0)

Agent Utilities implements a sophisticated **Graph-Native Durable Execution Engine** designed specifically for high-assurance, fault-tolerant algorithmic trading and complex multi-leg workflows.

## Overview
Unlike standard isolated execution databases (such as DBOS), this capability persists workflow checkpoints directly into the `agent-utilities` **LadybugDB Cypher Knowledge Graph**. This enables a unified intelligence view where execution states, trading signals, and agent memories all live in the exact same topological space.

### Key Benefits
- **Fault-Tolerant Auto-Hedging**: If the agent experiences a crash or unexpected restart mid-trade, the orchestrator retrieves the exact topological checkpoint upon restart. Multi-leg trades (e.g. Iron Condors) are fully recoverable.
- **Topological Integrity**: `DurableExecutionNode` instances are linked via `CheckpointEdge` relationships directly to the `TeamConfig` and `Specialist` nodes that generated them, providing a full causal provenance graph of why a trade was executed.
- **Zero External Dependencies**: By mapping persistence to the native LadybugDB graph, we eliminate the need for maintaining separate DBOS or Redis clusters.

## Implementation Details
The `DurableExecutionManager` (located in `agent_utilities.orchestration.durable_execution`) provides three main primitives:
1. `save_checkpoint(node_id, state, status='PENDING')`: Serializes the active execution context and merges it into the Knowledge Graph.
2. `resume_session()`: Queries the graph for the most recent `PENDING` checkpoint for the active session, allowing seamless resumption.
3. `mark_completed(node_id)`: Transitions the node to a terminal state to prevent replay.

### Example
```python
from agent_utilities import DurableExecutionManager

manager = DurableExecutionManager(session_id="trade_123")
manager.save_checkpoint("execution_leg_1", {"asset": "AAPL", "qty": 100})

# System crashes...
# Upon restart:
last_state = manager.resume_session()
if last_state:
    print(f"Resuming at {last_state['node_id']} with {last_state['state']}")
```
