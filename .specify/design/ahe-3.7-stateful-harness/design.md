# Design: Transactional State Manager and Sensory Validation (CONCEPT:AU-AHE.evaluation.backtest-harness)

## 1. Context & Motivation

This document details the architectural design for extending `agent-utilities` with stateful harness capabilities inspired by **arXiv:2605.18747**.

Specifically, we address two high-priority structural gaps:
1. **Transactional State Convergence (`BranchMergeStateLocker`)**: Introducing git-like parallel state branching and three-way merging to resolve race conditions in concurrent multi-agent workspaces.
2. **Sensory Execution Verification (`ContractValidator`)**: Implementing declarative pre-conditions and post-conditions for execution nodes to ensure safety and system correctness under sandbox constraints.

---

## 2. Component Diagram (C4 Level 3)

```mermaid
C4Component
    title agent-utilities — Harness State & Sensory Verification (Level 3)

    Container(orch_step, "expert_executor_step", "pydantic-graph Step", "Routes and executes specialists")

    Component(validator, "ContractValidator", "Python", "Validates pre- and post-conditions for nodes")
    Component(locker, "BranchMergeStateLocker", "Python", "Manages state forks, branch updates, and merges")

    ContainerDb(redis_db, "Redis Cache (Optional)", "Key-Value Store", "Caches branch states and locks")
    ContainerDb(local_db, "Local In-Memory Store", "Python dict", "Fallback thread-safe local persistence")

    Rel(orch_step, validator, "Invokes pre- and post-condition checks")
    Rel(orch_step, locker, "Forks states on branch, merges back on completion")
    Rel(locker, redis_db, "Saves/deletes Redis keys with version WATCH")
    Rel(locker, local_db, "Fallback memory reads/writes")
```

---

## 3. High-Level Design (Pillar Integration)

### 3.1 BranchMergeStateLocker
The `BranchMergeStateLocker` inherits from `OptimisticStateLocker` to preserve all existing locking functionality. It adds three core abstractions:
* **Fork**: Creates a staged parallel state copy associated with a unique branch name.
* **Update**: Modifies the branched state independently from the main thread.
* **Merge**: Integrates branch changes back to the base branch, executing three-way recursive merges and custom conflict resolvers.

### 3.2 ContractValidator
The `ContractValidator` registers a schema contract (`ToolContract`) per step/node. During execution:
* **Pre-Check**: Asserts environmental conditions before launching an agent.
* **Post-Check**: Asserts structural and logical criteria against output schemas (using Pydantic validation) before propagating state transitions.

---

## 4. Concurrency Model Decision

We implement a **hybrid Redis/Local memory model**. State branches are kept lightweight and fast in memory/Redis hash structures, bypassing disk-write delays. Physical file modification (source code files) remains backed by traditional git/filesystem operations.
