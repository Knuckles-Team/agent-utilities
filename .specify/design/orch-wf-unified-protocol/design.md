# Design Document: Orchestrator conformance is structural (PEP 544 `Protocol`), not inheritance

CONCEPT:AU-ORCH.execution.unified-orchestration-protocol

> `agent_utilities/orchestration/protocol.py` (`OrchestratorProtocol`).

## Decision — `OrchestratorProtocol` is a `typing.Protocol`, so `Orchestrator`, `AgentOrchestrationEngine`, `KGDrivenExecutionEngine`, and `ParallelEngine` all conform automatically via structural subtyping

The module docstring states the contract directly: any class that implements
`dispatch()` and `get_status()` with compatible signatures satisfies
`OrchestratorProtocol` **without inheriting from it** — PEP 544 structural
typing, marked `@runtime_checkable` so `isinstance(orch, OrchestratorProtocol)`
also works at runtime, not just for static type checkers. Four pre-existing,
independently-implemented classes conform this way today:
`Orchestrator` (`orchestration/manager.py`), `AgentOrchestrationEngine`
(`graph/graph_orchestrator.py`), `KGDrivenExecutionEngine`
(`graph/dynamic_graph_orchestrator.py`), and `ParallelEngine`
(`graph/parallel_engine.py`) — none of which needed to change their class
hierarchy to gain conformance. The doc also draws an explicit boundary on
scope: domain-specific facades (`SDDOrchestrator`, `EngineeringPatternOrchestrator`,
`DynamicToolOrchestrator`) are *consumers* of orchestration, not task
dispatchers themselves, and are deliberately NOT required to implement this
protocol.

**The rejected alternative is a shared abstract base class (`ABC`) with
`dispatch()`/`get_status()` as abstract methods**, requiring every conforming
orchestrator to inherit from it. That would have forced a retrofit of four
already-shipping, independently-evolved classes into a common inheritance
hierarchy — a structural change to each one's class definition purely to
satisfy a typing contract, with the usual multiple-inheritance/MRO risk if
any of the four already extends something else. `Protocol` gets the same
type-checking benefit (a function that accepts `OrchestratorProtocol` can be
called with any of the four, and a type checker verifies the shape) with zero
change to any of their definitions — conformance is a property of the
methods a class already has, not a declared relationship it has to opt into.

## Risk Assessment

- **Blast Radius**: `agent_utilities/orchestration/protocol.py` only — a pure
  typing artifact; it constrains callers that choose to type against it, it
  does not itself route or execute anything.
- **Backward Compatible**: Yes — adding or removing conformance to a
  `Protocol` never requires changing a conforming class's declared bases.
- **Known weak point**: structural typing means conformance is checked by
  method *signature* shape only — `dispatch()`/`get_status()` existing with
  compatible types is sufficient even if their actual semantics diverge
  significantly between the four conforming classes (e.g. what "status"
  means for a `ParallelEngine` wave vs. a single `Orchestrator` dispatch), so
  the protocol guarantees callability, not behavioral equivalence.
