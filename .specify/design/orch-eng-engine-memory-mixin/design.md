# Design Document: Memory CRUD is a focused mixin extracted from the engine, not inline on it or a separate composed class

CONCEPT:AU-ORCH.execution.engine-memory-management

> `agent_utilities/knowledge_graph/core/engine_memory.py` (`MemoryMixin`) —
> the single source file carrying this concept (1 marker site), composed into
> `IntelligenceGraphEngine` in `agent_utilities/knowledge_graph/core/engine.py`.

## The real decision

`engine_memory.py`'s own module docstring states the origin directly:
*"Extracted from engine.py. Contains CRUD operations for memory nodes"*
(`engine_memory.py:5`). `MemoryMixin` (`engine_memory.py:203-204`, *"Memory
node CRUD capabilities for the KG engine"*) is one of a family of focused
mixins `engine.py` composes together, named in `engine.py`'s own docstring:

> *"The engine is composed of focused mixins for maintainability:
> `engine_query.py`: Query, search, and retrieval methods.
> `engine_memory.py`: Memory CRUD operations.
> `engine_ingestion.py`: Episode, MCP, A2A, and skill ingestion.
> `engine_registry.py`: Identity, prompt, resource, and codemap management."*
> (`engine.py:12-16`)

`engine.py:44-53` shows the composition mechanically: `IntelligenceGraphEngine`
imports and mixes in `MemoryMixin` alongside `QueryMixin`, `IngestionMixin`,
`MCPDiscoveryMixin`, `TaskManagerMixin`, `RegistryMixin`, `AHEMixin`,
`EnterpriseEngineMixin`, `FederationMixin`, `FinanceEngineMixin`,
`InfrastructureEngineMixin`, and `MachineLearningEngineMixin` — a dozen-plus
mixins, each in its own file, each typed against a shared `_EngineProtocol`
(`engine_memory.py:13-15`) so it can call sibling-mixin methods without a
circular import back to `engine.py` itself.

`GraphComputeEngine` (`engine.py:53`, the base class `IntelligenceGraphEngine`
ultimately builds on) is the single operational authority for storage,
retrieval, and native graph algorithms — the docstring is explicit that
*"Optional backends are explicit interoperability or mirror targets; they do
not introduce a second read authority"* (`engine.py:9-11`). The mixin
decomposition is therefore a decomposition of **one** engine's surface area
into maintainable files, not a decomposition of authority into multiple
competing engines.

## The rejected alternative

The rejected alternative is named directly by the phrase *"extracted from
engine.py"*: a single monolithic `IntelligenceGraphEngine` class with query,
memory, ingestion, registry, AHE, enterprise, federation, finance,
infrastructure, and ML/RLM logic all defined inline in one file. That is
strictly what existed before the extraction — the comment records a real
prior state, not a hypothetical. A monolith of that shape becomes
increasingly hard to navigate and review as each concern grows (the query
mixin alone, `engine_query.py`, carries a large `_BIOMIMICRY_KEYWORDS` table
used for innovation-discovery search), and every unrelated change to any one
concern touches the same file.

The alternative on the *other* side — a fully separate `MemoryManager` class
composed by delegation (`self.memory = MemoryManager(self)`) rather than by
mixin inheritance — is implicitly rejected too: the mixin shares `self`
directly with every other mixin via `_EngineProtocol`, so `add_memory` and
friends can be called as ordinary methods on the one `IntelligenceGraphEngine`
instance (`engine.add_memory(...)`) rather than through an extra `.memory.`
indirection, preserving the "one engine" contract from the caller's
perspective while still splitting the implementation across files.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/core/engine_memory.py`,
  `agent_utilities/knowledge_graph/core/engine.py`,
  `agent_utilities/knowledge_graph/core/_engine_protocol.py`.
- **Backward Compatible**: Yes — a pure internal file-organization decision;
  the public `IntelligenceGraphEngine` method surface is unchanged by which
  file a method is defined in.
- **Known weak point**: the shared `_EngineProtocol` base makes every mixin
  implicitly coupled to the full attribute surface every other mixin expects
  to exist on `self` — a mixin can be edited in isolation, but reasoning about
  whether it is *safe* to edit in isolation requires knowing what the other
  ~12 mixins assume is present, which the file split does not by itself make
  visible.
