# Naming Conventions

## Class Naming Rules

This document establishes naming conventions to prevent further **Engine proliferation** (50+ `*Engine` classes observed at audit time).

### `*Engine` — Standalone Systems

Reserved for **top-level entry points** that own their own lifecycle (startup → operation → shutdown). These are the primary interfaces consumers import.

**Examples:**
- `IntelligenceGraphEngine` — KG core lifecycle owner
- `ParallelEngine` — Canonical parallel task execution
- `GraphComputeEngine` — Rust-backed graph algorithms

**Rule:** If your class is a mixin, strategy, or helper, it is **not** an Engine.

---

### `*Mixin` — Composed via MRO

For classes that are mixed into an Engine via multiple inheritance. They add behavior but do not function standalone.

**Examples:**
- `QueryMixin` — Adds query methods to IntelligenceGraphEngine
- `PersistenceMixin` — Adds save/load lifecycle hooks
- `ObservabilityMixin` — Adds metrics/tracing

**Rule:** Mixins must not have `__init__` (or only call `super().__init__`). They must not be instantiated directly.

---

### `*Strategy` — Pluggable Algorithms

For classes implementing a specific algorithm or policy that can be swapped. These follow the Strategy pattern.

**Examples:**
- `CompactionStrategy` — Token-aware compaction policy
- `RoutingStrategy` — Task-to-specialist routing logic
- `ScoreStrategy` — Relevance scoring algorithm

**Rule:** Strategies are stateless or nearly stateless. They implement a common interface and are injected into Engines.

---

### `*Coordinator` — Cross-System Orchestration

For classes that coordinate across multiple systems without owning execution themselves.

**Examples:**
- `DistributedCoordinator` — NATS/local task routing
- `RecoveryDaemon` — Homeostatic recovery
- `UnifiedMemoryManager` — Memory lifecycle coordination

**Rule:** Coordinators delegate execution to other components. They are orchestration glue, not execution engines.

---

### `*Protocol` — Structural Typing Interfaces

For `typing.Protocol` classes that define structural contracts.

**Examples:**
- `OrchestratorProtocol` — Dispatch + status interface
- `CapabilityHandlerProtocol` — Capability execution interface

**Rule:** Protocols must use `@runtime_checkable` when isinstance checks are needed at runtime.

---

## Migration Guide

Existing class names are **NOT** being renamed in this phase to avoid import breakage. This convention applies to **new code only**.

When creating new classes, check this guide before naming. If you find yourself creating a new `*Engine`, verify it truly owns a standalone lifecycle. Most new additions should be `*Strategy`, `*Mixin`, or `*Coordinator`.

Future rename work can use `typing.TypeAlias` for deprecation-safe migration:

```python
# Gradual migration pattern
from ._new_name import QueryMixin

# Deprecated alias for backward compatibility
EngineQueryExtension = QueryMixin  # TODO: remove in v2.0
```
