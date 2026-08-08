# Design Document: The engine owns deterministic memory PRIMITIVES; AU owns the POLICY — and that policy operates on a bounded working set, never a global scan

CONCEPT:AU-KG.memory.drive-one-agent-native

> Realised by `agent_utilities/knowledge_graph/memory/lifecycle.py:1-38`
> (module docstring stating the split), `:93-121` (`MemoryLifecycleConfig`,
> including `max_working_set`) and `:175-183` (`MemoryLifecycle`). The engine
> primitives it drives are `create_summary_node` (EG-220), `consolidate`
> (EG-221) and `reinforce`/`decay`/`evict`/`maintain` (EG-222). Introduced by
> commit `37d449d` ("feat(memory): agent-native memory lifecycle loop").

## Decision 1 — the engine stores and executes deterministically; AU decides

The introducing commit states the boundary in one line: *"The engine is
deterministic and stores/executes; AU owns the POLICY the paper leaves to the
agent."*

The engine (Rust) exposes storage-level primitives only — create a summary
node, consolidate a cluster, reinforce/decay/evict, run maintenance. Every
*judgement* lives in AU: which working set to operate on, when a cluster is
ripe for consolidation, when maintenance should run, and the LLM call that
generates the summary text.

This is partly constraint-driven and should be described honestly as such: the
summary generation is an LLM call, and an LLM call cannot run inside the Rust
engine, so *some* split was forced. What was genuinely chosen is where to put
the line — the engine could have owned scheduling and selection while calling
back out for text, which would have made the engine the policy-holder with AU
as a text service. Keeping every judgement on the AU side means the engine
stays deterministic and testable, and the policy can change without an engine
release.

## Decision 2 — localized maintenance over a bounded working set, never a global scan

This is the trade-off with a real rejected alternative, and the module states
it directly (`lifecycle.py:10-11`):

> *"select a bounded working set + the cluster of episodic memories ripe for
> consolidation ... 'localized maintenance' per the paper: we operate on a
> selected working set, never a global scan."*

`MemoryLifecycleConfig.max_working_set` (`:93-121`) is the hard bound that
realises it.

**The rejected alternative is a global scan** — sweep all memory, consolidate
everything eligible. It is more complete and it is what "run maintenance"
naively means. It was rejected because its cost scales with total memory size
rather than with how much has changed, which makes it exactly the wrong shape
for a loop meant to run continuously: the longer the system has been
accumulating memory, the more expensive each maintenance pass gets, so the
mechanism that keeps memory healthy becomes unaffordable precisely as memory
grows. A bounded working set makes each pass cost the same regardless of store
size.

The cost accepted in exchange is coverage: a memory outside every selected
working set is never consolidated. Maintenance becomes eventual and
probabilistic rather than exhaustive.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/memory/lifecycle.py` and
  the engine primitives it drives; scheduled via `schedule_engine.py`.
- **Backward Compatible**: Yes — additive lifecycle loop.
- **Known weak point**: nothing guarantees fair coverage across the store.
  Working-set selection could systematically favour recently-touched or
  well-connected memory, leaving a cold region permanently unmaintained, and
  there is no metric reporting what fraction of eligible memory has been
  visited. The "eventual" in eventual consolidation is not currently bounded.
