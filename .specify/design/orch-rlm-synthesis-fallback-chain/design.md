# Design Document: Multi-agent output synthesis is a fallback chain (rlm → hierarchical → flat), never one strategy dumped whole into a context window

CONCEPT:AU-ORCH.execution.rlm-synthesis-failed-falling

> `agent_utilities/graph/parallel_engine.py` (`_synthesize` and its four strategy
> implementations), `agent_utilities/models/execution_manifest.py` (`SynthesisSpec`),
> `agent_utilities/core/config.py` (the `SYNTHESIS_STRATEGY`/`SYNTHESIS_RATIO` knobs), and
> `docs/architecture/configuration.md:569` (the operator-facing catalog entry).

## Decision — four synthesis strategies behind one dispatcher, with the expensive one falling back to the cheaper one on failure, never to raw context-stuffing

`ParallelEngine._synthesize()` (`agent_utilities/graph/parallel_engine.py:1133-1180`) is the
single dispatcher for merging parallel-agent outputs, and its docstring names the governing
principle: "The key insight: outputs are stored as Pydantic objects and processed
programmatically, never dumped into context windows" (`parallel_engine.py:1142-1145`).
`SynthesisSpec.strategy` (`agent_utilities/models/execution_manifest.py:75-89`) selects among
four concrete implementations:

- **`flat`** (`parallel_engine.py:1182-1193`) — simple markdown concatenation, for small
  agent counts.
- **`hierarchical`** (`parallel_engine.py:1195-1256`) — groups results by `spec.ratio`
  (default 10, `SYNTHESIS_RATIO`), LLM-summarizes each group, then recurses on the
  sub-summaries if there are still more than `ratio` of them.
- **`rlm`** (`parallel_engine.py:1258-1313`) — for "massive-scale (50+ agent) output
  processing," serializes results as JSON into an `RLMEnvironment` context (truncated to 2000
  chars per output for the metadata view) and lets RLM programmatically process the outputs
  rather than concatenating them into one prompt.
- **`progressive`** (`parallel_engine.py:1315-1339`) — streams synthesis incrementally,
  merging one new result into a running summary at a time.

`synthesis_strategy` defaults to `"auto"` (`agent_utilities/core/config.py:5354-5356`, `"'auto'
selects based on agent count and output size"`), and `docs/architecture/configuration.md:569`
documents the same four-way choice for operators.

**The rejected alternative** is a single fixed synthesis approach — in practice, always
flattening every agent's output into one prompt for a final LLM pass. That is explicitly what
`_flat_synthesis` *is*, and the code keeps it only as the strategy for small agent counts and
as the ultimate fallback (`_synthesize._` default case at `parallel_engine.py:1179-1180`
falls to `_flat_synthesis` for any unrecognized strategy value) — not as the general-purpose
approach, because it is exactly the "dumped into context windows" failure mode the module
docstring rejects at scale. `hierarchical`/`progressive`/`rlm` each exist because flat
concatenation degrades or breaks outright once agent count or per-agent output size grows past
what one prompt can hold.

**A second, coupled decision**: the `rlm` strategy is not treated as unconditionally
available — it has its own fallback chain, and the concept's marker text ("RLM synthesis
failed, falling back...") names it directly. `_rlm_synthesis()` wraps the entire RLM path in a
`try`/`except`:

```python
except Exception as e:
    logger.warning(
        "[CONCEPT:AU-ORCH.execution.rlm-synthesis-failed-falling] RLM synthesis failed, falling back to "
        "hierarchical: %s", e,
    )
    return await self._hierarchical_synthesis(results, spec, query, graph_deps)
```

(`parallel_engine.py:1307-1313`). The fallback chain nests one level further: within
`hierarchical`, `_synthesize_group()`'s own per-group LLM call is wrapped the same way —
on failure it logs and returns the raw joined text instead (`parallel_engine.py:1359-1379`,
`"Group synthesis failed, using flat"`). So a failure at the most capable tier degrades one
tier at a time (`rlm` → `hierarchical` → raw-joined-text) rather than either propagating the
exception (losing the whole synthesis) or jumping straight to the weakest strategy
(discarding whatever partial structure the intermediate tier could still provide). The
rejected alternative here is letting an RLM synthesis failure bubble up as a hard error — the
code treats output synthesis as something that must always produce *some* answer, with quality
degrading gracefully rather than availability failing outright.

## Risk Assessment

- **Blast Radius**: `agent_utilities/graph/parallel_engine.py`,
  `agent_utilities/models/execution_manifest.py`, `agent_utilities/core/config.py`.
- **Backward Compatible**: Yes — `"auto"` preserves prior behavior for existing callers that
  never set `SYNTHESIS_STRATEGY` explicitly.
- **Known weak point**: the fallback chain is fully silent to the caller — a `rlm`-requested
  synthesis that silently degrades to flat-joined raw text still returns `200`/success with no
  signal in the return value itself that the requested strategy did not actually run; only the
  `logger.warning` records it.
