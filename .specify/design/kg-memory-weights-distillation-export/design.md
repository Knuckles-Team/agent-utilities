# Design Document: Memory→weights distillation is an EXPORT in core — a torch-free corpus plus a typed job spec — with the GPU train handed to data-science-mcp

CONCEPT:AU-KG.memory.memory-weights-distillation-export

> Realised by `agent_utilities/knowledge_graph/memory/weights_distillation.py:1-51`
> (module docstring and "Design notes"), `:105-127`
> (`DATA_SCIENCE_MCP_CONTRACT`), `:214-364` (`DistillationTargetSpec`,
> `DistillationCorpus`, `DistillationJob`), `:378-663`
> (`MemoryWeightsDistiller.select` / `export` / `_build_handoff`) and `:958`
> (the `graph_analyze action=distill_memory` action-core). Introduced by commit
> `8b2a40e9`.

## Decision — the dependency boundary decides where the work runs

Distilling accumulated memory into model weights is a training workload, and
the natural implementation puts training in the module that owns the memory.
This decision refuses that, on a constitutional rule the module docstring cites
directly:

> *"Dependency discipline (see `AGENTS.md`): core stays torch-free. This module
> only *reads* memory and *emits a corpus + a job spec* ... The actual GPU
> fine-tune runs in `agents/data-science-mcp`."*

So core's half is deliberately boring and fully deterministic: a bounded read
of consolidated and procedural `:Memory` nodes, rendered into a JSONL SFT /
preference corpus with a checksum, plus a typed `DistillationTargetSpec`
describing what should be trained. `DATA_SCIENCE_MCP_CONTRACT` (`:105-127`) is
the typed hand-off — the seam is a declared contract rather than an import.

**The rejected alternative is embedding the ML training stack — numpy, torch,
transformers, peft — into core so the fine-tune can run in-process.** It is the
shorter path and it keeps one process. It was rejected because core is imported
by everything in the ecosystem: every MCP server, every agent, every CLI. A
training dependency in core is a multi-gigabyte install, a CUDA/platform
constraint, and a large new attack and breakage surface imposed on every
consumer, the overwhelming majority of which will never train anything. Putting
the GPU work behind an MCP boundary means only the agent that actually trains
pays for the ability to.

The determinism and checksum on the export side are what make that boundary
safe rather than merely tidy: the corpus is a reproducible artifact that can be
verified on the far side of a process boundary, so the hand-off does not
require trusting the transport.

## Relationship to `live-data-science-mcp`

This concept is the export contract and the torch-free-core boundary. The
*dispatch behaviour* — whether `submit()` fires the train live or enqueues a
durable job — is a separate, later decision carried by
`CONCEPT:AU-KG.memory.live-data-science-mcp`, delineated in this same file at
`:42`. The two are documented separately because they trade off different
things: this one trades convenience for dependency hygiene; that one trades
immediacy for degradation behaviour.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/memory/weights_distillation.py`
  and the `data-science-mcp` agent on the far side of the contract.
- **Backward Compatible**: Yes — a new capability; nothing previously trained
  from memory.
- **Known weak point**: `DATA_SCIENCE_MCP_CONTRACT` is a hand-maintained typed
  contract across a process boundary, so the two sides can drift. Core cannot
  import the trainer to type-check against it — that is the whole point of the
  boundary — so a change on the `data-science-mcp` side is caught at runtime
  when a job is submitted, not at build time.
