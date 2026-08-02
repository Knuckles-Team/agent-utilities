# Design Document: `submit()` dispatches the train LIVE and degrades to a durable enqueued job — it never raises when data-science-mcp is unreachable

CONCEPT:AU-KG.memory.live-data-science-mcp

> Realised by `agent_utilities/knowledge_graph/memory/weights_distillation.py:152-157`
> (`_DISPATCH_TIMEOUT`), `:717-767` (`_default_dispatch`) and `:897-948`
> (`poll`). Introduced by commit `4818558e` ("feat(kg): wire LIVE
> memory→weights data-science-mcp dispatch + status poll"), whose message opens
> *"Follow-on to KG-2.316 (memory→weights EXPORT)."*

## Decision — flip the default from enqueue-only to live dispatch, but keep the enqueue path as the degrade route rather than deleting it

`CONCEPT:AU-KG.memory.memory-weights-distillation-export` established the
torch-free export and deliberately stopped short of running anything: its own
commit says *"the live LoRA train is the documented integration point"*, i.e.
the default `submit()` materialised a corpus, registered a `TrainingJob` node,
and stopped.

This decision changes that default. `submit()` now dispatches the LoRA/SFT
train live to `data-science-mcp` via the `train_model` workflow over
`graph_orchestrate` / `graph_workflows execute`, bridged sync→async and bounded
by a hard wall clock (`_DISPATCH_TIMEOUT`, 45s). The `TrainingJob` node is
marked `running` with the remote run handle, and `poll()` (`:897-948`) reads
real status back.

The commit states the substitution explicitly — the default now dispatches live
*"instead of only materializing a corpus + enqueuing a job node"*.

**The rejected alternative is therefore the previous default, and the shape of
the rejection is the interesting part: it was demoted, not deleted.** The
commit describes the degrade path — *"any unreachable/failed data-science-mcp
falls back to a durable `enqueued` job ... and never raises."*

Three properties are being chosen together here:

- **Live by default**, because a distillation that requires a second manual
  step to actually train is a pipeline nobody completes.
- **Bounded**, because the dispatch is a synchronous bridge into a remote
  workflow; without a hard wall clock an unresponsive trainer would hang the
  caller rather than fail it.
- **Never raises.** This is the load-bearing one. Distillation is background
  self-improvement, not a user request. An exception because the training agent
  is down would propagate into whatever maintenance pass invoked it, so an
  optional capability being unavailable would break a non-optional one. Falling
  back to the durable enqueued job preserves the work — the corpus is already
  materialised and the job node exists — so nothing is lost except immediacy.

The old default surviving as the fallback is what makes "never raises"
implementable rather than a swallowed error: there is a real, durable,
resumable state to degrade *to*.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/memory/weights_distillation.py`;
  the `data-science-mcp` agent and the workflow-execute path.
- **Backward Compatible**: The default behaviour of `submit()` changes from
  enqueue to dispatch. Callers that relied on `submit()` being cheap and
  non-blocking now pay up to `_DISPATCH_TIMEOUT`.
- **Known weak point**: "never raises" means a persistently unreachable trainer
  is indistinguishable, from the caller's side, from a working one — every
  submit succeeds and silently accumulates `enqueued` jobs. Nothing here alerts
  on a growing backlog or on a dispatch success rate that has fallen to zero,
  so the degrade path can be the steady state without anyone noticing.
