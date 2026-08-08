# Design Document: Checkpoint at the dispatcher transition through the consolidated `CheckpointManager` — restoring a capability a refactor had silently dropped

CONCEPT:AU-ORCH.routing.transition-state-checkpoint

> Realised by `agent_utilities/graph/_router_impl.py:972-994`, which saves graph
> state at the dispatcher transition boundary via the consolidated
> `core/checkpoint` `CheckpointManager` (KG backend).

## Decision — route checkpointing through the consolidated manager, and treat the silent-import-failure that preceded it as the thing being fixed

The comment at the marker site records the whole history:

> *"Routes through the consolidated CheckpointManager (KG backend). The old
> `graph/state_checkpoint.StateCheckpointer` was merged into `core/checkpoint`
> (Plan 03 Step 8); the prior import silently failed, dropping this capability
> — restored here."*

Two decisions are stacked here and both are worth stating.

The first is *where* to checkpoint: at the dispatcher transition boundary. That
is the point at which the routing decision is settled and the state about to be
handed to execution is coherent — checkpointing mid-route would capture a state
that is not a valid resume point.

The second, and the reason this concept exists at all, is *through what*.
`StateCheckpointer` was consolidated into `core/checkpoint` by an earlier
refactor. The call site's import of the old path did not fail loudly — it
failed silently, and the effect was not an error but an *absence*: graph state
simply stopped being checkpointed. Nothing raised, no test covered the negative
("state was saved"), and the system continued to work in every way except that
runs could no longer be resumed.

**The rejected alternative is the state this replaced — a dead import path
left in place after a consolidation.** It is worth naming as a rejected
alternative rather than a mere bug because the general form is a design
question: a capability wired through an optional/guarded import degrades to
nothing when the import target moves, and that degradation is invisible. The
fix does not add a try/except or a fallback; it points at the one consolidated
manager, so there is a single path that either exists or fails loudly at
import.

## Risk Assessment

- **Blast Radius**: `agent_utilities/graph/_router_impl.py`, plus the
  `core/checkpoint` `CheckpointManager` it now depends on.
- **Backward Compatible**: Yes, and strictly restorative — checkpointing was
  already meant to be happening.
- **Known weak point**: nothing asserts that a checkpoint was actually written
  at the transition. The original regression was invisible precisely because
  the absence of a checkpoint is not an observable error at the time it
  happens; it only becomes visible when a resume is attempted much later. That
  detection gap is unchanged by this fix — a future consolidation could
  reproduce it.
