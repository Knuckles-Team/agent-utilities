# Design Document: Three independently-proposed memory banks converge into ONE graph-native typed store that merges instead of deleting

CONCEPT:AU-KG.memory.memory-lifecycle-manager

> Realised by `agent_utilities/harness/evolving_memory.py:4-27` (module
> docstring naming the three converged proposals) and `:77-124`
> (`EvolvingMemoryStore`, `MemoryBank`, `add()` with dedup-by-signature).
> Introduced by commit `84d49ec8`.

## Decision — build the union once rather than building three overlapping stores

Three separate research-assimilation proposals under
`.specify/specs/research-evolution-20260606/` each called for a memory store,
and the module docstring names all three: **b4-03** (MEMO insight bank),
**b8-06** (Web2BigTable skill banks), and **b5-02** (BioMedArena typed
four-bank workspace). Each was independently reasonable, and each would have
produced a store with its own record type, its own append semantics and its own
lifecycle.

The decision is to treat that as one requirement rather than three.
`EvolvingMemoryStore` is the "build-once convergence": typed `MemoryBank`
records general enough to express all three shapes, one persistence path, one
lifecycle.

**The rejected alternative is implementing the three proposals as proposed** —
which is the default outcome, since each was separately specified and each
could have been built without reference to the others. It was rejected because
three parallel stores would fragment memory at the point where fragmentation
costs the most. Three stores means three answers to "what do we know", three
reconciliation policies, three sets of retrieval semantics, and no way to ask a
question that spans them. The overlap between the three proposals was large
enough that most of what each needed, the others also needed.

Two properties of the converged store carry that convergence:

- **Dedup by signature** on `add()` — three sources feeding one store will
  produce the same insight more than once, so deduplication is a precondition
  for merging the banks rather than an optimisation.
- **Reconcile by merge, never delete.** A record that conflicts with an
  existing one is merged into it. With one store serving several producers,
  delete-on-conflict would let one producer silently destroy another's
  contribution — the failure mode that separate stores at least made obvious.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/evolving_memory.py` and the
  harness paths that persist typed memory records.
- **Backward Compatible**: Yes — net-new store; the three proposals were never
  separately implemented.
- **Known weak point**: the generality that lets one record type express three
  proposals' needs also makes the schema permissive. `MemoryBank` records can
  represent shapes none of the three proposals intended, and merge-not-delete
  means a bad record is never removed, only merged with. The store has no
  compaction or retirement path, so incorrect entries accumulate rather than
  being corrected.
