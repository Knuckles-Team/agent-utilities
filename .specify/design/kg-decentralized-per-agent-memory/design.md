# Design Document: Each agent gets its OWN memory pools plus an append-only collaboration trace — not one centralized shared bank

CONCEPT:AU-KG.memory.ahe-record-this-base

> Realised by `agent_utilities/harness/decentralized_memory.py:4-90`
> (module and class docstrings, `MemoryPool`, `DecentralizedMemory`), exported
> at `agent_utilities/harness/__init__.py:129` and consumed at
> `agent_utilities/harness/agentic_evolution_engine.py:382-403`, which reads
> and rewards the pools. Operationalises DecentMem (arXiv:2605.22721).

## Decision — private per-agent exploit/explore pools, with attribution recorded in a separate append-only trace

Each agent owns a pair of memory pools (exploit and explore) that no other
agent reads or writes. Who solved what is recorded separately, in an
append-only collaboration trace.

**The rejected alternative is named explicitly and its three failure modes are
enumerated** (`decentralized_memory.py:6`):

> *"The prior behaviour we are surpassing is a single *centralized* shared
> memory bank: every agent reads and writes one pool, which mixes provenance,
> leaks one agent's mistakes into another's recall, and destroys the signal of
> *who* actually solved each piece of a task."*

Those three are worth separating, because they are distinct problems that a
shared bank causes simultaneously:

- **Mixed provenance** — a recalled memory carries no reliable indication of
  which agent produced it, so its trustworthiness cannot be assessed.
- **Cross-contamination** — an agent's wrong turn becomes another agent's
  premise. In a shared pool a mistake is not contained to the agent that made
  it; it propagates as evidence.
- **Destroyed attribution** — if every agent writes to one pool, there is no
  per-agent signal of competence. That is fatal for a harness whose whole
  purpose is evolutionary selection: you cannot reward the agent that solved
  the task if you cannot tell which one did.

The design keeps sharing where sharing is safe. Isolating pools would, on its
own, lose the ability to learn across agents entirely — so attribution moves
into the append-only trace, which records *who solved what* without letting one
agent's working memory contaminate another's recall. Sharing conclusions with
provenance is separated from sharing raw memory.

The exploit/explore pair within each agent is the second axis: it keeps
speculative material from being recalled with the same authority as material
that has worked.

## Naming note

The concept id is a slugified fragment from the OKF-CIS rename and reads
poorly; two of its three marker sites additionally carry a legacy `/AHE-3.33`
citation tail that the marker grammar truncates. Neither is evidence that the
concept is junk — the decision is real, well-evidenced and load-bearing. The
marker *text* would benefit from a mechanical cleanup pass to strip the legacy
tail; that is separate from this decision.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/decentralized_memory.py`,
  `agent_utilities/harness/agentic_evolution_engine.py`.
- **Backward Compatible**: Yes — a new memory structure for the harness.
- **Known weak point**: isolation has a real cost that this design absorbs
  rather than solves — a genuinely useful discovery by one agent does not reach
  the others' recall at all, only the trace. Knowledge that *should* propagate
  now has no path to, and nothing measures how much duplicated work that
  causes across the collective.
