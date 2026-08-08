# Design Document: The specialist roster is grown from the observed task stream, not authored by hand

CONCEPT:AU-ORCH.routing.emergent-specialization-discovery-pass

> Realised by `agent_utilities/graph/specialization_discovery.py:1-115`
> (`SpecializationDiscovery.discover`, `_cluster`). Introduced by commit
> `21069306` ("feat: multi-agent collective — market, specialization,
> hierarchical coord").

## Decision — cluster the *failing and expensive* task stream and propose a new archetype where coverage is thin

The module docstring names the prior condition exactly: *"AU's roles were
statically authored; nothing analyzed the task stream to discover where a new
specialist was needed."* Roles existed and worked, but the roster only ever
changed when a human noticed a gap and wrote one.

`SpecializationDiscovery.discover` closes that loop from evidence. It embeds
and clusters the task stream, and for each cluster computes the maximum
similarity to the existing archetypes. A cluster whose best match falls below a
coverage floor is a niche the current roster does not serve, and the pass
proposes a new specialist archetype for it.

Two aspects of the input selection are the actual design content:

- **It clusters the failing or expensive stream, not all traffic.** Clustering
  everything would mostly rediscover the niches already well served — those are
  the largest clusters by volume. The signal for "a specialist is missing" is
  not "many tasks look like this" but "tasks that look like this go badly or
  cost too much". Selecting on outcome is what makes the discovered clusters
  actionable.
- **The threshold is a coverage floor over similarity to existing archetypes,
  not cluster size.** A small cluster nothing can serve is a real gap; a huge
  cluster the planner already handles is not.

**The rejected alternative is the status quo it replaces: a hand-authored
roster.** It was not rejected as wrong — hand-authored roles are correct and
remain the bootstrap — but as unable to keep up. A static roster degrades
silently as the workload shifts, and the cost of a missing specialist is paid
diffusely (as failures and expensive escalations spread across many turns)
rather than announcing itself, so nobody notices until someone audits.

## Risk Assessment

- **Blast Radius**: `agent_utilities/graph/specialization_discovery.py`. The
  pass *proposes* archetypes; it does not silently install them.
- **Backward Compatible**: Yes — additive discovery over an existing roster.
- **Known weak point**: the coverage floor and the clustering parameters are
  unvalidated constants, and the pass has no notion of whether a *proposed*
  archetype turned out to be useful once adopted. It can therefore propose
  repeatedly into the same thin region without learning that previous proposals
  there did not help. It also inherits the selection bias of its input: tasks
  that fail for reasons unrelated to specialisation (a flaky backend, a bad
  prompt) cluster just as readily as genuine capability gaps.
