# Design Document: One universal context plane; every domain (code, ops, deploy, tickets, troubleshooting) is a registered provider, not a new subsystem

CONCEPT:AU-KG.retrieval.route-question-its-domain ·
CONCEPT:AU-KG.retrieval.kg-2 ·
CONCEPT:AU-KG.retrieval.kg-3 ·
CONCEPT:AU-KG.retrieval.kg-4 ·
CONCEPT:AU-KG.retrieval.ops-context

> `agent_utilities/knowledge_graph/retrieval/context_plane.py` (the registry +
> dispatch), with providers in `deploy_context.py`, `entity_context.py`,
> `troubleshoot_context.py`, and `ops_context.py`.

## Decision — `synthesize_context(domain, query, intent)` is the ONE pattern; new domains register a provider, they do not fork a new answer surface

`CONCEPT:AU-KG.retrieval.route-question-its-domain`

`context_plane.py:4-20` states the generalization directly: `code_context`
(KG-2.134) was the first instance of a grounded, cited answer with a stable
`{answer, citations, capability_id, used_primitives}` shape; this module
"generalizes it so **any** domain — code, ops/health, deployment, tickets,
processes — registers a provider and inherits the surface, the
`file:line`/id citation contract, and the action-outcome reward loop
(AHE-3.62)." A provider is a plain `fn(engine, *, query, intent, **opts) ->
dict` returning the standard answer shape; built-ins are lazy-imported
(`_BUILTIN_PROVIDERS`) so the plane has zero import-time dependency on any one
provider's heavy modules, and providers import the plane rather than the other
way around (no import cycle).

**The rejected alternative** is named by what the docstring calls out as the
point: "the enterprise cockpit is not a new subsystem — it is *more providers
on this one plane*." A domain-specific implementation (a bespoke "ops
dashboard," a bespoke "deploy status page," a bespoke "ticket search") would
each reinvent the citation contract, the intent-routing convention, and the
reward-loop hook independently — and would drift from each other the moment
one of them evolved its answer shape without the others following. Four
concrete providers below demonstrate the pattern is real, not aspirational —
each is a genuinely different piece of logic, unified only by the contract
they satisfy.

### Pointer — `CONCEPT:AU-KG.retrieval.kg-2` (the `deploy` provider)

`deploy_context.py:4-14`. Answers "where does this code run, and is my change
live?" by synthesizing deployment reality from git (canonical checkout HEAD +
dirty state), the mount-alias map, active worktrees, and the KG's
`serves`/`servedBy` route graph when present. **The rejected alternative is
explicit**: the provider "is honest about what it cannot see: a served
daemon's *loaded* revision is unknown from here, so it says so and points at
the restart that guarantees liveness, rather than guessing." Guessing a
daemon's loaded revision from checkout state alone would be wrong exactly
when it matters — right after a change that hasn't been picked up yet.

### Pointer — `CONCEPT:AU-KG.retrieval.kg-3` (the `entity` provider, reused across enterprise domains)

`entity_context.py:4-14`, registered for `tickets`/`deploys`/`process` in
`context_plane.py:49-58`. Answers "what's in the world-model / how many X /
show me recent X" over *any* node type in the KG. The comment at
`context_plane.py:50` states the rejected alternative: "Enterprise domains
are the entity provider with a label filter — registered here so the cockpit
grows with ingested data" — i.e. NOT a bespoke provider per enterprise domain
(a ticket dashboard, a deploy-inventory page, a process browser), each of
which would have to be built and ingested speculatively ahead of any
connector actually landing that data. One provider + a `DOMAIN_LABELS` map
means breadth grows exactly as fast as ingestion does, never ahead of it. The
`why` intent additionally delegates to `CONCEPT:AU-KG.retrieval.assimilated-from-mragent`'s
active reconstruction rather than a flat census, for relational questions.

### Pointer — `CONCEPT:AU-KG.retrieval.kg-4` (the `troubleshoot` provider)

`troubleshoot_context.py:4-20`. Answers "this agent run failed / this service
is unreachable / this container keeps crashing — what happened and how do I
trace it?" by pulling the run's `:RunTrace`/`:ToolCall` provenance and
emitting a **layered troubleshooting playbook** — the exact fleet tool to
reach for at each stack layer (app-trace → container log → system log → host
reachability → cross-cutting observability). **The rejected alternative is
explicit and named "anti-sprawl"**: "It builds no new log store." It composes
existing KG reads with a precise map onto the EXISTING fleet tools
(`cm__*`/`sm__*`/`tm__*`/`lgtm__*`/`graph_observe`) instead of standing up a
parallel log-aggregation system, so the operator/agent runs the right next
call instead of guessing — the same anti-guessing stance as the deploy
provider above, applied to a different symptom class.

### Pointer — `CONCEPT:AU-KG.retrieval.ops-context` (the `ops` provider)

`ops_context.py:4-15`. Answers "is the system healthy / why is the maint lane
backing up / what's blocked?" by synthesizing the KG's own operational data —
`WorkItem`s, their status/lane/kind, the dead-letter and failed backlog —
into one grounded answer with task/lane citations and a remediation hint.
Like every other provider here, it is pure best-effort Cypher reads that
"never raises," so a degraded backend yields a partial picture instead of a
crash — the same fail-soft posture the whole plane commits to, rather than
one provider being reliable and another being brittle.

## Risk Assessment

- **Blast Radius**: `context_plane.py`, `deploy_context.py`,
  `entity_context.py`, `troubleshoot_context.py`, `ops_context.py`,
  `capability_context.py` (a fifth registered provider, Seam 8 Phase 1 — see
  `.specify/design/kgr-capability-power-descriptor/design.md`).
- **Backward Compatible**: Yes — adding a domain is registering a provider,
  never modifying the dispatch contract.
- **Known weak point**: every provider commits to "never raises" / best-effort
  reads, which is the right default for availability but means a provider bug
  that silently returns an empty/wrong answer is much harder to distinguish
  from "nothing ingested yet" than a bug that crashes loudly would be.
