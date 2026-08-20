# Ontology integrity-policy activation (CONCEPT:AU-KG.ontology.integrity-bootstrap)

> Ledger: NE-152. How agent-utilities registers the engine's mandatory SHACL/ICV
> integrity policy on a dedicated ontology graph before loading any axioms into
> it, why the engine rejects ontology loads without this, and how an operator
> diagnoses "ontology not activated" in a running deployment.

## The invariant

The epistemic-graph engine's RDF write guard
(`EG-KG.ontology.rdf-update-guard`) rejects **every** `AddTriples` /
`RemoveTriples` against a named graph — even a request adding zero triples —
until that graph has a registered SHACL/ICV integrity policy (`IcvConfigure`).
There is no "no policy configured" pass-through; a graph with no policy is
**closed to writes**, permanently, by design. This is correct engine behavior:
it is agent-utilities' job to register a policy before it ever asks the engine
to load ontology content.

## What used to happen (the defect)

Every caller in this codebase that loaded ontology axioms into the engine's
dedicated per-tenant ontology graph called `add_triples` directly, with no
prior `IcvConfigure`. The engine correctly rejected every one of those calls.
The callers caught the rejection, recorded `active: false` on the hosted-
ontology record, logged a warning, and returned success to *their* caller —
so the process kept serving requests while ontology activation silently and
permanently failed, on every boot, forever. Nothing surfaced this in
readiness or health.

## What happens now

`agent_utilities/knowledge_graph/ontology/activation.py` is the ONE chokepoint
function, `ensure_ontology_graph_activated(gc, tenant=..., graph_name=...,
ontology_turtle=...)`, that every ontology-triple-loading call site now routes
through **before** calling `add_triples`:

- `ontology/lifecycle.py` — `OntologyLifecycle._load_axioms` (backs
  `load()`/`update()`, the hosted-ontology CRUD surface reached by
  `graph_ontology` MCP/REST and boot-time package-ontology sync).
- `ontology/lifecycle.py` — `OntologyLifecycle.activate_graph()`, called
  **unconditionally** by `mcp/kg_server.py`'s `_sync_ontologies_at_boot` before
  any package ontologies are synced, so the graph is activation-ready even on
  a boot with zero federated ontology content to load.
- `ontology/evolution.py` — `materialize_shadow` (the ontology-evolution
  proposal pipeline's throwaway shadow graph — ephemeral does not mean exempt
  from the guard).

`ensure_ontology_graph_activated`:

1. **Registers** the graph's SHACL/ICV policy via the engine's own
   `GraphComputeEngine.icv_configure` (→ `client.rdf.icv_configure`,
   `IcvConfigure` on the wire) — the only supported registration API. It never
   bypasses, relaxes, or reimplements the guard.
2. **Reads back and verifies** the registration took effect with a
   *zero-triple* `add_triples("")` probe. The guard evaluates graph
   authority/policy presence before it ever inspects the triples supplied, so
   an empty probe exercises the exact same check a real load would hit — using
   only the engine's own existing authority, never a second validator.
3. Only then does the caller proceed to load real ontology content.

### Binding

The first successful activation of a graph durably records
`(tenant, graph, policy_digest)` as an `:OntologyActivation` marker node in
that same graph (the same convention as the `:HostedOntology` registry).
Every later activation call for that graph must present the identical
tenant, graph, and policy digest, or it is rejected with
`OntologyPolicyBindingMismatchError` — never silently reconfigured. The
ontology **content** digest is recorded for provenance only; loading a new
ontology *version* over an already-activated graph (`update()`) is expected
and is not treated as a binding mismatch.

### Idempotence

Re-running activation against an already-activated graph (a restart, a second
boot, a repeated MCP call) is a no-op success: no second `IcvConfigure` call,
no error. (The engine's own `IcvConfigure` handler is additionally
policy-idempotent server-side for byte-identical calls — this module's check
is an independent, testable guarantee on top of that.)

### Bounded retry / backoff

Registration + verification retries transient failures (engine starting,
shard moving) with jittered exponential backoff, up to
`DEFAULT_MAX_ATTEMPTS` (6) attempts or `DEFAULT_TIME_CEILING_S` (30s),
whichever is hit first. Only the **first** failure and the **final give-up**
are logged — never one line per attempt.

## Failure modes and what they look like

| Symptom | Cause | Typed error | Log line |
|---|---|---|---|
| Activation never runs | No engine attached (offline/dev), or the engine handle lacks an RDF/ICV surface | `OntologyActivationError` | none (offline mode is not an error) |
| Repeated transient failures until the ceiling | Engine starting / shard placement moving during boot | `OntologyActivationTimeoutError` | one `WARNING` on first failure, one `ERROR` "gave up ... after N attempt(s)" |
| A graph's policy/tenant doesn't match what's already registered | A caller bug resolved the wrong tenant/graph, or a policy change was pushed outside this chokepoint | `OntologyPolicyBindingMismatchError` | one `ERROR` "binding mismatch" naming both the recorded and requested values |

## Readiness: fails closed

`agent_utilities/knowledge_graph/readiness.py`'s `collect_readiness_snapshot()`
includes an `ontology_activation` check (`_check_ontology_activation`):

- **`unavailable`, reason `no_engine_supplied`** — no engine handle passed to
  the readiness collector at all.
- **`unavailable`, reason `ontology_activation_not_attempted`** — activation
  has never run in this process for the tenant's ontology graph. This is the
  state the live defect used to hide behind a "ready" liveness probe.
- **`unavailable`, reason `<activation_timeout | policy_binding_mismatch |
  activation_error:...>`** — activation was attempted and gave up; the
  `detail` field carries the graph name and (for a mismatch) both the
  recorded and requested tenant/graph/policy digest.
- **`ready`** — activation succeeded (fresh or idempotent replay) in this
  process.

A degraded/unavailable `ontology_activation` check pulls the WHOLE snapshot's
`overall` away from `"ready"` (`readiness._rollup`), so
`is_snapshot_ready()` returns `False` — the service can no longer silently
serve as if ontology activation succeeded when it did not.

## Diagnosing "ontology not activated" live

1. **Check readiness.** Call `graph_analyze(action="readiness")` (or the REST
   twin) and read `checks.ontology_activation`. `state` + `reason` tell you
   which of the failure modes above you're in; `detail.graph` names the exact
   named graph.
2. **Check the boot log.** `_sync_ontologies_at_boot` logs `"Ontology graph
   activation failed at boot: <reason>"` at `ERROR` when `activate_graph()`
   doesn't succeed (and is silent — by design — only when there is genuinely
   no engine RDF surface, i.e. an offline/dev profile).
3. **Check `activation.py`'s own log lines.** A retry sequence logs exactly
   one `WARNING` ("... will retry with bounded backoff") on the first
   transient failure and one `ERROR` ("... gave up ... after N attempt(s)")
   on final exhaustion — grep for `Ontology activation` in the graph-os
   process log.
4. **A binding mismatch** logs an `ERROR` naming the recorded vs. requested
   tenant/graph/policy digest — this almost always means a caller resolved
   the wrong tenant (check `current_actor().tenant_id` / the `tenant=`
   argument passed to `OntologyLifecycle(...)`), not an engine problem.
5. **Do not** work around a failed activation by disabling the guard, adding
   a "skip integrity" path, or hand-loading triples through a lower-level
   client — there is no supported way to load ontology content into a graph
   that hasn't been activated, and there should not be one.

## Non-goals / what this does not change

- The engine's write guard itself is untouched — this is a client-side
  ordering fix, not an engine change.
- No new validator: policy content is real SHACL (see
  `activation.DEFAULT_ICV_SHAPES_TTL` — every hosted `owl:Class` /
  `owl:ObjectProperty` / `owl:DatatypeProperty` must be a named IRI, never a
  blank node), evaluated entirely by the engine's own SHACL implementation.
- No compatibility flag reintroduces the old "load without activating"
  behavior.
