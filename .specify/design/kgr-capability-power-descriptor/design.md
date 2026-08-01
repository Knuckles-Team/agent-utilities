# Design Document: A Capability Power Descriptor is generated from canonical sources every time — it is never hand-edited, so it cannot silently rot

CONCEPT:AU-KG.retrieval.capability-power-descriptor

> `agent_utilities/knowledge_graph/retrieval/capability_power_descriptor.py`
> (the descriptor + generator contract), `scripts/gen_capability_power.py`
> (generator), `scripts/check_cpd.py` (drift gate),
> `tests/gates/test_cpd_gate.py`.

## Decision — every CPD field is read from an existing authority or computed from one by a pure function; a drift gate enforces it, nothing edits the output by hand

`capability_power_descriptor.py:4-50` (Seam 8 Phase 1) states the core rule
in bold in the module docstring itself: **"Generated from canonical sources,
not edited per CPD."** Every field group names its exact source of truth —
tool surface (`id`/`title`/`one_line`/typed I/O) from the immutable
feature-aware `ToolSpec` universe plus an isolated canonical FastMCP build;
intent verbs from the deliberate, ordered `tool_specs.TOOL_VERBS` routing
authority, copied exactly (with a dependency-free `infer_intent_verbs`
fallback only for a candidate the authority doesn't yet cover); the action
inventory from the already-generated `GRAPHOS_ACTIONS` manifest; the REST
route from `kg_server.ACTION_TOOL_ROUTES` (the same map
`check_surface_parity.py` already gates); side-effects/durability/authz from
the EG-P0-1 generated capability ledger, matched to an AU action by a
best-effort, NAME-DERIVED fuzzy match that "records its own confidence and
the exact tokens that matched" — an unmatched action is left unmatched with
an honest note, "NEVER guessed"; and cost/latency/reliability numbers
TRANSCRIBED (not estimated) from EG's own measured benchmark docs.

**The rejected alternative is named directly and is the whole point of the
sentence it appears in**: a hand-authored or per-tool-edited CPD. Any
authority-derived field left to manual editing would drift the moment its
source authority changed underneath it — a renamed tool, an added REST route,
a re-benchmarked latency number — with nothing to catch the staleness. The
regeneration discipline (`scripts/gen_capability_power.py` as generator,
`scripts/check_cpd.py` as the CI drift gate backed by
`tests/gates/test_cpd_gate.py`) is what makes "cannot silently rot" an
enforced property rather than an aspiration: a CPD that no longer matches a
fresh regeneration from its sources fails the gate, rather than quietly
shipping stale.

A second, narrower rejected alternative for the EG↔AU crosswalk specifically:
guessing a match between AU's action-routed surface and EG's raw `Method`
enum when no confident name-derived match exists. The two are "maintained in
two different repos with no existing 1:1 crosswalk," and the module's answer
is to leave a low-confidence action explicitly unmatched with a note, rather
than force a plausible-looking but unverified pairing that a reader might
trust as ground truth.

## Risk Assessment

- **Blast Radius**: `capability_power_descriptor.py`, `capability_context.py`,
  `context_plane.py` (the `capability` domain provider — see
  `.specify/design/kgr-context-plane-domain-providers/design.md`),
  `scripts/check_cpd.py`, `scripts/gen_capability_power.py`,
  `docs/capabilities-power.md`/`.json`.
- **Backward Compatible**: Yes — the descriptor is a read-time projection
  over existing authorities; it does not change any of them.
- **Known weak point**: the EG↔AU action crosswalk match confidence is
  name-derived and best-effort — a real match whose AU action name and EG
  `Method` name happen to diverge lexically will be left unmatched (a false
  negative that under-reports capability, not a false positive that
  over-promises one, which is the safer failure direction but still a real
  coverage gap).
