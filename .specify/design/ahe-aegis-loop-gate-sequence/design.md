# Design Document: Wire HarnessX's AEGIS decomposition onto existing machinery, not a new gate stack

CONCEPT:AU-AHE.harness.run-aegis-loop-over ·
CONCEPT:AU-AHE.harness.manifest-verify ·
CONCEPT:AU-AHE.harness.per-dimension-ship-outcome ·
CONCEPT:AU-AHE.harness.variant-pool

> `agent_utilities/harness/aegis_loop.py` (all four), plus
> `agent_utilities/harness/manifest.py:94` (manifest-verify),
> `agent_utilities/mcp/tools/analysis_tools.py:2138` (run-aegis-loop-over's MCP
> surface) and `agent_utilities/harness/harness_foundry_benchmark.py:98`
> (variant-pool's reproduction benchmark).

## Decision — AEGIS's Digester→Planner→Evolver→Critic shape is realised over machinery that already existed, not built fresh

`CONCEPT:AU-AHE.harness.run-aegis-loop-over`

`aegis_loop.py`'s module docstring states the decision directly (`aegis_loop.py:1-9`):
HarnessX's AEGIS paper decomposes harness evolution into four stages, but rather
than implement a new per-edit gate matching the paper's own gate, `AegisLoop`
wires the shape onto pieces that already existed — the Critic is the existing
formal SHACL gate (`HarnessGate`) reasoning over the harness ontology, and the
Planner (`adaptation_landscape()`) reads the **accumulated edit distribution
already stored in the ontology**, so the loop sees cross-round concentration
the paper's own per-edit gate cannot see in isolation.

**The rejected alternative is reproducing HarnessX's gate wholesale** — a fresh,
paper-shaped verifier bolted on beside the existing ontology-driven gate. That
would duplicate state (two views of "has this pattern already shipped?") and
lose the cross-round signal that lets this implementation self-correct
*before* HarnessX's own reported failure mode: the sub-threshold-coupling
tipping point that collapsed its τ³-Bench run (`aegis_loop.py:6-9`). The stages
are dependency-injected (`evolver_fn`, `verifier_fn`, `smoke_fn`,
`normalize_fn`) so the loop can run fully offline with no LLM/engine, the same
injection pattern already used by `FastSlowController`/`SubstrateTrainer` —
another point where an existing pattern was reused instead of inventing a new
one for this loop specifically.

### Pointer — `CONCEPT:AU-AHE.harness.manifest-verify`

`aegis_loop.py:227-257`, `manifest.py:94-95`. The Critic is not one check but a
fixed four-stage **deterministic** gate sequence: manifest-verify (regression
check via the injected `verifier_fn`) → config-normalization (canonical-form
dedup, so a re-proposed identical edit can't masquerade as fresh progress) →
build/smoke test (does the edited processor/tool actually instantiate?) →
SHACL gate. **The rejected alternative is a single combined probabilistic
check** (e.g. one LLM-judged "does this look safe" pass) — the deterministic
sequence exists precisely so a duplicate or a non-instantiating edit is caught
*before* it ever reaches the SHACL concentration/regression reasoning, which is
comparatively expensive and semantic. `ManifestPrediction.smoke_passed`
(`manifest.py:94`) is set by this exact stage and is what the SHACL gate later
reasons over.

### Pointer — `CONCEPT:AU-AHE.harness.per-dimension-ship-outcome`

`aegis_loop.py:102,109-119,213-215,302-304`. A per-dimension ship-outcome
ledger (`_ledger: dict[str, list[bool]]`) tracks the last N ship attempts per
dimension and exposes a recent hit-rate. **This is a quantitative
under-exploration defense, deliberately separate from the SHACL concentration
gate**: concentration counts *how many* edits land in a dimension; the ledger
tracks whether they are *succeeding*. The rejected alternative — relying on
concentration alone — cannot distinguish "this dimension is over-mined" from
"this dimension is declining in yield," which are different problems needing
different responses. Selective invocation (the Evolver may return nothing,
short-circuiting the round as idle rather than forcing a low-quality edit) and
a patience-based early stop are the same discipline applied to when the loop
gives up entirely.

### Pointer — `CONCEPT:AU-AHE.harness.variant-pool`

`aegis_loop.py:99-101,148,259-266`, `harness_foundry_benchmark.py:95-108`. When
an edit fixes one task cluster but regresses another, the loop **forks a new
variant scoped to the improved cluster** instead of accepting or rejecting the
edit wholesale. `_bench_variant_isolation` names the rejected alternative
explicitly and reproduces it as a baseline: "Single-harness baseline: the edit
applies to the ONE harness covering all tasks, so its regression on taskB is
in-scope → the seesaw rejects" — i.e. HarnessX's own single-harness model must
reject any edit with a mixed effect, which is exactly the seesaw stagnation
the paper documents on heterogeneous benchmarks. With variant isolation the
no-regression SHACL shape is evaluated *per variant, over that variant's own
cluster*, so the out-of-scope regression routes to a different variant instead
of blocking the fix.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/aegis_loop.py`,
  `agent_utilities/harness/manifest.py`, `agent_utilities/harness/harness_gate.py`,
  `agent_utilities/mcp/tools/analysis_tools.py`,
  `agent_utilities/harness/harness_foundry_benchmark.py`.
- **Backward Compatible**: Yes — the loop is opt-in machinery invoked explicitly
  (`harness_evolve` MCP action or direct instantiation); it does not change any
  existing gate's default behavior.
- **Known weak point**: `variant_capacity` bounds the resident base-plus-fork
  pool (default 8); a pathological edit stream that keeps producing mixed
  fixes/regressions across many clusters can exhaust that capacity, at which
  point new forks are refused and the loop degrades back toward the
  single-harness seesaw behavior it exists to avoid.
