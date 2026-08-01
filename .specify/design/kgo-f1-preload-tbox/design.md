# Design Document: The bundled TBox preloads into the local OWL store at boot, best-effort, never blocking startup on a missing OWL dependency

CONCEPT:AU-KG.ontology.preload-tbox

> `agent_utilities/server/app.py:469-484`.

## Decision — load `ontology.ttl` into the local OWL backend at server startup so reasoning + the local SPARQL endpoint have schema immediately, degrading to a no-op when owlready2/rdflib aren't installed

`app.py:469-472` states the decision and its guard: "preload the bundled
ontology TBox into the local OWL store at startup so OWL reasoning + the
local SPARQL endpoint have the schema immediately (best-effort; a no-op when
owlready2/rdflib aren't installed, e.g. the most minimal tiny profile)." The
implementation (`app.py:473-484`) wraps the whole block in a guarded
`try`, resolves the backend via the SAME `create_owl_backend()` factory
`owl-closure-native` and `owl-rdf-bridge` use, and only calls `load_ontology`
when both the backend exists AND the bundled `ontology.ttl` file is present on
disk.

**The rejected alternative is lazy, first-request loading** — defer loading
the TBox until the first OWL-dependent request arrives, avoiding any startup
cost on a profile that never uses OWL reasoning. Preloading instead means the
FIRST real request never pays a cold-load latency penalty and the local
SPARQL endpoint is queryable against the schema from the moment the server
reports ready — the cost is a small, always-paid startup delay on every
profile that has the OWL dependencies installed, whether or not that
particular deployment ever issues an OWL-reasoning request. The best-effort
guard is what makes this acceptable for the "most minimal tiny profile"
mentioned in the comment: a profile without `owlready2`/`rdflib` installed
pays zero cost (the whole block degrades to nothing) rather than failing
server startup over an optional dependency.

## Risk Assessment

- **Blast Radius**: `server/app.py` startup sequence, the local OWL
  backend/SPARQL endpoint's initial state.
- **Backward Compatible**: Yes — best-effort; a profile without OWL deps
  starts exactly as before.
- **Known weak point**: the preload silently no-ops on ANY exception in the
  guarded block (missing deps, a malformed bundled `ontology.ttl`, a backend
  construction failure) with no distinct signal for "intentionally minimal
  profile" versus "something is actually broken" — an operator would only
  notice via the absence of TBox-dependent reasoning results later, not a
  startup-time error.
