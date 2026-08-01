# Design Document: Migrating a domain ontology out of core keeps the canonical import edge alive via a federated-IRI ledger, not a hard break

CONCEPT:AU-KG.ontology.package-federation-migration ·
CONCEPT:AU-KG.ontology.federation-provider-leg

> `agent_utilities/knowledge_graph/core/ontology_federation.py`.

### Pointer — `CONCEPT:AU-KG.ontology.federation-provider-leg`

`ontology_federation.py:1-35`. The general mechanism this migration rides on:
"the third leg of the fleet-federation mechanism (skills + prompts already
exist in `agent_utilities.core.providers`)." Any installed agent-package
contributes its own OWL/RDF module(s) by declaring a data-only
`agent_utilities.ontology_providers` entry point — no provider code is
imported to discover it, only the owning distribution's auditable file
manifest is resolved, and contributed `.ttl`s are then "treated **identically
to the bundled ontology modules**." **The rejected alternative is importing
provider code to discover its ontology assets** — the obvious approach, and
one that would make ontology discovery depend on a package's code being
importable/side-effect-free. Resolving from the distribution's file manifest
instead keeps discovery a pure filesystem/metadata operation: "adding the Nth
ontology provider adds zero bytes to the hub."

## Decision — moving a domain ontology into its owning `agents/*` package is tracked as ONE ledger entry, not a removal of the canonical `owl:imports` edge

`ontology_federation.py:56-60` states the migration mechanism directly: "the
~14-package migration fan-out: each domain ontology below now lives in its
owning `agents/*` package ... federated back in by IRI. `ontology_company.ttl`
(which stays in core) imports the banking + legal IRIs, so both must be
listed here for its import to resolve in a provider-less base install." The
canonical `ontology.ttl` keeps its `owl:imports` edge to a moved module's IRI
even after the module physically leaves the core wheel; `REGISTERED_FEDERATED_IRIS`
is the ledger the `check_ontology` gate consults so that edge is NOT flagged
as a dangling import in a base (provider-less) install (`ontology_federation.py:
50-55`). This is the concrete migration instance of the general per-package
federation mechanism (`CONCEPT:AU-KG.ontology.federation-provider-leg`).

**The rejected alternative is severing the import edge when a module moves**
— removing `owl:imports <.../banking>` from `ontology_company.ttl` the moment
the banking ontology moves into its own package would silently drop
`ontology_company.ttl`'s dependency on banking classes for any install that
doesn't also have the banking-owning package installed, and would require
re-adding the edge (and re-verifying nothing else broke) the moment the
package IS installed. Keeping the edge and tracking the target as "known
federated" instead means: a provider-less base install resolves the edge as a
harmless superset no-op (the classes just aren't reasoned over), while an
install WITH the provider gets the edge resolving for real — the SAME
`ontology.ttl` file works correctly in both configurations with no
conditional editing.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/core/ontology_federation.py`
  (`REGISTERED_FEDERATED_IRIS`), `scripts/check_ontology.py`'s CONNECTED
  check, every core `.ttl` file that imports a migrated domain's IRI.
- **Backward Compatible**: Yes — the ledger is purely additive tracking; a
  provider-less install behaves as it always did (superset no-op).
- **Known weak point**: the ledger is a manually curated tuple — "the
  ~14-package migration fan-out appends one line here per package it moves
  out" (`ontology_federation.py:55`) — nothing forces a NEW migration to add
  its IRI to this list; forgetting to would make `check_ontology` flag a
  legitimate provider-less-install superset import as a dangling reference.
