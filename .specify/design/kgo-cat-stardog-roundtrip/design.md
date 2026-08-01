# Design Document: Stardog is a round-trip catalog target — publish clears-then-adds, import consumes it back through the normal load path

CONCEPT:AU-KG.ontology.stardog-catalog-import ·
CONCEPT:AU-KG.ontology.stardog-catalog-overwrite

> `agent_utilities/knowledge_graph/core/ontology_publisher.py`
> (`OntologyPublisher.push_to_stardog`, `import_ontology_from_stardog`),
> exposed via `agent_utilities/mcp/kg_server.py` and
> `agent_utilities/mcp/tools/ontology_tools.py`.

## Decision — the platform both PUBLISHES to and CONSUMES from Stardog, not a one-way export

`CONCEPT:AU-KG.ontology.stardog-catalog-import`

`import_ontology_from_stardog` (`ontology_publisher.py:366-386`) is
"the reverse of `push_to_stardog`": it pulls the TBox already living in a
Stardog database/named graph back out as turtle and, when an `engine` is
given, runs it through the identical parse → validate → register → activate
lifecycle path every other hosted ontology uses (`OntologyLifecycle.load`),
so the engine reasons over the catalog that already lives there. Without an
engine it just returns the pulled turtle for offline inspection.

**The rejected alternative is a one-directional publish-only integration** —
the simpler design, and the one most triplestore integrations stop at (push
data out, never read it back). Making the path bidirectional means Stardog can
be treated as a genuine catalog of record: an ontology curated/edited directly
in Stardog's own tooling can be pulled back into the KG's reasoning engine
through the SAME validated lifecycle path a locally-loaded `.ttl` would use —
no separate, weaker "trust it because it came from Stardog" import path.

### Pointer — `CONCEPT:AU-KG.ontology.stardog-catalog-overwrite`

`ontology_publisher.py:174-188, 233-248` and the REST twin at
`kg_server.py:1302-1321` (`overwrite=bool(body.get("overwrite", True))` —
**defaults to True**). `push_to_stardog`'s `overwrite` flag decides between two
real alternatives, both implemented and named directly in the docstring: leave
it **False** and re-publishing an updated ontology *accumulates* duplicate/stale
triples alongside the old ones in the target named graph; or **True**, which
`CLEAR`s the target graph (scoped to `named_graph` when given, else the Stardog
`DEFAULT` graph) inside the same transaction before adding, so re-publishing
*replaces* the prior catalog slice atomically (`conn.begin()` /
`conn.rollback()` on failure). The REST route's default of `overwrite=True`
is a deliberate choice that re-publishing should behave like an update by
default, with accumulation only reachable by an explicit opt-out.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/core/ontology_publisher.py`,
  `mcp/kg_server.py:1287-1345`, any Stardog database configured as a catalog
  target.
- **Backward Compatible**: Yes — Stardog integration is opt-in and
  credential-gated (`STARDOG_ENDPOINT`/`STARDOG_DATABASE`/…); absent
  credentials, both actions return a clean `status: error`, not a crash.
- **Known weak point**: `overwrite=True` with no `named_graph` clears
  Stardog's `DEFAULT` graph outright — if two ontologies were ever published to
  the same Stardog database without distinct named graphs, an overwrite publish
  of one silently destroys the other's triples.
