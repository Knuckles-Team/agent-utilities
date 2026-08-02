# Design Document: Provider ontologies (and other package-contributed assets) are read from ONE materialized, XDG-first unified tree, not by walking each provider's `site-packages` at read time

CONCEPT:AU-OS.deployment.unified-install-tree

> `agent_utilities/knowledge_graph/core/ontology_federation.py:146-155`
> (`resolve_provider_ontologies`); reused by `scripts/check_ontology.py:250-256`
> (`_provider_ttls`, the gate that sweeps package-contributed ontology TTLs).

## Decision — `resolve_provider_ontologies` prefers a valid, materialized subtree per currently registered provider under the unified installed tree (XDG-first), falling back to live entry-point discovery only when no materialized subtree is ready — and this is THE ONE read-path every ontology-federation glob-point (including the `check_ontology` gate) uses, rather than each caller walking installed packages' `site-packages` directories itself

Federated ontology fragments come from dozens of installed fleet packages. Two ways
exist to find them at read time: walk every installed package's own `site-packages`
directory looking for ontology files, or read from ONE materialized, XDG-managed
tree that `agent-utilities install` already builds. This module chooses the
latter: "the runtime reads contributed ontologies from one place instead of walking
each provider's `site-packages`" (`ontology_federation.py:151-152`). Materialization
happens once (at install/materialize time); every subsequent read — the runtime's
own federation resolution AND the `check_ontology` gate's provider-TTL sweep, which
explicitly reuses this same resolver "so the gate sweeps package-contributed
ontologies from the same place the runtime does" (`check_ontology.py:252-253`) — is
a read against that one materialized tree, with unmarked or retired nested
directories deliberately excluded (not treated as provider contributions).

## Rejected alternative — walk every installed package's `site-packages` directory at read time to discover its ontology contributions

Directly resolving contributions by scanning installed packages is the more
"obvious" approach and needs no separate materialization step — but it means every
caller that needs provider ontologies (the runtime's federation resolver, the
`check_ontology` gate, and any future glob-point) repeats the SAME `site-packages`
walk independently, with every one of them needing to agree on identical discovery
rules (which nested directories count, which are excluded) to avoid drifting from
each other — exactly the kind of duplicated-logic drift risk the surface-parity and
concept-lineage designs elsewhere in this codebase are built to prevent. It also
couples ontology discovery directly to Python's package-installation layout, which
is slower to walk repeatedly and harder to reason about for a non-Python packaging
context. Materializing into one XDG-first tree ONCE and reading from it everywhere
means every caller — runtime and gate alike — sees the identical set of
contributions, computed by one resolver, with the live entry-point-discovery walk
kept only as an explicit fallback for the case where the materialized tree is not
yet populated.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/core/ontology_federation.py`,
  `scripts/check_ontology.py`, every consumer of `resolve_provider_ontologies`.
- **Backward Compatible**: Yes — falls back to live discovery when the
  materialized tree is not ready, so a fresh/pre-`install` environment still
  resolves ontologies, just via the slower path.
- **Known weak point**: the materialized tree can go stale relative to actually
  installed packages — a package upgraded or removed without re-running
  `agent-utilities install` leaves its old materialized subtree in place (or a new
  one absent) until the next materialization, a drift window neither this resolver
  nor the `check_ontology` gate that reuses it can detect on its own.
