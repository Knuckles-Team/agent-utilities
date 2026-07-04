#!/usr/bin/python
"""Per-package ontology federation — CONCEPT:KG-2.320.

The third leg of the fleet-federation mechanism (skills + prompts already exist in
:mod:`agent_utilities.core.providers`). Any installed agent-package may contribute
its own OWL/RDF ontology module(s) to the central hub by declaring a data-only
entry-point::

    # in the contributing package's pyproject.toml
    [project.entry-points."agent_utilities.ontology_providers"]
    servicenow-api = "servicenow_api.ontology"
    [tool.setuptools.package-data]
    servicenow_api = ["ontology/**"]

The contributed ``.ttl`` files (plus optional ``shapes/*.ttl``) then live inside the
contributor's own wheel and are treated **identically to the bundled ontology
modules** — parsed into the published TBox, pre-loaded into the live OWL reasoner so
``owl:imports`` resolve, and swept by the ``check_ontology`` valid/connected/SHACL
gate (KG-2.112). Adding the Nth ontology provider adds zero bytes to the hub.

Discovery reuses :func:`agent_utilities.core.providers.iter_provider_dirs` verbatim
(same ``importlib.metadata`` + ``importlib.resources`` resolution, same
failure-isolation and de-duplication) and flattens each resolved provider directory
to its concrete ``*.ttl`` files.

A moved-but-imported ontology (e.g. the canonical ``ontology.ttl`` keeps its
``owl:imports <http://knuckles.team/kg/servicenow>`` edge after the servicenow module
moves into the ``servicenow-api`` wheel) is kept non-dangling two ways:

* when the provider **is installed**, its IRI is declared by a discovered TTL, so it
  resolves like any bundled module; and
* when the provider is **not installed**, the IRI is a *known federated* reference
  (see :func:`registered_federated_iris`) that the gate tolerates as a superset
  no-op — federation never breaks the base install.
"""

from __future__ import annotations

import logging
from pathlib import Path

from agent_utilities.core.providers import (
    ONTOLOGY_PROVIDER_GROUP,
    iter_provider_dirs,
)

logger = logging.getLogger(__name__)

# Federated-IRI registry: the ledger of ontology IRIs that live in fleet packages
# rather than the agent-utilities wheel. The canonical ``ontology.ttl`` may keep an
# ``owl:imports`` edge to one of these even when the owning package is not currently
# installed; the ``check_ontology`` gate consults this set so such an import is NOT
# flagged dangling in a base (provider-less) install. The ~20-package migration
# fan-out appends one line here per package it moves out.
# CONCEPT:KG-2.325 — the ~14-package migration fan-out: each domain ontology below
# now lives in its owning agents/* package (see docs/architecture/ontology_library.md),
# federated back in by IRI. ``ontology_company.ttl`` (which stays in core) imports the
# banking + legal IRIs, so both must be listed here for its import to resolve in a
# provider-less base install.
REGISTERED_FEDERATED_IRIS: tuple[str, ...] = (
    "http://knuckles.team/kg/servicenow",
    "http://knuckles.team/kg/leanix",
    "http://knuckles.team/kg/erpnext",
    "http://knuckles.team/kg/archimate",
    "http://knuckles.team/kg/egeria",
    "http://knuckles.team/kg/quant",
    "http://knuckles.team/kg/trading",
    "http://knuckles.team/kg/banking",
    "http://knuckles.team/kg/legal",
    "http://knuckles.team/kg/media",
    "http://knuckles.team/kg/grafana",
    "http://knuckles.team/kg/observability",
    "http://knuckles.team/kg/social",
    "http://knuckles.team/kg/feed",
    "http://knuckles.team/kg/wellness",
    "http://knuckles.team/kg/database",
)


def registered_federated_iris() -> set[str]:
    """Return the set of known package-owned (federated) ontology IRIs.

    These are IRIs the canonical bundle may import even when the owning package is
    not installed — a superset no-op, not a dangling reference (CONCEPT:KG-2.320).
    """
    return set(REGISTERED_FEDERATED_IRIS)


def discover_provider_ontologies() -> list[tuple[str, Path]]:
    """Discover every contributed ontology ``.ttl`` across installed providers.

    Resolves each ``agent_utilities.ontology_providers`` entry-point to its data
    directory via :func:`iter_provider_dirs` (failure-isolated, sorted, deduped),
    then flattens to each concrete ``*.ttl`` inside that directory plus any
    ``shapes/*.ttl`` it ships.

    Returns:
        A list of ``(provider_name, ttl_path)`` tuples, deterministically ordered
        by provider name then file name. Empty when no provider is installed
        (federation is a superset no-op in the base install).
    """
    out: list[tuple[str, Path]] = []
    for provider, asset_dir in iter_provider_dirs(ONTOLOGY_PROVIDER_GROUP):
        ttls: list[Path] = sorted(asset_dir.glob("*.ttl"))
        shapes_dir = asset_dir / "shapes"
        if shapes_dir.is_dir():
            ttls.extend(sorted(shapes_dir.glob("*.ttl")))
        if not ttls:
            logger.debug(
                "Ontology provider %s (%s) resolved but ships no .ttl; skipping",
                provider,
                asset_dir,
            )
            continue
        for ttl in ttls:
            out.append((provider, ttl))
    return out


def resolve_provider_ontologies() -> list[tuple[str, Path]]:
    """XDG-first provider-ontology resolution (CONCEPT:OS-5.78).

    Prefer the materialized unified tree (``$XDG.../ontologies/<provider>/*.ttl``,
    written by ``agent-utilities install``); fall back to live entry-point discovery
    (:func:`discover_provider_ontologies`) when that tree is unpopulated (dev/editable
    or pre-install). This is the read-path every ontology federation glob-point uses,
    so the runtime reads contributed ontologies from one place instead of walking each
    provider's ``site-packages``.

    The ``agent-utilities`` provider dir is **excluded**: it mirrors the bundled core
    TBox, which every consumer already loads directly via its own ``ontology*.ttl``
    glob — including it here would double-load and trip the duplicate-IRI gate.
    """
    root = unified_ontologies_dir()
    if not root.is_dir():
        return discover_provider_ontologies()
    providers = [
        d for d in sorted(root.iterdir()) if d.is_dir() and d.name != "agent-utilities"
    ]
    if not providers:
        # Tree exists but ships no fleet provider (only the hub's own mirror, or
        # loose user TTLs) — defer to live discovery so entry-point providers still load.
        return discover_provider_ontologies()
    out: list[tuple[str, Path]] = []
    for pdir in providers:
        ttls = sorted(pdir.glob("*.ttl"))
        shapes = pdir / "shapes"
        if shapes.is_dir():
            ttls.extend(sorted(shapes.glob("*.ttl")))
        for ttl in ttls:
            out.append((pdir.name, ttl))
    return out


def unified_ontologies_dir() -> Path:
    """The XDG unified-tree ontologies root (``$XDG.../ontologies/``).

    Thin re-export of :func:`agent_utilities.core.unified_install.unified_ontologies_dir`
    (imported lazily to avoid an import cycle) so this module — the ontology
    federation home — owns the read-path lookup its consumers call.
    """
    from agent_utilities.core.unified_install import (
        unified_ontologies_dir as _root,
    )

    return _root()


def discover_provider_ontology_dirs() -> list[tuple[str, Path]]:
    """Return each installed ontology provider's ``(provider_name, asset_dir)``.

    Thin pass-through to :func:`iter_provider_dirs` for callers (IRI→file
    resolution, the unified installer) that need the directory rather than the
    flattened TTL list (CONCEPT:KG-2.320).
    """
    return iter_provider_dirs(ONTOLOGY_PROVIDER_GROUP)
