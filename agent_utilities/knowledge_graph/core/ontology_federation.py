#!/usr/bin/python
"""Per-package ontology federation — CONCEPT:AU-KG.ontology.federation-provider-leg.

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

Resolution uses the owning distribution's auditable file manifest without importing
provider code. It rejects ambiguous registrations and linked/special/unbounded source
trees, selects a content-addressed XDG generation only when it exactly matches the
current source, and flattens the resulting directory to concrete ``*.ttl`` files.

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

from pathlib import Path

from agent_utilities.core.provider_materialization import resolve_managed_generation
from agent_utilities.core.providers import (
    ONTOLOGY_PROVIDER_GROUP,
    current_provider_assets,
)

# Federated-IRI registry: the ledger of ontology IRIs that live in fleet packages
# rather than the agent-utilities wheel. The canonical ``ontology.ttl`` may keep an
# ``owl:imports`` edge to one of these even when the owning package is not currently
# installed; the ``check_ontology`` gate consults this set so such an import is NOT
# flagged dangling in a base (provider-less) install. The ~20-package migration
# fan-out appends one line here per package it moves out.
# CONCEPT:AU-KG.ontology.package-federation-migration — the ~14-package migration fan-out: each domain ontology below
# now lives in its owning agents/* package (see docs/architecture/ontology_library.md),
# federated back in by IRI. ``ontology_company.ttl`` (which stays in core) imports the
# banking + legal IRIs, so both must be listed here for its import to resolve in a
# provider-less base install.
REGISTERED_FEDERATED_IRIS: tuple[str, ...] = (
    "http://knuckles.team/kg/ansible",
    "http://knuckles.team/kg/archimate",
    "http://knuckles.team/kg/archivebox",
    "http://knuckles.team/kg/aris",
    "http://knuckles.team/kg/arr",
    "http://knuckles.team/kg/atlassian",
    "http://knuckles.team/kg/audio",
    "http://knuckles.team/kg/audiobookshelf",
    "http://knuckles.team/kg/banking",
    "http://knuckles.team/kg/caddy",
    "http://knuckles.team/kg/camunda",
    "http://knuckles.team/kg/ciso",
    "http://knuckles.team/kg/clarity",
    "http://knuckles.team/kg/container",
    "http://knuckles.team/kg/database",
    "http://knuckles.team/kg/datascience",
    "http://knuckles.team/kg/dockerhub",
    "http://knuckles.team/kg/documentdb",
    "http://knuckles.team/kg/egeria",
    "http://knuckles.team/kg/emerald",
    "http://knuckles.team/kg/erpnext",
    "http://knuckles.team/kg/fan",
    "http://knuckles.team/kg/feed",
    "http://knuckles.team/kg/firefly",
    "http://knuckles.team/kg/genius",
    "http://knuckles.team/kg/github",
    "http://knuckles.team/kg/gitlab",
    "http://knuckles.team/kg/grafana",
    "http://knuckles.team/kg/gramps",
    "http://knuckles.team/kg/homeassistant",
    "http://knuckles.team/kg/jena",
    "http://knuckles.team/kg/kafka",
    "http://knuckles.team/kg/keycloak",
    "http://knuckles.team/kg/langfuse",
    "http://knuckles.team/kg/leanix",
    "http://knuckles.team/kg/legal",
    "http://knuckles.team/kg/listmonk",
    "http://knuckles.team/kg/mattermost",
    "http://knuckles.team/kg/mealie",
    "http://knuckles.team/kg/media",
    "http://knuckles.team/kg/media-downloader",
    "http://knuckles.team/kg/microsoft",
    "http://knuckles.team/kg/nextcloud",
    "http://knuckles.team/kg/objectstore",
    "http://knuckles.team/kg/observability",
    "http://knuckles.team/kg/okta",
    "http://knuckles.team/kg/onetrust",
    "http://knuckles.team/kg/openbao",
    "http://knuckles.team/kg/owncast",
    "http://knuckles.team/kg/paperless",
    "http://knuckles.team/kg/plane",
    "http://knuckles.team/kg/portainer",
    "http://knuckles.team/kg/pulselink",
    "http://knuckles.team/kg/qbittorrent",
    "http://knuckles.team/kg/quant",
    "http://knuckles.team/kg/repository",
    "http://knuckles.team/kg/rom",
    "http://knuckles.team/kg/salesforce",
    "http://knuckles.team/kg/scholarx",
    "http://knuckles.team/kg/searxng",
    "http://knuckles.team/kg/servicenow",
    "http://knuckles.team/kg/social",
    "http://knuckles.team/kg/stirlingpdf",
    "http://knuckles.team/kg/systems",
    "http://knuckles.team/kg/technitium",
    "http://knuckles.team/kg/trading",
    "http://knuckles.team/kg/tunnel",
    "http://knuckles.team/kg/twenty",
    "http://knuckles.team/kg/uptimekuma",
    "http://knuckles.team/kg/vector",
    "http://knuckles.team/kg/wellness",
)


def registered_federated_iris() -> set[str]:
    """Return the set of known package-owned (federated) ontology IRIs.

    These are IRIs the canonical bundle may import even when the owning package is
    not installed — a superset no-op, not a dangling reference (CONCEPT:AU-KG.ontology.federation-provider-leg).
    """
    return set(REGISTERED_FEDERATED_IRIS)


def resolve_provider_ontologies() -> list[tuple[str, Path]]:
    """XDG-first provider-ontology resolution (CONCEPT:AU-OS.deployment.unified-install-tree).

    Prefer valid managed subtrees for currently registered providers under the
    materialized unified tree; fall back to live entry-point discovery when none are
    ready. Unmarked or retired nested directories are not provider contributions.
    This is the read-path every ontology federation glob-point uses, so the runtime
    reads contributed ontologies from one place instead of walking each provider's
    ``site-packages``.

    The ``agent-utilities`` provider dir is **excluded**: it mirrors the bundled core
    TBox, which every consumer already loads directly via its own ``ontology*.ttl``
    glob — including it here would double-load and trip the duplicate-IRI gate.
    """
    materialization_root = unified_ontologies_dir()
    out: list[tuple[str, Path]] = []
    for assets in current_provider_assets(ONTOLOGY_PROVIDER_GROUP):
        provider = assets.registration.name
        if provider == "agent-utilities":
            continue
        asset_dir = resolve_managed_generation(
            materialization_root / provider,
            provider=provider,
            leg="ontologies",
            registration=assets.registration.digest,
            source_manifest=assets.manifest,
        ) or assets.source_root
        ttls = sorted(asset_dir.glob("*.ttl"))
        shapes = asset_dir / "shapes"
        if shapes.is_dir():
            ttls.extend(sorted(shapes.glob("*.ttl")))
        for ttl in ttls:
            out.append((provider, ttl))
    return sorted(out, key=lambda item: (item[0].casefold(), item[1].as_posix()))


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
