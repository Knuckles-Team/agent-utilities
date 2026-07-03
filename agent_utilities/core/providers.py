#!/usr/bin/python
"""Entry-point provider discovery for fleet-contributed skills and prompts.

CONCEPT:OS-5.52 — Modular skill/prompt contribution.

Any installed agent-package can contribute its own skills and/or system-prompt
blueprints to the central hub by declaring a setuptools entry-point that points
at a *data-only* subpackage::

    # in the contributing package's pyproject.toml
    [project.entry-points."agent_utilities.skill_providers"]
    servicenow-api = "servicenow_api.skills"
    [project.entry-points."agent_utilities.prompt_providers"]
    servicenow-api = "servicenow_api.prompts"

The hub resolves each entry-point to the contributor's installed *data
directory* via ``importlib.resources`` — it imports only the named data
subpackage (which must carry no heavy dependencies) and never executes the
agent's business logic. This keeps ``agent-utilities`` lean: contributed assets
live inside each package's own wheel, and adding the Nth provider adds zero
bytes to the hub.

The discovery is failure-isolated: an uninstalled or unresolvable provider is
skipped, never fatal, so one broken package cannot break install/ingest for the
rest of the fleet.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from importlib.resources import as_file, files
from pathlib import Path

logger = logging.getLogger(__name__)

SKILL_PROVIDER_GROUP = "agent_utilities.skill_providers"
PROMPT_PROVIDER_GROUP = "agent_utilities.prompt_providers"
# CONCEPT:KG-2.320 — the third federation leg: any installed agent-package can
# contribute its own OWL/RDF ontology module(s) to the central hub by declaring an
# ``agent_utilities.ontology_providers`` entry-point pointing at a *data-only*
# ``<pkg>.ontology`` subpackage carrying ``*.ttl`` (+ optional ``shapes/*.ttl``).
# Resolved identically to skills/prompts via :func:`iter_provider_dirs`.
ONTOLOGY_PROVIDER_GROUP = "agent_utilities.ontology_providers"


def resolve_prompt_provider_dirs() -> list[tuple[str, Path]]:
    """XDG-first prompt-provider resolution (CONCEPT:OS-5.78).

    Prefer the materialized unified tree (``$XDG.../prompts/<provider>/``, written by
    ``agent-utilities install``); fall back to live entry-point discovery
    (``iter_provider_dirs(PROMPT_PROVIDER_GROUP)``) when it is unpopulated (dev/editable
    or pre-install). This lets the prompt ``registry_builder`` read contributed prompts
    from one place instead of walking each provider's ``site-packages``.

    The ``agent-utilities`` provider dir is **excluded**: the hub's base prompts are
    already loaded directly by ``registry_builder`` (bare ``prompt:<name>`` ids), and
    re-reading its mirror here would re-introduce the double-ingest the missing
    ``prompt_providers`` self-entry-point deliberately avoids.
    """
    try:
        from agent_utilities.core.unified_install import unified_prompts_dir

        root = unified_prompts_dir()
    except Exception:  # noqa: BLE001 — resolver must never break prompt ingestion
        root = None
    if root is not None and root.is_dir():
        providers = [
            d
            for d in sorted(root.iterdir())
            if d.is_dir() and d.name != "agent-utilities"
        ]
        if providers:
            return [(d.name, d) for d in providers]
    return iter_provider_dirs(PROMPT_PROVIDER_GROUP)


def iter_provider_dirs(group: str) -> list[tuple[str, Path]]:
    """Resolve every entry-point in ``group`` to ``(provider_name, asset_dir)``.

    Resolves each contributor's data directory via ``importlib.resources`` —
    never executes contributor business logic beyond importing the named data
    subpackage. Providers that cannot be resolved (not installed, bad value,
    missing directory) are skipped, not raised. Duplicate provider names are
    de-duplicated (first wins).

    Returns a list (deterministically ordered by entry-point name) so callers
    can apply a stable precedence.
    """
    out: list[tuple[str, Path]] = []
    seen: set[str] = set()
    try:
        eps = entry_points(group=group)
    except TypeError:  # pragma: no cover - very old importlib.metadata
        # Legacy importlib.metadata (<3.10): entry_points() returns a dict-like
        # SelectableGroups, not the keyword-filtered EntryPoints — the .get + the
        # default list are correct for that old shape only.
        eps = entry_points().get(group, [])  # type: ignore[attr-defined, arg-type]
    for ep in sorted(eps, key=lambda e: e.name):
        if ep.name in seen:
            continue
        try:
            with as_file(files(ep.value)) as resolved:
                path = Path(resolved)
            if path.is_dir():
                out.append((ep.name, path))
                seen.add(ep.name)
            else:
                logger.debug(
                    "Provider %s (%s) resolved to non-directory %s; skipping",
                    ep.name,
                    ep.value,
                    path,
                )
        except (ModuleNotFoundError, TypeError, FileNotFoundError, ValueError) as e:
            logger.debug("Could not resolve provider %s (%s): %s", ep.name, ep.value, e)
            continue
    return out
