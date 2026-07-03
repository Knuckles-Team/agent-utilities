#!/usr/bin/python
"""Unified XDG install of every provider contribution — CONCEPT:OS-5.77.

The three federation legs (skills + prompts + ontologies, discovered via the
``agent_utilities.*_providers`` entry-points in :mod:`agent_utilities.core.providers`)
are materialized into **one** XDG data tree so the runtime reads assets from a single
location instead of walking every installed package's ``site-packages`` on demand::

    $XDG_DATA_HOME/agent-utilities/skills/<provider>/<skill>/SKILL.md
    $XDG_DATA_HOME/agent-utilities/prompts/<provider>/*.json
    $XDG_DATA_HOME/agent-utilities/ontologies/<provider>/<pkg>.ttl (+ shapes/)

The hub's OWN contributions land in the same tree under provider name
``agent-utilities``. Its skills come through its ``skill_providers`` entry-point like
any package, but it deliberately declares **no** ``prompt_providers`` /
``ontology_providers`` entry-point (that would double-ingest the base prompts / core
TBox at the registry level), so its prompts and core/upper ontologies are sourced via
a **direct package-data path** here. For robustness against stale editable dist-info,
the hub's skills are ALSO sourced directly (deduped against the provider loop by name).

The materialization is idempotent and overwrite-on-reinstall (``force=True`` default):
each provider's destination subtree is replaced wholesale, so a reinstall always
reflects the currently-installed provider. Failure is isolated per provider — one
unresolvable/broken package never blocks the rest.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from agent_utilities.core.paths import (
    data_dir,
    ontology_dir,
    skills_dir,
    unified_prompts_dir,
)
from agent_utilities.core.providers import (
    ONTOLOGY_PROVIDER_GROUP,
    PROMPT_PROVIDER_GROUP,
    SKILL_PROVIDER_GROUP,
    iter_provider_dirs,
)

logger = logging.getLogger(__name__)

# The hub's own contributions are namespaced under this provider name in every leg.
OWN_PROVIDER = "agent-utilities"

_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")


def unified_skills_dir() -> Path:
    """The unified-tree skills root (``$XDG.../skills/``) — same dir the factory reads."""
    return skills_dir()


def unified_ontologies_dir() -> Path:
    """The unified-tree ontologies root (``$XDG.../ontologies/``)."""
    return ontology_dir()


def _copy_tree(src: Path, dst: Path, force: bool) -> int:
    """Copy a provider asset directory ``src`` → ``dst`` (whole subtree).

    Overwrite-on-reinstall when ``force``; otherwise a populated ``dst`` is left as-is.
    ``__pycache__``/compiled files are never copied. Returns the number of files
    materialized (0 when skipped).
    """
    if not src.is_dir():
        return 0
    if dst.exists():
        if not force:
            return 0
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=_IGNORE)
    return sum(1 for p in dst.rglob("*") if p.is_file())


def _copy_ontology(src: Path, dst: Path, force: bool) -> int:
    """Materialize the ``*.ttl`` (+ ``shapes/*.ttl``) an ontology provider ships.

    Unlike skills/prompts this flattens to just the ontology data (never the
    ``__init__.py`` / package machinery of a ``<pkg>.ontology`` module).
    Overwrite-on-reinstall when ``force``. Returns the count of TTL files copied.
    """
    ttls = sorted(src.glob("*.ttl"))
    shapes_src = src / "shapes"
    shapes = sorted(shapes_src.glob("*.ttl")) if shapes_src.is_dir() else []
    if not ttls and not shapes:
        return 0
    if dst.exists():
        if not force:
            return 0
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for t in ttls:
        shutil.copy2(t, dst / t.name)
        n += 1
    if shapes:
        (dst / "shapes").mkdir(parents=True, exist_ok=True)
        for s in shapes:
            shutil.copy2(s, dst / "shapes" / s.name)
            n += 1
    return n


def _install_own(force: bool, result: dict[str, dict[str, int]]) -> None:
    """Materialize the hub's OWN skills + prompts + core ontologies (direct path).

    Sourced from the installed ``agent_utilities`` package data — not via a
    self-entry-point — so they land in the unified tree under ``agent-utilities``
    regardless of entry-point/dist-info state.
    """
    import agent_utilities

    pkg = Path(agent_utilities.__file__).resolve().parent

    n = _copy_tree(pkg / "skills", unified_skills_dir() / OWN_PROVIDER, force)
    if n or (pkg / "skills").is_dir():
        result["skills"][OWN_PROVIDER] = n

    n = _copy_tree(pkg / "prompts", unified_prompts_dir() / OWN_PROVIDER, force)
    if n or (pkg / "prompts").is_dir():
        result["prompts"][OWN_PROVIDER] = n

    # Core + upper ontologies live directly in knowledge_graph/ (ontology.ttl,
    # ontology_*.ttl, shapes/*.ttl) — the same set the publisher/backend glob.
    n = _copy_ontology(
        pkg / "knowledge_graph", unified_ontologies_dir() / OWN_PROVIDER, force
    )
    result["ontologies"][OWN_PROVIDER] = n


def install_unified(force: bool = True) -> dict[str, Any]:
    """Materialize all three provider legs (+ the hub's own) into the XDG data tree.

    CONCEPT:OS-5.77. Idempotent; ``force`` (default) replaces each provider subtree so
    a reinstall reflects the installed set. Returns a report::

        {"data_dir": ..., "skills": {provider: n_files}, "prompts": {...},
         "ontologies": {provider: n_ttls}}

    where ``agent-utilities`` is the hub's own contribution in each leg.
    """
    result: dict[str, Any] = {
        "data_dir": str(data_dir()),
        "skills": {},
        "prompts": {},
        "ontologies": {},
    }

    # Fleet providers — the hub's own provider name is handled directly below, so it
    # is skipped here to avoid a redundant (and dist-info-dependent) second pass.
    for provider, src in iter_provider_dirs(SKILL_PROVIDER_GROUP):
        if provider == OWN_PROVIDER:
            continue
        try:
            result["skills"][provider] = _copy_tree(
                src, unified_skills_dir() / provider, force
            )
        except OSError as e:  # noqa: BLE001 — one bad provider never blocks the rest
            logger.warning("Skill provider %s not materialized: %s", provider, e)

    for provider, src in iter_provider_dirs(PROMPT_PROVIDER_GROUP):
        if provider == OWN_PROVIDER:
            continue
        try:
            result["prompts"][provider] = _copy_tree(
                src, unified_prompts_dir() / provider, force
            )
        except OSError as e:  # noqa: BLE001
            logger.warning("Prompt provider %s not materialized: %s", provider, e)

    for provider, src in iter_provider_dirs(ONTOLOGY_PROVIDER_GROUP):
        if provider == OWN_PROVIDER:
            continue
        try:
            result["ontologies"][provider] = _copy_ontology(
                src, unified_ontologies_dir() / provider, force
            )
        except OSError as e:  # noqa: BLE001
            logger.warning("Ontology provider %s not materialized: %s", provider, e)

    _install_own(force, result)
    return result
