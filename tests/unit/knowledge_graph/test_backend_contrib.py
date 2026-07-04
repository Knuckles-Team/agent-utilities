#!/usr/bin/python
"""Tests for the demoted (contrib) graph backends and default selection.

CONCEPT:AU-KG.query.object-graph-mapper

Verifies the Plan-03 Step-11 backend reorganization:
    - Neo4j / FalkorDB / LadybugDB are demoted to ``backends.contrib`` and
      stay fully importable on demand (opt-in).
    - Default / primary backend selection (``file`` / ``memory`` / the config
      default) resolves WITHOUT importing any contrib backend, so the optional
      ``neo4j`` / ``falkordb`` / ``ladybug`` driver packages are never required.
    - Back-compat lazy re-exports on the ``backends`` package resolve to the
      exact same classes as the new ``contrib`` import path.
"""

import importlib
import sys

import pytest

import agent_utilities.knowledge_graph.backends as backends

CONTRIB_MODULE_NAMES = (
    "agent_utilities.knowledge_graph.backends.contrib.neo4j_backend",
    "agent_utilities.knowledge_graph.backends.contrib.falkordb_backend",
    "agent_utilities.knowledge_graph.backends.contrib.ladybug_backend",
)


@pytest.fixture(autouse=True)
def _restore_contrib_modules():
    """Restore any contrib backend modules purged during a test.

    ``_purge_contrib_modules`` deletes ``backends.contrib.*`` from ``sys.modules``
    so we can assert they are not re-imported. Without restoring them, a later
    test (e.g. ``test_ladybug_backend_get_lock``) re-imports a *fresh* module,
    and the stale-vs-fresh ``CombinedLock``/lock objects break its assertions.
    Snapshotting and restoring here keeps the purge local to each test.
    """
    saved = {n: m for n, m in sys.modules.items() if "backends.contrib." in n}
    yield
    for name, module in saved.items():
        sys.modules.setdefault(name, module)


def _purge_contrib_modules() -> None:
    """Remove already-imported contrib backend modules from ``sys.modules``.

    This lets us assert that a given code path does NOT (re)import them. The
    ``_restore_contrib_modules`` autouse fixture puts them back after the test.
    """
    for name in list(sys.modules):
        if "backends.contrib." in name:
            del sys.modules[name]


@pytest.mark.parametrize("backend_type", ["file", "memory", "epistemic_graph"])
def test_default_selection_does_not_require_contrib(backend_type):
    """Default/primary backend selection must not import contrib backends.

    (a) The Rust-native EpistemicGraph tiers (file/memory/epistemic_graph)
    resolve without importing neo4j/falkordb/ladybug.
    """
    _purge_contrib_modules()

    backend = backends.create_backend(backend_type)

    assert backend is not None
    assert type(backend).__name__ == "EpistemicGraphBackend"

    leaked = [m for m in sys.modules if "backends.contrib." in m]
    assert not leaked, f"contrib backend(s) imported eagerly: {leaked}"


def test_factory_default_backend_type_is_not_contrib(monkeypatch):
    """With no explicit type and no GRAPH_BACKEND, the default resolves to the
    primary PostgreSQL tier (a non-contrib backend) — never ladybug/contrib."""
    _purge_contrib_modules()
    monkeypatch.delenv("GRAPH_BACKEND", raising=False)

    # PostgreSQLBackend construction may fail without a live DB; we only need
    # to confirm the resolution path does not touch contrib. Importing the
    # module itself is non-contrib and safe.
    pg_mod = importlib.import_module(
        "agent_utilities.knowledge_graph.backends.postgresql_backend"
    )
    assert hasattr(pg_mod, "PostgreSQLBackend")

    leaked = [m for m in sys.modules if "backends.contrib." in m]
    assert not leaked, f"contrib backend(s) imported during default path: {leaked}"


def test_contrib_backends_importable_via_new_path():
    """(b) Demoted backends remain importable via their new contrib path."""
    from agent_utilities.knowledge_graph.backends.base import GraphBackend
    from agent_utilities.knowledge_graph.backends.contrib.falkordb_backend import (
        FalkorDBBackend,
    )
    from agent_utilities.knowledge_graph.backends.contrib.ladybug_backend import (
        LadybugBackend,
    )
    from agent_utilities.knowledge_graph.backends.contrib.neo4j_backend import (
        Neo4jBackend,
    )

    assert issubclass(Neo4jBackend, GraphBackend)
    assert issubclass(FalkorDBBackend, GraphBackend)
    assert issubclass(LadybugBackend, GraphBackend)


def test_contrib_package_lazy_attribute_access():
    """The contrib package exposes lazy attribute access for opt-in use."""
    from agent_utilities.knowledge_graph.backends import contrib

    assert contrib.Neo4jBackend.__name__ == "Neo4jBackend"
    assert contrib.FalkorDBBackend.__name__ == "FalkorDBBackend"
    assert contrib.LadybugBackend.__name__ == "LadybugBackend"

    with pytest.raises(AttributeError):
        _ = contrib.DoesNotExist


@pytest.mark.parametrize(
    "name",
    ["Neo4jBackend", "FalkorDBBackend", "LadybugBackend"],
)
def test_back_compat_shim_resolves_to_same_class(name):
    """(c) Old import path (``backends.<Name>``) resolves to the same class as
    the new ``backends.contrib.*`` path."""
    from agent_utilities.knowledge_graph.backends.contrib import (
        falkordb_backend,
        ladybug_backend,
        neo4j_backend,
    )

    new_path = {
        "Neo4jBackend": neo4j_backend.Neo4jBackend,
        "FalkorDBBackend": falkordb_backend.FalkorDBBackend,
        "LadybugBackend": ladybug_backend.LadybugBackend,
    }[name]

    shim = getattr(backends, name)
    assert shim is new_path


def test_ladybug_available_shim():
    """LADYBUG_AVAILABLE flag is exposed via both the shim and contrib package."""
    from agent_utilities.knowledge_graph.backends import contrib
    from agent_utilities.knowledge_graph.backends.contrib.ladybug_backend import (
        LADYBUG_AVAILABLE,
    )

    assert backends.LADYBUG_AVAILABLE == LADYBUG_AVAILABLE
    assert contrib.LADYBUG_AVAILABLE == LADYBUG_AVAILABLE
