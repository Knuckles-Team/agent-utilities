"""Static backend-interface parity surrogate (CONCEPT:AU-KG.query.vendor-agnostic-traversal).

The engine-free *contract* portion of the cross-backend parity matrix. The live
``backend-parity-nightly`` matrix asserted backend BEHAVIOR, but it needed a running
epistemic-graph engine + Docker testcontainers, so it could never run in pre-commit
or the PR suite — only nightly (and was rarely green). This is its fast, import-only
stand-in: it asserts every concrete ``GraphBackend`` implements the SAME abstract
interface, so a backend that drops/renames a contract method is caught on every
commit instead of only when the nightly happens to stand that backend up. No engine,
no containers, no driver — pure class introspection (a concrete backend's
``__abstractmethods__`` is computed at class creation, never instantiated here).

The live conformance matrix (``tests/integration/backends``, ``-m live``) remains the
behavioral net; this guards the *shape*.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

import pytest

from agent_utilities.knowledge_graph.backends.base import GraphBackend

# Every concrete backend module. Imports are engine-free — the third-party drivers
# (psycopg/neo4j/falkordb/kuzu) are lazily imported / try-except-guarded, so the
# class definitions load with nothing deployed.
_BACKEND_MODULES = [
    "agent_utilities.knowledge_graph.backends.epistemic_graph_backend",
    "agent_utilities.knowledge_graph.backends.postgresql_backend",
    "agent_utilities.knowledge_graph.backends.age_backend",
    "agent_utilities.knowledge_graph.backends.fanout_backend",
    "agent_utilities.knowledge_graph.backends.contrib.neo4j_backend",
    "agent_utilities.knowledge_graph.backends.contrib.falkordb_backend",
    "agent_utilities.knowledge_graph.backends.contrib.ladybug_backend",
]

# The abstract contract every backend must satisfy (mirrors the @abstractmethod set
# on GraphBackend). A NEW abstractmethod added to the interface fails
# ``test_abstract_contract_method_set_is_stable`` until accounted for here — a
# conscious ratchet on the cross-backend contract, like the engine protocol-parity
# baseline.
_CONTRACT = frozenset(
    {
        "execute",
        "execute_batch",
        "create_schema",
        "add_embedding",
        "semantic_search",
        "prune",
        "close",
    }
)


def _concrete_backends() -> dict[str, type[GraphBackend]]:
    """Every concrete backend that INHERITS ``GraphBackend`` and is defined here.

    Filters on ``GraphBackend in __mro__`` (true inheritance), NOT ``issubclass`` —
    the latter also matches virtual subclasses registered via ``GraphBackend.register``
    (e.g. the transparent ``BrainGuardedBackend`` proxy, which satisfies the interface
    by ``__getattr__`` delegation at runtime, not by static methods). The static
    method-set contract only applies to real subclasses.
    """
    found: dict[str, type[GraphBackend]] = {}
    for mod_name in _BACKEND_MODULES:
        mod = importlib.import_module(mod_name)
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (
                GraphBackend in obj.__mro__
                and obj is not GraphBackend
                and obj.__module__ == mod_name
            ):
                found[obj.__name__] = obj
    return found


def test_every_concrete_backend_is_discoverable() -> None:
    # Guard against a silently-empty sweep declaring a falsely-clean gate.
    backends = _concrete_backends()
    assert len(backends) >= 6, f"only found {sorted(backends)} concrete backends"


def test_abstract_contract_method_set_is_stable() -> None:
    actual = set(GraphBackend.__abstractmethods__)
    assert actual == set(_CONTRACT), (
        "GraphBackend abstract contract changed — update this surrogate AND the live "
        f"parity matrix: expected {sorted(_CONTRACT)}, got {sorted(actual)}"
    )


@pytest.mark.parametrize(
    "name,cls",
    sorted(_concrete_backends().items()),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_backend_implements_full_interface(name: str, cls: type[Any]) -> None:
    # A concrete backend has NO unimplemented abstractmethods — Python would refuse to
    # instantiate it otherwise. A parity break (a dropped/renamed contract method)
    # leaves the inherited @abstractmethod here, so it can be caught without standing
    # the backend up.
    missing = set(getattr(cls, "__abstractmethods__", frozenset()))
    assert not missing, (
        f"{name} does not implement GraphBackend contract method(s): {sorted(missing)}"
    )
    # And every contract method resolves to a real callable on the class.
    for method in _CONTRACT:
        assert callable(getattr(cls, method, None)), f"{name} missing {method}()"
