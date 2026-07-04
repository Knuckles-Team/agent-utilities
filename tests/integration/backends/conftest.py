"""The parametrized backend matrix for the conformance suite.

CONCEPT:AU-KG.query.object-graph-mapper / KG-2.7 — one ``backend_under_test`` parameter per supported
backend, built through the production ``create_backend`` factory against the
ephemeral fixtures defined in ``tests/integration/conftest.py``.

The two zero-infra params (epistemic_graph + ladybug) run in the default PR
suite; the container-backed params carry ``pytest.mark.live`` so the full matrix
runs only under ``pytest -m live``.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any

import pytest

from agent_utilities.knowledge_graph.backends import create_backend

_BACKEND_PARAMS = [
    pytest.param("epistemic_graph", id="epistemic_graph"),
    pytest.param("ladybug", id="ladybug"),
    pytest.param("postgresql", id="pggraph", marks=pytest.mark.live),
    pytest.param("neo4j", id="neo4j", marks=pytest.mark.live),
    pytest.param("falkordb", id="falkordb", marks=pytest.mark.live),
]


@pytest.fixture(params=_BACKEND_PARAMS)
def backend_under_test(request: pytest.FixtureRequest) -> Iterator[Any]:
    """Yield a live ``GraphBackend`` for each supported backend.

    Resolves per-backend connection details from the ephemeral fixtures (only the
    one this param needs is instantiated, via ``getfixturevalue``), then builds
    the backend through ``create_backend`` so the test exercises the exact
    construction path real deployments use.
    """
    backend_type: str = request.param

    if backend_type == "epistemic_graph":
        if not os.environ.get("GRAPH_SERVICE_SOCKET"):
            pytest.skip(
                "epistemic-graph engine not running (GRAPH_SERVICE_SOCKET unset)"
            )
        kwargs: dict[str, Any] = {}
    elif backend_type == "ladybug":
        kwargs = {"db_path": request.getfixturevalue("ephemeral_ladybug")["db_path"]}
    elif backend_type == "postgresql":
        info = request.getfixturevalue("ephemeral_pg_age")
        kwargs = {"uri": info["uri"], "db_name": info["db_name"]}
    elif backend_type == "neo4j":
        info = request.getfixturevalue("ephemeral_neo4j")
        kwargs = {
            "uri": info["uri"],
            "user": info["user"],
            "password": info["password"],
        }
    elif backend_type == "falkordb":
        info = request.getfixturevalue("ephemeral_falkordb")
        kwargs = {
            "host": info["host"],
            "port": info["port"],
            "db_name": info["db_name"],
        }
    else:  # pragma: no cover — guarded by the param list above
        pytest.skip(f"unknown backend param {backend_type!r}")

    backend = create_backend(backend_type=backend_type, **kwargs)
    if backend is None:
        pytest.skip(
            f"{backend_type} backend unavailable "
            f"(optional driver not installed — see the matching pyproject extra)"
        )

    # Stash the construction recipe so durability tests can reopen the same store
    # (epistemic_graph is in-process/in-memory and has nothing to reopen).
    backend._parity_backend_type = backend_type  # type: ignore[attr-defined]
    backend._parity_kwargs = kwargs  # type: ignore[attr-defined]
    backend._parity_durable = backend_type != "epistemic_graph"  # type: ignore[attr-defined]

    try:
        yield backend
    finally:
        with contextlib.suppress(Exception):
            backend.close()
