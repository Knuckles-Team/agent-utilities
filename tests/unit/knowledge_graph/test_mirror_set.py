"""Mirror-set construction (CONCEPT:AU-KG.backend.mirror-health-repair).

A single-writer, file-locked mirror (LadybugDB/Kuzu) must be owned by exactly one
process — the host write daemon. The many client MCP processes share the same
``config.json``; if each built the ladybug mirror they would all try to open the
same DB file and contend on its exclusive OS lock. So ``_build_mirror_set`` builds
a file-locked mirror only when ``effective_daemon_role() == "host"`` and silently
skips it for client roles. Network mirrors (neo4j/falkordb) are built for every
role — many openers are fine.

The classes below also prove the availability fix: ``_build_mirror_set`` is the
exact function ``create_backend()`` calls (via ``backend_type="fanout"``) while
constructing the ONE operational authority at graph-os startup
(``mcp/kg_server.py``'s ``_get_engine`` → ``create_backend()`` →
``elif backend_type == "fanout": ... _build_mirror_set(...)``). Every
``kg_connections`` mirror (neo4j/falkordb/ladybug/...) is optional interop/BI/DR
tooling the epistemic-graph authority never depends on, so one mirror that
cannot be constructed (missing driver, unreachable host, bad credentials,
invalid ``connection_profile_ref``) must never propagate out of this function —
that would take the whole server down over an optional dependency, which is
exactly the bug: a bare ``ImportError: Neo4j driver is not installed`` used to
propagate, uncaught, all the way out of engine construction.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph import backends as B
from agent_utilities.knowledge_graph.core import host_lock


@pytest.fixture(autouse=True)
def _isolated_mirror_build_status():
    """``_build_mirror_set`` records into the process-wide
    ``B._MIRROR_BUILD_STATUS`` registry (CONCEPT:AU-KG.backend.mirror-health-repair)
    so ``get_mirror_build_status()`` reflects real activity across the whole
    process — but that means these tests, which call the REAL
    ``_build_mirror_set``, mutate real global state. Give every test a clean
    slate so one test's mirror names never leak into another's assertions."""
    B._MIRROR_BUILD_STATUS.clear()
    yield
    B._MIRROR_BUILD_STATUS.clear()


def _configure(monkeypatch):
    from agent_utilities.core.config import config as cfg

    monkeypatch.delenv("GRAPH_MIRROR_TARGETS", raising=False)
    monkeypatch.setattr(
        cfg, "graph_mirror_targets", ["team-falkor", "local-ladybug"], raising=False
    )
    monkeypatch.setattr(
        cfg,
        "kg_connections",
        [
            {"name": "team-falkor", "backend": "falkordb"},
            {"name": "local-ladybug", "backend": "ladybug", "db_path": "/tmp/x.db"},
        ],
        raising=False,
    )
    # Don't actually instantiate real backends — record the build attempts.
    monkeypatch.setattr(
        B, "_build_member", lambda spec: ("BK", spec.get("backend_type"))
    )


def test_file_locked_mirror_skipped_for_client_role(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(host_lock, "effective_daemon_role", lambda: "client")
    mirrors = B._build_mirror_set()
    assert "team-falkor" in mirrors  # network mirror always built
    assert "local-ladybug" not in mirrors  # file-locked → host-only


def test_file_locked_mirror_built_for_host_role(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(host_lock, "effective_daemon_role", lambda: "host")
    mirrors = B._build_mirror_set()
    assert "team-falkor" in mirrors
    assert "local-ladybug" in mirrors  # the host daemon owns it


def test_network_mirror_never_consults_role(monkeypatch):
    """A pure network-mirror set must not even resolve the daemon role."""
    from agent_utilities.core.config import config as cfg

    monkeypatch.delenv("GRAPH_MIRROR_TARGETS", raising=False)
    monkeypatch.setattr(cfg, "graph_mirror_targets", ["team-falkor"], raising=False)
    monkeypatch.setattr(
        cfg,
        "kg_connections",
        [{"name": "team-falkor", "backend": "falkordb"}],
        raising=False,
    )
    monkeypatch.setattr(
        B, "_build_member", lambda spec: ("BK", spec.get("backend_type"))
    )

    def _boom():
        raise AssertionError("role must not be resolved for network-only mirrors")

    monkeypatch.setattr(host_lock, "effective_daemon_role", _boom)
    assert "team-falkor" in B._build_mirror_set()


# --------------------------------------------------------------------------- #
# Availability fix: one mirror failing to construct must never take the
# operational authority (or any other mirror) down with it.
# --------------------------------------------------------------------------- #
def _configure_two_mirrors_one_broken(monkeypatch, *, broken_exc: Exception):
    """``team-falkor`` builds fine; ``prod-neo4j`` fails to construct with
    ``broken_exc`` (e.g. the real-world ``ImportError: Neo4j driver is not
    installed``)."""
    from agent_utilities.core.config import config as cfg

    monkeypatch.delenv("GRAPH_MIRROR_TARGETS", raising=False)
    monkeypatch.setattr(
        cfg, "graph_mirror_targets", ["team-falkor", "prod-neo4j"], raising=False
    )
    monkeypatch.setattr(
        cfg,
        "kg_connections",
        [
            {"name": "team-falkor", "backend": "falkordb"},
            {"name": "prod-neo4j", "backend": "neo4j"},
        ],
        raising=False,
    )

    def _fake_build_member(spec):
        if spec.get("backend_type") == "neo4j":
            raise broken_exc
        return ("BK", spec.get("backend_type"))

    monkeypatch.setattr(B, "_build_member", _fake_build_member)


def test_a_broken_mirror_does_not_raise_out_of_mirror_set_construction(monkeypatch):
    """THE bug fix: constructing ``prod-neo4j`` raises ``ImportError`` (the
    exact real-world failure — missing ``neo4j`` driver package). This must
    never propagate out of ``_build_mirror_set`` — the function
    ``create_backend()`` calls while building the ONE operational authority at
    graph-os startup. Before the fix, this exact exception took the whole
    server down.
    """
    _configure_two_mirrors_one_broken(
        monkeypatch,
        broken_exc=ImportError(
            "Neo4j driver is not installed. Please install with "
            "`pip install agent-utilities[neo4j]`"
        ),
    )

    mirrors = B._build_mirror_set()  # must not raise

    assert "prod-neo4j" not in mirrors


def test_other_mirrors_still_register_when_one_mirror_is_broken(monkeypatch):
    """The healthy mirror must still be built even though its sibling failed."""
    _configure_two_mirrors_one_broken(
        monkeypatch, broken_exc=ImportError("Neo4j driver is not installed")
    )

    mirrors = B._build_mirror_set()

    assert "team-falkor" in mirrors
    assert "prod-neo4j" not in mirrors


def test_broken_mirror_construction_failure_is_reported_with_real_cause(
    monkeypatch,
):
    """The failure must be OBSERVABLE (not silent) and must carry the real
    cause — the exception message, not just its type — per
    ``get_mirror_build_status()`` (read by the ``kg_mirrors`` runtime-health
    check).
    """
    _configure_two_mirrors_one_broken(
        monkeypatch,
        broken_exc=ImportError(
            "Neo4j driver is not installed. Please install with "
            "`pip install agent-utilities[neo4j]`"
        ),
    )

    B._build_mirror_set()

    status = B.get_mirror_build_status()
    assert status["prod-neo4j"]["ok"] is False
    assert "Neo4j driver is not installed" in status["prod-neo4j"]["reason"]
    assert status["prod-neo4j"]["backend_type"] == "neo4j"
    assert status["team-falkor"]["ok"] is True


def test_broken_mirror_logs_the_real_cause_not_just_the_exception_type(monkeypatch):
    """The anti-pattern this codebase has been bitten by repeatedly: logging
    only ``type(exc).__name__`` hides the actual, actionable reason. The
    ``logger.error(...)`` call for a broken mirror must be given the real
    exception object (not just its type) and ``exc_info=True`` so a traceback
    is attached.

    NOTE: this asserts what ``_build_mirror_set`` hands to the logger, not
    what a handler ultimately renders — ``agent_utilities``'s own process-wide
    privacy boundary (``core/log_privacy.py``'s ``install_log_privacy_boundary``,
    installed for every ``agent_utilities.*`` logger) reduces exception objects
    to their type name and drops tracebacks from every record in this package,
    by design, before any handler sees them ("logs retain exception types, not
    tracebacks/messages that may embed endpoints or paths"). That pre-existing,
    package-wide policy is orthogonal to this fix and applies identically to
    every other error log in the codebase (e.g. ``kg_server.py``'s
    ``_bundled_skill_contract``) — the real, unredacted cause for a broken
    mirror is what ``get_mirror_build_status()`` carries instead (plain dict
    data, never routed through logging), proven by the test above.
    """
    _configure_two_mirrors_one_broken(
        monkeypatch,
        broken_exc=ValueError("Neo4j requires a complete connection profile"),
    )
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(B.logger, "error", lambda *a, **k: calls.append((a, k)))

    B._build_mirror_set()

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert kwargs.get("exc_info") is True
    # The real exception OBJECT (not merely its type name) was handed to the
    # logger — %s-formatting it renders the full message, not just "ValueError".
    passed_exc = args[-1]
    assert isinstance(passed_exc, ValueError)
    assert str(passed_exc) == "Neo4j requires a complete connection profile"
    rendered = args[0] % args[1:]
    assert "Neo4j requires a complete connection profile" in rendered
    assert "ValueError" in rendered
    assert "prod-neo4j" in rendered


def test_backend_factory_returning_none_is_also_isolated_and_reported(monkeypatch):
    """A mirror factory returning ``None`` (e.g. a driver import guard that
    degrades gracefully instead of raising) is a distinct failure mode from a
    raised exception — it must be isolated and reported the same way: skipped,
    never fatal, and visible in ``get_mirror_build_status()``.
    """
    from agent_utilities.core.config import config as cfg

    monkeypatch.delenv("GRAPH_MIRROR_TARGETS", raising=False)
    monkeypatch.setattr(
        cfg, "graph_mirror_targets", ["team-falkor", "prod-neo4j"], raising=False
    )
    monkeypatch.setattr(
        cfg,
        "kg_connections",
        [
            {"name": "team-falkor", "backend": "falkordb"},
            {"name": "prod-neo4j", "backend": "neo4j"},
        ],
        raising=False,
    )

    def _fake_build_member(spec):
        if spec.get("backend_type") == "neo4j":
            return None
        return ("BK", spec.get("backend_type"))

    monkeypatch.setattr(B, "_build_member", _fake_build_member)

    mirrors = B._build_mirror_set()  # must not raise

    assert "team-falkor" in mirrors
    assert "prod-neo4j" not in mirrors
    status = B.get_mirror_build_status()
    assert status["prod-neo4j"]["ok"] is False
    assert status["team-falkor"]["ok"] is True
