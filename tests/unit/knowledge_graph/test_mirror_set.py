"""Mirror-set construction (CONCEPT:KG-2.74).

A single-writer, file-locked mirror (LadybugDB/Kuzu) must be owned by exactly one
process — the host write daemon. The many client MCP processes share the same
``config.json``; if each built the ladybug mirror they would all try to open the
same DB file and contend on its exclusive OS lock. So ``_build_mirror_set`` builds
a file-locked mirror only when ``effective_daemon_role() == "host"`` and silently
skips it for client roles. Network mirrors (neo4j/falkordb) are built for every
role — many openers are fine.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph import backends as B
from agent_utilities.knowledge_graph.core import host_lock


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
