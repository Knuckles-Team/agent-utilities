"""Shared recording engine double for the fail-closed ActionPolicy tests."""

from __future__ import annotations


class DurableEngine:
    """Minimal durable authority: the policy fails closed without receipt storage."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}

    def add_node(self, node_id, node_type, properties=None, **_kw):
        self.nodes[node_id] = {"type": node_type, **(properties or {})}

    def query_cypher(self, *_args, **_kwargs):
        return []


def allow_host_validation(monkeypatch, tmp_path, *executables: str) -> str:
    """Opt a live-path goal-loop test into the (default-off) host validation runner.

    The runner only runs bare, operator-allowlisted executable names resolved
    from ``PATH`` (no shell). Returns the bare name of the current interpreter,
    allowlisted and reachable on ``PATH``, so a test can build a side-effecting
    validation command as argv.
    """
    import os
    import sys
    from pathlib import Path

    from agent_utilities.core.config import config

    interpreter = Path(sys.executable)
    names = {interpreter.name, *executables}
    monkeypatch.setattr(config, "kg_loop_allow_host_validation", True)
    monkeypatch.setattr(config, "kg_loop_host_validation_executables", ",".join(names))
    monkeypatch.setenv("PATH", f"{interpreter.parent}{os.pathsep}{os.environ['PATH']}")
    return interpreter.name
