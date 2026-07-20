"""Meta-tests for the public graph-boundary architecture gate."""

from __future__ import annotations

from pathlib import Path

from scripts.check_public_graph_boundary import check


def _write(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_gate_rejects_direct_and_aliased_backend_execution(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "agent_utilities/gateway/bad.py",
        "def route(engine):\n    engine.backend.execute('MATCH (n) RETURN n')\n",
    )
    _write(
        tmp_path,
        "agent_utilities/server/routers/also_bad.py",
        "def route(engine):\n    backend = getattr(engine, 'backend')\n"
        "    backend.execute_batch([])\n",
    )
    assert len(check(tmp_path)) == 2


def test_gate_accepts_guarded_facade_query(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "agent_utilities/gateway/good.py",
        "def route(engine, session):\n"
        "    return engine.query_cypher('MATCH (n) RETURN n', session=session)\n",
    )
    assert check(tmp_path) == []


def test_gate_covers_mcp_query_surfaces(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "agent_utilities/mcp/tools/query_tools.py",
        "def tool(engine):\n"
        "    backend = getattr(engine, 'backend')\n"
        "    return backend.execute('MATCH (n) RETURN n')\n",
    )
    findings = check(tmp_path)
    assert len(findings) == 1
    assert "backend execute primitive" in findings[0]
