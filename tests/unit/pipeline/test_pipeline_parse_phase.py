"""EH-273: the ``parse`` phase batches code-file parsing in ONE engine
round-trip instead of one ``ParseFile`` RPC per file.

Mocks ``RustASTParser``/``EpistemicGraphClient`` since no live engine socket
is available in this test environment, so these prove the CALL PATTERN (one
batch RPC vs N per-file RPCs) and the fallback semantics, not live engine
timings.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_utilities.knowledge_graph.pipeline.phases import parse as parse_phase_mod
from agent_utilities.knowledge_graph.pipeline.types import PipelineContext
from agent_utilities.models.knowledge_graph import PipelineConfig


class _FakeGraph:
    """Minimal graph double: records add_node/add_edge calls.

    ``PipelineContext.graph`` is typed ``GraphComputeEngine`` and pydantic
    validates that even with ``arbitrary_types_allowed`` — but a REAL
    ``GraphComputeEngine`` needs an isolated test engine
    (``tests/conftest.py``'s ``_session_engine``), which this dev host
    cannot start (no live epistemic-graph process; same skip every other
    ``pipeline/phases/*`` test hits here — see ``test_pipeline_scan_parse.py``).
    EH-273's own logic (batch vs. per-file call pattern, fallback,
    mismatched-response handling) needs no real engine at all, so these
    tests build the context via ``model_construct`` (pydantic's
    validation-skipping constructor) with this lightweight double instead.
    """

    def __init__(self) -> None:
        self.nodes: list[tuple[Any, dict]] = []
        self.edges: list[tuple[Any, Any, dict]] = []

    def add_node(self, node_id, **props):
        self.nodes.append((node_id, props))

    def add_edge(self, source, target, **props):
        self.edges.append((source, target, props))


def _ctx(tmp_path: Path, files: list[str]) -> tuple[PipelineContext, list[str]]:
    cfg = PipelineConfig(workspace_path=str(tmp_path))
    ctx = PipelineContext.model_construct(
        config=cfg, graph=_FakeGraph(), backend=None, results={}, metadata={}
    )
    return ctx, files


def _parse_result(rel_path: str) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "node_id": f"symbol:{rel_path}::fn",
                "node_type": "SYMBOL",
                "properties": {"symbol_type": "Function", "name": "fn", "line": "1"},
            }
        ],
        "edges": [],
        "symbols_extracted": 1,
    }


@pytest.mark.asyncio
async def test_code_files_use_one_batched_round_trip(tmp_path, monkeypatch):
    """N code files -> ONE `client.graph.parse_files` call, not N `parse_file` calls."""
    py_files = []
    for i in range(5):
        p = tmp_path / f"m{i}.py"
        p.write_text(f"def fn{i}(): pass\n")
        py_files.append(str(p))

    ctx, files = _ctx(tmp_path, py_files)

    parse_file_calls: list[str] = []

    class FakeParser:
        socket_path = "/fake.sock"
        auth_secret = "s"
        verified_context = {}

        async def parse_file(self, rel_path, source):
            parse_file_calls.append(rel_path)
            return _parse_result(rel_path)

    batch_calls: list[list[tuple[str, bytes]]] = []

    class FakeGraphOps:
        async def parse_files(self, files):
            batch_calls.append(list(files))
            return [_parse_result(fp) for fp, _src in files]

    class FakeClient:
        def __init__(self):
            self.graph = FakeGraphOps()

        @classmethod
        async def connect(cls, **kwargs):
            return cls()

        async def close(self):
            pass

    monkeypatch.setattr(
        parse_phase_mod, "_ingest_markdown", parse_phase_mod._ingest_markdown
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "epistemic_graph.parser",
        SimpleNamespace(RustASTParser=lambda **kw: FakeParser()),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "epistemic_graph.client",
        SimpleNamespace(EpistemicGraphClient=FakeClient),
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.session.GraphSession.from_ambient",
        classmethod(lambda cls: SimpleNamespace(engine_verified_context=lambda: {})),
    )

    result = await parse_phase_mod.execute_parse(
        ctx, {"scan": SimpleNamespace(output=files)}
    )

    assert len(batch_calls) == 1  # ONE round-trip for all 5 files
    assert len(batch_calls[0]) == 5
    assert parse_file_calls == []  # per-file path never used on the happy path
    assert result["symbols_extracted"] == 5


@pytest.mark.asyncio
async def test_falls_back_to_per_file_when_batch_connection_unavailable(
    tmp_path, monkeypatch
):
    """Batch RPC unavailable -> falls back to the original per-file loop,
    which itself still degrades per file (preserves pre-EH-273 behavior)."""
    py_files = [str(tmp_path / "a.py"), str(tmp_path / "b.py")]
    for f in py_files:
        Path(f).write_text("def fn(): pass\n")
    ctx, files = _ctx(tmp_path, py_files)

    parse_file_calls: list[str] = []

    class FakeParser:
        socket_path = "/fake.sock"
        auth_secret = "s"
        verified_context = {}

        async def parse_file(self, rel_path, source):
            parse_file_calls.append(rel_path)
            return _parse_result(rel_path)

    class FakeClient:
        @classmethod
        async def connect(cls, **kwargs):
            raise ConnectionRefusedError("engine down")

    monkeypatch.setitem(
        __import__("sys").modules,
        "epistemic_graph.parser",
        SimpleNamespace(RustASTParser=lambda **kw: FakeParser()),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "epistemic_graph.client",
        SimpleNamespace(EpistemicGraphClient=FakeClient),
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.session.GraphSession.from_ambient",
        classmethod(lambda cls: SimpleNamespace(engine_verified_context=lambda: {})),
    )

    result = await parse_phase_mod.execute_parse(
        ctx, {"scan": SimpleNamespace(output=files)}
    )

    assert sorted(parse_file_calls) == ["a.py", "b.py"]
    assert result["symbols_extracted"] == 2


@pytest.mark.asyncio
async def test_markdown_files_are_never_sent_to_the_batch_rpc(tmp_path, monkeypatch):
    """Markdown stays on its regex-only path; only code files hit the engine."""
    md = tmp_path / "note.md"
    md.write_text("CONCEPT:KG-2.106 — a thing\n")
    py = tmp_path / "a.py"
    py.write_text("def fn(): pass\n")
    ctx, files = _ctx(tmp_path, [str(md), str(py)])

    batch_calls: list[list[tuple[str, bytes]]] = []

    class FakeGraphOps:
        async def parse_files(self, files):
            batch_calls.append(list(files))
            return [_parse_result(fp) for fp, _src in files]

    class FakeClient:
        def __init__(self):
            self.graph = FakeGraphOps()

        @classmethod
        async def connect(cls, **kwargs):
            return cls()

        async def close(self):
            pass

    class FakeParser:
        socket_path = "/fake.sock"
        auth_secret = "s"
        verified_context = {}

        async def parse_file(self, rel_path, source):  # pragma: no cover
            raise AssertionError("per-file path should not run here")

    monkeypatch.setitem(
        __import__("sys").modules,
        "epistemic_graph.parser",
        SimpleNamespace(RustASTParser=lambda **kw: FakeParser()),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "epistemic_graph.client",
        SimpleNamespace(EpistemicGraphClient=FakeClient),
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.session.GraphSession.from_ambient",
        classmethod(lambda cls: SimpleNamespace(engine_verified_context=lambda: {})),
    )

    result = await parse_phase_mod.execute_parse(
        ctx, {"scan": SimpleNamespace(output=files)}
    )

    assert len(batch_calls) == 1
    assert len(batch_calls[0]) == 1  # only the .py file
    assert result["symbols_extracted"] == 2  # 1 markdown CONCEPT + 1 symbol
