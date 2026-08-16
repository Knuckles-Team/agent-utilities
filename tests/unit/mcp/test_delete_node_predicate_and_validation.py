"""BUG-049 — a destructive action must never report success it did not perform.

`graph_write action=delete_node` used to be, in full:

    engine.delete_node(node_id)
    return f"Node {node_id} deleted."

`node_id` defaults to ``""``, was never validated, and ``node_type`` was ignored
entirely. Deleting by predicate therefore returned ``"Node  deleted."`` -- note the
empty id -- having removed nothing. Reproduced live against the running engine: the
type count was unchanged afterwards.

That is worse than a plain no-op, because the caller is told the opposite of what
happened. Here specifically it could convince an operator that private conversational
data had been purged when it was still resident.

The sibling branches in the same function (`add_node`, `add_edge`,
`register_external_graph`) all validated their required arguments. Only the two
DESTRUCTIVE branches did not.
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

import pytest


class _MockMCP:
    def __init__(self) -> None:
        self.funcs = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self.funcs[fn.__name__] = fn
            return fn

        return decorator

    def custom_route(self, *args, **kwargs):
        def decorator(fn):
            self.funcs[fn.__name__] = fn
            return fn

        return decorator


@pytest.fixture
def graph_write_and_engine(tmp_path, monkeypatch):
    """Register the real tool surface against a fake engine we can assert on.

    The engine is resolved AT CALL TIME via ``kg_server._resolve_target_engines``
    (not the build-time ``_get_engine``), so that is what must be patched — and it
    must stay patched while the tool runs. Getting this wrong makes every assertion
    pass against ``"Error: IntelligenceGraphEngine not active."``, i.e. a test that
    is green without the fix.

    Each test also passes the empty-string defaults EXPLICITLY. Calling the
    registered function directly bypasses FastMCP's ``Field`` default resolution,
    so an omitted ``node_id`` arrives as a truthy ``FieldInfo`` object rather than
    ``""`` and no falsy check can fire. The live surface resolves them to ``""`` --
    which is exactly how the original defect produced ``"Node  deleted."``.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    mock_mcp = _MockMCP()
    engine = MagicMock()
    engine.backend = MagicMock()
    engine.backend.read_only = False
    with patch(
        "agent_utilities.mcp.server_factory.create_mcp_server",
        return_value=(None, mock_mcp, []),
    ):
        with patch("agent_utilities.mcp.kg_server._get_engine", return_value=engine):
            from agent_utilities.mcp.kg_server import _build_server

            _build_server(bootstrap=False)

    from agent_utilities.mcp import kg_server

    registry = MagicMock()
    registry.is_writable.return_value = True
    monkeypatch.setattr(
        kg_server,
        "_resolve_target_engines",
        lambda _t: ([("primary", engine)], {}, False),
    )
    monkeypatch.setattr(kg_server, "get_connection_registry", lambda: registry)
    return mock_mcp.funcs["graph_write"], engine


@pytest.mark.asyncio
async def test_known_bad_input_no_id_and_no_type_refuses_and_deletes_nothing(
    graph_write_and_engine,
) -> None:
    """THE known-bad input: the exact call that used to say 'Node  deleted.'"""
    graph_write, engine = graph_write_and_engine

    result = await graph_write(action="delete_node", node_id="", node_type="")

    assert "Error" in str(result), (
        "an empty delete must FAIL, not report success -- this is the BUG-049 "
        f"regression; got {result!r}"
    )
    engine.delete_node.assert_not_called()


@pytest.mark.asyncio
async def test_delete_by_node_id_still_works(graph_write_and_engine) -> None:
    graph_write, engine = graph_write_and_engine

    result = await graph_write(action="delete_node", node_id="node-1", node_type="")

    engine.delete_node.assert_called_once_with("node-1")
    assert "node-1" in str(result)


@pytest.mark.asyncio
async def test_predicate_delete_removes_every_match_and_reports_the_real_count(
    graph_write_and_engine,
) -> None:
    """A predicate delete must actually delete, and report what it deleted."""
    graph_write, engine = graph_write_and_engine
    engine.get_nodes_by_label.return_value = [
        ("memento-1", {}),
        ("memento-2", {}),
        ("memento-3", {}),
    ]

    result = await graph_write(action="delete_node", node_id="", node_type="Memento")

    # Enumerated engine-side, NOT through Cypher: the query path applies RLS row
    # filtering, so a Cypher enumeration would silently under-delete.
    engine.get_nodes_by_label.assert_called_once_with("Memento", 0)
    assert [c.args[0] for c in engine.delete_node.call_args_list] == [
        "memento-1",
        "memento-2",
        "memento-3",
    ]
    assert "3" in str(result), f"must report the real count, got {result!r}"


@pytest.mark.asyncio
async def test_predicate_delete_reports_zero_rather_than_claiming_success(
    graph_write_and_engine,
) -> None:
    """Nothing matched is a truthful 0 -- never an unconditional 'deleted.'"""
    graph_write, engine = graph_write_and_engine
    engine.get_nodes_by_label.return_value = []

    result = await graph_write(action="delete_node", node_id="", node_type="NoSuchType")

    engine.delete_node.assert_not_called()
    assert "0" in str(result)


@pytest.mark.asyncio
async def test_predicate_delete_uses_backend_nodes_by_label_when_engine_lacks_it(
    graph_write_and_engine,
) -> None:
    """An IntelligenceGraphEngine exposes the label index on its BACKEND.

    The first version of this fix called ``engine.get_nodes_by_label`` only, which
    raised AttributeError against the engine ``_resolve_target_engines`` actually
    returns. Deleted nothing, but at least failed loudly rather than claiming
    success -- which is the whole point of BUG-049.
    """
    graph_write, engine = graph_write_and_engine
    del engine.get_nodes_by_label  # this engine shape does not have it
    engine.backend.nodes_by_label.return_value = [("m-1", {}), ("m-2", {})]

    result = await graph_write(action="delete_node", node_id="", node_type="Memento")

    engine.backend.nodes_by_label.assert_called_once_with("Memento", 0)
    assert [c.args[0] for c in engine.delete_node.call_args_list] == ["m-1", "m-2"]
    assert "2" in str(result)


@pytest.mark.asyncio
async def test_no_label_index_accessor_refuses_rather_than_scanning(
    graph_write_and_engine,
) -> None:
    """No accessor anywhere => explicit error, never a Cypher fallback."""
    graph_write, engine = graph_write_and_engine
    del engine.get_nodes_by_label
    del engine.backend.nodes_by_label
    del engine.backend.get_nodes_by_label

    result = await graph_write(action="delete_node", node_id="", node_type="Memento")

    assert "Error" in str(result) and "label index" in str(result)
    engine.delete_node.assert_not_called()


@pytest.mark.asyncio
async def test_delete_edge_known_bad_input_refuses(graph_write_and_engine) -> None:
    """delete_edge carried the identical missing-validation defect."""
    graph_write, engine = graph_write_and_engine

    result = await graph_write(
        action="delete_edge", source_id="a", target_id="", rel_type=""
    )

    assert "Error" in str(result), f"got {result!r}"
    engine.delete_edge.assert_not_called()
