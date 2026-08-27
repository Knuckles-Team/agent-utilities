"""Characterization tests for the WC1-AU-01 ``_graph_writer_loop`` decomposition.

``_graph_writer_loop`` (agent_utilities/knowledge_graph/core/engine_tasks.py) had
ZERO pre-existing test coverage: it is a ``while True:`` background-daemon loop
with no exit condition, so it was architecturally untestable as a single unit
even before this refactor (grep across tests/ for "GraphWriterDaemon",
"graph_writer_loop", "_merge_staged_node" etc. returns nothing). The
WC1-AU-01 decomposition split it into non-looping, directly-callable units
(``_process_one_staged_graph_item``, ``_write_staged_nodes``,
``_write_staged_edges``, ``_merge_staged_node``, plus several module-level pure
helpers) -- these tests characterize THOSE new units, since there was no prior
automated baseline to run before/after. Per the wave brief: "Author a new test
only for an uncovered branch and name it in your report."
"""

from __future__ import annotations

from unittest.mock import Mock

from agent_utilities.knowledge_graph.core.engine_tasks import (
    TaskManagerMixin,
    _build_staged_node_merge_query,
    _fold_staged_node_metadata,
    _is_non_code_symbol_type,
    _resolve_staged_node_label,
)


def _bare_task_manager() -> TaskManagerMixin:
    """A TaskManagerMixin instance without running its (heavy) __init__.

    Matches the established pattern in
    tests/unit/knowledge_graph/core/test_engine_tasks_embedding_backfill.py.
    """
    return object.__new__(TaskManagerMixin)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# module-level pure helpers
# ---------------------------------------------------------------------------


def test_resolve_staged_node_label_known_type():
    assert _resolve_staged_node_label("file") == "Code"


def test_resolve_staged_node_label_unknown_type_is_capitalized():
    assert _resolve_staged_node_label("custom_thing") == "CustomThing"


def test_is_non_code_symbol_type_true_for_file_on_code_label():
    assert _is_non_code_symbol_type("Code", "file") is True


def test_is_non_code_symbol_type_false_for_non_code_label():
    assert _is_non_code_symbol_type("Agent", "agent") is False


def test_is_non_code_symbol_type_false_when_raw_type_is_literally_code():
    assert _is_non_code_symbol_type("Code", "code") is False


def test_fold_staged_node_metadata_noop_without_metadata_column():
    props = {"name": "x", "extra": "y"}
    _fold_staged_node_metadata(props, {"name"})
    assert props == {"name": "x", "extra": "y"}


def test_fold_staged_node_metadata_folds_extra_keys():
    props = {"name": "x", "extra": "y"}
    _fold_staged_node_metadata(props, {"name", "metadata"})
    assert props == {"name": "x", "metadata": {"extra": "y"}}


def test_fold_staged_node_metadata_parses_existing_json_string():
    props = {"metadata": '{"already": 1}', "extra": "y"}
    _fold_staged_node_metadata(props, {"metadata"})
    assert props["metadata"] == {"already": 1, "extra": "y"}


def test_fold_staged_node_metadata_invalid_json_string_falls_back_to_empty():
    props = {"metadata": "not json", "extra": "y"}
    _fold_staged_node_metadata(props, {"metadata"})
    assert props["metadata"] == {"extra": "y"}


def test_build_staged_node_merge_query_with_properties():
    query, params = _build_staged_node_merge_query("Code", "n1", {"name": "foo.py"})
    assert query == "MERGE (n:Code {id: $id}) SET n.name = $props_name"
    assert params == {"id": "n1", "props_name": "foo.py"}


def test_build_staged_node_merge_query_no_properties():
    query, params = _build_staged_node_merge_query("Code", "n1", {})
    assert query == "MERGE (n:Code {id: $id})"
    assert params == {"id": "n1"}


# ---------------------------------------------------------------------------
# _merge_staged_node / _write_staged_nodes
# ---------------------------------------------------------------------------


def test_merge_staged_node_executes_merge_and_records_type():
    mgr = _bare_task_manager()
    mgr.backend = Mock()
    node = {"id": "n1", "type": "file", "name": "foo.py", "unrecognized_key": "z"}
    schema_cache = {"Code": {"id", "name", "symbol_type", "metadata"}}
    node_type_map: dict = {}

    mgr._merge_staged_node(node, schema_cache, node_type_map)

    assert node_type_map == {"n1": "Code"}
    mgr.backend.execute.assert_called_once()
    query, params = mgr.backend.execute.call_args[0]
    assert query.startswith("MERGE (n:Code {id: $id}) SET")
    assert params["id"] == "n1"
    assert params["props_name"] == "foo.py"
    # raw_type "file" != "code" on a Code-labelled node -> symbol_type preserved
    assert params["props_symbol_type"] == "file"
    # unrecognized_key isn't in valid_keys and "metadata" IS in valid_keys ->
    # folded into metadata (JSON-serialized since dict values are serialized).
    import json

    assert json.loads(params["props_metadata"]) == {"unrecognized_key": "z"}


def test_merge_staged_node_drops_unknown_props_when_no_metadata_column():
    mgr = _bare_task_manager()
    mgr.backend = Mock()
    node = {"id": "a1", "type": "agent", "name": "researcher", "junk": "drop-me"}
    schema_cache = {"Agent": {"id", "name"}}
    node_type_map: dict = {}

    mgr._merge_staged_node(node, schema_cache, node_type_map)

    _query, params = mgr.backend.execute.call_args[0]
    assert "props_junk" not in params
    assert params["props_name"] == "researcher"


def test_write_staged_nodes_skips_entries_missing_id_or_type():
    mgr = _bare_task_manager()
    mgr.backend = Mock()
    nodes = [
        {"id": "n1", "type": "file", "name": "a.py"},
        {"name": "no id or type here"},
    ]
    mgr._write_staged_nodes(nodes, {"Code": {"id", "name"}})
    assert mgr.backend.execute.call_count == 1


# ---------------------------------------------------------------------------
# _write_staged_edges
# ---------------------------------------------------------------------------


def test_write_staged_edges_links_valid_edges():
    mgr = _bare_task_manager()
    mgr.link_nodes = Mock()
    edges = [{"source": "a", "target": "b", "type": "calls"}]
    mgr._write_staged_edges(edges)
    mgr.link_nodes.assert_called_once_with("a", "b", "CALLS")


def test_write_staged_edges_skips_incomplete_edges():
    mgr = _bare_task_manager()
    mgr.link_nodes = Mock()
    edges = [{"source": "a", "type": "calls"}]  # missing "target"
    mgr._write_staged_edges(edges)
    mgr.link_nodes.assert_not_called()


def test_write_staged_edges_swallows_dangling_reference_errors():
    """A dangling edge must not raise -- the staged item is retried wholesale
    by the caller on failure, so one bad edge must not doom the whole batch."""
    mgr = _bare_task_manager()
    mgr.link_nodes = Mock(side_effect=RuntimeError("node not found"))
    edges = [{"source": "a", "target": "ghost", "type": "calls"}]
    mgr._write_staged_edges(edges)  # must not raise
    mgr.link_nodes.assert_called_once()


# ---------------------------------------------------------------------------
# _process_one_staged_graph_item
# ---------------------------------------------------------------------------


def test_process_one_staged_graph_item_writes_and_acks():
    mgr = _bare_task_manager()
    mgr.backend = Mock()
    mgr.link_nodes = Mock()
    mgr._submission_queue = Mock()

    item = (
        "item-1",
        "job-1",
        {
            "nodes": [{"id": "n1", "type": "file", "name": "a.py"}],
            "edges": [{"source": "n1", "target": "n2", "type": "imports"}],
        },
    )
    schema_cache = {"Code": {"id", "name"}}

    mgr._process_one_staged_graph_item(item, schema_cache)

    mgr.backend.execute.assert_called_once()
    mgr.link_nodes.assert_called_once_with("n1", "n2", "IMPORTS")
    mgr._submission_queue.ack_staged_graph.assert_called_once_with("item-1")
