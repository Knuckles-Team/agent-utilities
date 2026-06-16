"""Extraction (Rust ParseFile → entities) + classification (CONCEPT:KG-2.8)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.classify import (
    TestThresholds,
    classify_test,
)
from agent_utilities.knowledge_graph.enrichment.extractors.code_test import (
    entities_from_index_result,
    entities_from_parse_result,
    resolve_covers,
)


def _fn(name, is_test, **kw):
    props = {
        "symbol_type": "Function",
        "name": name,
        "line": "1",
        "ast_hash": f"hash-{name}",
        "file_path": kw.pop("file_path", "test_t.py"),
        "is_test": "true" if is_test else "false",
    }
    props.update({k: str(v) for k, v in kw.items()})
    return {"node_id": f"symbol:{name}", "node_type": "SYMBOL", "properties": props}


def test_maps_rust_parse_result_to_entities():
    parsed = {
        "nodes": [
            _fn(
                "test_mock_heavy",
                True,
                assert_count=1,
                mock_count=3,
                fixture_count=2,
                calls="MagicMock,compute",
            ),
            _fn("compute", False, file_path="app.py"),
            {
                "node_id": "symbol:C",
                "node_type": "SYMBOL",
                "properties": {
                    "symbol_type": "Class",
                    "name": "Widget",
                    "line": "1",
                    "ast_hash": "h",
                    "file_path": "app.py",
                },
            },
        ]
    }
    res = entities_from_parse_result("test_t.py", "chash", parsed)
    assert {t.name for t in res.tests} == {"test_mock_heavy"}
    assert {c.name for c in res.code} == {"compute", "Widget"}
    t = res.tests[0]
    assert t.assert_count == 1 and t.mock_count == 3 and t.fixture_count == 2


def test_classify_mock_heavy_and_dormant_and_assertionfree():
    th = TestThresholds()
    mock_heavy = entities_from_parse_result(
        "test_t.py", "h", {"nodes": [_fn("test_a", True, assert_count=0, mock_count=4)]}
    ).tests[0]
    codes = {i.code for i in classify_test(mock_heavy, th)}
    assert "MockHeavyTest" in codes and "AssertionFreeTest" in codes

    dormant = entities_from_parse_result(
        "test_t.py",
        "h",
        {"nodes": [_fn("test_b", True, is_skipped="true", marks="skip")]},
    ).tests[0]
    assert {i.code for i in classify_test(dormant, th)} == {"DormantTest"}

    healthy = entities_from_parse_result(
        "test_t.py",
        "h",
        {"nodes": [_fn("test_c", True, assert_count=3, mock_count=0, fixture_count=1)]},
    ).tests[0]
    assert classify_test(healthy, th) == []


def test_resolve_covers_links_test_to_called_code():
    t = entities_from_parse_result(
        "test_t.py",
        "h",
        {"nodes": [_fn("test_compute", True, assert_count=2, calls="compute,helper")]},
    )
    app = entities_from_parse_result(
        "app.py",
        "h2",
        {"nodes": [_fn("compute", False, file_path="app.py")]},
    )
    edges = resolve_covers([t, app])
    assert len(edges) == 1
    assert edges[0].rel_type == "COVERS"
    assert edges[0].source == "test:test_t.py::test_compute"
    assert edges[0].target == "code:app.py::compute"


def _isym(node_id, name, file_path, sym_type="Function"):
    return {
        "node_id": node_id,
        "node_type": "SYMBOL",
        "properties": {
            "symbol_type": sym_type,
            "name": name,
            "line": "1",
            "ast_hash": node_id,
            "file_path": file_path,
            "is_test": "false",
        },
    }


def test_entities_from_index_result_maps_symbols_and_resolved_edges():
    """One merged IndexResult → per-file entities + resolved CALLS/INHERITS edges
    bound to entity ids, carrying strategy/confidence (CONCEPT:KG-2.100)."""
    index = {
        "nodes": [
            _isym("symbol:caller", "caller", "app.py"),
            _isym("symbol:helper", "helper", "app.py"),
            _isym("symbol:Base", "Base", "m.py", sym_type="Class"),
            _isym("symbol:Child", "Child", "m.py", sym_type="Class"),
        ],
        "edges": [
            {
                "source": "symbol:caller",
                "target": "symbol:helper",
                "edge_type": "calls",
                "properties": {"name": "helper", "strategy": "same_file", "confidence": "0.90"},
            },
            {
                "source": "symbol:Child",
                "target": "symbol:Base",
                "edge_type": "inherits",
                "properties": {"name": "Base"},
            },
        ],
    }
    results, edges = entities_from_index_result(index, {"app.py": "h1", "m.py": "h2"})

    # Per-file entity extraction off the merged nodes.
    code_ids = {c.id for r in results for c in r.code}
    assert "code:app.py::caller" in code_ids and "code:m.py::Child" in code_ids

    calls = [e for e in edges if e.rel_type == "CALLS"]
    assert len(calls) == 1
    assert calls[0].source == "code:app.py::caller"
    assert calls[0].target == "code:app.py::helper"
    assert calls[0].props == {"strategy": "same_file", "confidence": "0.90"}

    inh = [e for e in edges if e.rel_type == "INHERITS"]
    assert len(inh) == 1
    assert inh[0].source == "code:m.py::Child" and inh[0].target == "code:m.py::Base"
