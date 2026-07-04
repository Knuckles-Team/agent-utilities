"""CONCEPT:EG-KG.storage.nonblocking-checkpoint — multi-language code discovery + language classification.

The Rust engine parses Python/JS/TS/Go/Rust/Java/C/C++/C#; the Python side must
(1) discover all of those source files (not just ``*.py``) and (2) carry the
``language`` + precise ``kind`` from the parser onto each ``CodeEntity``.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.extractors.code_test import (
    entities_from_parse_result,
)
from agent_utilities.knowledge_graph.enrichment.pipeline import (
    SOURCE_EXTENSIONS,
    discover_source_files,
)


def test_discovers_all_languages_and_skips_vendor(tmp_path):
    (tmp_path / "app.py").write_text("x = 1\n")
    (tmp_path / "Widget.java").write_text("class Widget {}\n")
    (tmp_path / "main.go").write_text("package main\n")
    (tmp_path / "lib.rs").write_text("pub fn f() {}\n")
    (tmp_path / "ui.tsx").write_text("export const A = () => null;\n")
    (tmp_path / "svc.cs").write_text("class S {}\n")
    (tmp_path / "README.md").write_text("# docs\n")  # not source
    # vendored deps must be skipped
    vendor = tmp_path / "node_modules" / "dep"
    vendor.mkdir(parents=True)
    (vendor / "index.js").write_text("module.exports = {};\n")

    found = {p.name for p in discover_source_files(tmp_path)}
    assert found == {"app.py", "Widget.java", "main.go", "lib.rs", "ui.tsx", "svc.cs"}
    assert "README.md" not in found
    assert "index.js" not in found  # node_modules skipped


def test_source_extensions_cover_the_major_languages():
    for ext in (
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".go",
        ".rs",
        ".java",
        ".c",
        ".cpp",
        ".cs",
    ):
        assert ext in SOURCE_EXTENSIONS


def test_language_and_kind_detail_wired_from_parser():
    # Simulate the Rust ParseFile output for a Java interface + method.
    parsed = {
        "nodes": [
            {
                "properties": {
                    "name": "Drawable",
                    "symbol_type": "Class",
                    "kind_detail": "interface",
                    "language": "java",
                    "line": "1",
                    "ast_hash": "abc",
                }
            },
            {
                "properties": {
                    "name": "draw",
                    "symbol_type": "Function",
                    "kind_detail": "method",
                    "language": "java",
                    "line": "2",
                    "ast_hash": "def",
                }
            },
        ]
    }
    res = entities_from_parse_result("Drawable.java", "hash", parsed)
    by_name = {c.name: c for c in res.code}
    assert by_name["Drawable"].kind == "interface"
    assert by_name["Drawable"].language == "java"
    assert by_name["draw"].kind == "method"
    assert by_name["draw"].language == "java"


def test_missing_kind_detail_falls_back_to_coarse_bucket():
    # Older engine builds emit no kind_detail/language — must not crash.
    parsed = {
        "nodes": [
            {"properties": {"name": "C", "symbol_type": "Class", "line": "1"}},
            {"properties": {"name": "f", "symbol_type": "Function", "line": "2"}},
        ]
    }
    res = entities_from_parse_result("legacy.py", "hash", parsed)
    by_name = {c.name: c for c in res.code}
    assert by_name["C"].kind == "class"
    assert by_name["f"].kind == "function"
    assert by_name["C"].language == ""
