"""Queries over enriched test entities (CONCEPT:KG-2.8 Phase 1).

"Which pytests need work" is now a graph query against stored enrichment facts —
not an ad-hoc script. Runs through the single ``GraphBackend.execute`` interface,
so it works on every backend.
"""

from __future__ import annotations

import json
from typing import Any

_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


def tests_needing_work(backend: Any, limit: int | None = None) -> list[dict[str, Any]]:
    """Return enriched Test nodes flagged ``needs_work``, with their issues.

    Sorted worst-first (by highest-severity issue). Each item carries the test's
    identity, metrics, and decoded issue list (evidence) for an agent/LLM to act
    on or explain.
    """
    rows = backend.execute(
        "MATCH (t:Test {needs_work: true}) "
        "RETURN t.id as id, t.name as name, t.file_path as file_path, "
        "t.line as line, t.assert_count as assert_count, "
        "t.mock_count as mock_count, t.fixture_count as fixture_count, "
        "t.is_skipped as is_skipped, t.issues as issues"
    )
    out: list[dict[str, Any]] = []
    for r in rows or []:
        issues = r.get("issues")
        if isinstance(issues, str):
            try:
                issues = json.loads(issues)
            except json.JSONDecodeError:
                issues = []
        r["issues"] = issues or []
        out.append(r)

    def worst(item: dict[str, Any]) -> int:
        sevs = [_SEVERITY_RANK.get(i.get("severity"), 3) for i in item["issues"]]
        return min(sevs) if sevs else 3

    out.sort(key=worst)
    return out[:limit] if limit else out


def how_implemented(backend: Any, name: str) -> list[dict[str, Any]]:
    """Explain how a symbol is implemented: summary, patterns, responsibilities."""
    rows = backend.execute(
        "MATCH (c:Code {name: $name}) "
        "RETURN c.id as id, c.name as name, c.kind as kind, c.file_path as file_path, "
        "c.patterns as patterns, c.summary as summary, "
        "c.responsibilities as responsibilities",
        {"name": name},
    )
    out = []
    for r in rows or []:
        resp = r.get("responsibilities")
        if isinstance(resp, str):
            try:
                resp = json.loads(resp)
            except json.JSONDecodeError:
                resp = []
        r["responsibilities"] = resp or []
        r["patterns"] = [p for p in (r.get("patterns", "") or "").split(",") if p]
        out.append(r)
    return out


def code_by_pattern(
    backend: Any, pattern: str, limit: int | None = None
) -> list[dict[str, Any]]:
    """Find code entities tagged with a given design pattern."""
    rows = backend.execute(
        "MATCH (c:Code) WHERE c.patterns CONTAINS $p "
        "RETURN c.name as name, c.kind as kind, c.file_path as file_path, "
        "c.patterns as patterns",
        {"p": pattern},
    )
    out = []
    for r in rows or []:
        r["patterns"] = [p for p in (r.get("patterns", "") or "").split(",") if p]
        if pattern in r["patterns"]:  # exact tag, not substring
            out.append(r)
    return out[:limit] if limit else out


def enrichment_coverage(backend: Any, graph_name: str | None = None) -> dict[str, Any]:
    """Per-category enrichment completeness for the KG (CONCEPT:KG-2.8).

    The gauge for "how enriched is the graph by category": structural ingest
    lands nodes immediately while the background daemon backfills LLM cards, so
    each category reports totals plus a ``coverage`` ratio in ``[0, 1]`` for its
    LLM-enriched dimension. Counts are computed client-side (the L1 Cypher
    subset can't ``GROUP BY``), so this works on every backend. ``graph_name``
    is accepted for API symmetry; the backend already targets a single graph.
    """

    def _rows(q: str) -> list[dict[str, Any]]:
        try:
            return backend.execute(q) or []
        except Exception:
            return []

    def _node(r: Any) -> dict[str, Any]:
        if isinstance(r, dict):
            inner = r.get("n")
            return inner if isinstance(inner, dict) else r
        return {}

    out: dict[str, Any] = {}

    # Codebase: capability-card coverage over real (ast_hash'd) Code symbols.
    code = [
        c
        for c in (_node(r) for r in _rows("MATCH (n:Code) RETURN n"))
        if c.get("ast_hash")
    ]
    code_total = len(code)
    code_with_cards = sum(1 for c in code if (c.get("summary") or "").strip())
    tests = [_node(r) for r in _rows("MATCH (n:Test) RETURN n")]
    tests_needing_work = sum(
        1 for t in tests if t.get("needs_work") in (True, "true", "True")
    )
    out["codebase"] = {
        "code_total": code_total,
        "code_with_cards": code_with_cards,
        "cards_pending": code_total - code_with_cards,
        "tests_total": len(tests),
        "tests_needing_work": tests_needing_work,
        "coverage": round(code_with_cards / code_total, 4) if code_total else 0.0,
    }

    # Documents: concept-extraction presence (proxy without edge traversal).
    docs = _rows("MATCH (n:Document) RETURN n")
    concepts = _rows("MATCH (n:Concept) RETURN n")
    out["documents"] = {
        "document_total": len(docs),
        "concept_total": len(concepts),
        "coverage": round(min(1.0, len(concepts) / len(docs)), 4) if docs else 0.0,
    }

    out["features"] = {"feature_total": len(_rows("MATCH (n:Feature) RETURN n"))}
    return out


def list_features(backend: Any, limit: int | None = None) -> list[dict[str, Any]]:
    """List discovered features (call-graph communities), largest first."""
    rows = backend.execute(
        "MATCH (f:Feature) "
        "RETURN f.id as id, f.name as name, f.summary as summary, "
        "f.size as size, f.patterns as patterns"
    )
    out = []
    for r in rows or []:
        try:
            r["size"] = int(r.get("size", 0) or 0)
        except (TypeError, ValueError):
            r["size"] = 0
        r["patterns"] = [p for p in (r.get("patterns", "") or "").split(",") if p]
        out.append(r)
    out.sort(key=lambda x: x["size"], reverse=True)
    return out[:limit] if limit else out
