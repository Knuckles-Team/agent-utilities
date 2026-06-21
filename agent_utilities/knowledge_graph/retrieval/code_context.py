#!/usr/bin/python
from __future__ import annotations

"""Synthesized, cited codebase Q&A over the KG (CONCEPT:KG-2.134 / KG-2.135).

``code_context(query, intent)`` composes the already-built code-intelligence
primitives — the typed/scope-resolved call graph (KG-2.100), embedder-free
similar-code (KG-2.101), code↔service routes (KG-2.102), git change-coupling
(KG-2.104), the ``CONCEPT:`` markers, and ingested docs — into **one grounded
explanation with ``file:line`` citations**. The point: an agent learns *how an
area works / where a symbol is used / what a change impacts* by querying the KG
instead of grep-then-read across the tree (the codebase-context-via-KG vision).

Design choices that make it *native* (default-on, fast, robust):

* **Deterministic & embedder-free.** Every section is a read over the resolved
  ``:Code`` graph (pure Cypher), so it answers even when the remote vLLM/embedder
  is unavailable. The prose is templated from the retrieved structure; every
  claim carries a citation — no hallucinated synthesis.
* **Degrades gracefully.** Sections whose enrichment has not run yet (docs,
  ``similar_to``, ``FILE_CHANGES_WITH``, ``MENTIONED_IN`` concepts) simply come
  back empty and are omitted — the answer is still grounded on what *is* present
  (the call graph), and richens automatically as the delta sweep enriches.
* **Cross-repo by name (KG-2.135).** Anchoring callers/usages by *symbol name*
  (not node id) aggregates references across every ingested repo in one query —
  ``run_agent``'s callers span agent-utilities, the frameworks, the agents. The
  ``/au`` source-mount is normalized to its canonical path so the same file does
  not cite twice.

The capability id every answer carries feeds the reads-avoided reward loop
(CONCEPT:AHE-3.61) via :class:`FeedbackService`.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.core.source_paths import normalize_path, repo_of

VALID_INTENTS = ("how", "usage", "impact")

# Stable :Code projection (mirrors query_tools._CODE_COLS for a consistent shape).
_CODE_COLS = (
    "{v}.id AS id, {v}.name AS name, {v}.file_path AS file_path, "
    "{v}.line AS line, {v}.language AS language, {v}.kind_detail AS kind, "
    "{v}.instance AS instance, {v}.source_system AS source_system"
)

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class CodeContextAnswer:
    """The synthesized, cited answer returned by :func:`build_code_context`."""

    query: str
    intent: str
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    sections: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    anchors: list[dict[str, Any]] = field(default_factory=list)
    capability_id: str = ""
    used_primitives: list[str] = field(default_factory=list)
    cross_repo: bool = False
    coverage: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "query": self.query,
            "intent": self.intent,
            "answer": self.answer,
            "citations": self.citations,
            "sections": self.sections,
            "anchors": self.anchors,
            "capability_id": self.capability_id,
            "used_primitives": self.used_primitives,
            "cross_repo": self.cross_repo,
            "coverage": self.coverage,
        }


def _rows(engine: Any, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Run a read-only query, tolerant of either engine read API; never raises."""
    try:
        qc = getattr(engine, "query_cypher", None)
        if callable(qc):
            return list(qc(cypher, params) or [])
        backend = getattr(engine, "backend", None)
        if backend is not None and hasattr(backend, "execute"):
            return list(backend.execute(cypher, params) or [])
    except Exception:  # pragma: no cover - read best-effort by design
        return []
    return []


def _code_row(row: dict[str, Any]) -> dict[str, Any]:
    """Project a :Code result row into a stable, path-normalized citation dict."""
    return {
        "id": row.get("id"),
        "symbol": row.get("name"),
        "file": normalize_path(row.get("file_path")),
        "line": _as_int(row.get("line")),
        "kind": row.get("kind"),
        "language": row.get("language"),
        "source_system": row.get("source_system"),
    }


# ---------------------------------------------------------------------------
# Anchor resolution
# ---------------------------------------------------------------------------
def resolve_anchors(
    engine: Any,
    *,
    query: str = "",
    node_id: str = "",
    source_system: str = "",
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Resolve the question to the :Code definition node(s) it is about.

    ``node_id`` (exact) wins; otherwise the longest identifier-like token in the
    query is matched against ``:Code.name`` (exact first, then case-insensitive
    contains) so a natural-language question still anchors on a real symbol.
    """
    cols = _CODE_COLS.format(v="c")
    if node_id:
        rows = _rows(
            engine,
            f"MATCH (c:Code) WHERE c.id = $id RETURN {cols} LIMIT {int(limit)}",
            {"id": node_id},
        )
        return [_code_row(r) for r in rows]

    tokens = sorted(set(_TOKEN_RE.findall(query or "")), key=len, reverse=True)
    src_clause = " AND c.source_system = $src" if source_system else ""
    params: dict[str, Any] = {}
    if source_system:
        params["src"] = source_system

    for tok in tokens[:6]:
        p = dict(params, tok=tok)
        rows = _rows(
            engine,
            f"MATCH (c:Code) WHERE c.name = $tok{src_clause} "
            f"RETURN {cols} LIMIT {int(limit)}",
            p,
        )
        if rows:
            return [_code_row(r) for r in rows]
    # Fall back to a case-insensitive contains on the longest token.
    if tokens:
        p = dict(params, tok=tokens[0].lower())
        rows = _rows(
            engine,
            f"MATCH (c:Code) WHERE toLower(c.name) CONTAINS $tok{src_clause} "
            f"RETURN {cols} LIMIT {int(limit)}",
            p,
        )
        return [_code_row(r) for r in rows]
    return []


# ---------------------------------------------------------------------------
# Section primitives (each best-effort; empty when its enrichment has not run)
# ---------------------------------------------------------------------------
def _callers(engine: Any, name: str, limit: int) -> list[dict[str, Any]]:
    """find_references — incoming `calls` edges, anchored by NAME (cross-repo)."""
    cols = _CODE_COLS.format(v="caller")
    rows = _rows(
        engine,
        f"MATCH (caller:Code)-[:calls]->(def:Code) WHERE def.name = $n "
        f"RETURN DISTINCT {cols} LIMIT {int(limit)}",
        {"n": name},
    )
    return [_code_row(r) for r in rows]


def _callees(engine: Any, name: str, depth: int, limit: int) -> list[dict[str, Any]]:
    cols = _CODE_COLS.format(v="callee")
    depth = max(1, min(6, int(depth)))
    rows = _rows(
        engine,
        f"MATCH (s:Code)-[:calls*1..{depth}]->(callee:Code) WHERE s.name = $n "
        f"RETURN DISTINCT {cols} LIMIT {int(limit)}",
        {"n": name},
    )
    return [_code_row(r) for r in rows]


def _impact(engine: Any, name: str, depth: int, limit: int) -> list[dict[str, Any]]:
    cols = _CODE_COLS.format(v="caller")
    depth = max(1, min(6, int(depth)))
    rows = _rows(
        engine,
        f"MATCH (caller:Code)-[:calls*1..{depth}]->(t:Code) WHERE t.name = $n "
        f"RETURN DISTINCT {cols} LIMIT {int(limit)}",
        {"n": name},
    )
    return [_code_row(r) for r in rows]


def _similar(engine: Any, node_id: str, limit: int) -> list[dict[str, Any]]:
    rows = _rows(
        engine,
        "MATCH (s:Code {id: $id})-[r]-(t:Code) "
        "WHERE type(r) IN ['similar_to', 'SIMILAR_TO'] "
        "RETURN t.id AS id, t.name AS name, t.file_path AS file_path, "
        "t.line AS line, t.kind_detail AS kind, t.language AS language, "
        "t.source_system AS source_system, r.score AS score",
        {"id": node_id},
    )
    out = [{**_code_row(r), "score": r.get("score")} for r in rows]
    out.sort(key=lambda d: float(d.get("score") or 0), reverse=True)
    return out[: int(limit)]


def _routes(engine: Any, name: str, limit: int) -> list[dict[str, Any]]:
    rows = _rows(
        engine,
        "MATCH (h:Code)-[r2]->(rt:Route) WHERE type(r2) IN ['SERVES', 'serves'] "
        "AND h.name = $n "
        "OPTIONAL MATCH (rt)-[r3]->(svc) WHERE type(r3) IN ['SERVED_BY', 'served_by'] "
        "RETURN rt.method AS method, rt.path AS path, h.name AS handler, "
        "svc.id AS service",
        {"n": name},
    )
    return [
        {
            "method": r.get("method"),
            "path": r.get("path"),
            "handler": r.get("handler"),
            "service": r.get("service"),
        }
        for r in rows[: int(limit)]
    ]


def _concepts(engine: Any, file_path: str, limit: int) -> list[dict[str, Any]]:
    """Owning CONCEPT: markers — concept –MENTIONED_IN→ the anchor's file."""
    if not file_path:
        return []
    rows = _rows(
        engine,
        "MATCH (c)-[r]->(f) WHERE type(r) IN ['MENTIONED_IN', 'mentioned_in'] "
        "AND (c.type = 'CONCEPT' OR c.concept_id IS NOT NULL) "
        "AND (f.file_path = $fp OR f.path = $fp OR f.id CONTAINS $fp) "
        "RETURN DISTINCT c.concept_id AS concept, c.definition AS definition",
        {"fp": file_path},
    )
    return [
        {"concept": r.get("concept"), "definition": r.get("definition")}
        for r in rows[: int(limit)]
        if r.get("concept")
    ]


def _coupling(engine: Any, file_path: str, limit: int) -> list[dict[str, Any]]:
    """Hidden git change-coupling — files that historically change together."""
    if not file_path:
        return []
    rows = _rows(
        engine,
        "MATCH (a)-[r]-(b) WHERE type(r) IN ['FILE_CHANGES_WITH', 'file_changes_with'] "
        "AND (a.file_path = $fp OR a.path = $fp OR a.id CONTAINS $fp) "
        "RETURN DISTINCT b.id AS coupled, b.path AS path, b.file_path AS file_path, "
        "r.support AS support",
        {"fp": file_path},
    )
    out = [
        {
            "file": normalize_path(
                r.get("file_path") or r.get("path") or r.get("coupled")
            ),
            "support": _as_int(r.get("support")),
        }
        for r in rows
    ]
    out.sort(key=lambda d: d.get("support") or 0, reverse=True)
    return out[: int(limit)]


def _docs(engine: Any, query: str, limit: int) -> list[dict[str, Any]]:
    """Ingested doc/chunk snippets matching the query terms (best-effort)."""
    tokens = sorted(set(_TOKEN_RE.findall(query or "")), key=len, reverse=True)
    if not tokens:
        return []
    rows = _rows(
        engine,
        "MATCH (d) WHERE (d:Chunk OR d:Document) AND d.text IS NOT NULL "
        "AND toLower(d.text) CONTAINS $tok "
        "RETURN d.id AS id, substring(d.text, 0, 280) AS snippet, "
        "d.source_path AS source_path LIMIT $k",
        {"tok": tokens[0].lower(), "k": int(limit)},
    )
    return [
        {
            "id": r.get("id"),
            "snippet": r.get("snippet"),
            "source": normalize_path(r.get("source_path")),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Cross-repo usage (CONCEPT:KG-2.135)
# ---------------------------------------------------------------------------
def cross_repo_usages(engine: Any, symbol: str, limit: int = 200) -> dict[str, Any]:
    """Every usage of a published ``symbol`` across the whole fleet, in one query.

    Anchors by name (not node id), so callers resolve across every ingested repo;
    groups them by repo/instance so "where is ``create_model`` used across the 62
    agent packages?" is a single answer. Definition sites are returned too. Honest
    about scope: this surfaces *resolved* ``calls`` references — an import that the
    intra-repo resolver left unbound is not yet a cross-repo edge.
    """
    defs = resolve_anchors(engine, query=symbol, limit=limit)
    callers = _callers(engine, symbol, limit)
    by_repo: dict[str, list[dict[str, Any]]] = {}
    for c in callers:
        repo = repo_of(c.get("file") or "")
        by_repo.setdefault(repo, []).append(c)
    return {
        "symbol": symbol,
        "definitions": defs,
        "usage_count": len(callers),
        "repos": sorted(by_repo),
        "usages_by_repo": by_repo,
    }


# ---------------------------------------------------------------------------
# Citation dedup + synthesis
# ---------------------------------------------------------------------------
def _dedup_cites(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, int | None]] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        key = (r.get("file") or "", r.get("line"))
        if r.get("file") and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _cite(r: dict[str, Any]) -> str:
    return f"{r.get('file')}:{r.get('line')}" if r.get("line") else str(r.get("file"))


def _make_capability_id(intent: str, anchors: list[dict[str, Any]], query: str) -> str:
    if anchors and anchors[0].get("symbol"):
        key = str(anchors[0]["symbol"])
    else:
        slug = re.sub(r"[^a-z0-9]+", "-", (query or "").lower()).strip("-")[:48]
        key = slug or "adhoc"
    return f"code_context:{intent}:{key}"


def build_code_context(
    engine: Any,
    *,
    query: str = "",
    intent: str = "how",
    node_id: str = "",
    source_system: str = "",
    top_k: int = 10,
    depth: int = 2,
    cross_repo: bool = False,
) -> dict[str, Any]:
    """Compose the code-intelligence primitives into one cited explanation.

    See the module docstring. Returns :meth:`CodeContextAnswer.as_dict`.
    """
    intent = (intent or "how").strip().lower()
    if intent not in VALID_INTENTS:
        intent = "how"
    limit = max(1, min(100, int(top_k)))

    anchors = resolve_anchors(
        engine, query=query, node_id=node_id, source_system=source_system, limit=8
    )
    sections: dict[str, list[dict[str, Any]]] = {}
    used: list[str] = []
    cites: list[dict[str, Any]] = list(anchors)

    if not anchors:
        # No symbol anchor — still try docs so a prose question isn't a dead end.
        docs = _docs(engine, query, limit)
        if docs:
            sections["docs"] = docs
            used.append("docs")
        answer = (
            f"No resolved code symbol matched '{query}'. "
            "The area may not be ingested yet (run a delta sweep), or try a more "
            "specific symbol name."
            if not docs
            else f"No code symbol matched '{query}', but related docs were found."
        )
        cap = _make_capability_id(intent, anchors, query)
        return CodeContextAnswer(
            query=query,
            intent=intent,
            answer=answer,
            citations=[],
            sections=sections,
            anchors=[],
            capability_id=cap,
            used_primitives=used,
            cross_repo=cross_repo,
            coverage={"anchors": 0},
        ).as_dict()

    primary = anchors[0]
    name = primary.get("symbol") or ""
    nid = primary.get("id") or ""
    fp = primary.get("file") or ""

    if intent == "how":
        callees = _callees(engine, name, depth, limit)
        if callees:
            sections["calls"] = callees
            cites += callees
            used.append("call_graph")
        concepts = _concepts(engine, fp, limit)
        if concepts:
            sections["concepts"] = concepts
            used.append("concepts")
        routes = _routes(engine, name, limit)
        if routes:
            sections["routes"] = routes
            used.append("routes")
        docs = _docs(engine, query or name, limit)
        if docs:
            sections["docs"] = docs
            used.append("docs")
    elif intent == "usage":
        callers = _callers(engine, name, limit)
        if callers:
            sections["callers"] = callers
            cites += callers
            used.append("call_graph")
        similar = _similar(engine, nid, limit) if nid else []
        if similar:
            sections["similar"] = similar
            cites += similar
            used.append("similar_code")
        routes = _routes(engine, name, limit)
        if routes:
            sections["routes"] = routes
            used.append("routes")
        # "Where is it used" is inherently a fleet-wide question — surface the
        # cross-repo usage view by default for usage intent (CONCEPT:KG-2.135).
        sections["cross_repo"] = [cross_repo_usages(engine, name, limit)]
        used.append("cross_repo")
        cross_repo = True
    else:  # impact
        impacted = _impact(engine, name, depth, limit)
        if impacted:
            sections["impacted_callers"] = impacted
            cites += impacted
            used.append("impact_of_change")
        coupling = _coupling(engine, fp, limit)
        if coupling:
            sections["change_coupling"] = coupling
            used.append("change_coupling")
        routes = _routes(engine, name, limit)
        if routes:
            sections["routes"] = routes
            used.append("routes")

    citations = _dedup_cites(cites)
    answer = _synthesize(query, intent, primary, sections, citations)
    cap = _make_capability_id(intent, anchors, query)
    return CodeContextAnswer(
        query=query,
        intent=intent,
        answer=answer,
        citations=citations,
        sections=sections,
        anchors=anchors,
        capability_id=cap,
        used_primitives=used,
        cross_repo=cross_repo,
        coverage={
            "anchors": len(anchors),
            "citations": len(citations),
            "sections": {k: len(v) for k, v in sections.items()},
        },
    ).as_dict()


def _synthesize(
    query: str,
    intent: str,
    primary: dict[str, Any],
    sections: dict[str, list[dict[str, Any]]],
    citations: list[dict[str, Any]],
) -> str:
    """Templated, grounded prose — every claim cites a real ``file:line``."""
    name = primary.get("symbol") or query
    kind = primary.get("kind") or "symbol"
    lines: list[str] = [f"`{name}` ({kind}) is defined at {_cite(primary)}."]

    if intent == "how":
        calls = sections.get("calls") or []
        if calls:
            names = ", ".join(f"`{c['symbol']}`" for c in calls[:6] if c.get("symbol"))
            lines.append(f"It calls {names}.")
        for c in sections.get("concepts") or []:
            lines.append(
                f"Implements concept {c['concept']}"
                + (f": {c['definition']}" if c.get("definition") else "")
                + "."
            )
        for rt in (sections.get("routes") or [])[:3]:
            lines.append(
                f"Serves HTTP {rt.get('method')} {rt.get('path')}"
                + (f" (service {rt['service']})" if rt.get("service") else "")
                + "."
            )
        for d in (sections.get("docs") or [])[:2]:
            if d.get("snippet"):
                lines.append(f"Doc: {d['snippet'].strip()} [{d.get('source') or ''}]")
    elif intent == "usage":
        callers = sections.get("callers") or []
        if callers:
            lines.append(
                f"Used by {len(callers)} caller(s): "
                + ", ".join(_cite(c) for c in callers[:6])
                + ("…" if len(callers) > 6 else "")
                + "."
            )
        else:
            lines.append("No resolved callers found (it may be an entry point).")
        cr = sections.get("cross_repo") or []
        if cr:
            info = cr[0]
            lines.append(
                f"Across the fleet: {info.get('usage_count', 0)} usages in "
                f"{len(info.get('repos', []))} repo(s) — {', '.join(info.get('repos', [])[:8])}."
            )
        sim = sections.get("similar") or []
        if sim:
            lines.append(
                "Near-clones: "
                + ", ".join(f"`{s['symbol']}`" for s in sim[:4] if s.get("symbol"))
                + "."
            )
    else:  # impact
        impacted = sections.get("impacted_callers") or []
        if impacted:
            lines.append(
                f"Changing it transitively impacts {len(impacted)} caller(s): "
                + ", ".join(_cite(c) for c in impacted[:6])
                + ("…" if len(impacted) > 6 else "")
                + "."
            )
        else:
            lines.append("No upstream callers resolved — low blast radius.")
        coupling = sections.get("change_coupling") or []
        if coupling:
            lines.append(
                "Historically co-changes with: "
                + ", ".join(c["file"] for c in coupling[:4] if c.get("file"))
                + "."
            )

    lines.append(
        f"({len(citations)} citation(s); query the cited file:line to edit, not to understand.)"
    )
    return " ".join(lines)
