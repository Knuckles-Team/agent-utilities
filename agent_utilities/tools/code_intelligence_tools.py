"""CONCEPT:KG-2.65 — Code-Intelligence Tools: the SWE agent's grounding surface.

OpenHands grounds its CodeActAgent by stuffing file contents into the context and compressing
with a summarizing condenser. We do the opposite: the code ontology is *already live* (KG-2.8
ingestion emits ``Code``/``Test`` nodes with ``calls``/``covers``/``dependsOn`` edges), so
"where is this defined?", "who calls this?", and "which tests cover this?" are **graph
queries**, not context-window gambles. This is the differentiator that lets the agent reason
about a large repo it has never read in full.

:class:`CodeIntelligence` is a pure, synchronous core over an engine's ``backend.execute`` (so it
unit-tests against a fake backend). The ``@tool`` wrappers adapt it to the Pydantic-AI
``RunContext[AgentDeps]`` surface and format results for the model. Edge-label matching is
case-tolerant (``CALLS``/``calls``) because backends normalize labels differently.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic_ai import RunContext

from agent_utilities.harness.tracing import trace

from ..models import AgentDeps
from .knowledge_tools import get_knowledge_engine
from .versioning import tool_version

logger = logging.getLogger(__name__)

_LIMIT = 50


class CodeIntelligence:
    """Symbol-graph queries over the live code ontology (KG-2.65)."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def _execute(self, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        backend = getattr(self.engine, "backend", None)
        execute = getattr(backend, "execute", None)
        if not callable(execute):
            return []
        try:
            rows = execute(cypher, params)
        except Exception as exc:  # noqa: BLE001 - dialect/availability differences degrade to empty
            logger.debug("code-intelligence query failed: %s", exc)
            return []
        return [r for r in (rows or []) if isinstance(r, dict)]

    @staticmethod
    def _match(symbol: str) -> dict[str, Any]:
        # Match on exact name/id or an id suffix (qualified-name tail), so "foo" finds
        # "pkg/mod.py::foo".
        return {"s": symbol, "suffix": f"::{symbol}"}

    def find_definition(self, symbol: str) -> list[dict[str, Any]]:
        cypher = (
            "MATCH (c:Code) WHERE c.name = $s OR c.id = $s OR c.id ENDS WITH $suffix "
            "RETURN c.id AS id, c.name AS name, c.summary AS summary, "
            "c.file AS file, c.path AS path, c.file_path AS file_path "
            f"LIMIT {_LIMIT}"
        )
        return self._execute(cypher, self._match(symbol))

    def who_calls(self, symbol: str) -> list[dict[str, Any]]:
        cypher = (
            "MATCH (caller:Code)-[r]->(c:Code) "
            "WHERE type(r) IN ['CALLS','calls'] AND (c.name = $s OR c.id ENDS WITH $suffix) "
            "RETURN DISTINCT caller.id AS id, caller.name AS name, caller.file AS file "
            f"LIMIT {_LIMIT}"
        )
        return self._execute(cypher, self._match(symbol))

    def impacted_tests(self, symbol: str) -> list[dict[str, Any]]:
        cypher = (
            "MATCH (t:Test)-[r]->(c:Code) "
            "WHERE type(r) IN ['COVERS','covers'] AND (c.name = $s OR c.id ENDS WITH $suffix) "
            "RETURN DISTINCT t.id AS id, t.name AS name "
            f"LIMIT {_LIMIT}"
        )
        return self._execute(cypher, self._match(symbol))

    def call_graph(self, symbol: str, depth: int = 2) -> list[dict[str, Any]]:
        depth = max(1, min(int(depth), 5))  # bound the traversal
        cypher = (
            f"MATCH (c:Code)-[r*1..{depth}]->(d:Code) "
            "WHERE (c.name = $s OR c.id ENDS WITH $suffix) "
            "AND all(x IN r WHERE type(x) IN ['CALLS','calls']) "
            "RETURN DISTINCT d.id AS id, d.name AS name "
            f"LIMIT {_LIMIT}"
        )
        return self._execute(cypher, self._match(symbol))

    def dependencies(self, module: str) -> list[dict[str, Any]]:
        cypher = (
            "MATCH (c:Code)-[r]->(d:Code) "
            "WHERE type(r) IN ['DEPENDS_ON','dependsOn','DEPENDSON'] "
            "AND (c.name = $s OR c.id ENDS WITH $suffix) "
            "RETURN DISTINCT d.id AS id, d.name AS name "
            f"LIMIT {_LIMIT}"
        )
        return self._execute(cypher, self._match(module))


def _fmt(rows: list[dict[str, Any]], empty: str) -> str:
    if not rows:
        return empty
    out = []
    for r in rows:
        loc = r.get("file") or r.get("path") or r.get("file_path") or ""
        line = f"- {r.get('name') or r.get('id')}"
        if r.get("id") and r.get("id") != r.get("name"):
            line += f" [{r['id']}]"
        if loc:
            line += f" ({loc})"
        if r.get("summary"):
            line += f": {str(r['summary'])[:160]}"
        out.append(line)
    return "\n".join(out)


# ── Pydantic-AI tool wrappers ────────────────────────────────────────────────


def _ci(ctx: RunContext[AgentDeps]) -> CodeIntelligence | None:
    engine = get_knowledge_engine(ctx)
    return CodeIntelligence(engine) if engine else None


@trace(name="find_definition", trace_type="TOOL")
@tool_version("1.0.0")
async def find_definition(ctx: RunContext[AgentDeps], symbol: str) -> str:
    """Locate where a code symbol (function/class/method) is defined, via the code graph.

    Prefer this over reading files blindly — it answers "where is X?" as a graph query.
    """
    ci = _ci(ctx)
    if ci is None:
        return "Knowledge Graph not available."
    return _fmt(ci.find_definition(symbol), f"No definition found for '{symbol}'.")


@trace(name="who_calls", trace_type="TOOL")
@tool_version("1.0.0")
async def who_calls(ctx: RunContext[AgentDeps], symbol: str) -> str:
    """List the call sites of a symbol (inbound CALLS edges) — who would break if it changes."""
    ci = _ci(ctx)
    if ci is None:
        return "Knowledge Graph not available."
    return _fmt(ci.who_calls(symbol), f"No callers found for '{symbol}'.")


@trace(name="impacted_tests", trace_type="TOOL")
@tool_version("1.0.0")
async def impacted_tests(ctx: RunContext[AgentDeps], symbol: str) -> str:
    """List the tests that cover a symbol (inbound COVERS edges) — run these after editing it."""
    ci = _ci(ctx)
    if ci is None:
        return "Knowledge Graph not available."
    return _fmt(ci.impacted_tests(symbol), f"No covering tests found for '{symbol}'.")


@trace(name="call_graph", trace_type="TOOL")
@tool_version("1.0.0")
async def call_graph(ctx: RunContext[AgentDeps], symbol: str, depth: int = 2) -> str:
    """Show the transitive callees of a symbol up to ``depth`` hops (outbound CALLS)."""
    ci = _ci(ctx)
    if ci is None:
        return "Knowledge Graph not available."
    return _fmt(ci.call_graph(symbol, depth), f"No call graph found for '{symbol}'.")


@trace(name="dependencies", trace_type="TOOL")
@tool_version("1.0.0")
async def dependencies(ctx: RunContext[AgentDeps], module: str) -> str:
    """List what a module/symbol depends on (outbound dependsOn edges)."""
    ci = _ci(ctx)
    if ci is None:
        return "Knowledge Graph not available."
    return _fmt(ci.dependencies(module), f"No dependencies found for '{module}'.")


CODE_INTELLIGENCE_TOOLS = [
    find_definition,
    who_calls,
    impacted_tests,
    call_graph,
    dependencies,
]
