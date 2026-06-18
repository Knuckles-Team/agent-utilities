#!/usr/bin/python
from __future__ import annotations

"""ARAService — the single dispatch point for ARA over MCP + REST (CONCEPT:KG-2.80).

One service, one source of truth: the MCP ``research_artifact`` tool and the gateway
``/api/research/*`` routes both call :meth:`ARAService.run`, so the surfaces never drift.
Actions:

- ``reason``   — run OWL/RDF reasoning over the ecosystem and harvest the extrapolated
                 cross-domain relationships as research topics (the keystone, KG-2.79);
- ``compile``  — legacy paper → OWL-native ecosystem-grounded ARA (KG-2.80, A4);
- ``review`` / ``seal`` — run the ARA Seal (L1/L2/L3) over an artifact and certify it (A5);
- ``capture``  — record a live research event with provenance and promote it (A3);
- ``get``      — read a compiled artifact + its claims back from the graph;
- ``list``     — list compiled research artifacts.

Dependencies (grounding/judge/embedder) are injectable so the service is unit-testable
without live LLM/engine; every action degrades to an error dict, never an exception that
escapes the tool boundary.

Concept: ara-service
"""

import logging
from collections.abc import Callable
from typing import Any

from .compiler import ARACompiler
from .live_manager import LiveResearchManager
from .reasoning_driver import OntologyReasoningDriver
from .seal import ARASeal

logger = logging.getLogger(__name__)

ACTIONS = (
    "reason",
    "compile",
    "review",
    "seal",
    "capture",
    "get",
    "list",
    "inquire",
)


class ARAService:
    """Dispatch ARA actions over one engine for both the MCP tool and REST."""

    def __init__(
        self,
        engine: Any,
        *,
        ground_fn: Any = None,
        judge_fn: Any = None,
        confidence_floor: float = 0.0,
    ) -> None:
        self._engine = engine
        self._ground_fn = ground_fn
        self._judge_fn = judge_fn
        self._confidence_floor = confidence_floor

    # -- dispatch --------------------------------------------------------- #
    def run(self, action: str, **kwargs: Any) -> dict[str, Any]:
        """Route ``action`` to its handler; unknown/erroring actions → error dict."""
        handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "reason": self._reason,
            "compile": self._compile,
            "review": self._review,
            "seal": self._review,  # alias
            "capture": self._capture,
            "get": self._get,
            "list": self._list,
            "inquire": self._inquire,
        }
        handler = handlers.get(action)
        if handler is None:
            return {"error": f"unknown action {action!r}; expected one of {ACTIONS}"}
        try:
            return handler(**kwargs)
        except TypeError as e:  # bad/missing kwargs for the action
            return {"error": f"{action}: {e}"}
        except Exception as e:  # noqa: BLE001 — never escape the tool boundary
            logger.debug("ARAService.%s failed: %s", action, e)
            return {"error": str(e)}

    # -- actions ---------------------------------------------------------- #
    def _inquire(
        self,
        *,
        topic: str = "",
        topic_id: str = "",
        materialize: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        """Run a native multi-perspective inquiry over ``topic`` (CONCEPT:KG-2.127).

        The on-demand twin of the loop's perspectival acquire: derives expert lenses,
        fans KG probes across their questions, and returns the contradiction map +
        self-critique (optionally materialized as KG nodes).
        """
        name = (topic or topic_id).strip()
        if not name:
            return {"error": "inquire needs a topic"}
        from ..perspective import PerspectiveEngine
        from ..search import acquire_for_topic

        tid = topic_id or f"topic:{name.lower().replace(' ', '-')[:80]}"
        engine = self._engine

        def _probe(question: str) -> list[str]:
            return acquire_for_topic(engine, {"id": tid, "name": question}, top_k=3)

        inquiry = PerspectiveEngine(engine).inquire({"id": tid, "name": name}, _probe)
        materialized = (
            PerspectiveEngine(engine).materialize(inquiry) if materialize else {}
        )
        cm = inquiry.contradiction_map
        pr = inquiry.peer_review
        return {
            "action": "inquire",
            "topic": name,
            "perspectives": [
                {"lens": p.lens, "sources": p.source_node_ids}
                for p in inquiry.perspectives
            ],
            "agreements": cm.agreements,
            "divergences": cm.divergences,
            "blind_spot": cm.blind_spot,
            "peer_review": {
                "dominant_lens": pr.dominant_lens,
                "missing_perspective": pr.missing_perspective,
                "weakest_link": pr.weakest_link,
                "frontier_question": pr.frontier_question,
                "confidence": pr.confidence,
            },
            "materialized": materialized,
        }

    def _reason(
        self, *, query: str = "", persist: bool = True, **_: Any
    ) -> dict[str, Any]:
        harvest = OntologyReasoningDriver(self._engine).extrapolate(persist=persist)
        return {
            "action": "reason",
            "query": query,
            "inferred_edges": harvest.inferred_edges[:50],
            "new_topics": harvest.new_topics,
            "stats": harvest.stats,
            "error": harvest.error,
        }

    def _compile(
        self,
        *,
        article_id: str = "",
        target_codebase: str | None = None,
        materialize: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        if not article_id:
            return {"error": "compile needs an article_id"}
        compiler = ARACompiler(self._engine, ground_fn=self._ground_fn)
        artifact, report = compiler.compile(
            article_id, target_codebase=target_codebase, materialize=materialize
        )
        return {
            "action": "compile",
            "artifact": artifact.model_dump(),
            "report": report.model_dump(),
        }

    def _review(
        self,
        *,
        article_id: str = "",
        level: str = "L1",
        materialize: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        if not article_id:
            return {"error": "review needs an article_id"}
        # recompile (no re-materialize) to obtain the artifact, then seal it
        artifact, _report = ARACompiler(
            self._engine, ground_fn=self._ground_fn
        ).compile(article_id, materialize=False)
        seal = ARASeal(
            self._engine,
            judge_fn=self._judge_fn,
            confidence_floor=self._confidence_floor,
        )
        report = seal.review(artifact, level=level, materialize=materialize)
        return {"action": "review", "report": report.model_dump()}

    def _capture(
        self,
        *,
        text: str = "",
        article_id: str = "",
        provenance: str = "ai_executed",
        actor: str = "",
        event_type: str = "",
        flush: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        if not text or not article_id:
            return {"error": "capture needs text and article_id"}
        lrm = LiveResearchManager(article_id)
        ev = lrm.capture(
            text,
            type=event_type or None,  # type: ignore[arg-type]
            provenance=provenance,  # type: ignore[arg-type]
            actor=actor,
        )
        flushed = lrm.flush(self._engine) if flush else 0
        return {"action": "capture", "event": ev.model_dump(), "flushed": flushed}

    def _get(self, *, article_id: str = "", **_: Any) -> dict[str, Any]:
        if not article_id:
            return {"error": "get needs an article_id"}
        node_id = f"research_artifact:{article_id}"
        node = self._read_node(node_id)
        if node is None:
            return {"error": f"artifact {node_id!r} not found"}
        return {"action": "get", "artifact": node, "claims": self._read_claims(node_id)}

    def _list(self, *, limit: int = 50, **_: Any) -> dict[str, Any]:
        rows = self._query(
            "MATCH (a:research_artifact) RETURN a.id AS id, a.name AS name LIMIT $limit",
            {"limit": int(limit)},
        )
        return {"action": "list", "artifacts": rows, "count": len(rows)}

    # -- graph helpers (best-effort, SUPPORTED shapes) -------------------- #
    def _query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict]:
        try:
            rows = self._engine.query_cypher(cypher, params or {})
        except Exception as e:  # noqa: BLE001
            logger.debug("ARAService query failed: %s", e)
            return []
        return [r for r in (rows or []) if isinstance(r, dict)]

    def _read_node(self, node_id: str) -> dict[str, Any] | None:
        rows = self._query(
            "MATCH (a:research_artifact) WHERE a.id = $id RETURN a.id AS id, "
            "a.name AS name, a.summary AS summary, a.source_url AS source_url",
            {"id": node_id},
        )
        return rows[0] if rows else None

    def _read_claims(self, node_id: str) -> list[dict[str, Any]]:
        return self._query(
            "MATCH (a:research_artifact)-[:CONTAINS]->(c:claim) WHERE a.id = $id "
            "RETURN c.id AS id, c.statement AS statement",
            {"id": node_id},
        )


__all__ = ["ARAService", "ACTIONS"]
