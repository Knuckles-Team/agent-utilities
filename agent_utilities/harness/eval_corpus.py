#!/usr/bin/python
from __future__ import annotations

"""KG-backed evaluation corpus sourced from real usage (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort).

Closes the Layer-6 gap where eval cases were synthetic/hand-authored. Cases are
persisted as ``eval_case`` graph nodes and can be sourced from (1) human ``eval``
corrections via :class:`FeedbackService` and (2) retrieval regressions caught by
``EvaluationCapture``. :meth:`run_corpus` executes the corpus through the existing
:class:`EvalRunner`, so corrections become regression tests automatically.

Works with no backend (in-memory) so it is unit-testable in isolation.
"""

import json
import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class EvalCorpus:
    """Persist, load, and run regression eval cases."""

    def __init__(self, backend: Any = None) -> None:
        self.backend = backend
        self._mem: list[dict[str, Any]] = []

    def add_case(
        self,
        query: str,
        expected_output: str,
        *,
        tags: list[str] | None = None,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
        assertion: str = "",
    ) -> str:
        """Add a regression case; returns its id. Persists to the graph if able.

        ``metadata`` carries arbitrary per-case context (e.g. the evidence,
        gold topics, or retrieved ids a context-aware scorer needs); it is
        surfaced on the loaded :class:`TestCase.metadata`. ``assertion`` is an
        optional plain-English pass/fail check (CONCEPT:AU-AHE.evaluation.failure-analysis-loop, Opik Test Suite
        style) judged by LLM-as-judge in lieu of expected-output scoring.
        """
        case_id = f"eval_case:{uuid.uuid4().hex[:12]}"
        rec = {
            "id": case_id,
            "query": query,
            "expected_output": expected_output,
            "tags": list(tags or []),
            "reason": reason,
            "metadata": dict(metadata or {}),
            "assertion": assertion,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._mem.append(rec)
        if self.backend is not None and hasattr(self.backend, "add_node"):
            try:
                self.backend.add_node(
                    case_id,
                    type="eval_case",
                    query=query,
                    expected_output=expected_output,
                    tags=",".join(rec["tags"]),
                    reason=reason,
                    metadata=json.dumps(rec["metadata"]) if rec["metadata"] else "",
                    assertion=assertion,
                    timestamp=rec["timestamp"],
                )
            except Exception as exc:  # pragma: no cover - persistence best-effort
                logger.debug("eval_case persist failed: %s", exc)
        return case_id

    def add_from_trace(
        self, trace: Any, *, assertion: str = "", tags: list[str] | None = None
    ) -> str:
        """Promote a production trace into a versioned regression case (CONCEPT:AU-AHE.evaluation.generationnode-records).

        The prod→dataset half of the closed loop: a ``TraceNode`` (or any object/dict with
        ``input``/``output``) becomes a ``DatasetItemNode(source=trace)`` AND an eval case
        whose expected output is the trace's output (or whose ``assertion`` is the check to
        re-run). ``source_trace_id`` records provenance so the case traces back to its trace.
        """

        def _g(obj: Any, key: str, default: str = "") -> str:
            if isinstance(obj, dict):
                return str(obj.get(key, default))
            return str(getattr(obj, key, default) or default)

        trace_id = _g(trace, "id")
        query = _g(trace, "input") or _g(trace, "name")
        output = _g(trace, "output")
        case_id = self.add_case(
            query=query or trace_id,
            expected_output=output,
            tags=(tags or []) + ["from_trace"],
            reason=f"promoted from trace {trace_id}",
            assertion=assertion,
            metadata={"source": "trace", "source_trace_id": trace_id},
        )
        # Mirror as a DatasetItemNode(source=trace) with provenance, when persistable.
        if self.backend is not None and hasattr(self.backend, "add_node"):
            try:
                self.backend.add_node(
                    f"dataset_item:{case_id}",
                    type="dataset_item",
                    source="trace",
                    input=query,
                    expected=output,
                    assertion=assertion,
                    source_trace_id=trace_id,
                )
            except Exception as exc:  # pragma: no cover - best-effort
                logger.debug("dataset_item persist failed: %s", exc)
        return case_id

    def load_cases(self) -> list[Any]:
        """Load cases as harness ``TestCase`` objects (graph first, else memory)."""
        from .continuous_evaluation_engine import TestCase

        rows: list[dict[str, Any]] = []
        if self.backend is not None and hasattr(self.backend, "execute"):
            try:
                rows = (
                    self.backend.execute(
                        "MATCH (c) WHERE c.type = 'eval_case' "
                        "RETURN c.id AS id, c.query AS query, "
                        "c.expected_output AS expected_output, c.tags AS tags, "
                        "c.assertion AS assertion"
                    )
                    or []
                )
            except Exception as exc:  # pragma: no cover - dialect tolerant
                logger.debug("load_cases query failed, using memory: %s", exc)
        if not rows:
            rows = self._mem
        cases: list[Any] = []
        for r in rows:
            if not r.get("query"):
                continue
            tags = r.get("tags")
            if isinstance(tags, str):
                tags = [t for t in tags.split(",") if t]
            metadata = r.get("metadata")
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata) if metadata else {}
                except json.JSONDecodeError:  # pragma: no cover - tolerant
                    metadata = {}
            cases.append(
                TestCase(
                    id=r.get("id", ""),
                    query=r["query"],
                    expected_output=r.get("expected_output", ""),
                    tags=tags or [],
                    metadata=metadata or {},
                    assertion=r.get("assertion", "") or "",
                )
            )
        return cases

    def run_corpus(
        self,
        actual_output_fn: Callable[[Any], str],
        runner: Any = None,
    ) -> list[Any]:
        """Run every case through ``EvalRunner``; returns the list of EvalResult."""
        from .continuous_evaluation_engine import EvalRunner

        runner = runner or EvalRunner()
        results = []
        for case in self.load_cases():
            try:
                actual = actual_output_fn(case)
            except Exception as exc:  # pragma: no cover - per-case isolation
                logger.debug("actual_output_fn failed for %s: %s", case.id, exc)
                continue
            results.append(runner.run_eval(case, actual))
        return results

    @property
    def size(self) -> int:
        return len(self._mem)
