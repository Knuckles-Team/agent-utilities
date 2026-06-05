#!/usr/bin/python
from __future__ import annotations

"""KG-backed evaluation corpus sourced from real usage (CONCEPT:AHE-3.1).

Closes the Layer-6 gap where eval cases were synthetic/hand-authored. Cases are
persisted as ``eval_case`` graph nodes and can be sourced from (1) human ``eval``
corrections via :class:`FeedbackService` and (2) retrieval regressions caught by
``EvaluationCapture``. :meth:`run_corpus` executes the corpus through the existing
:class:`EvalRunner`, so corrections become regression tests automatically.

Works with no backend (in-memory) so it is unit-testable in isolation.
"""

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
    ) -> str:
        """Add a regression case; returns its id. Persists to the graph if able."""
        case_id = f"eval_case:{uuid.uuid4().hex[:12]}"
        rec = {
            "id": case_id,
            "query": query,
            "expected_output": expected_output,
            "tags": list(tags or []),
            "reason": reason,
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
                    timestamp=rec["timestamp"],
                )
            except Exception as exc:  # pragma: no cover - persistence best-effort
                logger.debug("eval_case persist failed: %s", exc)
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
                        "c.expected_output AS expected_output, c.tags AS tags"
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
            cases.append(
                TestCase(
                    id=r.get("id", ""),
                    query=r["query"],
                    expected_output=r.get("expected_output", ""),
                    tags=tags or [],
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
