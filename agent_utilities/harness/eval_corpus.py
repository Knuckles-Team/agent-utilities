#!/usr/bin/python
"""KG-backed evaluation corpus sourced from real usage (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort).

Closes the Layer-6 gap where eval cases were synthetic/hand-authored. Cases are
persisted as ``eval_case`` graph nodes and can be sourced from (1) human ``eval``
corrections via :class:`FeedbackService` and (2) retrieval regressions caught by
``EvaluationCapture``. :meth:`run_corpus` executes the corpus through the existing
:class:`EvalRunner`, so corrections become regression tests automatically.

Works with no backend (in-memory) so it is unit-testable in isolation.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

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
        from agent_utilities.harness.optimization_backend import (
            opaque_program_reference,
        )

        case_id = f"eval_case:{uuid.uuid4().hex}"
        program_example_ref = opaque_program_reference("example", case_id)
        pending = {
            "query": query,
            "expected_output": expected_output,
            "tags": list(tags or []),
            "reason": reason,
            "metadata": dict(metadata or {}),
            "assertion": assertion,
        }
        sanitized, _privacy_report = PersistencePrivacyGuard().sanitize(pending)
        if not isinstance(sanitized, dict):  # defensive: the guard preserves mappings
            raise RuntimeError("eval case privacy normalization failed")
        case_metadata = sanitized.get("metadata")
        if not isinstance(case_metadata, dict):
            case_metadata = {}
        case_metadata["program_example_ref"] = program_example_ref
        rec = {
            "id": case_id,
            **sanitized,
            "metadata": case_metadata,
            "program_example_ref": program_example_ref,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._mem.append(rec)
        if self.backend is not None and hasattr(self.backend, "add_node"):
            try:
                self.backend.add_node(
                    case_id,
                    node_type="eval_case",
                    query=rec["query"],
                    expected_output=rec["expected_output"],
                    tags=",".join(rec["tags"]),
                    reason=rec["reason"],
                    metadata=json.dumps(rec["metadata"]) if rec["metadata"] else "",
                    program_example_ref=program_example_ref,
                    assertion=rec["assertion"],
                    timestamp=rec["timestamp"],
                )
            except Exception as exc:  # pragma: no cover - persistence best-effort
                logger.debug("eval_case persist failed (%s)", type(exc).__name__)
        return case_id

    def add_from_trace(
        self, trace: Any, *, assertion: str = "", tags: list[str] | None = None
    ) -> str:
        """Promote a production trace into a versioned regression case (CONCEPT:AU-AHE.evaluation.generationnode-records).

        The prod→dataset half of the closed loop: a ``TraceNode`` (or any object/dict with
        ``input``/``output``) becomes a ``DatasetItemNode(source=trace)`` AND an eval case
        whose expected output is the trace's output (or whose ``assertion`` is the check to
        re-run). ``source_trace_ref`` records non-reversible provenance without retaining
        the source system's trace identifier.
        """

        def _g(obj: Any, key: str, default: str = "") -> str:
            if isinstance(obj, dict):
                return str(obj.get(key, default))
            return str(getattr(obj, key, default) or default)

        from agent_utilities.harness.optimization_backend import (
            opaque_program_reference,
        )

        trace_id = _g(trace, "id")
        trace_ref = opaque_program_reference("trace", trace_id or "unidentified-trace")
        query = _g(trace, "input")
        output = _g(trace, "output")
        case_id = self.add_case(
            query=query or "governed trace execution",
            expected_output=output,
            tags=(tags or []) + ["from_trace"],
            reason="promoted from governed trace evidence",
            assertion=assertion,
            metadata={"source": "trace", "source_trace_ref": trace_ref},
        )
        # Mirror as a DatasetItemNode(source=trace) with provenance, when persistable.
        if self.backend is not None and hasattr(self.backend, "add_node"):
            try:
                dataset_item, _privacy_report = PersistencePrivacyGuard().sanitize(
                    {
                        "node_type": "dataset_item",
                        "source": "trace",
                        "input": query,
                        "expected": output,
                        "assertion": assertion,
                    }
                )
                if not isinstance(dataset_item, dict):
                    raise RuntimeError("dataset item privacy normalization failed")
                dataset_item["source_trace_ref"] = trace_ref
                self.backend.add_node(
                    f"dataset_item:{case_id}",
                    **dataset_item,
                )
            except Exception as exc:  # pragma: no cover - best-effort
                logger.debug("dataset_item persist failed (%s)", type(exc).__name__)
        return case_id

    def load_cases(self) -> list[Any]:
        """Load cases as harness ``TestCase`` objects (graph first, else memory)."""
        from .continuous_evaluation_engine import TestCase

        rows: list[dict[str, Any]] = []
        if self.backend is not None and hasattr(self.backend, "execute"):
            try:
                rows = (
                    self.backend.execute(
                        "MATCH (c) WHERE c.node_type = 'eval_case' "
                        "RETURN c.id AS id, c.query AS query, "
                        "c.expected_output AS expected_output, c.tags AS tags, "
                        "c.assertion AS assertion, c.metadata AS metadata, "
                        "c.program_example_ref AS program_example_ref"
                    )
                    or []
                )
            except Exception as exc:  # pragma: no cover - dialect tolerant
                logger.debug(
                    "load_cases query failed, using memory (%s)",
                    type(exc).__name__,
                )
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
            if not isinstance(metadata, dict):
                metadata = {}
            program_example_ref = str(r.get("program_example_ref") or "")
            if program_example_ref:
                metadata.setdefault("program_example_ref", program_example_ref)
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
                logger.debug(
                    "actual_output_fn failed for governed case (%s)",
                    type(exc).__name__,
                )
                continue
            results.append(runner.run_eval(case, actual))
        return results

    @property
    def size(self) -> int:
        return len(self._mem)
