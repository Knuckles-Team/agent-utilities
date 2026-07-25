#!/usr/bin/python
from __future__ import annotations

"""Code-correctness / security-audit gap track — the 4th discovery track.

CONCEPT:AU-AHE.harness.audit-gap-detector — Wave-6 D1-ext (the "Macroscope-level" review,
reframed as engine-native). An AI review runs over the **already-ingested code KG** — it
reads ``:CodeUnit``/``:Symbol`` nodes and their source straight from the graph (all
codebases + specs + git-history are already ingested, so there is no fresh scan; the
detector "naturally evolves within the epistemic-graph") — and detects the Macroscope
finding-classes:

* **resource-lifecycle / transaction-durability** — an unclosed resource or an uncommitted
  write that is silently lost.
* **id-consistency** — a run/id mismatch that breaks a later join.
* **serialization-validity** — truncated/invalid JSON or a lossy serialization.
* **secret-redaction** — a separator-normalization or redaction gap that leaks a secret.
* **error-handling** — a swallowed exception / bare ``except`` that hides a failure.
* **audit-integrity** — an audit/hook/probe that ALTERS the run it is supposed to observe.

Each finding ``submit_gap``s ONE canonical ``:Gap`` (source ``audit``, severity → priority)
that flows the SAME ``Gap → SDD → code-synth → W2.7 → resolved`` lifecycle as every other
track — so a blocked/flagged issue becomes governed, spec'd, tracked work rather than
block-and-forget. This is the honest fix for what coverage+mutation testing miss: a
semantic reviewer catches that class, and the KG makes the fix governed.

**Model** (user's choice): the local vLLM via the model factory (``role="reviewer"``);
large/high-risk units escalate to the configured escalation model (``role="critic"``). Both
are injectable ``(prompt) -> str`` callables for tests. **Opt-in** (``KG_LOOP_AUDIT``):
detection is a loop-driven stage that only runs when the operator turns it on — the
flywheel proposes, humans veto (no default-on autonomy). The optional pre-commit
front-end that runs the same review on a staged diff is a follow-up; this is the
engine-native detector.
"""

import json
import logging
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from ..knowledge_graph.research.gaps import SOURCE_AUDIT, submit_gap

logger = logging.getLogger(__name__)

LLMFn = Callable[[str], str]

#: The Macroscope finding-classes this detector reasons about.
FINDING_CLASSES = (
    "resource-lifecycle",
    "transaction-durability",
    "id-consistency",
    "serialization-validity",
    "secret-redaction",
    "error-handling",
    "audit-integrity",
)

#: LLM severity word → 0..1 (fed to ``severity_to_bucket`` so High/Critical expedite).
_SEVERITY = {"low": 0.3, "medium": 0.55, "high": 0.8, "critical": 0.95}

#: Property names an ingested code node may carry its source under.
_SOURCE_FIELDS = ("source", "content", "code", "body", "snippet", "text")

_REVIEW_PROMPT = """You are a meticulous code auditor reviewing ONE unit of an
already-ingested codebase for correctness and security defects that tests with high
coverage still miss. Only report REAL defects you can point to in the code — never
style, naming, or speculative issues.

Consider ONLY these finding classes:
- resource-lifecycle: an opened resource / DB transaction that is not closed or committed,
  so a write is silently lost.
- transaction-durability: an uncommitted or non-durable write path.
- id-consistency: an id/run key that will not match a later join.
- serialization-validity: truncated or invalid JSON / a lossy serialization.
- secret-redaction: a redaction / separator-normalization gap that can leak a secret.
- error-handling: a swallowed exception or bare except that hides a failure.
- audit-integrity: an audit / hook / probe that MUTATES the run it observes.

FILE: {file_path}
SYMBOL: {symbol}
SOURCE:
{source}

Output ONLY a JSON array (possibly empty) of objects with keys "finding_class" (one of the
classes above), "severity" ("low"|"medium"|"high"|"critical"), and "statement" (one concrete
sentence naming the defect and where). No prose outside the JSON."""


class AuditFinding(BaseModel):
    """One defect the reviewer found in a code unit."""

    finding_class: str
    severity: str = "medium"
    statement: str
    symbol_id: str = ""
    file_path: str = ""


def _parse_findings(raw: str) -> list[AuditFinding]:
    """Parse the reviewer's JSON array into validated findings (best-effort)."""
    try:
        start, end = raw.index("["), raw.rindex("]") + 1
        items = json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return []
    out: list[AuditFinding] = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        fclass = str(it.get("finding_class", "")).strip()
        statement = str(it.get("statement", "")).strip()
        if fclass not in FINDING_CLASSES or not statement:
            continue  # never invent a class or an empty finding
        sev = str(it.get("severity", "medium")).strip().lower()
        out.append(
            AuditFinding(
                finding_class=fclass,
                severity=sev if sev in _SEVERITY else "medium",
                statement=statement,
            )
        )
    return out


def _agent_llm_fn(role: str) -> LLMFn:
    """A ``(prompt) -> str`` backed by a model-factory role, "" when unreachable.

    Bounded by a hard wall-clock timeout so an unreachable/slow endpoint never hangs the
    audit stage (a blocking ``run_sync`` cannot be caught by try/except — run it on a
    worker thread), mirroring ``assimilation.plan_synthesis._llm_synth``. Built through the
    single R0 composition seam ``create_context_agent``.
    """

    def _fn(prompt: str) -> str:
        import concurrent.futures

        try:
            from agent_utilities.core.config import setting
            from agent_utilities.core.contextual_model import create_context_agent
            from agent_utilities.core.model_factory import create_model

            model = create_model(role=role)
            agent = create_context_agent(
                model=model, system_prompt="You are a precise code-defect auditor."
            )
            try:
                timeout_s = float(setting("AUDIT_REVIEW_TIMEOUT_S", "45"))
            except ValueError:
                timeout_s = 45.0
            ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                result: Any = ex.submit(agent.run_sync, prompt).result(
                    timeout=timeout_s
                )
            finally:
                ex.shutdown(wait=False)  # don't block on a wedged inference thread
            return str(
                getattr(result, "output", None) or getattr(result, "data", "") or ""
            )
        except Exception as exc:  # noqa: BLE001 — no model ⇒ no findings (graceful)
            logger.debug("audit review model unavailable: %s", type(exc).__name__)
            return ""

    return _fn


class AuditGapDetector:
    """Review ingested code units and file each finding as a canonical ``:Gap``.

    ``review_fn`` audits an ordinary unit (local vLLM); ``escalate_fn`` audits a large or
    high-risk unit (the configured escalation model). Both default to the model factory
    but are injectable for tests. Reads the code KG only — no fresh filesystem scan.
    """

    def __init__(
        self,
        engine: Any,
        *,
        review_fn: LLMFn | None = None,
        escalate_fn: LLMFn | None = None,
        escalate_chars: int = 6000,
    ) -> None:
        self.engine = engine
        self._review_fn = review_fn
        self._escalate_fn = escalate_fn
        self.escalate_chars = max(1, int(escalate_chars))

    def _review(self) -> LLMFn:
        if self._review_fn is None:
            self._review_fn = _agent_llm_fn("reviewer")
        return self._review_fn

    def _escalate(self) -> LLMFn:
        if self._escalate_fn is None:
            self._escalate_fn = _agent_llm_fn("critic")
        return self._escalate_fn

    def _read_code_units(self, limit: int) -> list[dict[str, Any]]:
        """Read ``:CodeUnit``/``:Symbol`` nodes (with source) from the code KG."""
        if self.engine is None:
            return []
        units: list[dict[str, Any]] = []
        seen: set[str] = set()
        for label in ("CodeUnit", "Symbol"):
            try:
                rows = self.engine.query_cypher(
                    f"MATCH (n:{label}) RETURN n LIMIT {int(limit)}"
                )
            except Exception as exc:  # noqa: BLE001 — one label failing never blocks
                logger.debug("audit read %s failed: %s", label, type(exc).__name__)
                continue
            for r in rows or []:
                props = r.get("n") if isinstance(r, dict) else None
                if not isinstance(props, dict):
                    continue
                nid = str(props.get("id") or "")
                if not nid or nid in seen:
                    continue
                source = next(
                    (str(props[f]) for f in _SOURCE_FIELDS if props.get(f)), ""
                )
                if not source.strip():
                    continue  # nothing to review
                seen.add(nid)
                units.append(
                    {
                        "id": nid,
                        "name": str(props.get("name") or nid),
                        "file_path": str(
                            props.get("file_path") or props.get("path") or ""
                        ),
                        "source": source,
                    }
                )
                if len(units) >= limit:
                    return units
        return units

    def _review_unit(self, unit: dict[str, Any]) -> list[AuditFinding]:
        prompt = _REVIEW_PROMPT.format(
            file_path=unit.get("file_path") or "?",
            symbol=unit.get("name") or unit.get("id") or "?",
            source=unit["source"],
        )
        # Escalate a large (or otherwise high-risk) unit to the stronger model.
        fn = (
            self._escalate()
            if len(unit["source"]) > self.escalate_chars
            else self._review()
        )
        findings = _parse_findings(fn(prompt) or "")
        for f in findings:
            f.symbol_id = unit["id"]
            f.file_path = unit.get("file_path", "")
        return findings

    def detect(self, *, limit: int = 25) -> list[dict[str, Any]]:
        """Review up to ``limit`` code units; file one canonical ``:Gap`` per finding.

        Returns the submitted gap dicts (severity → priority bucket, so High/Critical
        findings are expedited). Each gap flows the SAME lifecycle as every other track.
        """
        gaps: list[dict[str, Any]] = []
        for unit in self._read_code_units(limit):
            for finding in self._review_unit(unit):
                gap = submit_gap(
                    self.engine,
                    source=SOURCE_AUDIT,
                    signature=f"{finding.finding_class}:{unit['id']}",
                    statement=finding.statement,
                    domain=finding.finding_class,
                    severity=_SEVERITY.get(finding.severity, 0.55),
                    concept_ids=[unit["id"]],
                    evidence_refs=[unit.get("file_path", "")]
                    if unit.get("file_path")
                    else [],
                )
                if gap:
                    gaps.append(gap)
        if gaps:
            logger.info("[Wave6] audit detector filed %d canonical gap(s)", len(gaps))
        return gaps


def run_audit_gap_scan(engine: Any, *, limit: int = 25) -> dict[str, Any]:
    """Opt-in loop entry: run the audit detector when ``KG_LOOP_AUDIT`` is on.

    The engine-native tick (mirrors ``failure_analyzer.run_failure_ingest``): gated so a
    non-opted-in deployment is unaffected. When enabled, files a canonical ``:Gap`` per
    finding — the flywheel proposes; the specs it produces still sit behind the
    ``spec_promotion`` veto.
    """
    from agent_utilities.core.config import config

    if not getattr(config, "kg_loop_audit", False):
        return {"skipped": True, "reason": "KG_LOOP_AUDIT off"}
    gaps = AuditGapDetector(engine).detect(limit=limit)
    return {"gaps_filed": len(gaps), "gap_ids": [g["id"] for g in gaps]}


__all__ = [
    "FINDING_CLASSES",
    "AuditFinding",
    "AuditGapDetector",
    "run_audit_gap_scan",
]
