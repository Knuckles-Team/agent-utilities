#!/usr/bin/python
from __future__ import annotations

"""``spark_jobs.py`` -- au's governed façade over spark-mcp's live Transform
tools (CA-27, CONCEPT:AU-KG.compute.spark-transform-jobs).

**Overlap finding, recorded for CA-29/CA-53 (measured 2026-08-26).** This
lane's brief (``plans/company-architecture/lanes/CA-27-trino-spark-adapters.md``)
was written expecting a ``spark-transform-submit``/``spark-transform-status``
gap: two ``MCP_TOOL_PRESETS`` entries this lane defines and CA-29 registers in
``mcp_tool.py`` at end-of-wave, with ``spark_jobs.py`` failing loudly (naming
the missing preset) until that registration lands. That gap no longer exists.
CA-42 shipped real, live FastMCP tools directly on ``agents/spark-mcp``
(``spark_mcp/mcp/mcp_spark.py``): ``spark_submit_transform``,
``spark_list_transform_runs``, ``spark_rerun_transform`` -- plain per-tool
Pydantic-typed arguments, not the fleet's ``action``/``params_json`` envelope
convention ``MCP_TOOL_PRESETS`` models. This session additionally confirmed
Spark Connect itself is now live (``services/spark``'s ``spark-connect``
ClusterIP Service, selector ``app: spark-runner``, backed by a Running
``spark-runner`` pod) -- CA-27's sibling-lane note that Spark Connect was
"unverified" is stale as of this session.

So: there is nothing left for CA-29 to register for this pair, and this
module calls the real tool names directly through the fleet's existing
write-side one-shot caller (``McpToolSourceConnector.call_tool_once`` --
already public API, no registration required for a one-shot call). What this
lane still adds, because spark-mcp's tools do not provide it:

1. **Idempotent submission.** ``spark_submit_transform``'s server
   implementation mints a fresh ``run_id`` (``uuid.uuid4()``) on every call,
   with no manifest-level dedup (``spark_mcp/api/api_client_spark.py``'s
   ``submit_transform``) -- a bare retry after a dropped response would
   silently double-append the output table. :meth:`SparkJobsClient.
   submit_transform` computes a deterministic ``submission_id`` from the
   manifest (or accepts a caller-supplied one) and short-circuits a repeat
   submission to the cached result instead of re-executing (CA-27-W05).
2. **``poll_status(run_id)``.** spark-mcp exposes only
   ``spark_list_transform_runs`` (filterable by transform NAME, not run_id).
   This module resolves one run_id's status by consulting its own submission
   ledger for the transform name, then paging that tool.
3. **The R5 fence.** A submitted Transform result is not a KG fact until it
   carries run id / input snapshot id(s) / code version / confidence --
   :meth:`SparkJobsClient.fence` builds the
   :class:`~agent_utilities.knowledge_graph.backends.trino_backend.
   ChangeEnvelopeBuilder` for a completed submission (never for one still
   running or failed -- see :meth:`fence`'s docstring).

Never a kubectl-exec fallback (R3 / this lane's failure invariant): every
call here goes through the MCP tool surface; a connection failure to
spark-mcp (or to spark-mcp's own Spark Connect backend) surfaces as a typed
:class:`SparkJobError` naming the endpoint, never an opaque kubectl exit code.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from ...models.company_brain import DataClassification
from ..ingestion.change_envelope import ChangeEnvelope
from .trino_backend import ChangeEnvelopeBuilder

logger = logging.getLogger(__name__)

__all__ = [
    "SPARK_MCP_SERVER",
    "SPARK_SUBMIT_TOOL",
    "SPARK_LIST_RUNS_TOOL",
    "SPARK_RERUN_TOOL",
    "SparkJobError",
    "SparkPresetUnavailableError",
    "SparkTransformSubmission",
    "SparkJobsClient",
]

#: The real, live spark-mcp server + tool names this module calls (CA-42's
#: shipped surface -- see the module docstring's overlap finding). Named as
#: constants, not a preset dict, because there is no ``MCP_TOOL_PRESETS``
#: registration gap left to model.
SPARK_MCP_SERVER = "spark-mcp"
SPARK_SUBMIT_TOOL = "spark_submit_transform"
SPARK_LIST_RUNS_TOOL = "spark_list_transform_runs"
SPARK_RERUN_TOOL = "spark_rerun_transform"

#: Poll backoff (module docstring / lane budget): 2s initial, exponential to
#: a 30s cap, 30 minute total budget before ``status: "timeout"``.
_POLL_INITIAL_INTERVAL = 2.0
_POLL_MAX_INTERVAL = 30.0
_POLL_BUDGET_SECONDS = 30 * 60

_TERMINAL_STATUSES = frozenset({"succeeded", "failed"})

#: Best-effort markers a fastmcp/MCP transport uses for "no such tool" across
#: server/protocol versions -- distinguishes a genuinely-missing tool surface
#: (fail loudly, name it) from a live-but-erroring call (fail loudly with the
#: real error). Not exhaustive by construction; see :meth:`SparkJobsClient._call`.
_UNKNOWN_TOOL_MARKERS = (
    "unknown tool",
    "tool not found",
    "no such tool",
    "method not found",
    "-32601",
)


class SparkJobError(RuntimeError):
    """A Spark Transform submit/poll failure -- names the endpoint/tool, never
    an opaque kubectl-exec-style exit code (this lane's failure invariant)."""


class SparkPresetUnavailableError(SparkJobError):
    """The target spark-mcp server does not expose the expected Transform
    tool surface (``spark_submit_transform`` / ``spark_list_transform_runs``).

    Named explicitly (acceptance gate 5's intent, restated for the real tool
    surface -- see the module docstring's overlap finding): a caller sees
    exactly which tool is missing and on which server, never a bare
    ``KeyError``/``AttributeError``.
    """


def submission_id_for(manifest: dict[str, Any]) -> str:
    """Deterministic idempotency key from a Transform manifest.

    The same ``(transform, kind, body, inputs, output)`` always yields the
    same id, so a retried submission of byte-identical work is recognized
    client-side even though ``spark_submit_transform`` mints a fresh
    ``run_id`` on every server-side call (module docstring, point 1).
    """
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class SparkTransformSubmission:
    """One submitted (or replayed-from-ledger) Transform result."""

    submission_id: str
    run_id: str
    transform: str
    status: str
    output_snapshot_id: str | None
    output_table: str
    row_count: int | None
    error: str | None
    input_snapshot_ids: tuple[str, ...]
    raw: dict[str, Any]


def _input_snapshot_ids(manifest: dict[str, Any]) -> tuple[str, ...]:
    ids = []
    for ref in manifest.get("inputs", []) or []:
        version = ref.get("as_of_version") if isinstance(ref, dict) else None
        if version:
            ids.append(str(version))
    return tuple(ids)


class SparkJobsClient:
    """au's governed client for spark-mcp's live Transform tool surface.

    Holds an in-process idempotency ledger (``submission_id`` -> result,
    ``run_id`` -> ``submission_id``) -- CA-27-W01 scope is a single-process
    unit-test/live-proof harness; a durable cross-process ledger is future
    work (out of this lane's W1 size), noted rather than silently assumed.
    """

    def __init__(
        self,
        *,
        server: str = SPARK_MCP_SERVER,
        mcp_client: Any = None,
        url: str = "",
        timeout: float = 60.0,
    ) -> None:
        self._server = server
        self._mcp_client = mcp_client
        self._url = url
        self._timeout = timeout
        self._ledger: dict[str, SparkTransformSubmission] = {}
        self._run_index: dict[str, str] = {}

    # -- transport --------------------------------------------------------

    async def _call(self, tool: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            from ...protocols.source_connectors.connectors.mcp_tool import (
                McpToolSourceError,
                call_tool_once,
            )
        except ImportError as exc:
            raise SparkJobError(
                "spark_jobs needs fastmcp (install agent-utilities[mcp])"
            ) from exc
        try:
            result = await call_tool_once(
                server=self._server,
                client=self._mcp_client,
                url=self._url,
                tool=tool,
                action="",
                params=params,
                params_style="args",
                timeout=self._timeout,
            )
        except McpToolSourceError as exc:
            cause = exc.__cause__
            combined = f"{exc} {cause}".lower() if cause else str(exc).lower()
            if any(marker in combined for marker in _UNKNOWN_TOOL_MARKERS):
                raise SparkPresetUnavailableError(
                    f"spark-mcp server {self._server!r} does not expose tool "
                    f"{tool!r} (expected CA-42's Transform tool surface: "
                    f"{SPARK_SUBMIT_TOOL}/{SPARK_LIST_RUNS_TOOL}) -- {exc}"
                ) from exc
            raise SparkJobError(
                f"spark-mcp call to {self._server}:{tool} failed: {exc}"
            ) from exc
        if not isinstance(result, dict):
            raise SparkJobError(
                f"spark-mcp {tool!r} returned a non-dict result: {type(result).__name__}"
            )
        return result

    def _call_sync(self, tool: str, params: dict[str, Any]) -> dict[str, Any]:
        return asyncio.run(self._call(tool, params))

    # -- submit / idempotency ----------------------------------------------

    def submit_transform(
        self, manifest: dict[str, Any], *, submission_id: str | None = None
    ) -> SparkTransformSubmission:
        """Submit a Transform manifest, or replay a matching prior submission.

        Retrying the SAME ``submission_id`` (explicit, or the manifest's own
        deterministic hash) is a no-op: the cached
        :class:`SparkTransformSubmission` is returned WITHOUT calling
        ``spark_submit_transform`` again (CA-27-W05's idempotency gate).
        """
        sub_id = submission_id or submission_id_for(manifest)
        cached = self._ledger.get(sub_id)
        if cached is not None:
            logger.info(
                "spark_jobs: submission %s already applied as run_id=%s; retry is a no-op",
                sub_id,
                cached.run_id,
            )
            return cached
        result = self._call_sync(SPARK_SUBMIT_TOOL, dict(manifest))
        submission = SparkTransformSubmission(
            submission_id=sub_id,
            run_id=str(result.get("run_id", "")),
            transform=str(result.get("transform", manifest.get("transform", ""))),
            status=str(result.get("status", "unknown")),
            output_snapshot_id=result.get("output_snapshot_id"),
            output_table=str(
                result.get("output_table")
                or (manifest.get("output") or {}).get("table", "")
            ),
            row_count=result.get("row_count"),
            error=result.get("error"),
            input_snapshot_ids=_input_snapshot_ids(manifest),
            raw=result,
        )
        self._ledger[sub_id] = submission
        if submission.run_id:
            self._run_index[submission.run_id] = sub_id
        return submission

    # -- poll ---------------------------------------------------------------

    def poll_status(
        self, run_id: str, *, transform: str | None = None
    ) -> dict[str, Any]:
        """Resolve ``run_id``'s current status.

        spark-mcp's ``spark_list_transform_runs`` filters by transform NAME
        only (not run_id) -- so this needs the transform name, taken from
        this client's local ledger when the run was submitted through it, or
        supplied explicitly (e.g. after a process restart with an empty
        ledger). Backs off 2s -> 30s cap, 30 minute total budget, returning
        ``{"status": "timeout", ...}`` rather than blocking forever.
        """
        sub_id = self._run_index.get(run_id)
        cached = self._ledger.get(sub_id) if sub_id else None
        if cached is not None and cached.status in _TERMINAL_STATUSES:
            return {
                "status": cached.status,
                "output_snapshot_id": cached.output_snapshot_id,
                "raw": cached.raw,
            }
        known_transform = transform or (cached.transform if cached else None)
        if not known_transform:
            raise SparkJobError(
                f"poll_status: run_id {run_id!r} is not in this client's local "
                "ledger and no 'transform' name was supplied -- spark-mcp's "
                "spark_list_transform_runs can only be filtered by transform "
                "name, not run_id; pass transform= explicitly."
            )
        deadline = time.monotonic() + _POLL_BUDGET_SECONDS
        interval = _POLL_INITIAL_INTERVAL
        while True:
            result = self._call_sync(
                SPARK_LIST_RUNS_TOOL, {"limit": 100, "transform": known_transform}
            )
            for run in result.get("runs", []):
                if run.get("run_id") == run_id:
                    status = str(run.get("status", "unknown"))
                    if status in _TERMINAL_STATUSES:
                        return {
                            "status": status,
                            "output_snapshot_id": run.get("output_snapshot_id"),
                            "raw": run,
                        }
                    break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {"status": "timeout", "output_snapshot_id": None, "raw": None}
            time.sleep(min(interval, remaining))
            interval = min(interval * 2, _POLL_MAX_INTERVAL)

    # -- R5 fence -------------------------------------------------------------

    def fence(
        self,
        submission: SparkTransformSubmission,
        *,
        connector: str = "spark-transform",
        code_version: str,
    ) -> ChangeEnvelopeBuilder:
        """Build the R5 fence for a COMPLETED, SUCCEEDED submission.

        Refuses (``SparkJobError``) for a still-running or failed submission
        -- a failed/in-flight Transform's payload is never a fact (Company
        Architecture invariant I3); a caller must not construct a
        ``ChangeEnvelope`` for one.
        """
        if submission.status != "succeeded":
            raise SparkJobError(
                f"cannot fence submission {submission.submission_id!r}: status="
                f"{submission.status!r} (only a succeeded run may become a fact)"
            )
        if not submission.output_snapshot_id:
            raise SparkJobError(
                f"cannot fence submission {submission.submission_id!r}: no "
                "output_snapshot_id was reported"
            )
        input_ids = submission.input_snapshot_ids or (submission.output_snapshot_id,)
        return ChangeEnvelopeBuilder(
            connector=connector,
            run_id=submission.run_id,
            input_snapshot_ids=input_ids,
            code_version=code_version,
            confidence=1.0,
        )

    def build_envelope(
        self,
        submission: SparkTransformSubmission,
        *,
        connector: str = "spark-transform",
        code_version: str,
        classification: DataClassification = DataClassification.INTERNAL,
    ) -> ChangeEnvelope:
        """Convenience: :meth:`fence` + ``.build()`` in one call, keyed by the
        output table as the KG ``source_object_id``."""
        builder = self.fence(submission, connector=connector, code_version=code_version)
        return builder.build(
            source_object_id=submission.output_table,
            payload={
                "run_id": submission.run_id,
                "output_table": submission.output_table,
                "output_snapshot_id": submission.output_snapshot_id,
                "row_count": submission.row_count,
            },
            classification=classification,
        )
