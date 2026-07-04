#!/usr/bin/python
from __future__ import annotations

"""Research cohort — one-command batch ingest + barrier synthesis (CONCEPT:AU-KG.coordination.research-cohort-barrier).

A *cohort* is the unit the evolution pipeline actually works in: "ingest THESE N
papers + M codebases in one go, then produce the comparative matrix." Without it,
a caller has to fan tasks out by hand and has no signal for *when the batch is
done* so the assimilation/matrix synthesis can run.

:func:`create_cohort` fans every member out as an ordinary durable task — papers
as ``content_url`` ingests, repos as ``codebase`` ingests — each tagged with the
``cohort_id`` so progress is a read-time COUNT (no per-cohort counters to race).
It then submits ONE ``cohort_synthesize`` gate task.

The gate is a **self-polling barrier**, deliberately NOT a ``depends_on`` join: a
``depends_on`` barrier is *cancelled* the moment any single member fails
(``_deps_state`` → ``broken``), so one poison paper would wedge the whole cohort.
Instead the gate re-defers itself (``scheduled``) each poll until every member is
**terminal** — completed OR failed — or a deadline passes, then runs the
assimilation pass (which materializes the feature matrix, KG-2.173) over whatever
was ingested. Pipeline parallelism comes for free: members drain concurrently
across the worker pool / functional lanes while the gate waits.

Concept: research-cohort
"""

import json
import logging
import re
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from ...models.knowledge_graph import RegistryNodeType

logger = logging.getLogger(__name__)

#: a cohort finalizes after this even if some members never reach a terminal state.
DEFAULT_MAX_WAIT_S = 3600.0
#: how long the gate sleeps (as ``scheduled``) between readiness checks.
POLL_INTERVAL_S = 60.0
SYNTHESIZE_TASK_TYPE = "cohort_synthesize"

_DONE = {"completed", "done", "success"}
_FAILED = {"failed", "dead_letter", "cancelled", "error"}
_TERMINAL = _DONE | _FAILED


def _decode(meta: Any) -> dict[str, Any]:
    from ..core.engine_tasks import _decode_metadata

    return _decode_metadata(meta) or {}


def _arxiv_id(ref: str) -> str:
    """Bare arXiv id from a URL / ``arxiv:ID`` / bare id (else the ref unchanged)."""
    s = str(ref).strip().rstrip("/")
    m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?$", s)
    return m.group(1) if m else s


def _paper_pdf_path(pid: str) -> str:
    """Path to a pre-downloaded paper PDF in the research store, if it exists."""
    try:
        from ...core import paths

        p = paths.research_dir() / "papers" / f"{pid}.pdf"
        return str(p) if p.exists() else ""
    except Exception:  # noqa: BLE001 — best-effort; empty → handler downloads
        return ""


def create_cohort(
    engine: Any,
    *,
    papers: list[str] | None = None,
    repos: list[str] | None = None,
    goal: str = "",
    max_wait_s: float = DEFAULT_MAX_WAIT_S,
) -> dict[str, Any]:
    """Fan a batch of papers + repos out as cohort-tagged tasks + a synthesize gate.

    ``papers`` are arXiv ids / URLs ingested via the ``research_paper_fetch`` lane —
    so each becomes an **Article** node (the matrix feature type), full-text from a
    pre-downloaded PDF when present (CONCEPT:AU-KG.research.so-cohort), and its ``article_id`` is
    recorded on the task as durable cohort provenance (CONCEPT:AU-KG.research.provenance). ``repos``
    are local paths / git URLs ingested via the ``codebase`` lane. Returns the
    ``cohort_id`` and the submitted job ids.
    """
    papers = [p for p in (papers or []) if p]
    repos = [r for r in (repos or []) if r]
    cohort_id = f"cohort-{uuid.uuid4().hex[:8]}"
    now = time.time()
    deadline = now + float(max_wait_s)

    engine.add_node(
        cohort_id,
        node_type=RegistryNodeType.RESEARCH_COHORT.value,
        properties={
            "goal": goal,
            "status": "ingesting",
            "member_count": len(papers) + len(repos),
            "papers": len(papers),
            "repos": len(repos),
            "created_at": datetime.now(UTC).isoformat(),
            "deadline_unix": deadline,
            "concept": "AU-KG.coordination.research-cohort-barrier",
        },
    )

    members: list[str] = []
    for i, ref in enumerate(papers):
        pid = _arxiv_id(ref)
        url = ref if str(ref).startswith("http") else f"https://arxiv.org/abs/{pid}"
        # research_paper_fetch → an Article node via the research pipeline, using the
        # pre-downloaded PDF for full text when present; the handler records the
        # resulting article_id on the task (cohort provenance, KG-2.192).
        members.append(
            engine.submit_task(
                pid,
                False,
                {},
                task_type="research_paper_fetch",
                extra_meta={
                    "cohort_id": cohort_id,
                    "paper": {
                        "id": pid,
                        "url": url,
                        "pdf_path": _paper_pdf_path(pid),
                        "score": 1.0,
                    },
                },
                job_id=f"{cohort_id}:p{i}",
                skip_dedupe=True,
            )
        )
    for i, repo in enumerate(repos):
        members.append(
            engine.submit_task(
                repo,
                True,
                {},
                task_type="codebase",
                extra_meta={"cohort_id": cohort_id},
                job_id=f"{cohort_id}:r{i}",
                skip_dedupe=True,
            )
        )

    synth = engine.submit_task(
        f"cohort:{cohort_id}",
        False,
        {},
        task_type=SYNTHESIZE_TASK_TYPE,
        extra_meta={"cohort_id": cohort_id, "deadline_unix": deadline},
        job_id=f"{cohort_id}:synth",
        skip_dedupe=True,
    )
    logger.info(
        "cohort %s: %d papers + %d repos fanned out → gate %s",
        cohort_id,
        len(papers),
        len(repos),
        synth,
    )
    return {
        "cohort_id": cohort_id,
        "members": members,
        "synthesize_job": synth,
        "papers": len(papers),
        "repos": len(repos),
    }


def cohort_member_status(engine: Any, cohort_id: str) -> dict[str, int]:
    """Read-time COUNT of a cohort's member tasks by status (the gate itself excluded)."""
    counts = {
        "total": 0,
        "pending": 0,
        "running": 0,
        "scheduled": 0,
        "blocked": 0,
        "completed": 0,
        "failed": 0,
        "terminal": 0,
    }
    try:
        rows = engine._control_cypher(
            "MATCH (t:Task) RETURN t.status as s, t.metadata as meta"
        )
    except Exception:  # noqa: BLE001 — status read is best-effort
        return counts
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        meta = _decode(row.get("meta"))
        if (
            meta.get("cohort_id") != cohort_id
            or meta.get("type") == SYNTHESIZE_TASK_TYPE
        ):
            continue
        counts["total"] += 1
        s = str(row.get("s") or "").lower()
        if s in _DONE:
            counts["completed"] += 1
        elif s in _FAILED:
            counts["failed"] += 1
        elif s == "running":
            counts["running"] += 1
        elif s == "scheduled":
            counts["scheduled"] += 1
        elif s == "blocked":
            counts["blocked"] += 1
        else:
            counts["pending"] += 1
        if s in _TERMINAL:
            counts["terminal"] += 1
    return counts


def cohort_source_ids(engine: Any, cohort_id: str) -> set[str]:
    """Graph node ids this cohort's members produced (CONCEPT:AU-KG.research.provenance provenance).

    Each member task records the node it created (``research_paper_fetch`` stamps
    ``article_id``), so the cohort's source set is recovered from durable task
    provenance WITHOUT scanning the graph — this is exactly what scopes the matrix
    to the cohort (CONCEPT:AU-KG.ingest.fetch-only-requested-ids) instead of the whole 15k-feature graph.
    """
    ids: set[str] = set()
    try:
        rows = engine._control_cypher(
            "MATCH (t:Task) RETURN t.status as s, t.metadata as meta"
        )
    except Exception:  # noqa: BLE001 — provenance read is best-effort
        return ids
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        meta = _decode(row.get("meta"))
        if meta.get("cohort_id") != cohort_id:
            continue
        for key in ("article_id", "node_id", "source_id"):
            val = meta.get(key)
            if val:
                ids.add(str(val))
    return ids


def cohort_ready(
    engine: Any, cohort_id: str, *, deadline_unix: float = 0.0
) -> tuple[bool, dict[str, int]]:
    """``(ready, member_counts)`` — ready once every member is terminal, the deadline
    has passed, or the cohort has no members (trivially done)."""
    st = cohort_member_status(engine, cohort_id)
    empty = st["total"] == 0
    all_terminal = st["total"] > 0 and st["terminal"] >= st["total"]
    past_deadline = deadline_unix > 0 and time.time() > deadline_unix
    return (empty or all_terminal or past_deadline, st)


def _read_cohort_node(engine: Any, cohort_id: str) -> dict[str, Any]:
    graph = getattr(engine, "graph", None)
    if graph is None:
        return {}
    try:
        for nid, data in graph.nodes(data=True):
            if nid == cohort_id and isinstance(data, dict):
                return data
    except TypeError:  # pragma: no cover - non-standard graph
        return {}
    return {}


def finalize_cohort(engine: Any, cohort_id: str) -> dict[str, Any]:
    """Run the assimilation pass SCOPED to the cohort's sources + materialize its
    feature matrix, then mark the cohort ``synthesized``.

    Scoping (CONCEPT:AU-KG.ingest.fetch-only-requested-ids) makes this O(cohort) not O(graph): only the cohort's
    Article ids (recovered from task provenance, KG-2.192) are matched/ranked, and
    the matrix is materialized to the cohort's own node ``feature_matrix:<cohort>``
    so it never clobbers the ecosystem-wide matrix.
    """
    from .loop_controller import run_assimilation_pass

    restrict = cohort_source_ids(engine, cohort_id)
    rep = run_assimilation_pass(
        engine,
        force=True,
        restrict_to=restrict,
        matrix_node_id=f"feature_matrix:{cohort_id}",
    )
    matrix = rep.get("feature_matrix") or {}
    st = cohort_member_status(engine, cohort_id)
    try:
        engine.add_node(
            cohort_id,
            node_type=RegistryNodeType.RESEARCH_COHORT.value,
            properties={
                "status": "synthesized",
                "synthesized_at": datetime.now(UTC).isoformat(),
                "member_status": json.dumps(st),
                "feature_matrix": json.dumps(matrix.get("counts", {})),
                "matrix_node": str(matrix.get("node_id", "")),
            },
        )
    except Exception as e:  # noqa: BLE001 — node update is best-effort
        logger.debug("cohort %s finalize node update failed: %s", cohort_id, e)
    return {
        "cohort_id": cohort_id,
        "members": st,
        "assimilate": {
            k: rep.get(k)
            for k in ("auto_satisfied", "related", "open_gaps", "synergy_bundles")
        },
        "feature_matrix": matrix,
    }


def cohort_status(engine: Any, cohort_id: str) -> dict[str, Any]:
    """The unified cohort progress view (members + node state) for the MCP/REST surface."""
    node = _read_cohort_node(engine, cohort_id)
    st = cohort_member_status(engine, cohort_id)
    fm = node.get("feature_matrix")
    if isinstance(fm, str):
        try:
            fm = json.loads(fm)
        except (json.JSONDecodeError, TypeError):
            fm = {}
    return {
        "cohort_id": cohort_id,
        "status": str(node.get("status", "unknown")),
        "goal": str(node.get("goal", "")),
        "member_count": int(node.get("member_count", st["total"]) or 0),
        "members": st,
        "feature_matrix": fm or {},
    }


__all__ = [
    "create_cohort",
    "cohort_member_status",
    "cohort_ready",
    "finalize_cohort",
    "cohort_status",
    "SYNTHESIZE_TASK_TYPE",
    "DEFAULT_MAX_WAIT_S",
    "POLL_INTERVAL_S",
]
