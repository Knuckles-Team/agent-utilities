"""CONCEPT:ECO-4.43 — Git issue/PR -> SWE task resolver.

OpenHands ships a per-platform resolver that turns a GitHub/GitLab/Bitbucket issue or PR into an
agent task, with a suggested-tasks taxonomy. We match it and surpass it the agent-utilities way:
the issue/PR is **ingested into the KG as a ``GitTask`` object**, so the suggested-tasks views
(merge-conflicts / failing-checks / unresolved-comments / open-issue / open-PR) are graph
queries, not bespoke per-platform code. The task is then enqueued onto the existing durable
dispatch queue (ORCH-1.45) as a reference, targeting the ``swe_engineer`` agent (ORCH-1.47).

Parsing (:func:`parse_webhook`) and classification (:func:`classify`) are pure and cover the
GitHub and GitLab webhook shapes; ingestion + enqueue are thin and injectable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# The suggested-tasks taxonomy (mirrors OpenHands' task kinds).
OPEN_ISSUE = "open_issue"
OPEN_PR = "open_pr"
FAILING_CHECKS = "failing_checks"
MERGE_CONFLICTS = "merge_conflicts"
UNRESOLVED_COMMENTS = "unresolved_comments"
SUGGESTED_TASK_KINDS = (
    OPEN_ISSUE,
    OPEN_PR,
    FAILING_CHECKS,
    MERGE_CONFLICTS,
    UNRESOLVED_COMMENTS,
)


@dataclass
class GitTaskRef:
    """A normalized engineering task derived from an issue/PR."""

    source: str  # "github" | "gitlab"
    repo: str  # "owner/name"
    repo_url: str
    number: int
    kind: str  # one of SUGGESTED_TASK_KINDS
    title: str
    body: str = ""
    base_ref: str = ""

    @property
    def task_id(self) -> str:
        return f"gittask:{self.source}:{self.repo}#{self.number}"

    @property
    def problem_statement(self) -> str:
        return f"{self.title}\n\n{self.body}".strip()


def classify(
    is_pr: bool,
    *,
    mergeable: Any = None,
    checks_failing: bool = False,
    unresolved_comments: bool = False,
) -> str:
    """Pick the suggested-task kind. Most specific signal wins (PR pathologies before open-PR)."""
    if not is_pr:
        return OPEN_ISSUE
    if checks_failing:
        return FAILING_CHECKS
    if mergeable is False:
        return MERGE_CONFLICTS
    if unresolved_comments:
        return UNRESOLVED_COMMENTS
    return OPEN_PR


def parse_webhook(
    payload: dict[str, Any], *, source: str | None = None
) -> GitTaskRef | None:
    """Parse a GitHub or GitLab webhook into a :class:`GitTaskRef` (returns None if unsupported)."""
    source = source or _detect_source(payload)
    try:
        if source == "github":
            return _parse_github(payload)
        if source == "gitlab":
            return _parse_gitlab(payload)
    except Exception as exc:  # noqa: BLE001 - malformed payloads are skipped, not fatal
        logger.warning("git webhook parse failed: %s", exc)
    return None


def _detect_source(payload: dict[str, Any]) -> str:
    if "object_kind" in payload or "object_attributes" in payload:
        return "gitlab"
    return "github"


def _parse_github(payload: dict[str, Any]) -> GitTaskRef | None:
    repo = payload.get("repository", {}) or {}
    repo_full = repo.get("full_name", "")
    repo_url = repo.get("clone_url") or repo.get("html_url", "")
    if "pull_request" in payload:
        pr = payload["pull_request"]
        kind = classify(
            True,
            mergeable=pr.get("mergeable"),
            unresolved_comments=bool(pr.get("review_comments")),
        )
        return GitTaskRef(
            "github",
            repo_full,
            repo_url,
            int(pr.get("number", 0)),
            kind,
            pr.get("title", ""),
            pr.get("body") or "",
            (pr.get("base", {}) or {}).get("ref", ""),
        )
    if "issue" in payload:
        issue = payload["issue"]
        return GitTaskRef(
            "github",
            repo_full,
            repo_url,
            int(issue.get("number", 0)),
            OPEN_ISSUE,
            issue.get("title", ""),
            issue.get("body") or "",
        )
    return None


def _parse_gitlab(payload: dict[str, Any]) -> GitTaskRef | None:
    attrs = payload.get("object_attributes", {}) or {}
    project = payload.get("project", {}) or {}
    repo_full = project.get("path_with_namespace", "")
    repo_url = project.get("git_http_url") or project.get("web_url", "")
    kind_field = payload.get("object_kind") or attrs.get("noteable_type", "")
    number = int(attrs.get("iid") or attrs.get("id") or 0)
    if kind_field in ("merge_request",):
        kind = classify(
            True,
            mergeable=None
            if attrs.get("merge_status") != "cannot_be_merged"
            else False,
        )
        return GitTaskRef(
            "gitlab",
            repo_full,
            repo_url,
            number,
            kind,
            attrs.get("title", ""),
            attrs.get("description") or "",
            attrs.get("target_branch", ""),
        )
    if kind_field in ("issue",):
        return GitTaskRef(
            "gitlab",
            repo_full,
            repo_url,
            number,
            OPEN_ISSUE,
            attrs.get("title", ""),
            attrs.get("description") or "",
        )
    return None


def ingest_task(engine: Any, ref: GitTaskRef) -> str | None:
    """Persist the task as a ``GitTask`` KG object linked to its repo. Returns the node id."""
    if engine is None:
        return None
    try:
        engine.add_node(
            ref.task_id,
            "GitTask",
            properties={
                "name": ref.title,
                "source": ref.source,
                "repo": ref.repo,
                "repo_url": ref.repo_url,
                "number": ref.number,
                "kind": ref.kind,
                "problem_statement": ref.problem_statement[:4000],
                "base_ref": ref.base_ref,
                "status": "open",
            },
        )
        engine.add_edge(ref.task_id, f"repo:{ref.repo}", "TARGETS_REPO")
    except Exception as exc:  # noqa: BLE001 - KG optional
        logger.debug("git task ingest failed: %s", exc)
        return None
    return ref.task_id


def suggested_tasks(
    engine: Any, *, repo: str | None = None, kind: str | None = None
) -> list[dict]:
    """Query open GitTasks as a graph query — the surpass over per-platform suggested-task code."""
    backend = getattr(engine, "backend", None)
    execute = getattr(backend, "execute", None)
    if not callable(execute):
        return []
    where = ["g.status = 'open'"]
    params: dict[str, Any] = {}
    if repo:
        where.append("g.repo = $repo")
        params["repo"] = repo
    if kind:
        where.append("g.kind = $kind")
        params["kind"] = kind
    cypher = (
        f"MATCH (g:GitTask) WHERE {' AND '.join(where)} "
        "RETURN g.id AS id, g.repo AS repo, g.kind AS kind, g.name AS name LIMIT 100"
    )
    try:
        return [r for r in (execute(cypher, params) or []) if isinstance(r, dict)]
    except Exception:  # noqa: BLE001
        return []


def enqueue_swe_task(
    ref: GitTaskRef, *, agent_name: str = "swe_engineer", queue: Any = None
) -> dict:
    """Enqueue the resolved task onto the durable dispatch queue (ORCH-1.45) as a reference."""
    from agent_utilities.orchestration.agent_dispatch import (
        KIND_ORCHESTRATOR_TASK,
        AgentTurnEnvelope,
        enqueue_agent_turn,
    )

    envelope = AgentTurnEnvelope(
        session_id=ref.task_id,
        kind=KIND_ORCHESTRATOR_TASK,
        payload_ref=ref.task_id,
        agent_name=agent_name,
    )
    return enqueue_agent_turn(envelope, queue=queue)


def resolve_and_dispatch(
    payload: dict[str, Any],
    engine: Any,
    *,
    source: str | None = None,
    queue: Any = None,
) -> dict[str, Any] | None:
    """Full pipeline: parse -> ingest as KG object -> enqueue an swe task. Returns the handle."""
    ref = parse_webhook(payload, source=source)
    if ref is None:
        return None
    ingest_task(engine, ref)
    handle = enqueue_swe_task(ref, queue=queue)
    return {"task_id": ref.task_id, "kind": ref.kind, "repo": ref.repo, **handle}
