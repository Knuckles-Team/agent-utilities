"""Issue-tracker write-back sinks — GitLab / GitHub / Plane (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

Closes the inference→action loop: KG-derived findings (TRM end-of-life/vuln risk,
tech-debt, failing tests) become *filed tickets*. Fail-closed
(``GITLAB_ENABLE_WRITE`` / ``GITHUB_ENABLE_WRITE`` / ``PLANE_ENABLE_WRITE``),
dry-run-first. ``creations`` items: ``{name/title, body, project_id|owner+repo}``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from ..core import WritebackContext, WritebackResult, register_sink

logger = logging.getLogger(__name__)


def _resolve_client(ops: dict[str, Any], module: str) -> Any | None:
    client = ops.get("client")
    if client is not None:
        return client
    try:
        mod = __import__(f"{module}.auth", fromlist=["get_client"])
        return mod.get_client()
    except Exception:  # noqa: BLE001
        logger.debug("%s write client unavailable", module, exc_info=True)
        return None


def _title(c: dict[str, Any]) -> str | None:
    return c.get("title") or c.get("name")


class _IssueSinkBase(ABC):
    domain = ""
    enable_flag = ""
    module = ""

    def _client(self, ops: dict[str, Any]) -> Any | None:
        return _resolve_client(ops, self.module)

    @abstractmethod
    def _create(self, client: Any, title: str, body: str, c: dict[str, Any]) -> None:
        """File one issue/ticket on the tracker via the resolved client."""

    def run(
        self, ctx: WritebackContext, ops: dict[str, Any], *, dry_run: bool
    ) -> WritebackResult:
        result = WritebackResult(target=self.domain)
        client = self._client(ops)
        if client is None and not dry_run:
            result.skipped += 1
            return result
        for c in ops.get("creations") or []:
            title = _title(c)
            if not title:
                continue
            body = str(c.get("body") or c.get("summary") or "")
            if dry_run:
                result.proposals.append({"op": "create_issue", "title": title})
                continue
            try:
                self._create(client, title, body, c)
                result.created += 1
            except Exception:  # noqa: BLE001
                logger.debug("%s create_issue failed", self.domain, exc_info=True)
                result.errors += 1
        return result


class GitLabIssueSink(_IssueSinkBase):
    domain = "gitlab"
    enable_flag = "GITLAB_ENABLE_WRITE"
    module = "gitlab_api"

    def _create(self, client, title, body, c):
        client.create_issue(
            project_id=c.get("project_id") or c.get("project"),
            title=title,
            description=body,
        )


class GitHubIssueSink(_IssueSinkBase):
    domain = "github"
    enable_flag = "GITHUB_ENABLE_WRITE"
    module = "github_agent"

    def _create(self, client, title, body, c):
        client.create_issue(c.get("owner"), c.get("repo"), title, body=body)


class PlaneIssueSink(_IssueSinkBase):
    domain = "plane"
    enable_flag = "PLANE_ENABLE_WRITE"
    module = "plane_agent"

    def _create(self, client, title, body, c):
        client.create_work_item(
            c.get("project_id") or c.get("project"),
            {"name": title, "description_html": body},
        )


class JiraIssueSink(_IssueSinkBase):
    domain = "jira"
    enable_flag = "JIRA_ENABLE_WRITE"
    module = "atlassian_agent"

    def _client(self, ops):
        if ops.get("client") is not None:
            return ops["client"]
        try:
            from atlassian_agent.auth import get_jira_cloud_client

            return get_jira_cloud_client()
        except Exception:  # noqa: BLE001
            return None

    def _create(self, client, title, body, c):
        client.jira_cloud_create_issue(
            payload={
                "fields": {
                    "project": {"key": c.get("project") or c.get("project_id")},
                    "summary": title,
                    "description": body,
                    "issuetype": {"name": c.get("issue_type", "Task")},
                }
            }
        )


# CONCEPT:AU-KG.enrichment.ticket-status-comment-writeback — Ticket status and comment writeback
# ── Status-transition + comment sinks (CONCEPT:AU-KG.enrichment.ticket-status-comment-writeback) ─────────────────────
#
# The action half of the ticket→PR loop: KG/agent decisions move an issue through
# its workflow (In Progress → Ready for QA → Done) and leave an audit comment. These
# are HIGH-STAKES (they mutate live tracker state), so they never auto-execute —
# ``run_writeback`` queues them for approval (``graph_writeback action=approve``).
# ``ops["transitions"]`` items: ``{key|id|work_item_id, status|state, project_id?,
# comment?}``.


def _adf_comment(text: str) -> dict[str, Any]:
    """Wrap plain text as a minimal Atlassian Document Format comment payload (v3)."""
    return {
        "body": {
            "type": "doc",
            "version": 1,
            "content": [
                {"type": "paragraph", "content": [{"type": "text", "text": str(text)}]}
            ],
        }
    }


class _TransitionSinkBase(ABC):
    domain = ""
    enable_flag = ""
    risk_tier = "high_stakes"

    @abstractmethod
    def _client(self, ops: dict[str, Any]) -> Any | None:
        """Resolve the tracker client (or ``None`` when unavailable)."""

    @abstractmethod
    def _apply(self, client: Any, item: dict[str, Any]) -> None:
        """Transition one ticket to its target status and add any comment."""

    def run(
        self, ctx: WritebackContext, ops: dict[str, Any], *, dry_run: bool
    ) -> WritebackResult:
        result = WritebackResult(target=self.domain)
        items = ops.get("transitions") or []
        client = None if dry_run else self._client(ops)
        if not dry_run and client is None:
            result.skipped += len(items)
            return result
        for item in items:
            ref = item.get("key") or item.get("work_item_id") or item.get("id")
            target = item.get("status") or item.get("state")
            if not ref or not target:
                result.skipped += 1
                continue
            if dry_run:
                result.proposals.append(
                    {"op": "transition", "ticket": str(ref), "to": str(target)}
                )
                continue
            try:
                self._apply(client, item)
                result.enriched += 1
            except Exception:  # noqa: BLE001
                logger.debug("%s transition failed", self.domain, exc_info=True)
                result.errors += 1
        return result


class JiraTransitionSink(_TransitionSinkBase):
    domain = "jira_transition"
    enable_flag = "JIRA_ENABLE_WRITE"

    def _client(self, ops):
        if ops.get("client") is not None:
            return ops["client"]
        try:
            from atlassian_agent.auth import get_jira_cloud_client

            return get_jira_cloud_client()
        except Exception:  # noqa: BLE001
            return None

    def _apply(self, client, item):
        key = item.get("key") or item.get("id")
        target = str(item.get("status") or item.get("state") or "")
        resp = client.jira_cloud_get_transitions(issue_id_or_key=key)
        data = getattr(resp, "data", resp) or {}
        wanted = target.lower()
        tid = None
        for t in data.get("transitions") or []:
            names = {
                str(t.get("name") or "").lower(),
                str((t.get("to") or {}).get("name") or "").lower(),
            }
            if wanted in names:
                tid = t.get("id")
                break
        if tid is None:
            raise ValueError(f"no transition to {target!r} available for {key}")
        client.jira_cloud_do_transition(
            issue_id_or_key=key, payload={"transition": {"id": tid}}
        )
        if comment := item.get("comment"):
            client.jira_cloud_add_comment(
                issue_id_or_key=key, payload=_adf_comment(comment)
            )


class PlaneStateSink(_TransitionSinkBase):
    domain = "plane_state"
    enable_flag = "PLANE_ENABLE_WRITE"

    def _client(self, ops):
        if ops.get("client") is not None:
            return ops["client"]
        try:
            from plane_agent.auth import get_client

            return get_client()
        except Exception:  # noqa: BLE001
            return None

    def _apply(self, client, item):
        project = item.get("project_id") or item.get("project")
        wid = item.get("work_item_id") or item.get("id")
        state = item.get("state") or item.get("status")
        if not (project and wid and state):
            raise ValueError("plane transition needs project_id, work_item_id, state")
        client.update_work_item(project, wid, {"state": state})
        if comment := item.get("comment"):
            client.create_work_item_comment(
                project, wid, {"comment_html": f"<p>{comment}</p>"}
            )


register_sink(GitLabIssueSink())
register_sink(GitHubIssueSink())
register_sink(PlaneIssueSink())
register_sink(JiraIssueSink())
register_sink(JiraTransitionSink())
register_sink(PlaneStateSink())
