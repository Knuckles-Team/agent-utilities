"""Issue-tracker write-back sinks — GitLab / GitHub / Plane (CONCEPT:KG-2.9).

Closes the inference→action loop: KG-derived findings (TRM end-of-life/vuln risk,
tech-debt, failing tests) become *filed tickets*. Fail-closed
(``GITLAB_ENABLE_WRITE`` / ``GITHUB_ENABLE_WRITE`` / ``PLANE_ENABLE_WRITE``),
dry-run-first. ``creations`` items: ``{name/title, body, project_id|owner+repo}``.
"""

from __future__ import annotations

import logging
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


class _IssueSinkBase:
    domain = ""
    enable_flag = ""
    module = ""

    def _create(self, client: Any, title: str, body: str, c: dict[str, Any]) -> None:
        raise NotImplementedError  # ABSTRACT-OK

    def run(
        self, ctx: WritebackContext, ops: dict[str, Any], *, dry_run: bool
    ) -> WritebackResult:
        result = WritebackResult(target=self.domain)
        client = _resolve_client(ops, self.module)
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


register_sink(GitLabIssueSink())
register_sink(GitHubIssueSink())
register_sink(PlaneIssueSink())
