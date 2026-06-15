"""Spec ↔ ticket ↔ agent bidirectional linking (CONCEPT:KG-2.9).

Ties KG SDD specs/features to Plane & Jira work items: push spec content onto the
item (as it is drafted→completed), link the item to the KG spec, assign it (to an
agent's bot-user, an explicit user, or the configured act-as user), and comment.
Reads back "who owns what" (items assigned to a user) into the KG.

Fail-closed (``PLANE_ENABLE_WRITE`` / ``JIRA_ENABLE_WRITE``) + dry-run-first.

Agent assignability resolves in this order: explicit ``assignee`` → the agent's
mapped bot-user (``AGENT_USER_MAP`` config) → the configured act-as user
(``PLANE_ACT_AS_USER`` / ``JIRA_ACT_AS_USER``); a metadata fallback stamps the agent
id onto the item when no real user maps.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)


def _plane_client(client: Any | None) -> Any | None:
    if client is not None:
        return client
    try:
        from plane_agent.auth import get_client

        return get_client()
    except Exception:  # noqa: BLE001
        return None


def _jira_client(client: Any | None) -> Any | None:
    if client is not None:
        return client
    try:
        from atlassian_agent.auth import get_jira_cloud_client

        return get_jira_cloud_client()
    except Exception:  # noqa: BLE001
        return None


def _agent_user(target: str, agent: str | None) -> str | None:
    if not agent:
        return None
    try:
        mapping = json.loads(setting("AGENT_USER_MAP", "{}") or "{}")
    except Exception:  # noqa: BLE001
        mapping = {}
    entry = mapping.get(agent) or {}
    return entry.get(target) if isinstance(entry, dict) else None


def _resolve_assignee(
    target: str, assignee: str | None, agent: str | None
) -> str | None:
    """explicit user → agent's bot-user → configured act-as user."""
    return (
        assignee
        or _agent_user(target, agent)
        or (setting(f"{target.upper()}_ACT_AS_USER", "") or None)
    )


def _spec_text(spec: dict[str, Any]) -> tuple[str, str]:
    title = spec.get("title") or spec.get("feature_id") or "Spec"
    stories = spec.get("user_stories") or []
    lines = [f"Spec: {spec.get('feature_id', '')}"]
    for s in stories if isinstance(stories, list) else []:
        if isinstance(s, dict):
            lines.append(f"- {s.get('title') or s.get('description', '')}")
    for nfr in spec.get("non_functional_requirements") or []:
        lines.append(f"NFR: {nfr}")
    return title, "\n".join(lines)


def _spec_node_id(spec: dict[str, Any]) -> str:
    return f"spec:{spec.get('feature_id', spec.get('title', 'spec'))}"


def _write_tracked_by(
    backend: Any, spec: dict[str, Any], target: str, issue_id: str
) -> None:
    """Best-effort: stamp :trackedBy on the KG spec node + external_links mirror."""
    if backend is None:
        return
    try:
        backend.execute(
            "MERGE (n {id: $id}) SET n.trackedBy = $tb, n.domain = coalesce(n.domain,'sdd')",
            {"id": _spec_node_id(spec), "tb": f"{target}:{issue_id}"},
        )
    except Exception:  # noqa: BLE001
        logger.debug("trackedBy write failed", exc_info=True)


def link_spec(
    spec: dict[str, Any],
    *,
    target: str,
    issue_id: str,
    project_id: str | None = None,
    assignee: str | None = None,
    agent: str | None = None,
    comment: str | None = None,
    backend: Any = None,
    client: Any = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Push spec → ticket: update content, link, assign, comment. Fail-closed."""
    target = target.lower()
    enabled = bool(setting(f"{target.upper()}_ENABLE_WRITE", False, cast=bool))
    if not dry_run and not enabled:
        return {
            "status": "refused",
            "reason": f"{target.upper()}_ENABLE_WRITE not set; refusing live write",
        }
    title, body = _spec_text(spec)
    who = _resolve_assignee(target, assignee, agent)
    actions = [
        {"op": "update_item", "issue": issue_id, "title": title},
        {"op": "link_spec", "issue": issue_id, "spec": _spec_node_id(spec)},
    ]
    if who:
        actions.append({"op": "assign", "issue": issue_id, "user": who})
    elif agent:
        actions.append({"op": "tag_agent_metadata", "issue": issue_id, "agent": agent})
    if comment:
        actions.append({"op": "comment", "issue": issue_id})

    if dry_run:
        # record the external link in the KG even on preview (cheap, reversible)
        spec.setdefault("external_links", {})[target] = issue_id
        return {
            "status": "completed",
            "dry_run": True,
            "target": target,
            "proposals": actions,
        }

    errors = 0
    spec_url = f"kg://spec/{_spec_node_id(spec)}"
    try:
        if target == "plane":
            cl = _plane_client(client)
            data: dict[str, Any] = {
                "name": title,
                "description_html": body.replace("\n", "<br/>"),
            }
            if who:
                data["assignees"] = [who]
            elif agent:
                data["metadata"] = {"kg_agent": agent}
            cl.update_work_item(project_id, issue_id, data)  # type: ignore[union-attr]  # client None-checked above
            cl.create_work_item_link(  # type: ignore[union-attr]  # client None-checked above
                project_id, issue_id, {"url": spec_url, "title": f"KG spec: {title}"}
            )
            if comment:
                cl.create_work_item_comment(  # type: ignore[union-attr]  # client None-checked above
                    project_id, issue_id, {"comment_html": comment}
                )
        elif target == "jira":
            cl = _jira_client(client)
            cl.jira_cloud_edit_issue(  # type: ignore[union-attr]  # client None-checked above
                issue_id, payload={"fields": {"summary": title, "description": body}}
            )
            if who:
                cl.jira_cloud_assign_issue(issue_id, payload={"accountId": who})  # type: ignore[union-attr]  # client None-checked above
            cl.jira_cloud_create_or_update_remote_issue_link(  # type: ignore[union-attr]  # client None-checked above
                issue_id,
                payload={
                    "globalId": spec_url,
                    "object": {"url": spec_url, "title": f"KG spec: {title}"},
                },
            )
            if comment:
                cl.jira_cloud_add_comment(issue_id, payload={"body": comment})  # type: ignore[union-attr]  # client None-checked above
        else:
            return {"status": "error", "error": f"unsupported tracker {target!r}"}
    except Exception as e:  # noqa: BLE001
        logger.debug("link_spec %s failed", target, exc_info=True)
        return {"status": "error", "target": target, "error": str(e)}

    spec.setdefault("external_links", {})[target] = issue_id
    _write_tracked_by(backend, spec, target, issue_id)
    return {
        "status": "completed",
        "dry_run": False,
        "target": target,
        "issue": issue_id,
        "assignee": who,
        "errors": errors,
    }


def pull_assigned(
    target: str,
    *,
    user: str | None = None,
    project_id: str | None = None,
    client: Any = None,
) -> dict[str, Any]:
    """Read items assigned to ``user`` (default the act-as user) — "what do I own?"."""
    target = target.lower()
    user = user or (setting(f"{target.upper()}_ACT_AS_USER", "") or None)
    items: list[dict[str, Any]] = []
    try:
        if target == "plane":
            cl = _plane_client(client)
            resp = cl.list_work_items(project_id) if cl else None
            rows = getattr(resp, "data", resp) or []
            rows = rows.get("results", rows) if isinstance(rows, dict) else rows
            for w in rows if isinstance(rows, list) else []:
                assignees = w.get("assignees") or w.get("assignee_ids") or []
                if user is None or user in assignees:
                    items.append(
                        {
                            "id": w.get("id"),
                            "name": w.get("name"),
                            "state": w.get("state"),
                            "assignees": assignees,
                        }
                    )
        elif target == "jira":
            cl = _jira_client(client)
            jql = f'assignee = "{user}"' if user else "assignee is not EMPTY"
            searcher = getattr(cl, "jira_cloud_search", None) if cl else None
            resp = searcher(payload={"jql": jql}) if callable(searcher) else None
            data = getattr(resp, "data", resp) or {}
            for it in data.get("issues", []) if isinstance(data, dict) else []:
                f = it.get("fields", {})
                items.append(
                    {
                        "id": it.get("key"),
                        "name": f.get("summary"),
                        "assignee": (f.get("assignee") or {}).get("accountId"),
                    }
                )
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "target": target, "error": str(e)}
    return {"status": "completed", "target": target, "user": user, "items": items}
