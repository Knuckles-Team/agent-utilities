"""Fleet prompts become ``:Prompt`` nodes through the harvest's promotion half.

CONCEPT:AU-ECO.mcp.cross-process-prompt-harvest — the promotion half of the
cross-process prompt harvest: a prompt body pulled off a fleet MCP child is
written through the SAME ``ingest_prompt_node`` primitive
``ingest_prompts_to_graph`` uses for the packaged base, so a fleet prompt has
the exact same node shape/id scheme regardless of which path discovered it. A
prompt whose body never arrived, or does not parse as the blueprint schema,
is not promoted at all but records WHY.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.ingestion.fleet_prompt_harvest import (
    promote_harvested_prompts,
)


class _FakeEngine:
    """Minimal duck-typed stand-in covering only what ``ingest_prompt_node`` touches."""

    def __init__(self) -> None:
        self.upsert_calls: list[tuple[str, str, dict[str, Any]]] = []

    def _serialize_node(self, node: Any, label: str | None = None) -> dict[str, Any]:
        return {"id": node.id, "name": getattr(node, "name", None)}

    def _upsert_node(self, label: str, node_id: str, data: dict[str, Any]) -> None:
        self.upsert_calls.append((label, node_id, data))

    def prompt_ids_upserted(self) -> list[str]:
        return [c[1] for c in self.upsert_calls if c[0] == "Prompt"]


def _entry(name: str, provider: str = "servicenow-api", **overrides: Any) -> dict:
    entry = {
        "name": name,
        "provider": provider,
        "uri": f"prompt://{provider}/{name}",
        "body": f'{{"name": "{name}", "description": "d", "content": "do the thing"}}',
    }
    entry.update(overrides)
    return entry


def _server(*entries: dict, error: str | None = None) -> dict:
    return {"error": error, "tools": [], "skills": [], "prompts": list(entries)}


def test_a_harvested_prompt_body_is_promoted_to_a_prompt_node():
    engine = _FakeEngine()
    report = promote_harvested_prompts(
        engine, {"servicenow-mcp": _server(_entry("incident-triage"))}
    )

    assert report["promoted"] == 1
    assert report["blocked"] == 0
    assert report["errors"] == 0
    assert engine.prompt_ids_upserted() == ["prompt:servicenow-api/incident-triage"]


def test_a_prompt_with_no_body_is_blocked_with_a_named_reason():
    engine = _FakeEngine()
    report = promote_harvested_prompts(
        engine,
        {
            "servicenow-mcp": _server(
                _entry(
                    "incident-triage",
                    body=None,
                    harvest_error="server served an empty prompt body",
                )
            )
        },
    )

    assert report["promoted"] == 0
    assert report["blocked"] == 1
    assert (
        report["blocked_detail"]["servicenow-mcp/servicenow-api/incident-triage"]
        == "server served an empty prompt body"
    )
    assert engine.prompt_ids_upserted() == []


def test_a_body_that_does_not_parse_as_json_is_an_error_not_a_silent_drop():
    engine = _FakeEngine()
    report = promote_harvested_prompts(
        engine,
        {
            "servicenow-mcp": _server(
                _entry("incident-triage", body="not-json-at-all")
            )
        },
    )

    assert report["promoted"] == 0
    assert report["errors"] == 1
    assert (
        "JSONDecodeError"
        in report["error_detail"]["servicenow-mcp/servicenow-api/incident-triage"]
    )
    assert engine.prompt_ids_upserted() == []


def test_multiple_fleet_providers_are_namespaced_independently():
    engine = _FakeEngine()
    report = promote_harvested_prompts(
        engine,
        {
            "servicenow-mcp": _server(_entry("triage", provider="servicenow-api")),
            "gitlab-mcp": _server(_entry("triage", provider="gitlab-api")),
        },
    )

    assert report["promoted"] == 2
    assert sorted(engine.prompt_ids_upserted()) == [
        "prompt:gitlab-api/triage",
        "prompt:servicenow-api/triage",
    ]


def test_an_unreachable_server_contributes_no_prompts():
    engine = _FakeEngine()
    report = promote_harvested_prompts(
        engine, {"down-mcp": _server(error="connection refused")}
    )

    assert report["promoted"] == 0
    assert report["blocked"] == 0
    assert report["errors"] == 0
