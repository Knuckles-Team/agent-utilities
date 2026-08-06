"""The multiplexer reads prompt BODIES off an already-connected child.

CONCEPT:AU-ECO.mcp.cross-process-prompt-harvest — the ``prompt://`` sibling of
``test_skill_body_harvest.py``. ``_probe_prompts`` mirrors ``_probe_skills``:
a fleet prompt is invisible to graph-os's in-process
``resolve_prompt_provider_dirs()`` (the fleet package is deliberately not
co-installed in graph-os's venv), so it can only ever become a ``:Prompt``
node by reading the body back over the multiplexer's already-open probe
session. These tests pin the body read, its bounds, and the rule that a body
which cannot be read produces a NAMED reason rather than a silent gap.
"""

from __future__ import annotations

import pytest

from agent_utilities.mcp.multiplexer import (
    _MAX_PROMPT_BODY_BYTES,
    MCPMultiplexer,
)


class _Resource:
    def __init__(self, uri: str, description: str = "") -> None:
        self.uri = uri
        self.description = description


class _Text:
    def __init__(self, text: str) -> None:
        self.text = text


class _Result:
    def __init__(self, contents) -> None:
        self.contents = contents


class _Session:
    """A child that serves prompt resources, optionally failing some reads."""

    def __init__(self, bodies: dict[str, object], fail_times: int = 0) -> None:
        self._bodies = bodies
        self._fail_times = fail_times
        self.reads: list[str] = []

    async def list_resources(self):
        class _R:
            resources = [_Resource(uri, "desc") for uri in self._bodies]

        _R.resources = [_Resource(uri, "desc") for uri in self._bodies]
        return _R

    async def read_resource(self, uri):
        self.reads.append(str(uri))
        if self._fail_times > 0:
            self._fail_times -= 1
            raise RuntimeError("Rate limit exceeded for client: global")
        body = self._bodies[str(uri)]
        return _Result([_Text(body)])


def _mux() -> MCPMultiplexer:
    return MCPMultiplexer.__new__(MCPMultiplexer)


@pytest.mark.asyncio
async def test_probe_returns_each_prompt_with_its_body_and_provider():
    session = _Session(
        {
            "prompt://servicenow-api/incident-triage": '{"name": "incident-triage"}',
            "prompt://servicenow-api/change-review": '{"name": "change-review"}',
        }
    )

    prompts = await _mux()._probe_prompts("servicenow-mcp", session)

    assert {p["name"] for p in prompts} == {"incident-triage", "change-review"}
    assert all(p["provider"] == "servicenow-api" for p in prompts)
    assert all(p["body"].startswith("{") for p in prompts)
    assert not any(p.get("harvest_error") for p in prompts)


@pytest.mark.asyncio
async def test_a_skill_resource_is_not_mistaken_for_a_prompt():
    session = _Session(
        {
            "skill://servicenow-incident-management/SKILL.md": "# incidents",
            "prompt://servicenow-api/incident-triage": '{"name": "incident-triage"}',
        }
    )

    prompts = await _mux()._probe_prompts("servicenow-mcp", session)

    assert len(prompts) == 1
    assert prompts[0]["name"] == "incident-triage"


@pytest.mark.asyncio
async def test_a_rate_limited_read_is_retried_rather_than_stranded():
    """A child defending itself must not permanently strand its own corpus."""
    session = _Session(
        {"prompt://p/x": '{"name": "x"}'},
        fail_times=3,
    )

    prompts = await _mux()._probe_prompts("child-mcp", session)

    assert prompts[0]["body"] == '{"name": "x"}'
    assert "harvest_error" not in prompts[0]
    assert len(session.reads) == 4


@pytest.mark.asyncio
async def test_an_unreadable_body_records_a_named_reason_and_no_body():
    """Fail closed and say why — never a silently body-less prompt."""
    session = _Session({"prompt://p/x": '{"name": "x"}'}, fail_times=99)

    prompts = await _mux()._probe_prompts("child-mcp", session)

    assert "body" not in prompts[0]
    assert "Rate limit exceeded" in prompts[0]["harvest_error"]


@pytest.mark.asyncio
async def test_an_empty_body_is_rejected_with_a_named_reason():
    session = _Session({"prompt://p/x": "   \n  "})

    prompts = await _mux()._probe_prompts("child-mcp", session)

    assert "body" not in prompts[0]
    assert prompts[0]["harvest_error"] == "server served an empty prompt body"


@pytest.mark.asyncio
async def test_an_oversized_body_is_rejected_with_a_named_reason():
    session = _Session({"prompt://p/x": "x" * (_MAX_PROMPT_BODY_BYTES + 1)})

    prompts = await _mux()._probe_prompts("child-mcp", session)

    assert "body" not in prompts[0]
    assert prompts[0]["harvest_error"] == "prompt body exceeded its size boundary"
