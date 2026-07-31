"""The multiplexer reads skill BODIES off an already-connected child.

CONCEPT:AU-ECO.mcp.cross-process-skill-harvest — the discovery half of the
harvest. ``_probe_skills`` used to return names only, which is why a fleet
skill could never be promoted to a runnable resource: there was no instruction
body to promote. These tests pin the body read, its bounds, and the rule that a
body which cannot be read produces a NAMED reason rather than a silent gap.
"""

from __future__ import annotations

import pytest

from agent_utilities.mcp.multiplexer import (
    _MAX_SKILL_BODY_BYTES,
    MCPMultiplexer,
    _resource_body_text,
)


class _Resource:
    def __init__(self, uri: str, description: str = "") -> None:
        self.uri = uri
        self.description = description


class _Text:
    def __init__(self, text: str) -> None:
        self.text = text


class _Blob:
    def __init__(self, blob: bytes) -> None:
        self.blob = blob
        self.text = None


class _Result:
    def __init__(self, contents) -> None:
        self.contents = contents


class _Session:
    """A child that serves skill resources, optionally failing some reads."""

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
        if isinstance(body, bytes):
            return _Result([_Blob(body)])
        return _Result([_Text(body)])


def _mux() -> MCPMultiplexer:
    return MCPMultiplexer.__new__(MCPMultiplexer)


@pytest.mark.asyncio
async def test_probe_returns_each_skill_with_its_instruction_body():
    session = _Session(
        {
            "skill://servicenow-incident-management/SKILL.md": "# incidents\n\nbody",
            "skill://servicenow-cmdb/SKILL.md": "# cmdb\n\nbody",
        }
    )

    skills = await _mux()._probe_skills("servicenow-mcp", session)

    assert {s["name"] for s in skills} == {
        "servicenow-incident-management",
        "servicenow-cmdb",
    }
    assert all(s["instructions"].startswith("# ") for s in skills)
    assert not any(s.get("harvest_error") for s in skills)


@pytest.mark.asyncio
async def test_a_rate_limited_read_is_retried_rather_than_stranded():
    """A child defending itself must not permanently strand its own corpus."""
    session = _Session({"skill://s/SKILL.md": "# s\n\nbody"}, fail_times=3)

    skills = await _mux()._probe_skills("child-mcp", session)

    assert skills[0]["instructions"] == "# s\n\nbody"
    assert "harvest_error" not in skills[0]
    assert len(session.reads) == 4


@pytest.mark.asyncio
async def test_an_unreadable_body_records_a_named_reason_and_no_instructions():
    """Fail closed and say why — never a silently body-less skill."""
    session = _Session({"skill://s/SKILL.md": "# s"}, fail_times=99)

    skills = await _mux()._probe_skills("child-mcp", session)

    assert "instructions" not in skills[0]
    assert "Rate limit exceeded" in skills[0]["harvest_error"]


@pytest.mark.asyncio
async def test_an_empty_body_is_rejected_with_a_named_reason():
    session = _Session({"skill://s/SKILL.md": "   \n  "})

    skills = await _mux()._probe_skills("child-mcp", session)

    assert "instructions" not in skills[0]
    assert skills[0]["harvest_error"] == "server served an empty skill body"


@pytest.mark.asyncio
async def test_an_oversized_body_is_rejected_with_a_named_reason():
    session = _Session({"skill://s/SKILL.md": "x" * (_MAX_SKILL_BODY_BYTES + 1)})

    skills = await _mux()._probe_skills("child-mcp", session)

    assert "instructions" not in skills[0]
    assert skills[0]["harvest_error"] == "skill body exceeded its size boundary"


def test_a_binary_payload_is_not_coerced_into_an_instruction_body():
    """A blob is not a skill body; accepting one would fabricate instructions."""
    with pytest.raises(RuntimeError, match="not text"):
        _resource_body_text(_Result([_Blob(b"\x00\x01")]))

    with pytest.raises(RuntimeError, match="no contents"):
        _resource_body_text(_Result([]))
