"""Public read-boundary checks; runtime lifecycle and hosted proof are pending."""

from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from agent_utilities.api import CurrentPlanRead, CurrentPlanRequest, CurrentPlanSnapshot
from agent_utilities.api.agent_control_plane import compose_agent_control_plane
from agent_utilities.api.session import GraphSession, ScopeError, use_session
from agent_utilities.security.actor_identity import ActorType
from agent_utilities.security.brain_context import ActorContext

pytestmark = pytest.mark.spec("AU-HARNESS-R014.9.1", "AU-HARNESS-R014.10.1")


def session():
    return GraphSession(
        actor=ActorContext(
            actor_id="agent:test",
            actor_type=ActorType.AI_AGENT,
            tenant_id="tenant:test",
            authenticated=True,
        ),
        tenant="tenant:test",
        scopes=frozenset({"kg:read", "kg:write"}),
        policy_version="policy:test",
        audience="graph-os",
    )


class Client:
    @contextmanager
    def use_verified_context(self, claims):
        yield


class Reader:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def read(self, request, *, session):
        self.calls.append((request, session))
        return self.result


def available(**changes):
    now = datetime.now(UTC)
    values = dict(
        run_id="run-one",
        revision=0,
        observed_at=now,
        expires_at=now + timedelta(seconds=30),
        items=(),
    )
    values.update(changes)
    return CurrentPlanRead(
        availability="available", snapshot=CurrentPlanSnapshot(**values)
    )


@pytest.mark.asyncio
async def test_public_read_passes_verified_session_and_preserves_empty_plan():
    owner = session()
    port = Reader(available())
    request = CurrentPlanRequest(run_id="run-one")
    control = compose_agent_control_plane(Client(), owner, current_plan_reader=port)
    with use_session(owner):
        assert await control.current_plan(request) == port.result
    assert port.calls == [(request, owner)]


@pytest.mark.asyncio
async def test_absent_port_is_unavailable_but_still_requires_read_scope():
    owner = session()
    request = CurrentPlanRequest(run_id="run-one")
    with use_session(owner):
        assert (
            await compose_agent_control_plane(Client(), owner).current_plan(request)
        ).snapshot is None
    denied = replace(owner, scopes=frozenset())
    with use_session(denied), pytest.raises(ScopeError):
        await compose_agent_control_plane(Client(), denied).current_plan(request)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["wrong_run", "expired", "future"])
async def test_invalid_snapshot_identity_or_time_is_unavailable(case):
    now = datetime.now(UTC)
    changes = {
        "wrong_run": {"run_id": "different"},
        "expired": {
            "observed_at": now - timedelta(seconds=60),
            "expires_at": now - timedelta(seconds=30),
        },
        "future": {"observed_at": now + timedelta(seconds=10)},
    }[case]
    owner = session()
    control = compose_agent_control_plane(
        Client(), owner, current_plan_reader=Reader(available(**changes))
    )
    with use_session(owner):
        result = await control.current_plan(CurrentPlanRequest(run_id="run-one"))
    assert result == CurrentPlanRead(availability="unavailable")


@pytest.mark.asyncio
async def test_malformed_constructed_request_never_reaches_port():
    owner = session()
    port = Reader(CurrentPlanRead(availability="unavailable"))
    control = compose_agent_control_plane(Client(), owner, current_plan_reader=port)
    with use_session(owner), pytest.raises(ValidationError):
        await control.current_plan(CurrentPlanRequest.model_construct(run_id=123))
    assert port.calls == []
