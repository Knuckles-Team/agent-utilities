"""AU-SEC-R006: the shared elevation client keeps EG's two-person rules at every surface."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from agent_utilities.security.elevation import (
    APPROVAL_SCOPE,
    ElevationApproval,
    ElevationRefused,
    ElevationRequest,
    ElevationRevocation,
    ElevationService,
    ElevationSurface,
)
from tests.unit.security.elevation_fakes import FakeElevationEngine

HOUR_MS = 60 * 60 * 1000


def _ask(**overrides: Any) -> ElevationRequest:
    values: dict[str, Any] = {
        "scopes": [{"graph": "tenant-a", "action": "write"}],
        "span_ms": HOUR_MS,
        "justification": "incident 42 data repair",
    }
    values.update(overrides)
    return ElevationRequest.model_validate(values)


def _claims(agent: str = "bob", **overrides: Any) -> dict[str, Any]:
    claims: dict[str, Any] = {
        "agent_id": agent,
        "principal": agent,
        "delegation": [],
        "scopes": ["kg:read", APPROVAL_SCOPE],
    }
    claims.update(overrides)
    return claims


@pytest.fixture
def engine() -> FakeElevationEngine:
    return FakeElevationEngine()


@pytest.fixture
def service(engine: FakeElevationEngine) -> ElevationService:
    return ElevationService(engine, caller="alice", clock_ms=lambda: engine.now_ms)


async def _requested(engine: FakeElevationEngine, service: ElevationService) -> Any:
    engine.caller = "alice"
    view = await service.request(_ask())
    engine.caller = "bob"
    return view


async def test_a_request_grants_nothing_and_carries_no_identity(
    engine: FakeElevationEngine, service: ElevationService
) -> None:
    view = await service.request(_ask())
    assert view.status == "requested" and view.remaining_ms == 0
    assert view.grantee == "alice", "the grantee is the verified caller"
    assert view.own, "a console never offers to approve the caller's own request"
    assert view.elevation_id.startswith("elevation-")
    assert engine.sent == [
        {
            "op": {
                "op": "request",
                "request": {
                    "scopes": [{"graph": "tenant-a", "action": "write"}],
                    "span_ms": HOUR_MS,
                    "justification": "incident 42 data repair",
                    "elevation_id": view.elevation_id,
                },
            }
        }
    ]


async def test_a_direct_approver_with_the_exact_scope_activates_the_window(
    engine: FakeElevationEngine, service: ElevationService
) -> None:
    asked = await _requested(engine, service)
    active = await service.approve(
        ElevationApproval(
            elevation_id=asked.elevation_id, request_digest=asked.request_digest
        ),
        surface=ElevationSurface.OPERATOR_CONSOLE,
        claims=_claims(),
    )
    assert active.status == "active"
    assert active.remaining_ms == HOUR_MS, "the countdown starts at approval"
    engine.now_ms += HOUR_MS // 4
    listed = await service.list_elevations()
    assert listed[0].remaining_ms == 3 * HOUR_MS // 4


@pytest.mark.parametrize(
    ("surface", "claims", "code"),
    [
        (ElevationSurface.AGENT_TOOL, _claims(), "ELEVATION_APPROVAL_SURFACE"),
        (ElevationSurface.A2A, _claims(), "ELEVATION_APPROVAL_SURFACE"),
        (
            ElevationSurface.OPERATOR_CONSOLE,
            _claims(delegation=["bob", "agent:chat:1"]),
            "ELEVATION_APPROVER_DELEGATED",
        ),
        (
            ElevationSurface.OPERATOR_CONSOLE,
            _claims(scopes=["*", "kg:admin", "rbac:*"]),
            "ELEVATION_APPROVER_SCOPE",
        ),
    ],
)
async def test_an_approval_is_refused_before_anything_is_sent(
    engine: FakeElevationEngine,
    service: ElevationService,
    surface: ElevationSurface,
    claims: dict[str, Any],
    code: str,
) -> None:
    asked = await _requested(engine, service)
    with pytest.raises(ElevationRefused) as refused:
        await service.approve(
            ElevationApproval(
                elevation_id=asked.elevation_id, request_digest=asked.request_digest
            ),
            surface=surface,
            claims=claims,
        )
    assert refused.value.code == code
    assert engine.ops() == ["request"]


@pytest.mark.parametrize(
    ("approver", "digest", "elevation_id", "code"),
    [
        ("alice", None, None, "ELEVATION_SELF_APPROVAL"),
        ("bob", "sha256:other", None, "ELEVATION_STALE_VIEW"),
        ("bob", None, "elevation-unknown", "ELEVATION_NOT_FOUND"),
    ],
)
async def test_an_approval_must_name_someone_else_s_exact_request(
    engine: FakeElevationEngine,
    service: ElevationService,
    approver: str,
    digest: str | None,
    elevation_id: str | None,
    code: str,
) -> None:
    asked = await _requested(engine, service)
    engine.caller = approver
    with pytest.raises(ElevationRefused) as refused:
        await service.approve(
            ElevationApproval(
                elevation_id=elevation_id or asked.elevation_id,
                request_digest=digest or asked.request_digest,
            ),
            surface=ElevationSurface.OPERATOR_CONSOLE,
            claims=_claims(approver),
        )
    assert refused.value.code == code
    assert "approve" not in engine.ops()


async def test_revocation_ends_the_window_now(
    engine: FakeElevationEngine, service: ElevationService
) -> None:
    asked = await _requested(engine, service)
    await service.approve(
        ElevationApproval(
            elevation_id=asked.elevation_id, request_digest=asked.request_digest
        ),
        surface=ElevationSurface.OPERATOR_CONSOLE,
        claims=_claims(),
    )
    revoked = await service.revoke(ElevationRevocation(elevation_id=asked.elevation_id))
    assert revoked.status == "revoked" and revoked.remaining_ms == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"scopes": [{"graph": "tenant-*", "action": "read"}]},
        {"scopes": [{"graph": "__admin__", "action": "read"}]},
        {"scopes": [{"graph": "tenant-a", "action": "admin"}]},
        {"scopes": []},
        {"span_ms": 24 * HOUR_MS + 1},
        {"span_ms": 0},
        {"actor": {"agent_id": "someone-else"}},
        {"approver": "bob"},
    ],
)
def test_a_request_body_is_exact_and_bounded(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _ask(**overrides)
