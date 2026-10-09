"""AU-CONTEXT-R007.1: typed delegation seam tests."""

from __future__ import annotations

import pytest

import agent_utilities.api.finance_delegation as finance_delegation
from agent_utilities.api.finance_delegation import (
    KellySizingRequest,
    KellySizingResponse,
    kelly_size_via_eg_or_local,
    resolve_finance_primitives_client,
)


@pytest.mark.spec("AU-CONTEXT-R007.1")
def test_resolve_returns_none_when_eg_finance_core_is_absent() -> None:
    # EG has not shipped a finance-core client yet; this must return None,
    # not raise and not fabricate a client.
    assert resolve_finance_primitives_client() is None


@pytest.mark.spec("AU-CONTEXT-R007.1")
def test_delegation_seam_falls_back_to_local_when_eg_is_unavailable() -> None:
    calls: list[KellySizingRequest] = []

    def local_fallback(request: KellySizingRequest) -> KellySizingResponse:
        calls.append(request)
        fraction = request.win_probability - (
            (1 - request.win_probability) / request.win_loss_ratio
        )
        return KellySizingResponse(suggested_fraction=max(0.0, min(1.0, fraction)))

    request = KellySizingRequest(win_probability=0.6, win_loss_ratio=2.0)
    response = kelly_size_via_eg_or_local(request, local_fallback=local_fallback)

    assert calls == [request]
    assert response.suggested_fraction == max(0.0, min(1.0, 0.6 - (0.4 / 2.0)))


@pytest.mark.spec("AU-CONTEXT-R007.1")
def test_delegation_seam_prefers_eg_client_when_one_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate EG's finance-core client being installed and resolved: the
    # seam must delegate to it and must NOT fall back to the caller-supplied
    # local calculation.
    class _FakeClient:
        def __init__(self) -> None:
            self.calls: list[KellySizingRequest] = []

        def kelly_sizing(self, request: KellySizingRequest) -> KellySizingResponse:
            self.calls.append(request)
            return KellySizingResponse(suggested_fraction=0.42)

    fake_client = _FakeClient()
    monkeypatch.setattr(
        finance_delegation,
        "resolve_finance_primitives_client",
        lambda: fake_client,
    )

    def local_fallback(request: KellySizingRequest) -> KellySizingResponse:
        raise AssertionError(
            "local fallback must not run when an EG client is installed"
        )

    request = KellySizingRequest(win_probability=0.6, win_loss_ratio=2.0)
    response = kelly_size_via_eg_or_local(request, local_fallback=local_fallback)

    assert fake_client.calls == [request]
    assert response.suggested_fraction == 0.42
