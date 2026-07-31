"""Tests for `agent_utilities.mcp.protocol_compat.check_mcp_sdk_floor` (D-ISR-2).

D-ISR-2: nothing asserted that the runtime's installed `mcp`/`fastmcp` matched
`pyproject.toml`'s declared `[mcp]` floor, so v2-targeted source passed every gate
and only failed at import time in production (`child_resilience.py` hard-importing
the SDK-v2 `MCPError` name against a pod that had resolved `mcp` 1.29.0). This is
the CI regression guard: it fails loudly, at test time, the moment this repo's own
lock resolves an `mcp`/`fastmcp` pair that no longer satisfies the declared floor —
instead of shipping a mismatch that only breaks at serve time.
"""

from __future__ import annotations

import importlib.metadata

from agent_utilities.mcp.protocol_compat import check_mcp_sdk_floor


def test_installed_sdk_satisfies_the_declared_floor() -> None:
    """The CI gate: THIS repo's currently-resolved `mcp`/`fastmcp` must satisfy the
    `[mcp]` extra's declared floor. A future lock regression that resolves an older
    SDK line fails here, at build/test time, rather than as a swallowed ImportError
    deep in a deployed child transport."""
    outcome = check_mcp_sdk_floor()
    assert outcome["ok"] is True, outcome["detail"]


def test_detects_a_v1_sdk_against_a_v2_declared_floor(monkeypatch) -> None:
    """Reproduces the exact D-ISR-2 production mismatch: a runtime that resolved
    `mcp` 1.29.0 / `fastmcp` 3.4.5 while `agent-utilities[mcp]` declares
    `fastmcp>=4.0.0b1` (SDK v2)."""
    real_version = importlib.metadata.version

    def fake_version(name: str) -> str:
        if name == "mcp":
            return "1.29.0"
        if name == "fastmcp":
            return "3.4.5"
        return real_version(name)

    monkeypatch.setattr(importlib.metadata, "version", fake_version)

    outcome = check_mcp_sdk_floor()
    assert outcome["ok"] is False
    assert "fastmcp 3.4.5" in outcome["detail"]
    assert "mcp 1.29.0" in outcome["detail"]


def test_reports_ok_none_when_distribution_declares_no_mcp_extra() -> None:
    """A distribution with no `[mcp]`-gated `fastmcp` requirement (e.g. a package
    that never declared the extra) is a distinct outcome from a real mismatch —
    the check must not report `ok=False` for something it cannot evaluate."""
    outcome = check_mcp_sdk_floor(distribution="pytest")
    assert outcome["ok"] is None


def test_missing_fastmcp_install_fails_closed(monkeypatch) -> None:
    """If `fastmcp` itself is not installed at all, that's a real failure, not a
    skip — the `[mcp]` extra was declared but never actually resolved."""
    real_version = importlib.metadata.version

    def fake_version(name: str) -> str:
        if name == "fastmcp":
            raise importlib.metadata.PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(importlib.metadata, "version", fake_version)

    outcome = check_mcp_sdk_floor()
    assert outcome["ok"] is False
    assert "fastmcp is not installed" in outcome["detail"]


def test_doctor_check_wires_into_the_registered_surface() -> None:
    """Live-path guard: the check is reachable through `agent-utilities doctor`
    (`CHECKS["mcp_sdk_floor"]`), not just importable as a standalone helper."""
    from agent_utilities.deployment.doctor import CHECKS

    assert "mcp_sdk_floor" in CHECKS
    result = CHECKS["mcp_sdk_floor"]()
    assert result["name"] == "mcp_sdk_floor"
    assert result["status"] in {"ok", "warn", "fail", "skip"}
