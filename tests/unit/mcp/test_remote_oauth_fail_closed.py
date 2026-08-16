"""Fail-closed regressions for the remote-HTTP-MCP browser-OAuth prerequisites
(U-11, U-41, U-43, U-44, U-45).

No browser-OAuth broker is implemented in this repo (verified by source
inspection: no ``TokenStorage``/``OAuthClientProvider`` consumer, no server
callback route, no per-principal token store — see the lane report for the
full trace). Per the hard constraint in this theme, a fleet-global process
must never own or replay a per-user browser OAuth grant, and the correct
outcome when a safe design isn't in place is to FAIL CLOSED rather than
invent a shared-credential shortcut.

These tests do not build that broker. They pin the boundary conditions that
make today's absence of the feature safe (rather than merely incomplete),
so a later change implementing U-43's broker cannot silently reintroduce the
exact shared-credential shortcut this theme rules out:

* ``MCP_CLIENT_AUTH`` (the fleet-global outbound child-auth selector) has no
  "oauth"/browser-flow value — selecting one fails closed with a clear
  config error, not silent anonymous fallback or an invented mapping onto
  the service-account grant (U-11 / U-41's "no static token, no legacy
  shortcut" requirement).
* The multiplexer's remote-child session/auth construction path never reads
  the per-request delegated-user identity context
  (``agent_utilities.mcp.delegated_auth``) — i.e. today's one
  ``ChildRuntime``/session per configured server (U-44's root cause) cannot
  yet be handed a per-user token even by accident, because no code path
  connects the two. If this test starts failing, someone has wired a user
  token into the fleet-global child/session maps WITHOUT the principal-scoped
  isolation U-44 requires, and that needs a deliberate design decision, not
  an incidental import.
* The local, single-user, loopback-based interactive OAuth helper
  (``agent_utilities.security.browser_auth``) is not imported by the
  multiplexer or the MCP server factory — U-43's exact fix is "do not reuse
  or weaken the local helper" for the multi-user serving-plane responsibility;
  this pins that it has not been reused.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

_MCP_PACKAGE_DIR = Path(__file__).resolve().parents[3] / "agent_utilities" / "mcp"


def _source(module_filename: str) -> str:
    path = _MCP_PACKAGE_DIR / module_filename
    assert path.is_file(), f"expected {path} to exist"
    return path.read_text(encoding="utf-8")


class TestNoOAuthShortcutInFleetGlobalClientAuth:
    """U-11 / U-41: selecting a browser-OAuth-shaped scheme fails closed."""

    @pytest.mark.parametrize(
        "attempted_value",
        ["oauth", "oauth-browser", "browser", "dcr", "authorization-code"],
    )
    def test_oauth_shaped_mcp_client_auth_value_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch, attempted_value: str
    ) -> None:
        monkeypatch.setenv("MCP_CLIENT_AUTH", attempted_value)
        import agent_utilities.mcp.client_credentials as module

        importlib.reload(module)
        try:
            with pytest.raises(RuntimeError, match="unsupported value"):
                module._auth_mode()
            with pytest.raises(RuntimeError, match="unsupported value"):
                module.child_auth(None)
        finally:
            monkeypatch.delenv("MCP_CLIENT_AUTH", raising=False)
            importlib.reload(module)


class TestFleetGlobalSessionsCannotYetCarryUserIdentity:
    """U-44 / U-45 canary: today's single fleet-global ChildRuntime/session per
    server cannot be handed a per-user browser-delegated token, because no
    code path connects the two. This is a structural fence, not a feature —
    it must fail loudly (forcing a conscious design decision referencing
    U-44's principal-scoped isolation requirement) the moment that changes."""

    def test_multiplexer_does_not_reference_delegated_user_identity(self) -> None:
        source = _source("multiplexer.py")
        for forbidden in (
            "delegated_auth",
            "get_user_token",
            "get_user_claims",
            "get_delegated_token",
        ):
            assert forbidden not in source, (
                f"multiplexer.py now references {forbidden!r} — a per-user delegated "
                "identity primitive reaching the fleet-global children/sessions maps. "
                "U-44 requires principal-scoped storage, principal-scoped discovered "
                "catalogs, and isolated remote MCP connections per operation before "
                "this is safe; do not wire delegated user identity into the existing "
                "server-global runtime without that isolation."
            )

    def test_server_factory_does_not_reference_delegated_user_identity_for_child_auth(
        self,
    ) -> None:
        source = _source("server_factory.py")
        # delegated_auth.py itself documents this module as the INBOUND
        # RFC 8693 exchange path for THIS server's own callers, not outbound
        # remote-child auth; server_factory legitimately wires the inbound
        # oidc-proxy auth type, so only assert on the outbound-shaped names.
        for forbidden in ("get_user_token", "get_delegated_token"):
            assert forbidden not in source, (
                f"server_factory.py now references {forbidden!r}; confirm this is "
                "inbound-session handling, not outbound remote-child OAuth reusing "
                "the caller's own delegated token (U-44/U-45)."
            )


class TestLocalInteractiveHelperIsNotReusedAsAServerBroker:
    """U-43 canary: the local loopback/copy-paste OAuth helper
    (`security/browser_auth.py`) is a process-local, single-shared-secret-slot
    tool. It must not be imported by the multiplexer or MCP server factory —
    that would be exactly the reuse-for-a-different-trust-model U-43 rejects."""

    @pytest.mark.parametrize("module_filename", ["multiplexer.py", "server_factory.py"])
    def test_browser_auth_helper_is_not_imported(self, module_filename: str) -> None:
        source = _source(module_filename)
        assert "browser_auth" not in source, (
            f"{module_filename} references security.browser_auth — U-43 requires a "
            "dedicated server-side broker (server callback route, encrypted "
            "principal-scoped storage, one-time expiring state/session records), "
            "not reuse of the local single-user interactive helper."
        )
