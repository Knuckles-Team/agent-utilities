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
  (``agent_utilities.graphos_client.delegated_auth``) — i.e. today's one
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

import pytest


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
        import agent_utilities.graphos_client.client_credentials as module

        importlib.reload(module)
        try:
            with pytest.raises(RuntimeError, match="unsupported value"):
                module._auth_mode()
            with pytest.raises(RuntimeError, match="unsupported value"):
                module.child_auth(None)
        finally:
            monkeypatch.delenv("MCP_CLIENT_AUTH", raising=False)
            importlib.reload(module)
