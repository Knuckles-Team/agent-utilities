"""CONCEPT:X1 -- the tiny-profile local-process mint is least-privilege.

``request_identity._mint_local_process_authority`` is the single chokepoint for
credential-free authority: it mints only the grants in
``_LOCAL_PROCESS_GRANTS`` and refuses any that would carry ``kg:admin``/``*``.
"""

from __future__ import annotations

import pytest


class TestLocalProcessGrantChokepoint:
    """CONCEPT:X1: ``_mint_local_process_authority`` is the single chokepoint
    for credential-free authority; it mints only table-defined grants and
    refuses any that would project ``kg:admin`` or ``*``."""

    def test_every_grant_is_free_of_admin_scopes(self):
        from agent_utilities.security import request_identity as ri

        for grant in ri.LocalProcessGrant:
            session = ri._mint_local_process_authority(grant)
            assert not session.scopes & {"kg:admin", "*"}, grant

    @pytest.mark.parametrize("role", ["kg:admin", "*"])
    def test_chokepoint_refuses_an_admin_grant(self, monkeypatch, role):
        from agent_utilities.security import request_identity as ri

        widened = dict(ri._LOCAL_PROCESS_GRANTS)
        widened[ri.LocalProcessGrant.AMBIENT] = ("graph-os:local-process", (role,))
        monkeypatch.setattr(ri, "_LOCAL_PROCESS_GRANTS", widened)
        # ``*`` is outside the served allowlist, so widen that too: the
        # chokepoint guard, not the allowlist, must be what refuses it.
        monkeypatch.setattr(
            ri, "_GRAPH_AUTH_SCOPES", ri._GRAPH_AUTH_SCOPES | frozenset({"*"})
        )
        with pytest.raises(PermissionError, match="local-process authority"):
            ri.mint_local_process_session()
