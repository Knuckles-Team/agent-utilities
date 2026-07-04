"""graph_secret MCP tool — engine-backed encrypted secret store (CONCEPT:AU-OS.identity.encrypted-secret-store).

Thin wrapper over :class:`agent_utilities.security.secrets_client.SecretsClient`
(the one core). Secret set/get/list/delete reach the durable, engine-encrypted
``__secrets__`` graph (or the enterprise Vault backend) through the same client
that every Python import uses, so this surface never re-implements storage. The
REST twin is ``/graph/secret`` (auto-mounted from ``ACTION_TOOL_ROUTES`` in
``kg_server.py``); both dispatch into the same client so they never drift.

Mutations (``set``/``delete``) are governed by the ActionPolicy gate
(``secret.set`` / ``secret.delete``, approval_required); reads (``get``/``list``)
are not gated, mirroring the read posture of the other graph surfaces. Secret
VALUES are never returned by ``list`` and are only returned by ``get`` to the
authorized caller.

CONCEPT:AU-OS.identity.encrypted-secret-store — Engine-backed encrypted secret store (MCP + REST surfaces)
"""

from __future__ import annotations

import json

from pydantic import Field

from agent_utilities.mcp import kg_server


def _gate(kind: str, target: str, reason: str) -> tuple[bool, dict]:
    """Run the ActionPolicy gate for a secret mutation. CONCEPT:AU-OS.identity.encrypted-secret-store."""
    from agent_utilities.orchestration.action_policy import (
        ActionRequest,
        get_action_policy,
    )

    decision = get_action_policy(kg_server._get_engine()).decide(
        ActionRequest(
            kind=kind,
            target=target,
            source="mcp",
            reason=reason or f"{kind} on secret store",
        )
    )
    info = {
        "decision": decision.decision,
        "tier": decision.tier,
        "reason": decision.reason,
        "approval_id": decision.approval_id,
    }
    return decision.allowed, info


def register_secret_tools(mcp):
    """Register the ``graph_secret`` tool onto the MCP server. CONCEPT:AU-OS.identity.encrypted-secret-store"""

    @mcp.tool(
        name="graph_secret",
        description=(
            "Manage secrets (CONCEPT:AU-OS.identity.encrypted-secret-store) in the durable, engine-encrypted "
            "__secrets__ store (secret VALUES are sealed by the engine's "
            "encryption-at-rest; key NAMES + metadata stay queryable). Actions: "
            "'set' (key+value [+metadata] → store/overwrite, GOVERNED by "
            "ActionPolicy secret.set), 'get' (key → value or null), 'list' (→ key "
            "names only, never values), 'delete' (key → removed bool, GOVERNED by "
            "ActionPolicy secret.delete). The enterprise OpenBao/Vault backend is "
            "used transparently when configured."
        ),
        tags=["graph-os", "security", "secret"],
    )
    async def graph_secret(
        action: str = Field(
            default="get",
            description="set | get | list | delete",
        ),
        key: str = Field(default="", description="Secret key (set/get/delete)."),
        value: str = Field(default="", description="Secret value (set)."),
        metadata: dict | None = Field(
            default=None, description="Optional non-secret metadata (set)."
        ),
        reason: str = Field(
            default="", description="Why this mutation is happening (audit trail)."
        ),
    ) -> str:
        from agent_utilities.security.secrets_client import create_secrets_client

        client = create_secrets_client()

        if action == "set":
            if not key:
                return json.dumps({"error": "set requires key"})
            allowed, info = _gate("secret.set", key, reason)
            if not allowed:
                return json.dumps({"error": "policy_denied", **info})
            client.set(key, value, **(metadata or {}))
            return json.dumps({"status": "stored", "key": key, **info})
        if action == "get":
            if not key:
                return json.dumps({"error": "get requires key"})
            return json.dumps({"key": key, "value": client.get(key)})
        if action == "list":
            return json.dumps({"keys": client.list_keys()})
        if action == "delete":
            if not key:
                return json.dumps({"error": "delete requires key"})
            allowed, info = _gate("secret.delete", key, reason)
            if not allowed:
                return json.dumps({"error": "policy_denied", **info})
            return json.dumps({"deleted": client.delete(key), "key": key, **info})
        return json.dumps({"error": f"unknown action: {action}"})

    kg_server.REGISTERED_TOOLS["graph_secret"] = graph_secret
