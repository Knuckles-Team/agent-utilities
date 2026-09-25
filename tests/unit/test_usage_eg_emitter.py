"""The AU usage emitter writes only EG metadata facts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.usage.eg_emitter import UsageFactEmitter
from agent_utilities.usage.models import ParsedSessionBundle, UsageEvent, UsageSession

KEY = b"stable-test-usage-identity-key-00000000"


class FakeNodes:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    async def create_if_absent(self, node_id: str, properties: dict) -> bool:
        if node_id in self.rows:
            return False
        self.rows[node_id] = properties
        return True

    async def properties(self, node_id: str) -> dict | None:
        return self.rows.get(node_id)


def _client(tenant: str = "tenant-a") -> SimpleNamespace:
    claims = {
        "tenant": tenant,
        "principal": "service:usage",
        "agent_id": "service:usage",
    }
    return SimpleNamespace(
        nodes=FakeNodes(),
        _auth_secret="fixture-secret",
        _effective_verified_context=lambda: claims,
    )


def _bundle(tenant: str = "tenant-a") -> ParsedSessionBundle:
    return ParsedSessionBundle(
        session=UsageSession(
            id="source-run", tenant_id=tenant, started_at="2026-09-25T00:00:00Z"
        ),
        usage_events=[
            UsageEvent(
                session_id="source-run",
                tenant_id=tenant,
                origin="runtime",
                model="provider/model",
                input_tokens=12,
                cost_usd=0.012345,
            )
        ],
    )


@pytest.mark.asyncio
async def test_emits_private_metadata_once() -> None:
    client = _client()
    emitter = UsageFactEmitter(client, KEY)
    assert await emitter.emit_bundle(_bundle()) == 1
    assert await emitter.emit_bundle(_bundle()) == 0
    row = next(iter(client.nodes.rows.values()))
    assert row["input_tokens"] == 12
    assert row["cost_microusd"] == 12345
    assert row["model_ref"].startswith("pref_model_")
    assert "source-run" not in str(row)
    assert "provider/model" not in str(row)


@pytest.mark.asyncio
async def test_rejects_tenant_spoofing() -> None:
    client = _client()
    with pytest.raises(PermissionError, match="signed authority"):
        await UsageFactEmitter(client, KEY).emit_bundle(_bundle("tenant-b"))
    assert client.nodes.rows == {}
