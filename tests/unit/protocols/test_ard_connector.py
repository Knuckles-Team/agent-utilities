"""ARD consume connector + _sync_ard live-path tests (CONCEPT:AU-ECO.connector.ingest-external-ard-registry / KG-2.188).

Offline: a ``fetch_fn`` returns a canned ``ai-catalog.json``, so no network or live
registry is needed. The live-path test drives the real connector through ``_sync_ard``
and asserts the typed KG entities it would write.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.protocols.source_connectors.registry import (
    discover,
    get_connector_class,
)
from agent_utilities.security import ard_signing


def _manifest(*, sign: bool = False, domain: str = "registry.test") -> str:
    pub = {"domain": domain, "name": "Peer"}
    entries = [
        {
            "id": "mcp:portainer",
            "type": "application/mcp-server+json",
            "name": "portainer",
            "description": "container management",
            "tags": ["portainer", "docker"],
            "publisher": pub,
        },
        {
            "id": "skill:deploy",
            "type": "application/ai-skill",
            "name": "Deploy",
            "description": "deploy a service",
            "tags": ["deploy"],
            "publisher": pub,
        },
    ]
    if sign:
        for e in entries:
            e["signature"] = ard_signing.sign_datapoint(e)
    return json.dumps(
        {
            "ardSpecVersion": "draft-0",
            "publisher": pub,
            "publisherKey": ard_signing.public_key_b64(),
            "resources": entries,
        }
    )


def test_ard_connector_is_discovered() -> None:
    discover()
    assert get_connector_class("ard") is not None


def test_connector_loads_resources_offline() -> None:
    cls = get_connector_class("ard")
    conn = cls(
        catalog_url="https://registry.test",
        verify=False,
        fetch_fn=lambda _u: _manifest(),
    )
    docs = list(conn.load())
    assert {d.metadata["ard_media_type"] for d in docs} == {
        "application/mcp-server+json",
        "application/ai-skill",
    }
    assert all(d.doc_type == "ard_resource" for d in docs)


def test_media_type_filter() -> None:
    cls = get_connector_class("ard")
    conn = cls(
        catalog_url="https://registry.test",
        media_types=["application/ai-skill"],
        verify=False,
        fetch_fn=lambda _u: _manifest(),
    )
    docs = list(conn.load())
    assert [d.id for d in docs] == ["skill:deploy"]


@pytest.mark.skipif(
    not ard_signing.signing_available(), reason="cryptography not installed"
)
def test_signature_verification_drops_bad_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cls = get_connector_class("ard")
    # Signed manifest whose publisher domain matches the catalog host → accepted.
    conn = cls(
        catalog_url="https://registry.test",
        verify=True,
        fetch_fn=lambda _u: _manifest(sign=True),
    )
    assert len(list(conn.load())) == 2
    assert conn.verify_failures == 0

    # Tamper: a signed entry whose body no longer matches its signature is dropped.
    def _tampered(_u: str) -> str:
        data = json.loads(_manifest(sign=True))
        data["resources"][0]["description"] = "mutated after signing"
        return json.dumps(data)

    conn2 = cls(catalog_url="https://registry.test", verify=True, fetch_fn=_tampered)
    docs = list(conn2.load())
    assert "mcp:portainer" not in [d.id for d in docs]
    assert conn2.verify_failures == 1


def test_require_signature_drops_unsigned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARD_REQUIRE_SIGNATURE", "true")
    cls = get_connector_class("ard")
    conn = cls(
        catalog_url="https://registry.test",
        verify=True,
        fetch_fn=lambda _u: _manifest(sign=False),
    )
    assert list(conn.load()) == []
    assert conn.verify_failures == 2


class _FakeEngine:
    """Captures ingest_external_batch + serves a None backend (watermarks are no-ops)."""

    backend = None

    def __init__(self) -> None:
        self.batches: list[tuple] = []

    def ingest_external_batch(self, domain, entities, relationships=None):
        self.batches.append((domain, entities, relationships or []))
        return {"written": len(entities)}


def test_sync_ard_live_path_writes_typed_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    """AU-P1-5: ``_sync_ard`` is envelope-native (CONCEPT:AU-KG.ingest.envelope-atomic-transaction)
    — one ``ingest_envelope`` call per resource/registry/capability entity, so
    ``engine.batches`` now holds one entry per entity instead of a single
    all-resources batch. Aggregate across every call; the underlying intent
    (typed nodes + registeredIn/providesCapability links) is unchanged.
    """
    from agent_utilities.knowledge_graph.core.source_sync import _sync_ard

    monkeypatch.setenv(
        "ARD_REGISTRIES",
        json.dumps(
            [{"name": "peer", "catalog_url": "https://registry.test", "verify": False}]
        ),
    )
    engine = _FakeEngine()
    res = _sync_ard(engine, mode="full", ids=None, client=lambda _u: _manifest())

    assert res["status"] == "ok"
    assert res["nodes_hydrated"] > 0
    assert engine.batches and all(domain == "ard" for domain, _, _ in engine.batches)
    entities = [e for _domain, es, _rels in engine.batches for e in es]
    labels = {e["type"] for e in entities}
    assert {"ResourceRegistry", "MCPServer", "Skill", "ServiceCapability"} <= labels
    rel_types = {r["type"] for _domain, _es, rels in engine.batches for r in rels}
    assert {"registeredIn", "providesCapability"} <= rel_types
