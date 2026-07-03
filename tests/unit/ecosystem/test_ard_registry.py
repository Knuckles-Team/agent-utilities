"""ARD registry publish-side tests (CONCEPT:ECO-4.95 / OS-5.60).

Offline: the multiplexer catalog is injected as a pre-probed dict and the KG engine is a
fake, so no server is spawned and no graph is required.
"""

from __future__ import annotations

from agent_utilities.ecosystem import ard_registry as r
from agent_utilities.security import ard_signing

_CATALOG = {
    "portainer-agent": {
        "tools": [
            {"name": "list_containers", "description": "list docker containers"},
            {"name": "restart_container", "description": "restart a container"},
        ],
        "error": None,
    },
    "down-server": {"tools": [], "error": "unreachable"},
}


class _FakeEngine:
    def __init__(self, skills: list[dict]) -> None:
        self._skills = skills

    def query_cypher(self, _query: str, _params: dict | None = None) -> list[dict]:
        return self._skills


def test_build_catalog_emits_signed_server_entries() -> None:
    cat = r.build_ai_catalog(multiplexer=_CATALOG)
    assert cat["ardSpecVersion"]
    assert cat["publisherKey"]
    servers = [e for e in cat["resources"] if e["type"] == r.MEDIA_MCP_SERVER]
    assert len(servers) == 1  # the unreachable server is skipped
    entry = servers[0]
    assert entry["name"] == "portainer-agent"
    assert "portainer" in entry["tags"]
    assert entry["exampleQueries"]  # representative queries derived from synonyms/tools
    # The entry's signature verifies against the manifest's publisher key.
    unsigned = {k: v for k, v in entry.items() if k != "signature"}
    assert ard_signing.verify_datapoint(
        unsigned, entry["signature"], cat["publisherKey"]
    )


def test_build_catalog_includes_kg_skills_as_ai_skill() -> None:
    engine = _FakeEngine(
        [{"id": "deploy", "name": "Deploy", "description": "deploy a service"}]
    )
    cat = r.build_ai_catalog(multiplexer=_CATALOG, engine=engine)
    skills = [e for e in cat["resources"] if e["type"] == r.MEDIA_AI_SKILL]
    assert any(e["name"] == "Deploy" for e in skills)


def test_search_filters_by_media_type() -> None:
    engine = _FakeEngine(
        [{"id": "deploy", "name": "deploy containers", "description": "deploy"}]
    )
    # Only ai-skill requested → no mcp-server results even though the catalog has servers.
    out = r.ard_search(
        "deploy containers",
        types=[r.MEDIA_AI_SKILL],
        page_size=5,
        multiplexer=_CATALOG,
        engine=engine,
    )
    assert all(res["type"] == r.MEDIA_AI_SKILL for res in out["results"])
    assert any("deploy" in res["name"] for res in out["results"])
