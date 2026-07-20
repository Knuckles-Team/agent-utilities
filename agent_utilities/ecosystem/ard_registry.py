"""CONCEPT:AU-ECO.mcp.eco-serves-two-ard — Agentic Resource Discovery (ARD) registry (publish side).

ARD (a draft open spec from Hugging Face, Microsoft, Google, GoDaddy and others)
separates capability *discovery* from *execution*: instead of hardcoding an MCP URL,
an agent discovers tools/skills/agents at runtime through two artifacts —

* a **static manifest** ``ai-catalog.json`` served at ``/.well-known/ai-catalog.json``
  describing each resource (publisher, name, description, tags, example queries, media
  type, signature), and
* a **dynamic registry API** ``POST /search`` returning ranked results for an NL query
  with a media-type filter.

This module is the ONE core both serving surfaces call (the gateway REST router in
``server/routers/ard.py`` and the graph-os ``@mcp.custom_route`` mirror in
``mcp/kg_server.py``) — keeping them in lockstep per the surface-parity rule. We map our
existing fleet onto ARD's envelope rather than building anything new:

* every fleet MCP server (probed via the multiplexer catalog) → an
  ``application/mcp-server+json`` resource, with tags + example queries derived from
  :func:`derive_capability_synonyms`;
* every KG ``:Skill`` node → an ``application/ai-skill`` resource;
* ranking for ``/search`` reuses :meth:`MCPMultiplexer.discover_tools` (token overlap
  blended with KG semantic search) — the same engine ``find_tools`` rides.

Entries are Ed25519-signed (``security/ard_signing``, OS-5.60) so a consuming agent can
verify them against the manifest's ``publisherKey`` before trusting a capability.

The exact ARD JSON shape is intentionally quarantined to this module (and the consume
parser in ``connectors/ard.py``) so a draft-spec field rename is a one-file edit; the
assumed spec revision is stamped as ``ardSpecVersion``.
"""

from __future__ import annotations

import logging
import re
import socket
from typing import Any

from agent_utilities.core.config import setting
from agent_utilities.security import ard_signing

logger = logging.getLogger(__name__)

# ── ARD media types (the resource kinds we publish) ──────────────────────────
MEDIA_MCP_SERVER = "application/mcp-server+json"
MEDIA_MCP_SERVER_CARD = "application/mcp-server-card+json"
MEDIA_AI_SKILL = "application/ai-skill"
ALL_MEDIA_TYPES = (MEDIA_MCP_SERVER, MEDIA_MCP_SERVER_CARD, MEDIA_AI_SKILL)


def spec_version() -> str:
    """The ARD draft revision this generator targets (stamped into the manifest)."""
    return str(setting("ARD_SPEC_VERSION", default="draft-0"))


def publisher() -> dict[str, str]:
    """This registry's domain-anchored publisher identity (ARD verification)."""
    from agent_utilities.prompting.builder import load_identity

    ident = load_identity()
    domain = str(setting("ARD_PUBLISHER_DOMAIN", default=socket.gethostname()))
    return {
        "domain": domain,
        "name": ident.get("name", "agent-utilities"),
        "description": ident.get("description", "")[:280],
    }


# ── Multiplexer-backed fleet probing ─────────────────────────────────────────


def _probe_fleet(multiplexer: Any = None) -> dict[str, dict]:
    """Return the ``{server: {"tools": [...], "error": str|None}}`` fleet catalog.

    ``multiplexer`` may be a pre-probed catalog dict (tests/callers that hold one), a
    live :class:`MCPMultiplexer`, or ``None`` to build one from the fleet config — the
    same resolution :func:`knowledge_graph.core.source_sync._sync_fleet` uses.
    """
    if isinstance(multiplexer, dict):
        return multiplexer
    try:
        from ..knowledge_graph.core.source_sync import _resolve_fleet_config
        from ..mcp.multiplexer import MCPMultiplexer
        from ..protocols.source_connectors.connectors.mcp_package import _run_async
    except Exception as exc:  # noqa: BLE001 — multiplexer optional at import
        logger.debug("ard: multiplexer unavailable: %s", exc)
        return {}
    mux = multiplexer
    if mux is None:
        config_path = _resolve_fleet_config()
        if config_path is None:
            return {}
        try:
            mux = MCPMultiplexer(config_path)
        except Exception as exc:  # noqa: BLE001
            logger.debug("ard: multiplexer build failed: %s", exc)
            return {}
    try:
        return _run_async(mux.probe_catalog())
    except Exception as exc:  # noqa: BLE001 — probe is best-effort
        logger.debug("ard: probe failed: %s", exc)
        return {}


def _example_queries(server: str, tools: list[dict]) -> list[str]:
    """Representative NL queries for a server (ARD ``exampleQueries``).

    Built from the de-suffixed product synonyms + a couple of tool names so a
    consumer's NL search ("manage portainer containers") lands on the resource.
    """
    from ..knowledge_graph.core.source_sync import derive_capability_synonyms

    syns = derive_capability_synonyms(server)
    queries = [f"use {s}" for s in syns[:2]]
    for entry in tools[:2]:
        name = (entry.get("name") or "").replace("_", " ").strip()
        if name:
            queries.append(name)
    # De-dup, preserve order.
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def _server_entries(catalog: dict[str, dict], pub: dict[str, str]) -> list[dict]:
    """Map probed fleet servers → ARD ``application/mcp-server+json`` resources."""
    from ..knowledge_graph.core.source_sync import derive_capability_synonyms

    entries: list[dict] = []
    for server, info in (catalog or {}).items():
        if not isinstance(info, dict) or info.get("error"):
            continue
        tools = [t for t in (info.get("tools") or []) if isinstance(t, dict)]
        if not tools:
            continue
        tool_names = [t.get("name", "") for t in tools if t.get("name")]
        desc = f"MCP server '{server}' exposing {len(tool_names)} tools: " + ", ".join(
            tool_names[:8]
        )
        entries.append(
            _sign_entry(
                {
                    "id": f"mcp:{server}",
                    "type": MEDIA_MCP_SERVER,
                    "name": server,
                    "description": desc[:600],
                    "tags": derive_capability_synonyms(server),
                    "exampleQueries": _example_queries(server, tools),
                    "toolCount": len(tool_names),
                    "publisher": pub,
                }
            )
        )
    return entries


def _skill_entries(engine: Any, pub: dict[str, str], *, limit: int = 500) -> list[dict]:
    """Map KG ``:Skill`` nodes → ARD ``application/ai-skill`` resources."""
    eng = engine or _active_engine()
    if eng is None:
        return []
    try:
        rows = eng.query_cypher(
            "MATCH (s:Skill) WHERE s.id IS NOT NULL "
            "RETURN s.id AS id, s.name AS name, s.description AS description "
            f"LIMIT {int(limit)}"
        )
    except Exception as exc:  # noqa: BLE001 — KG cold/absent ⇒ no skill entries
        logger.debug("ard: skill query failed: %s", exc)
        return []
    entries: list[dict] = []
    for r in rows or []:
        if not isinstance(r, dict) or not r.get("id"):
            continue
        name = r.get("name") or str(r["id"])
        entries.append(
            _sign_entry(
                {
                    "id": f"skill:{r['id']}",
                    "type": MEDIA_AI_SKILL,
                    "name": name,
                    "description": (r.get("description") or "")[:600],
                    "tags": _tokens(name),
                    "publisher": pub,
                }
            )
        )
    return entries


def _active_engine() -> Any:
    try:
        from ..knowledge_graph.core.engine import IntelligenceGraphEngine

        return IntelligenceGraphEngine.get_active()
    except Exception:  # noqa: BLE001
        return None


def _tokens(text: str) -> list[str]:
    return sorted(
        {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 1}
    )


def _sign_entry(entry: dict) -> dict:
    """Attach an Ed25519 signature over the entry's canonical form (sans signature)."""
    sig = ard_signing.sign_datapoint(entry)
    if sig:
        entry["signature"] = sig
    return entry


# ── Public API: the two surfaces call these ──────────────────────────────────


def build_ai_catalog(*, multiplexer: Any = None, engine: Any = None) -> dict:
    """Build the ARD ``ai-catalog.json`` manifest for ``/.well-known/ai-catalog.json``.

    CONCEPT:AU-ECO.mcp.eco-serves-two-ard. Assembles fleet MCP servers + KG skills into ARD resources, signs
    each with our Ed25519 key, and stamps the publisher identity + public key so a
    consumer can verify. Degrades gracefully: an unreachable fleet or cold KG simply
    yields fewer resources, never an error.
    """
    pub = publisher()
    catalog = _probe_fleet(multiplexer)
    resources = _server_entries(catalog, pub) + _skill_entries(engine, pub)
    pub_key = ard_signing.public_key_b64()
    return {
        "ardSpecVersion": spec_version(),
        "publisher": pub,
        "publisherKey": pub_key,
        "signed": bool(pub_key and ard_signing.is_configured()),
        "generatedFrom": "agent-utilities",
        "resourceCount": len(resources),
        "resources": resources,
    }


def ard_search(
    query_text: str,
    *,
    types: list[str] | None = None,
    page_size: int = 5,
    multiplexer: Any = None,
    engine: Any = None,
) -> dict:
    """Rank discoverable resources for an NL query (ARD ``POST /search`` core).

    CONCEPT:AU-ECO.mcp.eco-serves-two-ard. Fleet tools are ranked by :meth:`MCPMultiplexer.discover_tools`
    (token overlap blended with KG semantic search — the same path ``find_tools`` uses)
    and surfaced as ``application/mcp-server+json``; KG skills are lexically matched and
    surfaced as ``application/ai-skill``. ``types`` filters by media type (ARD
    ``filter.type``); ``page_size`` bounds the result count.
    """
    wanted = set(types) if types else set(ALL_MEDIA_TYPES)
    page_size = max(1, int(page_size or 5))
    pub = publisher()
    results: list[dict] = []
    unavailable: dict[str, str] = {}

    if MEDIA_MCP_SERVER in wanted or MEDIA_MCP_SERVER_CARD in wanted:
        results.extend(
            _search_servers(query_text, page_size, multiplexer, pub, unavailable)
        )
    if MEDIA_AI_SKILL in wanted:
        results.extend(_search_skills(query_text, page_size, engine, pub))

    results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    return {
        "ardSpecVersion": spec_version(),
        "publisher": pub,
        "results": results[:page_size],
        "unavailable": unavailable,
    }


def _search_servers(
    query_text: str,
    page_size: int,
    multiplexer: Any,
    pub: dict[str, str],
    unavailable: dict[str, str],
) -> list[dict]:
    try:
        from ..knowledge_graph.core.source_sync import _resolve_fleet_config
        from ..mcp.multiplexer import MCPMultiplexer
        from ..protocols.source_connectors.connectors.mcp_package import _run_async
    except Exception:  # noqa: BLE001
        return []
    mux = multiplexer if not isinstance(multiplexer, dict) else None
    if mux is None:
        config_path = _resolve_fleet_config()
        if config_path is None:
            return []
        try:
            mux = MCPMultiplexer(config_path)
        except Exception:  # noqa: BLE001
            return []
    try:
        ranked = _run_async(mux.discover_tools(query_text, top_k=page_size * 2))
    except Exception as exc:  # noqa: BLE001
        unavailable["fleet"] = str(exc)
        return []
    unavailable.update(ranked.get("unavailable") or {})
    out: list[dict] = []
    for hit in ranked.get("results") or []:
        server = hit.get("server", "")
        tool = hit.get("tool", "")
        if not server or tool == "*":
            continue
        out.append(
            {
                "id": f"mcp:{server}:{tool}",
                "type": MEDIA_MCP_SERVER,
                "name": f"{server}.{tool}",
                "description": hit.get("description", "") or "",
                "score": float(hit.get("score", 0.0) or 0.0),
                "server": server,
                "tool": tool,
                "publisher": pub,
            }
        )
    return out


def _search_skills(
    query_text: str, page_size: int, engine: Any, pub: dict[str, str]
) -> list[dict]:
    entries = _skill_entries(engine, pub, limit=1000)
    q = set(_tokens(query_text))
    if not q:
        return []
    scored: list[dict] = []
    for e in entries:
        text = f"{e.get('name', '')} {e.get('description', '')}"
        overlap = len(q & set(_tokens(text)))
        if overlap <= 0:
            continue
        scored.append({**e, "score": overlap / max(len(q), 1)})
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:page_size]
