"""Deterministic, zero-LLM social/text entity extraction.

CONCEPT:AU-KG.ingest.deterministic-social-entity-mining.

Mines a source platform's OWN structured entity metadata (hashtags, @-mentions,
outbound URLs) directly from an already-fetched record — no LLM call, no extra
network round trip, no cost. This is the free-first stage a connector ingesting
social/text-platform content should run BEFORE any LLM-based enrichment of the
same record: cheaper, deterministic, reproducible, and auditable. Complements
the staged deterministic-write -> LLM-enrich pipeline
(``knowledge_graph/ingestion/staged_pipeline.py``) by giving the deterministic
stage something real to extract for this content shape, rather than that stage
only doing structural writes.

Two connector-agnostic entrypoints:

* :func:`extract_structured_entities` — schema-defensive extraction of
  hashtags/mentions/urls from a record's own nested ``entities``-shaped
  metadata (the common X/Twitter/Mastodon-style shape), tolerant of the exact
  upstream API-version differences a raw payload carries (a v1.1 ``legacy``
  wrapper vs a v2 top-level ``entities``, ``tag`` vs ``text`` for hashtags,
  ``screen_name`` vs ``username`` for mentions, ``expanded_url`` vs ``url`` for
  links) via the shared dotted-path digger so a connector needs no per-shape
  branching of its own.
* :func:`resolve_known_tools` — a curated, exact + registered-suffix
  domain -> tool lookup that turns a document's own outbound links into named
  ``:Tool`` references without any model call: a cheap, high-precision
  resolver feeding the ontology.

:func:`to_kg_rows` renders both into the ``{"id", "node_type", ...}`` /
``{"source", "target", "relationship"}`` row shape
``memory.native_ingest.ingest_entities`` accepts, stamping an
``extraction_stage="deterministic"`` provenance property and a ``confidence``
so this free stage's output stays distinguishable from any later LLM-derived
enrichment of the same document — the "each stage's output distinguishable in
provenance" half of cost-escalation staging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from ...etl.transforms import coalesce, dig

__all__ = [
    "KNOWN_TOOL_DOMAINS",
    "StructuredEntities",
    "extract_structured_entities",
    "resolve_known_tools",
    "to_kg_rows",
]


# Curated domain -> canonical tool/product name. Deliberately small and
# high-precision (an exact host match, or a registered-suffix match for
# subdomains such as ``xyz.github.io``) rather than an LLM guess — a genuinely
# free, deterministic entity-linking resolver feeding the ontology. Extend by
# adding an entry, not by routing tool detection through a model call.
KNOWN_TOOL_DOMAINS: dict[str, str] = {
    # Code & dev
    "github.com": "GitHub",
    "gitlab.com": "GitLab",
    "bitbucket.org": "Bitbucket",
    "stackoverflow.com": "Stack Overflow",
    "npmjs.com": "npm",
    "pypi.org": "PyPI",
    "crates.io": "crates.io",
    "docker.com": "Docker",
    "hub.docker.com": "Docker Hub",
    "vercel.com": "Vercel",
    "netlify.com": "Netlify",
    "supabase.com": "Supabase",
    "cloudflare.com": "Cloudflare",
    "aws.amazon.com": "AWS",
    "cloud.google.com": "Google Cloud",
    "azure.microsoft.com": "Azure",
    "linear.app": "Linear",
    "atlassian.com": "Atlassian",
    # AI / ML
    "huggingface.co": "Hugging Face",
    "arxiv.org": "arXiv",
    "openai.com": "OpenAI",
    "anthropic.com": "Anthropic",
    "perplexity.ai": "Perplexity",
    "replicate.com": "Replicate",
    "together.ai": "Together AI",
    "groq.com": "Groq",
    "mistral.ai": "Mistral",
    "cohere.com": "Cohere",
    "kaggle.com": "Kaggle",
    "wandb.ai": "Weights & Biases",
    # Design
    "figma.com": "Figma",
    "framer.com": "Framer",
    "canva.com": "Canva",
    # Productivity
    "notion.so": "Notion",
    "obsidian.md": "Obsidian",
    "airtable.com": "Airtable",
    "miro.com": "Miro",
    "loom.com": "Loom",
    # Media / content
    "youtube.com": "YouTube",
    "youtu.be": "YouTube",
    "substack.com": "Substack",
    "medium.com": "Medium",
    "news.ycombinator.com": "Hacker News",
    "dev.to": "dev.to",
    # Community
    "discord.com": "Discord",
    "discord.gg": "Discord",
    "slack.com": "Slack",
    "reddit.com": "Reddit",
    "t.me": "Telegram",
    # Finance / crypto
    "coinbase.com": "Coinbase",
    "binance.com": "Binance",
    "uniswap.org": "Uniswap",
    "opensea.io": "OpenSea",
    "etherscan.io": "Etherscan",
}


@dataclass
class StructuredEntities:
    """Zero-cost entities mined from a record's own structured metadata."""

    hashtags: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.hashtags or self.mentions or self.urls or self.tools)


def _domain(url: str) -> str | None:
    """Best-effort hostname extraction, ``www.``-stripped. ``None`` on a malformed URL."""
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return None
    if not host:
        return None
    return host[4:] if host.startswith("www.") else host


def resolve_known_tools(urls: list[str]) -> list[str]:
    """Resolve a document's outbound links to known tool/product names.

    Exact host match first, then a registered-suffix match (``xyz.github.io``
    resolves via ``github.com``... only when explicitly registered — this does
    NOT do a blind ``endswith`` over every known domain, only over domains
    genuinely meant to match subdomains). Order-preserving, deduplicated.
    """
    seen: set[str] = set()
    tools: list[str] = []
    for url in urls:
        host = _domain(url)
        if not host:
            continue
        name = KNOWN_TOOL_DOMAINS.get(host)
        if name is None and "." in host:
            # Subdomain of a known domain, e.g. ``docs.github.com`` -> GitHub.
            _, _, parent = host.partition(".")
            name = KNOWN_TOOL_DOMAINS.get(parent)
        if name and name not in seen:
            seen.add(name)
            tools.append(name)
    return tools


def extract_structured_entities(
    record: dict[str, Any] | None,
    *,
    exclude_url_hosts: tuple[str, ...] = (),
) -> StructuredEntities:
    """Zero-cost extraction from a record's own structured ``entities`` metadata.

    Schema-defensive over the two shapes a raw social/text-platform payload
    commonly carries: a v2-style top-level ``entities.*`` and a v1.1-style
    ``legacy.entities.*``, tolerating field-name drift within each (``tag`` vs
    ``text`` for hashtags, ``screen_name`` vs ``username`` for mentions,
    ``expanded_url`` vs ``url`` for links) via the shared dotted-path digger
    (:func:`agent_utilities.knowledge_graph.etl.transforms.dig`) rather than a
    bespoke per-connector traversal. Never raises — an absent/malformed
    ``entities`` block yields an empty :class:`StructuredEntities`.

    ``exclude_url_hosts`` filters out link-shortener/self-referential hosts
    (e.g. the source platform's own domain) that carry no external signal.
    """
    if not isinstance(record, dict):
        return StructuredEntities()

    hashtag_objs = (
        coalesce(record, "entities.hashtags", "legacy.entities.hashtags", default=[])
        or []
    )
    hashtags = sorted(
        {
            str(dig(h, "tag") or dig(h, "text") or "").strip().lower()
            for h in hashtag_objs
            if isinstance(h, dict)
        }
        - {""}
    )

    mention_objs = (
        coalesce(
            record,
            "entities.user_mentions",
            "legacy.entities.user_mentions",
            default=[],
        )
        or []
    )
    mentions = sorted(
        {
            str(dig(m, "screen_name") or dig(m, "username") or "").strip().lower()
            for m in mention_objs
            if isinstance(m, dict)
        }
        - {""}
    )

    url_objs = (
        coalesce(record, "entities.urls", "legacy.entities.urls", default=[]) or []
    )
    urls: list[str] = []
    seen_urls: set[str] = set()
    for u in url_objs:
        if not isinstance(u, dict):
            continue
        link = str(dig(u, "expanded_url") or dig(u, "url") or "").strip()
        if not link or link in seen_urls:
            continue
        host = _domain(link) or ""
        if any(host == ex or host.endswith(f".{ex}") for ex in exclude_url_hosts):
            continue
        seen_urls.add(link)
        urls.append(link)

    return StructuredEntities(
        hashtags=hashtags,
        mentions=mentions,
        urls=urls,
        tools=resolve_known_tools(urls),
    )


def to_kg_rows(
    entities: StructuredEntities,
    *,
    document_id: str,
    confidence: float = 1.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Render extracted entities as ``ingest_entities``-ready ``(nodes, edges)``.

    Every node is stamped ``extraction_stage="deterministic"`` (plus
    ``confidence``, default ``1.0`` — a directly-read structured field, not a
    model guess) so this free-first pass stays distinguishable in provenance
    from any later LLM-derived enrichment of the same document.
    """
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    def _stamped(node: dict[str, Any]) -> dict[str, Any]:
        node["extraction_stage"] = "deterministic"
        node["confidence"] = confidence
        return node

    for tag in entities.hashtags:
        node_id = f"hashtag:{tag}"
        nodes.append(_stamped({"id": node_id, "node_type": "Hashtag", "name": tag}))
        edges.append(
            {
                "source": document_id,
                "target": node_id,
                "relationship": "taggedWithHashtag",
            }
        )

    for handle in entities.mentions:
        node_id = f"mention:{handle}"
        nodes.append(_stamped({"id": node_id, "node_type": "Mention", "name": handle}))
        edges.append(
            {"source": document_id, "target": node_id, "relationship": "mentionsHandle"}
        )

    for tool in entities.tools:
        node_id = f"tool:{tool.lower().replace(' ', '-')}"
        nodes.append(_stamped({"id": node_id, "node_type": "Tool", "name": tool}))
        edges.append(
            {
                "source": document_id,
                "target": node_id,
                "relationship": "referencesTool",
            }
        )

    return nodes, edges
