#!/usr/bin/python
from __future__ import annotations

"""Knowledge Pack Models — Domain-specific data presets for the Knowledge Graph.

CONCEPT:KG-2.6 — Knowledge Packs

A KnowledgePack defines a set of actual data instances (Nodes and Edges) that can be
seeded into the Knowledge Graph as a cohesive bundle. While a `SchemaPack` defines the
*structure* (which ontology types are allowed), a `KnowledgePack` provides the *content*
(e.g., specific papers, repositories, entities).

Knowledge Packs use deterministic ID generation to ensure idempotent imports, allowing
them to be easily shared, versioned, and injected into different environments.

Usage::

    from agent_utilities.models.knowledge_pack import KnowledgePackBundle, KnowledgePackImporter

    # Load from YAML
    bundle = KnowledgePackImporter.load("presets/finance/awesome-deep-trading-pack.yaml")

    # Seed into KG
    counts = KnowledgePackImporter.seed_into_kg(bundle, engine)

"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


def generate_deterministic_id(node_type: str, identifier: str) -> str:
    """Generate a deterministic ID for a node based on its type and a unique identifier.

    Args:
        node_type: The string representation of the RegistryNodeType.
        identifier: A unique string for this node (e.g., URL, exact title, or name).

    Returns:
        A deterministic ID string prefixed with `kp:`.
    """
    hash_input = f"{node_type}:{identifier}".encode()
    hash_hex = hashlib.sha256(hash_input).hexdigest()[:16]
    return f"kp:{hash_hex}"


@dataclass
class KnowledgePackBundle:
    """A bundle of Knowledge Graph nodes and edges for portable distribution.

    Attributes:
        name: The human-readable name of the pack (e.g., 'awesome-deep-trading').
        domain: The domain this pack belongs to (e.g., 'finance').
        version: The version of the pack.
        description: A brief description of the pack contents.
        nodes: A list of dictionary representations of nodes to upsert.
        edges: A list of dictionary representations of edges to upsert.
        metadata: Additional metadata about the pack.
    """

    name: str
    domain: str = "general"
    version: str = "1.0"
    description: str = ""
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON/YAML-compatible dict."""
        return {
            "name": self.name,
            "domain": self.domain,
            "version": self.version,
            "description": self.description,
            "metadata": self.metadata,
            "nodes": self.nodes,
            "edges": self.edges,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgePackBundle:
        """Deserialize from a dict."""
        return cls(
            name=data.get("name", "unnamed-pack"),
            domain=data.get("domain", "general"),
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
            nodes=data.get("nodes", []),
            edges=data.get("edges", []),
        )


class KnowledgePackExporter:
    """Export Knowledge Packs to human-readable formats."""

    @staticmethod
    def to_yaml(bundle: KnowledgePackBundle, path: str | Path | None = None) -> str:
        """Serialize a bundle to YAML.

        Args:
            bundle: The KnowledgePackBundle to serialize.
            path: Optional file path to write the output.

        Returns:
            The YAML string.
        """
        try:
            import yaml
        except ImportError:
            logger.warning(
                "[KG-2.7] PyYAML not available — falling back to JSON export."
            )
            return KnowledgePackExporter.to_json(bundle, path)

        data = bundle.to_dict()
        yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False)

        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml_str)
            logger.info("[KG-2.7] Knowledge Pack exported to %s (YAML)", path)

        return yaml_str

    @staticmethod
    def to_json(bundle: KnowledgePackBundle, path: str | Path | None = None) -> str:
        """Serialize a bundle to JSON.

        Args:
            bundle: The KnowledgePackBundle to serialize.
            path: Optional file path to write the output.

        Returns:
            The JSON string.
        """
        data = bundle.to_dict()
        json_str = json.dumps(data, indent=2, default=str)

        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json_str)
            logger.info("[KG-2.7] Knowledge Pack exported to %s (JSON)", path)

        return json_str


def _searxng_connector_for(query: str) -> Any | None:
    """Build a KG-2.59 ``mcp_tool`` searxng source for one query, if configured.

    Reuse-policy seam: external web retrieval goes through the ``mcp_tool``
    ``searxng-search`` preset when a searxng server is present in the
    workspace ``mcp_config.json``. Returns ``None`` when no searxng server is
    configured (or the connector stack is unavailable), in which case callers
    fall back to the zero-infra crawl4ai path.
    """
    try:
        from agent_utilities.protocols.source_connectors.connectors.mcp_package import (
            _load_mcp_config,
        )
        from agent_utilities.protocols.source_connectors.registry import (
            build_connector,
        )

        servers = _load_mcp_config()
        if not any(name in servers for name in ("searxng", "searxng-mcp")):
            return None
        return build_connector(
            "mcp_tool", {"preset": "searxng-search", "params": {"query": query}}
        )
    except Exception as exc:  # noqa: BLE001 — unavailable source ≠ hydration failure
        logger.debug("[KG-2.7] searxng mcp_tool source unavailable: %s", exc)
        return None


class KnowledgePackHydrator:
    """Hydrates Knowledge Pack nodes by extracting content from their URLs."""

    @staticmethod
    def _hydrate_via_searxng(
        urls: list[str], node_map: dict[str, dict[str, Any]]
    ) -> list[str]:
        """Hydrate web URLs through the searxng-mcp ``mcp_tool`` source.

        For each URL, queries the configured searxng server (KG-2.59 preset)
        and uses the matching result's content. Returns the URLs that were
        *not* hydrated — because no searxng server is configured, the call
        failed, or no result matched — so the caller can fall back to the
        zero-infra crawl4ai path for exactly those.
        """
        remaining: list[str] = []
        for url in urls:
            connector = _searxng_connector_for(url)
            if connector is None:
                remaining.append(url)
                continue
            try:
                match = next(
                    (
                        doc
                        for doc in connector.load()
                        if doc.id == url and doc.text.strip()
                    ),
                    None,
                )
            except Exception as exc:  # noqa: BLE001 — fall back per URL
                logger.warning(
                    "[KG-2.7] searxng-mcp fetch failed for %s (%s); "
                    "falling back to crawl4ai.",
                    url,
                    exc,
                )
                remaining.append(url)
                continue
            if match is not None:
                node_map[url]["content"] = match.text.strip()
            else:
                remaining.append(url)
        return remaining

    @staticmethod
    async def hydrate(bundle: KnowledgePackBundle) -> None:
        """Hydrate nodes with content extracted from their URLs.

        PDFs are converted using pymupdf4llm. Web pages go through the
        searxng-mcp ``mcp_tool`` source when a searxng server is configured
        (KG-2.59 reuse policy), then Crawl4AI as the zero-infra fallback,
        then plain requests as the last resort.

        Args:
            bundle: The KnowledgePackBundle to hydrate.
        """
        urls_to_crawl = []
        node_map = {}

        # First pass: Extract PDF content directly, and gather web URLs
        for node in bundle.nodes:
            url = node.get("url")
            if not url:
                continue

            # We'll use the ID as a safe way to map results back to nodes
            node.get("id")
            node_map[url] = node

            if url.lower().endswith(".pdf"):
                try:
                    logger.info("[KG-2.7] Extracting PDF content for %s", url)
                    import tempfile

                    import pymupdf4llm
                    import requests

                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False
                    ) as tmp:
                        resp = requests.get(url, timeout=30)
                        resp.raise_for_status()
                        tmp.write(resp.content)
                        tmp_path = tmp.name

                    md_text = pymupdf4llm.to_markdown(tmp_path)
                    node["content"] = md_text

                    Path(tmp_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning("[KG-2.7] Failed to extract PDF %s: %s", url, e)
            else:
                urls_to_crawl.append(url)

        if not urls_to_crawl:
            return

        # Second pass: when a searxng server is configured, route web
        # retrieval through the mcp_tool source preset (KG-2.59 reuse
        # policy); crawl4ai remains the zero-infra fallback below.
        urls_to_crawl = KnowledgePackHydrator._hydrate_via_searxng(
            urls_to_crawl, node_map
        )
        if not urls_to_crawl:
            return

        # Final fallback pass: extract web content using crawl4ai (zero-infra)
        try:
            from crawl4ai import (
                AsyncWebCrawler,
                BrowserConfig,
                CacheMode,
                CrawlerRunConfig,
            )

            browser_config = BrowserConfig(
                headless=True,
                verbose=False,
                extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
            )
            crawl_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                magic=True,
                wait_until="networkidle",
                page_timeout=30000,
            )

            logger.info(
                "[KG-2.7] Extracting web content for %d URLs", len(urls_to_crawl)
            )
            async with AsyncWebCrawler(config=browser_config) as crawler:
                results = await crawler.arun_many(
                    urls=urls_to_crawl, config=crawl_config
                )
                for result in results:
                    if result.success:
                        # Extract markdown using crawl4ai's preferred format
                        md = ""
                        if hasattr(result, "markdown_v2") and result.markdown_v2:
                            md = getattr(
                                result.markdown_v2,
                                "raw_markdown",
                                str(result.markdown_v2),
                            )
                        elif hasattr(result, "markdown") and hasattr(
                            result.markdown, "raw_markdown"
                        ):
                            md = result.markdown.raw_markdown
                        else:
                            md = str(getattr(result, "markdown", ""))

                        target_node = node_map.get(result.url)
                        if target_node:
                            target_node["content"] = md.strip()
                    else:
                        logger.warning(
                            "[KG-2.7] Failed to extract web content for %s: %s",
                            result.url,
                            result.error_message,
                        )

        except Exception as e:
            logger.warning(
                "[KG-2.7] crawl4ai/Playwright failed or not installed (%s). Falling back to basic requests for web content.",
                e,
            )
            import requests

            for url in urls_to_crawl:
                try:
                    resp = requests.get(url, timeout=15)
                    resp.raise_for_status()
                    target_node = node_map.get(url)
                    if target_node:
                        # Fallback stores raw HTML if no markdown parser is available
                        target_node["content"] = resp.text
                except Exception as ex:
                    logger.warning("[KG-2.7] Failed to fallback fetch %s: %s", url, ex)


class KnowledgePackImporter:
    @staticmethod
    def load(path: str | Path) -> KnowledgePackBundle:
        """Load a Knowledge Pack from a YAML or JSON file.

        Args:
            path: Path to the bundle file (.yaml, .yml, or .json).

        Returns:
            A deserialized KnowledgePackBundle.

        Raises:
            FileNotFoundError: If the bundle file doesn't exist.
            ValueError: If the file format is unsupported.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge Pack file not found: {path}")

        content = path.read_text()

        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml

                data = yaml.safe_load(content) or {}
            except ImportError:
                raise ValueError(
                    "PyYAML is required to load YAML bundles. "
                    "Install with: pip install pyyaml"
                ) from None
        elif path.suffix == ".json":
            data = json.loads(content)
        else:
            raise ValueError(
                f"Unsupported format: {path.suffix}. Use .yaml, .yml, or .json"
            )

        bundle = KnowledgePackBundle.from_dict(data)
        logger.info(
            "[KG-2.7] Loaded Knowledge Pack '%s' (%d nodes, %d edges)",
            bundle.name,
            len(bundle.nodes),
            len(bundle.edges),
        )
        return bundle

    @staticmethod
    def seed_into_kg(
        bundle: KnowledgePackBundle,
        engine: IntelligenceGraphEngine,
    ) -> dict[str, int]:
        """Seed a Knowledge Pack's contents into the Knowledge Graph.

        Uses the deterministic IDs present in the nodes to perform idempotent upserts.

        Args:
            bundle: The bundle to seed.
            engine: The IntelligenceGraphEngine instance.

        Returns:
            Dict with counts: ``nodes_seeded``, ``edges_seeded``, ``errors``.
        """
        counts = {"nodes_seeded": 0, "edges_seeded": 0, "errors": 0}

        # Seed Nodes
        for node_data in bundle.nodes:
            try:
                node_type = node_data.get("type")
                node_id = node_data.get("id")

                if not node_type or not node_id:
                    logger.warning(
                        "[KG-2.7] Skipping node missing type or id: %s", node_data
                    )
                    counts["errors"] += 1
                    continue

                engine._upsert_node(node_type, node_id, node_data)
                counts["nodes_seeded"] += 1
            except Exception as e:
                logger.warning(
                    "[KG-2.7] Failed to seed node %s: %s", node_data.get("id"), e
                )
                counts["errors"] += 1

        # Seed Edges
        for edge_data in bundle.edges:
            try:
                source = edge_data.get("source")
                target = edge_data.get("target")
                edge_type = edge_data.get("type")

                if not source or not target or not edge_type:
                    logger.warning(
                        "[KG-2.7] Skipping edge missing source/target/type: %s",
                        edge_data,
                    )
                    counts["errors"] += 1
                    continue

                engine.link_nodes(
                    source, target, edge_type, edge_data.get("metadata", {})
                )
                counts["edges_seeded"] += 1
            except Exception as e:
                logger.warning(
                    "[KG-2.7] Failed to seed edge %s -> %s: %s",
                    edge_data.get("source"),
                    edge_data.get("target"),
                    e,
                )
                counts["errors"] += 1

        logger.info(
            "[KG-2.7] Knowledge Pack seeding complete for '%s': %d nodes, %d edges, %d errors",
            bundle.name,
            counts["nodes_seeded"],
            counts["edges_seeded"],
            counts["errors"],
        )
        return counts
