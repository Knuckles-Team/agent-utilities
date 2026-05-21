#!/usr/bin/python
"""Ontology Loader — Remote OWL Import Resolver.

CONCEPT:KG-2.6 — Modular Ontology Federation

Resolves owl:imports declarations in TTL files, fetching remote ontologies
from HTTP URLs or SPARQL endpoints and merging them into the local rdflib
graph. Enables the "inherit from central, extend locally" enterprise pattern.

Supports:
- File-based imports (file:// URIs)
- HTTP/HTTPS remote imports
- TTL-based caching with configurable TTL
- Recursive import resolution
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default cache directory
_DEFAULT_CACHE_DIR = (
    Path.home() / ".local" / "share" / "agent-utilities" / "ontology_cache"
)
_DEFAULT_CACHE_TTL_SECONDS = 3600  # 1 hour


class OntologyLoader:
    """Resolves and loads modular ontologies with owl:imports support.

    CONCEPT:KG-2.6 — Enterprise Ontology Federation

    Enables hierarchical ontology inheritance where:
    - A central enterprise ontology is served at a known URL
    - Domain deployments import it via owl:imports
    - Local customizations extend/specialize the imported classes
    - The loader caches remote ontologies for offline resilience

    Example::

        loader = OntologyLoader()
        graph = loader.load_with_imports("ontology.ttl")
        # graph now contains the main ontology + all imported ontologies
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        cache_ttl_seconds: int = _DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        """Initialize the loader.

        Args:
            cache_dir: Directory for caching remote ontologies.
            cache_ttl_seconds: Cache expiry time in seconds.
        """
        self._cache_dir = Path(cache_dir or _DEFAULT_CACHE_DIR)
        self._cache_ttl = cache_ttl_seconds
        self._loaded_uris: set[str] = set()

    def load_with_imports(
        self,
        ontology_path: str | Path,
        base_graph: Any | None = None,
    ) -> Any:
        """Load an ontology and recursively resolve all owl:imports.

        Args:
            ontology_path: Path to the main ontology TTL file.
            base_graph: Optional existing rdflib.Graph to merge into.

        Returns:
            An rdflib.Graph with the main ontology and all imports merged.
        """
        import rdflib

        graph = base_graph or rdflib.Graph()
        path = Path(ontology_path)

        if not path.exists():
            logger.error("Ontology file not found: %s", path)
            return graph

        # Parse the main ontology
        graph.parse(str(path), format="turtle")
        self._loaded_uris.add(str(path.resolve()))

        # Resolve imports
        self._resolve_imports(graph, path.parent)

        logger.info(
            "Loaded ontology with imports: %d triples from %d sources",
            len(graph),
            len(self._loaded_uris),
        )
        return graph

    def _resolve_imports(self, graph: Any, base_dir: Path) -> None:
        """Recursively resolve owl:imports declarations.

        Args:
            graph: rdflib.Graph to scan for imports and merge into.
            base_dir: Base directory for resolving relative file paths.
        """
        import rdflib

        OWL = rdflib.Namespace("http://www.w3.org/2002/07/owl#")

        # Find all owl:imports triples
        imports = list(graph.objects(predicate=OWL.imports))

        for import_uri in imports:
            uri_str = str(import_uri)
            if uri_str in self._loaded_uris:
                continue  # Skip already loaded

            self._loaded_uris.add(uri_str)

            try:
                imported_data = self._fetch_ontology(uri_str, base_dir)
                if imported_data:
                    # Parse into a temporary graph first
                    temp = rdflib.Graph()
                    temp.parse(data=imported_data, format="turtle")

                    # Merge into main graph
                    for triple in temp:
                        graph.add(triple)

                    # Recursively resolve imports from the imported ontology
                    self._resolve_imports(graph, base_dir)

                    logger.debug(
                        "Imported ontology: %s (%d triples)", uri_str, len(temp)
                    )
            except Exception as e:
                logger.warning("Failed to import ontology %s: %s", uri_str, e)

    def _fetch_ontology(self, uri: str, base_dir: Path) -> str | None:
        """Fetch ontology content from a URI.

        Tries in order:
        1. Local file (if URI maps to a local path)
        2. Cache (if not expired)
        3. HTTP fetch (with cache write)

        Args:
            uri: The ontology URI to fetch.
            base_dir: Base directory for resolving file paths.

        Returns:
            Ontology content as string, or None if unavailable.
        """
        # Try local file first — map URI to local filename
        local_path = self._uri_to_local_path(uri, base_dir)
        if local_path and local_path.exists():
            logger.debug("Loading import from local file: %s", local_path)
            return local_path.read_text(encoding="utf-8")

        # Check cache
        cached = self._read_cache(uri)
        if cached is not None:
            logger.debug("Loading import from cache: %s", uri)
            return cached

        # HTTP fetch for remote URIs
        if uri.startswith("http://") or uri.startswith("https://"):
            return self._fetch_remote(uri)

        logger.warning("Cannot resolve import URI: %s", uri)
        return None

    def _uri_to_local_path(self, uri: str, base_dir: Path) -> Path | None:
        """Map a URI to a potential local file path.

        Handles patterns like:
        - http://knuckles.team/kg/enterprise → ontology_enterprise.ttl
        - http://knuckles.team/kg → ontology.ttl
        - file:///path/to/file.ttl → /path/to/file.ttl

        Args:
            uri: The URI to resolve.
            base_dir: Base directory for relative paths.

        Returns:
            Path if a local file mapping exists, None otherwise.
        """
        if uri.startswith("file://"):
            return Path(uri.replace("file://", ""))

        # Map http://knuckles.team/kg/X to ontology_X.ttl
        if "knuckles.team/kg" in uri:
            parts = uri.split("knuckles.team/kg")
            if len(parts) > 1:
                suffix = parts[1].strip("/")
                if suffix:
                    candidate = base_dir / f"ontology_{suffix}.ttl"
                else:
                    candidate = base_dir / "ontology.ttl"
                return candidate

        return None

    def _fetch_remote(self, url: str) -> str | None:
        """Fetch ontology from an HTTP URL.

        Args:
            url: HTTP/HTTPS URL to fetch.

        Returns:
            Content string, or None on failure.
        """
        try:
            import requests

            response = requests.get(
                url,
                headers={"Accept": "text/turtle, application/rdf+xml"},
                timeout=15,
            )
            response.raise_for_status()
            content = response.text

            # Write to cache
            self._write_cache(url, content)

            logger.info("Fetched remote ontology: %s (%d bytes)", url, len(content))
            return content
        except Exception as e:
            logger.warning("Failed to fetch remote ontology %s: %s", url, e)
            return None

    def _cache_path(self, uri: str) -> Path:
        """Generate a cache file path for a URI."""
        uri_hash = hashlib.sha256(uri.encode()).hexdigest()[:16]
        return self._cache_dir / f"ontology_{uri_hash}.ttl"

    def _read_cache(self, uri: str) -> str | None:
        """Read ontology from cache if not expired."""
        cache_file = self._cache_path(uri)
        if not cache_file.exists():
            return None

        age = time.time() - cache_file.stat().st_mtime
        if age > self._cache_ttl:
            logger.debug("Cache expired for %s (age: %.0fs)", uri, age)
            return None

        return cache_file.read_text(encoding="utf-8")

    def _write_cache(self, uri: str, content: str) -> None:
        """Write ontology to cache."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self._cache_path(uri)
        cache_file.write_text(content, encoding="utf-8")
        logger.debug("Cached ontology: %s → %s", uri, cache_file)

    def clear_cache(self) -> int:
        """Clear all cached ontologies.

        Returns:
            Number of cache files removed.
        """
        if not self._cache_dir.exists():
            return 0

        count = 0
        for f in self._cache_dir.glob("ontology_*.ttl"):
            f.unlink()
            count += 1

        logger.info("Cleared %d cached ontologies", count)
        return count
