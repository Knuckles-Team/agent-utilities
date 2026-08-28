#!/usr/bin/python
from __future__ import annotations

"""Skill-Graph Distiller — KG subgraph → packageable reference tree + manifest.

CONCEPT:AU-AHE.optimization.physical-distillation-engine — Physical Knowledge Distillation (read side).

The companion of ``physical_distiller.py``: where that module writes *evolved
skill/tool/prompt adaptations* back to the filesystem, this module **reads** a
coherent slice of the epistemic Knowledge Graph (e.g. "everything about
ServiceNow") and materialises it as a neutral ``reference/`` markdown tree plus
a ``kg_manifest.json`` provenance record.

That tree is deliberately *format-agnostic*: it is exactly what
``skill-graph-builder`` (``generate_skill.py``) already consumes as a "local
markdown directory" source, so the existing TOC/SKILL.md generator turns it into
a versioned, distributable skill-graph with **no** changes to that generator.
The ``kg_manifest.json`` makes the package round-trippable — another KG can
re-ingest it and dedup-merge (see ``deduplicator.py``).

Repo-boundary note: this module knows nothing about the SKILL.md format. The
seam mirrors how ``generate_skill.py`` already shells out to ``crawl.py``.

The graph is reached over the out-of-process MessagePack/UDS client
(``epistemic_graph.client``) — there is no PyO3. The node set is selected, then
its properties + induced edges are pulled in a **single** ``GetSubgraph``
round-trip (``fetch_subgraph``).

CLI::

    python -m agent_utilities.knowledge_graph.distillation.skill_graph_distiller \
        --query "ServiceNow incident management" --depth 2 --out-dir /tmp/sn

Library::

    from agent_utilities.knowledge_graph.distillation import SkillGraphDistiller
    manifest = await SkillGraphDistiller.from_env().distill(
        query="ServiceNow", depth=2, out_dir="/tmp/sn",
    )
"""

import argparse
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

# Manifest schema id — bump when the on-disk shape changes.
MANIFEST_SCHEMA = "skill-graph-kg-manifest/v1"

# Text-bearing node-property keys, in fidelity priority order. ``content`` is the
# curated KB Article body; ``trusted_answer`` is the full IdeaBlock chunk;
# ``summary``/``description`` are lossy fallbacks (e.g. Concept nodes).
_BODY_KEYS = ("content", "trusted_answer", "summary", "description")

# Property keys that may carry a node's type (the ingestion plane writes ``type``;
# some paths write ``node_type``).
_TYPE_KEYS = ("type", "node_type")

# Property keys that may carry a human title.
_TITLE_KEYS = ("title", "name")

# Property keys that may carry a source attribution.
_SOURCE_KEYS = ("source_url", "file_path", "source", "url")

# Edge-property keys that may carry the relationship label.
_REL_KEYS = ("rel_type", "type", "relationship_type")

# Relationship types worth surfacing as inline "Related" cross-links between
# materialised files.
_CROSSLINK_RELS = {
    "MENTIONS",
    "RELATES_TO",
    "ADDRESSES",
    "ADDRESSED_BY",
    "SIMILAR_TO",
    "DEPENDS_ON",
    "CONTAINS",
    "PART_OF",
    "IMPLEMENTS",
    "REALIZES",
}


def _slugify(text: str, fallback: str = "node") -> str:
    """Filesystem-safe, link-stable slug derived from a title or node id."""
    text = (text or "").strip()
    if not text:
        text = fallback
    # Drop an ``ns:`` style prefix (doc:, ideablock:, concept:) for readability.
    if ":" in text and "/" not in text and " " not in text:
        text = text.split(":", 1)[1] or text
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-_.")
    slug = re.sub(r"-{2,}", "-", slug)
    return (slug or fallback)[:80]


def _first(props: dict | None, keys: tuple[str, ...]) -> Any:
    for k in keys:
        v = props.get(k) if props else None
        if v:
            return v
    return None


class SkillGraphDistiller:
    """Reads a KG subgraph and materialises a ``reference/`` tree + manifest.

    Args:
        client: A connected async ``EpistemicGraphClient`` (or compatible).
        graph_name: Tenant graph the client is bound to (recorded in the
            manifest for provenance; the client itself is already bound).
    """

    def __init__(self, client: Any, graph_name: str = "__commons__") -> None:
        self.client = client
        self.graph_name = graph_name
        # ``Any`` (not ``Any | None``): a ``False`` sentinel marks "tried & failed"
        # without mypy inferring a ``bool`` member that lacks ``get_text_embedding``.
        self._embed_model: Any = None

    # ── construction ──────────────────────────────────────────────────────

    @classmethod
    async def connect(
        cls,
        *,
        graph_name: str | None = None,
        socket_path: str | None = None,
        tcp_addr: str | None = None,
        auth_secret: str | None = None,
    ) -> SkillGraphDistiller:
        """Acquire an async facade over the one process graph client."""
        import asyncio

        from agent_utilities.knowledge_graph.core.graph_compute import (
            GraphComputeEngine,
        )

        gname = graph_name or setting("KG_GRAPH_NAME", "__commons__")
        if socket_path or tcp_addr or auth_secret:
            raise ValueError(
                "per-call engine endpoint/auth overrides are disabled; configure "
                "the process graph authority through deployment settings"
            )
        engine = await asyncio.to_thread(
            GraphComputeEngine.get_or_create, graph_name=gname
        )
        return cls(engine.async_client, graph_name=gname)

    # ── stage 1: select ───────────────────────────────────────────────────

    async def select_subgraph(
        self,
        *,
        seed: str | None = None,
        query: str | None = None,
        depth: int = 2,
        max_nodes: int = 400,
        seed_results: int = 5,
    ) -> dict[str, Any]:
        """Pick the node set for the package.

        Seeds either from explicit node id(s) or by embedding ``query`` and
        running semantic search, then performs an undirected, hop-ordered BFS to
        ``depth`` (closest-first, capped at ``max_nodes``). Hop-ordering means a
        cap naturally keeps the most relevant — nearest — nodes.

        Returns ``{"anchors": [...], "node_ids": [...]}``.
        """
        anchors: list[str] = []
        if seed:
            anchors = [seed]
        elif query:
            anchors = await self._semantic_seed(query, seed_results)
        else:
            raise ValueError("select_subgraph requires either `seed` or `query`")

        if not anchors:
            return {"anchors": [], "node_ids": []}

        # Undirected, hop-bounded BFS. ``neighbors`` is bidirectional, so a
        # topic Concept also pulls back the Documents that MENTION it.
        seen: set[str] = set(anchors)
        frontier: list[str] = list(anchors)
        for _hop in range(max(0, depth)):
            frontier = await self._expand_one_hop(frontier, seen, max_nodes)
            if not frontier or len(seen) >= max_nodes:
                break

        return {"anchors": anchors, "node_ids": sorted(seen)}

    async def _expand_one_hop(
        self, frontier: list[str], seen: set[str], max_nodes: int
    ) -> list[str]:
        """One BFS hop: pull neighbors of each frontier node into ``seen``
        (mutated in place) and return the next frontier, capped at
        ``max_nodes``."""
        next_frontier: list[str] = []
        for node_id in frontier:
            if len(seen) >= max_nodes:
                break
            try:
                neigh = await self.client.nodes.neighbors(node_id)
            except Exception:  # noqa: BLE001
                neigh = []
            for nid in neigh:
                if nid not in seen:
                    seen.add(nid)
                    next_frontier.append(nid)
                    if len(seen) >= max_nodes:
                        break
        return next_frontier

    async def _semantic_seed(self, query: str, n: int) -> list[str]:
        emb = self._embed(query)
        if emb is None:
            logger.warning("No embedding model; cannot seed by query %r", query)
            return []
        try:
            hits = await self.client.graph.semantic_search(emb, n_results=n)
        except Exception as e:  # noqa: BLE001
            logger.warning("semantic_search failed: %s", e)
            return []
        # hits: list[(node_id, score)]
        return [h[0] for h in hits if h and h[0]]

    def _embed(self, text: str) -> list[float] | None:
        if self._embed_model is None:
            try:
                from agent_utilities.core.embedding_utilities import (
                    create_embedding_model,
                )

                self._embed_model = create_embedding_model()
            except Exception as e:  # noqa: BLE001
                logger.warning("embedding model unavailable: %s", e)
                self._embed_model = False  # sentinel: tried & failed
        if not self._embed_model:
            return None
        try:
            return self._embed_model.get_text_embedding(text)
        except Exception as e:  # noqa: BLE001
            logger.warning("embed failed: %s", e)
            return None

    # ── stage 2: taxonomy (folders) ───────────────────────────────────────

    async def derive_taxonomy(
        self, node_ids: list[str], props: dict[str, dict], resolution: float = 1.0
    ) -> dict[str, list[str]]:
        """Group selected nodes into ``reference/<cluster>/`` folders using
        community detection. Falls back to a single flat group on any failure or
        when the graph yields a single community.

        Returns an ordered ``{cluster_name: [node_id, ...]}`` mapping covering
        every node in ``node_ids`` exactly once.
        """
        selected = set(node_ids)
        communities = await self._detect_communities(resolution)
        clusters, assigned = self._cluster_communities(communities, selected, props)

        leftover = [n for n in node_ids if n not in assigned]
        if leftover:
            # If nothing clustered, keep it flat (no subfolder) under "".
            key = "" if not clusters else "general"
            clusters[key] = leftover

        # A lone cluster is just a flat tree — drop the subfolder.
        if len(clusters) == 1:
            only = next(iter(clusters.values()))
            return {"": only}
        return clusters

    async def _detect_communities(self, resolution: float) -> list:
        """Run community detection, folding any failure into a flat taxonomy."""
        try:
            return await self.client.graph.community_detection(resolution)
        except Exception as e:  # noqa: BLE001
            logger.info("community_detection unavailable (%s); flat taxonomy", e)
            return []

    def _cluster_communities(
        self,
        communities: list,
        selected: set[str],
        props: dict[str, dict],
    ) -> tuple[dict[str, list[str]], set[str]]:
        """Turn raw communities into named ``{cluster: [node_id, ...]}``
        groups, dropping singletons (they fold into "general" upstream).
        Returns the clusters plus the set of node ids they cover."""
        clusters: dict[str, list[str]] = {}
        assigned: set[str] = set()
        for comm in communities or []:
            members = [n for n in comm if n in selected and n not in assigned]
            if len(members) < 2:  # singletons fold into "general"
                continue
            name = self._cluster_name(members, props, taken=set(clusters))
            clusters[name] = members
            assigned.update(members)
        return clusters, assigned

    def _cluster_name(
        self, members: list[str], props: dict[str, dict], taken: set[str]
    ) -> str:
        """Name a cluster from its most representative Concept's title."""
        # Prefer a Concept node's title, else any title, else the first id.
        best = None
        for nid in members:
            p = props.get(nid) or {}
            ntype = str(_first(p, _TYPE_KEYS) or "")
            title = _first(p, _TITLE_KEYS)
            if title and ntype.lower() == "concept":
                best = title
                break
            if title and best is None:
                best = title
        base = _slugify(str(best) if best else members[0], fallback="cluster")
        name = base
        i = 2
        while name in taken:
            name = f"{base}-{i}"
            i += 1
        return name

    # ── stage 3: materialize ──────────────────────────────────────────────

    async def materialize(
        self,
        selection: dict[str, Any],
        taxonomy: dict[str, list[str]],
        props: dict[str, dict],
        edges: list[tuple[str, str, str]],
        out_dir: str | Path,
        *,
        selector: dict[str, Any],
    ) -> dict[str, Any]:
        """Write the ``reference/`` tree + ``kg_manifest.json`` under ``out_dir``.

        Only nodes that carry a body (one of ``_BODY_KEYS``) become files;
        body-less container nodes (e.g. a ``Document`` whose text lives on its
        child Concept/IdeaBlock nodes) contribute structure and cross-links but
        not empty files.
        """
        out = Path(out_dir)
        ref = out / "reference"
        ref.mkdir(parents=True, exist_ok=True)

        # Standardized ingestion stores a Document's full body AND its verbatim
        # chunks (IdeaBlock --PART_OF--> Document). When the parent Document is
        # itself materialised, its chunks are redundant — emit the doc, not
        # doc+chunks. Collect such covered children up front so they are recorded
        # in the manifest but never written as separate files.
        edge_records = edges
        covered_children = self._covered_children(edge_records, props)
        file_for, manifest_nodes = self._place_nodes(
            ref, taxonomy, props, covered_children
        )
        crosslinks = self._build_crosslinks(edge_records, file_for, manifest_nodes)
        files_written = self._write_reference_files(ref, file_for, props, crosslinks)

        manifest = self._build_manifest(
            selection=selection,
            taxonomy=taxonomy,
            selector=selector,
            files_written=files_written,
            edge_records=edge_records,
            manifest_nodes=manifest_nodes,
        )
        (out / "kg_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return manifest

    @staticmethod
    def _covered_children(
        edges: list[tuple[str, str, str]], props: dict[str, dict]
    ) -> set[str]:
        """Chunks (IdeaBlock --PART_OF--> Document) whose parent Document
        already carries a body — recorded in the manifest but never written
        as a separate, redundant file."""
        covered: set[str] = set()
        for src, dst, rel in edges:
            if rel.upper() in ("PART_OF", "CONTAINS"):
                child, parent = (src, dst) if rel.upper() == "PART_OF" else (dst, src)
                parent_props = props.get(parent) or {}
                if _first(parent_props, _BODY_KEYS):
                    covered.add(child)
        return covered

    def _place_nodes(
        self,
        ref: Path,
        taxonomy: dict[str, list[str]],
        props: dict[str, dict],
        covered_children: set[str],
    ) -> tuple[dict[str, str], list[dict[str, Any]]]:
        """Assign each selected node either a manifest-only entry (no body,
        or a chunk covered by its parent) or a written file path — disambiguating
        collisions across the whole tree — and create the cluster directories
        that will receive files. Returns ``(node_id -> relative path, manifest
        node entries)``."""
        file_for: dict[str, str] = {}
        used_rel: set[str] = set()
        manifest_nodes: list[dict[str, Any]] = []

        for cluster, members in taxonomy.items():
            cluster_dir = ref / cluster if cluster else ref
            for nid in members:
                entry, rel = self._place_one_node(
                    nid, cluster, props, covered_children, used_rel
                )
                manifest_nodes.append(entry)
                if rel is not None:
                    used_rel.add(rel)
                    file_for[nid] = rel
                    cluster_dir.mkdir(parents=True, exist_ok=True)

        return file_for, manifest_nodes

    @staticmethod
    def _place_one_node(
        nid: str,
        cluster: str,
        props: dict[str, dict],
        covered_children: set[str],
        used_rel: set[str],
    ) -> tuple[dict[str, Any], str | None]:
        """Classify one node: a manifest-only entry (no body, or a chunk
        covered by its parent Document — ``rel=None``) or a manifest entry
        with a unique ``cluster/slug.md`` relative path."""
        p = props.get(nid) or {}
        body = _first(p, _BODY_KEYS)
        ntype = str(_first(p, _TYPE_KEYS) or "Node")
        title = str(_first(p, _TITLE_KEYS) or _slugify(nid))
        if not body or nid in covered_children:
            # No body, or a chunk already covered by its parent Document:
            # recorded in the manifest, but no (empty/duplicate) file.
            entry: dict[str, Any] = {
                "id": nid,
                "type": ntype,
                "title": title,
                "file": None,
                "source_url": _first(p, _SOURCE_KEYS),
            }
            if nid in covered_children:
                entry["covered_by_parent"] = True
            return entry, None
        rel = SkillGraphDistiller._unique_rel_path(cluster, title or nid, used_rel)
        entry = {
            "id": nid,
            "type": ntype,
            "title": title,
            "file": f"reference/{rel}",
            "source_url": _first(p, _SOURCE_KEYS),
        }
        return entry, rel

    @staticmethod
    def _unique_rel_path(cluster: str, title_or_id: str, used_rel: set[str]) -> str:
        """Slugify ``title_or_id`` into a ``cluster/slug.md`` path, disambiguating
        collisions against ``used_rel`` with a ``-2``, ``-3``, ... suffix."""
        slug = _slugify(title_or_id, fallback="node")
        cand = slug
        i = 2
        rel = f"{cluster}/{cand}.md" if cluster else f"{cand}.md"
        while rel in used_rel:
            cand = f"{slug}-{i}"
            rel = f"{cluster}/{cand}.md" if cluster else f"{cand}.md"
            i += 1
        return rel

    @staticmethod
    def _build_crosslinks(
        edge_records: list[tuple[str, str, str]],
        file_for: dict[str, str],
        manifest_nodes: list[dict[str, Any]],
    ) -> dict[str, list[tuple[str, str]]]:
        """Adjacency among *written* files, for inline "## Related" links."""
        crosslinks: dict[str, list[tuple[str, str]]] = {}
        for src, dst, rel in edge_records:
            if rel.upper() not in _CROSSLINK_RELS:
                continue
            if src in file_for and dst in file_for and src != dst:
                dst_title = next(
                    (n["title"] for n in manifest_nodes if n["id"] == dst), dst
                )
                crosslinks.setdefault(src, []).append((dst_title, file_for[dst]))
        return crosslinks

    def _write_reference_files(
        self,
        ref: Path,
        file_for: dict[str, str],
        props: dict[str, dict],
        crosslinks: dict[str, list[tuple[str, str]]],
    ) -> int:
        """Render each written node's markdown file (front-matter + body +
        "## Related" cross-links) and return the count written."""
        files_written = 0
        for nid, rel in file_for.items():
            p = props.get(nid) or {}
            body = str(_first(p, _BODY_KEYS) or "")
            title = str(_first(p, _TITLE_KEYS) or _slugify(nid))
            src_url = _first(p, _SOURCE_KEYS)
            fm = [
                "---",
                f"title: {title}",
                f"kg_node_id: {nid}",
                f"kg_node_type: {_first(p, _TYPE_KEYS) or 'Node'}",
            ]
            if src_url:
                fm.append(f"source_url: {src_url}")
            fm.append("---")
            parts = ["\n".join(fm), "", f"# {title}", "", body.strip(), ""]
            links = crosslinks.get(nid)
            if links:
                parts.append("## Related")
                for dst_title, dst_rel in sorted(set(links)):
                    # Links are relative to the file's own folder; use a path up
                    # to reference root for cross-cluster correctness.
                    parts.append(f"- [{dst_title}]({self._rel_link(rel, dst_rel)})")
                parts.append("")
            (ref / rel).write_text("\n".join(parts), encoding="utf-8")
            files_written += 1
        return files_written

    def _build_manifest(
        self,
        *,
        selection: dict[str, Any],
        taxonomy: dict[str, list[str]],
        selector: dict[str, Any],
        files_written: int,
        edge_records: list[tuple[str, str, str]],
        manifest_nodes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Assemble the ``kg_manifest.json`` payload for a skill-graph package."""
        node_ids = selection["node_ids"]
        return {
            "schema": MANIFEST_SCHEMA,
            "ontology": "agent-utilities",
            "agent_utilities_version": _pkg_version(),
            "graph_name": self.graph_name,
            "selector": selector,
            "snapshot_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stats": {
                "nodes": len(node_ids),
                "files": files_written,
                "clusters": len([c for c in taxonomy if c]) or 1,
                "edges": len(edge_records),
            },
            "anchors": selection.get("anchors", []),
            "clusters": {k: v for k, v in taxonomy.items()},
            "nodes": manifest_nodes,
            "edges": [{"src": s, "dst": d, "type": r} for s, d, r in edge_records],
        }

    @staticmethod
    def _rel_link(from_rel: str, to_rel: str) -> str:
        """Relative markdown link from one reference file to another."""
        from_dir = Path(from_rel).parent
        try:
            return os.path.relpath(to_rel, from_dir if str(from_dir) != "." else "")
        except ValueError:
            return to_rel

    # ── orchestration ─────────────────────────────────────────────────────

    async def fetch_subgraph(
        self, node_ids: list[str]
    ) -> tuple[dict[str, dict], list[tuple[str, str, str]]]:
        """Fetch node properties + induced edges in ONE round-trip via the
        engine's batched ``GetSubgraph``.

        Returns ``(props_by_id, edge_records)``. ``GetSubgraph`` is part of the
        current engine contract; a missing or malformed response fails loudly.
        """
        sub = await self.client.graph.get_subgraph(list(node_ids))
        if not isinstance(sub, dict) or not isinstance(sub.get("nodes"), list):
            raise RuntimeError("engine GetSubgraph returned an invalid response")
        props: dict[str, dict] = {
            node["id"]: (node.get("properties") or {})
            for node in sub["nodes"]
            if isinstance(node, dict) and node.get("id")
        }
        for nid in node_ids:
            props.setdefault(nid, {})
        edges = self._parse_edges(sub.get("edges", []))
        return props, edges

    @staticmethod
    def _parse_edges(raw_edges: list) -> list[tuple[str, str, str]]:
        """Normalize the engine's raw edge records to ``(src, dst, rel)``,
        dropping anything malformed."""
        edges: list[tuple[str, str, str]] = []
        for edge in raw_edges:
            if not isinstance(edge, dict):
                continue
            src, dst = edge.get("source"), edge.get("target")
            if not (src and dst):
                continue
            rel = str(_first(edge.get("properties") or {}, _REL_KEYS) or "RELATED")
            edges.append((src, dst, rel))
        return edges

    async def distill(
        self,
        *,
        seed: str | None = None,
        query: str | None = None,
        depth: int = 2,
        max_nodes: int = 400,
        resolution: float = 1.0,
        out_dir: str | Path,
    ) -> dict[str, Any]:
        """Full pipeline: select → fetch props → taxonomy → materialize."""
        selector = {
            "seed": seed,
            "query": query,
            "depth": depth,
            "max_nodes": max_nodes,
            "resolution": resolution,
        }
        selection = await self.select_subgraph(
            seed=seed, query=query, depth=depth, max_nodes=max_nodes
        )
        if not selection["node_ids"]:
            logger.warning("Selection empty for selector=%s", selector)
            # Still emit an (empty) manifest so callers get a deterministic shape.
            selection = {"anchors": selection.get("anchors", []), "node_ids": []}
            return await self.materialize(
                selection, {"": []}, {}, [], out_dir, selector=selector
            )
        props, edges = await self.fetch_subgraph(selection["node_ids"])
        taxonomy = await self.derive_taxonomy(
            selection["node_ids"], props, resolution=resolution
        )
        manifest = await self.materialize(
            selection, taxonomy, props, edges, out_dir, selector=selector
        )
        logger.info(
            "Distilled %d nodes → %d files in %d clusters at %s",
            manifest["stats"]["nodes"],
            manifest["stats"]["files"],
            manifest["stats"]["clusters"],
            out_dir,
        )
        return manifest

    # ── workflow distillation (paired graph-native procedure skill) ───────

    # Node types whose subgraph reads as an ordered procedure.
    _PROCEDURE_TYPES = {
        "procedure",
        "playbook",
        "policy",
        "action",
        "task",
        "step",
        "process",
    }

    @staticmethod
    def _token(text: str) -> str:
        """A single ``[a-zA-Z0-9_-]+`` token (what the workflow validator wants)."""
        tok = re.sub(r"[^a-zA-Z0-9_-]+", "_", (text or "").strip()).strip("_-")
        return (tok or "step")[:60]

    @staticmethod
    def _toposort(nodes: list[str], precedes: list[tuple[str, str]]) -> list[str]:
        """Kahn topological sort; ``precedes`` = (before, after) pairs. On a cycle,
        the remaining nodes are appended in their original order (deterministic)."""
        succ, indeg = SkillGraphDistiller._build_adjacency(nodes, precedes)
        return SkillGraphDistiller._kahn_order(nodes, succ, indeg)

    @staticmethod
    def _build_adjacency(
        nodes: list[str], precedes: list[tuple[str, str]]
    ) -> tuple[dict[str, list[str]], dict[str, int]]:
        """Successor list + in-degree map for Kahn's algorithm, over edges
        whose endpoints are both in ``nodes``."""
        nset = set(nodes)
        succ: dict[str, list[str]] = {n: [] for n in nodes}
        indeg: dict[str, int] = {n: 0 for n in nodes}
        for a, b in precedes:
            if a in nset and b in nset:
                succ[a].append(b)
                indeg[b] += 1
        return succ, indeg

    @staticmethod
    def _kahn_order(
        nodes: list[str], succ: dict[str, list[str]], indeg: dict[str, int]
    ) -> list[str]:
        """Kahn's algorithm over a precomputed adjacency; a cycle's leftover
        nodes are appended in their original order (deterministic)."""
        ready = [n for n in nodes if indeg[n] == 0]
        order: list[str] = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for m in succ[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)
        if len(order) < len(nodes):  # cycle — append the rest deterministically
            order += [n for n in nodes if n not in set(order)]
        return order

    async def distill_workflow(
        self,
        *,
        seed: str | None = None,
        query: str | None = None,
        depth: int = 2,
        max_nodes: int = 200,
        out_dir: str | Path,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Distill a procedure subgraph into a graph-native skill-workflow.

        Maps the KG's procedural structure onto a workflow step-DAG: nodes of a
        procedure type (or any node participating in a ``PRECEDES`` edge) become
        steps; ``PRECEDES`` edges become ``depends_on`` ordering. Emits a
        ``SKILL.md`` consumable/validatable by ``skill-workflow-builder`` plus a
        ``kg_manifest.json`` for provenance. (CONCEPT:AU-AHE.optimization.physical-distillation-engine / KG-2.7)
        """
        selector = {
            "seed": seed,
            "query": query,
            "depth": depth,
            "max_nodes": max_nodes,
            "mode": "workflow",
        }
        selection = await self.select_subgraph(
            seed=seed, query=query, depth=depth, max_nodes=max_nodes
        )
        node_ids = selection["node_ids"]
        props, edges = await self.fetch_subgraph(node_ids)

        ordered, precedes, deps = self._derive_workflow_steps(node_ids, props, edges)
        wf_name = self._workflow_name(name, seed, query)
        markdown = self._render_workflow_markdown(
            wf_name, seed, query, ordered, props, deps
        )

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "SKILL.md").write_text(markdown, encoding="utf-8")

        manifest = self._build_workflow_manifest(
            selector=selector, ordered=ordered, deps=deps, precedes=precedes
        )
        (out / "kg_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        logger.info(
            "Distilled workflow %s with %d steps at %s", wf_name, len(ordered), out_dir
        )
        return {"name": wf_name, "steps": len(ordered), "manifest": manifest}

    def _derive_workflow_steps(
        self,
        node_ids: list[str],
        props: dict[str, dict],
        edges: list[tuple[str, str, str]],
    ) -> tuple[list[str], list[tuple[str, str]], dict[str, list[int]]]:
        """Map ``PRECEDES`` edges + procedure-typed nodes onto an ordered step
        list plus a ``node_id -> [before-step-index, ...]`` dependency map."""
        precedes = [(s, d) for (s, d, r) in edges if r.upper() == "PRECEDES"]
        step_ids = self._collect_step_ids(node_ids, props, precedes)
        ordered = self._toposort(sorted(step_ids), precedes)
        deps = self._step_dependencies(ordered, precedes)
        return ordered, precedes, deps

    def _collect_step_ids(
        self,
        node_ids: list[str],
        props: dict[str, dict],
        precedes: list[tuple[str, str]],
    ) -> set[str]:
        """Nodes on a ``PRECEDES`` edge, plus any procedure-typed node."""
        step_ids: set[str] = set()
        for s, d in precedes:
            step_ids.update((s, d))
        for nid in node_ids:
            ntype = str(_first(props.get(nid) or {}, _TYPE_KEYS) or "").lower()
            if ntype in self._PROCEDURE_TYPES:
                step_ids.add(nid)
        return step_ids

    @staticmethod
    def _step_dependencies(
        ordered: list[str], precedes: list[tuple[str, str]]
    ) -> dict[str, list[int]]:
        """``after`` node id -> [step index of each ``before`` it depends on]."""
        pos = {nid: i for i, nid in enumerate(ordered)}
        deps: dict[str, list[int]] = {}
        for before, after in precedes:
            if before in pos and after in pos:
                deps.setdefault(after, []).append(pos[before])
        return deps

    def _workflow_name(
        self, name: str | None, seed: str | None, query: str | None
    ) -> str:
        """Derive the ``*-workflow`` slug used for the SKILL.md name/filename."""
        wf_name = name or (seed or query or "kg-workflow")
        wf_name = self._token(wf_name).lower().replace("_", "-")
        if not wf_name.endswith("-workflow"):
            wf_name = f"{wf_name}-workflow"
        return wf_name

    def _render_workflow_markdown(
        self,
        wf_name: str,
        seed: str | None,
        query: str | None,
        ordered: list[str],
        props: dict[str, dict],
        deps: dict[str, list[int]],
    ) -> str:
        """Render the distilled ``SKILL.md`` body for a workflow's step-DAG."""
        lines = [
            "---",
            f"name: {wf_name}",
            "description: >-",
            f"  Graph-native procedure distilled from the Knowledge Graph "
            f"({'seed ' + seed if seed else 'query: ' + (query or '')}).",
            "domain: kg-distilled",
            "agent: orchestrator",
            "tags: [kg-distilled, workflow, procedure]",
            "concept: CONCEPT:AU-KG.query.vendor-agnostic-traversal",
            "---",
            "",
            f"# {wf_name}",
            "",
            "Distilled from KG procedure nodes; `PRECEDES` edges → step ordering.",
            "",
        ]
        for i, nid in enumerate(ordered):
            p = props.get(nid) or {}
            token = self._token(_first(p, _TITLE_KEYS) or nid)
            dep_nums = sorted(set(deps.get(nid, [])))
            dep_clause = (
                f" [depends_on: {', '.join(f'Step {n}' for n in dep_nums)}]"
                if dep_nums
                else ""
            )
            body = str(_first(p, _BODY_KEYS) or _first(p, _TITLE_KEYS) or nid).strip()
            lines.append(f"### Step {i}: {token}{dep_clause}")
            lines.append("**Agent**: `orchestrator`")
            lines.append("")
            lines.append(body)
            lines.append(f"Expected: {token}_result")
            lines.append("")
        return "\n".join(lines)

    def _build_workflow_manifest(
        self,
        *,
        selector: dict[str, Any],
        ordered: list[str],
        deps: dict[str, list[int]],
        precedes: list[tuple[str, str]],
    ) -> dict[str, Any]:
        """Assemble the ``kg_manifest.json`` payload for a skill-workflow."""
        return {
            "schema": MANIFEST_SCHEMA,
            "kind": "skill-workflow",
            "ontology": "agent-utilities",
            "agent_utilities_version": _pkg_version(),
            "graph_name": self.graph_name,
            "selector": selector,
            "snapshot_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "steps": [
                {
                    "step": i,
                    "node_id": nid,
                    "depends_on": sorted(set(deps.get(nid, []))),
                }
                for i, nid in enumerate(ordered)
            ],
            "precedes": [{"before": a, "after": b} for a, b in precedes],
        }

    async def close(self) -> None:
        try:
            await self.client.close()
        except Exception:  # noqa: BLE001
            pass


def _pkg_version() -> str:
    try:
        from importlib.metadata import version

        return version("agent-utilities")
    except Exception:  # noqa: BLE001
        return "unknown"


async def _amain(args: argparse.Namespace) -> int:
    distiller = await SkillGraphDistiller.connect(graph_name=args.graph_name)
    try:
        if args.workflow:
            result = await distiller.distill_workflow(
                seed=args.seed,
                query=args.query,
                depth=args.depth,
                max_nodes=args.max_nodes,
                out_dir=args.out_dir,
                name=args.name,
            )
            summary = {
                "kind": "skill-workflow",
                "name": result["name"],
                "steps": result["steps"],
            }
        else:
            manifest = await distiller.distill(
                seed=args.seed,
                query=args.query,
                depth=args.depth,
                max_nodes=args.max_nodes,
                resolution=args.resolution,
                out_dir=args.out_dir,
            )
            summary = {"kind": "skill-graph", "stats": manifest["stats"]}
    finally:
        await distiller.close()
    print(
        json.dumps(
            {
                "out_dir": str(Path(args.out_dir).resolve()),
                "manifest": str((Path(args.out_dir) / "kg_manifest.json").resolve()),
                **summary,
            },
            indent=2,
        )
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill a KG subgraph into a reference/ tree + kg_manifest.json "
        "(consumable by skill-graph-builder as a local-directory source)."
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--seed", help="Anchor node id to grow the subgraph from.")
    g.add_argument("--query", help="Natural-language seed (semantic search anchor).")
    parser.add_argument(
        "--depth", type=int, default=2, help="BFS hop depth (default 2)."
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=400,
        help="Cap on selected nodes (default 400).",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Community-detection resolution → folder granularity (default 1.0).",
    )
    parser.add_argument(
        "--graph-name",
        default=None,
        help="Tenant graph (default $KG_GRAPH_NAME or __commons__).",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--workflow",
        action="store_true",
        help="Distill a graph-native skill-WORKFLOW (procedure step-DAG from "
        "PRECEDES edges) instead of a documentation skill-graph.",
    )
    parser.add_argument(
        "--name", default=None, help="Optional name for the distilled workflow."
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
