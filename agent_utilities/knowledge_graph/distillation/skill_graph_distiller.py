#!/usr/bin/python
from __future__ import annotations

"""Skill-Graph Distiller — KG subgraph → packageable reference tree + manifest.

CONCEPT:AHE-3.9 — Physical Knowledge Distillation (read side).

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
round-trip (``fetch_subgraph``); a legacy per-node fallback keeps it working
against engines that predate that batched method.

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

    def __init__(self, client: Any, graph_name: str = "__bus__") -> None:
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
        """Connect a client using ``AgentConfig`` defaults (mirrors
        ``GraphComputeEngine``'s endpoint resolution) and return a distiller."""
        from epistemic_graph.client import EpistemicGraphClient

        gname = graph_name or os.environ.get("KG_GRAPH_NAME", "__bus__")

        # Resolve endpoint/secret the same way the compute engine does, so the
        # distiller reads from exactly the graph the ingestion plane writes to.
        if not (socket_path or tcp_addr):
            try:
                from agent_utilities.core.config import AgentConfig
                from agent_utilities.knowledge_graph.core.graph_compute import (
                    resolve_engine_auth,
                )

                config = AgentConfig()
                if auth_secret is None:
                    # Shared per-install secret (CONCEPT:OS-5.14) — same
                    # resolution the compute engine uses.
                    auth_secret, _insecure = resolve_engine_auth(config)
                # Shard-aware (CONCEPT:KG-2.58): route to the graph's
                # HRW-owning shard instead of assuming endpoints[0], so the
                # distiller reads the same shard the ingestion plane wrote.
                from agent_utilities.knowledge_graph.core.shard_topology import (
                    resolve_endpoints,
                    shard_endpoint_for,
                )

                ep = shard_endpoint_for(gname, resolve_endpoints(config))
                if ep.startswith("tcp://"):
                    tcp_addr = ep[6:]
                elif ep.startswith("unix://"):
                    socket_path = ep[7:]
                else:
                    socket_path = ep
            except Exception:  # noqa: BLE001 — fall through to client env defaults
                logger.debug("AgentConfig unavailable; using client env defaults")

        client = await EpistemicGraphClient.connect(
            socket_path=socket_path,
            tcp_addr=tcp_addr,
            auth_secret=auth_secret,
            graph_name=gname,
        )
        return cls(client, graph_name=gname)

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
            frontier = next_frontier
            if not frontier or len(seen) >= max_nodes:
                break

        return {"anchors": anchors, "node_ids": sorted(seen)}

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
        try:
            communities = await self.client.graph.community_detection(resolution)
        except Exception as e:  # noqa: BLE001
            logger.info("community_detection unavailable (%s); flat taxonomy", e)
            communities = []

        clusters: dict[str, list[str]] = {}
        assigned: set[str] = set()
        for comm in communities or []:
            members = [n for n in comm if n in selected and n not in assigned]
            if len(members) < 2:  # singletons fold into "general"
                continue
            name = self._cluster_name(members, props, taken=set(clusters))
            clusters[name] = members
            assigned.update(members)

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

        node_ids = selection["node_ids"]

        # Standardized ingestion stores a Document's full body AND its verbatim
        # chunks (IdeaBlock --PART_OF--> Document). When the parent Document is
        # itself materialised, its chunks are redundant — emit the doc, not
        # doc+chunks. Collect such covered children up front so they are recorded
        # in the manifest but never written as separate files.
        all_edges = edges
        covered_children: set[str] = set()
        for src, dst, rel in all_edges:
            if rel.upper() in ("PART_OF", "CONTAINS"):
                child, parent = (src, dst) if rel.upper() == "PART_OF" else (dst, src)
                parent_props = props.get(parent) or {}
                if _first(parent_props, _BODY_KEYS):
                    covered_children.add(child)
        # Map node_id -> relative-to-reference file path for the nodes we write.
        file_for: dict[str, str] = {}
        used_rel: set[str] = set()

        manifest_nodes: list[dict[str, Any]] = []

        for cluster, members in taxonomy.items():
            cluster_dir = ref / cluster if cluster else ref
            for nid in members:
                p = props.get(nid) or {}
                body = _first(p, _BODY_KEYS)
                ntype = str(_first(p, _TYPE_KEYS) or "Node")
                title = str(_first(p, _TITLE_KEYS) or _slugify(nid))
                if not body or nid in covered_children:
                    # No body, or a chunk already covered by its parent Document:
                    # recorded in the manifest, but no (empty/duplicate) file.
                    entry = {
                        "id": nid,
                        "type": ntype,
                        "title": title,
                        "file": None,
                        "source_url": _first(p, _SOURCE_KEYS),
                    }
                    if nid in covered_children:
                        entry["covered_by_parent"] = True
                    manifest_nodes.append(entry)
                    continue
                slug = _slugify(title or nid, fallback="node")
                # Disambiguate collisions across the whole tree.
                cand = slug
                i = 2
                rel = f"{cluster}/{cand}.md" if cluster else f"{cand}.md"
                while rel in used_rel:
                    cand = f"{slug}-{i}"
                    rel = f"{cluster}/{cand}.md" if cluster else f"{cand}.md"
                    i += 1
                used_rel.add(rel)
                file_for[nid] = rel
                manifest_nodes.append(
                    {
                        "id": nid,
                        "type": ntype,
                        "title": title,
                        "file": f"reference/{rel}",
                        "source_url": _first(p, _SOURCE_KEYS),
                    }
                )
                cluster_dir.mkdir(parents=True, exist_ok=True)

        # Reuse the single batched edge read from above (no per-edge round-trip).
        edge_records = all_edges

        # Build adjacency among *written* files for inline cross-links.
        crosslinks: dict[str, list[tuple[str, str]]] = {}
        for src, dst, rel in edge_records:
            if rel.upper() not in _CROSSLINK_RELS:
                continue
            if src in file_for and dst in file_for and src != dst:
                dst_title = next(
                    (n["title"] for n in manifest_nodes if n["id"] == dst), dst
                )
                crosslinks.setdefault(src, []).append((dst_title, file_for[dst]))

        # Write the files.
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

        manifest = {
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
        (out / "kg_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return manifest

    @staticmethod
    def _rel_link(from_rel: str, to_rel: str) -> str:
        """Relative markdown link from one reference file to another."""
        from_dir = Path(from_rel).parent
        try:
            return os.path.relpath(to_rel, from_dir if str(from_dir) != "." else "")
        except ValueError:
            return to_rel

    async def _collect_edges(self, node_ids: set[str]) -> list[tuple[str, str, str]]:
        """Return ``(src, dst, rel_type)`` for edges fully inside the selection.

        Uses a single ``edges.list()`` and unpacks edge-property blobs locally.
        """
        try:
            raw_edges = await self.client.edges.list()
        except Exception as e:  # noqa: BLE001
            logger.info("edges.list() unavailable (%s)", e)
            return []
        import msgpack

        out: list[tuple[str, str, str]] = []
        for edge in raw_edges or []:
            try:
                src, dst = edge[0], edge[1]
            except (IndexError, TypeError):
                continue
            if src not in node_ids or dst not in node_ids:
                continue
            rel = "RELATED"
            blob = edge[2] if len(edge) > 2 else None
            if blob:
                try:
                    if isinstance(blob, list | tuple):
                        blob = bytes(blob)
                    eprops = msgpack.unpackb(blob, raw=False) if blob else {}
                    rel = str(_first(eprops, _REL_KEYS) or rel)
                except Exception:  # noqa: BLE001
                    pass
            out.append((src, dst, rel))
        return out

    # ── orchestration ─────────────────────────────────────────────────────

    async def fetch_subgraph(
        self, node_ids: list[str]
    ) -> tuple[dict[str, dict], list[tuple[str, str, str]]]:
        """Fetch node properties + induced edges in ONE round-trip via the
        engine's batched ``GetSubgraph``.

        Returns ``(props_by_id, edge_records)``. Falls back to the legacy path
        (per-node ``properties`` + full ``edges.list`` scan) when the engine
        predates the batched method, so the distiller works against any engine.
        """
        try:
            sub = await self.client.graph.get_subgraph(list(node_ids))
            if isinstance(sub, dict) and "nodes" in sub:
                props: dict[str, dict] = {
                    n["id"]: (n.get("properties") or {})
                    for n in sub.get("nodes", [])
                    if n.get("id")
                }
                for nid in node_ids:  # guarantee every requested id is present
                    props.setdefault(nid, {})
                edges: list[tuple[str, str, str]] = []
                for e in sub.get("edges", []):
                    src, dst = e.get("source"), e.get("target")
                    if not (src and dst):
                        continue
                    rel = str(_first(e.get("properties") or {}, _REL_KEYS) or "RELATED")
                    edges.append((src, dst, rel))
                return props, edges
        except Exception as e:  # noqa: BLE001
            logger.info(
                "get_subgraph unavailable (%s); falling back to per-node reads", e
            )
        # Legacy fallback.
        props = await self.fetch_props(node_ids)
        edges = await self._collect_edges(set(node_ids))
        return props, edges

    async def fetch_props(self, node_ids: list[str]) -> dict[str, dict]:
        """Per-node property read (legacy fallback for engines without the
        batched ``GetSubgraph``)."""
        props: dict[str, dict] = {}
        for nid in node_ids:
            try:
                p = await self.client.nodes.properties(nid)
            except Exception:  # noqa: BLE001
                p = None
            props[nid] = p or {}
        return props

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
        nset = set(nodes)
        succ: dict[str, list[str]] = {n: [] for n in nodes}
        indeg: dict[str, int] = {n: 0 for n in nodes}
        for a, b in precedes:
            if a in nset and b in nset:
                succ[a].append(b)
                indeg[b] += 1
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
        ``kg_manifest.json`` for provenance. (CONCEPT:AHE-3.9 / KG-2.7)
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

        precedes = [(s, d) for (s, d, r) in edges if r.upper() == "PRECEDES"]
        step_ids: set[str] = set()
        for s, d in precedes:
            step_ids.update((s, d))
        for nid in node_ids:
            ntype = str(_first(props.get(nid) or {}, _TYPE_KEYS) or "").lower()
            if ntype in self._PROCEDURE_TYPES:
                step_ids.add(nid)

        ordered = self._toposort(sorted(step_ids), precedes)
        pos = {nid: i for i, nid in enumerate(ordered)}
        # after → [before step numbers]
        deps: dict[str, list[int]] = {}
        for before, after in precedes:
            if before in pos and after in pos:
                deps.setdefault(after, []).append(pos[before])

        wf_name = name or (seed or query or "kg-workflow")
        wf_name = self._token(wf_name).lower().replace("_", "-")
        if not wf_name.endswith("-workflow"):
            wf_name = f"{wf_name}-workflow"

        lines = [
            "---",
            f"name: {wf_name}",
            "description: >-",
            f"  Graph-native procedure distilled from the Knowledge Graph "
            f"({'seed ' + seed if seed else 'query: ' + (query or '')}).",
            "domain: kg-distilled",
            "agent: orchestrator",
            "tags: [kg-distilled, workflow, procedure]",
            "concept: CONCEPT:KG-2.7",
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

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")

        manifest = {
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
        (out / "kg_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        logger.info(
            "Distilled workflow %s with %d steps at %s", wf_name, len(ordered), out_dir
        )
        return {"name": wf_name, "steps": len(ordered), "manifest": manifest}

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
        help="Tenant graph (default $KG_GRAPH_NAME or __bus__).",
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
