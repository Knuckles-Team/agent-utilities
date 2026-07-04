#!/usr/bin/python
from __future__ import annotations

"""Skill-Graph Importer — re-ingest a distilled skill-graph back into a KG.

CONCEPT:AU-AHE.optimization.physical-distillation-engine / KG-2.7 — The round-trip counterpart of ``SkillGraphDistiller``.

A skill-graph distilled by ``SkillGraphDistiller`` ships a ``kg_manifest.json``
(original node ids/types, edges, ontology, snapshot) alongside its ``reference/``
markdown tree. This importer reads that manifest and faithfully reconstructs the
subgraph in a *recipient* KG — preserving original node ids and edges — so a
curated, shareable knowledge package can be merged into another brain.

Because node ids are preserved and chunk ids are deterministic, re-importing is
idempotent (overwrites, never duplicates). For cross-package convergence (two
people's "ServiceNow" packages), pass ``dedup=True`` to run the existing
IdeaBlock deduplicator (``engine.distill_knowledge``) after import.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Where each node type's body text is stored (mirrors the distiller's read keys).
_BODY_KEY_FOR_TYPE = {
    "document": "content",
    "idea_block": "trusted_answer",
    "concept": "summary",
    "article": "content",
}


def _extract_body(md_text: str) -> str:
    """Recover the node body from a distilled reference file.

    Strips the YAML frontmatter, the leading ``# Title`` heading, and the
    trailing ``## Related`` cross-link section the distiller appended.
    """
    text = md_text
    # Drop leading frontmatter block.
    fm = re.match(r"^---\s*\n.*?\n---\s*\n", text, re.DOTALL)
    if fm:
        text = text[fm.end() :]
    # Drop a single leading "# Heading" line.
    text = re.sub(r"^\s*#\s+.*\n", "", text, count=1)
    # Truncate at the appended "## Related" section.
    idx = text.find("\n## Related")
    if idx != -1:
        text = text[:idx]
    return text.strip()


def import_skill_graph_pack(
    engine: Any, pack_dir: str | Path, *, dedup: bool = False
) -> dict[str, Any]:
    """Reconstruct a distilled skill-graph's subgraph into ``engine``'s KG.

    Args:
        engine: An ``IntelligenceGraphEngine`` (must expose ``backend`` with
            ``add_node``/``add_edge``).
        pack_dir: A distilled skill-graph directory (contains ``kg_manifest.json``
            and ``reference/``).
        dedup: If True, run ``engine.distill_knowledge()`` afterwards to merge
            duplicate IdeaBlocks against existing knowledge.

    Returns:
        A stats dict ``{nodes, edges, files, dedup}``.
    """
    pack = Path(pack_dir)
    manifest_path = pack / "kg_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No kg_manifest.json in {pack_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    backend = getattr(engine, "backend", None) or getattr(engine, "graph", None)
    add_node = getattr(backend, "add_node", None)
    add_edge = getattr(backend, "add_edge", None)
    if not callable(add_node):
        raise RuntimeError("engine backend lacks add_node")

    nodes_written = 0
    for node in manifest.get("nodes", []):
        nid = node.get("id")
        if not nid:
            continue
        ntype = node.get("type") or "Node"
        title = node.get("title") or nid
        props: dict[str, Any] = {"type": ntype, "name": title}
        if node.get("source_url"):
            props["source_url"] = node["source_url"]
        rel = node.get("file")
        if rel:
            fp = pack / rel
            if fp.exists():
                body = _extract_body(fp.read_text(encoding="utf-8"))
                body_key = _BODY_KEY_FOR_TYPE.get(ntype.lower(), "content")
                props[body_key] = body
        try:
            add_node(nid, **props)
            nodes_written += 1
        except Exception as e:  # noqa: BLE001
            logger.warning("import: add_node failed for %s: %s", nid, e)

    edges_written = 0
    if callable(add_edge):
        for edge in manifest.get("edges", []):
            src, dst, rel = edge.get("src"), edge.get("dst"), edge.get("type")
            if not (src and dst):
                continue
            try:
                add_edge(src, dst, rel_type=rel or "RELATED")
                edges_written += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("import: add_edge %s->%s failed: %s", src, dst, e)

    deduped = None
    if dedup and hasattr(engine, "distill_knowledge"):
        try:
            deduped = engine.distill_knowledge()
        except Exception as e:  # noqa: BLE001
            logger.warning("import: dedup pass failed: %s", e)

    stats = {
        "nodes": nodes_written,
        "edges": edges_written,
        "files": sum(1 for n in manifest.get("nodes", []) if n.get("file")),
        "ontology": manifest.get("ontology"),
        "snapshot_ts": manifest.get("snapshot_ts"),
        "dedup": bool(deduped),
    }
    logger.info("Imported skill-graph pack %s: %s", pack_dir, stats)
    return stats
