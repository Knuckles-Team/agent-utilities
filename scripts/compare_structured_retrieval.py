#!/usr/bin/env python3
"""Offline, visibility-matched section retrieval comparison for EH-652.

The candidate is a no-training table-of-contents leaf ranker. It establishes a
cheap ablation before a corpus-specific STAIR model is considered. This script
does not change the served retrieval path.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_utilities.knowledge_graph.ontology.document_processing import (
    SectionNode,
    SectionTreeConfig,
    build_section_tree,
    iter_sections,
)
from agent_utilities.knowledge_graph.retrieval.hierarchical_document_retriever import (
    HierarchicalDocumentRetriever,
)
from agent_utilities.knowledge_graph.retrieval.reasoning_reranker import (
    LexicalRelevanceScorer,
)


@dataclass(frozen=True)
class CaseResult:
    document: str
    query: str
    gold_id: str
    lexical_ids: list[str]
    toc_ids: list[str]
    allowed_ids: list[str]
    lexical_ms: float
    toc_ms: float
    cited_ranges_valid: bool


def _leaves(roots: list[SectionNode]) -> list[SectionNode]:
    return [node for node in iter_sections(roots) if not node.children]


def _paths(roots: list[SectionNode]) -> dict[str, tuple[str, ...]]:
    paths: dict[str, tuple[str, ...]] = {}

    def visit(nodes: list[SectionNode], parent: tuple[str, ...]) -> None:
        for node in nodes:
            path = (*parent, node.title)
            paths[node.node_id] = path
            visit(node.children, path)

    visit(roots, ())
    return paths


def _visible_tree(
    roots: list[SectionNode], allowed_paths: set[tuple[str, ...]]
) -> list[SectionNode]:
    """Prune hidden leaves before either retriever sees the section map."""

    def keep(node: SectionNode, parent: tuple[str, ...]) -> SectionNode | None:
        path = (*parent, node.title)
        children = [child for item in node.children if (child := keep(item, path))]
        if not children and node.children:
            return None
        if not node.children and path not in allowed_paths:
            return None
        return node.model_copy(update={"children": children})

    return [node for item in roots if (node := keep(item, ())) is not None]


def _toc_rank(query: str, roots: list[SectionNode], limit: int) -> list[str]:
    """Rank allowed leaf IDs from title and ancestry; read no section body."""
    scorer = LexicalRelevanceScorer()
    paths = _paths(roots)
    ranked = sorted(
        _leaves(roots),
        key=lambda node: (
            -scorer.score(query, " ".join(paths[node.node_id])),
            paths[node.node_id],
            node.node_id,
        ),
    )
    return [node.node_id for node in ranked[:limit]]


def _case(
    document: str, text: str, roots: list[SectionNode], item: dict[str, Any]
) -> CaseResult:
    all_paths = _paths(roots)
    leaves = _leaves(roots)
    path_to_id = {all_paths[node.node_id]: node.node_id for node in leaves}
    if len(path_to_id) != len(leaves):
        raise ValueError(f"{document}: duplicate leaf title paths")
    gold_path = tuple(item["gold_path"])
    if gold_path not in path_to_id:
        raise ValueError(f"{document}: gold leaf path not found: {gold_path!r}")
    allowed = {tuple(path) for path in item.get("allowed_paths", path_to_id.keys())}
    if gold_path not in allowed or not allowed <= path_to_id.keys():
        raise ValueError(f"{document}: invalid visibility set for {gold_path!r}")
    visible = _visible_tree(roots, allowed)
    visible_ids = {node.node_id for node in _leaves(visible)}
    query = str(item["query"])

    started = time.perf_counter()
    matches = HierarchicalDocumentRetriever().retrieve(
        query, tree=visible, top_k=len(list(iter_sections(visible))), beam_width=3
    )
    lexical_ms = (time.perf_counter() - started) * 1000
    lexical_ids = [match.node_id for match in matches if match.node_id in visible_ids][
        :3
    ]
    citations_ok = all(
        0 <= match.char_start < match.char_end <= len(text)
        for match in matches
        if match.node_id in visible_ids
    ) and all(
        0 <= node.char_start < node.char_end <= len(text) for node in _leaves(visible)
    )

    started = time.perf_counter()
    toc_ids = _toc_rank(query, visible, 3)
    toc_ms = (time.perf_counter() - started) * 1000
    return CaseResult(
        document=document,
        query=query,
        gold_id=path_to_id[gold_path],
        lexical_ids=lexical_ids,
        toc_ids=toc_ids,
        allowed_ids=sorted(visible_ids),
        lexical_ms=lexical_ms,
        toc_ms=toc_ms,
        cited_ranges_valid=citations_ok,
    )


def evaluate(corpus_root: Path, fixture: dict[str, Any]) -> dict[str, Any]:
    results: list[CaseResult] = []
    build_ms = 0.0
    for document in fixture["documents"]:
        relative = Path(document["path"])
        path = (corpus_root / relative).resolve()
        if not path.is_relative_to(corpus_root.resolve()) or not path.is_file():
            raise ValueError(f"document is outside corpus or missing: {relative}")
        text = path.read_text(encoding="utf-8")
        started = time.perf_counter()
        roots = build_section_tree(text, config=SectionTreeConfig(thin=False))
        build_ms += (time.perf_counter() - started) * 1000
        for item in document["cases"]:
            results.append(_case(str(relative), text, roots, item))
    if not results:
        raise ValueError("fixture has no query cases")

    def metrics(name: str) -> dict[str, Any]:
        ranks = [getattr(row, f"{name}_ids") for row in results]
        timings = [getattr(row, f"{name}_ms") for row in results]
        positions = [
            ids.index(row.gold_id) if row.gold_id in ids else None
            for row, ids in zip(results, ranks, strict=True)
        ]
        return {
            "recall_at_1": sum(pos == 0 for pos in positions) / len(results),
            "recall_at_3": sum(pos is not None for pos in positions) / len(results),
            "ndcg_at_3": sum(
                1 / math.log2(pos + 2) for pos in positions if pos is not None
            )
            / len(results),
            "p50_ms": statistics.median(timings),
            "p95_ms": sorted(timings)[math.ceil(0.95 * len(timings)) - 1],
            "invalid_leaf_ids": sum(
                node_id not in row.allowed_ids
                for row, ids in zip(results, ranks, strict=True)
                for node_id in ids
            ),
        }

    return {
        "cases": len(results),
        "tree_build_ms": build_ms,
        "lexical": metrics("lexical"),
        "toc_leaf": metrics("toc"),
        "citations_valid": all(row.cited_ranges_valid for row in results),
        "rows": [row.__dict__ for row in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--corpus-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    print(json.dumps(evaluate(args.corpus_root, fixture), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
