#!/usr/bin/env python3
"""Offline, visibility-matched section retrieval comparison for EH-652.

The candidate is a no-training table-of-contents leaf ranker. It establishes a
cheap ablation before a corpus-specific STAIR model is considered. This script
does not change the served retrieval path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import time
from collections import Counter
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
    bm25_ids: list[str]
    hybrid_ids: list[str]
    allowed_ids: list[str]
    lexical_ms: float
    toc_ms: float
    bm25_ms: float
    hybrid_ms: float
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


_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _body_rank(
    query: str, roots: list[SectionNode], limit: int
) -> tuple[list[str], list[str], float, float]:
    """CPU-only BM25 body baseline and BM25+ToC lexical fusion.

    Both rank the same visibility-pruned leaves. This is an offline lexical
    surrogate for a hybrid retriever, not the served vector+graph implementation.
    """
    started = time.perf_counter()
    leaves = _leaves(roots)
    paths = _paths(roots)
    docs = [
        Counter(_tokens(" ".join(paths[node.node_id]) + " " + node.text))
        for node in leaves
    ]
    lengths = [sum(words.values()) for words in docs]
    avg_length = sum(lengths) / len(lengths) if lengths else 1.0
    query_terms = set(_tokens(query))
    df = Counter(term for words in docs for term in words)
    scored: list[tuple[float, SectionNode]] = []
    for node, words, length in zip(leaves, docs, lengths, strict=True):
        body_score = 0.0
        for term in query_terms:
            count = words[term]
            if not count:
                continue
            idf = math.log(1 + (len(leaves) - df[term] + 0.5) / (df[term] + 0.5))
            body_score += (
                idf * count * 2.2 / (count + 1.2 * (0.25 + 0.75 * length / avg_length))
            )
        scored.append((body_score, node))

    def key(item: tuple[float, SectionNode]) -> tuple[tuple[str, ...], str]:
        return paths[item[1].node_id], item[1].node_id

    bm25 = sorted(scored, key=lambda item: (-item[0], *key(item)))
    bm25_ms = (time.perf_counter() - started) * 1000
    toc = LexicalRelevanceScorer()
    fused = [
        (body_score, toc.score(query, " ".join(paths[node.node_id])), node)
        for body_score, node in scored
    ]
    max_body = max((item[0] for item in scored), default=0.0)
    hybrid = sorted(
        fused,
        key=lambda item: (
            -(0.7 * item[0] / max_body + 0.3 * item[1]) if max_body else -item[1],
            paths[item[2].node_id],
            item[2].node_id,
        ),
    )
    return (
        [item[1].node_id for item in bm25[:limit]],
        [item[2].node_id for item in hybrid[:limit]],
        bm25_ms,
        (time.perf_counter() - started) * 1000,
    )


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
    bm25_ids, hybrid_ids, bm25_ms, hybrid_ms = _body_rank(query, visible, 3)
    return CaseResult(
        document=document,
        query=query,
        gold_id=path_to_id[gold_path],
        lexical_ids=lexical_ids,
        toc_ids=toc_ids,
        bm25_ids=bm25_ids,
        hybrid_ids=hybrid_ids,
        allowed_ids=sorted(visible_ids),
        lexical_ms=lexical_ms,
        toc_ms=toc_ms,
        bm25_ms=bm25_ms,
        hybrid_ms=hybrid_ms,
        cited_ranges_valid=citations_ok,
    )


def _update_cost(
    text: str, roots: list[SectionNode], update: dict[str, Any]
) -> dict[str, Any]:
    """Measure a local body edit, whole-tree rebuild, and leaf-index refresh.

    The edit is inserted at the end of the named leaf's span. This is an
    intentionally conservative full rebuild baseline, not an incremental index.
    """
    paths = _paths(roots)
    leaves = {paths[node.node_id]: node for node in _leaves(roots)}
    target_path = tuple(update["leaf_path"])
    if target_path not in leaves:
        raise ValueError(f"update leaf not found: {target_path!r}")
    addition = str(update["append_text"])
    if not addition.strip() or addition.lstrip().startswith("#"):
        raise ValueError("update must append non-heading body text")
    target = leaves[target_path]
    revised = text[: target.char_end] + "\n" + addition + "\n" + text[target.char_end :]
    started = time.perf_counter()
    next_roots = build_section_tree(revised, config=SectionTreeConfig(thin=False))
    rebuild_ms = (time.perf_counter() - started) * 1000
    next_paths = _paths(next_roots)
    next_leaves = {next_paths[node.node_id]: node for node in _leaves(next_roots)}
    if next_leaves.keys() != leaves.keys():
        raise ValueError("body-only edit changed the leaf identity set")
    changed = [
        path
        for path in leaves
        if hashlib.sha256(leaves[path].text.encode()).digest()
        != hashlib.sha256(next_leaves[path].text.encode()).digest()
    ]
    if target_path not in changed:
        raise ValueError("declared update did not change its target leaf")
    started = time.perf_counter()
    _body_rank("index refresh", next_roots, len(next_leaves))
    body_recompute_probe_ms = (time.perf_counter() - started) * 1000
    return {
        "leaf_path": list(target_path),
        "bytes_added": len(revised.encode()) - len(text.encode()),
        "leaf_count": len(leaves),
        "changed_leaf_count": len(changed),
        "toc_leaf_keys_changed": len(leaves.keys() ^ next_leaves.keys()),
        "rebuild_ms": rebuild_ms,
        "bm25_full_recompute_probe_ms": body_recompute_probe_ms,
    }


def evaluate(corpus_root: Path, fixture: dict[str, Any]) -> dict[str, Any]:
    results: list[CaseResult] = []
    build_ms = 0.0
    updates: list[dict[str, Any]] = []
    for document in fixture["documents"]:
        relative = Path(document["path"])
        path = (corpus_root / relative).resolve()
        if not path.is_relative_to(corpus_root.resolve()) or not path.is_file():
            raise ValueError(f"document is outside corpus or missing: {relative}")
        text = path.read_text(encoding="utf-8")
        if expected := document.get("sha256"):
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            if observed != expected:
                raise ValueError(f"document digest changed: {relative}")
        started = time.perf_counter()
        roots = build_section_tree(text, config=SectionTreeConfig(thin=False))
        build_ms += (time.perf_counter() - started) * 1000
        for item in document["cases"]:
            results.append(_case(str(relative), text, roots, item))
        for update in document.get("updates", []):
            updates.append(
                {"document": str(relative), **_update_cost(text, roots, update)}
            )
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
        "bm25_body": metrics("bm25"),
        "hybrid_lexical": metrics("hybrid"),
        "citations_valid": all(row.cited_ranges_valid for row in results),
        "updates": updates,
        "baseline_note": "hybrid_lexical is CPU-only BM25 body + ToC; no vector/model or served graph arm",
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
