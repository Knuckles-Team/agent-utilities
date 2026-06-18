"""Shared per-paper research grading + prioritized enqueue (CONCEPT:KG-2.114/2.121).

The single research decision every feed source funnels through. Extracted from the
old scholarx-only ``run_rss_feed_screen`` so the native ``rss`` connector, the
ScholarX arXiv source, and FreshRSS-arXiv items all grade and enqueue the SAME way:

  keyword score (``score_paper``) + novelty probe (``_paper_novelty``; ``None`` on
  embedder outage → keyword-only) →
    score ≥ relevant  → enqueue a ``research_paper_fetch`` :Task, ``prio_bucket``
                         derived from the grade (best-graded fetched FIRST)
    score ≥ marginal  → abstract-only ingest inline
    else              → rejected
"""

from __future__ import annotations

import asyncio
from typing import Any


def _run_coro(coro: Any) -> Any:
    """Run an async coroutine from sync code, whether or not a loop is running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()


def grade_and_enqueue_paper(engine: Any, paper: dict[str, Any]) -> dict[str, Any]:
    """Grade one paper and enqueue a prioritized fetch / abstract ingest.

    ``paper`` keys: ``id`` (canonical, e.g. ``arxiv:2601.0009``), ``title``,
    ``abstract``, ``authors``, ``url``, ``pdf_url``. Returns
    ``{"tier", "score", "bucket"?}``. The ``research_paper_fetch`` enqueue uses the
    queue's own target-dedup (``skip_dedupe=False``) so re-grading the same paper
    before its fetch completes does not pile up duplicate tasks.
    """
    from agent_utilities.automation.research_pipeline import ResearchPipelineRunner

    runner = ResearchPipelineRunner(engine=engine)
    cfg = runner.config
    relevant, marginal = cfg.relevant_threshold, cfg.marginal_threshold

    aid = str(paper.get("id") or "").strip()
    title = paper.get("title", "") or ""
    abstract = paper.get("abstract", "") or ""
    authors = paper.get("authors", []) or []
    url = paper.get("url", "") or ""

    score, domains = runner.score_paper(title, abstract)
    novelty = runner._paper_novelty(title, abstract)  # None on embedder outage
    # Already-built (low-novelty) high-keyword paper → demote to memory-only.
    if novelty is not None and novelty < 0.25 and score >= relevant:
        score = marginal

    if score >= relevant:
        # Higher grade → lower (more urgent) bucket → fetched first.
        bucket = 0 if score >= 2 * relevant else 1
        engine.submit_task(
            target_path=url or paper.get("pdf_url", "") or aid,
            is_codebase=False,
            provenance={"source_url": url},
            task_type="research_paper_fetch",
            skip_dedupe=False,
            priority=bucket,
            extra_meta={
                "paper": {
                    "id": aid,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "url": url,
                    "pdf_url": paper.get("pdf_url", ""),
                    "score": score,
                    "domains": domains,
                }
            },
        )
        return {"tier": "queued_full", "score": round(score, 2), "bucket": bucket}

    if score >= marginal:
        _run_coro(
            runner.ingest_paper_marginal(
                aid,
                title,
                abstract,
                authors,
                source_url=url,
                relevance_score=score,
                domains=domains,
            )
        )
        return {"tier": "ingested_marginal", "score": round(score, 2)}

    return {"tier": "rejected", "score": round(score, 2)}
