"""Characterization for ``TaskManagerMixin._run_relevance_sweep`` (CXA-AU-03).

No existing test exercised this function's actual scoring logic before this
lane's refactor (CCN 54 -> decomposed into ``_relevance_sweep_defer_check``,
``_relevance_sweep_target_centroid``, ``_relevance_sweep_unique_papers``,
``_relevance_sweep_repo_set``, ``_score_relevance_paper``,
``_score_relevance_repo``, and small shared helpers). These tests pin the
three branches lizard's CCN counted: the throttle-defer early return, the
no-target-data early return, and the full paper+repo scoring path (including
the per-item try/except that must swallow a query failure and keep going).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin


class _FakeRelevanceEngine(TaskManagerMixin):
    """Minimal stand-in: only what ``_run_relevance_sweep`` touches on ``self``."""

    def __init__(self, responses: list[tuple[str, list[dict]]]) -> None:
        self._responses = responses
        self.persisted: list[tuple] = []
        self.queries: list[str] = []

    def query_cypher(self, query: str, params: dict | None = None) -> list[dict]:
        self.queries.append(query)
        for substring, rows in self._responses:
            if substring in query:
                return rows
        raise AssertionError(f"unexpected query in characterization stub: {query}")

    def _persist_relevance_score(self, *args, **kwargs) -> None:
        self.persisted.append(args)


def _not_throttled():
    return MagicMock(should_yield_background=False)


@pytest.mark.asyncio
async def test_relevance_sweep_defers_when_bulk_ingest_active():
    engine = _FakeRelevanceEngine(responses=[])
    with patch(
        "agent_utilities.core.background_throttle.get_throttle",
        return_value=MagicMock(should_yield_background=True),
    ):
        result = await engine._run_relevance_sweep("job-1", "some-repo")
    assert result == {
        "status": "deferred",
        "reason": "bulk_ingest_or_foreground",
        "job_id": "job-1",
    }
    # Deferred before any Cypher query is issued.
    assert engine.queries == []


@pytest.mark.asyncio
async def test_relevance_sweep_reports_no_target_data_when_no_embeddings_found():
    engine = _FakeRelevanceEngine(
        responses=[
            ("c.embedding AS emb LIMIT 200", []),  # primary target Code query: no rows
            ("a.target_path CONTAINS $name", []),  # fallback Article query: no rows
        ]
    )
    with patch(
        "agent_utilities.core.background_throttle.get_throttle",
        return_value=_not_throttled(),
    ):
        result = await engine._run_relevance_sweep("job-1", "ghost-repo")
    assert result["status"] == "no_target_data"
    assert result["target"] == "ghost-repo"


@pytest.mark.asyncio
async def test_relevance_sweep_scores_papers_and_repos_and_sorts_descending():
    responses = [
        ("c.embedding AS emb LIMIT 200", [{"id": "c1", "emb": [1.0, 0.0]}]),  # target centroid
        (
            "DISTINCT a.target_path",
            [{"paper_path": "papers/one.pdf"}],
        ),  # unique papers
        (
            "c.file_path AS path LIMIT 2000",
            [
                {
                    "id": "code1",
                    # WD1-GATE-01A: this used to be rooted under a home
                    # directory, which `check_tracked_privacy.py` flags as a
                    # machine-specific home path in runtime source (it scans
                    # the tracked .py source text itself, not just the value
                    # at runtime). Only the 7-segment shape and the
                    # "agent-packages" + differing-repo-name content matter to
                    # `_relevance_sweep_repo_set`'s `parts[5] if
                    # "agent-packages" in path else parts[4]` indexing below —
                    # a differently-rooted path reproduces both identically
                    # without embedding a literal the scanner matches.
                    "path": "/srv/a/b/agent-packages/otherrepo/file.py",
                }
            ],
        ),  # repo set: split("/") has 7 parts, "agent-packages" present -> parts[5]
        # == "otherrepo" (must differ from target_codebase, or the sweep's own
        # self-exclusion filters it out of repo_set entirely)
        (
            "a.target_path = $path",
            [
                {
                    "id": "a1",
                    "emb": [1.0, 0.0],
                    "content": "an agent orchestration paper about mcp and memory",
                }
            ],
        ),  # per-paper chunks — perfectly aligned with the target centroid
        (
            "c.file_path CONTAINS $name",  # repo chunks (Step 5); the target
            # query above matched on "c.embedding AS emb LIMIT 200" first, so this only matches
            # the repo-chunks query, which requests LIMIT 100.
            [{"id": "code1", "emb": [0.0, 1.0], "content": "unrelated code"}],
        ),
    ]
    engine = _FakeRelevanceEngine(responses=responses)
    with patch(
        "agent_utilities.core.background_throttle.get_throttle",
        return_value=_not_throttled(),
    ):
        result = await engine._run_relevance_sweep("job-1", "reponame")

    assert result["status"] == "completed"
    assert result["target_codebase"] == "reponame"
    assert result["items_scored"] == 2
    assert result["type"] == "relevance_sweep"
    assert "scored_at" in result

    ids = {item["id"] for item in result["top_10"]}
    assert ids == {"paper:one", "repo:otherrepo"}

    # The paper's embedding is identical to the target centroid (perfect
    # cosine similarity) and its content matches several concept/arch
    # keywords, so it must outrank the orthogonal, keyword-free repo.
    assert result["top_10"][0]["id"] == "paper:one"
    assert result["top_10"][0]["score"] >= result["top_10"][1]["score"]

    # Both items were persisted via _persist_relevance_score.
    assert len(engine.persisted) == 2
    persisted_ids = {call[0] for call in engine.persisted}
    assert persisted_ids == {"paper:one", "repo:otherrepo"}


@pytest.mark.asyncio
async def test_relevance_sweep_swallows_per_paper_scoring_failure_and_continues():
    """A query failure scoring ONE paper must not abort the whole sweep --
    the original code wraps each paper/repo iteration in its own try/except
    and logs a warning rather than propagating."""

    class _RaisingOnChunksEngine(_FakeRelevanceEngine):
        def query_cypher(self, query: str, params: dict | None = None):
            if "a.target_path = $path" in query:
                raise RuntimeError("boom")
            return super().query_cypher(query, params)

    responses = [
        ("c.embedding AS emb LIMIT 200", [{"id": "c1", "emb": [1.0, 0.0]}]),
        ("DISTINCT a.target_path", [{"paper_path": "papers/broken.pdf"}]),
        ("c.file_path AS path LIMIT 2000", []),
    ]
    engine = _RaisingOnChunksEngine(responses=responses)
    with patch(
        "agent_utilities.core.background_throttle.get_throttle",
        return_value=_not_throttled(),
    ):
        result = await engine._run_relevance_sweep("job-1", "reponame")

    assert result["status"] == "completed"
    assert result["items_scored"] == 0
    assert result["top_10"] == []
    assert engine.persisted == []


# ---------------------------------------------------------------------------
# BUG-CX-062: ``job_id`` is accepted by ``_run_relevance_sweep`` but never
# forwarded anywhere -- not into the log lines, not into the returned result
# dict on any of its three exit paths (deferred / no_target_data /
# completed). That breaks job correlation: a caller (or anything reading a
# persisted task result) that only has the result payload cannot tell which
# job produced it. The fix threads ``job_id`` into every returned dict.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_relevance_sweep_completed_result_carries_the_job_id():
    responses = [
        ("c.embedding AS emb LIMIT 200", [{"id": "c1", "emb": [1.0, 0.0]}]),
        ("DISTINCT a.target_path", []),
        ("c.file_path AS path LIMIT 2000", []),
    ]
    engine = _FakeRelevanceEngine(responses=responses)
    with patch(
        "agent_utilities.core.background_throttle.get_throttle",
        return_value=_not_throttled(),
    ):
        result = await engine._run_relevance_sweep("job-xyz-789", "reponame")

    assert result["status"] == "completed"
    assert result["job_id"] == "job-xyz-789", (
        "job_id was accepted but not forwarded into the completed-sweep result"
    )


@pytest.mark.asyncio
async def test_relevance_sweep_no_target_data_result_carries_the_job_id():
    engine = _FakeRelevanceEngine(
        responses=[
            ("c.embedding AS emb LIMIT 200", []),
            ("a.target_path CONTAINS $name", []),
        ]
    )
    with patch(
        "agent_utilities.core.background_throttle.get_throttle",
        return_value=_not_throttled(),
    ):
        result = await engine._run_relevance_sweep("job-abc-123", "ghost-repo")

    assert result["status"] == "no_target_data"
    assert result["job_id"] == "job-abc-123", (
        "job_id was accepted but not forwarded into the no-target-data result"
    )


@pytest.mark.asyncio
async def test_relevance_sweep_deferred_result_carries_the_job_id():
    engine = _FakeRelevanceEngine(responses=[])
    with patch(
        "agent_utilities.core.background_throttle.get_throttle",
        return_value=MagicMock(should_yield_background=True),
    ):
        result = await engine._run_relevance_sweep("job-def-456", "some-repo")

    assert result["status"] == "deferred"
    assert result["job_id"] == "job-def-456", (
        "job_id was accepted but not forwarded into the deferred result"
    )
