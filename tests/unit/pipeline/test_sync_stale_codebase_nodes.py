"""Regression coverage for the stale codebase-node sweep."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent_utilities.knowledge_graph.pipeline.phases.sync import (
    _sweep_stale_codebase_nodes,
)


def _context() -> SimpleNamespace:
    return SimpleNamespace(
        metadata={"ingestion_timestamp": "2026-08-29T12:00:00+00:00"},
        config=SimpleNamespace(workspace_path="/workspace/project"),
    )


def test_stale_sweep_reads_exact_set_then_deletes_by_id_batches() -> None:
    """The stale predicate remains on a read; writes use the native subset."""
    backend = MagicMock()
    backend.execute_read.return_value = [
        {"id": "old-file"},
        {"id": "missing-timestamp"},
        {"id": "old-file"},
    ]

    _sweep_stale_codebase_nodes(_context(), backend)

    backend.execute_read.assert_called_once_with(
        "MATCH (n:Code) WHERE n.file_path STARTS WITH $workspace_path AND "
        "(n.last_seen_timestamp < $ts OR n.last_seen_timestamp IS NULL) "
        "RETURN n.id AS id",
        {
            "workspace_path": "/workspace/project",
            "ts": "2026-08-29T12:00:00+00:00",
        },
    )
    backend.execute.assert_called_once_with(
        "MATCH (n:Code) WHERE n.id IN $ids DETACH DELETE n",
        {"ids": ["old-file", "missing-timestamp"]},
    )


def test_stale_sweep_fails_closed_without_read_capability() -> None:
    """An unsupported backend must not receive a guessed destructive write."""
    backend = SimpleNamespace(execute=MagicMock())

    _sweep_stale_codebase_nodes(_context(), backend)

    backend.execute.assert_not_called()


def test_stale_sweep_skips_malformed_read_results() -> None:
    """A degraded read must not turn into a broad delete."""
    backend = MagicMock()
    backend.execute_read.return_value = {"id": "should-not-delete"}

    _sweep_stale_codebase_nodes(_context(), backend)

    backend.execute.assert_not_called()


def test_stale_sweep_skips_partial_read_results() -> None:
    """A single malformed row must not cause a partial destructive write."""
    backend = MagicMock()
    backend.execute_read.return_value = [{"id": "valid"}, object()]

    _sweep_stale_codebase_nodes(_context(), backend)

    backend.execute.assert_not_called()
