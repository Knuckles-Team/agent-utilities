"""GAP-2 wiring: the ``skill_workflows`` background job must also ingest the
ATOMIC-skill corpus, not just workflows.

Root cause this pins: ``package_install_ingest.py::_ingest_skills_leg`` pairs
``ingest_skill_workflows`` + ``ingest_atomic_skills`` on ONE watermarked
``:Schedule`` tick, but that schedule fires only on an ``install-manifest.json``
change from the universal-installer -- a deployment whose skill corpus is
baked into the image (never routed through the installer) produces no such
manifest, so the atomic leg's only wired call site was unreachable there. The
``skill_workflows`` MCP action / ``task_type`` was already this corpus's
manual, on-demand, non-manifest-gated full-sweep entrypoint for workflows;
this test pins that it now reaches BOTH legs, same reused primitives
(``ingest_skill_workflows``, ``ingest_atomic_skills``), no new ingestion path.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin


class _FakeTaskEngine(TaskManagerMixin):
    """Minimal stand-in: only what ``_run_background_task``'s
    ``skill_workflows`` branch (and the shared ``finally: self._checkpoint_db()``
    every task type runs through) touches on ``self``."""

    def __init__(self) -> None:
        self.updates: list[tuple[str, str, dict]] = []
        self.backend = None  # no WAL-checkpointable backend in this fake

    def _update_task_status(self, job_id: str, status: str, payload: dict) -> None:
        self.updates.append((job_id, status, payload))


@pytest.mark.asyncio
async def test_skill_workflows_job_also_runs_atomic_skill_ingest():
    engine = _FakeTaskEngine()

    with (
        patch(
            "agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest.ingest_skill_workflows",
            return_value={"workflows": 3, "steps": 9, "skill_links": 2, "skipped": 0, "errors": 0},
        ) as workflows_mock,
        patch(
            "agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest.ingest_atomic_skills",
            return_value={"skills": 5, "skipped": 1, "not_skill": 2, "errors": 0},
        ) as atomic_mock,
    ):
        await engine._run_background_task(
            job_id="job:1",
            target=Path("universal-skills"),
            is_codebase=False,
            task_type="skill_workflows",
        )

    workflows_mock.assert_called_once()
    atomic_mock.assert_called_once()
    # Both legs share the SAME resolved corpus root argument.
    assert workflows_mock.call_args.kwargs.get("root") == atomic_mock.call_args.kwargs.get("root")

    job_id, status, payload = engine.updates[-1]
    assert job_id == "job:1"
    assert status == "completed"
    assert payload["workflows"] == 3
    assert payload["atomic_skills"] == 5
    assert payload["atomic_skipped"] == 1
    assert payload["atomic_not_skill"] == 2
    assert payload["atomic_errors"] == 0


@pytest.mark.asyncio
async def test_skill_workflows_job_survives_atomic_leg_failure():
    """One bad leg must not swallow the other -- mirrors
    ``_ingest_skills_leg``'s own fail-independently contract."""
    engine = _FakeTaskEngine()

    with (
        patch(
            "agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest.ingest_skill_workflows",
            return_value={"workflows": 1, "steps": 1, "skill_links": 0, "skipped": 0, "errors": 0},
        ) as workflows_mock,
        patch(
            "agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest.ingest_atomic_skills",
            side_effect=RuntimeError("boom"),
        ),
    ):
        await engine._run_background_task(
            job_id="job:2",
            target=Path("universal-skills"),
            is_codebase=False,
            task_type="skill_workflows",
        )

    workflows_mock.assert_called_once()
    job_id, status, payload = engine.updates[-1]
    assert status == "completed"
    assert payload["workflows"] == 1
    assert payload["atomic_skills"] == 0
    assert payload["atomic_errors"] == 1
