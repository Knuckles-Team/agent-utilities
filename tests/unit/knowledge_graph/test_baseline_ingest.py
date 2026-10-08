"""Baseline ingest after daemon boot (spec: baseline-ingestion)."""

from __future__ import annotations

import contextvars
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from agent_utilities.knowledge_graph.ingestion import baseline_ingest as bi

_SETTINGS = "agent_utilities.knowledge_graph.ingestion.baseline_ingest._settings"


def _settings(**overrides: Any) -> SimpleNamespace:
    values = {
        "kg_baseline_ingest": True,
        "kg_baseline_skill_providers": "agent-utilities,graph-os,universal-skills",
        "kg_baseline_codebases": "none",
        "kg_baseline_max_codebases": 32,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class RecordingEngine:
    def __init__(self, fail_targets: set[str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._fail = fail_targets or set()

    def submit_task(self, **kwargs: Any) -> str:
        if kwargs["target_path"] in self._fail:
            raise ValueError("outside workspace")
        self.calls.append(kwargs)
        return kwargs["job_id"]


def _workspace(tmp_path: Path) -> dict[str, Any]:
    for name in ("agent-utilities", "graph-os", "universal-skills", "gitlab-api"):
        sub = "skills/" if name == "universal-skills" else ""
        sub = "agents/" if name == "gitlab-api" else sub
        (tmp_path / "agent-packages" / f"{sub}{name}").mkdir(parents=True)
    return {
        "path": str(tmp_path),
        "subdirectories": {
            "agent-packages": {
                "repositories": [
                    {"url": "https://example.invalid/agent-utilities.git"},
                    {"url": "https://example.invalid/graph-os.git"},
                    {"url": "https://example.invalid/.github.git"},
                    {"url": "https://example.invalid/not-cloned.git"},
                ],
                "subdirectories": {
                    "skills": {
                        "repositories": [
                            {"url": "https://example.invalid/universal-skills.git"}
                        ]
                    },
                    "agents": {
                        "repositories": [
                            {"url": "https://example.invalid/gitlab-api.git"}
                        ]
                    },
                },
            }
        },
    }


def test_plan_orders_fast_before_medium_and_covers_every_leg(tmp_path: Path) -> None:
    data = _workspace(tmp_path)
    with (
        patch(_SETTINGS, return_value=_settings(kg_baseline_codebases="core")),
        patch.object(bi, "_load_workspace_manifest", return_value=data),
    ):
        items = bi.plan_baseline()
    legs = [item.leg for item in items]
    assert legs[:4] == ["prompts", "skills", "skills", "skills"]
    assert [item.name for item in items if item.leg == "codebase"] == [
        "agent-utilities",
        "graph-os",
        "universal-skills",
    ]
    classes = [item.queue_class for item in items]
    assert classes == sorted(classes, key=["fast", "medium", "slow_heavy"].index)
    assert {item.target for item in items if item.leg == "skills"} == {
        "skill-provider:agent-utilities",
        "skill-provider:graph-os",
        "skill-provider:universal-skills",
    }


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        ("all", ["agent-utilities", "graph-os", "universal-skills", "gitlab-api"]),
        ("graph-os, gitlab-api", ["graph-os", "gitlab-api"]),
        ("none", []),
    ],
)
def test_workspace_scope(tmp_path: Path, scope: str, expected: list[str]) -> None:
    names = [name for name, _ in bi.workspace_repositories(_workspace(tmp_path), scope)]
    assert sorted(names) == sorted(expected)


def test_codebase_cap_bounds_the_plan(tmp_path: Path) -> None:
    data = _workspace(tmp_path)
    with (
        patch(
            _SETTINGS,
            return_value=_settings(
                kg_baseline_codebases="all", kg_baseline_max_codebases=1
            ),
        ),
        patch.object(bi, "_load_workspace_manifest", return_value=data),
    ):
        items = bi.plan_baseline()
    assert sum(item.leg == "codebase" for item in items) == 1


def test_workspace_scan_failure_keeps_skills_and_prompts() -> None:
    with (
        patch(_SETTINGS, return_value=_settings(kg_baseline_codebases="core")),
        patch.object(bi, "_load_workspace_manifest", side_effect=OSError("denied")),
    ):
        items = bi.plan_baseline()
    assert {item.leg for item in items} == {"prompts", "skills"}


def test_enqueue_stamps_tier_metadata_and_isolates_rejections() -> None:
    items = [
        bi._prompt_item(),
        bi.BaselineItem("codebase", "far", "/elsewhere/far", "codebase", True),
        bi.BaselineItem("codebase", "near", "/ws/near", "codebase", True),
    ]
    engine = RecordingEngine(fail_targets={"/elsewhere/far"})
    report = bi.enqueue_baseline(engine, items, "boot1")
    assert [entry["name"] for entry in report["queued"]] == ["prompt-library", "near"]
    assert report["rejected"] == [
        {
            "leg": "codebase",
            "name": "far",
            "queue_class": "medium",
            "reason": "ValueError",
        }
    ]
    prompt, code = engine.calls
    assert prompt["task_type"] == "scheduled_job"
    assert prompt["priority"] == 1
    assert prompt["extra_meta"]["payload"] == {
        "kind": "maint",
        "ref": "baseline_prompts",
    }
    assert prompt["extra_meta"]["baseline"]["stage"] == "S1"
    assert code["priority"] == 2
    assert code["is_codebase"] is True
    assert code["job_id"] == "baseline-boot1-codebase-near"
    assert code["extra_meta"]["baseline"] == {
        "leg": "codebase",
        "name": "near",
        "queue_class": "medium",
        "stage": "S2",
    }


def test_run_never_raises() -> None:
    with patch.object(bi, "plan_baseline", side_effect=RuntimeError("boom")):
        assert bi.run_baseline_ingest(RecordingEngine()) is None


def test_start_runs_once_on_an_authorized_thread() -> None:
    started: list[str] = []

    class FakeThread:
        def __init__(self, target, args) -> None:
            self._target, self._args = target, args

        def start(self) -> None:
            started.append("start")

    def _thread(_session, target, *, name, args):
        assert name == "KG-Baseline-Ingest"
        return FakeThread(target, args)

    engine = SimpleNamespace()
    with (
        patch(_SETTINGS, return_value=_settings()),
        patch(
            "agent_utilities.knowledge_graph.core.engine_tasks._authorized_background_thread",
            side_effect=_thread,
        ),
    ):
        assert bi.start_baseline_ingest(engine, object()) is True
        assert bi.start_baseline_ingest(engine, object()) is False
    assert started == ["start"]


def test_start_respects_the_switch_and_swallows_launch_errors() -> None:
    with patch(_SETTINGS, return_value=_settings(kg_baseline_ingest=False)):
        assert bi.start_baseline_ingest(SimpleNamespace(), object()) is False
    with (
        patch(_SETTINGS, return_value=_settings()),
        patch(
            "agent_utilities.knowledge_graph.core.engine_tasks._authorized_background_thread",
            side_effect=PermissionError("no session"),
        ),
    ):
        assert bi.start_baseline_ingest(SimpleNamespace(), object()) is False


def test_skill_corpus_root_resolution() -> None:
    assert bi.resolve_skill_corpus_root("universal-skills") is None
    assert bi.resolve_skill_corpus_root("/explicit/root") == "/explicit/root"
    with patch(
        "agent_utilities.core.providers.iter_provider_dirs",
        return_value=[("graph-os", Path("/pkg/graph_os/skills"))],
    ):
        assert bi.resolve_skill_corpus_root("skill-provider:graph-os") == str(
            Path("/pkg/graph_os/skills")
        )
        with pytest.raises(LookupError):
            bi.resolve_skill_corpus_root("skill-provider:absent")


def test_prompt_library_runs_off_loop_with_the_caller_context() -> None:
    marker: contextvars.ContextVar[str] = contextvars.ContextVar("marker")
    seen: dict[str, Any] = {}

    async def _ingest() -> None:
        seen["marker"] = marker.get()
        seen["thread"] = threading.current_thread().name

    marker.set("verified")
    with patch(
        "agent_utilities.agent.registry_builder.ingest_prompts_to_graph",
        side_effect=_ingest,
    ):
        bi.ingest_prompt_library()
    assert seen["marker"] == "verified"
    assert seen["thread"].startswith("kg-baseline")


def test_prompt_tick_is_an_allowlisted_maintenance_ref() -> None:
    from agent_utilities.core.schedule_engine import _MAINTENANCE_REF_ALLOWLIST
    from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin

    assert bi.PROMPTS_MAINTENANCE_REF in _MAINTENANCE_REF_ALLOWLIST
    assert callable(getattr(TaskManagerMixin, f"_tick_{bi.PROMPTS_MAINTENANCE_REF}"))
