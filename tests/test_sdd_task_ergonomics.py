"""Tests for SDD task-management ergonomics (CONCEPT:ORCH-1.47)."""

from __future__ import annotations

import pytest

from agent_utilities.models.sdd import Task, Tasks
from agent_utilities.sdd import SDDManager


def _t(tid: str, *, deps=None, status="pending", priority="medium", subtasks=None):
    return Task(
        id=tid,
        title=f"task {tid}",
        depends_on=deps or [],
        status=status,
        priority=priority,
        subtasks=subtasks or [],
    )


# --------------------------- pure-logic: next_task --------------------------- #
def test_next_task_respects_dependencies():
    tasks = Tasks(tasks=[_t("1"), _t("2", deps=["1"])])
    # 2 is blocked by 1, so 1 comes first.
    assert tasks.next_task().id == "1"


def test_next_task_priority_breaks_ties():
    tasks = Tasks(
        tasks=[
            _t("1", priority="low"),
            _t("2", priority="critical"),
            _t("3", priority="high"),
        ]
    )
    assert tasks.next_task().id == "2"


def test_next_task_prefers_subtasks_of_in_progress_parent():
    parent = _t("1", status="in_progress", subtasks=[_t("1.1"), _t("1.2")])
    tasks = Tasks(tasks=[parent, _t("2", priority="critical")])
    # Even though task 2 is critical, the in-progress parent's subtask wins.
    assert tasks.next_task().id == "1.1"


def test_next_task_none_when_all_done():
    tasks = Tasks(tasks=[_t("1", status="completed")])
    assert tasks.next_task() is None


def test_next_task_skips_blocked_by_unfinished_dep():
    tasks = Tasks(tasks=[_t("1", status="in_progress"), _t("2", deps=["1"])])
    # 1 is in progress (still actionable), 2 blocked -> 1.
    assert tasks.next_task().id == "1"


# --------------------------- pure-logic: cycles ------------------------------ #
def test_detect_cycles_finds_cycle():
    tasks = Tasks(tasks=[_t("1", deps=["2"]), _t("2", deps=["1"])])
    cycles = tasks.detect_cycles()
    assert cycles
    assert set(cycles[0]) >= {"1", "2"}


def test_detect_cycles_none_for_dag():
    tasks = Tasks(tasks=[_t("1"), _t("2", deps=["1"]), _t("3", deps=["1", "2"])])
    assert tasks.detect_cycles() == []


def test_validate_dependencies_flags_problems():
    tasks = Tasks(
        tasks=[_t("1", deps=["1"]), _t("2", deps=["99"]), _t("3", deps=["4"]),
               _t("4", deps=["3"])]
    )
    errors = tasks.validate_dependencies()
    assert any("itself" in e for e in errors)
    assert any("unknown task 99" in e for e in errors)
    assert any(e.startswith("dependency cycle") for e in errors)


# --------------------------- SDDManager: persistence ------------------------- #
def test_parse_prd_structural(tmp_path):
    mgr = SDDManager(tmp_path)
    prd = """# Build login
1. Create the user model
2. Add the auth endpoint
- Write integration tests
"""
    tasks = mgr.parse_prd(prd, "auth")
    assert len(tasks.tasks) == 4
    # Sequential dependency chain.
    assert tasks.tasks[0].depends_on == []
    assert tasks.tasks[1].depends_on == [tasks.tasks[0].id]
    # Round-trips from disk.
    loaded = mgr.get_tasks("auth")
    assert loaded is not None
    assert len(loaded.tasks) == 4


def test_analyze_complexity_heuristic_persists_report(tmp_path):
    mgr = SDDManager(tmp_path)
    long_desc = "word " * 120
    tasks = Tasks(
        feature_id="f",
        tasks=[
            Task(id="1", title="simple", description="tiny"),
            Task(id="2", title="big", description=long_desc, depends_on=["1"],
                 file_paths=["a.py", "b.py"]),
        ],
    )
    mgr.save(tasks, "f")
    report = mgr.analyze_complexity("f")
    scores = {r["task_id"]: r["complexity_score"] for r in report["complexity_analysis"]}
    assert scores["2"] > scores["1"]
    assert (tmp_path / ".specify" / "reports" / "task-complexity-f.json").exists()
    # Scores written back onto the tasks.
    reloaded = mgr.get_tasks("f")
    assert reloaded.tasks[1].complexity_score > 0


def test_analyze_complexity_injectable_scorer(tmp_path):
    mgr = SDDManager(tmp_path)
    mgr.save(Tasks(feature_id="f", tasks=[Task(id="1", title="x")]), "f")

    def fake_scorer(task):
        return {"complexity_score": 9.0, "recommended_subtasks": 5,
                "expansion_prompt": "split it"}

    report = mgr.analyze_complexity("f", scorer=fake_scorer)
    assert report["complexity_analysis"][0]["complexity_score"] == 9.0
    assert report["complexity_analysis"][0]["recommended_subtasks"] == 5


def test_next_task_raises_on_cycle(tmp_path):
    mgr = SDDManager(tmp_path)
    mgr.save(
        Tasks(feature_id="f", tasks=[_t("1", deps=["2"]), _t("2", deps=["1"])]), "f"
    )
    with pytest.raises(ValueError, match="dependency cycle"):
        mgr.next_task("f")


def test_set_task_status_persists(tmp_path):
    mgr = SDDManager(tmp_path)
    mgr.save(Tasks(feature_id="f", tasks=[_t("1")]), "f")
    mgr.set_task_status("f", "1", "completed")
    assert str(mgr.get_tasks("f").tasks[0].status) == "completed"


def test_scope_down_preserves_done_subtasks(tmp_path):
    mgr = SDDManager(tmp_path)
    parent = Task(
        id="1",
        title="big",
        recommended_subtasks=5,
        complexity_score=8.0,
        subtasks=[_t("1.1", status="completed"), _t("1.2", status="pending")],
    )
    mgr.save(Tasks(feature_id="f", tasks=[parent]), "f")
    result = mgr.scope_task("f", "1", "down", strength="regular")
    # Pending subtask dropped, completed one kept.
    sub_ids = {s.id for s in result.subtasks}
    assert "1.1" in sub_ids
    assert "1.2" not in sub_ids
    assert result.recommended_subtasks == 3


def test_scope_up_increases_recommendation(tmp_path):
    mgr = SDDManager(tmp_path)
    mgr.save(
        Tasks(feature_id="f", tasks=[Task(id="1", recommended_subtasks=2,
                                          complexity_score=3.0)]),
        "f",
    )
    result = mgr.scope_task("f", "1", "up", strength="heavy")
    assert result.recommended_subtasks == 6


def test_branch_and_list_contexts(tmp_path):
    mgr = SDDManager(tmp_path)
    mgr.save(Tasks(feature_id="main", tasks=[_t("1")]), "main")
    mgr.branch_tasks("main", "experiment")
    contexts = set(mgr.list_task_contexts())
    assert {"main", "experiment"} <= contexts
