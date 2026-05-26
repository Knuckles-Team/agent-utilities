"""Tests for Plan and Tasks Watcher & Ingester.

CONCEPT:KG-2.6 — Implementation Plan & Tasks versioning and KG lineage.
"""

from unittest.mock import MagicMock, patch

from agent_utilities.sdd.watcher import (
    _get_latest_version_from_history,
    _get_md5,
    _parse_plan_metadata,
    get_workspace_path,
    ingest_plan_version,
    ingest_tasks_version,
    process_kg_ingest_location,
    process_plan_file,
    process_scholarx_file,
    process_skill_file,
    process_tasks_file,
    run_watcher_scan,
)


def test_get_md5():
    content = "Hello, world!"
    expected_hash = "6cd3556deb0da54bca060b4c39479839"
    assert _get_md5(content) == expected_hash


def test_parse_plan_metadata():
    content = "# My Goal\n\n## Approach\nThis is the approach.\n\n## Tasks"
    metadata = _parse_plan_metadata(content)
    assert metadata["title"] == "My Goal"
    assert metadata["approach"] == "This is the approach."

    # Fallback when no Approach section exists
    content_no_approach = "# Short Goal\nSome simple content."
    metadata_no_approach = _parse_plan_metadata(content_no_approach)
    assert metadata_no_approach["title"] == "Short Goal"
    assert metadata_no_approach["approach"] == content_no_approach


def test_get_latest_version_from_history(tmp_path):
    history_dir = tmp_path / "history"
    history_dir.mkdir()

    # Empty history
    version, seen_hashes = _get_latest_version_from_history(history_dir, "plan_feat")
    assert version == 0
    assert len(seen_hashes) == 0

    # Add some files
    f1 = history_dir / "plan_feat_v1_20260524_120000.md"
    f1.write_text("content 1")
    f2 = history_dir / "plan_feat_v2_20260524_130000.md"
    f2.write_text("content 2")
    f3 = history_dir / "other_feat_v3_20260524_140000.md"
    f3.write_text("content 3")

    version, seen_hashes = _get_latest_version_from_history(history_dir, "plan_feat")
    assert version == 2
    assert _get_md5("content 1") in seen_hashes
    assert _get_md5("content 2") in seen_hashes
    assert _get_md5("content 3") not in seen_hashes


def test_ingest_plan_version():
    mock_engine = MagicMock()
    mock_engine.backend = MagicMock()

    plan_id = ingest_plan_version(
        engine=mock_engine,
        feature_id="feat1",
        title="Test Title",
        approach="Approach description",
        version=2,
        session_id="sess123",
        content_hash="hashabc",
        raw_content="Raw content body",
    )

    assert plan_id == "plan:feat1:v2"
    assert mock_engine.backend.execute.call_count >= 3


def test_ingest_tasks_version():
    mock_engine = MagicMock()
    mock_engine.backend = MagicMock()

    tasks_id = ingest_tasks_version(
        engine=mock_engine,
        feature_id="feat1",
        title="Tasks list",
        version=1,
        session_id="sess123",
        content_hash="hashabc",
        raw_content="Raw content body",
    )

    assert tasks_id == "tasks:feat1:v1"
    assert mock_engine.backend.execute.call_count >= 2


def test_process_plan_file(tmp_path):
    mock_engine = MagicMock()
    mock_engine.backend = MagicMock()

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    plan_file = workspace / ".specify" / "specs" / "feat1" / "plan.md"
    plan_file.parent.mkdir(parents=True, exist_ok=True)
    plan_content = "# Feature 1\n\n## Approach\nBuild it well."
    plan_file.write_text(plan_content)

    # Process first time -> creates v1
    process_plan_file(mock_engine, plan_file, workspace)

    history_dir = workspace / ".specify" / "history" / "plans"
    assert history_dir.exists()
    history_files = list(history_dir.glob("plan_feat1_v1_*.md"))
    assert len(history_files) == 1
    assert history_files[0].read_text() == plan_content

    # Process second time with same content -> should skip (v1 remains)
    process_plan_file(mock_engine, plan_file, workspace)
    history_files_all = list(history_dir.glob("plan_feat1_*.md"))
    assert len(history_files_all) == 1

    # Modify content -> creates v2
    new_content = "# Feature 1\n\n## Approach\nBuild it even better."
    plan_file.write_text(new_content)
    process_plan_file(mock_engine, plan_file, workspace)
    history_files_v2 = list(history_dir.glob("plan_feat1_v2_*.md"))
    assert len(history_files_v2) == 1


def test_process_tasks_file(tmp_path):
    mock_engine = MagicMock()
    mock_engine.backend = MagicMock()

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    tasks_file = workspace / ".specify" / "specs" / "feat1" / "tasks.md"
    tasks_file.parent.mkdir(parents=True, exist_ok=True)
    tasks_content = "- [ ] Task 1"
    tasks_file.write_text(tasks_content)

    # Process first time -> creates v1
    process_tasks_file(mock_engine, tasks_file, workspace)

    history_dir = workspace / ".specify" / "history" / "tasks"
    assert history_dir.exists()
    history_files = list(history_dir.glob("tasks_feat1_v1_*.md"))
    assert len(history_files) == 1


def test_run_watcher_scan(tmp_path):
    mock_engine = MagicMock()
    mock_engine.backend = MagicMock()

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Set up specs dir
    specs_dir = workspace / ".specify" / "specs" / "feat_xyz"
    specs_dir.mkdir(parents=True)
    (specs_dir / "plan.md").write_text("# Plan XYZ")
    (specs_dir / "tasks.md").write_text("- [ ] Task XYZ")

    with patch("os.path.expanduser", return_value=str(tmp_path / "gemini")):
        run_watcher_scan(mock_engine, workspace)

        # Verify history was created for both plan and tasks
        history_plans = list(
            (workspace / ".specify" / "history" / "plans").glob("plan_feat_xyz_v1_*.md")
        )
        history_tasks = list(
            (workspace / ".specify" / "history" / "tasks").glob(
                "tasks_feat_xyz_v1_*.md"
            )
        )
        assert len(history_plans) == 1
        assert len(history_tasks) == 1


def test_get_workspace_path(tmp_path):
    # Test fallback to cwd or parents
    with patch.dict("os.environ", {}, clear=True):
        with patch("os.getcwd", return_value=str(tmp_path)):
            # By default no pyproject.toml, so returns tmp_path
            assert get_workspace_path() == tmp_path

            # Create pyproject.toml
            (tmp_path / "pyproject.toml").write_text("")
            assert get_workspace_path() == tmp_path


def test_process_skill_file(tmp_path):
    mock_engine = MagicMock()
    mock_engine.backend = MagicMock()

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    skill_file = workspace / "my_skill" / "SKILL.md"
    skill_file.parent.mkdir(parents=True, exist_ok=True)
    skill_content = "---\nname: My Custom Skill\ndescription: Performs cool tasks\n---\n# My Skill Docs"
    skill_file.write_text(skill_content)

    process_skill_file(mock_engine, skill_file, workspace)

    history_dir = workspace / ".specify" / "history" / "skills"
    assert history_dir.exists()
    history_files = list(history_dir.glob("skill_my_custom_skill_v1_*.md"))
    assert len(history_files) == 1
    assert history_files[0].read_text() == skill_content


def test_process_scholarx_file(tmp_path):
    mock_engine = MagicMock()
    mock_engine.submit_task = MagicMock()

    pdf_file = tmp_path / "paper.pdf"
    pdf_file.write_text("dummy pdf content")

    process_scholarx_file(mock_engine, pdf_file)

    assert mock_engine.submit_task.call_count == 1
    mock_engine.submit_task.assert_called_with(
        target_path=str(pdf_file.resolve()),
        is_codebase=False,
        task_type="document",
        provenance={"source": "watcher_scholarx"},
    )


def test_process_kg_ingest_location(tmp_path):
    mock_engine = MagicMock()
    mock_engine.submit_task = MagicMock()

    yaml_file = tmp_path / "inventory.yaml"
    yaml_file.write_text("nodes: []")

    process_kg_ingest_location(mock_engine, yaml_file)

    assert mock_engine.submit_task.call_count == 1
    mock_engine.submit_task.assert_called_with(
        target_path=str(yaml_file.resolve()),
        is_codebase=False,
        task_type="document",
        provenance={"source": "watcher_kg_ingest"},
    )


def test_start_sdd_watcher():
    """Test start_sdd_watcher behaves correctly under various configuration conditions."""
    from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin

    from typing import Any

    class TestEngine(TaskManagerMixin):
        def __init__(self):
            self._workers_running = False
            self._watcher_thread_running = False
            self.backend: Any = None

        def add_node(
            self,
            node_id: str,
            node_type: str,
            properties: dict[str, Any] | None = None,
            ephemeral: bool = False,
        ) -> Any:
            return None

        def link_nodes(
            self,
            source_id: str,
            target_id: str,
            rel_type: str,
            properties: dict | None = None,
            ephemeral: bool = False,
        ) -> None:
            return None

        def query_cypher(
            self, cypher: str, params: dict | None = None
        ) -> list[dict[str, Any]]:
            return []

    # 1. Test when enable_sdd_watcher is False
    engine = TestEngine()
    with (
        patch("os.environ.get", return_value=""),
        patch("sys.argv", ["pytest"]),
        patch("agent_utilities.core.config.config.enable_sdd_watcher", False),
        patch("threading.Thread") as mock_thread,
    ):
        engine.start_sdd_watcher()
        mock_thread.assert_not_called()
        assert getattr(engine, "_watcher_thread_running", False) is False

    # 2. Test when enable_sdd_watcher is True
    engine = TestEngine()
    with (
        patch("os.environ.get", return_value=None),
        patch("sys.argv", ["app.py"]),
        patch("agent_utilities.core.config.config.enable_sdd_watcher", True),
        patch(
            "agent_utilities.sdd.watcher.get_workspace_path",
            return_value="/fake/workspace",
        ),
        patch("threading.Thread") as mock_thread,
    ):
        mock_thread_inst = MagicMock()
        mock_thread.return_value = mock_thread_inst

        engine.start_sdd_watcher()

        assert mock_thread.call_count == 1
        kwargs = mock_thread.call_args[1]
        assert kwargs["daemon"] is True
        assert kwargs["name"] == "KGPlanWatcherThread"
        mock_thread_inst.start.assert_called_once()
        assert engine._watcher_thread_running is True


def test_watcher_paused_flag():
    """Test that setting _WATCHER_PAUSED = True skips scans during the watcher loop."""
    import agent_utilities.sdd.watcher as watcher
    from pathlib import Path

    mock_engine = MagicMock()
    mock_workspace = Path("/fake/workspace")

    # Verify that normal scan executes run_watcher_scan
    with (
        patch("agent_utilities.sdd.watcher.run_watcher_scan") as mock_scan,
        patch("time.sleep", side_effect=[None, KeyboardInterrupt]),
    ):
        watcher._WATCHER_PAUSED = False  # type: ignore
        try:
            watcher.run_plan_watcher_loop(mock_engine, mock_workspace)
        except KeyboardInterrupt:
            pass

        assert mock_scan.call_count >= 1

    # Verify that paused scan does not execute run_watcher_scan
    with (
        patch("agent_utilities.sdd.watcher.run_watcher_scan") as mock_scan,
        patch("time.sleep", side_effect=[None, KeyboardInterrupt]),
    ):
        watcher._WATCHER_PAUSED = True  # type: ignore
        try:
            watcher.run_plan_watcher_loop(mock_engine, mock_workspace)
        except KeyboardInterrupt:
            pass

        mock_scan.assert_not_called()

    # Reset flag
    watcher._WATCHER_PAUSED = False  # type: ignore
