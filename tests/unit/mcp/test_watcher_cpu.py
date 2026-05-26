import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from agent_utilities.sdd.watcher import (
    process_plan_file,
    process_tasks_file,
    process_skill_file,
    _SEEN_MTIMES,
    _SEEN_HASHES,
)

class TestWatcherCPU(unittest.TestCase):
    """Deterministic unit tests for the background watcher's CPU and I/O optimization."""

    def setUp(self):
        # Clear global caches before each test
        _SEEN_MTIMES.clear()
        _SEEN_HASHES.clear()

    def test_process_plan_file_mtime_caching(self):
        """Verify that process_plan_file skips reading/hashing on cache hit."""
        engine = MagicMock()
        file_path = Path("/mock/workspace/.specify/specs/feature_x/plan.md")
        workspace_path = Path("/mock/workspace")

        # Mock path attributes and stat
        mock_stat = MagicMock()
        mock_stat.st_mtime = 1234567.89

        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "stat", return_value=mock_stat), \
             patch.object(Path, "resolve", return_value=file_path), \
             patch.object(Path, "mkdir") as mock_mkdir, \
             patch.object(Path, "write_text") as mock_write_file, \
             patch("agent_utilities.sdd.watcher._get_latest_version_from_history", return_value=(0, set())), \
             patch("agent_utilities.sdd.watcher._parse_plan_metadata", return_value={"title": "Plan X", "approach": "Simple"}), \
             patch("agent_utilities.sdd.watcher.ingest_plan_version") as mock_ingest, \
             patch.object(Path, "read_text", return_value="title: Plan X\napproach: Simple") as mock_read:

            # 1. First execution: should read the file and update mtime cache
            process_plan_file(engine, file_path, workspace_path)
            self.assertEqual(mock_read.call_count, 1)
            self.assertIn(str(file_path), _SEEN_MTIMES)
            self.assertEqual(_SEEN_MTIMES[str(file_path)], 1234567.89)

            # 2. Second execution with same mtime: should bypass read_text entirely
            mock_read.reset_mock()
            process_plan_file(engine, file_path, workspace_path)
            self.assertEqual(mock_read.call_count, 0)  # Verify zero disk read calls!

            # 3. Third execution with updated mtime: should read the file again
            mock_stat.st_mtime = 1234568.99
            mock_read.reset_mock()
            process_plan_file(engine, file_path, workspace_path)
            self.assertEqual(mock_read.call_count, 1)
            self.assertEqual(_SEEN_MTIMES[str(file_path)], 1234568.99)

    def test_process_tasks_file_mtime_caching(self):
        """Verify that process_tasks_file skips reading/hashing on cache hit."""
        engine = MagicMock()
        file_path = Path("/mock/workspace/.specify/specs/feature_x/tasks.md")
        workspace_path = Path("/mock/workspace")

        mock_stat = MagicMock()
        mock_stat.st_mtime = 987654.32

        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "stat", return_value=mock_stat), \
             patch.object(Path, "resolve", return_value=file_path), \
             patch.object(Path, "mkdir") as mock_mkdir, \
             patch.object(Path, "write_text") as mock_write_file, \
             patch("agent_utilities.sdd.watcher._get_latest_version_from_history", return_value=(0, set())), \
             patch("agent_utilities.sdd.watcher.ingest_tasks_version") as mock_ingest, \
             patch.object(Path, "read_text", return_value="- [ ] Task 1\n- [ ] Task 2") as mock_read:

            # 1. First execution: should read and cache mtime
            process_tasks_file(engine, file_path, workspace_path)
            self.assertEqual(mock_read.call_count, 1)
            self.assertIn(str(file_path), _SEEN_MTIMES)

            # 2. Second execution: should bypass disk read entirely
            mock_read.reset_mock()
            process_tasks_file(engine, file_path, workspace_path)
            self.assertEqual(mock_read.call_count, 0)

    def test_process_skill_file_mtime_caching(self):
        """Verify that process_skill_file skips reading/hashing on cache hit."""
        engine = MagicMock()
        file_path = Path("/mock/workspace/skills/my_skill/SKILL.md")
        workspace_path = Path("/mock/workspace")

        mock_stat = MagicMock()
        mock_stat.st_mtime = 555666.77

        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "stat", return_value=mock_stat), \
             patch.object(Path, "resolve", return_value=file_path), \
             patch.object(Path, "mkdir") as mock_mkdir, \
             patch.object(Path, "write_text") as mock_write_file, \
             patch("agent_utilities.sdd.watcher._get_latest_version_from_history", return_value=(0, set())), \
             patch.object(Path, "read_text", return_value="---\nname: My Skill\ndescription: Test\n---") as mock_read:

            # 1. First execution: should read and cache mtime
            process_skill_file(engine, file_path, workspace_path)
            self.assertEqual(mock_read.call_count, 1)
            self.assertIn(str(file_path), _SEEN_MTIMES)

            # 2. Second execution: should bypass disk read entirely
            mock_read.reset_mock()
            process_skill_file(engine, file_path, workspace_path)
            self.assertEqual(mock_read.call_count, 0)
