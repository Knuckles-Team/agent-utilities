#!/usr/bin/python
"""Tests for CONCEPT:KG-2.3 — Structural Fingerprint Engine."""

import os
import textwrap
import tempfile

import pytest

from agent_utilities.knowledge_graph.fingerprint import (
    ChangeLevel,
    FingerprintManager,
    StructuralFingerprint,
    classify_change,
    compute_fingerprint,
    detect_stale_files,
)


@pytest.fixture
def python_file(tmp_path):
    """Create a temporary Python file for testing."""
    code = textwrap.dedent("""\
        import os
        from pathlib import Path

        __all__ = ["greet", "Greeter"]

        def greet(name: str, loud: bool = False) -> str:
            \"\"\"Say hello.\"\"\"
            msg = f"Hello, {name}!"
            return msg.upper() if loud else msg

        class Greeter:
            \"\"\"A greeter class.\"\"\"
            def __init__(self, prefix: str = "Hi"):
                self.prefix = prefix

            def say(self, name: str) -> str:
                return f"{self.prefix}, {name}!"
    """)
    f = tmp_path / "greet.py"
    f.write_text(code)
    return str(f)


class TestComputeFingerprint:
    """Test fingerprint computation."""

    def test_computes_python_fingerprint(self, python_file):
        """Should extract functions, classes, imports from Python."""
        fp = compute_fingerprint(python_file)

        assert fp is not None
        assert fp.content_hash  # Non-empty SHA-256
        assert fp.structural_hash
        assert fp.computed_at

        # Should have extracted functions
        func_names = [f["name"] for f in fp.functions]
        assert "greet" in func_names

        # Should have extracted classes
        class_names = [c["name"] for c in fp.classes]
        assert "Greeter" in class_names

        # Should have extracted imports
        assert any("os" in imp for imp in fp.imports)
        assert any("Path" in imp for imp in fp.imports)

        # Should have extracted __all__
        assert "greet" in fp.exports
        assert "Greeter" in fp.exports

    def test_nonexistent_file_returns_none(self):
        """Missing file should return None."""
        fp = compute_fingerprint("/nonexistent/path/to/file.py")
        assert fp is None

    def test_non_python_file(self, tmp_path):
        """Non-Python files should get content-only fingerprint."""
        f = tmp_path / "config.yaml"
        f.write_text("key: value\nother: stuff\n")
        fp = compute_fingerprint(str(f))

        assert fp is not None
        # For non-Python, structural_hash should still be computed
        assert fp.content_hash
        assert fp.structural_hash


class TestClassifyChange:
    """Test change classification logic."""

    def test_none_when_identical(self, python_file):
        """Same file should classify as NONE."""
        fp1 = compute_fingerprint(python_file)
        fp2 = compute_fingerprint(python_file)

        assert classify_change(fp1, fp2) == ChangeLevel.NONE

    def test_cosmetic_on_comment_change(self, tmp_path):
        """Comment-only changes should classify as COSMETIC."""
        code1 = textwrap.dedent("""\
            def hello():
                # Original comment
                return "hi"
        """)
        code2 = textwrap.dedent("""\
            def hello():
                # Updated comment with more detail
                return "hi"
        """)
        f = tmp_path / "hello.py"

        f.write_text(code1)
        fp1 = compute_fingerprint(str(f))

        f.write_text(code2)
        fp2 = compute_fingerprint(str(f))

        assert classify_change(fp1, fp2) == ChangeLevel.COSMETIC

    def test_structural_on_new_param(self, tmp_path):
        """Adding a function parameter should classify as STRUCTURAL."""
        code1 = textwrap.dedent("""\
            def hello(name):
                return f"Hi {name}"
        """)
        code2 = textwrap.dedent("""\
            def hello(name, greeting="Hi"):
                return f"{greeting} {name}"
        """)
        f = tmp_path / "hello.py"

        f.write_text(code1)
        fp1 = compute_fingerprint(str(f))

        f.write_text(code2)
        fp2 = compute_fingerprint(str(f))

        assert classify_change(fp1, fp2) == ChangeLevel.STRUCTURAL

    def test_structural_on_new_function(self, tmp_path):
        """Adding a new function should classify as STRUCTURAL."""
        code1 = "def a(): pass\n"
        code2 = "def a(): pass\ndef b(): pass\n"
        f = tmp_path / "funcs.py"

        f.write_text(code1)
        fp1 = compute_fingerprint(str(f))

        f.write_text(code2)
        fp2 = compute_fingerprint(str(f))

        assert classify_change(fp1, fp2) == ChangeLevel.STRUCTURAL

    def test_structural_when_new_file(self):
        """New file (old=None) should classify as STRUCTURAL."""
        fp = StructuralFingerprint(
            file_path="new.py",
            content_hash="abc",
            structural_hash="def",
        )
        assert classify_change(None, fp) == ChangeLevel.STRUCTURAL

    def test_structural_when_deleted(self):
        """Deleted file (new=None) should classify as STRUCTURAL."""
        fp = StructuralFingerprint(
            file_path="old.py",
            content_hash="abc",
            structural_hash="def",
        )
        assert classify_change(fp, None) == ChangeLevel.STRUCTURAL


class TestStructuralFingerprint:
    """Test fingerprint serialization."""

    def test_roundtrip_serialization(self, python_file):
        """Fingerprint should survive dict roundtrip."""
        fp = compute_fingerprint(python_file)
        assert fp is not None

        data = fp.to_dict()
        restored = StructuralFingerprint.from_dict(data)

        assert restored.content_hash == fp.content_hash
        assert restored.structural_hash == fp.structural_hash
        assert restored.file_path == fp.file_path


class TestFingerprintManager:
    """Test workspace-level fingerprint management."""

    def test_scan_workspace(self, tmp_path):
        """Should scan and fingerprint all Python files."""
        # Create a mini workspace
        (tmp_path / "module.py").write_text("def func(): pass\n")
        (tmp_path / "utils.py").write_text("class Helper: pass\n")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cache.py").write_text("# cached\n")

        manager = FingerprintManager(str(tmp_path))
        fps = manager.scan()

        # Should find the two Python files but not __pycache__
        assert len(fps) == 2
        paths = [os.path.basename(p) for p in fps.keys()]
        assert "module.py" in paths
        assert "utils.py" in paths

    def test_diff_detects_changes(self, tmp_path):
        """Should detect structural changes between snapshots."""
        f = tmp_path / "changing.py"
        f.write_text("def a(): pass\n")

        manager1 = FingerprintManager(str(tmp_path))
        snapshot1 = manager1.scan()

        # Modify the file structurally
        f.write_text("def a(): pass\ndef b(x: int): pass\n")

        manager2 = FingerprintManager(str(tmp_path))
        manager2.scan()

        changes = manager2.diff(snapshot1)
        assert len(changes) >= 1
        # The change should be STRUCTURAL (new function added)
        for path, level in changes.items():
            if "changing.py" in path:
                assert level == ChangeLevel.STRUCTURAL

    def test_get_structural_changes(self, tmp_path):
        """Should return only STRUCTURAL changes."""
        f1 = tmp_path / "stable.py"
        f1.write_text("def func(): pass\n")
        f2 = tmp_path / "evolving.py"
        f2.write_text("def old(): pass\n")

        manager1 = FingerprintManager(str(tmp_path))
        snapshot1 = manager1.scan()

        # Only comment change in stable.py (COSMETIC)
        f1.write_text("def func():\n    # new comment\n    pass\n")
        # Signature change in evolving.py (STRUCTURAL)
        f2.write_text("def old(): pass\ndef new(x: str): pass\n")

        manager2 = FingerprintManager(str(tmp_path))
        manager2.scan()
        structural = manager2.get_structural_changes(snapshot1)

        # Only evolving.py should be in structural changes
        structural_basenames = [os.path.basename(p) for p in structural]
        assert "evolving.py" in structural_basenames


class TestDetectStaleFiles:
    """Test git-based staleness detection."""

    def test_handles_non_git_directory(self, tmp_path):
        """Non-git directory should return empty list gracefully."""
        result = detect_stale_files(str(tmp_path))
        assert result == []
