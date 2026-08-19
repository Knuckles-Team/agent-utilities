"""Meta-proof for the NE-015 HTTP import/dependency closure ratchet."""

from __future__ import annotations

from pathlib import Path

from scripts.check_http_transport_closure import validate


def _metadata(root: Path, *, lock_body: str = "") -> tuple[Path, Path]:
    pyproject = root / "pyproject.toml"
    pyproject.write_text(
        """
[project]
dependencies = []
[project.optional-dependencies]
http = ["httpx>=0.28.1"]
""",
        encoding="utf-8",
    )
    lock = root / "uv.lock"
    lock.write_text(lock_body, encoding="utf-8")
    return pyproject, lock


def test_new_direct_http_import_is_rejected(tmp_path: Path) -> None:
    package = tmp_path / "agent_utilities"
    package.mkdir()
    (package / "new_client.py").write_text(
        "import httpx\n\nclient = httpx.Client()\n", encoding="utf-8"
    )
    pyproject, lock = _metadata(tmp_path)

    errors = validate(package=package, pyproject=pyproject, lock=lock)

    assert any("new_client.py" in error for error in errors)
    assert any("direct httpx import" in error for error in errors)


def test_new_direct_httpx2_import_is_rejected(tmp_path: Path) -> None:
    package = tmp_path / "agent_utilities"
    package.mkdir()
    (package / "new_client.py").write_text(
        "import httpx2\n\nclient = httpx2.Client()\n", encoding="utf-8"
    )
    pyproject, lock = _metadata(tmp_path)

    errors = validate(package=package, pyproject=pyproject, lock=lock)

    assert any("direct httpx2 import" in error for error in errors)


def test_lock_removal_is_rejected_while_resolved_consumer_remains(
    tmp_path: Path,
) -> None:
    package = tmp_path / "agent_utilities"
    package.mkdir()
    (package / "new_client.py").write_text("# no direct imports\n", encoding="utf-8")
    pyproject, lock = _metadata(
        tmp_path,
        lock_body="""
version = 1

[[package]]
name = "consumer"
version = "1.0.0"
dependencies = [{ name = "httpx" }]
""",
    )

    errors = validate(package=package, pyproject=pyproject, lock=lock)

    assert any("uv.lock omits httpx" in error for error in errors)
