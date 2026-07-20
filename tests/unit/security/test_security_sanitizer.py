"""Security regression tests for the repository sanitizer.

The sanitizer is copied into standalone ecosystem projects, so these tests stay
dependency-free and exercise only its public module-level functions.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def sanitizer():
    script = Path(__file__).parents[3] / "scripts" / "security_sanitizer.py"
    spec = importlib.util.spec_from_file_location("security_sanitizer", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(repo: Path, relative: str, content: str) -> Path:
    target = repo / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def test_secret_diagnostic_never_echoes_secret_or_source_line(tmp_path, sanitizer):
    credential = "sk-" + "lf-" + ("q" * 24)
    _write(tmp_path, "src/settings.py", f'credential = "{credential}"\n')

    violations = sanitizer.scan_repository(tmp_path)

    assert any("Langfuse secret" in item for item in violations)
    assert all(credential not in item for item in violations)
    assert all("credential =" not in item for item in violations)


def test_inline_scanner_bypass_comments_are_not_honored(tmp_path, sanitizer):
    credential = "sk-" + "lf-" + ("r" * 24)
    _write(tmp_path, "src/settings.py", f'credential = "{credential}"  # nosec\n')

    violations = sanitizer.scan_repository(tmp_path)

    assert any("Langfuse secret" in item for item in violations)


def test_placeholder_word_in_variable_name_cannot_hide_secret(tmp_path, sanitizer):
    credential = "qwertyuiopasdfghjklzxcvb"
    _write(tmp_path, "src/settings.py", f'secret_example = "{credential}"\n')

    violations = sanitizer.scan_repository(tmp_path)

    assert any("Generic Secret Assignment" in item for item in violations)


def test_invalid_utf8_fails_closed(tmp_path, sanitizer):
    target = tmp_path / "src" / "settings.conf"
    target.parent.mkdir()
    target.write_bytes(b"prefix\xffsuffix")

    violations = sanitizer.scan_repository(tmp_path)

    assert any("could not be inspected" in item for item in violations)


def test_oversized_source_fails_closed_without_reading_it(
    tmp_path, sanitizer, monkeypatch
):
    monkeypatch.setattr(sanitizer, "MAX_SCAN_BYTES", 4)
    _write(tmp_path, "src/settings.conf", "12345")

    violations = sanitizer.scan_repository(tmp_path)

    assert any("exceeds security scan boundary" in item for item in violations)


def test_repository_symlink_is_not_followed(tmp_path, sanitizer):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.conf"
    credential = "sk-" + "lf-" + ("s" * 24)
    outside.write_text(credential, encoding="utf-8")
    link = tmp_path / "linked.conf"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    violations = sanitizer.scan_repository(tmp_path)

    assert not violations


def test_git_inventory_cannot_escape_repository(tmp_path, sanitizer, monkeypatch):
    monkeypatch.setattr(
        sanitizer.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="../outside.conf\n"),
    )
    monkeypatch.setattr(sanitizer.os, "walk", lambda *args, **kwargs: [])

    assert sanitizer.get_repo_files(tmp_path) == []


def test_fallback_inventory_scans_hidden_source_directories(
    tmp_path, sanitizer, monkeypatch
):
    hidden = _write(tmp_path, ".github/workflows/check.yml", "safe: true\n")
    monkeypatch.setattr(
        sanitizer.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("git unavailable")),
    )

    assert hidden in sanitizer.get_repo_files(tmp_path)
