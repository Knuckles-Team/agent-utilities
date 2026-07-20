"""Static contract for the exact offline GraphOS release image."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "docker" / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
PYTHON_IMAGE = (
    "python:3.12-slim@sha256:"
    "57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de"
)


def test_agent_local_is_closed_hash_locked_and_path_origin_free() -> None:
    source = DOCKERFILE.read_text(encoding="utf-8")
    local_stage = source.split("FROM builder-base AS builder-agent-local", 1)[1]
    local_stage = local_stage.split("FROM python:", 1)[0]

    for required in (
        "--offline",
        "--no-index",
        "--find-links /tmp/release-wheelhouse",
        "--only-binary :all:",
        "--require-hashes",
        "RELEASE_REQUIREMENTS_SHA256",
        'm.version("agent-utilities")',
        'm.version("epistemic-graph")',
        'm.version("langfuse-agent")',
        'shutil.which("graph-os")',
        'glob("*.dist-info/direct_url.json")',
        "import epistemic_graph.numeric",
        "sys.version_info[:2] == (3, 12)",
    ):
        assert required in local_stage

    assert " @ file:" not in local_stage
    assert "uv pip install" in local_stage
    assert "UV_OVERRIDE" not in local_stage


def test_release_context_admits_only_wheels_and_hash_locked_requirements() -> None:
    source = DOCKERFILE.read_text(encoding="utf-8")
    ignored = DOCKERIGNORE.read_text(encoding="utf-8").splitlines()

    assert source.count(f"FROM {PYTHON_IMAGE}") == 2
    assert "python:3.11" not in source
    assert source.rsplit("FROM runtime-base AS ", 1)[1].startswith("agent\n")
    assert "release-wheelhouse/*" in ignored
    assert "!release-wheelhouse/*.whl" in ignored
    assert "!release-wheelhouse/*.txt" in ignored
