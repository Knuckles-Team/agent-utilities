"""Source contracts for the tag-triggered protected PyPI release path."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github/workflows/release.yml"


def _document() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _workflow() -> dict:
    document = _document()
    # PyYAML's YAML 1.1 loader reads the GitHub Actions ``on`` key as ``True``.
    return document.get("on", document.get(True))


def _job(name: str) -> dict:
    return _document()["jobs"][name]


def _run_text(job: dict) -> str:
    return "\n".join(
        step.get("run", "") for step in job.get("steps", []) if "run" in step
    )


def _job_text(job: dict) -> str:
    return yaml.safe_dump(job) + "\n" + _run_text(job)


def test_release_triggers_keep_main_and_pull_request_semantics() -> None:
    triggers = _workflow()
    assert triggers["push"]["branches"] == ["main"]
    assert triggers["push"]["tags"] == ["v*"]

    pull_request = triggers["pull_request"]
    assert "agent_utilities/**" in pull_request["paths"]
    assert ".github/workflows/release.yml" in pull_request["paths"]


def test_pypi_publish_is_tag_only_and_uses_the_protected_environment() -> None:
    publish = _job("publish-pypi")

    assert publish["needs"] == "numeric-runtime-gate"
    assert publish["if"] == (
        "github.event_name == 'push' && github.ref_type == 'tag' && "
        "startsWith(github.ref_name, 'v')"
    )
    assert publish["environment"] == "pypi-publish"
    assert any(
        step.get("env", {}).get("UV_PUBLISH_TOKEN") == "${{ secrets.PYPI_API_TOKEN }}"
        for step in publish["steps"]
    )

    source = _run_text(publish)
    assert "uv publish --check-url https://pypi.org/simple" in source


def test_release_identity_checks_bind_tag_version_and_commit() -> None:
    source = _job_text(_job("publish-pypi"))

    for required in (
        "EVENT_NAME: ${{ github.event_name }}",
        "REF_TYPE: ${{ github.ref_type }}",
        "REF: ${{ github.ref }}",
        "TAG_NAME: ${{ github.ref_name }}",
        "HEAD_SHA: ${{ github.sha }}",
        "REPOSITORY: ${{ github.repository }}",
        "semver_re='^v[0-9]+\\.[0-9]+\\.[0-9]+$'",
        'gh api "repos/${REPOSITORY}/git/ref/tags/${TAG_NAME}"',
        'gh api "repos/${REPOSITORY}/git/tags/${tag_object_sha}"',
        'if [[ "$tag_commit" != "$HEAD_SHA" ]]',
        'git rev-parse --verify "${TAG_NAME}^{commit}"',
        'if [[ "$checkout_commit" != "$HEAD_SHA" ]]',
        'Path("agent_utilities/_version.py")',
        'tag_version="${TAG_NAME#v}"',
        'if [[ "$tag_version" != "$source_version"',
        '"$tag_version" != "$wheel_version"',
    ):
        assert required in source


def test_pypi_verification_requires_the_exact_published_release() -> None:
    source = _job_text(_job("publish-pypi"))

    for required in (
        "EXPECTED_TAG: ${{ github.ref_name }}",
        'expected_version="${EXPECTED_TAG#v}"',
        "https://pypi.org/pypi/agent-utilities/{version}/json",
        'info.get("name") != "agent-utilities"',
        'info.get("version") != version',
        "if version not in releases:",
        'item.get("filename") == expected_wheel',
        "for attempt in $(seq 1 12); do",
        "PyPI exact release verified: agent-utilities=={version}",
    ):
        assert required in source
