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


def test_build_needs_gates_clone_scanners_and_the_engine_release_order_gate() -> None:
    """The cross-repo release-order check (`engine-release-order`) moved out
    of `gates` into its own job so a red result there no longer also hides
    `gates`' test suite. The release-order protection itself must be
    unchanged: nothing can reach `build` -- and therefore nothing downstream
    of it (numeric-runtime-gate, publish-pypi, publish-docker) -- while the
    epistemic-graph floor is not resolvable on PyPI. `build.needs` is the one
    place that guarantee lives now; pin it directly so dropping
    `engine-release-order` from that list fails here instead of silently
    reopening the gap."""
    build = _job("build")

    assert set(build["needs"]) == {"gates", "clone-scanners", "engine-release-order"}

    engine_release_order = _job("engine-release-order")
    assert engine_release_order["name"] == (
        "epistemic-graph release is resolvable on PyPI"
    )
    assert "needs" not in engine_release_order

    source = _run_text(engine_release_order)
    assert "scripts/release/check_eg_pypi_resolvable.py" in source

    # The gate used to live inside `gates`; it must not have been duplicated
    # there as well as moved.
    gates_source = _job_text(_job("gates"))
    assert "check_eg_pypi_resolvable.py" not in gates_source


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
    assert "publish_validated_wheel.py" in source
    assert "--check-url" not in source
    assert "--skip-existing" not in source


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


def test_publication_uses_pinned_central_contract_after_identity_check() -> None:
    steps = _job("publish-pypi")["steps"]
    contract = next(
        step
        for step in steps
        if step.get("name") == "Checkout centralized publication contract"
    )
    assert contract["with"] == {
        "repository": "Knuckles-Team/pipelines",
        "ref": "4f80a968dbf09b1f4d35bbff11b77f9e2fcd3ef6",
        "path": ".pipeline-contract",
        "persist-credentials": False,
    }
    publish = next(
        step for step in steps if "publish_validated_wheel.py" in step.get("run", "")
    )
    identity = next(
        step
        for step in steps
        if step.get("name")
        == "Require exact release tag, source version, and wheel version"
    )
    assert steps.index(identity) < steps.index(contract) < steps.index(publish)
    assert publish["env"]["SOURCE_COMMIT"] == "${{ github.sha }}"
    assert publish["env"]["PIPELINES_CONTRACT_COMMIT"] == contract["with"]["ref"]
    assert publish["env"]["EXPECTED_TAG"] == "${{ github.ref_name }}"
    assert '"${EXPECTED_TAG#v}"' in publish["run"]
    assert '"$RUNNER_TEMP/au-publication.json"' in publish["run"]
    assert "python -I " in publish["run"]
    assert "urllib.request" not in _run_text(_job("publish-pypi"))
    assert "version not in releases" not in _run_text(_job("publish-pypi"))


def test_publication_preserves_runtime_and_artifact_proof() -> None:
    runtime = _job("numeric-runtime-gate")
    assert runtime["needs"] == "build"
    assert runtime["strategy"]["matrix"]["os"] == ["ubuntu-latest", "windows-latest"]
    assert runtime["strategy"]["fail-fast"] is False
    assert "str(next(Path('dist').glob('agent_utilities-*.whl')))" in _run_text(runtime)
    smokes = [
        step
        for step in runtime["steps"]
        if step.get("working-directory") == "${{ runner.temp }}"
    ]
    assert len(smokes) == 2
    assert "check_numeric_runtime.py" in smokes[0]["run"]
    assert "persistence_privacy" in smokes[1]["run"]
    build = _job("build")
    assert build["needs"] == ["gates", "clone-scanners"]
    assert "release_wheel_reproducibility=passed" in _run_text(build)
    assert "check_wheel_privacy.py" in _run_text(build)
    assert _job("docker-publish-approval")["environment"] == "docker-publish"
    assert _job("docker-publish-approval")["needs"] == "publish-pypi"
    assert _job("publish-docker")["needs"] == "docker-publish-approval"
