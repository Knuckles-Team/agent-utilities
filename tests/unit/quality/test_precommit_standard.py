"""The local hooks are exactly the pipelines pre-commit standard (reference/precommit-standard.md)."""

from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[3]


def _hooks(name: str) -> dict[str, dict]:
    config = yaml.safe_load((_ROOT / ".config" / name).read_text(encoding="utf-8"))
    return {hook["id"]: hook for repo in config["repos"] for hook in repo["hooks"]}


def _stages(hook: dict) -> list[str]:
    return hook.get("stages", ["pre-commit"])


def test_local_hooks_are_the_standard_set() -> None:
    hooks = _hooks("pre-commit.yaml")
    commit = {i for i, h in hooks.items() if _stages(h) == ["pre-commit"]}
    push = {i for i, h in hooks.items() if _stages(h) == ["pre-push"]}
    assert commit == {
        "trailing-whitespace", "end-of-file-fixer", "check-merge-conflict",
        "check-added-large-files", "check-yaml", "check-toml", "check-json",
        "detect-private-key", "ruff-format", "ruff-check",
        "lane-guard",  # stated exception: guards the local canonical checkout
    }  # fmt: skip
    assert push == {"uv-lock", "spec-trailers", "fast-tests"}
    assert set(hooks) == commit | push


def test_no_manual_stage_anywhere() -> None:
    for name in ("pre-commit.yaml", "ci-gates.yaml"):
        for hook_id, hook in _hooks(name).items():
            assert "manual" not in _stages(hook), (name, hook_id)


def test_ci_runs_the_ci_only_gate_file() -> None:
    workflow = (_ROOT / ".github" / "workflows" / "advisory.yml").read_text(encoding="utf-8")
    assert "pre-commit run --config .config/ci-gates.yaml --all-files" in workflow
    assert not set(_hooks("pre-commit.yaml")) & set(_hooks("ci-gates.yaml"))
