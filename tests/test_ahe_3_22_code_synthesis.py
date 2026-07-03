"""Autonomous code-synthesis stage for promoted proposals (CONCEPT:AHE-3.22).

The deployed loop could branch a ``kind="code"`` change but never *generated*
the diff. These tests cover the new generator seam: target resolution, the
single-file synthesizer (with a stub LLM), the ``extra_files`` bridge into the
unchanged ``synthesize_change_set`` sandbox path, and the live ``governed_publish``
wiring — including the prose fallback for un-attributed proposals.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "unit"))

from fleet_autonomy_fakes import FakeEngine  # noqa: E402

from agent_utilities.knowledge_graph.research import code_synthesis  # noqa: E402
from agent_utilities.knowledge_graph.research.change_publisher import (  # noqa: E402
    LocalBranchPublisher,
    governed_publish,
)
from agent_utilities.knowledge_graph.research.change_synthesis import (  # noqa: E402
    FileChange,
    synthesize_change_set,
)
from agent_utilities.knowledge_graph.research.code_synthesis import (  # noqa: E402
    resolve_target_file,
    synthesize_code,
)

pytestmark = pytest.mark.concept("AHE-3.22")


class _StubSynthesizer:
    """A deterministic generator — no LLM. Records whether it was asked."""

    def __init__(self, content: str | None = "STUB_AHE_3_22 = 1\n") -> None:
        self.content = content
        self.calls: list[str] = []

    def generate(
        self, *, goal: str, target_path: str, current_source: str
    ) -> str | None:
        self.calls.append(target_path)
        return self.content


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "pkg" / "notes.md").write_text("# notes\n", encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------


class TestResolveTarget:
    def test_resolves_existing_py_target(self, repo: Path):
        assert (
            resolve_target_file({"file_path": "pkg/mod.py"}, repo_root=str(repo))
            == "pkg/mod.py"
        )

    def test_unattributed_proposal_is_none(self, repo: Path):
        assert resolve_target_file({"goal": "do a thing"}, repo_root=str(repo)) is None

    def test_non_python_target_is_none(self, repo: Path):
        assert (
            resolve_target_file({"file_path": "pkg/notes.md"}, repo_root=str(repo))
            is None
        )

    def test_missing_file_is_none(self, repo: Path):
        assert (
            resolve_target_file({"file_path": "pkg/ghost.py"}, repo_root=str(repo))
            is None
        )

    def test_escaping_path_is_none(self, repo: Path):
        assert (
            resolve_target_file({"file_path": "../etc/x.py"}, repo_root=str(repo))
            is None
        )


# ---------------------------------------------------------------------------
# Single-file synthesis
# ---------------------------------------------------------------------------


class TestSynthesizeCode:
    def test_attributed_proposal_yields_single_file_edit(self, repo: Path):
        stub = _StubSynthesizer()
        out = synthesize_code(
            {"file_path": "pkg/mod.py", "goal": "tweak"},
            synthesizer=stub,
            repo_root=str(repo),
        )
        assert out is not None
        assert [f.path for f in out] == ["pkg/mod.py"]
        assert out[0].content == "STUB_AHE_3_22 = 1\n"
        assert stub.calls == ["pkg/mod.py"]

    def test_unattributed_proposal_skips_generation(self, repo: Path):
        stub = _StubSynthesizer()
        assert (
            synthesize_code({"goal": "x"}, synthesizer=stub, repo_root=str(repo))
            is None
        )
        assert stub.calls == []  # generator never invoked without a target

    def test_unchanged_or_empty_output_is_none(self, repo: Path):
        assert (
            synthesize_code(
                {"file_path": "pkg/mod.py"},
                synthesizer=_StubSynthesizer(
                    content="VALUE = 1"
                ),  # == current (stripped)
                repo_root=str(repo),
            )
            is None
        )
        assert (
            synthesize_code(
                {"file_path": "pkg/mod.py"},
                synthesizer=_StubSynthesizer(content=""),
                repo_root=str(repo),
            )
            is None
        )


# ---------------------------------------------------------------------------
# The extra_files bridge into change synthesis (sandbox path unchanged)
# ---------------------------------------------------------------------------


class TestExtraFilesBridge:
    def test_generated_files_take_the_code_branch(self):
        proposal = {
            "id": "p:1",
            "name": "Improve mod",
            "goal": "g",
        }  # no embedded files
        change = synthesize_change_set(
            proposal, extra_files=[FileChange(path="pkg/mod.py", content="VALUE = 2\n")]
        )
        assert change.kind == "code"
        assert [f.path for f in change.files] == ["pkg/mod.py"]
        assert change.validation is not None and change.validation.ok
        assert change.publishable

    def test_no_extra_files_falls_back_to_prose(self):
        change = synthesize_change_set(
            {"id": "p:2", "name": "Improve mod", "goal": "g"}
        )
        assert change.kind == "sdd_plan"

    def test_embedded_files_win_over_extra_files(self):
        proposal = {
            "id": "p:3",
            "name": "x",
            "files": [{"path": "pkg/a.py", "content": "A = 1\n"}],
        }
        change = synthesize_change_set(
            proposal, extra_files=[FileChange(path="pkg/b.py", content="B = 2\n")]
        )
        assert [f.path for f in change.files] == ["pkg/a.py"]

    def test_syntactically_broken_generated_file_is_not_publishable(self):
        change = synthesize_change_set(
            {"id": "p:4", "name": "x", "goal": "g"},
            extra_files=[FileChange(path="pkg/broken.py", content="def broken(:\n")],
        )
        assert change.kind == "code"
        assert change.validation is not None and not change.validation.ok
        assert not change.publishable


# ---------------------------------------------------------------------------
# Live path — governed_publish drives the generator by default
# ---------------------------------------------------------------------------


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def target_repo(tmp_path: Path) -> Path:
    r = tmp_path / "target"
    r.mkdir()
    _git("init", "-q", "-b", "main", ".", cwd=r)
    (r / "README.md").write_text("seed\n", encoding="utf-8")
    _git("add", "-A", cwd=r)
    _git(
        "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init", cwd=r
    )
    return r


class TestGovernedPublishLivePath:
    def test_governed_publish_emits_generated_code(
        self, monkeypatch, target_repo, tmp_path
    ):
        # Fake the generator so the live path does not need an LLM or the real repo.
        monkeypatch.setattr(
            code_synthesis,
            "synthesize_code",
            lambda proposal, **kw: [
                FileChange(path="pkg/gen.py", content="GENERATED = 1\n")
            ],
        )
        engine = FakeEngine()
        # Relax the merge_promotion tier to auto so the gate allows (no human grant).
        engine.add_node(
            "rule:promo-auto",
            "governance_rule",
            properties={
                "scope": "action_policy",
                "kind": "merge_promotion",
                "target": "*",
                "tier": "auto",
            },
        )
        publisher = LocalBranchPublisher(
            engine, repo_path=target_repo, worktree_root=tmp_path / "wt"
        )
        report = governed_publish(
            engine,
            {"id": "proposal:gen-1", "name": "Gen", "goal": "g"},
            publisher=publisher,
        )
        assert report["status"] == "published"
        assert report["change_kind"] == "code"
        assert report["code_synthesis"]["files"] == ["pkg/gen.py"]
        assert report["publish"]["branch"].startswith("evolution/")

    def test_unattributed_proposal_still_publishes_prose(self, target_repo, tmp_path):
        # No monkeypatch: real synthesize_code returns None for a no-target proposal,
        # so the prose SDD skeleton is published exactly as before AHE-3.22.
        engine = FakeEngine()
        engine.add_node(
            "rule:promo-auto",
            "governance_rule",
            properties={
                "scope": "action_policy",
                "kind": "merge_promotion",
                "target": "*",
                "tier": "auto",
            },
        )
        publisher = LocalBranchPublisher(
            engine, repo_path=target_repo, worktree_root=tmp_path / "wt"
        )
        report = governed_publish(
            engine,
            {"id": "proposal:prose-1", "name": "Prose only", "goal": "g"},
            publisher=publisher,
        )
        assert report["status"] == "published"
        assert report["change_kind"] == "sdd_plan"
        assert "code_synthesis" not in report
