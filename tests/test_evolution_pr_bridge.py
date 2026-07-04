"""End-to-end tests for the evolution→branch bridge (CONCEPT:AU-AHE.harness.evolution-branch-bridge).

Promoted proposals become reviewable git branches: change synthesis (embedded
file artifacts → code change sets, prose → SDD plan skeletons), RLM-sandbox
validation, the ChangePublisher seam with the default LocalBranchPublisher
(fresh worktree, local branch, NO push), and the governed publication flow
through the OS-5.24 ActionPolicy's reserved ``merge_promotion`` kind.

@pytest.mark.concept("AU-AHE.harness.evolution-branch-bridge")
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "unit"))

from fleet_autonomy_fakes import FakeEngine  # noqa: E402

from agent_utilities.knowledge_graph.research.auto_merge import (  # noqa: E402
    GovernedAutoMerger,
    MergePolicy,
)
from agent_utilities.knowledge_graph.research.change_publisher import (  # noqa: E402
    LocalBranchPublisher,
    PublishResult,
    get_change_publisher,
    governed_publish,
    publish_proposal,
    set_change_publisher,
)
from agent_utilities.knowledge_graph.research.change_synthesis import (  # noqa: E402
    ChangeSet,
    FileChange,
    extract_embedded_files,
    extract_named_tests,
    synthesize_change_set,
)

pytestmark = pytest.mark.concept("AU-AHE.harness.evolution-branch-bridge")


class BridgeEngine(FakeEngine):
    """FakeEngine + the node-by-id lookup the bridge's proposal loader uses."""

    def query_cypher(self, query: str, params: dict | None = None):
        params = params or {}
        if "WHERE n.id" in query:
            node = self.nodes.get(params.get("id"))
            return [{"n": dict(node)}] if node else []
        return super().query_cypher(query, params)


def _git(*args: str, cwd: Path) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


@pytest.fixture
def target_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", "-b", "main", ".", cwd=repo)
    (repo / "README.md").write_text("seed\n", encoding="utf-8")
    _git("add", "-A", cwd=repo)
    _git(
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-q",
        "-m",
        "init",
        cwd=repo,
    )
    return repo


def _publisher(engine, repo: Path, tmp_path: Path, **kwargs) -> LocalBranchPublisher:
    return LocalBranchPublisher(
        engine,
        repo_path=repo,
        worktree_root=tmp_path / "worktrees",
        **kwargs,
    )


def _code_proposal(**extra) -> dict:
    return {
        "id": "proposal:code-1",
        "name": "Wire retrieval cache",
        "goal": "Add a small retrieval cache module",
        "files": [
            {"path": "pkg/cache.py", "content": "CACHE: dict = {}\n"},
        ],
        **extra,
    }


def _prose_proposal() -> dict:
    return {
        "id": "proposal:prose-1",
        "name": "Improve ranking heuristics",
        "goal": "Ranking misses synergy signals",
        "description": "Blend synergy-bundle membership into the rank score.",
        "concept_ids": ["AU-KG.query.vendor-agnostic-traversal"],
    }


# ---------------------------------------------------------------------------
# Change synthesis
# ---------------------------------------------------------------------------


class TestChangeSynthesis:
    def test_code_proposal_yields_validated_code_change(self):
        change = synthesize_change_set(_code_proposal())
        assert change.kind == "code"
        assert change.proposal_id == "proposal:code-1"
        assert [f.path for f in change.files] == ["pkg/cache.py"]
        assert change.validation is not None and change.validation.ok
        assert change.publishable

    def test_syntax_error_fails_sandbox_validation(self):
        proposal = _code_proposal(
            files=[{"path": "pkg/broken.py", "content": "def broken(:\n"}]
        )
        change = synthesize_change_set(proposal)
        assert change.validation is not None
        assert not change.validation.ok
        assert not change.publishable
        assert any(
            c.name.startswith("syntax:") and not c.passed
            for c in change.validation.checks
        )

    def test_missing_intra_changeset_module_fails(self):
        proposal = _code_proposal(
            files=[
                {
                    "path": "pkg/uses_sibling.py",
                    "content": "import pkg.never_shipped\n",
                }
            ]
        )
        change = synthesize_change_set(proposal)
        assert change.validation is not None and not change.validation.ok

    def test_external_imports_are_deferred_not_fatal(self):
        proposal = _code_proposal(
            files=[
                {
                    "path": "pkg/uses_repo_context.py",
                    "content": "import some_module_only_the_repo_has\n",
                }
            ]
        )
        change = synthesize_change_set(proposal)
        assert change.validation is not None and change.validation.ok

    def test_unsafe_paths_are_dropped(self):
        files = extract_embedded_files(
            {
                "files": [
                    {"path": "../escape.py", "content": "x"},
                    {"path": "/abs/path.py", "content": "x"},
                    {"path": "ok/inside.py", "content": "x"},
                ]
            }
        )
        assert [f.path for f in files] == ["ok/inside.py"]

    def test_files_json_node_property_form(self):
        node_props = {
            "id": "proposal:node-form",
            "name": "Node form",
            "goal": "g",
            "files_json": json.dumps([{"path": "a/b.py", "content": "Y = 2\n"}]),
            "tests_json": json.dumps(["tests/test_x.py::TestY", "../bad.py"]),
        }
        change = synthesize_change_set(node_props, validate=False)
        assert change.kind == "code"
        assert [f.path for f in change.files] == ["a/b.py"]
        assert extract_named_tests(node_props) == ["tests/test_x.py::TestY"]

    def test_prose_proposal_yields_sdd_plan_skeleton(self):
        change = synthesize_change_set(_prose_proposal())
        assert change.kind == "sdd_plan"
        assert change.publishable
        paths = [f.path for f in change.files]
        assert ".specify/specs/improve-ranking-heuristics/spec.md" in paths
        assert ".specify/specs/improve-ranking-heuristics/tasks.md" in paths
        spec_md = change.files[0].content
        assert "proposal:prose-1" in spec_md
        assert "Ranking misses synergy signals" in spec_md
        assert "AU-KG.query.vendor-agnostic-traversal" in spec_md


# ---------------------------------------------------------------------------
# LocalBranchPublisher
# ---------------------------------------------------------------------------


class TestLocalBranchPublisher:
    def test_publishes_local_branch_with_commit(self, target_repo, tmp_path):
        engine = BridgeEngine()
        pub = _publisher(engine, target_repo, tmp_path, regression_check=lambda s: True)
        change = synthesize_change_set(_code_proposal(), validate=False)

        result = pub.publish(change)

        assert result.ok
        assert result.branch.startswith("evolution/")
        assert len(result.commit_sha) == 40
        assert result.gate_result == "pass"
        # The branch exists in the target repo; the commit cites the proposal.
        assert result.branch in _git("branch", "--list", result.branch, cwd=target_repo)
        message = _git("log", "-1", "--format=%B", result.branch, cwd=target_repo)
        assert "proposal:code-1" in message
        assert "AU-AHE.harness.evolution-branch-bridge" in message
        # Published file landed in the fresh worktree, not the canonical tree.
        assert (Path(result.worktree_path) / "pkg/cache.py").exists()
        assert not (target_repo / "pkg").exists()
        assert _git("status", "--porcelain", cwd=target_repo) == ""

    def test_records_publication_on_the_graph(self, target_repo, tmp_path):
        engine = BridgeEngine()
        pub = _publisher(
            engine, target_repo, tmp_path, regression_check=lambda s: False
        )
        result = pub.publish(synthesize_change_set(_code_proposal(), validate=False))

        assert result.ok and result.gate_result == "hold"
        records = engine.by_type("ProposalPublication")
        assert len(records) == 1
        assert records[0]["branch"] == result.branch
        assert records[0]["commit_sha"] == result.commit_sha
        assert records[0]["gate_result"] == "hold"
        assert ("proposal:code-1", records[0]["id"], "PUBLISHED_AS") in engine.edges

    def test_refuses_unpublishable_change_set(self, target_repo, tmp_path):
        engine = BridgeEngine()
        pub = _publisher(engine, target_repo, tmp_path)
        change = synthesize_change_set(
            _code_proposal(files=[{"path": "bad.py", "content": "def x(:\n"}])
        )
        result = pub.publish(change)
        assert not result.ok
        assert "not publishable" in result.detail
        assert _git("branch", "--list", "evolution/*", cwd=target_repo) == ""

    def test_prose_proposal_publishes_sdd_skeleton_branch(self, target_repo, tmp_path):
        pub = _publisher(BridgeEngine(), target_repo, tmp_path)
        result = pub.publish(synthesize_change_set(_prose_proposal()))
        assert result.ok
        files = _git(
            "ls-tree", "-r", "--name-only", result.branch, cwd=target_repo
        ).splitlines()
        assert ".specify/specs/improve-ranking-heuristics/spec.md" in files
        assert ".specify/specs/improve-ranking-heuristics/tasks.md" in files

    def test_runs_proposal_named_tests_in_worktree(self, target_repo, tmp_path):
        pub = _publisher(BridgeEngine(), target_repo, tmp_path)
        change = ChangeSet(
            proposal_id="proposal:tested",
            title="Tested change",
            kind="code",
            files=[
                FileChange("pkg2/__init__.py", ""),
                FileChange("pkg2/answer.py", "ANSWER = 42\n"),
                FileChange(
                    "tests/test_answer.py",
                    "from pkg2.answer import ANSWER\n\n\n"
                    "def test_answer():\n    assert ANSWER == 42\n",
                ),
            ],
            tests=["tests/test_answer.py"],
        )
        result = pub.publish(change)
        assert result.ok
        assert result.tests_passed is True
        assert result.test_report is not None
        assert result.test_report["targets"] == ["tests/test_answer.py"]

    def test_no_remote_means_nothing_pushed(self, target_repo, tmp_path):
        pub = _publisher(BridgeEngine(), target_repo, tmp_path)
        result = pub.publish(synthesize_change_set(_prose_proposal()))
        assert result.ok
        assert _git("remote", cwd=target_repo) == ""

    def test_registry_resolves_injected_publisher(self):
        class Recorder:
            name = "recorder"

            def publish(self, change_set, metadata=None):
                return PublishResult(ok=True, proposal_id=change_set.proposal_id)

        sentinel = Recorder()
        set_change_publisher(sentinel)
        try:
            assert get_change_publisher() is sentinel
        finally:
            set_change_publisher(None)
        assert isinstance(get_change_publisher(), LocalBranchPublisher)


# ---------------------------------------------------------------------------
# Governed publication (ActionPolicy merge_promotion gate)
# ---------------------------------------------------------------------------


def _grant_pending_approval(engine: BridgeEngine, proposal_id: str) -> str:
    pending = [
        n
        for n in engine.by_type("ActionApproval")
        if n.get("kind") == "merge_promotion" and n.get("target") == proposal_id
    ]
    assert pending, "expected a queued merge_promotion approval"
    pending[0]["status"] = "approved"
    return pending[0]["id"]


class TestGovernedPublish:
    def test_default_policy_queues_approval(self, target_repo, tmp_path):
        engine = BridgeEngine()
        report = governed_publish(
            engine,
            _code_proposal(),
            publisher=_publisher(engine, target_repo, tmp_path),
        )
        assert report["status"] == "approval_queued"
        assert report["approval_id"]
        assert "publish" not in report
        # The shipped tier queued — nothing was published.
        assert _git("branch", "--list", "evolution/*", cwd=target_repo) == ""

    def test_granted_approval_publishes_and_is_consumed(self, target_repo, tmp_path):
        engine = BridgeEngine()
        proposal = _code_proposal()
        publisher = _publisher(engine, target_repo, tmp_path)
        first = governed_publish(engine, proposal, publisher=publisher)
        approval_id = _grant_pending_approval(engine, first["proposal_id"])

        report = governed_publish(engine, proposal, publisher=publisher)

        assert report["status"] == "published"
        assert report["approval_id"] == approval_id
        assert report["publish"]["branch"].startswith("evolution/")
        assert engine.nodes[approval_id]["status"] == "executed"
        executions = engine.by_type("ActionExecution")
        assert executions and executions[0]["kind"] == "merge_promotion"
        assert executions[0]["ok"] is True

    def test_kg_rule_can_relax_tier_to_auto(self, target_repo, tmp_path):
        engine = BridgeEngine()
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
        report = governed_publish(
            engine,
            _code_proposal(),
            publisher=_publisher(engine, target_repo, tmp_path),
        )
        assert report["status"] == "published"
        assert report["decision"] == "allow"

    def test_granted_but_invalid_change_set_fails_validation(
        self, target_repo, tmp_path
    ):
        engine = BridgeEngine()
        proposal = _code_proposal(
            files=[{"path": "pkg/broken.py", "content": "def broken(:\n"}]
        )
        publisher = _publisher(engine, target_repo, tmp_path)
        first = governed_publish(engine, proposal, publisher=publisher)
        approval_id = _grant_pending_approval(engine, first["proposal_id"])

        report = governed_publish(engine, proposal, publisher=publisher)

        assert report["status"] == "validation_failed"
        assert engine.nodes[approval_id]["status"] == "failed"
        assert _git("branch", "--list", "evolution/*", cwd=target_repo) == ""

    def test_publish_proposal_by_node_id(self, target_repo, tmp_path):
        engine = BridgeEngine()
        engine.add_node(
            "proposal:seeded",
            "team",
            properties={
                "name": "Seeded proposal",
                "goal": "Materialize an embedded patch",
                "files_json": json.dumps(
                    [{"path": "pkg/seeded.py", "content": "SEEDED = True\n"}]
                ),
            },
        )
        publisher = _publisher(engine, target_repo, tmp_path)
        first = publish_proposal(engine, "proposal:seeded", publisher=publisher)
        assert first["status"] == "approval_queued"
        _grant_pending_approval(engine, "proposal:seeded")

        report = publish_proposal(engine, "proposal:seeded", publisher=publisher)

        assert report["status"] == "published"
        branch = report["publish"]["branch"]
        files = _git("ls-tree", "-r", "--name-only", branch, cwd=target_repo)
        assert "pkg/seeded.py" in files.splitlines()

    def test_publish_proposal_unknown_id(self):
        report = publish_proposal(BridgeEngine(), "proposal:ghost")
        assert report["status"] == "not_found"


# ---------------------------------------------------------------------------
# GovernedAutoMerger wiring
# ---------------------------------------------------------------------------


class TestMergerBridgeWiring:
    def _spec(self) -> dict:
        return {
            **_code_proposal(),
            "quality_score": 0.95,
        }

    def test_merged_proposal_queues_publication(self, target_repo, tmp_path):
        engine = BridgeEngine()
        merger = GovernedAutoMerger(
            engine,
            policy=MergePolicy(enabled=True),
            governance_validator=lambda s: True,
            promoter=lambda s: True,
            publisher=_publisher(engine, target_repo, tmp_path),
        )
        evaluation = merger.consider(self._spec())
        assert evaluation.merged
        assert evaluation.publication is not None
        assert evaluation.publication["status"] == "approval_queued"
        assert engine.by_type("ActionApproval")

    def test_disabled_policy_never_publishes(self, target_repo, tmp_path):
        engine = BridgeEngine()
        merger = GovernedAutoMerger(
            engine,
            policy=MergePolicy(enabled=False),
            governance_validator=lambda s: True,
            promoter=lambda s: True,
            publisher=_publisher(engine, target_repo, tmp_path),
        )
        evaluation = merger.consider(self._spec())
        assert not evaluation.merged
        assert evaluation.publication is None
        assert not engine.by_type("ActionApproval")

    def test_relaxed_tier_publishes_end_to_end(self, target_repo, tmp_path):
        """Seeded proposal → governance pass → branch + gate verdict recorded."""
        engine = BridgeEngine()
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
        merger = GovernedAutoMerger(
            engine,
            policy=MergePolicy(enabled=True),
            governance_validator=lambda s: True,
            regression_check=lambda s: True,
            promoter=lambda s: True,
            publisher=_publisher(
                engine, target_repo, tmp_path, regression_check=lambda s: True
            ),
        )
        evaluation = merger.consider(self._spec())
        assert evaluation.merged
        assert evaluation.publication["status"] == "published"
        publish = evaluation.publication["publish"]
        assert publish["branch"].startswith("evolution/")
        assert publish["gate_result"] == "pass"
        records = engine.by_type("ProposalPublication")
        assert records and records[0]["branch"] == publish["branch"]
