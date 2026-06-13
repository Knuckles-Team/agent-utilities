"""Capability ratchet + verified apply→verify→rollback (CONCEPT:AHE-3.24, AHE-3.23).

A published worktree is re-measured against a persisted capability baseline; a
measured regression (ManifestVerifier ``*_revert`` recommendation, or any tracked
capability dropping below baseline) abandons the branch instead of merging it. The
first run bootstraps the baseline; a passing run advances it monotonically. The
recorded verdict is consulted by the AHE-3.20 promotion-governance gate.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "unit"))

from fleet_autonomy_fakes import FakeEngine  # noqa: E402

from agent_utilities.knowledge_graph.research.capability_ratchet import (  # noqa: E402
    CapabilityRatchet,
    RatchetVerdict,
    latest_ratchet_result,
)
from agent_utilities.knowledge_graph.research.change_publisher import (  # noqa: E402
    LocalBranchPublisher,
    governed_publish,
)
from agent_utilities.knowledge_graph.research.promotion_governance import (  # noqa: E402
    PromotionGovernanceValidator,
)

pytestmark = pytest.mark.concept("AHE-3.24")


class RatchetEngine:
    """Minimal engine: stores nodes and answers the two ratchet queries."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}

    def add_node(self, node_id: str, node_type: str, properties: dict | None = None) -> None:
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def by_type(self, t: str) -> list[dict]:
        return [n for n in self.nodes.values() if n["type"] == t]

    def query_cypher(self, query: str, params: dict | None = None):
        params = params or {}
        if "CapabilityScoreVector" in query:
            return [
                {"scores": n["scores_json"], "ts": n["recorded_at"]}
                for n in self.by_type("CapabilityScoreVector")
            ]
        if "CapabilityRatchetResult" in query:
            pid = params.get("pid")
            return [
                {"result": n["result"], "ts": n["recorded_at"]}
                for n in self.by_type("CapabilityRatchetResult")
                if n.get("proposal_id") == pid
            ]
        return []


def _ratchet(engine, scores: dict[str, float]) -> CapabilityRatchet:
    # suite_runner injected → no real pytest subprocess.
    return CapabilityRatchet(engine, suite_runner=lambda _wt: dict(scores))


# ---------------------------------------------------------------------------
# Ratchet logic (AHE-3.24)
# ---------------------------------------------------------------------------


class TestRatchet:
    def test_bootstrap_establishes_baseline_without_blocking(self, tmp_path):
        engine = RatchetEngine()
        verdict = _ratchet(engine, {"cap": 0.8}).evaluate(str(tmp_path), proposal_id="p:1")
        assert verdict.passed and verdict.recommendation == "bootstrap"
        assert engine.by_type("CapabilityScoreVector")  # baseline persisted

    def test_regression_blocks_and_records_hold(self, tmp_path):
        engine = RatchetEngine()
        _ratchet(engine, {"cap": 0.9}).evaluate(str(tmp_path), proposal_id="p:base")
        verdict = _ratchet(engine, {"cap": 0.5}).evaluate(
            str(tmp_path), proposal_id="p:regress"
        )
        assert not verdict.passed
        assert verdict.regressions == ["cap"]
        assert verdict.recommendation == "full_revert"  # ManifestVerifier: delta < 0
        holds = [n for n in engine.by_type("CapabilityRatchetResult") if n["result"] == "hold"]
        assert holds

    def test_improvement_passes_and_advances_baseline(self, tmp_path):
        engine = RatchetEngine()
        _ratchet(engine, {"cap": 0.8}).evaluate(str(tmp_path), proposal_id="p:base")
        verdict = _ratchet(engine, {"cap": 0.95}).evaluate(
            str(tmp_path), proposal_id="p:better"
        )
        assert verdict.passed and verdict.recommendation == "confirm"
        # monotone advance: the newest baseline reflects the higher score.
        latest = _ratchet(engine, {})._load_baseline()
        assert latest == {"cap": 0.95}

    def test_no_probes_is_not_measured_and_does_not_block(self, tmp_path):
        engine = RatchetEngine()
        verdict = CapabilityRatchet(engine, suite_runner=lambda _wt: {}).evaluate(
            str(tmp_path), proposal_id="p:x"
        )
        assert verdict.passed and verdict.recommendation == "not_measured"
        assert not engine.by_type("CapabilityScoreVector")  # nothing recorded

    def test_real_measure_skips_absent_probes(self, tmp_path):
        # No injected runner: targets don't exist in tmp_path → measured nothing.
        assert CapabilityRatchet(None).measure(str(tmp_path)) == {}


# ---------------------------------------------------------------------------
# Governance predicate (AHE-3.24)
# ---------------------------------------------------------------------------


class TestGovernancePredicate:
    def test_recorded_hold_blocks_promotion(self, tmp_path):
        engine = RatchetEngine()
        _ratchet(engine, {"cap": 0.9}).evaluate(str(tmp_path), proposal_id="p:base")
        _ratchet(engine, {"cap": 0.4}).evaluate(str(tmp_path), proposal_id="prop:1")
        assert latest_ratchet_result(engine, "prop:1") == "hold"
        check = PromotionGovernanceValidator(engine)._check_capability_ratchet({}, "prop:1")
        assert not check.passed

    def test_no_record_defers(self):
        check = PromotionGovernanceValidator(RatchetEngine())._check_capability_ratchet(
            {}, "prop:none"
        )
        assert check.passed


# ---------------------------------------------------------------------------
# Verified verdict on the live publish path (AHE-3.23)
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
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init", cwd=r)
    return r


class _StubRatchet:
    def __init__(self, verdict: RatchetVerdict) -> None:
        self._verdict = verdict
        self.seen: list[str] = []

    def evaluate(self, worktree_path, *, change_set=None, proposal_id=""):
        self.seen.append(worktree_path)
        return self._verdict


def _auto_engine() -> FakeEngine:
    engine = FakeEngine()
    engine.add_node(
        "rule:promo-auto",
        "governance_rule",
        properties={"scope": "action_policy", "kind": "merge_promotion", "target": "*", "tier": "auto"},
    )
    return engine


@pytest.mark.concept("AHE-3.23")
class TestVerifiedRollback:
    def test_capability_regression_abandons_the_branch(self, target_repo, tmp_path):
        engine = _auto_engine()
        ratchet = _StubRatchet(
            RatchetVerdict(passed=False, recommendation="full_revert", reason="cap dropped")
        )
        publisher = LocalBranchPublisher(engine, repo_path=target_repo, worktree_root=tmp_path / "wt")
        report = governed_publish(
            engine,
            {"id": "proposal:bad", "name": "Bad", "goal": "g", "files": [{"path": "pkg/x.py", "content": "X = 1\n"}]},
            publisher=publisher,
            capability_ratchet=ratchet,
        )
        assert report["status"] == "reverted"
        assert report["capability_ratchet"]["recommendation"] == "full_revert"
        # branch removed — the (unpushed) publication is fully undone.
        assert _git("branch", "--list", "evolution/*", cwd=target_repo) == ""
        assert ratchet.seen  # the ratchet was actually consulted on the worktree

    def test_capability_confirmed_keeps_the_branch(self, target_repo, tmp_path):
        engine = _auto_engine()
        ratchet = _StubRatchet(
            RatchetVerdict(passed=True, recommendation="confirm", reason="ok")
        )
        publisher = LocalBranchPublisher(engine, repo_path=target_repo, worktree_root=tmp_path / "wt")
        report = governed_publish(
            engine,
            {"id": "proposal:good", "name": "Good", "goal": "g", "files": [{"path": "pkg/x.py", "content": "X = 1\n"}]},
            publisher=publisher,
            capability_ratchet=ratchet,
        )
        assert report["status"] == "published"
        assert report["capability_ratchet"]["recommendation"] == "confirm"
        assert "evolution/" in _git("branch", "--list", "evolution/*", cwd=target_repo)
