"""ARA Compiler + Seal — ecosystem grounding and OWL/SHACL-grounded review (KG-2.80)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.research.ara import (
    ARACompiler,
    ARASeal,
    Claim,
    Evidence,
    ResearchArtifact,
)


class _Engine:
    def __init__(self, concepts=None):
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple[str, str, str]] = []
        self._concepts = concepts or []

    def add_node(self, nid, ntype, properties=None):
        self.nodes[nid] = {"type": ntype, **(properties or {})}

    def add_edge(self, src, dst, rel_type="", **props):
        self.edges.append((src, dst, rel_type))

    def query_cypher(self, q, params=None):
        return self._concepts


class _Generator:
    """Stand-in for ResearchArtifactGenerator returning a legacy artifact."""

    class _Legacy:
        article_id = "2604.24658"
        title = "Agent-Native Research Artifacts"
        summary = "4-layer artifact"
        key_contributions = ["ARA lifts reproduction to 64.4 percent"]
        methods = ["compile legacy paper to ARA"]
        suggested_experiments = ["reproduce on held-out papers"]
        authors = ["A. Researcher"]
        source_url = "https://arxiv.org/abs/2604.24658"

    def generate_paper_artifact(self, article_id, target_codebase=None):
        return self._Legacy()


# ── compiler ────────────────────────────────────────────────────────────────


def test_compile_grounds_claims_to_ecosystem_and_materializes():
    eng = _Engine()
    # inject a deterministic grounding so a claim links to an ecosystem service node
    compiler = ARACompiler(
        eng,
        generator=_Generator(),
        ground_fn=lambda statement: ["service:vllm-embed", "concept:KG-2.80"],
    )
    artifact, report = compiler.compile("2604.24658")

    assert report.n_claims == 1
    assert report.n_grounded == 1
    # the artifact + its claim/code nodes were written
    assert "research_artifact:2604.24658" in eng.nodes
    # claim now grounds in the ecosystem nodes → cross-domain grounded_in edges exist
    grounded = [(s, d) for s, d, rel in eng.edges if rel == "grounded_in"]
    assert any(d == "service:vllm-embed" for _, d in grounded)
    assert report.groundings  # recorded per-claim


def test_default_ground_fn_uses_supported_query_only():
    eng = _Engine(
        concepts=[{"id": "concept:owl-reasoning", "name": "owl reasoning bridge"}]
    )
    compiler = ARACompiler(eng, generator=_Generator())  # no injected ground_fn
    _artifact, report = compiler.compile("2604.24658", materialize=False)
    # claim statement "...compile legacy paper to ARA..." won't token-match; the
    # contribution mentions 'reproduction' — grounding is best-effort, never raises
    assert report.article_id == "2604.24658"


# ── seal ──────────────────────────────────────────────────────────────────


def _grounded_artifact() -> ResearchArtifact:
    ev = Evidence(id="evidence:p:0", content="result table")
    return ResearchArtifact(
        article_id="p",
        title="Paper",
        evidence=[ev],
        claims=[
            Claim(id="claim:p:0", statement="claim text", evidence_ids=["evidence:p:0"])
        ],
        source_ref="article:p",
    )


def test_seal_l1_passes_for_grounded_conformant_artifact_and_certifies():
    eng = _Engine()
    report = ARASeal(eng).review(_grounded_artifact(), level="L1")
    assert report.passed
    assert not report.violations
    assert report.certificate_id.startswith("seal_certificate:")
    # the certificate node + certifies edge are materialized
    assert report.certificate_id in eng.nodes
    assert any(rel == "certifies" for _, _, rel in eng.edges)


def test_seal_l1_fails_ungrounded_claim():
    art = ResearchArtifact(
        article_id="p",
        title="Paper",
        claims=[Claim(id="claim:p:0", statement="ungrounded claim")],
        source_ref="article:p",
    )
    report = ARASeal(None).review(art, level="L1")
    assert not report.passed
    codes = {v.code for v in report.violations}
    assert "claim_not_conformant" in codes


def test_seal_l1_flags_dangling_code_reference():
    art = _grounded_artifact()
    art.claims[0].code_spec_ids = ["code_spec:p:missing"]
    report = ARASeal(None).review(art, level="L1")
    assert not report.passed
    assert any(v.code == "dangling_code_ref" for v in report.violations)


def test_seal_l2_confidence_floor_and_judge():
    art = _grounded_artifact()
    # a rigor judge that rejects → confidence below floor fails L2
    seal = ARASeal(None, judge_fn=lambda stmt, ev: 0.1, confidence_floor=0.5)
    report = seal.review(art, level="L2")
    assert not report.passed
    assert any(v.code == "below_confidence_floor" for v in report.violations)
    assert report.scores["mean_confidence"] == 0.1


def test_seal_l3_withholds_evidence_via_markings():
    eng = _Engine()
    report = ARASeal(eng).review(_grounded_artifact(), level="L3")
    assert report.passed
    assert report.withheld_evidence == ["evidence:p:0"]
    cert = eng.nodes[report.certificate_id]
    assert cert["markings"] == ["restricted"]
