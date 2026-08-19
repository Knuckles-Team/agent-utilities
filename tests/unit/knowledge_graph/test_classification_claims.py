"""NE-010/U-47 AU-CLAIMS track — tests retargeted onto the REAL, already-wired
implementation: ``agent_utilities.knowledge_graph.ontology.classification_claims``
(GOC-86, commit ``f415dacd2``), live behind the ``ontology_classification_claims``
MCP tool (``mcp/tools/ontology_tools.py``) and REST route (``kg_server.py``).

This track's original brief assumed U-47 was unimplemented and specified a
new Pydantic/``RegistryNode``-based contract. It was not unimplemented: an
audit (this file replaces a since-deleted parallel module,
``knowledge_graph/ingestion/classification_claims.py``) found the real
module already satisfies MOST of the brief's acceptance criteria — often
more thoroughly than the parallel draft did (its idempotent-replay guard and
its append-only lifecycle-event audit trail are both stronger). Six genuine,
confirmed gaps remained and are closed here, additively, IN the real module
(never by forking a second one):

* a candidate-method (``derived``/``generated``) claim could be constructed
  directly at ``status="promoted"``/``"rejected"`` via the raw dataclass
  constructor, bypassing ``ClassificationPromotionLedger`` entirely and
  recording no reviewer — closed by a new structural ``__post_init__``
  check (mirrors the pre-existing deterministic-claim check exactly).
* no model-provenance fields existed for a ``method="generated"`` claim —
  added ``model_profile_version``/``prompt_digest`` (additive, paired:
  either both are set or neither is).
* ``ClassificationPromotionLedger.review``/``promote``/``reject`` never
  checked a caller's tenant/graph scope against the claim's own — added
  optional ``tenant``/``graph`` kwargs (``None`` = no check, so every
  pre-existing caller, including the live MCP tool, is unaffected) that
  raise ``PermissionError`` on a mismatch when supplied.
* ``resolve_claim_evidence`` never surfaced a fragment's excerpt digest —
  added ``content_hash`` to its RETURN clause and every response row
  (including redacted ones — a digest cannot leak the payload it
  summarizes).

Two criteria from the original brief remain genuinely NOT met by the real
module and are NOT force-fixed here (see the coordinator report for why):
the module has no ``object_ref``/``literal_value`` split (one ``value: str``
field always) — reworking that touches the live MCP tool's wire contract and
every existing caller. Deterministic-replay/restart/cross-graph proofs
already exist as a thorough acceptance test
(``ontology/test_classification_claims_acceptance.py::test_full_w07_acceptance_scenario``);
this file adds smaller, focused versions plus everything new above, never a
duplicate of that scenario.

@pytest.mark.concept("AU-KG.ontology.classification-claim-multi-category")
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.ontology.classification_claims import (
    ClassificationClaim,
    ClassificationPromotionLedger,
    query_categories,
    query_claim_history,
    query_claims,
    record_claim,
    resolve_claim_evidence,
)

pytestmark = pytest.mark.concept("AU-KG.ontology.classification-claim-multi-category")


# ── a minimal fake engine — nodes/edges round-trip through query_cypher ─────


class FakeEngine:
    """One physical graph. A SECOND instance stands in for a distinct
    physical graph/tenant boundary — the same analogue the existing
    acceptance test uses (never a shortcut around the real boundary)."""

    def __init__(self, name: str = "graph-a") -> None:
        self.name = name
        self.nodes: dict[str, dict[str, Any]] = {}

    def add_node(
        self, node_id: str, node_type: str, properties: dict[str, Any] | None = None
    ) -> None:
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def link_nodes(self, source: str, target: str, rel_type: str, **kw: Any) -> None:
        pass

    def query_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        params = params or {}
        if "ClassificationClaim" in query and "Fragment" not in query:
            rows = [
                n for n in self.nodes.values() if n.get("type") == "ClassificationClaim"
            ]
            sid = params.get("sid")
            if sid is not None:
                rows = [r for r in rows if r.get("subject_id") == sid]
            status = params.get("status")
            if status is not None:
                rows = [r for r in rows if r.get("status") == status]
            category = params.get("category")
            if category is not None:
                rows = [r for r in rows if r.get("category") == category]
            return rows
        if "Fragment" in query and "Artifact" in query:
            fid = params.get("fid")
            frag = self.nodes.get(fid)
            if not frag or frag.get("type") != "Fragment":
                return []
            artifact = self.nodes.get(frag.get("artifact_id"), {})
            return [
                {
                    "id": frag["id"],
                    "text": frag.get("text"),
                    "address": frag.get("address"),
                    "content_hash": frag.get("content_hash"),
                    "classification": artifact.get("classification"),
                }
            ]
        return []

    def add_fragment(
        self, fragment_id: str, artifact_id: str, text: str, *, content_hash: str = ""
    ) -> None:
        self.nodes[fragment_id] = {
            "id": fragment_id,
            "type": "Fragment",
            "artifact_id": artifact_id,
            "text": text,
            "address": fragment_id,
            "content_hash": content_hash or f"sha256:{'a' * 64}",
        }

    def add_artifact(
        self, artifact_id: str, *, classification: str = "internal"
    ) -> None:
        self.nodes[artifact_id] = {
            "id": artifact_id,
            "type": "Artifact",
            "classification": classification,
        }

    def claim_count(self) -> int:
        return sum(
            1 for n in self.nodes.values() if n.get("type") == "ClassificationClaim"
        )


@pytest.fixture
def engine() -> FakeEngine:
    e = FakeEngine("graph-a")
    e.add_artifact("art:demo", classification="internal")
    e.add_fragment(
        "frag:a", "art:demo", "def handler(): ...", content_hash="sha256:" + "1" * 64
    )
    e.add_fragment(
        "frag:b", "art:demo", "owner: team-x", content_hash="sha256:" + "2" * 64
    )
    return e


# ── 1. one artifact, independent simultaneous categories, snapshot+hash ─────


class TestMultiCategorySimultaneousClaims:
    def test_one_artifact_carries_independent_simultaneous_classifications(
        self, engine: FakeEngine
    ) -> None:
        code = ClassificationClaim.observe(
            subject_id="art:demo",
            category="code",
            value="python",
            evidence_refs=["frag:a"],
            source_snapshot="sha:v1",
        )
        language = ClassificationClaim.observe(
            subject_id="art:demo",
            category="language",
            value="python3.12",
            evidence_refs=["frag:a"],
            source_snapshot="sha:v1",
        )
        security = ClassificationClaim.declare(
            subject_id="art:demo",
            category="security-critical",
            value="true",
            evidence_refs=["frag:b"],
            source_snapshot="sha:v1",
        )
        architecture = ClassificationClaim.propose(
            subject_id="art:demo",
            category="architecture-layer",
            value="service-layer",
            evidence_refs=["frag:a"],
            source_snapshot="sha:v1",
            method="generated",
            policy_approved=True,
            model_profile_version="claude-sonnet-5",
            prompt_digest="sha256:" + "d" * 64,
        )
        for claim in (code, language, security, architecture):
            record_claim(engine, claim)

        # All four coexist for the SAME subject — none overwrote another.
        assert engine.claim_count() == 4
        categories = {c.category for c in query_claims(engine, "art:demo")}
        assert categories == {
            "code",
            "language",
            "security-critical",
            "architecture-layer",
        }

        # Every row resolves to a snapshot id and (via resolve_claim_evidence)
        # an excerpt hash.
        for claim in (code, language, security, architecture):
            assert claim.source_snapshot == "sha:v1"
            resolved = resolve_claim_evidence(
                engine, claim, viewer_clearance="internal"
            )
            assert resolved
            for row in resolved:
                assert row["source_snapshot"] == "sha:v1"
                assert row["content_hash"] is not None
                assert row["content_hash"].startswith("sha256:")

        # The generated claim entered as CANDIDATE and carries model provenance.
        assert architecture.status == "candidate"
        assert architecture.model_profile_version
        assert architecture.prompt_digest


# ── 2. category-intersection query returns citations ────────────────────────


class TestCategoryIntersectionReturnsCitations:
    def test_intersection_read_returns_citations_resolving_to_snapshot_and_hash(
        self, engine: FakeEngine
    ) -> None:
        for category, value, fragment in (
            ("code", "python", "frag:a"),
            ("security-critical", "true", "frag:b"),
            ("ownership", "team-x", "frag:b"),
        ):
            record_claim(
                engine,
                ClassificationClaim.observe(
                    subject_id="art:demo",
                    category=category,
                    value=value,
                    evidence_refs=[fragment],
                    source_snapshot="sha:v1",
                ),
            )
        # A different subject only partially overlaps — must not pollute
        # the intersection read for art:demo.
        engine.add_artifact("art:other")
        record_claim(
            engine,
            ClassificationClaim.observe(
                subject_id="art:other",
                category="code",
                value="rust",
                evidence_refs=["frag:a"],
                source_snapshot="sha:v1",
            ),
        )

        wanted = {"code", "security-critical", "ownership"}
        categories = query_categories(engine, "art:demo")
        assert wanted <= categories  # the intersection check

        citations = []
        for category in wanted:
            for claim in query_claims(
                engine, "art:demo", category=category, status="promoted"
            ):
                for row in resolve_claim_evidence(
                    engine, claim, viewer_clearance="internal"
                ):
                    citations.append({**row, "category": category})
        assert len(citations) == 3
        for row in citations:
            assert row["source_snapshot"] == "sha:v1"
            assert row["content_hash"]


# ── 3. deterministic replay produces zero delta ──────────────────────────────


class TestReplayIdempotence:
    def test_replaying_the_same_extraction_adds_zero_nodes(
        self, engine: FakeEngine
    ) -> None:
        def emit() -> None:
            record_claim(
                engine,
                ClassificationClaim.observe(
                    subject_id="art:demo",
                    category="code",
                    value="python",
                    evidence_refs=["frag:a"],
                    source_snapshot="sha:v1",
                ),
            )

        emit()
        before = engine.claim_count()
        emit()  # exact same inputs, second pass
        after = engine.claim_count()
        assert after == before
        assert len(query_claims(engine, "art:demo", category="code")) == 1


# ── 4. a new source version supersedes without deleting history ─────────────


class TestSupersession:
    def test_new_source_version_supersedes_prior_claim_without_deleting_it(
        self, engine: FakeEngine
    ) -> None:
        ledger = ClassificationPromotionLedger(engine)
        v1 = ClassificationClaim.observe(
            subject_id="art:demo",
            category="code",
            value="python3.11",
            evidence_refs=["frag:a"],
            source_snapshot="sha:v1",
        )
        record_claim(engine, v1)

        v2 = ClassificationClaim.observe(
            subject_id="art:demo",
            category="code",
            value="python3.12",
            evidence_refs=["frag:a"],
            source_snapshot="sha:v2",
        )
        updated_v1, recorded_v2 = ledger.supersede(
            v1, v2, reason="re-extracted at sha:v2"
        )

        assert updated_v1.status == "superseded"
        assert updated_v1.superseded_by == v2.claim_id
        history = query_claim_history(engine, "art:demo", "code")
        assert {c.claim_id for c in history} == {v1.claim_id, v2.claim_id}
        assert any(c.status == "superseded" for c in history)
        assert any(c.status == "promoted" and c.value == "python3.12" for c in history)


# ── 5. candidate-method claims: cannot be created as accepted, need provenance ──


class TestCandidateClaimContract:
    def test_a_candidate_method_claim_cannot_be_created_directly_as_promoted(
        self,
    ) -> None:
        from agent_utilities.knowledge_graph.ontology.classification_claims import (
            claim_id_for,
        )

        cid = claim_id_for("subj", "cat", "val", "derived", ["frag:a"], "sha:v1")
        with pytest.raises(ValueError, match="never implicit promotion"):
            ClassificationClaim(
                claim_id=cid,
                subject_id="subj",
                category="cat",
                value="val",
                method="derived",
                status="promoted",
                evidence_refs=("frag:a",),
                source_snapshot="sha:v1",
                # reviewer intentionally omitted
            )

    def test_a_candidate_method_claim_cannot_be_created_directly_as_rejected(
        self,
    ) -> None:
        from agent_utilities.knowledge_graph.ontology.classification_claims import (
            claim_id_for,
        )

        cid = claim_id_for("subj", "cat", "val", "generated", ["frag:a"], "sha:v1")
        with pytest.raises(ValueError, match="never implicit promotion"):
            ClassificationClaim(
                claim_id=cid,
                subject_id="subj",
                category="cat",
                value="val",
                method="generated",
                status="rejected",
                evidence_refs=("frag:a",),
                source_snapshot="sha:v1",
            )

    def test_reaching_promoted_with_a_reviewer_recorded_is_allowed(self) -> None:
        from agent_utilities.knowledge_graph.ontology.classification_claims import (
            claim_id_for,
        )

        cid = claim_id_for("subj", "cat", "val", "derived", ["frag:a"], "sha:v1")
        claim = ClassificationClaim(
            claim_id=cid,
            subject_id="subj",
            category="cat",
            value="val",
            method="derived",
            status="promoted",
            evidence_refs=("frag:a",),
            source_snapshot="sha:v1",
            reviewer="reviewer-1",
        )
        assert claim.status == "promoted"
        assert claim.reviewer == "reviewer-1"

    def test_whitespace_only_reviewer_is_not_a_governance_decision(self) -> None:
        from agent_utilities.knowledge_graph.ontology.classification_claims import (
            claim_id_for,
        )

        cid = claim_id_for("subj", "cat", "val", "generated", ["frag:a"], "sha:v1")
        with pytest.raises(ValueError, match="non-blank reviewer"):
            ClassificationClaim(
                claim_id=cid,
                subject_id="subj",
                category="cat",
                value="val",
                method="generated",
                status="promoted",
                evidence_refs=("frag:a",),
                source_snapshot="sha:v1",
                reviewer=" \t",
            )

    def test_non_string_reviewer_cannot_bypass_governed_promotion(self) -> None:
        from agent_utilities.knowledge_graph.ontology.classification_claims import (
            claim_id_for,
        )

        cid = claim_id_for("subj", "cat", "val", "derived", ["frag:a"], "sha:v1")
        with pytest.raises(ValueError, match="non-blank reviewer"):
            ClassificationClaim(
                claim_id=cid,
                subject_id="subj",
                category="cat",
                value="val",
                method="derived",
                status="rejected",
                evidence_refs=("frag:a",),
                source_snapshot="sha:v1",
                reviewer=object(),  # type: ignore[arg-type]
            )

    def test_generated_claim_via_propose_without_model_provenance_is_allowed_but_unset(
        self,
    ) -> None:
        # NOT retroactively required (see module docstring) — the live MCP
        # tool does not collect it yet — but stays cleanly unset, never
        # fabricated, when omitted.
        claim = ClassificationClaim.propose(
            subject_id="art:demo",
            category="skill-resource",
            value="true",
            evidence_refs=["frag:a"],
            source_snapshot="sha:v1",
            method="generated",
            policy_approved=True,
        )
        assert claim.model_profile_version == ""
        assert claim.prompt_digest == ""

    def test_model_provenance_must_be_set_in_pairs_not_half(self) -> None:
        from agent_utilities.knowledge_graph.ontology.classification_claims import (
            claim_id_for,
        )

        cid = claim_id_for("subj", "cat", "val", "generated", ["frag:a"], "sha:v1")
        with pytest.raises(ValueError, match="must be set together"):
            ClassificationClaim(
                claim_id=cid,
                subject_id="subj",
                category="cat",
                value="val",
                method="generated",
                status="candidate",
                evidence_refs=("frag:a",),
                source_snapshot="sha:v1",
                model_profile_version="claude-sonnet-5",
                # prompt_digest intentionally omitted
            )

    def test_model_provenance_set_together_via_propose(self) -> None:
        claim = ClassificationClaim.propose(
            subject_id="art:demo",
            category="architecture-layer",
            value="service-layer",
            evidence_refs=["frag:a"],
            source_snapshot="sha:v1",
            method="generated",
            policy_approved=True,
            model_profile_version="claude-sonnet-5",
            prompt_digest="sha256:" + "e" * 64,
        )
        assert claim.status == "candidate"
        assert claim.model_profile_version == "claude-sonnet-5"
        assert claim.prompt_digest.startswith("sha256:")


# ── 6. promotion requires the right scope and same tenant/graph ─────────────


class TestPromotionScope:
    def _candidate(self) -> ClassificationClaim:
        return ClassificationClaim.propose(
            subject_id="art:demo",
            category="skill-resource",
            value="true",
            evidence_refs=["frag:a"],
            source_snapshot="sha:v1",
            method="generated",
            policy_approved=True,
            tenant="tenantA",
            graph="g1",
        )

    def test_promotion_refuses_a_mismatched_tenant(self, engine: FakeEngine) -> None:
        ledger = ClassificationPromotionLedger(engine)
        reviewed = ledger.review(self._candidate())
        with pytest.raises(PermissionError, match="tenant"):
            ledger.promote(reviewed, reviewer="r1", tenant="tenantB", graph="g1")

    def test_promotion_refuses_a_mismatched_graph(self, engine: FakeEngine) -> None:
        ledger = ClassificationPromotionLedger(engine)
        reviewed = ledger.review(self._candidate())
        with pytest.raises(PermissionError, match="graph"):
            ledger.promote(reviewed, reviewer="r1", tenant="tenantA", graph="g2")

    def test_correctly_scoped_promotion_succeeds(self, engine: FakeEngine) -> None:
        ledger = ClassificationPromotionLedger(engine)
        reviewed = ledger.review(self._candidate(), tenant="tenantA", graph="g1")
        promoted = ledger.promote(reviewed, reviewer="r1", tenant="tenantA", graph="g1")
        assert promoted.status == "promoted"
        assert promoted.reviewer == "r1"

    def test_unscoped_promotion_still_works_for_backward_compatibility(
        self, engine: FakeEngine
    ) -> None:
        # The live MCP tool does not pass tenant/graph yet — omitting them
        # (the default, None) must keep working exactly as before.
        ledger = ClassificationPromotionLedger(engine)
        reviewed = ledger.review(self._candidate())
        promoted = ledger.promote(reviewed, reviewer="r1")
        assert promoted.status == "promoted"


# ── 7. another graph returns zero rows ───────────────────────────────────────


class TestGraphIsolation:
    def test_another_graph_engine_returns_zero_rows(self, engine: FakeEngine) -> None:
        record_claim(
            engine,
            ClassificationClaim.observe(
                subject_id="art:demo",
                category="code",
                value="python",
                evidence_refs=["frag:a"],
                source_snapshot="sha:v1",
            ),
        )
        other_graph = FakeEngine("graph-b")
        assert query_claims(other_graph, "art:demo") == []
        assert query_categories(other_graph, "art:demo") == set()


# ── 8. same-storage restart returns identical state ──────────────────────────


class TestRestartDurability:
    def test_a_fresh_ledger_over_the_same_engine_reproduces_prior_reads(
        self, engine: FakeEngine
    ) -> None:
        ledger = ClassificationPromotionLedger(engine)
        candidate = ClassificationClaim.propose(
            subject_id="art:demo",
            category="skill-resource",
            value="true",
            evidence_refs=["frag:a"],
            source_snapshot="sha:v1",
            method="generated",
            policy_approved=True,
        )
        reviewed = ledger.review(candidate)
        promoted = ledger.promote(reviewed, reviewer="r1")

        before_categories = query_categories(engine, "art:demo")
        before_history = ledger.history(promoted.claim_id)

        # Simulate a restart: a FRESH ledger instance over the SAME
        # underlying engine/store.
        restarted_ledger = ClassificationPromotionLedger(engine)
        after_categories = query_categories(engine, "art:demo")
        after_history = restarted_ledger.history(promoted.claim_id)

        assert after_categories == before_categories
        assert after_history == before_history
        assert query_claims(engine, "art:demo")[0].claim_id == promoted.claim_id
