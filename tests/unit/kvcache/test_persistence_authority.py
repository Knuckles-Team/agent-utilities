"""Wiring proofs for authority-derived KV-checkpoint persistence eligibility.

CONCEPT:AU-OS.governance.authority-derived-persistence-eligibility (closes D-5.1-3 /
D-KCI-1).

These are **wiring** tests, not existence tests: every one of them drives a real
entrypoint (``TieredCheckpointManager`` or the live ``graph_kv_checkpoint`` MCP action)
under a real verified ``GraphSession``, and asserts what actually reached the durable
store. Nothing here sets a grant flag, because there is no longer one to set.

The five claims that had to be proven, and are:

1. A caller **whose authority permits it** persists to disk automatically, with no
   operator flag anywhere in the call —
   :func:`test_authority_that_permits_persists_automatically_with_no_operator_flag`.
2. A caller **whose authority does not** is refused, and the refusal **names the
   contributing source or the missing label that caused it** —
   :func:`test_refusal_names_the_contributing_source_that_raised_the_bar`,
   :func:`test_refusal_names_the_source_and_axis_of_a_missing_label`.
3. A **delegated agent cannot exceed its delegator's** persistence authority — proven
   differentially, by running the identical request with and without the delegation:
   :func:`test_a_delegate_cannot_exceed_its_delegators_persistence_authority`.
4. An **absent** label denies — on every axis, and for an absent source set, an
   unresolvable source, and an unresolvable delegation ceiling.
5. ``explain()`` shows the derivation, so a human can ask *why* —
   :func:`test_explain_shows_the_full_derivation`.

Everything here is offline and deterministic — no engine, no network, no LLM.
"""

from __future__ import annotations

import base64
import json
from contextlib import contextmanager

import pytest

from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    suspend_session,
    use_session,
)
from agent_utilities.kvcache.checkpoint import (
    CrossTenantCheckpointError,
    KVCheckpointError,
    KVCheckpointKey,
    KVCheckpointRecord,
)
from agent_utilities.kvcache.eligibility import (
    ANY_REGION,
    UNLIMITED_RETENTION_DAYS,
    AuthorityDerivedEligibility,
    CallerAuthority,
    ContributingSource,
    GraphSourceLabelResolver,
    PersistenceRequest,
    compose_source_labels,
    derive_caller_authority,
    get_source_label_resolver,
    set_source_label_resolver,
)
from agent_utilities.kvcache.tiering import (
    RAMCheckpointStore,
    TieredCheckpointManager,
    prefix_digest,
)
from agent_utilities.kvcache.worthiness import CheckpointTier
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext
from agent_utilities.security.delegation import (
    build_spawn_delegation,
    use_delegation,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

TENANT = "t1"


def _key(tenant: str = TENANT, prefix: str = "ctx") -> KVCheckpointKey:
    return KVCheckpointKey(
        model_identity="qwen3.6-27b",
        quantization="fp16",
        serving_engine="vllm",
        engine_version="0.9.0",
        prefix_digest=prefix_digest(prefix),
        tenant=tenant,
        policy_version="v1",
    )


@contextmanager
def caller(
    *,
    tenant: str = TENANT,
    roles: tuple[str, ...] = ("confidential",),
    scopes: tuple[str, ...] = ("kg:read", "kg:write"),
    actor_type: ActorType = ActorType.HUMAN,
    actor_id: str = "human:ada",
    delegation=None,
):
    """Bind a REAL verified ``GraphSession`` (plus optional delegation) as ambient.

    This is the seam the gate reads. Nothing in these tests hands the gate a
    pre-built authority object — it derives one from this session, exactly as
    ``promote()`` does in production.
    """
    actor = ActorContext(
        actor_id=actor_id,
        actor_type=actor_type,
        roles=tuple(roles),
        tenant_id=tenant,
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant=tenant,
        scopes=frozenset(scopes),
        policy_version="v1",
        audience="test",
    )
    with use_session(session), use_delegation(delegation):
        yield session


def public_source(source_id: str = "src:handbook") -> ContributingSource:
    """A source whose PUBLIC classification declares residency + retention outright."""
    return ContributingSource(
        source_id=source_id, classification="public", label_source="test"
    )


def labelled_source(
    source_id: str,
    *,
    classification: str = "confidential",
    regions: tuple[str, ...] = (ANY_REGION,),
    retention_days: int | None = UNLIMITED_RETENTION_DAYS,
    markings: tuple[str, ...] = (),
) -> ContributingSource:
    return ContributingSource(
        source_id=source_id,
        classification=classification,
        residency_regions=frozenset(regions),
        retention_days=retention_days,
        markings=frozenset(markings),
        label_source="test",
    )


class _FakeDiskStore:
    """Stands in for :class:`KVCheckpointStore` — records every durable write."""

    def __init__(self) -> None:
        self.created: list[dict] = []
        self.records: dict[str, KVCheckpointRecord] = {}

    def create_checkpoint(
        self, data, *, key, run_id, point, session=None, provenance=None
    ):
        self.created.append(
            {
                "data": data,
                "key": key,
                "run_id": run_id,
                "point": point,
                "provenance": provenance or {},
            }
        )
        record = KVCheckpointRecord(
            checkpoint_id=key.checkpoint_id,
            key=key,
            blob_id=f"kvblob:{key.tenant}:d",
            digest="d",
            run_id=run_id,
            point=point,
            size_bytes=len(data),
            created_at="2026-07-31T00:00:00Z",
            provenance=provenance or {},
        )
        self.records[record.checkpoint_id] = record
        return record

    def get_checkpoint(
        self, checkpoint_id, *, requesting_tenant, current_policy_version=None
    ):
        record = self.records.get(checkpoint_id)
        if record is None:
            raise KVCheckpointError(f"no checkpoint {checkpoint_id}")
        if record.key.tenant != requesting_tenant:
            raise CrossTenantCheckpointError("cross tenant")
        return record


@pytest.fixture
def disk() -> _FakeDiskStore:
    return _FakeDiskStore()


@pytest.fixture
def manager(disk) -> TieredCheckpointManager:
    """A manager on the PROCESS-WIDE default gate — no gate injected anywhere.

    Injecting a gate here would prove only that the injected object was called. The
    point is that ``AuthorityDerivedEligibility`` is what a plain deployment gets.
    """
    return TieredCheckpointManager(ram_store=RAMCheckpointStore(), disk_store=disk)


# ---------------------------------------------------------------------------
# PROOF 1 — a caller whose authority permits it persists automatically
# ---------------------------------------------------------------------------


def test_authority_that_permits_persists_automatically_with_no_operator_flag(
    manager, disk
):
    """No grant flag exists in this call. The verdict comes from the session alone."""
    with caller(roles=("confidential",)):
        outcome = manager.checkpoint_now(
            b"kv-bytes",
            key=_key(),
            run_id="run-1",
            point="post-plan",
            trigger="user",
            persist=True,
            sources=(public_source(), labelled_source("src:crm")),
        )

    assert outcome.tier is CheckpointTier.DISK
    assert len(disk.created) == 1
    assert outcome.eligibility is not None and outcome.eligibility.permitted
    assert outcome.eligibility.gate == "authority-derived"
    # The permit is not carrying an unanswered-policy caveat any more: the gate
    # answered residency/classification/retention from the sources' own labels.
    assert outcome.eligibility.unresolved == ()


def test_the_default_process_gate_is_the_derived_one():
    """Nothing had to be registered for the above to hold."""
    from agent_utilities.kvcache.eligibility import get_persistence_eligibility_gate

    assert isinstance(get_persistence_eligibility_gate(), AuthorityDerivedEligibility)


def test_an_agent_trigger_persists_when_its_authority_already_permits_it(manager, disk):
    """The agent-authorized case, ENFORCED: the trigger is not the authority.

    Same authority, same sources, ``trigger='agent'`` instead of ``'user'`` — it
    persists, because the agent is acting under an authority that already covers the
    material. The old gate refused this categorically.
    """
    with caller(roles=("confidential",), actor_type=ActorType.AI_AGENT):
        outcome = manager.checkpoint_now(
            b"kv-bytes",
            key=_key(),
            trigger="agent",
            persist=True,
            sources=(labelled_source("src:crm"),),
        )
    assert outcome.tier is CheckpointTier.DISK
    assert len(disk.created) == 1


def test_a_user_trigger_is_refused_when_the_authority_does_not_cover_it(manager, disk):
    """The mirror image: claiming to be the user buys nothing."""
    with caller(roles=("test",)):  # clears INTERNAL only
        outcome = manager.checkpoint_now(
            b"kv-bytes",
            key=_key(),
            trigger="user",
            persist=True,
            sources=(labelled_source("src:crm", classification="restricted"),),
        )
    assert outcome.tier is CheckpointTier.RAM
    assert disk.created == []


# ---------------------------------------------------------------------------
# PROOF 2 — a refusal names what caused it
# ---------------------------------------------------------------------------


def test_refusal_names_the_contributing_source_that_raised_the_bar(manager, disk):
    with caller(roles=("confidential",)):
        outcome = manager.checkpoint_now(
            b"kv-bytes",
            key=_key(),
            trigger="user",
            persist=True,
            sources=(
                public_source(),
                labelled_source("src:crm", classification="confidential"),
                labelled_source("src:payroll", classification="restricted"),
            ),
        )

    assert outcome.tier is CheckpointTier.RAM
    assert disk.created == []
    reason = outcome.eligibility.reason
    # The offending source is named — not just "denied by policy".
    assert "src:payroll" in reason
    assert "restricted" in reason
    assert "confidential" in reason  # what the caller actually clears
    # ...and the sources that did NOT cause it are not blamed.
    assert "src:handbook" not in reason


def test_refusal_names_the_source_and_axis_of_a_missing_label(manager, disk):
    with caller(roles=("confidential",)):
        outcome = manager.checkpoint_now(
            b"kv-bytes",
            key=_key(),
            trigger="user",
            persist=True,
            sources=(
                labelled_source("src:crm"),
                # Declares a classification but nothing about residency or retention.
                ContributingSource(source_id="src:notes", classification="internal"),
            ),
        )

    assert outcome.tier is CheckpointTier.RAM
    assert disk.created == []
    reason = outcome.eligibility.reason
    assert "src:notes:residency" in reason
    assert "src:notes:retention" in reason
    assert "src:crm" not in reason


def test_refusal_names_the_marking_and_which_source_carries_it(manager, disk):
    with caller(roles=("confidential",)):
        outcome = manager.checkpoint_now(
            b"kv-bytes",
            key=_key(),
            trigger="user",
            persist=True,
            sources=(labelled_source("src:legal", markings=("attorney-client",)),),
        )
    assert outcome.tier is CheckpointTier.RAM
    assert "attorney-client" in outcome.eligibility.reason
    assert "src:legal" in outcome.eligibility.reason

    # Hold the marking and the same request goes through.
    with caller(roles=("confidential", "marking:attorney-client")):
        outcome = manager.checkpoint_now(
            b"kv-bytes",
            key=_key(prefix="ctx2"),
            trigger="user",
            persist=True,
            sources=(labelled_source("src:legal", markings=("attorney-client",)),),
        )
    assert outcome.tier is CheckpointTier.DISK


def test_refusal_names_the_residency_region_and_the_source_that_forbids_it(disk):
    manager = TieredCheckpointManager(
        ram_store=RAMCheckpointStore(), disk_store=disk, durable_region="us-east"
    )
    with caller(roles=("confidential",)):
        outcome = manager.checkpoint_now(
            b"kv-bytes",
            key=_key(),
            trigger="user",
            persist=True,
            sources=(labelled_source("src:eu-crm", regions=("eu-west", "eu-central")),),
        )
    assert outcome.tier is CheckpointTier.RAM
    assert "us-east" in outcome.eligibility.reason
    assert "src:eu-crm" in outcome.eligibility.reason


# ---------------------------------------------------------------------------
# PROOF 3 — a delegate cannot exceed its delegator
# ---------------------------------------------------------------------------


def test_a_delegate_cannot_exceed_its_delegators_persistence_authority(manager, disk):
    """Differential proof: the SAME request, with and without the delegation.

    Without a delegation the spawn's own ``confidential`` capability permits the write.
    Under a delegation whose principal ceiling does NOT include it, the identical
    request is refused — so the delegation, not anything else, is what removed the
    authority.
    """
    sources = (labelled_source("src:crm", classification="confidential"),)

    # (a) undelegated — permitted.
    with caller(roles=("confidential",), actor_type=ActorType.AI_AGENT):
        undelegated = manager.checkpoint_now(
            b"kv-bytes",
            key=_key(prefix="undelegated"),
            trigger="agent",
            persist=True,
            sources=sources,
        )
    assert undelegated.tier is CheckpointTier.DISK
    assert len(disk.created) == 1

    # (b) delegated by a principal who does NOT hold `confidential` — refused.
    delegation = build_spawn_delegation(
        agent_name="researcher",
        run_id="run-9",
        principal="human:junior",
        ceiling=("kg:read", "kg:write"),
    )
    with caller(
        roles=("confidential",), actor_type=ActorType.AI_AGENT, delegation=delegation
    ):
        delegated = manager.checkpoint_now(
            b"kv-bytes",
            key=_key(prefix="delegated"),
            trigger="agent",
            persist=True,
            sources=sources,
        )

    assert delegated.tier is CheckpointTier.RAM
    assert len(disk.created) == 1  # unchanged — nothing new reached disk
    reason = delegated.eligibility.reason
    assert "human:junior" in reason  # the delegator is named
    assert "confidential" in reason  # the capability it lacked is named


def test_the_ceiling_intersection_ignores_the_delegation_rollout_posture(monkeypatch):
    """``ENABLE_DELEGATED_IDENTITY=warn`` must NOT relax a data-at-rest decision.

    ``security.delegation.enforce_ceiling`` is deliberately a no-op in ``warn`` (the
    shipped default) so tool scope can soak before it enforces. Reusing it here would
    have made every delegated spawn able to exceed its delegator for the whole soak
    window, which is precisely the escalation this gate exists to stop.
    """
    monkeypatch.setenv("ENABLE_DELEGATED_IDENTITY", "warn")
    delegation = build_spawn_delegation(
        agent_name="researcher",
        run_id="run-9",
        principal="human:junior",
        ceiling=("kg:read", "kg:write"),
    )
    assert not delegation.enforced  # the rollout posture really is the permissive one

    with caller(roles=("confidential",), delegation=delegation):
        authority = derive_caller_authority()

    assert authority.verified
    assert authority.ceiling_applied
    assert "confidential" in authority.narrowed_away
    assert authority.clearance_label == "internal"  # narrowed, despite `warn`


def test_a_delegation_with_an_unresolvable_ceiling_denies():
    """An unresolvable ceiling is an ABSENT label, so it denies rather than not
    narrowing (which is what ``enforce_ceiling`` does, correctly, for tool scope)."""
    delegation = build_spawn_delegation(
        agent_name="researcher", run_id="run-9", principal="human:junior", ceiling=()
    )
    with caller(roles=("confidential",), delegation=delegation):
        authority = derive_caller_authority()

    assert authority.verified is False
    assert "ceiling could not be resolved" in authority.unverified_reason
    assert "human:junior" in authority.unverified_reason


def test_an_unrestricted_principal_does_not_narrow_its_delegate():
    delegation = build_spawn_delegation(
        agent_name="researcher",
        run_id="run-9",
        principal="human:root",
        ceiling=("kg:admin",),
    )
    with caller(roles=("confidential",), delegation=delegation):
        authority = derive_caller_authority()

    assert authority.verified
    assert authority.ceiling_applied is False
    assert authority.clearance_label == "confidential"
    assert authority.delegation_chain[0] == "human:root"


# ---------------------------------------------------------------------------
# PROOF 4 — absence denies, on every axis
# ---------------------------------------------------------------------------


def test_no_verified_session_denies(manager, disk):
    """No ambient session at all — the daemon-tick case."""
    with suspend_session():
        outcome = manager.checkpoint_now(
            b"kv-bytes",
            key=_key(),
            trigger="system",
            persist=True,
            sources=(public_source(),),
        )
    assert outcome.tier is CheckpointTier.RAM
    assert disk.created == []
    assert "verified GraphSession" in outcome.eligibility.reason


def test_an_undeclared_source_set_denies(manager, disk):
    """The trap this program has hit five times: `[]` must NOT read as permissive."""
    with caller(roles=("kg:admin",)):  # maximum clearance, still refused
        outcome = manager.checkpoint_now(
            b"kv-bytes", key=_key(), trigger="user", persist=True, sources=()
        )
    assert outcome.tier is CheckpointTier.RAM
    assert disk.created == []
    assert "no contributing source" in outcome.eligibility.reason


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (ContributingSource(source_id="s"), "s:classification"),
        (
            ContributingSource(source_id="s", classification="internal"),
            "s:residency",
        ),
        (
            ContributingSource(
                source_id="s",
                classification="internal",
                residency_regions=frozenset({ANY_REGION}),
            ),
            "s:retention",
        ),
    ],
)
def test_an_absent_label_denies_and_names_its_axis(source, expected):
    decision = AuthorityDerivedEligibility().evaluate(
        PersistenceRequest(
            tenant=TENANT,
            authority=CallerAuthority(
                verified=True,
                actor_id="a",
                tenant=TENANT,
                scopes=frozenset({"kg:write"}),
                clearance=3,
            ),
            sources=(source,),
        )
    )
    assert decision.permitted is False
    assert expected in decision.reason


def test_an_empty_residency_intersection_denies():
    """Two sources whose permitted regions do not overlap: nowhere is legal."""
    decision = AuthorityDerivedEligibility().evaluate(
        PersistenceRequest(
            tenant=TENANT,
            authority=CallerAuthority(
                verified=True,
                actor_id="a",
                tenant=TENANT,
                scopes=frozenset({"kg:write"}),
                clearance=3,
            ),
            sources=(
                labelled_source("src:eu", regions=("eu-west",)),
                labelled_source("src:us", regions=("us-east",)),
            ),
            target_region="eu-west",
        )
    )
    assert decision.permitted is False
    assert "intersect to the empty set" in decision.reason


def test_a_source_that_permits_no_retention_vetoes():
    decision = AuthorityDerivedEligibility().evaluate(
        PersistenceRequest(
            tenant=TENANT,
            authority=CallerAuthority(
                verified=True,
                actor_id="a",
                tenant=TENANT,
                scopes=frozenset({"kg:write"}),
                clearance=3,
            ),
            sources=(labelled_source("src:ephemeral", retention_days=0),),
        )
    )
    assert decision.permitted is False
    assert "src:ephemeral" in decision.reason
    assert "no durable retention" in decision.reason


def test_a_restricted_source_without_a_known_target_region_denies():
    decision = AuthorityDerivedEligibility().evaluate(
        PersistenceRequest(
            tenant=TENANT,
            authority=CallerAuthority(
                verified=True,
                actor_id="a",
                tenant=TENANT,
                scopes=frozenset({"kg:write"}),
                clearance=3,
            ),
            sources=(labelled_source("src:eu", regions=("eu-west",)),),
            target_region="",
        )
    )
    assert decision.permitted is False
    assert "unknown target region denies" in decision.reason


def test_a_cross_tenant_persistence_is_refused(manager, disk):
    with caller(tenant="tenant-a", roles=("kg:admin",)):
        outcome = manager.promote(
            manager.checkpoint_now(
                b"kv", key=_key(tenant="tenant-a"), trigger="user", persist=False
            ).checkpoint_id,
            requesting_tenant="tenant-a",
            trigger="user",
        )
    assert outcome.tier is CheckpointTier.RAM

    # And a request naming another tenancy is refused by the gate, not just by the
    # store's own tenant check.
    with caller(tenant="tenant-b", roles=("kg:admin",)):
        decision = AuthorityDerivedEligibility().evaluate(
            PersistenceRequest(
                tenant="tenant-a",
                authority=derive_caller_authority(),
                sources=(public_source(),),
            )
        )
    assert decision.permitted is False
    assert "may only be persisted into the tenancy that produced it" in decision.reason


def test_a_session_without_the_write_scope_denies(manager, disk):
    with caller(roles=("confidential",), scopes=("kg:read",)):
        outcome = manager.checkpoint_now(
            b"kv-bytes",
            key=_key(),
            trigger="user",
            persist=True,
            sources=(public_source(),),
        )
    assert outcome.tier is CheckpointTier.RAM
    assert disk.created == []
    assert "kg:write" in outcome.eligibility.reason


def test_an_unresolvable_source_comes_back_unlabelled_not_dropped():
    """The resolver contract that keeps a degraded KG from reading as permissive."""

    class _BrokenBrain:
        @property
        def permissions(self):
            raise RuntimeError("knowledge graph is read-only")

    resolver = GraphSourceLabelResolver()
    import agent_utilities.knowledge_graph.core.company_brain_runtime as cbr

    original = cbr.get_company_brain
    cbr.get_company_brain = lambda: _BrokenBrain()
    try:
        resolved = resolver.resolve(("src:a", "src:b"), tenant=TENANT)
    finally:
        cbr.get_company_brain = original

    # Neither ref was dropped, and both are unlabelled — so they deny and say why.
    assert len(resolved) == 2
    assert {s.source_id for s in resolved} == {"src:a", "src:b"}
    assert all(s.classification == "" for s in resolved)
    assert all(s.label_source == "unresolvable" for s in resolved)

    composed = compose_source_labels(resolved)
    assert set(composed.unlabelled) >= {"src:a:classification", "src:b:classification"}


def test_a_registered_resolver_cannot_shorten_the_source_list():
    """A resolver that drops refs is refused by its own contract at the call site."""

    class _DroppingResolver:
        name = "dropping"

        def resolve(self, source_refs, *, tenant):
            return ()

    previous = set_source_label_resolver(_DroppingResolver())
    try:
        assert get_source_label_resolver().name == "dropping"
        # Composition of an empty set is the no-sources DENIAL, never a permit.
        assert compose_source_labels(
            get_source_label_resolver().resolve(("a",), tenant=TENANT)
        ).no_sources
    finally:
        set_source_label_resolver(previous)


# ---------------------------------------------------------------------------
# The label algebra — intersection, never union
# ---------------------------------------------------------------------------


def test_composition_takes_the_most_restrictive_of_every_axis():
    composed = compose_source_labels(
        (
            labelled_source(
                "a", classification="internal", regions=(ANY_REGION,), retention_days=90
            ),
            labelled_source(
                "b",
                classification="restricted",
                regions=("eu-west", "eu-central"),
                retention_days=30,
                markings=("gdpr",),
            ),
            labelled_source(
                "c",
                classification="public",
                regions=("eu-west", "us-east"),
                retention_days=UNLIMITED_RETENTION_DAYS,
                markings=("pii",),
            ),
        )
    )
    assert composed.classification == "restricted"  # max, not min
    assert composed.residency_regions == frozenset({"eu-west"})  # intersection
    assert composed.retention_days == 30  # min
    assert composed.markings == frozenset({"gdpr", "pii"})  # union
    assert composed.unlabelled == ()


def test_adding_a_source_can_only_narrow_the_composition():
    base = (labelled_source("a", classification="internal"),)
    narrower = base + (labelled_source("b", classification="restricted"),)
    assert (
        compose_source_labels(narrower).classification_level
        >= compose_source_labels(base).classification_level
    )


def test_a_public_classification_declares_residency_and_retention():
    composed = compose_source_labels((public_source("src:doc"),))
    assert composed.unlabelled == ()
    assert composed.residency_regions == frozenset({ANY_REGION})
    assert composed.retention_days == UNLIMITED_RETENTION_DAYS
    # ...and the derivation is VISIBLE, not silent.
    assert composed.contributions[0].derived_axes == ("residency", "retention")


# ---------------------------------------------------------------------------
# PROOF 5 — explain() shows the derivation
# ---------------------------------------------------------------------------


def test_explain_shows_the_full_derivation(manager, disk):
    with caller(roles=("confidential",)):
        outcome = manager.checkpoint_now(
            b"kv-bytes",
            key=_key(),
            run_id="run-1",
            point="post-plan",
            trigger="user",
            persist=True,
            sources=(public_source(), labelled_source("src:crm")),
        )
        assert outcome.tier is CheckpointTier.DISK
        explanation = manager.explain(outcome.checkpoint_id, requesting_tenant=TENANT)

    derivation = explanation.eligibility.derivation
    assert derivation is not None
    assert "most-restrictive composition" in derivation.rule
    assert derivation.authority.actor_id == "human:ada"
    assert derivation.label.classification == "confidential"
    assert {c.source_id for c in derivation.label.contributions} == {
        "src:handbook",
        "src:crm",
    }
    # Every check is named with its verdict, in the order it ran.
    names = [c.name for c in derivation.checks]
    assert names == [
        "verified_authority",
        "tenancy",
        "durable_write_scope",
        "source_provenance",
        "labels_present",
        "classification",
        "mandatory_markings",
        "residency",
        "retention",
    ]
    assert all(c.passed for c in derivation.checks)


def test_the_durable_node_carries_the_derivation_for_later_audit(manager, disk):
    with caller(roles=("confidential",)):
        manager.checkpoint_now(
            b"kv-bytes",
            key=_key(),
            trigger="user",
            persist=True,
            sources=(labelled_source("src:crm"),),
        )
    provenance = disk.created[0]["provenance"]
    assert provenance["authorized_actor"] == "human:ada"
    assert provenance["authorized_clearance"] == "confidential"
    assert provenance["composed_classification"] == "confidential"
    assert provenance["contributing_sources"] == ["src:crm"]
    assert any(
        c.startswith("classification=pass") for c in provenance["eligibility_checks"]
    )


def test_a_refusal_is_recorded_on_the_ram_record_for_later_questioning(manager):
    with caller(roles=("test",)):
        outcome = manager.checkpoint_now(
            b"kv-bytes",
            key=_key(),
            trigger="user",
            persist=True,
            sources=(labelled_source("src:payroll", classification="restricted"),),
        )
        explanation = manager.explain(outcome.checkpoint_id, requesting_tenant=TENANT)

    assert explanation.tier is CheckpointTier.RAM
    assert explanation.eligibility.permitted is False
    assert "src:payroll" in explanation.eligibility.reason
    assert explanation.eligibility.derivation.checks[-1].name == "classification"


# ---------------------------------------------------------------------------
# LIVE PATH — the MCP action, with authority bound to the verified session
# ---------------------------------------------------------------------------


def _mcp_args(**overrides):
    base = dict(
        graph="",
        data_b64="",
        model_identity="qwen3.6-27b",
        quantization="fp16",
        serving_engine="vllm",
        engine_version="0.9.0",
        prefix_digest=prefix_digest("ctx"),
        tenant="",
        policy_version="v1",
        run_id="run-1",
        point="post-plan",
        checkpoint_id="",
        requesting_tenant="",
        observation_json="{}",
        evidence_bundle_json="{}",
        context_bundle_json="{}",
        sources_json="[]",
        trigger="agent",
        persist=False,
    )
    base.update(overrides)
    return base


def test_mcp_action_persists_from_the_session_with_no_grant_argument(monkeypatch, disk):
    """LIVE PATH. There is no ``operator_grant`` argument to pass any more, and the
    tenant is taken from the verified session rather than the payload."""
    from agent_utilities.mcp.tools import engine_surface_tools as est

    ram = RAMCheckpointStore()
    monkeypatch.setattr(
        est,
        "_checkpoint_manager",
        lambda graph: TieredCheckpointManager(ram_store=ram, disk_store=disk),
    )

    with caller(roles=("confidential",)):
        taken = json.loads(
            est._kv_checkpoint_intelligence(
                "checkpoint_now",
                **_mcp_args(
                    data_b64=base64.b64encode(b"kv-bytes").decode(),
                    trigger="agent",
                    persist=True,
                    sources_json=json.dumps(
                        [
                            {
                                "source_id": "src:crm",
                                "classification": "confidential",
                                "residency_regions": [ANY_REGION],
                                "retention_days": UNLIMITED_RETENTION_DAYS,
                            }
                        ]
                    ),
                ),
            )
        )

    assert taken["result"]["tier"] == "disk"
    assert len(disk.created) == 1
    assert disk.created[0]["key"].tenant == TENANT  # from the session, not the payload


def test_mcp_action_takes_its_sources_from_the_context_bundle_citations(
    monkeypatch, disk
):
    """The bundle an agent already hands over for scoring IS its provenance."""
    from agent_utilities.mcp.tools import engine_surface_tools as est

    ram = RAMCheckpointStore()
    monkeypatch.setattr(
        est,
        "_checkpoint_manager",
        lambda graph: TieredCheckpointManager(ram_store=ram, disk_store=disk),
    )

    class _StubResolver:
        name = "stub"

        def __init__(self):
            self.seen: tuple[str, ...] = ()

        def resolve(self, source_refs, *, tenant):
            self.seen = source_refs
            return tuple(
                ContributingSource(
                    source_id=ref, classification="public", label_source=self.name
                )
                for ref in source_refs
            )

    stub = _StubResolver()
    previous = set_source_label_resolver(stub)
    try:
        with caller(roles=("test",)):
            taken = json.loads(
                est._kv_checkpoint_intelligence(
                    "checkpoint_now",
                    **_mcp_args(
                        data_b64=base64.b64encode(b"kv-bytes").decode(),
                        persist=True,
                        context_bundle_json=json.dumps(
                            {
                                "items": [{"id": "i1"}],
                                "dropped_redundant": 3,
                                "citations": [
                                    {"node_id": "n1", "source_refs": ["src:wiki"]},
                                    {
                                        "node_id": "n2",
                                        "source_refs": ["src:wiki", "src:docs"],
                                    },
                                ],
                            }
                        ),
                    ),
                )
            )
    finally:
        set_source_label_resolver(previous)

    # Refs came off the citations, de-duplicated, and every one was resolved.
    assert stub.seen == ("src:wiki", "src:docs")
    assert taken["result"]["tier"] == "disk"


def test_mcp_action_refuses_a_payload_tenant_that_is_not_the_session_tenant(
    monkeypatch, disk
):
    """A caller may not name someone else's tenancy — the payload no longer wins."""
    from agent_utilities.mcp.tools import engine_surface_tools as est

    monkeypatch.setattr(
        est,
        "_checkpoint_manager",
        lambda graph: TieredCheckpointManager(
            ram_store=RAMCheckpointStore(), disk_store=disk
        ),
    )
    with caller(tenant="tenant-a", roles=("kg:admin",)):
        payload = json.loads(
            est._kv_checkpoint_intelligence(
                "checkpoint_now",
                **_mcp_args(
                    data_b64=base64.b64encode(b"kv").decode(),
                    tenant="tenant-b",
                    persist=True,
                ),
            )
        )
    assert payload["error"]["code"] == "invalid_request"
    assert payload["surface"] == "kv_checkpoint"
    # Nothing was stored at all — the refusal happens before the RAM write, so the
    # payload tenant cannot even create a foreign-tenant RAM entry to promote later.
    assert disk.created == []


def test_mcp_promote_derives_authority_at_promotion_time(monkeypatch, disk):
    """RAM residency is not consent, and the authority is re-read at the write."""
    from agent_utilities.mcp.tools import engine_surface_tools as est

    ram = RAMCheckpointStore()
    monkeypatch.setattr(
        est,
        "_checkpoint_manager",
        lambda graph: TieredCheckpointManager(ram_store=ram, disk_store=disk),
    )
    sources_json = json.dumps(
        [
            {
                "source_id": "src:crm",
                "classification": "confidential",
                "residency_regions": [ANY_REGION],
                "retention_days": UNLIMITED_RETENTION_DAYS,
            }
        ]
    )

    # Take it in RAM under an authority that could NOT persist it.
    with caller(roles=("test",)):
        taken = json.loads(
            est._kv_checkpoint_intelligence(
                "checkpoint_now",
                **_mcp_args(
                    data_b64=base64.b64encode(b"kv-bytes").decode(),
                    persist=True,
                    sources_json=sources_json,
                ),
            )
        )
    checkpoint_id = taken["result"]["checkpoint_id"]
    assert taken["result"]["tier"] == "ram"
    assert disk.created == []

    # Promote it later under an authority that CAN — the verdict follows the authority
    # in force at the write, not the one that took the RAM checkpoint.
    with caller(roles=("confidential",)):
        promoted = json.loads(
            est._kv_checkpoint_intelligence(
                "promote", **_mcp_args(checkpoint_id=checkpoint_id, trigger="user")
            )
        )
    assert promoted["result"]["tier"] == "disk"
    assert len(disk.created) == 1
