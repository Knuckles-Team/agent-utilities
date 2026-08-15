"""U-07 regression: a background WorkItem claim must resolve the verified
AMBIENT session's tenant, never a hardcoded bootstrap/default tenant.

``work_item.claim_next()`` (used by the ingest task-worker poll loop,
``engine_tasks.py::_claim_next_task``) is called with NO explicit ``tenant=``
by both call sites (blind claim), so its request-building helper
(``_claim_request``) falls through to ``_work_tenant("")`` ->
``resolve_session(required_scope="kg:write").tenant`` -- the AMBIENT verified
session's own tenant, restored inside the background worker thread by
``_run_with_background_authority``'s ``with use_session(session):`` (see
``engine_tasks.py``'s ``_authorized_background_thread`` family). This test
pins that resolution: a background thread whose captured session carries a
REAL tenant (not the platform default/bootstrap tenant) must produce a claim
request scoped to THAT tenant, not a different/default one -- proving the
worker loop cannot silently drift onto the bootstrap tenant's queue while
running under a real tenant's authority.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.models.company_brain import ActorType
from agent_utilities.orchestration import work_item as wi
from agent_utilities.security.brain_context import ActorContext


def _session(tenant: str) -> GraphSession:
    actor = ActorContext(
        actor_id="worker:ingest-daemon",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("kg:read", "kg:write"),
        tenant_id=tenant,
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant=tenant,
        scopes=frozenset({"kg:read", "kg:write"}),
        policy_version="test-policy-v1",
        audience="test-audience",
    )


def test_blind_claim_request_resolves_the_ambient_session_tenant_not_default():
    """A background worker's captured, real-tenant session must produce a
    claim request scoped to that tenant."""
    with use_session(_session("acme-real-tenant")):
        request = wi._claim_request(
            None,
            item_id=None,
            queue="ingest_task",
            tenant=None,  # exactly how claim_next()'s two call sites invoke it
            resource_class=None,
            fairness_group=None,
            token="worker-token:1",
            now=1_700_000_000.0,
            lease_ttl_s=60.0,
        )
    assert request.tenant_ref == "acme-real-tenant"


def test_blind_claim_request_never_silently_resolves_to_a_different_tenant():
    """A second, DIFFERENT tenant's ambient session must resolve to ITS OWN
    tenant, not silently reuse the previous one or a hardcoded default --
    proving the resolution is genuinely per-call/per-thread, not cached."""
    with use_session(_session("beta-other-tenant")):
        request = wi._claim_request(
            None,
            item_id=None,
            queue="ingest_task",
            tenant=None,
            resource_class=None,
            fairness_group=None,
            token="worker-token:2",
            now=1_700_000_000.0,
            lease_ttl_s=60.0,
        )
    assert request.tenant_ref == "beta-other-tenant"
    assert request.tenant_ref != "acme-real-tenant"
    assert request.tenant_ref not in {"default", "bootstrap", ""}
