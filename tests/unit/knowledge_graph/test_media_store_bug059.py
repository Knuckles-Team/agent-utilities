"""BUG-059 governance tests for :class:`MediaStore` (CONCEPT:AU-KG.identity.asset-occurrence).

``memory/media_store.py`` is the worst instance of the 13 BUG-059
CHOKEPOINT_BYPASS sites: it ran its OWN ad-hoc, silently-degrading
``owner``/``tenant`` occurrence scheme (defaulting to ``""`` when no actor
was bound) instead of ever calling the governed ``stamp_ownership``/
``stamp_classification`` (``core.tenant_sharing``) every other node write in
the KG goes through -- because the module's cross-modal ACID commit
(``client.txn.add_node``) cannot reuse
``IntelligenceGraphEngine._upsert_node``/``GraphComputeEngine.add_node``
directly (see the module docstring).

Disposition: REPLACE/AUGMENT, not merely justify. ``stamp_ownership``/
``stamp_classification`` are now the authoritative KG ownership/ACL
mechanism and are applied to every node this module writes, upstream of the
native txn call. The module's own ``owner``/``tenant`` occurrence fields are
kept, unchanged, as the domain-level "who this belongs to" fields -- this is
additive governance, not a rename.

Reuses the fake ``client``/``compute`` scaffolding from
``test_media_store_identity.py`` (same file the existing identity-chain
tests already exercise) rather than duplicating it.
"""

from __future__ import annotations

import contextvars

import pytest

from agent_utilities.knowledge_graph.core.session import GraphSession
from agent_utilities.knowledge_graph.memory.media_store import MediaStore
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, IdentityRequiredError
from tests.unit.knowledge_graph.test_media_store_identity import (
    IMG,
    _digest,
    _FakeClient,
    _FakeCompute,
)


def _auth_session(tenant: str, actor_id: str = "user:1") -> GraphSession:
    return GraphSession(
        actor=ActorContext(
            actor_id=actor_id,
            actor_type=ActorType.HUMAN,
            tenant_id=tenant,
            authenticated=True,
        ),
        tenant=tenant,
    )


def _unauth_session(tenant: str = "acme") -> GraphSession:
    """A session whose actor was never authenticated — the known-bad input
    every ``stamp_ownership`` call site in the KG is expected to refuse.
    ``GraphSession.actor`` is a required (never ``None``) field, so "no
    actor" in practice is this: the default, unauthenticated ``ActorContext``."""
    return GraphSession(actor=ActorContext(), tenant=tenant)


# --------------------------------------------------------------------------- #
# store_media                                                                 #
# --------------------------------------------------------------------------- #


def test_store_media_requires_a_bound_actor():
    """Known-bad input: an explicit, but unauthenticated, session. BEFORE
    BUG-059's fix this silently minted an occurrence with owner="" -- readable
    by nobody in particular but also not refused. AFTER, it raises."""
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))

    with pytest.raises(IdentityRequiredError):
        store.store_media(
            IMG,
            media_type="image",
            mime_type="image/png",
            session=_unauth_session(),
        )

    # Nothing landed — refused before any partial state persisted.
    assert client.txn.nodes == {}
    assert client.blob.incref_calls == []


def test_store_media_requires_a_bound_actor_with_no_ambient_session_either():
    """Same known-bad input, but via the ambient path (no explicit ``session``
    kwarg at all, and nothing bound in the calling context) -- the shape the
    module's own docstring describes ("today's ambient actor")."""
    from agent_utilities.knowledge_graph.core.session import SessionRequiredError

    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))

    def isolated():
        with pytest.raises(SessionRequiredError):
            store.store_media(IMG, media_type="image", mime_type="image/png")

    contextvars.Context().run(isolated)
    assert client.txn.nodes == {}


def test_store_media_stamps_governed_ownership_alongside_the_domain_fields():
    """The occurrence carries BOTH the module's own domain-level owner/tenant
    fields (unchanged) AND the KG's real governance markers (new)."""
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))

    res = store.store_media(
        IMG,
        media_type="image",
        mime_type="image/png",
        session=_auth_session("acme", "user:alice"),
    )

    assert res is not None
    props = client.txn.nodes[res.occurrence_id]
    # Domain-level fields — unchanged, still authoritative for app semantics.
    assert props["owner"] == "user:alice"
    assert props["tenant"] == "acme"
    # KG governance markers — NEW, this is the BUG-059 fix.
    assert props["_owner_id"] == "user:alice"
    assert props["tenant_id"] == "acme"
    assert props["_shared_scope"] == "private"
    assert props["classification"] == "confidential"

    # The shared, deduped :Blob node gets classification (so it is readable
    # at all — secured_reads.permit() default-denies an unregistered ACL) but
    # deliberately NO ownership stamp (it has no single owner by design).
    blob_props = client.txn.nodes[res.blob_id]
    assert blob_props["classification"] == "confidential"
    assert "_owner_id" not in blob_props


def test_store_media_privileged_actor_stays_unowned_like_every_other_write():
    """A privileged (kg:admin) actor's write stays intentionally unowned —
    stamp_ownership's own documented policy for platform data, unchanged by
    this fix; only the missing-actor-entirely case is newly refused."""
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))
    admin_session = GraphSession(
        actor=ActorContext(
            actor_id="svc:ingest",
            actor_type=ActorType.SYSTEM,
            tenant_id="acme",
            authenticated=True,
            roles=("kg:admin",),
        ),
        tenant="acme",
    )

    res = store.store_media(
        IMG, media_type="image", mime_type="image/png", session=admin_session
    )

    assert res is not None
    props = client.txn.nodes[res.occurrence_id]
    assert props["tenant_id"] == "acme"
    assert "_owner_id" not in props  # privileged write stays unowned


# --------------------------------------------------------------------------- #
# migrate_legacy_asset                                                        #
# --------------------------------------------------------------------------- #


def test_migrate_legacy_asset_requires_a_bound_actor():
    client = _FakeClient()
    digest = _digest(b"legacy-bytes")
    legacy_id = f"media:{digest}"
    client.txn.nodes[legacy_id] = {
        "type": "MediaAsset",
        "content_digest": digest,
        "media_type": "image",
        "mime_type": "image/png",
        "source": "legacy-platform",
        "file_size_bytes": 11,
    }
    store = MediaStore(_FakeCompute(client))

    with pytest.raises(IdentityRequiredError):
        store.migrate_legacy_asset(legacy_id, session=_unauth_session())


def test_migrate_legacy_asset_stamps_governed_ownership():
    client = _FakeClient()
    digest = _digest(b"legacy-bytes-2")
    legacy_id = f"media:{digest}"
    client.txn.nodes[legacy_id] = {
        "type": "MediaAsset",
        "content_digest": digest,
        "media_type": "image",
        "mime_type": "image/png",
        "source": "legacy-platform",
        "file_size_bytes": 11,
    }
    store = MediaStore(_FakeCompute(client))

    res = store.migrate_legacy_asset(legacy_id, session=_auth_session("acme", "user:bob"))

    assert res is not None
    props = client.txn.nodes[res.occurrence_id]
    assert props["_owner_id"] == "user:bob"
    assert props["tenant_id"] == "acme"
    assert props["classification"] == "confidential"


# --------------------------------------------------------------------------- #
# record_extraction — read-modify-write; setdefault must not reassign owner   #
# --------------------------------------------------------------------------- #


def test_record_extraction_does_not_reassign_ownership_to_the_extraction_caller():
    """The occurrence was created by ``user:alice``; a later async extraction
    callback (a different, or no, actor) must never become its owner —
    stamp_ownership's setdefault semantics only backfill a MISSING stamp."""
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))
    created = store.store_media(
        IMG,
        media_type="image",
        mime_type="image/png",
        session=_auth_session("acme", "user:alice"),
    )
    assert created is not None

    ok = store.record_extraction(
        created.occurrence_id,
        model="vision-sidecar",
        extracted_text="a chart",
        session=_auth_session("acme", "user:extraction-worker"),
    )

    assert ok is True
    props = client.txn.nodes[created.occurrence_id]
    assert props["_owner_id"] == "user:alice"  # unchanged — NOT the worker
    assert props["extraction_model"] == "vision-sidecar"


def test_record_extraction_requires_a_bound_actor():
    """Known-bad input, same as store_media: an explicit but unauthenticated
    session raises rather than silently succeeding."""
    client = _FakeClient()
    store = MediaStore(_FakeCompute(client))
    created = store.store_media(
        IMG,
        media_type="image",
        mime_type="image/png",
        session=_auth_session("acme", "user:alice"),
    )
    assert created is not None

    with pytest.raises(IdentityRequiredError):
        store.record_extraction(
            created.occurrence_id,
            model="vision-sidecar",
            session=_unauth_session(),
        )
