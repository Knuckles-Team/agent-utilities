"""GOC-19 — WorkItem submission command preparation (carrier + digest replay).

Known-bad-input proofs required by the lane doc: a replayed command with the
same idempotency key must not double-apply (proven here at the digest layer:
two calls to :func:`prepare_submit_work_item_command` with identical logical
fields always produce the SAME ``command_digest``, and any field that changes
the command's meaning changes it), and a command constructed under a
malformed/unverified carrier must be rejected before it is ever canonicalized.
"""

from __future__ import annotations

import pytest

from agent_utilities.orchestration.work_item_command import (
    CommandCarrierRejected,
    canonical_command_digest,
    prepare_submit_work_item_command,
    require_verified_carrier,
)


def _carrier(**overrides: object) -> dict[str, object]:
    claims: dict[str, object] = {
        "principal": "principal:sha256:" + "a" * 64,
        "tenant": "tenant-a",
        "audience": "graph-os",
        "agent_id": "agent-1",
        "roles": ["kg:write"],
        "scopes": ["kg:write"],
        "delegation": [],
        "policy_version": "policy-v1",
    }
    claims.update(overrides)
    return claims


def _command_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "carrier_claims": _carrier(),
        "kind": "ingest_task",
        "idempotency_key": "idem-1",
        "policy_digest": "policy-digest-1",
        "catalog_digest": "catalog-digest-1",
        "model_digest": "model-digest-1",
    }
    kwargs.update(overrides)
    return kwargs


class TestRequireVerifiedCarrier:
    def test_well_formed_carrier_passes_through(self) -> None:
        claims = _carrier()
        assert require_verified_carrier(claims) is claims

    def test_missing_required_field_is_rejected(self) -> None:
        claims = _carrier()
        del claims["tenant"]
        with pytest.raises(CommandCarrierRejected, match="missing required field"):
            require_verified_carrier(claims)

    def test_unrecognized_field_is_rejected(self) -> None:
        claims = _carrier(bogus="nope")
        with pytest.raises(CommandCarrierRejected, match="unrecognized field"):
            require_verified_carrier(claims)

    def test_non_dict_is_rejected(self) -> None:
        with pytest.raises(CommandCarrierRejected):
            require_verified_carrier("not-a-dict")  # type: ignore[arg-type]


class TestCanonicalCommandDigest:
    def test_digest_is_bare_64_char_hex(self) -> None:
        digest = canonical_command_digest({"a": 1})
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_key_order_does_not_affect_digest(self) -> None:
        assert canonical_command_digest({"a": 1, "b": 2}) == canonical_command_digest(
            {"b": 2, "a": 1}
        )

    def test_different_content_changes_digest(self) -> None:
        assert canonical_command_digest({"a": 1}) != canonical_command_digest({"a": 2})


class TestPrepareSubmitWorkItemCommand:
    def test_rejects_a_malformed_carrier_before_touching_anything_else(self) -> None:
        claims = _carrier()
        del claims["roles"]
        with pytest.raises(CommandCarrierRejected):
            prepare_submit_work_item_command(**_command_kwargs(carrier_claims=claims))

    def test_rejects_a_requested_tenant_that_does_not_match_the_verified_carrier(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="does not match the verified carrier"):
            prepare_submit_work_item_command(
                **_command_kwargs(tenant="a-different-tenant")
            )

    def test_command_carries_the_verified_carrier_tenant_not_a_caller_supplied_one(
        self,
    ) -> None:
        command = prepare_submit_work_item_command(**_command_kwargs())
        assert command["context"]["tenant"] == "tenant-a"

    # ---- KNOWN-BAD: replay must not double-apply -----------------------------

    def test_replaying_the_same_logical_request_yields_the_same_command_digest(
        self,
    ) -> None:
        first = prepare_submit_work_item_command(**_command_kwargs())
        second = prepare_submit_work_item_command(**_command_kwargs())
        assert first["command_digest"] == second["command_digest"]

    def test_depends_on_ordering_does_not_change_the_digest(self) -> None:
        a = prepare_submit_work_item_command(
            **_command_kwargs(depends_on=["dep-a", "dep-b"])
        )
        b = prepare_submit_work_item_command(
            **_command_kwargs(depends_on=["dep-b", "dep-a"])
        )
        assert a["command_digest"] == b["command_digest"]

    def test_metadata_key_ordering_does_not_change_the_digest(self) -> None:
        a = prepare_submit_work_item_command(
            **_command_kwargs(metadata={"x": 1, "y": 2})
        )
        b = prepare_submit_work_item_command(
            **_command_kwargs(metadata={"y": 2, "x": 1})
        )
        assert a["command_digest"] == b["command_digest"]

    # ---- KNOWN-BAD: a changed field must NOT be treated as a replay ----------

    def test_a_different_payload_produces_a_different_digest(self) -> None:
        a = prepare_submit_work_item_command(**_command_kwargs(kind="ingest_task"))
        b = prepare_submit_work_item_command(**_command_kwargs(kind="loop_task"))
        assert a["command_digest"] != b["command_digest"]

    def test_a_different_idempotency_key_produces_a_different_digest(self) -> None:
        a = prepare_submit_work_item_command(**_command_kwargs(idempotency_key="k1"))
        b = prepare_submit_work_item_command(**_command_kwargs(idempotency_key="k2"))
        assert a["command_digest"] != b["command_digest"]

    # ---- Bounds -----------------------------------------------------------

    def test_rejects_empty_required_fields(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            prepare_submit_work_item_command(**_command_kwargs(kind=""))
        with pytest.raises(ValueError, match="idempotency_key"):
            prepare_submit_work_item_command(**_command_kwargs(idempotency_key=""))
        with pytest.raises(ValueError, match="policy_digest"):
            prepare_submit_work_item_command(**_command_kwargs(policy_digest=""))

    def test_rejects_too_many_dependencies(self) -> None:
        from agent_utilities.orchestration.work_item_command import MAX_DEPENDENCIES

        too_many = [f"dep-{i}" for i in range(MAX_DEPENDENCIES + 1)]
        with pytest.raises(ValueError, match="depends_on"):
            prepare_submit_work_item_command(**_command_kwargs(depends_on=too_many))

    def test_rejects_oversize_input_ref(self) -> None:
        from agent_utilities.orchestration.work_item_command import (
            MAX_PAYLOAD_REF_BYTES,
        )

        oversize = "x" * (MAX_PAYLOAD_REF_BYTES + 1)
        with pytest.raises(ValueError, match="input_ref"):
            prepare_submit_work_item_command(**_command_kwargs(input_ref=oversize))

    def test_deduplicates_and_sorts_dependencies(self) -> None:
        command = prepare_submit_work_item_command(
            **_command_kwargs(depends_on=["dep-b", "dep-a", "dep-b"])
        )
        assert command["depends_on"] == ["dep-a", "dep-b"]
