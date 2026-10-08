"""AU-ADMISSION-GENESIS (NE-064/NE-065) — error-message contract for au's own
engine-identity admission credential (``EPISTEMIC_GRAPH_SIGNER_KEYS_JSON``).

This test does not exercise a live engine (see the module docstring of
``agent_utilities/security/system_rbac_admission.py``, "PREPARE-ONLY"). It
proves the operator-facing failure text in ``resolve_admission_authority``
points at the reference doc, not just a bare instruction.

The reference doc itself (``references/engine-identity-admission.md``) and
the genesis skill that links to it from Phase 5 and the Phase 8 exit gate
moved to graph-os's ``graphos-genesis`` skill on 2026-10-03, along with every
other domain-tier skill; graph-os's own
``tests/skills/test_engine_identity_admission_doc.py`` covers the doc's
content, placeholders, credential-shape safety, and skill linkage there now.
"""

from __future__ import annotations

import pytest

from agent_utilities.security import admission_authority, brain_context


def test_missing_signer_key_error_points_at_the_reference_doc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(admission_authority.SIGNER_REGISTRY_ENV, raising=False)
    actor = brain_context.ActorContext(actor_id="some:principal", authenticated=True)
    token = brain_context.set_actor(actor)
    try:
        with pytest.raises(admission_authority.AdmissionAuthorityError) as exc_info:
            admission_authority.resolve_admission_authority()
    finally:
        brain_context.reset_actor(token)

    message = str(exc_info.value)
    # Names the exact principal and the exact registry to provision it into,
    # so the operator never has to read source to act on it.
    assert "some:principal" in message
    assert admission_authority.SIGNER_REGISTRY_ENV in message
    # Routes to the full provisioning/verification procedure.
    assert "engine-identity-admission.md" in message
