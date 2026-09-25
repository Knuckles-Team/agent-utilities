"""AU-ADMISSION-GENESIS (NE-064/NE-065) — the operator-facing error contract of
the engine-identity admission credential (``EPISTEMIC_GRAPH_SIGNER_KEYS_JSON``).

The reference procedure itself (``references/engine-identity-admission.md``)
ships with the graph-os ``graphos-genesis`` skill and is contract-tested there.
This test proves the failure text in ``resolve_admission_authority`` names the
principal and the registry and routes the operator to that procedure.
"""

from __future__ import annotations

import pytest

from agent_utilities.security import admission_authority, brain_context

# ---------------------------------------------------------------------------
# Deliverable 3: the failure text is legible and points at the new doc.
# ---------------------------------------------------------------------------


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
    assert "graphos-genesis" in message
