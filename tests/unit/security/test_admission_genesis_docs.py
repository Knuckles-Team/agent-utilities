"""AU-ADMISSION-GENESIS (NE-064/NE-065) — documentation and error-message
contract for au's own engine-identity admission credential
(``engine-admission/provisioner`` / ``EPISTEMIC_GRAPH_SIGNER_KEYS_JSON``).

These tests do not exercise a live engine (see the module docstring of
``agent_utilities/security/system_rbac_admission.py``, "PREPARE-ONLY"). They
prove three things instead:

- the operator-facing failure text in ``resolve_provisioner_authority``
  points at the new reference doc, not just the CLI command;
- the new reference doc (``references/engine-identity-admission.md``) exists,
  covers the required sections, names the real chain (exact secret key, env
  var, role, control graph name) so it is followable without reading source,
  and never contains a value that looks like a real credential; and
- the genesis ``SKILL.md``/``security-and-operations.md`` actually link to it
  from Phase 5 and the Phase 8 exit gate, per AU-ADMISSION-GENESIS deliverable
  2 — a doc nobody is routed to is not documentation, it is a dead file.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent_utilities.security import system_rbac_admission as sra

REPO_ROOT = Path(__file__).resolve().parents[3]
SKILL_DIR = REPO_ROOT / "agent_utilities" / "skills" / "workflows" / "agent-os-genesis"
REFERENCE = SKILL_DIR / "references" / "engine-identity-admission.md"
SKILL_MD = SKILL_DIR / "SKILL.md"
SECURITY_OPS = SKILL_DIR / "references" / "security-and-operations.md"


class _FakeSecretsClient:
    def __init__(self, value: str | None) -> None:
        self._value = value

    def get(self, key: str) -> str | None:
        return self._value


# ---------------------------------------------------------------------------
# Deliverable 3: the failure text is legible and points at the new doc.
# ---------------------------------------------------------------------------


def test_missing_provisioner_error_points_at_the_new_reference_doc() -> None:
    secrets = _FakeSecretsClient(None)
    with pytest.raises(sra.SystemAdmissionError) as exc_info:
        sra.resolve_provisioner_authority(secrets_client=secrets)

    message = str(exc_info.value)
    # Still names the exact missing key and the seeding command (unchanged
    # behavior — regression guard for the pre-existing contract).
    assert sra.DEFAULT_PROVISIONER_SECRET_KEY in message
    assert "python -m agent_utilities.security.cli" in message
    # New: routes the operator to the full provisioning/verification
    # procedure instead of leaving them with only a bare `set` command.
    assert "engine-identity-admission.md" in message


# ---------------------------------------------------------------------------
# Deliverable 1: the reference doc exists, is followable without reading
# source, and never leaks a credential-shaped value.
# ---------------------------------------------------------------------------


def test_reference_doc_exists_and_is_substantial() -> None:
    assert REFERENCE.is_file()
    text = REFERENCE.read_text(encoding="utf-8")
    assert len(text.splitlines()) > 100


@pytest.mark.parametrize(
    "must_contain",
    [
        # The chain, exact names — file/env/secret names named literally so
        # an operator never has to go read source to follow it.
        "engine-admission/provisioner",
        "EPISTEMIC_GRAPH_SIGNER_KEYS_JSON",
        "control:system",
        "__control__",
        "CypherEngineError",
        "resolve_provisioner_authority",
        "register_identity",
        "RegisterIdentity",
        # The four design problems, unmistakably present and not softened
        # into euphemism.
        "unconstrained authority over identity",
        "shared symmetric secret",
        "Bootstrap circularity",
        "No rotation path",
        # Rotation/revocation is concrete, not hand-waved.
        "## Rotation and revocation",
        "openssl rand -hex 32",
        # Failure modes keyed to the real symptoms named in the brief.
        "Pattern(\"tenant__homelab__*\")",
        "SystemAdmissionError",
        # Honest current-state disclosure (hard rule: never claim it works
        # live on this deployment).
        "does not exist",
        "read-only",
    ],
)
def test_reference_doc_covers_required_content(must_contain: str) -> None:
    text = REFERENCE.read_text(encoding="utf-8")
    assert must_contain in text, f"missing required content: {must_contain!r}"


def test_reference_doc_never_contains_a_credential_shaped_value() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    # No 64-hex-char (or longer contiguous hex) string anywhere — a real
    # 256-bit signer key rendered as hex would be exactly this shape. The
    # doc must only ever show the *command* that generates one, never an
    # example output.
    assert re.search(r"\b[0-9a-fA-F]{32,}\b", text) is None

    # Placeholders are present and obviously placeholders.
    assert "<signer-id>" in text
    assert "<hex-key>" in text


def test_reference_doc_does_not_claim_live_provisioning_on_this_deployment() -> None:
    text = REFERENCE.read_text(encoding="utf-8")
    assert "Live state on this deployment" in text
    # Explicitly marks the credential as unprovisioned on this deployment,
    # and the only available OpenBao token as read-only — the hard rule from
    # the AU-ADMISSION-GENESIS brief: never claim the chain currently works.
    assert "does not exist" in text
    assert "read-only" in text
    assert "not as something currently" in text


# ---------------------------------------------------------------------------
# Deliverable 2: genesis actually routes an operator to the new doc.
# ---------------------------------------------------------------------------


def test_skill_phase5_links_the_new_reference() -> None:
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "## Phase 5" in text
    phase5_start = text.index("## Phase 5")
    phase6_start = text.index("## Phase 6")
    phase5 = text[phase5_start:phase6_start]
    assert "engine-identity-admission.md" in phase5


def test_skill_phase8_has_an_admission_exit_gate() -> None:
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "## Phase 8" in text
    phase8_start = text.index("## Phase 8")
    output_start = text.index("## Output")
    phase8 = text[phase8_start:output_start]
    assert "engine-identity-admission.md" in phase8
    assert "CypherEngineError" in phase8


def test_security_and_operations_distinguishes_the_two_signer_concepts() -> None:
    text = SECURITY_OPS.read_text(encoding="utf-8")
    assert "engine-identity-admission.md" in text
    # The pre-existing build/artifact signing section must still be present
    # and the doc must now disambiguate it from the admission signer.
    assert "Signing keys specifically" in text
