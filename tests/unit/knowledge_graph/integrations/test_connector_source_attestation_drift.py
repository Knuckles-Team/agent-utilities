"""A release bump must never invalidate an already-signed source attestation.

CONCEPT:AU-KG.ontology.derived-compatibility-band. The regression these tests pin
down is D-OB-6: ``.bumpversion.cfg`` rewrote the literal compatibility band inside
:mod:`connector_source_attestation` on every bump while the gate compared that band
for exact equality, so a routine patch release invalidated all 68 signed provider
attestations at once.
"""

from __future__ import annotations

import base64
import configparser
import importlib.util
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from agent_utilities.knowledge_graph.integrations import connector_source_attestation
from agent_utilities.knowledge_graph.integrations.connector_source_attestation import (
    build_source_attestation,
    compatibility_violations,
    source_attestation_violations,
    source_compatibility,
)
from agent_utilities.knowledge_graph.ontology import ontology_integrity
from agent_utilities.knowledge_graph.ontology.connector_manifest import (
    ConnectorManifest,
    IntegrityInfo,
    ProvenanceSpec,
    SyncSpec,
)

ROOT = Path(__file__).resolve().parents[4]

_CHECK_SPEC = importlib.util.spec_from_file_location(
    "check_version_consistency_bump_coupling",
    ROOT / "scripts" / "check_version_consistency.py",
)
version_gate = importlib.util.module_from_spec(_CHECK_SPEC)
assert _CHECK_SPEC.loader is not None
_CHECK_SPEC.loader.exec_module(version_gate)

#: The exact band every one of the 68 preserved Jul-28 provider bundles declares.
PRESERVED_FLEET_COMPATIBILITY = {
    "agent_utilities": ">=2.1.0,<3",
    "bundle_schema": "2",
    "epistemic_graph": ">=2.23.1,<3",
}


@pytest.fixture(autouse=True)
def _clear_compatibility_cache() -> Any:
    connector_source_attestation.reset_compatibility_cache()
    yield
    connector_source_attestation.reset_compatibility_cache()


@pytest.fixture
def signer(monkeypatch: pytest.MonkeyPatch) -> ontology_integrity.ReleaseSigner:
    key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    monkeypatch.setenv("ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL", key)
    monkeypatch.setenv(
        "ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF",
        "env://ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL",
    )
    return ontology_integrity.ReleaseSigner.from_runtime()


def _pin(monkeypatch: pytest.MonkeyPatch, runtime: str, engine: str) -> None:
    """Repoint the two derived version authorities, as a release bump would."""

    connector_source_attestation.reset_compatibility_cache()
    monkeypatch.setattr(
        connector_source_attestation, "_released_runtime_version", lambda: runtime
    )
    monkeypatch.setattr(
        connector_source_attestation, "_released_engine_version", lambda: engine
    )


def _manifest() -> ConnectorManifest:
    return ConnectorManifest(
        connector="fixture-connector",
        sync=[
            SyncSpec(
                preset="widgets",
                server="widget-mcp",
                tool="list_widgets",
                tool_schema_sha256="4" * 64,
            )
        ],
        provenance=ProvenanceSpec(integrity=IntegrityInfo(hash="5" * 64)),
    )


def test_compatibility_band_is_a_minor_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    _pin(monkeypatch, "2.1.1", "2.23.2")

    assert source_compatibility() == {
        "agent_utilities": ">=2.1.0,<3",
        "bundle_schema": "2",
        "epistemic_graph": ">=2.23.0,<3",
    }


@pytest.mark.parametrize("patch", ["2.1.2", "2.1.9", "2.1.147"])
def test_a_patch_bump_recomputes_the_identical_band(
    monkeypatch: pytest.MonkeyPatch, patch: str
) -> None:
    _pin(monkeypatch, "2.1.1", "2.23.2")
    before = source_compatibility()

    _pin(monkeypatch, patch, "2.23.9")

    assert source_compatibility() == before


def test_a_signed_attestation_survives_a_simulated_patch_bump(
    monkeypatch: pytest.MonkeyPatch, signer: ontology_integrity.ReleaseSigner
) -> None:
    """The exact D-OB-6 scenario: bump the patch, re-validate what was signed."""

    _pin(monkeypatch, "2.1.1", "2.23.2")
    manifest = _manifest()
    attestation = build_source_attestation(
        manifest,
        artifacts={"connector_manifest.yml": "6" * 64},
        now=datetime(2026, 7, 28, tzinfo=UTC),
        release_signer=signer,
    )
    signed_bytes = dict(attestation)

    _pin(monkeypatch, "2.1.2", "2.23.2")

    assert attestation == signed_bytes
    assert source_attestation_violations(attestation, manifest) == []
    assert ontology_integrity.verify_release_signature(
        ontology_integrity.canonical_signed_document_hash(
            {**attestation, "signature": None}
        ),
        attestation["signature"],
        signer_id=attestation["signer"],
        algorithm=attestation["signature_algorithm"],
        public_key=attestation["signing_public_key"],
        trusted_public_keys=(attestation["signing_public_key"],),
    )


def test_a_minor_bump_also_leaves_signed_attestations_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin(monkeypatch, "2.9.9", "2.23.2")

    assert compatibility_violations(PRESERVED_FLEET_COMPATIBILITY) == []


def test_the_preserved_fleet_bundles_are_admissible_at_the_released_version() -> None:
    """All 68 Jul-28 bundles failed on this one field; they must now pass."""

    assert compatibility_violations(PRESERVED_FLEET_COMPATIBILITY) == []


def test_a_major_bump_is_the_only_boundary_that_invalidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin(monkeypatch, "3.0.0", "2.23.2")

    assert compatibility_violations(PRESERVED_FLEET_COMPATIBILITY) == [
        "source attestation agent_utilities band excludes the released version"
    ]


@pytest.mark.parametrize(
    "band",
    [">=2.1.0", "<3", "not-a-specifier", "==2.1.1"],
)
def test_an_unbounded_or_malformed_band_is_rejected(
    monkeypatch: pytest.MonkeyPatch, band: str
) -> None:
    _pin(monkeypatch, "2.1.1", "2.23.2")

    assert compatibility_violations(
        {**PRESERVED_FLEET_COMPATIBILITY, "agent_utilities": band}
    )


def test_a_foreign_bundle_schema_is_still_compared_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin(monkeypatch, "2.1.1", "2.23.2")

    assert compatibility_violations(
        {**PRESERVED_FLEET_COMPATIBILITY, "bundle_schema": "3"}
    ) == ["source attestation bundle schema differs"]


def test_bumpversion_never_targets_an_attestation_input() -> None:
    """The structural half of the fix: the bump cannot reach the band at all."""

    targets = set(version_gate.bumpversion_targets(ROOT))

    assert targets.isdisjoint(version_gate.BUMP_FORBIDDEN_PATHS)


def test_the_gate_fails_closed_if_the_coupling_is_reintroduced(tmp_path: Path) -> None:
    parser = configparser.ConfigParser()
    parser.read_string((ROOT / ".bumpversion.cfg").read_text(encoding="utf-8"))
    section = (
        "bumpversion:file:"
        "agent_utilities/knowledge_graph/integrations/connector_source_attestation.py"
    )
    parser.add_section(section)
    parser.set(section, "search", '">={current_version},<3"')
    parser.set(section, "replace", '">={new_version},<3"')
    with (tmp_path / ".bumpversion.cfg").open("w", encoding="utf-8") as handle:
        parser.write(handle)

    reintroduced = set(version_gate.bumpversion_targets(tmp_path))

    assert not reintroduced.isdisjoint(version_gate.BUMP_FORBIDDEN_PATHS)
