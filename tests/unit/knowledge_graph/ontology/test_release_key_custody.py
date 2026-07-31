"""Release signing custody: versioned, lock-agreeing, and never silently swapped.

CONCEPT:AU-KG.ontology.release-key-rotation. D-OB-5 in full: the signing seed lived
only in a local ``runtime-secrets.json``, was overwritten in place with a different
seed, and nothing noticed until admission time — by which point every lock entry and
all 68 provider attestations pinned a key that no longer existed anywhere.

Three mechanisms are pinned here:

* signing refuses a key the locks do not pin (detection moved to signing time),
* signing refuses custody that has no version history (the overwrite is what killed
  the original seed; OpenBao KV v2 kept its copy precisely because it is versioned),
* trust is granted by a recorded rotation, never by whichever key is simply present.
"""

from __future__ import annotations

import base64
import secrets as secrets_module
from pathlib import Path

import pytest
import yaml

from agent_utilities.knowledge_graph.ontology import ontology_integrity

PINNED_KEY = "QkigdPNpcUU7x7NcSkwCsUXIQpaFncMQbYSoiSgI2SY"
SWAPPED_KEY = "rJhTqqFwZFHnKJxdp2hfRzCX6Cz4mKbocugAxKTdiE4"


def _lock(tmp_path: Path, *keys: str) -> Path:
    entries = {
        f"agents/connector-{index}/connector_manifest.yml": {
            "hash": "0" * 64,
            "signing_public_key": key,
            "certification_signing_public_key": key,
        }
        for index, key in enumerate(keys)
    }
    path = tmp_path / "ontology.lock"
    path.write_text(yaml.safe_dump(entries, sort_keys=True), encoding="utf-8")
    return path


def _signer(public_key: str) -> ontology_integrity.ReleaseSigner:
    return ontology_integrity.ReleaseSigner(
        signer_id=ontology_integrity.DEFAULT_SIGNER_ID, public_key=public_key
    )


def test_the_locks_pinned_keys_are_read_from_both_signature_fields(
    tmp_path: Path,
) -> None:
    path = _lock(tmp_path, PINNED_KEY, SWAPPED_KEY)

    assert ontology_integrity.lock_pinned_public_keys(path) == frozenset(
        {PINNED_KEY, SWAPPED_KEY}
    )


def test_signing_with_the_pinned_key_is_allowed(tmp_path: Path) -> None:
    ontology_integrity.assert_signing_key_matches_locks(
        _signer(PINNED_KEY), lock_path=_lock(tmp_path, PINNED_KEY)
    )


def test_signing_with_a_swapped_key_is_refused_at_signing_time(
    tmp_path: Path,
) -> None:
    """The exact D-OB-5 condition, caught before any artifact is written."""

    with pytest.raises(ontology_integrity.ReleaseSigningError):
        ontology_integrity.assert_signing_key_matches_locks(
            _signer(SWAPPED_KEY), lock_path=_lock(tmp_path, PINNED_KEY)
        )


def test_a_declared_rotation_is_the_one_way_past_the_check(tmp_path: Path) -> None:
    ontology_integrity.assert_signing_key_matches_locks(
        _signer(SWAPPED_KEY), lock_path=_lock(tmp_path, PINNED_KEY), rotating=True
    )


def test_an_empty_lock_cannot_pin_anything(tmp_path: Path) -> None:
    ontology_integrity.assert_signing_key_matches_locks(
        _signer(SWAPPED_KEY), lock_path=tmp_path / "absent.lock"
    )


def test_publication_refuses_custody_without_version_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ``env://`` seed is a file that can be overwritten beyond recovery."""

    key = base64.urlsafe_b64encode(secrets_module.token_bytes(32)).decode().rstrip("=")
    monkeypatch.setenv("ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL", key)
    monkeypatch.setenv(
        "ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF",
        "env://ONTOLOGY_RELEASE_SIGNING_TEST_MATERIAL",
    )

    with pytest.raises(ontology_integrity.ReleaseSigningError, match="versioned"):
        ontology_integrity.release_signer_for_publication(
            lock_path=_lock(tmp_path, PINNED_KEY)
        )


@pytest.mark.parametrize("scheme", ontology_integrity.DURABLE_SECRET_SCHEMES)
def test_versioned_custody_schemes_are_accepted(scheme: str) -> None:
    assert scheme.endswith("://")
    assert "env://" not in ontology_integrity.DURABLE_SECRET_SCHEMES


def test_the_rotation_ledger_designates_the_active_trust_anchor(
    tmp_path: Path,
) -> None:
    record = tmp_path / "signing-key-rotation.yml"
    record.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "graphos.io/v1",
                "kind": "ReleaseSigningKeyRotation",
                "rotations": [
                    {
                        "rotated_at": "2026-01-01T00:00:00Z",
                        "from_public_key": None,
                        "to_public_key": PINNED_KEY,
                        "reason": "initial trust anchor",
                    },
                    {
                        "rotated_at": "2026-07-30T00:00:00Z",
                        "from_public_key": PINNED_KEY,
                        "to_public_key": SWAPPED_KEY,
                        "reason": "operator-authorised rotation",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    assert ontology_integrity.active_release_public_key(record) == SWAPPED_KEY


def test_no_rotation_record_designates_no_key(tmp_path: Path) -> None:
    assert ontology_integrity.active_release_public_key(tmp_path / "absent.yml") is None


def test_a_malformed_rotation_ledger_fails_closed(tmp_path: Path) -> None:
    record = tmp_path / "signing-key-rotation.yml"
    record.write_text(
        yaml.safe_dump({"rotations": [{"to_public_key": "not-a-key"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ontology_integrity.ReleaseSigningError):
        ontology_integrity.active_release_public_key(record)
