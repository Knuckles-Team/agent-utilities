from __future__ import annotations

from agent_utilities.security.persistence_privacy import (
    PersistencePrivacyGuard,
    persistence_reference,
    sanitize_for_persistence,
)


def test_sanitizes_identifiers_person_fields_secrets_and_machine_paths() -> None:
    payload = {
        "domain": {"name": "Synthetic Service Domain"},
        "person": {"name": "Example Person"},
        "owner_name": "Example Owner",
        "email": "contact@example.test",
        "token": "not-a-real-token",
        "workspace_path": "/home/agent-user/workspace/project",
        "content": "Reach contact@example.test from /home/example/private/file.md",
    }

    clean, report = sanitize_for_persistence(payload)

    assert clean["domain"]["name"] == "Synthetic Service Domain"
    assert clean["person"]["name"] == "[REDACTED_PERSON]"
    assert clean["owner_name"] == "[REDACTED_PERSON]"
    assert clean["email"] == "[REDACTED_EMAIL]"
    assert clean["token"] == "[REDACTED_SECRET]"
    assert clean["workspace_path"] == "[REDACTED_LOCATION]"
    assert "contact@example.test" not in clean["content"]
    assert "/home/example" not in clean["content"]
    assert report.changed is True
    assert set(report.detected_types) >= {
        "email",
        "personal_field",
        "secret_field",
        "location_field",
        "posix_user_path",
    }


def test_runtime_identity_terms_are_redacted_without_appearing_in_report() -> None:
    identity_term = "Sample Identity"
    guard = PersistencePrivacyGuard(deny_terms=[identity_term])

    clean, report = guard.sanitize_text(f"Prepared for {identity_term}")

    assert identity_term not in clean
    assert clean == "Prepared for [REDACTED_IDENTITY_TERM]"
    assert report.as_dict() == {
        "redactions": 1,
        "detected_types": ["identity_term"],
    }


def test_opaque_objects_never_persist_repr_content() -> None:
    class ObjectWithSensitiveRepr:
        def __repr__(self) -> str:
            return "ObjectWithSensitiveRepr(secret='value')"

    clean, report = sanitize_for_persistence(ObjectWithSensitiveRepr())

    assert clean == "[REDACTED_OBJECT:ObjectWithSensitiveRepr]"
    assert report.detected_types == ("opaque_object",)


def test_runtime_identity_is_derived_and_redacted_without_configuration(
    monkeypatch,
) -> None:
    monkeypatch.setenv("USER", "synthetic-local-identity")

    clean, report = PersistencePrivacyGuard().sanitize_text(
        "workspace owner synthetic-local-identity"
    )

    assert "synthetic-local-identity" not in clean
    assert "identity_term" in report.detected_types


def test_generic_hermetic_home_directory_is_not_an_identity_term(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", "/tmp/privacy-test/home")
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("LOGNAME", raising=False)
    monkeypatch.delenv("USERNAME", raising=False)

    clean, report = PersistencePrivacyGuard().sanitize_text("home-assistant")

    assert clean == "home-assistant"
    assert "identity_term" not in report.detected_types


def test_remote_blob_reference_is_preserved_but_local_reference_is_redacted() -> None:
    clean, report = sanitize_for_persistence(
        {
            "remote": {"blob_ref": "s3://synthetic-bucket/object.bin"},
            "local": {"blob_ref": "file:///home/example/private/object.bin"},
        }
    )

    assert clean["remote"]["blob_ref"] == "s3://synthetic-bucket/object.bin"
    assert clean["local"]["blob_ref"] == "[REDACTED_FILE_URI]"
    assert set(report.detected_types) >= {"file_uri", "posix_user_path"}


def test_persistence_reference_is_stable_and_never_contains_raw_identity() -> None:
    raw = "synthetic-person@example.test"

    first = persistence_reference("source_object", raw, namespace="connector")
    second = persistence_reference("source_object", raw, namespace="connector")

    assert first == second
    assert first.startswith("pref_source_object_")
    assert raw not in first


def test_host_and_generic_path_fields_are_location_redacted() -> None:
    clean, report = sanitize_for_persistence(
        {"hostname": "synthetic-device", "path": "relative/private/location"}
    )

    assert clean == {
        "hostname": "[REDACTED_LOCATION]",
        "path": "[REDACTED_LOCATION]",
    }
    assert "location_field" in report.detected_types


def test_nested_source_url_aliases_are_location_redacted() -> None:
    clean, report = sanitize_for_persistence(
        {
            "record": {
                "canonical": [{"href": "https://internal.example/item"}],
                "pdf_url": "https://internal.example/item.pdf",
                "source_uri": "connector://private/item",
            }
        }
    )

    record = clean["record"]
    assert record["canonical"][0]["href"] == "[REDACTED_LOCATION]"
    assert record["pdf_url"] == "[REDACTED_LOCATION]"
    assert record["source_uri"] == "[REDACTED_LOCATION]"
    assert report.redactions == 3


def test_camel_case_web_url_is_location_redacted() -> None:
    clean, report = sanitize_for_persistence(
        {"webUrl": "https://private-host.invalid/opaque-record"}
    )

    assert clean == {"webUrl": "[REDACTED_LOCATION]"}
    assert report.detected_types == ("location_field",)
