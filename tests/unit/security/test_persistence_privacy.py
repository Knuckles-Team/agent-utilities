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


def test_iban_shaped_structural_ids_survive_distinct_and_unmodified() -> None:
    """Regression test for the IBAN/uuid4-hex collision.

    ``dc0836cf29134aa2a38231c319e6497e`` and ``da90004e126c486abf4579b4646be2d0``
    are deterministic, literal hex strings written directly into this test
    (never ``uuid4()``, so this test can never flake) that both happen to
    satisfy the free-text IBAN shape: two letters in [a-f], two digits, then
    11-30 more alphanumeric groups. Before the ``_STRUCTURAL_ID_FIELDS``
    exemption, `_sanitize_string` would collapse both to the literal
    "[REDACTED_IBAN]", producing duplicate "id" values (a measured 5.37%
    collision rate over 20,000 generated uuid4 hex ids).
    """

    first_id = "action_decision:dc0836cf29134aa2a38231c319e6497e"
    second_id = "action_decision:da90004e126c486abf4579b4646be2d0"
    assert first_id != second_id

    payload = {
        "nodes": [
            {"id": first_id, "type": "action_decision"},
            {"id": second_id, "type": "action_decision"},
        ]
    }

    clean, report = sanitize_for_persistence(payload)

    cleaned_ids = [node["id"] for node in clean["nodes"]]
    assert cleaned_ids == [first_id, second_id]
    assert cleaned_ids[0] != cleaned_ids[1]
    assert "[REDACTED_" not in cleaned_ids[0]
    assert "[REDACTED_" not in cleaned_ids[1]
    assert "iban" not in report.detected_types


def test_live_colliding_action_decision_ids_survive_distinct_and_intact() -> None:
    """Exact reproduction of the live collision measured against
    ``/api/enhanced/graph/nodes``: these two ``action_decision`` ids
    collapsed into one duplicate ``occurrence:[REDACTED_IBAN]``-style key,
    crashing the graph canvas and repointing edges in
    ``get_graph_relationships``.
    """

    node_a = {"id": "action_decision:dc0836cf29134aa2a38231c319e6497e"}
    node_b = {"id": "action_decision:da90004e126c486abf4579b4646be2d0"}

    clean_a, _ = sanitize_for_persistence(node_a)
    clean_b, _ = sanitize_for_persistence(node_b)

    assert clean_a["id"] == "action_decision:dc0836cf29134aa2a38231c319e6497e"
    assert clean_b["id"] == "action_decision:da90004e126c486abf4579b4646be2d0"
    assert clean_a["id"] != clean_b["id"]


def test_real_iban_in_free_text_field_is_still_redacted() -> None:
    """The scoping fix must not weaken IBAN detection in genuine free text --
    only exempt values stored directly under structural-identifier keys."""

    real_iban = "GB29NWBK60161331926819"
    clean, report = sanitize_for_persistence(
        {
            "description": f"Wire the refund to {real_iban} by Friday.",
            "content": f"Account on file: {real_iban}",
        }
    )

    assert real_iban not in clean["description"]
    assert real_iban not in clean["content"]
    assert "[REDACTED_IBAN]" in clean["description"]
    assert "[REDACTED_IBAN]" in clean["content"]
    assert "iban" in report.detected_types


def test_relationship_source_and_target_ids_survive_unmodified() -> None:
    """`get_graph_relationships`-shaped payloads must not have their edge
    endpoints mangled by the free-text pattern pass."""

    relationship = {
        "source": "action_decision:dc0836cf29134aa2a38231c319e6497e",
        "target": "action_decision:da90004e126c486abf4579b4646be2d0",
        "type": "DERIVED_FROM",
    }

    clean, report = sanitize_for_persistence(relationship)

    assert clean["source"] == "action_decision:dc0836cf29134aa2a38231c319e6497e"
    assert clean["target"] == "action_decision:da90004e126c486abf4579b4646be2d0"
    assert clean["source"] != clean["target"]
    assert "iban" not in report.detected_types
