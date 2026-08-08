"""``graph_config`` — governed AgentConfig admin (CONCEPT:AU-OS.config.two-surfaces-by-default).

The three properties that make this tool safe to ship, each pinned against the
concrete way it would otherwise be unsafe:

* **Model-derived, not hand-maintained** — ``describe`` reports a field's real
  docstring/type/default/alias straight from ``AgentConfig``, so a field added
  tomorrow is discoverable today.
* **Secrets redacted by reference, never by value** — a ``vault://``/``env://``
  reference is shown (that IS the answer an operator wants); an inline secret is
  never echoed by ``get``/``describe``/``diff`` and can never be written by
  ``set``. A real Keycloak client_secret leak in this workspace is why.
* **Refusal by default on ``set``** — unknown key, model-rejected value, or a
  policy denial each stop the write BEFORE anything is persisted.
"""

from __future__ import annotations

import pytest

from agent_utilities.core import config_admin


@pytest.fixture(autouse=True)
def _clear_docstring_cache():
    config_admin._attribute_docstrings.cache_clear()
    yield
    config_admin._attribute_docstrings.cache_clear()


# --------------------------------------------------------------------------- #
# describe — derived from the pydantic model
# --------------------------------------------------------------------------- #


def test_describe_reports_docstring_type_default_current_and_alias():
    field = config_admin.describe("MCP_ALWAYS_LOAD")["field"]

    assert field["key"] == "MCP_ALWAYS_LOAD"
    assert field["field"] == "mcp_always_load"
    assert field["type"] == "list"
    assert "tunnel-manager-mcp" in field["default"]
    # The description is the field's OWN attribute docstring, not a second
    # hand-maintained table that could silently disagree with the model.
    assert "find_tools" in (field["description"] or "")
    assert field["restart_required"] is False


def test_describe_is_derived_so_a_new_field_needs_no_hand_registration():
    """Every AgentConfig field is describable. A hand-maintained allowlist would
    fail this the moment anyone added a field."""
    from agent_utilities.core.config import AgentConfig

    inventory = config_admin.describe()
    assert inventory["count"] == len(AgentConfig.model_fields)
    assert len(config_admin.field_index()) == len(AgentConfig.model_fields)


def test_describe_accepts_either_the_alias_or_the_field_name():
    by_alias = config_admin.describe("MCP_ALWAYS_LOAD_TOOLS")["field"]
    by_name = config_admin.describe("mcp_always_load_tools")["field"]
    assert by_alias == by_name


def test_describe_can_be_narrowed_without_reading_source():
    result = config_admin.describe(contains="always_load")
    keys = {f["key"] for f in result["fields"]}
    assert keys == {"MCP_ALWAYS_LOAD", "MCP_ALWAYS_LOAD_TOOLS"}


def test_unknown_key_is_refused_not_guessed():
    with pytest.raises(config_admin.ConfigAdminError) as exc:
        config_admin.describe("TOTALLY_MADE_UP_KEY")
    assert exc.value.code == "unknown_key"


# --------------------------------------------------------------------------- #
# Redaction — by reference, never by value
# --------------------------------------------------------------------------- #


def test_a_runtime_reference_is_shown_verbatim():
    """The reference is exactly what an operator needs ("which vault path?") and
    discloses nothing."""
    value, redacted = config_admin.redact(
        "OIDC_CLIENT_SECRET_REF", "vault://kv/graphos#client_secret"
    )
    assert value == "vault://kv/graphos#client_secret"
    assert redacted is False


def test_an_inline_secret_is_never_echoed():
    value, redacted = config_admin.redact(
        "OIDC_CLIENT_SECRET_REF", "aVeryRealKeycloakClientSecret"
    )
    assert value == config_admin.REDACTED
    assert redacted is True
    # Not even a prefix, suffix or length hint survives.
    assert "aVeryReal" not in value
    assert len(value) == 3


def test_a_structured_sensitive_value_is_redacted_wholesale():
    """There is no safe partial view of a structure whose key says it holds
    credentials — a per-element filter would leak the ones it didn't recognise."""
    value, redacted = config_admin.redact("MESSAGING_DISCORD_TOKEN", ["tok-a", "tok-b"])
    assert value == config_admin.REDACTED
    assert redacted is True


def test_get_redacts_a_sensitive_literal(monkeypatch):
    monkeypatch.setattr(
        config_admin, "_effective", lambda name: "aVeryRealKeycloakClientSecret"
    )
    result = config_admin.get("OIDC_CLIENT_SECRET_REF")

    assert result["value"] == config_admin.REDACTED
    assert result["redacted"] is True
    assert result["secret"] is True


def test_describe_redacts_a_sensitive_literal(monkeypatch):
    monkeypatch.setattr(config_admin, "_effective", lambda name: "literal-secret")
    field = config_admin.describe("OIDC_CLIENT_SECRET_REF")["field"]

    assert field["current"] == config_admin.REDACTED
    assert field["redacted"] is True


def test_diff_reports_the_changed_key_without_disclosing_either_value(monkeypatch):
    """A deployment whose only difference is a credential must still be
    diffable — the KEY is the finding, the value is not."""
    monkeypatch.setattr(
        config_admin,
        "_effective",
        lambda name: (
            "literal-secret"
            if name == "oidc_client_secret_ref"
            else config_admin._default(name)
        ),
    )
    result = config_admin.diff()
    entry = next(e for e in result["changed"] if e["key"] == "OIDC_CLIENT_SECRET_REF")

    assert entry["effective"] == config_admin.REDACTED
    assert entry["redacted"] is True


def test_diff_shows_only_fields_that_differ_from_their_default(monkeypatch):
    monkeypatch.setattr(
        config_admin,
        "_effective",
        lambda name: (
            ["only-this-one-mcp"]
            if name == "mcp_always_load"
            else config_admin._default(name)
        ),
    )
    result = config_admin.diff()

    assert [e["key"] for e in result["changed"]] == ["MCP_ALWAYS_LOAD"]
    assert result["changed"][0]["effective"] == ["only-this-one-mcp"]
    assert "container-manager-mcp" in result["changed"][0]["default"]


# --------------------------------------------------------------------------- #
# set — governed, never a blind write
# --------------------------------------------------------------------------- #


def test_set_refuses_an_unknown_key_before_writing(monkeypatch):
    written = []
    monkeypatch.setattr(
        "agent_utilities.core.config.save_config_item",
        lambda k, v: written.append((k, v)),
    )
    with pytest.raises(config_admin.ConfigAdminError) as exc:
        config_admin.set_value("NOT_A_REAL_SETTING", "x")

    assert exc.value.code == "unknown_key"
    assert written == []


def test_set_refuses_a_value_the_model_rejects_before_writing(monkeypatch):
    written = []
    monkeypatch.setattr(
        "agent_utilities.core.config.save_config_item",
        lambda k, v: written.append((k, v)),
    )
    # mcp_dynamic_discovery_timeout is a bounded float; a string is not one.
    with pytest.raises(config_admin.ConfigAdminError) as exc:
        config_admin.set_value("MCP_DYNAMIC_DISCOVERY_TIMEOUT", "not-a-number")

    assert exc.value.code == "validation_failed"
    assert written == []


def test_set_refuses_an_inline_secret(monkeypatch):
    written = []
    monkeypatch.setattr(
        "agent_utilities.core.config.save_config_item",
        lambda k, v: written.append((k, v)),
    )
    with pytest.raises(config_admin.ConfigAdminError) as exc:
        config_admin.set_value("OIDC_CLIENT_SECRET_REF", "aVeryRealClientSecret")

    assert exc.value.code == "inline_secret_refused"
    assert written == []


def test_set_denies_when_the_policy_gate_denies(monkeypatch):
    written = []
    monkeypatch.setattr(
        "agent_utilities.core.config.save_config_item",
        lambda k, v: written.append((k, v)),
    )
    monkeypatch.setattr(
        config_admin,
        "_gate",
        lambda key, reason: (False, {"decision": "denied", "reason": "needs approval"}),
    )
    result = config_admin.set_value("MCP_ALWAYS_LOAD", '["a-mcp"]')

    assert result["applied"] is False
    assert result["error"] == "policy_denied"
    assert written == []


def test_set_denies_when_the_policy_gate_itself_is_unavailable(monkeypatch):
    """An unavailable gate must never read as permission."""
    import agent_utilities.orchestration.action_policy as policy

    monkeypatch.setattr(
        policy,
        "get_action_policy",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("engine down")),
    )
    allowed, info = config_admin._gate("MCP_ALWAYS_LOAD", "why")

    assert allowed is False
    assert info["decision"] == "denied"


def test_set_writes_through_the_standard_precedence_and_records_provenance(monkeypatch):
    written = []
    monkeypatch.setattr(
        "agent_utilities.core.config.save_config_item",
        lambda k, v: written.append((k, v)),
    )
    monkeypatch.setattr(
        config_admin, "_gate", lambda key, reason: (True, {"decision": "allow"})
    )
    provenance = []
    monkeypatch.setattr(
        config_admin,
        "_record_provenance",
        lambda key, **kw: provenance.append((key, kw)) or "configchange:1",
    )
    result = config_admin.set_value(
        "MCP_ALWAYS_LOAD", '["tunnel-manager-mcp"]', reason="pin one server"
    )

    assert result["applied"] is True
    # Validated by the model, so what is persisted is the parsed list, not a string.
    assert written == [("MCP_ALWAYS_LOAD", ["tunnel-manager-mcp"])]
    assert result["provenance_id"] == "configchange:1"
    assert provenance[0][0] == "MCP_ALWAYS_LOAD"
    assert provenance[0][1]["reason"] == "pin one server"


def test_set_can_change_the_always_load_declaration(monkeypatch):
    """The operator's actual ask: AgentConfig — including always-load — must be
    modifiable through an MCP tool."""
    written = {}
    monkeypatch.setattr(
        "agent_utilities.core.config.save_config_item",
        lambda k, v: written.__setitem__(k, v),
    )
    monkeypatch.setattr(
        config_admin, "_gate", lambda key, reason: (True, {"decision": "allow"})
    )
    monkeypatch.setattr(config_admin, "_record_provenance", lambda key, **kw: None)

    config_admin.set_value("MCP_ALWAYS_LOAD_TOOLS", '["github-mcp:github_issues"]')

    assert written["MCP_ALWAYS_LOAD_TOOLS"] == ["github-mcp:github_issues"]


def test_a_failed_provenance_write_does_not_undo_an_applied_change(monkeypatch):
    """Provenance is best-effort by design: a downed backend must not roll back
    a change the operator was authorised to make and which is already durable."""
    written = []
    monkeypatch.setattr(
        "agent_utilities.core.config.save_config_item",
        lambda k, v: written.append((k, v)),
    )
    monkeypatch.setattr(
        config_admin, "_gate", lambda key, reason: (True, {"decision": "allow"})
    )
    result = config_admin.set_value("MCP_ALWAYS_LOAD", '["a-mcp"]')

    assert result["applied"] is True
    assert written == [("MCP_ALWAYS_LOAD", ["a-mcp"])]


# --------------------------------------------------------------------------- #
# reload — explicit about what it cannot apply
# --------------------------------------------------------------------------- #


def test_reload_names_the_fields_that_still_need_a_restart(monkeypatch):
    monkeypatch.setattr("agent_utilities.core.config.load_config", lambda: None)
    result = config_admin.reload()

    assert result["reloaded"] is True
    assert "STATE_DB_URI" in result["restart_required_fields"]
    assert result["live_fields"] > 0


# --------------------------------------------------------------------------- #
# Tool surface
# --------------------------------------------------------------------------- #


def test_dispatch_refuses_an_unknown_action():
    with pytest.raises(config_admin.ConfigAdminError) as exc:
        config_admin.dispatch("delete_everything")
    assert exc.value.code == "unknown_action"


def test_graph_config_is_registered_on_the_graphos_surface():
    from agent_utilities.mcp._graphos_action_manifest import GRAPHOS_ACTIONS

    actions = {e["action"] for e in GRAPHOS_ACTIONS if e["tool"] == "graph_config"}
    assert actions == {"get", "set", "describe", "reload", "diff"}


def test_only_set_is_treated_as_a_mutation():
    from agent_utilities.mcp.tool_specs import READ_ONLY_ACTIONS

    assert READ_ONLY_ACTIONS["graph_config"] == frozenset(
        {"describe", "get", "diff", "reload"}
    )


def test_config_set_is_policy_gated_at_approval_tier():
    from agent_utilities.orchestration.action_policy import DEFAULT_POLICY

    rule = next(r for r in DEFAULT_POLICY["rules"] if r.get("kind") == "config.set")
    assert rule["tier"] == "approval_required"
