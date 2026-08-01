from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest

from agent_utilities.protocols.source_connectors.tool_schema import (
    ToolSchemaContractError,
    canonical_input_schema,
    compatibility_fingerprint,
    validate_live_tool_contract,
)


def _tool(name: str = "records", *, action_type: str = "string") -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        inputSchema={
            "title": "Presentation-only title",
            "type": "object",
            "properties": {
                "action": {"type": action_type, "description": "routing action"},
                "params_json": {"type": "string"},
            },
            "required": ["params_json", "action"],
        },
    )


def test_live_contract_validates_name_arguments_and_compatibility_fingerprint():
    tool = _tool()
    expected = compatibility_fingerprint(
        "records", canonical_input_schema(tool, include_presentation=False)
    )

    contract = validate_live_tool_contract(
        SimpleNamespace(tools=[tool]),
        tool_name="records",
        expected_schema_sha256=expected,
        required_argument_types={"action": "string", "params_json": "string"},
    )

    assert contract.compatibility_sha256 == expected
    assert len(contract.schema_sha256) == 64


def test_v1_object_with_only_inputschema_still_reads_via_fallback():
    """D-FE-4: an MCP SDK v1-shaped object (only the old attribute) must keep
    working — ``_field`` falls back to ``inputSchema`` when ``input_schema``
    is absent."""
    tool = _tool()
    assert not hasattr(tool, "input_schema")
    schema = canonical_input_schema(tool, include_presentation=False)
    assert schema["properties"]["action"]["type"] == "string"


def test_v2_object_prefers_input_schema_and_never_touches_the_deprecated_alias():
    """D-FE-4: a v2-shaped object exposing both ``input_schema`` (real field)
    and ``inputSchema`` (deprecated compatibility property) must be read via
    ``input_schema`` only — touching the deprecated property at all is what
    emits ``FastMCPDeprecationWarning`` on every governed schema check."""

    class _DeprecatedAlias:
        def __get__(self, obj: object, objtype: type | None = None) -> object:
            warnings.warn(
                "Accessing 'Tool.inputSchema' is deprecated; MCP SDK v2 "
                "renamed this field to 'input_schema'",
                DeprecationWarning,
                stacklevel=2,
            )
            return obj._real_input_schema  # type: ignore[attr-defined]

    class _V2Tool:
        inputSchema = _DeprecatedAlias()

        def __init__(self, schema: dict) -> None:
            self.name = "records"
            self.input_schema = schema
            self._real_input_schema = schema

    tool = _V2Tool(_tool().inputSchema)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        schema = canonical_input_schema(tool, include_presentation=False)
    assert not caught, f"unexpected warnings: {[str(w.message) for w in caught]}"
    assert schema["properties"]["action"]["type"] == "string"


def test_presentation_drift_does_not_invalidate_signed_compatibility_pin():
    first = _tool()
    second = _tool()
    second.inputSchema["title"] = "Updated documentation"
    expected = compatibility_fingerprint(
        "records", canonical_input_schema(first, include_presentation=False)
    )

    contract = validate_live_tool_contract(
        [second], tool_name="records", expected_schema_sha256=expected
    )

    assert contract.compatibility_sha256 == expected


def test_presentation_drift_changes_exact_but_not_compatibility_fingerprint():
    first = _tool()
    second = _tool()
    second.inputSchema["title"] = "Updated documentation"
    second.inputSchema["properties"]["action"]["description"] = "updated"

    first_contract = validate_live_tool_contract(
        [first], tool_name="records", required_argument_types={"action": "string"}
    )
    second_contract = validate_live_tool_contract(
        [second], tool_name="records", required_argument_types={"action": "string"}
    )

    assert first_contract.schema_sha256 != second_contract.schema_sha256
    assert first_contract.compatibility_sha256 == second_contract.compatibility_sha256


@pytest.mark.parametrize(
    ("tools", "tool_name", "digest", "required"),
    [
        ([_tool("renamed")], "records", "", {"action": "string"}),
        ([_tool()], "records", "0" * 64, {"action": "string"}),
        ([_tool(action_type="integer")], "records", "", {"action": "string"}),
        ([_tool(), _tool()], "records", "", {"action": "string"}),
    ],
)
def test_live_contract_fails_closed_on_tool_or_schema_drift(
    tools, tool_name, digest, required
):
    with pytest.raises(ToolSchemaContractError):
        validate_live_tool_contract(
            tools,
            tool_name=tool_name,
            expected_schema_sha256=digest,
            required_argument_types=required,
        )
