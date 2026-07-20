from __future__ import annotations

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
