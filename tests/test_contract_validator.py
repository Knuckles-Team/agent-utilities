"""Tool contracts: pre-conditions, post-condition schema and verifier."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from agent_utilities.harness.contract_validator import ContractValidator, ToolContract


class _Out(BaseModel):
    count: int


def _boom(_):
    raise RuntimeError("verifier crashed")


@pytest.fixture
def validator():
    v = ContractValidator.instance()
    v._contracts.clear()
    yield v
    v._contracts.clear()


def test_unregistered_node_passes(validator):
    assert validator.validate_pre("none", {}) and validator.validate_post("none", {})
    assert ContractValidator() is validator


@pytest.mark.parametrize(
    ("contract", "pre_ctx", "pre_ok", "out", "post_ok"),
    [
        (
            ToolContract(node_id="n", pre_condition=lambda c: c.get("ready", False)),
            {"ready": True},
            True,
            {},
            True,
        ),
        (
            ToolContract(node_id="n", pre_condition=lambda c: c["ready"]),
            {},
            False,
            {},
            True,
        ),
        (
            ToolContract(node_id="n", post_condition_schema=_Out),
            {},
            True,
            {"count": "x"},
            False,
        ),
        (
            ToolContract(
                node_id="n",
                post_condition_schema=_Out,
                post_condition_verifier=lambda o: o["count"] > 0,
            ),
            {},
            True,
            {"count": 2},
            True,
        ),
        (ToolContract(node_id="n", post_condition_verifier=_boom), {}, True, {}, False),
    ],
)
def test_contract_checks(validator, contract, pre_ctx, pre_ok, out, post_ok):
    validator.register_contract(contract)
    assert validator.get_contract("n") is contract
    assert validator.validate_pre("n", pre_ctx) is pre_ok
    assert validator.validate_post("n", out) is post_ok
