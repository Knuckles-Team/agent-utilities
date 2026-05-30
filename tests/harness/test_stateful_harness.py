"""Tests for Stateful Harness and Sensory Verification (AHE-3.7).

CONCEPT: AHE-3.7 Distributed Agent State Manager - Concurrency Branching & Merging
"""

from pydantic import BaseModel, Field

from agent_utilities.harness.contract_validator import ContractValidator, ToolContract
from agent_utilities.harness.distributed_state_manager import BranchMergeStateLocker


class MockPostSchema(BaseModel):
    status: str = Field(..., description="Status must be success")
    value: int = Field(..., description="Value must be positive")


def test_concurrency_branching_and_ff_merge():
    """Verify that state branching works and fast-forward merge succeeds when base is unchanged."""
    locker = BranchMergeStateLocker(use_redis=False)
    base_key = "test_base"

    # Set up initial base state
    locker.update_state(base_key, {"main_key": "main_value"}, expected_version=0)

    # Fork state
    fork = locker.fork_state(base_key, "branch_alpha")
    assert fork["base_version"] == 1
    assert fork["data"]["main_key"] == "main_value"

    # Update branch state
    updated = locker.update_branch_state(
        base_key,
        "branch_alpha",
        {"main_key": "main_value", "branch_key": "alpha_value"},
    )
    assert updated is True

    # Retrieve branch state
    branch_state = locker.get_branch_state(base_key, "branch_alpha")
    assert branch_state is not None
    assert isinstance(branch_state, dict)
    assert branch_state["data"]["branch_key"] == "alpha_value"

    # Merge branch state (should fast-forward since base has version 1)
    merge_ok = locker.merge_state(base_key, "branch_alpha")
    assert merge_ok is True

    # Verify merged main state
    base_state = locker.get_state(base_key)
    assert base_state is not None
    assert isinstance(base_state, dict)
    assert base_state["version"] == 2
    assert base_state["data"]["branch_key"] == "alpha_value"

    # Verify branch is cleaned up
    assert locker.get_branch_state(base_key, "branch_alpha") is None


def test_concurrency_merge_conflict_resolution():
    """Verify three-way merge logic and custom conflict resolution."""
    locker = BranchMergeStateLocker(use_redis=False)
    base_key = "test_base_conflict"

    # Set up initial base state
    locker.update_state(base_key, {"shared_dict": {"k1": "v1"}}, expected_version=0)

    # Fork state
    locker.fork_state(base_key, "branch_beta")

    # Modifying base concurrently (version becomes 2)
    locker.update_state(
        base_key,
        {"shared_dict": {"k1": "v1", "concurrent_key": "concurrent_val"}},
        expected_version=1,
    )

    # Updating branch state with a different modification
    locker.update_branch_state(
        base_key,
        "branch_beta",
        {"shared_dict": {"k1": "v1", "branch_key": "branch_val"}},
    )

    # Default three-way merge: should recursively merge the shared_dict!
    merge_ok = locker.merge_state(base_key, "branch_beta")
    assert merge_ok is True

    # Verify recursive merge output
    base_state = locker.get_state(base_key)
    assert base_state is not None
    assert isinstance(base_state, dict)
    assert base_state["data"]["shared_dict"]["concurrent_key"] == "concurrent_val"
    assert base_state["data"]["shared_dict"]["branch_key"] == "branch_val"


def test_custom_conflict_resolver():
    """Verify custom resolver callback is called during conflict merge."""
    locker = BranchMergeStateLocker(use_redis=False)
    base_key = "test_custom_resolver"

    locker.update_state(base_key, {"counter": 10}, expected_version=0)
    locker.fork_state(base_key, "branch_gamma")

    # Increment counter concurrently in base
    locker.update_state(base_key, {"counter": 15}, expected_version=1)

    # Set counter concurrently in branch
    locker.update_branch_state(base_key, "branch_gamma", {"counter": 20})

    # Custom resolver sums the values or takes max
    def custom_resolver(base_data, branch_data):
        return {
            "counter": max(base_data.get("counter", 0), branch_data.get("counter", 0))
        }

    merge_ok = locker.merge_state(base_key, "branch_gamma", resolver=custom_resolver)
    assert merge_ok is True

    # Verify merged counter is 20
    base_state = locker.get_state(base_key)
    assert base_state is not None
    assert isinstance(base_state, dict)
    assert base_state["data"]["counter"] == 20


def test_declarative_contract_validation():
    """Verify declarative pre-conditions and Pydantic post-conditions."""
    validator = ContractValidator.instance()

    # Pre-condition: check if input data contains valid API keys or tags
    def pre_validator(ctx_dict):
        return ctx_dict.get("api_ready", False) is True

    contract = ToolContract(
        id="test_mcp_node",
        pre_condition=pre_validator,
        post_condition_schema=MockPostSchema,
    )

    validator.register_contract(contract)

    # Validate Pre-condition
    assert validator.validate_pre("test_mcp_node", {"api_ready": True}) is True
    assert validator.validate_pre("test_mcp_node", {"api_ready": False}) is False

    # Validate Post-condition (Success Case)
    assert (
        validator.validate_post("test_mcp_node", {"status": "success", "value": 100})
        is True
    )

    # Validate Post-condition (Failure Case - Missing Value)
    assert validator.validate_post("test_mcp_node", {"status": "success"}) is False

    # Validate Post-condition (Failure Case - Invalid Type)
    assert (
        validator.validate_post(
            "test_mcp_node", {"status": "success", "value": "not-an-int"}
        )
        is False
    )
