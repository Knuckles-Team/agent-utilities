"""Focused acceptance fixtures for NE-092's immutable workflow control core."""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from agent_utilities.control_plane.workflows import (
    ApprovedBinding,
    ArtifactRef,
    ExactBindingRegistry,
    InMemoryWorkflowRepository,
    StepContract,
    WorkflowBudget,
    WorkflowCatalog,
    WorkflowConflictError,
    WorkflowDomainError,
    WorkflowDefinition,
    WorkflowStep,
    WorkflowTemplate,
)


def _digest(letter: str) -> str:
    """A distinct, deterministic, CONTRACT-VALID digest per label.

    ``Digest`` requires ``^sha256:[0-9a-f]{64}$``. The previous ``letter * 64``
    form produced a non-hex string for any non-hex label -- i, j, o, p, q, r, s
    and t are all used here -- which the model correctly rejected. Deriving the
    body from the label keeps every call site and its distinctness while
    actually satisfying the contract.
    """
    return "sha256:" + hashlib.sha256(letter.encode("utf-8")).hexdigest()


def _budget(**overrides: int) -> WorkflowBudget:
    values = {
        "max_steps": 8,
        "max_depth": 8,
        "max_fanout": 4,
        "max_tool_calls": 4,
        "max_tokens": 10_000,
        "max_seconds": 120,
        "max_cost_micros": 10_000,
    }
    values.update(overrides)
    return WorkflowBudget(**values)


def _binding(kind: str, binding_id: str, letter: str) -> ApprovedBinding:
    return ApprovedBinding(
        kind=kind,
        binding_id=binding_id,
        version="1.0.0",
        digest=_digest(letter),
    )


def _step(step_id: str, binding: ApprovedBinding, depends_on: tuple[str, ...] = ()) -> WorkflowStep:
    artifact = ArtifactRef(ref=f"schema:{step_id}", digest=_digest("e"))
    return WorkflowStep(
        step_id=step_id,
        binding=binding,
        contract=StepContract(
            contract_id=f"contract:{step_id}",
            input_schema=artifact,
            output_schema=artifact,
        ),
        depends_on=depends_on,
    )


def _definition(
    *,
    version: str = "1.0.0",
    steps: tuple[WorkflowStep, ...] | None = None,
    budget: WorkflowBudget | None = None,
    policy_budget: WorkflowBudget | None = None,
) -> WorkflowDefinition:
    template = WorkflowTemplate(
        template_id="template:research",
        workflow_id="workflow:research",
        summary="bounded research workflow",
        owner_ref="agent:platform",
    )
    policy = _binding("policy", "policy:research", "p")
    agent = _binding("agent", "agent:research", "a")
    if steps is None:
        steps = (_step("fetch", agent), _step("summarize", agent, ("fetch",)))
    return WorkflowDefinition(
        workflow_id=template.workflow_id,
        template_id=template.template_id,
        template_digest=template.template_digest,
        version=version,
        policy_binding=policy,
        budget=budget or _budget(),
        policy_budget=policy_budget or _budget(),
        steps=steps,
    )


def _catalog(*definitions: WorkflowDefinition) -> WorkflowCatalog:
    bindings = {definition.policy_binding for definition in definitions}
    for definition in definitions:
        bindings.update(step.binding for step in definition.steps)
        bindings.update(
            step.policy_binding
            for step in definition.steps
            if step.policy_binding is not None
        )
    return WorkflowCatalog(
        InMemoryWorkflowRepository(),
        bindings=ExactBindingRegistry(bindings),
    )


def test_definition_digest_is_independent_of_step_and_dependency_order() -> None:
    first = _definition()
    second = _definition(
        steps=tuple(reversed(first.steps)),
    )

    assert first.definition_digest == second.definition_digest
    assert first.identity.workflow_id == "workflow:research"
    assert first.identity.template_id == "template:research"


def test_cycle_and_missing_dependency_fail_closed() -> None:
    agent = _binding("agent", "agent:cycle", "a")
    cyclic = (
        _step("a", agent, ("b",)),
        _step("b", agent, ("a",)),
    )
    with pytest.raises(ValidationError, match="workflow_cycle"):
        _definition(steps=cyclic)

    missing = (_step("a", agent, ("does-not-exist",)),)
    with pytest.raises(ValidationError, match="workflow_dependency_missing"):
        _definition(steps=missing)


def test_budget_escalation_and_graph_bounds_are_rejected() -> None:
    with pytest.raises(ValidationError, match="budget_escalation"):
        _definition(budget=_budget(max_tokens=20), policy_budget=_budget(max_tokens=10))

    agent = _binding("agent", "agent:depth", "a")
    chain = (
        _step("a", agent),
        _step("b", agent, ("a",)),
        _step("c", agent, ("b",)),
    )
    with pytest.raises(ValidationError, match="workflow_depth_budget_exceeded"):
        _definition(steps=chain, budget=_budget(max_depth=2), policy_budget=_budget(max_depth=2))


def test_alias_and_inline_payload_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ApprovedBinding(
            kind="agent",
            binding_id="agent:research",
            version="latest",
            digest=_digest("a"),
        )

    with pytest.raises(ValidationError):
        WorkflowStep.model_validate(
            {
                "step_id": "step:unsafe",
                "binding": {
                    "kind": "agent",
                    "binding_id": "agent:research",
                    "version": "1.0.0",
                    "digest": _digest("a"),
                },
                "contract": {
                    "contract_id": "contract:unsafe",
                    "input_schema": {"ref": "schema:in", "digest": _digest("i")},
                    "output_schema": {"ref": "schema:out", "digest": _digest("o")},
                },
                "body": "must never cross the definition boundary",
            }
        )


def test_catalog_requires_exact_bindings_and_cas_release() -> None:
    definition = _definition()
    catalog = _catalog(definition)

    pointer = catalog.release(definition, channel="stable")
    resolved = catalog.resolve(definition.workflow_id, channel="stable")
    assert resolved.definition.definition_digest == definition.definition_digest
    assert resolved.pointer_generation == pointer.generation
    assert resolved.resolution_digest.startswith("sha256:")
    assert "body" not in catalog.summary(definition.workflow_id, channel="stable").model_dump()

    with pytest.raises(WorkflowConflictError, match="release_pointer_cas_conflict"):
        catalog.release(definition, channel="stable")

    unapproved = _definition(
        version="2.0.0",
        steps=(_step("fetch", _binding("agent", "agent:unapproved", "u")),),
    )
    with pytest.raises(WorkflowDomainError, match="binding_unresolved_or_unapproved"):
        catalog.publish(unapproved)


def test_rollback_targets_an_existing_prior_immutable_version() -> None:
    first = _definition(version="1.0.0")
    second = _definition(version="2.0.0")
    catalog = _catalog(first, second)
    first_pointer = catalog.release(first, channel="stable")
    second_pointer = catalog.release(second, channel="stable", expected=first_pointer)

    rolled_back = catalog.rollback(
        second.workflow_id,
        channel="stable",
        prior_version="1.0.0",
        expected=second_pointer,
    )
    assert rolled_back.version == "1.0.0"
    assert rolled_back.generation == second_pointer.generation + 1

    with pytest.raises(WorkflowConflictError, match="rollback_target_not_prior"):
        catalog.rollback(
            second.workflow_id,
            channel="stable",
            prior_version="2.0.0",
            expected=rolled_back,
        )


def test_definition_catalog_has_no_workitem_claim_or_lease_authority() -> None:
    catalog = _catalog(_definition())
    assert not hasattr(catalog, "claim_work_item")
    assert not hasattr(catalog, "renew_work_item_lease")
    assert not hasattr(catalog, "commit_work_item_result")
