#!/usr/bin/python
from __future__ import annotations

"""Governed runtime capability authoring (CONCEPT:AU-AHE.harness.canonical-gap-lifecycle × harness runtime authoring).

``pydantic_ai_harness.capability_creation.CapabilityCreation`` lets an agent write a new
``AbstractCapability`` to disk, have the harness statically validate it, and — per the
harness's own documented orchestrator-loop pattern — inject it into the agent's capability
set on the run's fixed activation boundary: capabilities resolve ONCE at
``agent.run(...)`` start, so an authored capability is live on the **next** run, never the
current turn (there is no setter on a live run's capability chain).

That activation boundary is correct and load-bearing; what is missing is GOVERNANCE across
it. The harness's own reference integration (``creation.store.load_active()`` feeding
straight into ``agent.run(..., capabilities=extra)``) activates anything that passed
*static* validation — no human/Claude review, no distinction from a hand-written change
that would otherwise go through code review. This module closes that gap by routing every
authored capability through the SAME ``Gap → SDD → promote`` lifecycle any other change to
this codebase enters (``research/gaps.py`` + ``research/spec_proposals.py``, Wave-6):

    author_capability (harness, static validation)
        -> HELD immediately (never left "active" unreviewed)
        -> submit_gap(source="agent_authored_capability")           [Gap: open]
        -> persist_spec_proposal(..., gap_id=...)                    [SpecProposal: pending_review]
        -> review_spec(..., "approve" | "reject")                    [human / Claude review — SAME gate]
        -> load_governed_active_capabilities() re-activates ONLY the [status flips: active on approval]
           capabilities whose spec cleared review

An authored capability is therefore live on the orchestrator's next GOVERNED run after
approval — one notch stricter than the harness's own "next run after authoring," and never
a bypass of the review a human-authored change would get. Rejecting the spec (or simply
never approving it) leaves the capability held forever; nothing here auto-promotes.

This module does not replace ``CapabilityCreation`` — it is not itself the tool surface an
agent calls (that stays the harness's ``author_capability``/``list_authored_capabilities``/
``disable_authored_capability`` tools, unmodified). It sits between that tool surface's
on-disk store and the orchestrator's per-run activation call, exactly where the harness
README says the integration contract lives ("the orchestrator drives the loop, so it owns
the one-line contract: thread the store's active capabilities into each run").
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The ``:Gap`` discovery-track name for an agent-authored runtime capability —
#: folds into the SAME canonical ``:Gap`` every other discovery track uses
#: (production-failure / research / skill-coverage / code-audit).
AUTHORED_CAPABILITY_GAP_SOURCE = "agent_authored_capability"


def _spec_title(name: str) -> str:
    return f"Activate authored capability: {name}"


def submit_authored_capability_for_review(
    engine: Any,
    *,
    directory: Path | str,
    name: str,
    code: str,
    reason: str = "",
) -> dict[str, Any]:
    """Write + statically validate ``code`` as capability ``name``, then hold it and
    route it into the Gap->SDD->promote review lifecycle instead of the harness's
    own immediate "active" default.

    Returns a status dict:

    * ``{"written": False, "error": ...}`` — an invalid capability ``name`` (before
      anything was written).
    * ``{"written": True, "valid": False, "status": "validation_failed", ...}`` — the
      harness's own static validation rejected the code (not exactly one
      no-arg-constructible ``AbstractCapability`` subclass). Nothing enters the
      review lifecycle for code that does not even pass that bar.
    * ``{"written": True, "valid": True, "status": "pending_review", "gap_id": ...,
      "spec_id": ...}`` — the normal path: written, held, and a reviewable
      ``:Gap``/``:SpecProposal`` pair now exists.
    * ``status in ("gap_submit_failed", "spec_submit_failed")`` — a KG persistence
      failure; the capability is still written+held on disk (inspectable / retryable),
      just not yet entered into the review lifecycle.
    """
    from pydantic_ai_harness.capability_creation._store import CapabilityStore

    directory = Path(directory)
    store = CapabilityStore(directory=directory)
    try:
        record = store.write(name, code)
    except ValueError as exc:
        return {"written": False, "error": str(exc)}

    # Close the loophole: the harness's own `store.write()` marks a syntactically/
    # structurally valid capability 'active' immediately. Any caller feeding
    # `store.load_active()` straight into the next `agent.run(...)` — the harness's
    # own documented reference pattern — would inject it unreviewed. Hold it right
    # away so the ONLY path back to 'active' is `load_governed_active_capabilities`
    # below, gated on spec approval.
    store.disable(name)

    valid = bool(record.class_name) and record.last_error is None
    result: dict[str, Any] = {
        "written": True,
        "valid": valid,
        "class_name": record.class_name or None,
        "last_error": record.last_error,
        "status": "validation_failed" if not valid else "held_pending_review",
    }
    if not valid:
        # Nothing for a reviewer to approve yet — code that fails the harness's own
        # static bar never opens a Gap.
        return result

    from agent_utilities.knowledge_graph.enrichment.distill import SpecDraft
    from agent_utilities.knowledge_graph.research.gaps import submit_gap
    from agent_utilities.knowledge_graph.research.spec_proposals import (
        persist_spec_proposal,
    )

    module_path = directory / record.module_file
    gap = submit_gap(
        engine,
        source=AUTHORED_CAPABILITY_GAP_SOURCE,
        signature=name,
        statement=(
            f"Agent authored a runtime capability {name!r} ({record.class_name}) "
            f"and requests activation. Reason: {reason or '(none given)'}"
        ),
        domain="runtime_capability_authoring",
        severity=0.5,
    )
    if gap is None:
        result["status"] = "gap_submit_failed"
        return result
    result["gap_id"] = gap["id"]

    spec = SpecDraft(
        title=_spec_title(name),
        target_codebase=str(directory),
        problem=(
            f"Agent-authored capability {name!r} ({record.class_name}) passed the "
            f"harness's static validation but is HELD, not active — every "
            f"agent-authored capability enters the SAME Gap->SDD->promote review a "
            f"human-authored change would (CONCEPT:AU-AHE.harness.canonical-gap-lifecycle)."
        ),
        approach=(
            f"Review the authored source at {module_path} and either approve "
            f"(the next governed run activates it) or reject (it stays held forever)."
        ),
        value=reason or "agent-requested runtime capability",
        value_score=0.5,
        target_file=str(module_path),
    )
    spec_id = persist_spec_proposal(
        engine, spec, gap_id=gap["id"], target_file=str(module_path)
    )
    if spec_id is None:
        result["status"] = "spec_submit_failed"
        return result
    result["spec_id"] = spec_id
    result["status"] = "pending_review"
    return result


def load_governed_active_capabilities(engine: Any, directory: Path | str) -> list[Any]:
    """The governed replacement for ``CapabilityStore.load_active()``.

    An authored capability activates ONLY once its originating ``:SpecProposal`` has
    cleared review (``approved`` or ``published``) — never merely because it passed
    static validation. On every call this reconciles the harness's own on-disk
    'active'/'disabled' manifest status against the CURRENT spec status (so a
    same-session approval takes effect on the very next call, and a later rejection
    or revert holds a previously-approved capability back out), then delegates to
    the harness's own ``load_active()`` to actually import+construct the cleared set.

    This is the one-line orchestrator integration point the harness README asks for
    ("the orchestrator drives the loop, so it owns the one-line contract: thread the
    store's active capabilities into each run") — a caller replaces
    ``creation.store.load_active()`` with
    ``load_governed_active_capabilities(engine, directory)`` and gets the same
    return shape (``list[AbstractCapability]``) with governance now in the path.
    """
    from pydantic_ai_harness.capability_creation._store import CapabilityStore

    from agent_utilities.knowledge_graph.research.spec_proposals import (
        get_spec,
        spec_id_for,
    )

    directory = Path(directory)
    store = CapabilityStore(directory=directory)

    for record in store.list_all():
        if not record.class_name:
            continue  # never passed static validation — nothing to reconcile
        spec = get_spec(engine, spec_id_for(_spec_title(record.name)))
        cleared = bool(spec) and spec.get("status") in ("approved", "published")
        if cleared and record.status != "active":
            # Re-validate + re-mark active from the ALREADY-WRITTEN source (never
            # re-executes agent-supplied code through any other path) now that
            # review cleared it.
            module_path = directory / record.module_file
            try:
                code = module_path.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning(
                    "governed capability %r cleared review but its source file "
                    "is unreadable (%s) — staying held",
                    record.name,
                    exc,
                )
                continue
            store.write(record.name, code)
        elif not cleared and record.status == "active":
            # Defensive: a spec that regressed to rejected/reverted after a prior
            # approval (or any direct write bypassing this module) is held back out.
            store.disable(record.name)

    return store.load_active()


__all__ = [
    "AUTHORED_CAPABILITY_GAP_SOURCE",
    "load_governed_active_capabilities",
    "submit_authored_capability_for_review",
]
