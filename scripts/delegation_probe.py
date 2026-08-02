#!/usr/bin/env python3
"""Exercise the graph-os delegation CORE directly, in-process — no MCP, no pod restart.

Every entrypoint (graph-os MCP `graph_orchestrate`, agent-webui/REST, Telegram/Mattermost
via `messaging/router.py`) converges on ONE function:

    agent_utilities.orchestration.manager.Orchestrator.execute_agent()   # manager.py:424

So the fastest way to validate delegation is to call that directly and read the real
traceback, instead of round-tripping through MCP and reading an opaque envelope. Fixing
the core here fixes it for every surface at once.

Run it INSIDE the graph-os pod (live engine + config already present):

    kubectl -n platform exec -i <pod> -c graph-os -- python3 - \
        --skill servicenow-incident-management --server servicenow-mcp \
        --tool servicenow_get_incidents --profile \
        < scripts/delegation_probe.py

Stages run in dependency order and STOP at the first failure, because a later stage's
error is meaningless once an earlier one is broken:

    1 config      AgentConfig builds at all
    2 identity    the verified actor + ambient GraphSession are minted
    3 engine      the KG engine is reachable under that authority
    4 grounding   required evidence compilation is within budget and accepted
    5 model       the local LLM is reachable and answers
    6 skill       the named skill actually resolves from the graph
    7 toolset     the durable server catalog admits the exact tool (no transport yet)
    8 delegate    execute_agent() end to end
    9 provenance  the run's RunTrace/:ToolCall are readable BACK out of the graph

Exit code is the number of the first failing stage (0 = all passed), so it is usable
as a gate in a loop.

HONESTY CONTRACT
----------------
The engine is fail-closed on TWO nested security boundaries, and both are FEATURES:

  * ``IdentityRequiredError``  — a verified ``ActorContext`` must be ambient.
  * ``SessionRequiredError``   — a verified ambient ``GraphSession`` must be ambient.

This probe does **not** hand-construct either one. It calls
``agent_utilities.mcp.kg_server._mint_process_session(transport)`` — literally the same
function the served graph-os process calls at boot (kg_server.py:4218). For a network
transport that performs a real OAuth2 client-credentials grant against the configured
IdP and validates the resulting token through the same JWKS decoder as an inbound HTTP
request. Tenant, scopes, audience, policy revision and placement route are therefore all
derived from a genuinely verified credential, exactly as in production.

Consequence: if identity/session binding is broken in production, **this probe fails at
stage 2 too**. It cannot mask that class of failure. The `--identity-mode synthetic`
escape hatch exists only for offline debugging and prints a loud warning, because it
*would* mask an IdP-side failure.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import io
import json
import os
import pstats
import time
import traceback
from typing import Any

STAGES = [
    "config",
    "identity",
    "engine",
    "grounding",
    "model",
    "skill",
    "toolset",
    "delegate",
    "provenance",
]

# Populated by the identity stage and reused by everything downstream.
_STATE: dict[str, Any] = {}


def _emit(stage: str, ok: bool, detail: str = "", elapsed: float | None = None) -> None:
    mark = "PASS" if ok else "FAIL"
    t = f" [{elapsed:6.2f}s]" if elapsed is not None else ""
    print(f"  {mark:4s} {stage:9s}{t} {detail}"[:600], flush=True)


def _chain(exc: BaseException) -> str:
    """Full causal chain — the thing the MCP envelope throws away."""
    out: list[str] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        out.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
    return "\n      caused by ".join(out)


# ---------------------------------------------------------------------------
# Stage 1 — config
# ---------------------------------------------------------------------------
async def _stage_config() -> Any:
    from agent_utilities.core.config import config

    # The shared singleton is what every other layer reads; building a private
    # AgentConfig() here would validate a different object than delegation uses.
    return config


# ---------------------------------------------------------------------------
# Stage 2 — identity + ambient GraphSession (the real production mint)
# ---------------------------------------------------------------------------
def _mint_session(cfg: Any, mode: str, transport: str) -> Any:
    """Return a verified GraphSession using the same code path graph-os boots with."""
    if mode == "process":
        from agent_utilities.mcp.kg_server import _mint_process_session

        return _mint_process_session(transport)
    if mode == "local-process":
        from agent_utilities.security.request_identity import (
            mint_local_process_session,
        )

        return mint_local_process_session()
    if mode == "synthetic":
        print(
            "  WARN  identity  --identity-mode synthetic is bypassing IdP validation; "
            "an IdP-side production failure CANNOT be observed in this run.",
            flush=True,
        )
        from agent_utilities.security.request_identity import (
            actor_from_claims,
            mint_graph_session,
        )

        claims = {
            "sub": os.environ.get("PROBE_SUBJECT", "delegation-probe"),
            "tenant_id": os.environ.get("PROBE_TENANT", "") or "",
            "roles": ["kg:admin"],
            "scope": "kg:admin",
        }
        if not claims["tenant_id"]:
            raise RuntimeError(
                "synthetic identity mode requires PROBE_TENANT — the engine will not "
                "accept a session without a verified tenant claim"
            )
        return mint_graph_session(actor_from_claims(claims))
    raise ValueError(f"unknown identity mode {mode!r}")


async def _stage_identity(cfg: Any, mode: str, transport: str) -> str:
    from agent_utilities.knowledge_graph.core.session import set_session
    from agent_utilities.security.brain_context import set_actor

    session = await asyncio.to_thread(_mint_session, cfg, mode, transport)
    # engine_verified_context() is the authority check the served plane performs
    # per tool call; if it raises, the session is incomplete and everything
    # downstream would fail with a less informative error.
    claims = session.engine_verified_context()

    # Bind BOTH contexts for the rest of the process. contextvars set on the main
    # task propagate into asyncio.to_thread and child tasks, so every stage below
    # inherits this exact authority.
    set_actor(session.actor)
    set_session(session)
    _STATE["session"] = session

    return (
        f"mode={mode} principal={claims.get('principal')} tenant={session.tenant} "
        f"graph={session.graph!r} scopes={sorted(session.scopes)} "
        f"endpoint={session.endpoint} group={session.placement_group} "
        f"epoch={session.catalog_epoch} policy={session.policy_version}"
    )


# ---------------------------------------------------------------------------
# Stage 3 — engine
# ---------------------------------------------------------------------------
async def _stage_engine(cfg: Any) -> Any:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    # Prefer the ALREADY-RUNNING engine: constructing a second one in-process would
    # not be the thing delegation actually uses, and would mask a live misbinding.
    eng = IntelligenceGraphEngine.get_active()
    created = False
    if eng is None:
        eng = IntelligenceGraphEngine.get_or_create(defer_background_start=True)
        created = True
    # cheap authoritative read — proves the transport, not just construction
    rows = await asyncio.to_thread(
        eng.backend.execute, "MATCH (n) RETURN count(n) AS c LIMIT 1"
    )
    _STATE["engine"] = eng
    count = rows[0].get("c") if rows else "?"
    return f"{type(eng).__name__} reachable nodes={count} (created={created})"


# ---------------------------------------------------------------------------
# Stage 4 — grounding (the mandatory evidence compile, measured UNBOUNDED)
# ---------------------------------------------------------------------------
def _required_grounding_failure(
    samples: list[float], budget: float, quality_gate_failed: Any
) -> str | None:
    """Return the required-mode failure reason, if the measurement rejects a run."""
    failures: list[str] = []
    over = sum(1 for sample in samples if sample > budget)
    if over:
        failures.append(
            f"latency budget exceeded in {over}/{len(samples)} samples "
            f"(budget={budget:.1f}s)"
        )
    if quality_gate_failed:
        failures.append("retrieval_quality_gate_failed")
    return "; ".join(failures) or None


def _any_retrieval_quality_gate_failed(sample_failures: list[bool]) -> bool:
    """Fail required grounding when any completed evidence compile was rejected."""
    return any(sample_failures)


def _grounding_gate_failure(
    grounding: str, samples: list[float], budget: float, quality_gate_failed: Any
) -> str | None:
    """Apply the measurement result only when the caller selected required grounding."""
    if grounding != "required":
        return None
    return _required_grounding_failure(samples, budget, quality_gate_failed)


def _grounding_sample_plan(
    grounding: str, requested_samples: int | None
) -> tuple[int, str]:
    """Resolve functional preflight versus explicitly requested benchmarking."""
    if requested_samples is None:
        return (1, "preflight") if grounding == "required" else (0, "functional")
    if requested_samples < 0:
        raise ValueError("--grounding-samples must be zero or greater")
    if grounding == "required" and requested_samples == 0:
        raise ValueError(
            "--grounding=required cannot use --grounding-samples=0; "
            "omit the option for its one-sample preflight"
        )
    if requested_samples == 0:
        return 0, "functional"
    return requested_samples, "benchmark"


async def _stage_grounding(
    model_class: str,
    budget_s: float,
    grounding: str,
    sample_count: int,
    sample_mode: str,
) -> str:
    """Measure what the mandatory evidence compilation ACTUALLY costs.

    ``contextual_model._compiled_evidence_and_bundle_bounded`` wraps this in a
    hard ``_CONTEXT_COMPILE_TIMEOUT_S`` (10.0s) budget and, under the default
    ``grounding='required'``, RAISES when it is exceeded. That makes the compile
    cost the single most important number in delegation — and nothing has ever
    measured it, because the bounded wrapper destroys the evidence by design.

    Here it runs with a deliberately generous ceiling so we learn the true
    distribution instead of only "it exceeded 10s".
    """
    if sample_count <= 0:
        _grounding_sample_plan(grounding, sample_count)
    if sample_count == 0:
        _STATE["grounding_samples"] = []
        return (
            f"synthetic_compile=skipped mode={sample_mode} policy={grounding} "
            "=> proceeding to the real delegation"
        )

    from agent_utilities.core import contextual_model as cm
    from agent_utilities.core.model_factory import create_model

    model = create_model(role=model_class) if model_class else create_model()
    model_name = str(getattr(model, "model_name", "") or "unknown")
    budget = cm._CONTEXT_COMPILE_TIMEOUT_S

    from pydantic_ai.messages import ModelRequest, UserPromptPart

    messages = [
        ModelRequest(
            parts=[UserPromptPart(content="List recent ServiceNow incidents.")]
        )
    ]

    samples: list[float] = []
    sample_quality_failures: list[bool] = []
    for _ in range(sample_count):
        t0 = time.monotonic()
        try:
            compiled = await asyncio.wait_for(
                asyncio.to_thread(
                    cm._compiled_evidence_and_bundle, messages, model_name
                ),
                timeout=budget_s,
            )
            samples.append(time.monotonic() - t0)
            bundle = (
                compiled[1]
                if isinstance(compiled, tuple) and len(compiled) > 1
                else None
            )
            sample_quality_failures.append(
                bool(getattr(bundle, "retrieval_quality_gate_failed", False))
            )
        except TimeoutError:
            samples.append(float("inf"))
            break

    _STATE["grounding_samples"] = samples
    gate_failed = _any_retrieval_quality_gate_failed(sample_quality_failures)
    over = sum(1 for s in samples if s > budget)
    rendered = ", ".join(
        "timeout" if s == float("inf") else f"{s:.2f}s" for s in samples
    )
    required_failure = _grounding_gate_failure(grounding, samples, budget, gate_failed)
    if required_failure:
        raise RuntimeError("grounding='required' fails closed: " + required_failure)
    verdict = (
        f"OVER the {budget:.1f}s production budget in {over}/{len(samples)} samples "
        "— every default grounding='required' delegation FAILS CLOSED"
        if over
        else f"within the {budget:.1f}s production budget"
    )
    return (
        f"synthetic_compile={sample_mode} requested={sample_count} "
        f"completed={len(sample_quality_failures)} compile={rendered} "
        f"budget={budget:.1f}s quality_gate_failed={gate_failed} => {verdict}"
    )


# ---------------------------------------------------------------------------
# Stage 5 — model
# ---------------------------------------------------------------------------
async def _stage_model(cfg: Any, model_class: str, live: bool) -> str:
    from agent_utilities.core.model_factory import create_model

    model = create_model(role=model_class) if model_class else create_model()
    name = getattr(model, "model_name", None) or type(model).__name__
    if not live:
        return f"{name} (constructed; --live-model to actually call it)"

    from agent_utilities.core.contextual_model import create_context_agent

    agent = create_context_agent(model, default_capabilities=False)
    t0 = time.monotonic()
    res = await agent.run("Reply with the single word: ready")
    dt = time.monotonic() - t0
    out = str(getattr(res, "output", res))[:60].replace("\n", " ")
    return f"{name} answered in {dt:.2f}s: {out!r}"


# ---------------------------------------------------------------------------
# Stage 5 — skill
# ---------------------------------------------------------------------------
async def _stage_skill(skill: str) -> str:
    """Resolve the skill the way delegation will — fail here, not three layers down."""
    eng = _STATE["engine"]
    rows = await asyncio.to_thread(
        eng.backend.execute,
        "MATCH (n) WHERE n.name = $name "
        "RETURN n.node_type AS node_type, n.name AS name, n.id AS id LIMIT 5",
        {"name": skill},
    )
    if not rows:
        raise LookupError(
            f"skill {skill!r} does not resolve in the graph — delegation binds skills "
            f"fail-closed, so this is the real blocker, not the tool call"
        )
    _STATE["skill_rows"] = rows

    # RESOLVING IS NOT RUNNING. `agent_runner._resolve_agent_from_kg` only accepts a
    # CallableResource with resource_type='AGENT_SKILL'; a Skill / WorkflowDefinition
    # node with the same name resolves here but raises
    # "ingested skill '<name>' was not found or runnable" three layers down. Say so
    # HERE, where it is cheap to read, instead of after a multi-minute delegation.
    kinds = {str(r.get("node_type") or "") for r in rows}
    runnable = "CallableResource" in kinds
    if not runnable:
        raise LookupError(
            f"skill {skill!r} exists as {sorted(kinds)} but has NO CallableResource "
            f"node, so delegation cannot run it (only describe it). This is an "
            f"INGESTION gap, not a delegation defect: re-ingest it as "
            f"resource_type='AGENT_SKILL' with provider_ref set to its owning MCP "
            f"server."
        )
    return (
        f"runnable=True kinds={sorted(kinds)} "
        + json.dumps(rows[:2], default=str)[:240]
    )


# ---------------------------------------------------------------------------
# Stage 7 — toolset catalog
# ---------------------------------------------------------------------------
async def _stage_toolset(server: str, tool: str) -> str:
    """Prove the exact server/tool binding without owning an MCP transport.

    Delegation creates its own refresh-capable ``MCPToolset`` and owns its
    connection, schema notifications, cancellation, and teardown. Reusing a
    preflight session would split that ownership across two stages; opening one
    merely to list tools adds a throwaway handshake. The durable catalog is the
    same least-privilege contract ``run_agent`` validates before it builds that
    runtime toolset, so use it here and leave the one live session to delegation.
    """
    from agent_utilities.orchestration.agent_runner import (
        _catalog_toolset_binding,
        _fleet_server_url,
    )

    url = await asyncio.to_thread(_fleet_server_url, server)
    if not url:
        raise RuntimeError(
            f"catalog server={server!r} has no resolved MCP endpoint; delegation "
            "would not be able to construct its authenticated toolset"
        )
    server_meta = await asyncio.to_thread(
        _catalog_toolset_binding,
        _STATE.get("engine"),
        server,
        allowed_tools=[tool] if tool else None,
    )
    names = [
        str(item.get("name") or "").strip()
        for item in server_meta.get("tools") or []
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]
    if not names:
        raise RuntimeError(
            f"catalog server={server!r} url={url!r} declares ZERO tools — the "
            "probe cannot prove an exact binding without an ingested catalog"
        )
    bound = tool in names if tool else None
    if tool and not bound:
        raise RuntimeError(
            f"catalog server={server!r} url={url!r} declares {len(names)} tools "
            f"but NOT the requested {tool!r} — sample={names[:10]}"
        )
    return (
        f"catalog server={server!r} url={url!r} tools={len(names)} "
        f"{tool!r}_bound={bound} transport=deferred-to-delegate sample={names[:6]}"
    )


# ---------------------------------------------------------------------------
# Stage 8 — delegate
# ---------------------------------------------------------------------------
def _required_tool_summary_failure(
    a: argparse.Namespace, summary: dict[str, Any], output: object = ""
) -> str | None:
    """Fail closed when a tool-required governed run is not successful."""
    if not getattr(a, "require_tool", False):
        return None
    requested = str(getattr(a, "tool", "") or "")
    if not requested:
        return "--require-tool requires an explicit --tool"
    outcome = str(summary.get("outcome") or "missing").lower()
    if outcome == "ok":
        if str(output or "").strip():
            return None
        return f"required tool {requested!r} produced no returned response"
    failure = summary.get("failure")
    return (
        f"required tool {requested!r} did not complete successfully: "
        f"outcome={outcome!r} failure={failure!r}"
    )


async def _stage_delegate(a: argparse.Namespace) -> str:
    import uuid

    from agent_utilities.orchestration.manager import Orchestrator

    # `run_agent` returns a bare string unless the caller opts into the rich
    # wrapper. Pin an explicit run_id AND ask for the run summary so the run is
    # addressable: without both, a delegation is unverifiable after the fact —
    # you cannot read its RunTrace back out of the graph.
    run_id = a.run_id or f"probe-{uuid.uuid4().hex}"
    _STATE["run_id"] = run_id
    print(f"  ...  delegate  run_id={run_id}", flush=True)

    # The Orchestrator is engine-bound; hand it the SAME live engine the earlier
    # stages proved reachable rather than letting it resolve a second one.
    orch = Orchestrator(_STATE["engine"])
    t0 = time.monotonic()

    if a.entry == "execute_capability":
        # This is what the graph-os MCP `graph_orchestrate` tool calls: it also
        # runs KG capability resolution before dispatch, so it is the closest
        # in-process equivalent to the MCP surface.
        res = await orch.execute_capability(
            task=a.task,
            skill_name=a.skill,
            tool_server=a.server,
            allowed_tools=[a.tool] if a.tool else None,
            required_tools=[a.tool] if (a.tool and a.require_tool) else None,
            execution_mode=a.mode,
            budget_tokens=a.budget,
            max_steps=a.max_steps,
            grounding=a.grounding,
            model_class=a.model_class,
        )
    else:
        # `run_agent` enforces "skill_name must match the dispatched agent_name",
        # so a skill has to be named on BOTH keywords — passing skill_name alone
        # raises ValueError before anything runs.
        res = await orch.execute_agent(
            agent_name=a.skill,
            task=a.task,
            skill_name=a.skill,
            tool_server=a.server,
            allowed_tools=[a.tool] if a.tool else None,
            required_tools=[a.tool] if (a.tool and a.require_tool) else None,
            execution_mode=a.mode,
            budget_tokens=a.budget,
            max_steps=a.max_steps,
            grounding=a.grounding,
            model_class=a.model_class,
            run_id=run_id,
            include_run_summary=True,
        )
    dt = time.monotonic() - t0
    _STATE["result"] = res

    payload: dict[str, Any]
    if isinstance(res, str):
        try:
            parsed = json.loads(res)
            payload = parsed if isinstance(parsed, dict) else {"output": res}
        except (TypeError, ValueError):
            payload = {"output": res}
    elif isinstance(res, dict):
        payload = res
    else:
        payload = {"output": str(res)}

    summary = payload.get("run_summary") or {}
    # `execute_capability` mints its OWN run id (the `run_id=` kwarg is only on
    # `execute_agent`), so always prefer the id the runtime actually assigned —
    # otherwise the provenance read-back queries a handle that never existed and
    # reports a false negative.
    actual = payload.get("run_id") or summary.get("run_id") or ""
    if actual:
        _STATE["run_id"] = str(actual)
    # The durable trace id is a PSEUDONYMISED reference (`trace:pref_run_<sha256>`),
    # not the run id — deriving it from run_id is impossible by design, so the run
    # summary's `trace_ref` is the ONLY handle that can address this run's RunTrace.
    # A provenance check that ignores it reports a false negative.
    if summary.get("trace_ref"):
        _STATE["trace_ref"] = str(summary["trace_ref"])
    output = str(payload.get("output", ""))[:300].replace("\n", " ")
    print(
        "\n  RUN SUMMARY: " + json.dumps(summary, default=str)[:900] + "\n", flush=True
    )
    required_failure = _required_tool_summary_failure(a, summary, payload.get("output"))
    if required_failure:
        raise RuntimeError(required_failure)
    return (
        f"[{dt:.2f}s] run={payload.get('run_id') or run_id} "
        f"route={summary.get('route')} outcome={summary.get('outcome')} "
        f"stage={summary.get('stage_reached')} trace={summary.get('trace_ref')} "
        f"out={output!r}"
    )


# ---------------------------------------------------------------------------
# Stage 8 — provenance read-back (the actual gate: is the run VISIBLE?)
# ---------------------------------------------------------------------------
_PROVENANCE_QUERIES: tuple[tuple[str, str], ...] = (
    (
        "run_trace",
        "MATCH (t:RunTrace {id: $trace_id}) RETURN t.id AS id, t.agent_name AS agent, "
        "t.status AS status, t.duration_ms AS duration_ms, t.skill_ref AS skill_ref, "
        "t.server_ref AS server_ref, t.grounding_status AS grounding LIMIT 1",
    ),
    (
        "tool_calls",
        "MATCH (t:RunTrace {id: $trace_id})-[]->(c:ToolCall) "
        "RETURN c.id AS id, c.tool_name AS tool, c.status AS status LIMIT 20",
    ),
    (
        "tool_calls_by_run",
        "MATCH (c:ToolCall) WHERE c.run_id = $trace_id "
        "RETURN c.id AS id, c.tool_name AS tool, c.status AS status LIMIT 20",
    ),
    (
        "uses_skill",
        "MATCH (t:RunTrace {id: $trace_id})-[:USES_SKILL]->(s) "
        "RETURN s.name AS skill, s.node_type AS node_type LIMIT 5",
    ),
)


_SUCCESSFUL_TOOL_STATUSES = frozenset({"ok", "success", "succeeded", "completed"})


def _required_tool_provenance_failure(
    a: argparse.Namespace, rows: list[dict[str, Any]]
) -> str | None:
    """Require one successful, exact-name ToolCall for ``--require-tool``."""
    if not getattr(a, "require_tool", False):
        return None
    requested = str(getattr(a, "tool", "") or "")
    if not requested:
        return "--require-tool requires an explicit --tool"
    matching = [row for row in rows if str(row.get("tool") or "") == requested]
    if not matching:
        observed = sorted({str(row.get("tool") or "") for row in rows})
        return (
            f"required ToolCall {requested!r} is absent from provenance; "
            f"observed={observed!r}"
        )
    statuses = [str(row.get("status") or "missing").lower() for row in matching]
    if not any(status in _SUCCESSFUL_TOOL_STATUSES for status in statuses):
        return (
            f"required ToolCall {requested!r} has no successful provenance row; "
            f"statuses={statuses!r}"
        )
    return None


def _required_run_trace_failure(
    a: argparse.Namespace, rows: list[dict[str, Any]]
) -> str | None:
    """Require a successful terminal RunTrace for a tool-required probe."""
    if not getattr(a, "require_tool", False):
        return None
    if not rows:
        return "required tool execution has no RunTrace provenance row"
    statuses = [str(row.get("status") or "missing").lower() for row in rows]
    if not any(status in _SUCCESSFUL_TOOL_STATUSES for status in statuses):
        return (
            f"required tool execution has no completed RunTrace; statuses={statuses!r}"
        )
    return None


async def _stage_provenance(a: argparse.Namespace) -> str:
    """Read the delegation's own provenance back OUT of the graph.

    A run that produced text but wrote no readable RunTrace has not proven
    delegation — it has proven only that an LLM replied. This stage is therefore
    part of the gate, not a nicety.
    """
    eng = _STATE["engine"]
    run_id = _STATE.get("run_id") or a.run_id
    if not run_id:
        raise RuntimeError("no run_id captured — cannot read provenance back")

    # Prefer the run's OWN pseudonymised trace_ref; fall back to id-derived
    # spellings only when the run summary did not carry one.
    ref = _STATE.get("trace_ref") or ""
    trace_ids = [t for t in (ref, ref.removeprefix("trace:")) if t]
    trace_ids += [run_id, f"trace:{run_id}", f"trace-{run_id}"]
    found: dict[str, Any] = {}
    for label, cypher in _PROVENANCE_QUERIES:
        for tid in trace_ids:
            rows = await asyncio.to_thread(
                eng.backend.execute, cypher, {"trace_id": tid, "run_id": run_id}
            )
            if rows:
                found[label] = {"trace_id": tid, "rows": rows}
                break
    print(
        "\n  PROVENANCE READ-BACK: " + json.dumps(found, default=str)[:2000] + "\n",
        flush=True,
    )
    if "run_trace" not in found:
        raise LookupError(
            f"no :RunTrace readable for run_id={run_id!r} — the delegation produced "
            f"output but left no durable provenance, so the run is unverifiable"
        )
    run_trace_rows = list(found["run_trace"].get("rows", []))
    required_failure = _required_run_trace_failure(a, run_trace_rows)
    if required_failure:
        raise RuntimeError(required_failure)
    tool_rows = list(found.get("tool_calls", {}).get("rows", []))
    if not tool_rows:
        tool_rows = list(found.get("tool_calls_by_run", {}).get("rows", []))
    required_failure = _required_tool_provenance_failure(a, tool_rows)
    if required_failure:
        raise RuntimeError(required_failure)
    return f"RunTrace={found['run_trace']['rows'][0]} tool_calls={len(tool_rows)}"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
async def run(a: argparse.Namespace) -> int:
    print(
        f"\ndelegation probe — skill={a.skill!r} server={a.server!r} tool={a.tool!r} "
        f"mode={a.mode} identity={a.identity_mode}\n"
    )
    timings: list[tuple[str, float]] = []
    cfg = None
    for n, stage in enumerate(STAGES, 1):
        if a.stop_after and STAGES.index(a.stop_after) + 1 < n:
            break
        t0 = time.monotonic()
        try:
            if stage == "config":
                cfg = await _stage_config()
                d = (
                    f"backend={getattr(cfg, 'kg_backend', '?')} "
                    f"audience={getattr(cfg, 'auth_jwt_audience', '') or getattr(cfg, 'mcp_jwt_audience', '')!r} "
                    f"policy={getattr(cfg, 'kg_policy_version', '')!r}"
                )
            elif stage == "identity":
                d = await _stage_identity(cfg, a.identity_mode, a.transport)
            elif stage == "engine":
                eng = await _stage_engine(cfg)
                d = eng
            elif stage == "grounding":
                d = await _stage_grounding(
                    a.model_class,
                    a.grounding_budget,
                    a.grounding,
                    a.grounding_samples,
                    a.grounding_sample_mode,
                )
            elif stage == "model":
                d = await _stage_model(cfg, a.model_class, a.live_model)
            elif stage == "skill":
                d = await _stage_skill(a.skill) if a.skill else "(skipped: no --skill)"
            elif stage == "toolset":
                d = (
                    await _stage_toolset(a.server, a.tool)
                    if a.server
                    else "(skipped: no --server)"
                )
            elif stage == "delegate":
                d = await _stage_delegate(a)
            else:
                d = await _stage_provenance(a)
            dt = time.monotonic() - t0
            timings.append((stage, dt))
            _emit(stage, True, d, dt)
        except Exception as exc:  # noqa: BLE001 — the probe's whole job is to report the chain
            dt = time.monotonic() - t0
            timings.append((stage, dt))
            _emit(stage, False, "", dt)
            print(
                f"\n  ROOT CAUSE at stage {n} ({stage}):\n      {_chain(exc)}\n",
                flush=True,
            )
            if a.traceback:
                traceback.print_exc()
            _print_timings(timings)
            return n
    print("\n  all stages passed\n")
    _print_timings(timings)
    return 0


def _print_timings(timings: list[tuple[str, float]]) -> None:
    total = sum(t for _, t in timings) or 1e-9
    print("  === wall-clock by stage ===", flush=True)
    for stage, dt in timings:
        bar = "#" * int(40 * dt / total)
        print(f"    {stage:9s} {dt:7.2f}s {100 * dt / total:5.1f}%  {bar}", flush=True)
    print(f"    {'TOTAL':9s} {total:7.2f}s", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--skill", default="")
    p.add_argument("--server", default="")
    p.add_argument("--tool", default="")
    p.add_argument("--task", default="Retrieve exactly 2 records and summarise them.")
    p.add_argument("--mode", default="auto", choices=["auto", "pydantic_graph"])
    p.add_argument(
        "--grounding", default="required", choices=["required", "best_effort", "none"]
    )
    p.add_argument("--model-class", default="standard")
    p.add_argument("--budget", type=int, default=24000)
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument(
        "--require-tool",
        action="store_true",
        help="demand recorded ToolCall provenance",
    )
    p.add_argument("--stop-after", choices=STAGES, help="run only up to this stage")
    p.add_argument(
        "--profile", action="store_true", help="cProfile the run, print top cumulative"
    )
    p.add_argument(
        "--identity-mode",
        default="process",
        choices=["process", "local-process", "synthetic"],
        help=(
            "process = the exact mint graph-os boots with (real IdP round-trip); "
            "local-process = the packaged-local ephemeral JWT authority; "
            "synthetic = UNSAFE offline debugging only, masks IdP failures"
        ),
    )
    p.add_argument(
        "--transport",
        default="streamable-http",
        help="transport whose process-authority path to mint (stdio uses local authority)",
    )
    p.add_argument(
        "--live-model",
        action="store_true",
        help="actually call the LLM in the model stage instead of only constructing it",
    )
    p.add_argument(
        "--entry",
        default="execute_capability",
        choices=["execute_capability", "execute_agent"],
        help=(
            "execute_capability = what the MCP graph_orchestrate tool calls "
            "(KG capability resolution + dispatch); execute_agent = the bare core"
        ),
    )
    p.add_argument(
        "--grounding-budget",
        type=float,
        default=90.0,
        help="ceiling for the UNBOUNDED grounding measurement (not the production budget)",
    )
    p.add_argument(
        "--grounding-samples",
        type=int,
        default=None,
        help=(
            "synthetic compile count: required defaults to one preflight; "
            "best_effort/none default to zero; a positive value benchmarks"
        ),
    )
    p.add_argument(
        "--run-id",
        default="",
        help="pin the delegation's run id so its RunTrace is addressable afterwards",
    )
    p.add_argument("--traceback", action="store_true")
    a = p.parse_args()
    try:
        a.grounding_samples, a.grounding_sample_mode = _grounding_sample_plan(
            a.grounding, a.grounding_samples
        )
    except ValueError as exc:
        p.error(str(exc))

    if not a.profile:
        return asyncio.run(run(a))

    pr = cProfile.Profile()
    pr.enable()
    rc = asyncio.run(run(a))
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(30)
    print("\n=== profile (top 30 cumulative) ===\n" + s.getvalue()[:6000])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
