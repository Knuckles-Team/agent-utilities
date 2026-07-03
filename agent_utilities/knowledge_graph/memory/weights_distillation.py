"""CONCEPT:KG-2.316 — Memory→weights distillation EXPORT path (AU-side).

The *parametric-consolidation bridge* of the agent-native-memory stack. Where
:mod:`agent_utilities.knowledge_graph.memory.lifecycle` (KG-2.307) consolidates
episodic memory into semantic summaries (the engine EG-220/221 primitives) and
:mod:`agent_utilities.knowledge_graph.memory.distillation` (KG-2.309) distills
recurring clusters into *procedural* memory nodes, this module goes one hop
further: it turns that consolidated / procedural memory into a **training-ready
corpus** — a JSONL SFT (``{prompt, completion}``) or preference
(``{prompt, chosen, rejected}``) dataset plus a typed target *spec* — so the
memory can be folded into model **weights** as a LoRA/SFT fine-tune. This is the
"memory → weights" bridge that goes *beyond* retrieval-time context assembly
(EG-195): instead of re-reading the memory graph at inference, the distilled
knowledge is baked into an adapter.

Dependency discipline (see ``AGENTS.md``): **core stays torch-free.** This module
only *reads* memory and *emits a corpus + a job spec* — deterministic, no numpy /
torch / transformers / peft. The actual GPU fine-tune runs in
``agents/data-science-mcp`` (the ML/training pillar), reached over MCP. This is
the EXPORT half; the training half is the documented integration point below.

Pipeline (one ``export`` cycle):

  * **read** a BOUNDED working set of ``:Memory`` nodes — reuses the KG-2.307
    lifecycle's localized reader, never a global scan.
  * **select** the nodes in scope (:meth:`MemoryWeightsDistiller.select`) — by
    memory tier (``scopes``: default ``procedural`` + ``semantic``, i.e. the
    consolidated tiers), ACTIVE status, an optional time-window, a trust floor,
    and optional ``target_entities``. Deterministic id-order, capped at
    ``max_examples``.
  * **render** each node into one training example
    (:meth:`MemoryWeightsDistiller.to_example`) — an instruction/response pair
    (``{prompt, completion}``) or, when the node carries an explicit
    ``chosen``/``rejected`` preference, a preference triple. Matches the corpus
    keys ``data-science-mcp`` ``training_data.py`` already consumes.
  * **bundle** into a :class:`DistillationCorpus` (examples + the
    :class:`DistillationTargetSpec` + a content ``checksum`` for reproducibility).
  * **submit** (optional) the corpus + spec to ``data-science-mcp`` via
    :meth:`MemoryWeightsDistiller.submit`, returning a typed
    :class:`DistillationJob`. Default submit is durable + torch-free (writes the
    JSONL + a job manifest and, best-effort, registers a job node the fleet can
    pick up) **and now dispatches the train LIVE** (CONCEPT:KG-2.318): it runs the
    ``train_model`` workflow on ``data-science-mcp`` through the
    ``graph_orchestrate execute_workflow`` seam (:data:`DATA_SCIENCE_MCP_CONTRACT`),
    marking the job ``running`` with the remote run handle. The dispatch is bounded
    + robust: when the orchestration engine / data-science-mcp is unreachable it
    degrades to a durable ``enqueued`` job (materialized + a job node the fleet
    picks up) and never raises. The heavy LoRA/SFT train still executes **in
    data-science-mcp** (GPU-gated, GB10) — core only orchestrates it over MCP and
    polls the ``TrainingJob`` / checkpoint state back.

Design notes (mirror KG-2.307 / KG-2.309):

  * **Deterministic + off by default.** Export is pure and reproducible (same
    memory ⇒ byte-identical JSONL + checksum). ``submit`` never runs training.
  * **Reuses existing seams.** The bounded reader is the KG-2.307 lifecycle's; the
    corpus keys are ``data-science-mcp``'s; the hand-off is the established
    ``train_model`` workflow reached via ``graph_orchestrate``.
  * **Injectable submitter; clean degrade.** Tests inject a stub submitter; with
    no writable memory dir / no engine job surface the default submit degrades to
    a plain ``exported`` job (files written, nothing enqueued) and never raises.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .lifecycle import MemoryLifecycle, MemoryLifecycleConfig, _age_hours

logger = logging.getLogger(__name__)

# Retired / superseded tiers are never distilled into weights.
_INACTIVE_STATUSES = frozenset(
    {"ARCHIVED", "MERGED", "RETIRED", "DELETED", "SUPERSEDED"}
)

# The consolidated tiers a fine-tune corpus is distilled FROM, by default: the
# procedural artifacts (KG-2.309) and the semantic summaries (KG-2.307 / EG-220).
_DEFAULT_SCOPES: tuple[str, ...] = ("procedural", "semantic")

# Methods that consume a preference (chosen/rejected) corpus rather than SFT.
_PREF_METHODS = frozenset({"dpo", "orpo", "kto", "preference"})

# Default instruction synthesized for a memory node that carries only prose.
_DEFAULT_INSTRUCTION_TEMPLATE = (
    "Recall the consolidated {memory_type} knowledge about {topic}."
)


# The typed data-science-mcp hand-off CONTRACT (CONCEPT:KG-2.316). The export side
# produces a corpus in ``corpus_format[method]`` and hands it to the training side
# via ONE of the ``dispatch`` seams; the live LoRA train runs there (GPU-gated).
DATA_SCIENCE_MCP_CONTRACT: dict[str, Any] = {
    "server": "data-science-mcp",
    "corpus_format": {
        "sft": ["prompt", "completion"],
        "dpo": ["prompt", "chosen", "rejected"],
    },
    # Preferred: drive the whole DAG (curate → train → eval → register) as one call.
    "workflow": {
        "tool": "graph_orchestrate",
        "action": "execute_workflow",
        "name": "train_model",
    },
    # Or call the data-science-mcp tools directly (plan-by-default, execute=true):
    "mcp_tools": {
        "build_dataset": "build_training_dataset",
        "train": {"sft": "train_sft", "dpo": "train_dpo"},
        "merge_adapters": "merge_adapters_ties",
        "register_checkpoint": "register_checkpoint",
    },
    # The deploy seam: a trained adapter goes live via the model-registry role bind.
    "deploy": "model_registry.resolve_role",
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, "") or default


def _as_str_list(val: Any) -> list[str]:
    if isinstance(val, (list, tuple)):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str) and val.strip():
        return [p.strip() for p in val.split(",") if p.strip()]
    return []


# Hard wall-clock bound on the live data-science-mcp dispatch (CONCEPT:KG-2.318).
# A named constant, not a flag (Configuration discipline): a single correct upper
# bound so an unreachable/slow train hand-off fails fast into the durable
# ``enqueued`` degrade rather than blocking the export path.
_DISPATCH_TIMEOUT: float = 45.0


def _run_coro_sync(make_coro: Callable[[], Any], timeout: float) -> Any:
    """Run an async coroutine to completion from a synchronous caller (KG-2.318).

    ``submit`` is a sync call reached from inside the already-async ``graph_analyze``
    MCP tool, so we cannot ``await`` the orchestration hand-off directly. When no
    event loop is running we ``asyncio.run`` it; when one IS running (the live MCP
    path) we run it on a dedicated worker thread with its own loop. ``make_coro`` is
    a zero-arg factory so the coroutine is created inside the loop that will await
    it. Bounded by ``timeout`` so a stuck dispatch can never hang the export.
    """
    import asyncio

    try:
        asyncio.get_running_loop()
        loop_running = True
    except RuntimeError:
        loop_running = False

    if not loop_running:
        return asyncio.run(asyncio.wait_for(make_coro(), timeout))

    import concurrent.futures

    def _worker() -> Any:
        return asyncio.run(asyncio.wait_for(make_coro(), timeout))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_worker).result(timeout=timeout + 10)


def _dispatch_train_workflow(
    engine: Any, name: str, task: dict[str, Any], timeout: float = _DISPATCH_TIMEOUT
) -> dict[str, Any]:
    """Run the data-science-mcp ``train_model`` workflow via graph_orchestrate (KG-2.318).

    This is the concrete LIVE hand-off: it drives the same
    ``graph_orchestrate action=execute_workflow`` seam an operator would call —
    :class:`~agent_utilities.orchestration.manager.Orchestrator` ``execute_workflow`` —
    so the exported corpus + :class:`DistillationTargetSpec` (carried in ``task``)
    reach ``data-science-mcp`` for the GPU LoRA/SFT train. Returns the workflow
    result dict (carrying a ``run_id`` handle); raises on any failure so the caller
    can degrade to a durable ``enqueued`` job.
    """
    from agent_utilities.orchestration.manager import Orchestrator

    orchestrator = Orchestrator(engine)

    def _make() -> Any:
        return orchestrator.execute_workflow(name, task=json.dumps(task, default=str))

    result = _run_coro_sync(_make, timeout)
    return result if isinstance(result, dict) else {"result": result}


@dataclass
class DistillationTargetSpec:
    """The fine-tune TARGET of a memory→weights distillation (CONCEPT:KG-2.316).

    Describes *what* the exported corpus is meant to train: the base model, the
    adapter shape (LoRA rank/alpha), the training method, and *which* slice of
    memory was distilled (tiers/scopes + optional time-window + entity filter).
    This spec travels with the corpus to ``data-science-mcp`` so the train run is
    fully specified by the export.
    """

    base_model: str = ""
    method: str = "sft"  # "sft" | "dpo" (preference)
    adapter_type: str = "lora"
    adapter_rank: int = 16
    adapter_alpha: int = 32
    adapter_dropout: float = 0.05
    # Memory tiers the corpus is distilled FROM (the consolidated tiers by default).
    scopes: list[str] = field(default_factory=lambda: list(_DEFAULT_SCOPES))
    # Only distil memories written within this many days (None ⇒ no window).
    time_window_days: int | None = None
    # Optional target_entity/category scoping (empty ⇒ all in-scope memories).
    target_entities: list[str] = field(default_factory=list)
    # Drop memories below this trust score.
    min_trust: float = 0.0
    # Hard cap on exported examples (token/latency budget for the fine-tune).
    max_examples: int = 512
    instruction_template: str = _DEFAULT_INSTRUCTION_TEMPLATE

    def __post_init__(self) -> None:
        self.method = str(self.method or "sft").strip().lower()
        self.scopes = _as_str_list(self.scopes) or list(_DEFAULT_SCOPES)
        self.target_entities = _as_str_list(self.target_entities)

    @property
    def is_preference(self) -> bool:
        return self.method in _PREF_METHODS

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "method": self.method,
            "adapter_type": self.adapter_type,
            "adapter_rank": self.adapter_rank,
            "adapter_alpha": self.adapter_alpha,
            "adapter_dropout": self.adapter_dropout,
            "scopes": list(self.scopes),
            "time_window_days": self.time_window_days,
            "target_entities": list(self.target_entities),
            "min_trust": self.min_trust,
            "max_examples": self.max_examples,
        }

    @classmethod
    def from_params(cls, params: dict[str, Any] | None) -> DistillationTargetSpec:
        """Build a spec from a loose (MCP/REST) params dict, with env defaults."""
        p = dict(params or {})
        env_model = _env_str("AGENT_UTILITIES_DISTILL_BASE_MODEL")
        window = p.get("time_window_days")
        return cls(
            base_model=str(p.get("base_model") or env_model or ""),
            method=str(p.get("method") or "sft"),
            adapter_type=str(p.get("adapter_type") or "lora"),
            adapter_rank=int(p.get("adapter_rank") or 16),
            adapter_alpha=int(p.get("adapter_alpha") or 32),
            adapter_dropout=float(p.get("adapter_dropout") or 0.05),
            scopes=_as_str_list(p.get("scopes")) or list(_DEFAULT_SCOPES),
            time_window_days=int(window) if window not in (None, "") else None,
            target_entities=_as_str_list(p.get("target_entities")),
            min_trust=float(p.get("min_trust") or 0.0),
            max_examples=int(
                p.get("max_examples")
                or _env_int("AGENT_UTILITIES_DISTILL_MAX_EXAMPLES", 512)
            ),
        )


@dataclass
class DistillationCorpus:
    """A training-ready corpus distilled from memory (CONCEPT:KG-2.316).

    ``examples`` are plain dicts keyed exactly as ``data-science-mcp`` consumes:
    ``{prompt, completion}`` for SFT or ``{prompt, chosen, rejected}`` for a
    preference method. Rendering is deterministic so a re-export of unchanged
    memory yields a byte-identical JSONL and the same :attr:`checksum`.
    """

    spec: DistillationTargetSpec
    examples: list[dict[str, Any]]
    source_ids: list[str]
    stats: dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        """Render the corpus as canonical JSONL (sorted keys, deterministic)."""
        return "\n".join(
            json.dumps(ex, ensure_ascii=False, sort_keys=True) for ex in self.examples
        )

    @property
    def checksum(self) -> str:
        """Stable content hash of the exported JSONL (reproducibility handle)."""
        return hashlib.sha256(self.to_jsonl().encode("utf-8")).hexdigest()

    def summary(self, sample: int = 2) -> dict[str, Any]:
        """A compact, JSON-safe view (counts + spec + a small sample) for surfaces."""
        return {
            "example_count": len(self.examples),
            "source_count": len(self.source_ids),
            "format": "preference" if self.spec.is_preference else "sft",
            "checksum": self.checksum,
            "spec": self.spec.to_dict(),
            "stats": dict(self.stats),
            "sample": self.examples[: max(0, int(sample))],
        }


@dataclass
class DistillationJob:
    """A typed hand-off of a distilled corpus to data-science-mcp (CONCEPT:KG-2.316).

    ``status`` lifecycle: ``exported`` (corpus materialized, nothing enqueued) →
    ``enqueued`` (a durable job node the fleet can pick up) → (in data-science-mcp)
    ``running`` → ``succeeded`` / ``failed``. The train transition is the
    integration point; core only takes it to ``exported`` / ``enqueued``.
    """

    job_id: str
    status: str
    spec: dict[str, Any]
    corpus_ref: str
    checksum: str
    example_count: int
    handoff: dict[str, Any]
    detail: str = ""
    submitted_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "spec": self.spec,
            "corpus_ref": self.corpus_ref,
            "checksum": self.checksum,
            "example_count": self.example_count,
            "handoff": self.handoff,
            "detail": self.detail,
            "submitted_at": self.submitted_at,
        }


class MemoryWeightsDistiller:
    """Export consolidated/procedural memory as a LoRA/SFT corpus (CONCEPT:KG-2.316).

    Instantiate with the same ``engine`` the scheduler hands maintenance handlers
    (it exposes ``.backend`` for the bounded reader and, optionally, a job surface
    for :meth:`submit`). Two injectable seams:

    * ``submitter`` — ``(corpus, spec) -> DistillationJob``, overrides the WHOLE
      submit step (defaults to :meth:`_default_submit`, durable + torch-free).
    * ``dispatcher`` — ``(handoff, corpus, spec) -> dict``, the LIVE data-science-mcp
      hand-off within the default submit (CONCEPT:KG-2.318). Defaults to
      :meth:`_default_dispatch`, which runs the ``train_model`` workflow over
      ``graph_orchestrate execute_workflow``; tests inject a mock MCP client here.
    """

    def __init__(
        self,
        engine: Any,
        spec: DistillationTargetSpec | None = None,
        submitter: Callable[
            [DistillationCorpus, DistillationTargetSpec], DistillationJob
        ]
        | None = None,
        dispatcher: Callable[
            [dict[str, Any], DistillationCorpus, DistillationTargetSpec],
            dict[str, Any],
        ]
        | None = None,
        max_working_set: int = 2000,
    ) -> None:
        self.engine = engine
        self.spec = spec or DistillationTargetSpec()
        self._submitter = submitter
        self._dispatcher = dispatcher
        # Reuse the KG-2.307 lifecycle purely for its BOUNDED working-set reader.
        self._lifecycle = MemoryLifecycle(
            engine,
            config=MemoryLifecycleConfig(
                enabled=False, max_working_set=max_working_set
            ),
        )

    # ── Read (localized, bounded — reuses KG-2.307) ───────────────────────────
    def _read_working_set(self, now: datetime) -> list[dict[str, Any]]:
        return self._lifecycle._read_working_set(now)

    # ── Selection ─────────────────────────────────────────────────────────────
    def select(
        self,
        nodes: list[dict[str, Any]] | None = None,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Return the in-scope memory nodes to distil, deterministically ordered.

        A node is in scope (CONCEPT:KG-2.316) when its ``memory_type`` is one of
        ``spec.scopes``, its status is ACTIVE (not retired/merged/…), it carries
        content, it clears the trust floor and the optional time-window, and — if
        ``target_entities`` is set — its ``target_entity``/``category`` matches.
        Ordered by id and capped at ``spec.max_examples`` so the export is stable.
        """
        now = now or datetime.now(UTC)
        if nodes is None:
            nodes = self._read_working_set(now)

        scopes = {s.lower() for s in self.spec.scopes}
        entities = {e.lower() for e in self.spec.target_entities}
        window = self.spec.time_window_days
        picked: list[dict[str, Any]] = []
        for n in nodes:
            if str(n.get("memory_type", "")).lower() not in scopes:
                continue
            if str(n.get("status", "ACTIVE")).upper() in _INACTIVE_STATUSES:
                continue
            if not self._content(n):
                continue
            try:
                if float(n.get("trust_score", 1.0)) < self.spec.min_trust:
                    continue
            except (TypeError, ValueError):
                pass
            if window is not None and _age_hours(n, now) > window * 24.0:
                continue
            if entities:
                ent = str(n.get("target_entity") or n.get("category") or "").lower()
                if ent not in entities:
                    continue
            picked.append(n)

        picked.sort(key=lambda g: str(g.get("id", "")))
        return picked[: max(1, int(self.spec.max_examples))]

    # ── Rendering ──────────────────────────────────────────────────────────────
    @staticmethod
    def _content(node: dict[str, Any]) -> str:
        for key in ("content", "summary_text", "description", "name"):
            val = node.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""

    def _topic(self, node: dict[str, Any]) -> str:
        return str(
            node.get("target_entity")
            or node.get("category")
            or node.get("name")
            or "prior experience"
        )

    def _instruction(self, node: dict[str, Any]) -> str:
        # Prefer an explicit instruction the memory already carries.
        for key in ("prompt", "instruction", "question"):
            val = node.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return self.spec.instruction_template.format(
            memory_type=str(node.get("memory_type", "semantic")),
            topic=self._topic(node),
        )

    def to_example(self, node: dict[str, Any]) -> dict[str, Any] | None:
        """Render one memory node into a training example, or ``None`` if unusable.

        Preference method + an explicit ``chosen``/``rejected`` on the node ⇒ a
        ``{prompt, chosen, rejected}`` triple; otherwise an SFT
        ``{prompt, completion}`` pair (instruction synthesized from the node's
        scope when the memory carries no explicit prompt).
        """
        chosen = node.get("chosen")
        rejected = node.get("rejected")
        if (
            self.spec.is_preference
            and isinstance(chosen, str)
            and isinstance(rejected, str)
        ):
            if chosen.strip() and rejected.strip():
                return {
                    "prompt": self._instruction(node),
                    "chosen": chosen.strip(),
                    "rejected": rejected.strip(),
                    "source_id": str(node.get("id", "")),
                }
            return None

        # SFT: prefer an explicit response field, else the node content.
        completion = ""
        for key in ("response", "completion", "answer"):
            val = node.get(key)
            if isinstance(val, str) and val.strip():
                completion = val.strip()
                break
        if not completion:
            completion = self._content(node)
        if not completion:
            return None
        return {
            "prompt": self._instruction(node),
            "completion": completion,
            "source_id": str(node.get("id", "")),
        }

    # ── Export ─────────────────────────────────────────────────────────────────
    def export(
        self,
        nodes: list[dict[str, Any]] | None = None,
        now: datetime | None = None,
    ) -> DistillationCorpus:
        """Produce a :class:`DistillationCorpus` from the in-scope memory."""
        now = now or datetime.now(UTC)
        selected = self.select(nodes, now)
        examples: list[dict[str, Any]] = []
        source_ids: list[str] = []
        by_scope: dict[str, int] = {}
        for n in selected:
            ex = self.to_example(n)
            if ex is None:
                continue
            examples.append(ex)
            sid = str(n.get("id", ""))
            if sid:
                source_ids.append(sid)
            scope = str(n.get("memory_type", "")).lower()
            by_scope[scope] = by_scope.get(scope, 0) + 1
        stats = {
            "selected": len(selected),
            "examples": len(examples),
            "by_scope": by_scope,
            "format": "preference" if self.spec.is_preference else "sft",
        }
        return DistillationCorpus(
            spec=self.spec, examples=examples, source_ids=source_ids, stats=stats
        )

    # ── Hand-off to data-science-mcp ──────────────────────────────────────────
    def _build_handoff(self, corpus_ref: str) -> dict[str, Any]:
        """The concrete data-science-mcp call the training side runs (KG-2.316).

        A ready-to-dispatch payload built from :data:`DATA_SCIENCE_MCP_CONTRACT`:
        the ``train_model`` workflow invocation (preferred) carrying the corpus
        ref + spec, plus the equivalent direct-tool sequence. The actual run is the
        integration point (GPU-gated, executes in data-science-mcp).
        """
        task = {
            "objective": "memory_to_weights_lora",
            "corpus_ref": corpus_ref,
            "corpus_format": DATA_SCIENCE_MCP_CONTRACT["corpus_format"][
                "dpo" if self.spec.is_preference else "sft"
            ],
            "spec": self.spec.to_dict(),
        }
        train_tool = DATA_SCIENCE_MCP_CONTRACT["mcp_tools"]["train"][
            "dpo" if self.spec.is_preference else "sft"
        ]
        return {
            "contract": "KG-2.316",
            "server": DATA_SCIENCE_MCP_CONTRACT["server"],
            # Preferred single-call hand-off (drives the full train DAG).
            "workflow": {
                **DATA_SCIENCE_MCP_CONTRACT["workflow"],
                "task": task,
            },
            # Equivalent direct-tool sequence (plan-by-default; execute=true to run).
            "tools": [
                {
                    "tool": DATA_SCIENCE_MCP_CONTRACT["mcp_tools"]["build_dataset"],
                    "args": {"corpus_ref": corpus_ref},
                },
                {
                    "tool": train_tool,
                    "args": {
                        "base_model": self.spec.base_model,
                        "adapter_rank": self.spec.adapter_rank,
                        "execute": True,
                    },
                },
                {
                    "tool": DATA_SCIENCE_MCP_CONTRACT["mcp_tools"][
                        "register_checkpoint"
                    ],
                    "args": {},
                },
            ],
            "integration_point": (
                "live LoRA train runs in data-science-mcp (GPU-gated, GB10); "
                "dispatch the workflow above to execute it"
            ),
        }

    # CONCEPT:KG-2.318 — LIVE data-science-mcp train dispatch + TrainingJob status poll
    def _default_dispatch(
        self, handoff: dict[str, Any], corpus: DistillationCorpus
    ) -> dict[str, Any]:
        """LIVE data-science-mcp hand-off — run ``train_model`` (CONCEPT:KG-2.318).

        Drives the contract's ``workflow`` entry (``graph_orchestrate
        execute_workflow name=train_model``) so the exported corpus + spec reach
        ``data-science-mcp`` for the GPU LoRA/SFT train. Bounded + robust: only
        attempted when the engine is an orchestration-capable
        :class:`IntelligenceGraphEngine`; any unreachable/failed dispatch returns
        ``dispatched=False`` so the caller keeps a durable ``enqueued`` job the
        fleet can still pick up. Never raises.
        """
        workflow = (handoff or {}).get("workflow") or {}
        name = str(workflow.get("name") or "train_model")
        task = dict(workflow.get("task") or {})
        try:
            from agent_utilities.knowledge_graph.core.engine import (
                IntelligenceGraphEngine,
            )

            if not isinstance(self.engine, IntelligenceGraphEngine):
                return {
                    "dispatched": False,
                    "status": "enqueued",
                    "detail": (
                        "no orchestration engine; job enqueued for "
                        "data-science-mcp pickup"
                    ),
                }
            result = _dispatch_train_workflow(self.engine, name, task)
            run_id = str(result.get("run_id") or result.get("session_id") or "")
            return {
                "dispatched": True,
                "status": "running",
                "run_id": run_id,
                "via": "execute_workflow",
                "workflow": name,
                "detail": f"dispatched {name} to data-science-mcp (run {run_id})",
            }
        except Exception as e:  # noqa: BLE001 — degrade to durable enqueue
            logger.warning("[KG-2.318] live train dispatch failed: %s", e)
            return {
                "dispatched": False,
                "status": "enqueued",
                "detail": (
                    f"data-science-mcp dispatch unreachable ({e}); "
                    "job enqueued for pickup"
                ),
            }

    def _run_dispatch(
        self, handoff: dict[str, Any], corpus: DistillationCorpus
    ) -> dict[str, Any]:
        """Invoke the dispatcher seam (injected or default); never raises."""
        try:
            if self._dispatcher is not None:
                return self._dispatcher(handoff, corpus, self.spec)
            return self._default_dispatch(handoff, corpus)
        except Exception as e:  # noqa: BLE001 — a bad dispatcher can't abort submit
            logger.warning("[KG-2.318] dispatcher raised: %s", e)
            return {"dispatched": False, "status": "enqueued", "detail": str(e)}

    def _default_submit(self, corpus: DistillationCorpus) -> DistillationJob:
        """Durable, torch-free hand-off: materialize + LIVE-dispatch + register.

        Writes the JSONL + a job manifest under the memory dir, builds the
        data-science-mcp hand-off payload, **dispatches the train live** through the
        dispatcher seam (CONCEPT:KG-2.318 — the ``train_model`` workflow over
        ``graph_orchestrate``), and registers a durable ``TrainingJob`` node the
        fleet + poll can read back. A live dispatch marks the job ``running`` with
        the remote run handle; an unreachable data-science-mcp degrades to a durable
        ``enqueued`` job; with no writable dir or engine job surface it degrades to
        an in-memory ``exported`` job. Never raises.
        """
        job_id = f"distill-lora-{uuid.uuid4().hex[:10]}"
        submitted_at = datetime.now(UTC).isoformat()
        corpus_ref = f"inline:{job_id}"
        detail = ""
        status = "exported"

        # 1. Materialize the corpus + manifest (best-effort).
        try:
            from .memory_engine import memory_dir

            out_dir = memory_dir() / "distillation"
            out_dir.mkdir(parents=True, exist_ok=True)
            corpus_path = out_dir / f"{job_id}.jsonl"
            corpus_path.write_text(corpus.to_jsonl(), encoding="utf-8")
            corpus_ref = str(corpus_path)
        except Exception as e:  # noqa: BLE001 — degrade, never abort the export
            logger.warning("[KG-2.316] corpus materialization failed: %s", e)

        handoff = self._build_handoff(corpus_ref)

        # 2. LIVE hand-off: dispatch the train to data-science-mcp (KG-2.318).
        dispatch = self._run_dispatch(handoff, corpus)
        handoff["dispatch"] = dispatch
        run_id = str(dispatch.get("run_id") or "")
        # The status the durable job should carry: ``running`` when the train was
        # dispatched live, else ``enqueued`` (awaiting a data-science-mcp worker).
        intended_status = "running" if dispatch.get("dispatched") else "enqueued"

        # 3. Best-effort: register a durable job node the fleet + poll can read back.
        add_node = getattr(self.engine, "add_node", None)
        if callable(add_node):
            try:
                add_node(
                    job_id,
                    "TrainingJob",
                    properties={
                        "concept": "KG-2.318",
                        "status": intended_status,
                        "kind": "memory_to_weights_lora",
                        "base_model": self.spec.base_model,
                        "method": self.spec.method,
                        "corpus_ref": corpus_ref,
                        "checksum": corpus.checksum,
                        "example_count": len(corpus.examples),
                        "server": DATA_SCIENCE_MCP_CONTRACT["server"],
                        "dispatched": bool(dispatch.get("dispatched")),
                        "run_id": run_id,
                        "submitted_at": submitted_at,
                    },
                )
                status = intended_status
                detail = str(
                    dispatch.get("detail")
                    or "registered TrainingJob node; awaiting data-science-mcp"
                )
            except Exception as e:  # noqa: BLE001 — enqueue is best-effort
                logger.warning("[KG-2.318] job-node registration failed: %s", e)
                detail = f"materialized only (enqueue failed: {e})"
        else:
            detail = "materialized only (engine has no job surface)"

        # 4. Persist the final manifest (status + dispatch outcome) for poll.
        try:
            from .memory_engine import memory_dir

            manifest_path = memory_dir() / "distillation" / f"{job_id}.json"
            if manifest_path.parent.exists():
                manifest_path.write_text(
                    json.dumps(
                        {
                            "job_id": job_id,
                            "status": status,
                            "spec": self.spec.to_dict(),
                            "checksum": corpus.checksum,
                            "example_count": len(corpus.examples),
                            "corpus_ref": corpus_ref,
                            "handoff": handoff,
                            "run_id": run_id,
                            "submitted_at": submitted_at,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        except Exception as e:  # noqa: BLE001 — manifest is best-effort
            logger.warning("[KG-2.318] manifest persist failed: %s", e)

        return DistillationJob(
            job_id=job_id,
            status=status,
            spec=self.spec.to_dict(),
            corpus_ref=corpus_ref,
            checksum=corpus.checksum,
            example_count=len(corpus.examples),
            handoff=handoff,
            detail=detail,
            submitted_at=submitted_at,
        )

    def submit(self, corpus: DistillationCorpus) -> DistillationJob:
        """Hand a distilled corpus off to data-science-mcp (CONCEPT:KG-2.316)."""
        if self._submitter is not None:
            return self._submitter(corpus, self.spec)
        return self._default_submit(corpus)

    def poll(self, job_id: str) -> dict[str, Any]:
        """Poll a submitted job's status, reading train state back (CONCEPT:KG-2.318).

        Prefers the live ``TrainingJob`` engine node — which data-science-mcp
        updates as the LoRA/SFT train advances ``running``→``succeeded``/``failed``
        and links the produced ``register_checkpoint`` node — and surfaces that
        checkpoint when present, so the poll reflects the REAL remote train state,
        not just the last state core wrote. Falls back to the on-disk manifest.
        """
        # Prefer a live engine job node when available.
        get_node = getattr(self.engine, "get_node", None)
        if callable(get_node):
            try:
                node = get_node(job_id)
                if isinstance(node, dict) and node:
                    result: dict[str, Any] = {
                        "job_id": job_id,
                        "status": node.get("status", "unknown"),
                        "source": "engine",
                        "node": node,
                    }
                    if node.get("run_id"):
                        result["run_id"] = node["run_id"]
                    # KG-2.318: surface the checkpoint the train registered back.
                    ckpt_ref = node.get("checkpoint_ref") or node.get("checkpoint")
                    if ckpt_ref:
                        try:
                            ckpt = get_node(str(ckpt_ref))
                            if isinstance(ckpt, dict) and ckpt:
                                result["checkpoint"] = ckpt
                        except Exception as e:  # noqa: BLE001 — checkpoint optional
                            logger.debug("[KG-2.318] checkpoint lookup failed: %s", e)
                    return result
            except Exception as e:  # noqa: BLE001 — fall through to the manifest
                logger.debug("[KG-2.318] job node lookup failed: %s", e)
        # Fall back to the on-disk manifest.
        try:
            from .memory_engine import memory_dir

            manifest = memory_dir() / "distillation" / f"{job_id}.json"
            if manifest.exists():
                data = json.loads(manifest.read_text(encoding="utf-8"))
                return {
                    "job_id": job_id,
                    "status": data.get("status", "unknown"),
                    "source": "manifest",
                    "manifest": data,
                }
        except Exception as e:  # noqa: BLE001 — degrade to not_found
            logger.debug("[KG-2.316] manifest poll failed: %s", e)
        return {"job_id": job_id, "status": "not_found"}


def distill_memory_to_weights(
    engine: Any,
    *,
    params: dict[str, Any] | None = None,
    submit: bool = False,
    nodes: list[dict[str, Any]] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Action-core for ``graph_analyze action=distill_memory`` (CONCEPT:KG-2.316/2.318).

    The single method BOTH surfaces (the ``graph_analyze`` MCP tool and its
    ``POST /graph/analyze`` REST twin) dispatch into. Reads consolidated/procedural
    memory, exports a LoRA/SFT corpus + spec, and — when ``submit`` — LIVE-dispatches
    it to data-science-mcp (CONCEPT:KG-2.318), returning a JSON-safe summary (corpus
    stats + spec + job). A ``poll_job_id`` param instead reads a submitted job's live
    ``TrainingJob``/checkpoint state back (the status-poll surface).
    """
    params = dict(params or {})
    # KG-2.318 — poll a submitted job's live train state (both surfaces).
    poll_job_id = str(params.pop("poll_job_id", "") or "").strip()
    if poll_job_id:
        return {
            "status": "ok",
            "concept": "KG-2.318",
            "poll": MemoryWeightsDistiller(engine).poll(poll_job_id),
        }
    spec = DistillationTargetSpec.from_params(params)
    distiller = MemoryWeightsDistiller(engine, spec=spec)
    corpus = distiller.export(nodes=nodes, now=now)
    result: dict[str, Any] = {
        "status": "ok",
        "concept": "KG-2.316",
        "corpus": corpus.summary(),
    }
    if submit:
        job = distiller.submit(corpus)
        result["job"] = job.to_dict()
    else:
        result["handoff"] = distiller._build_handoff("<pending export>")
    return result
