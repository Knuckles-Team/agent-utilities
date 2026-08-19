"""Epistemic RAG as the default delegated-run context path (W3.7, CONCEPT:AU-KG.retrieval.context-compiler).

**The assembly path.** ``execute_agent``/``execute_workflow`` (``orchestration.manager.
Orchestrator.execute_agent`` -> ``orchestration.agent_runner.run_agent``) constructs
every one of its sub-agents — single-server, focused-tools, direct-completion, and
every node of the full multi-agent graph (router/dispatcher/planner/specialist) —
through :func:`agent_utilities.agent.factory.create_agent`, which is built on
:func:`agent_utilities.core.contextual_model.create_context_agent`. Verified by grep
(``rg "from pydantic_ai import Agent"``): exactly ONE runtime construction site exists
in the whole package, ``core/contextual_model.py`` itself; every other agent
constructor (``agent/factory.py``, ``graph/executor.py``,
``graph/hierarchical_planner.py``, ``graph/_router_impl.py``, ``graph/lifecycle.py``)
calls ``create_context_agent``. So the model-transport wrapper installed there
(:func:`~agent_utilities.core.contextual_model.wrap_model_with_context`) IS the
delegated-run context-assembly path — every model call a delegated run makes goes
through it, and it is what these tests exercise (directly, and one level up through
``orchestration.agent_runner._run_direct_completion``, a real execute_agent
sub-path — never a reimplementation of the wiring).

Covers the W3.7 acceptance:
* default-on proven — a fresh config (``MODEL_CONTEXT_COMPILER_ENABLED`` unset)
  takes the ContextCompiler path.
* flag-off works — the config-contract escape hatch actually disables compilation,
  and only from deployment config, never from request/task content.
* citations/proof references are embedded in the ACTUAL text reaching the model,
  end-to-end — not merely constructible on the ``ContextBundle`` API in isolation.
* the Seam-6 (X6) TTFT + KV-cache-hit metrics are recorded on this path, in the
  established ``observability/gateway_metrics.py`` Prometheus pattern.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.core import contextual_model
from agent_utilities.core.contextual_model import (
    _compile_messages,
    _compiled_evidence_and_bundle,
    _context_compiler_enabled,
    _record_delegated_run_ttft,
    use_context_compiler_engine,
)
from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    get_company_brain,
    reset_company_brain,
)
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.knowledge_graph.ontology.permissioning import (
    clear_markings,
    use_marking_authority,
)
from agent_utilities.models.company_brain import ActorType, DataClassification, NodeACL
from agent_utilities.security.brain_context import ActorContext

# ---------------------------------------------------------------------------
# Shared fixtures / fakes (mirrors tests/retrieval/test_context_compiler_mandatory.py)
# ---------------------------------------------------------------------------


class _FakeMarkingStore:
    """Minimal in-memory durable-store stand-in for the mandatory-marking seam."""

    @staticmethod
    def execute(_query, _params):
        return []


@pytest.fixture(autouse=True)
def _clean_state():
    reset_company_brain()
    clear_markings()
    with use_marking_authority(_FakeMarkingStore()):
        yield
    reset_company_brain()
    clear_markings()


class _CitationBearingEngine:
    """A ``search_hybrid`` retriever whose candidates carry real provenance
    (``source_refs``/``evidence_refs``/``confidence``) so a compiled bundle is
    genuinely citation-bearing, not merely non-empty."""

    def __init__(self) -> None:
        self.calls = 0

    def search_hybrid(
        self,
        query: str,
        *,
        top_k: int = 8,
        as_of: str | None = None,
        session: object | None = None,
    ) -> list[dict[str, object]]:
        del query, top_k, as_of, session
        self.calls += 1
        return [
            {
                "id": "evidence-1",
                "content": "Refunds require manager approval per policy 4.2.",
                "source_refs": ["policy-doc:4.2"],
                "confidence": 0.9,
                "score": 1.0,
            },
            {
                "id": "evidence-2",
                "content": "Approval must be logged in the ticket within 24h.",
                "source_refs": ["policy-doc:4.3"],
                "evidence_refs": ["ticket-log-schema"],
                "confidence": 0.8,
                "score": 0.8,
            },
        ]


class _BoomEngine:
    """A retriever that fails the test if it is ever touched — used to prove the
    flag-off escape hatch never reaches retrieval at all."""

    def search_hybrid(self, *args: object, **kwargs: object) -> list[dict[str, object]]:
        raise AssertionError(
            "retrieval must never run when MODEL_CONTEXT_COMPILER_ENABLED=false"
        )


def _session(*, scopes: frozenset[str] = frozenset({"kg:read"})) -> GraphSession:
    return GraphSession(
        actor=ActorContext(
            actor_id="test-principal",
            actor_type=ActorType.AI_AGENT,
            roles=(),
            tenant_id="test-tenant",
            authenticated=True,
        ),
        tenant="test-tenant",
        graph="test-graph",
        scopes=scopes,
        policy_version="policy-v1",
    )


def _grant_public(*node_ids: str) -> None:
    for nid in node_ids:
        get_company_brain().permissions.set_acl(
            NodeACL(node_id=nid, classification=DataClassification.PUBLIC)
        )


def _user_message(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        parts=[SimpleNamespace(part_kind="user-prompt", content=text)]
    )


# ---------------------------------------------------------------------------
# 1) Default-on
# ---------------------------------------------------------------------------


def test_fresh_config_routes_through_compiler_by_default(monkeypatch) -> None:
    """Acceptance: 'default-on proven (fresh config -> compiler path taken)'."""
    monkeypatch.delenv("MODEL_CONTEXT_COMPILER_ENABLED", raising=False)
    assert _context_compiler_enabled() is True

    _grant_public("evidence-1", "evidence-2")
    engine = _CitationBearingEngine()
    with use_context_compiler_engine(engine), use_session(_session()):
        governed, bundle = _compiled_evidence_and_bundle(
            [_user_message("What is the refund policy?")], "test-model"
        )

    assert engine.calls == 1, "the compiler path was not actually taken"
    assert bundle is not None
    assert len(governed) == 2  # [compiled-evidence, *original]


def test_original_conversation_is_preserved_after_the_evidence_prefix(
    monkeypatch,
) -> None:
    monkeypatch.delenv("MODEL_CONTEXT_COMPILER_ENABLED", raising=False)
    _grant_public("evidence-1", "evidence-2")
    engine = _CitationBearingEngine()
    original = [_user_message("What is the refund policy?")]
    with use_context_compiler_engine(engine), use_session(_session()):
        governed, _bundle = _compiled_evidence_and_bundle(original, "test-model")
    assert governed[1:] == original


# ---------------------------------------------------------------------------
# 2) Citations / proof references end-to-end
# ---------------------------------------------------------------------------


def test_compiled_evidence_is_citation_bearing_end_to_end(monkeypatch) -> None:
    """Acceptance: 'Every execute_agent context is citation-bearing' — asserted on
    the ACTUAL text prepended to the model request, not merely on the
    ``ContextBundle`` API in isolation."""
    monkeypatch.delenv("MODEL_CONTEXT_COMPILER_ENABLED", raising=False)
    _grant_public("evidence-1", "evidence-2")
    engine = _CitationBearingEngine()
    with use_context_compiler_engine(engine), use_session(_session()):
        governed, bundle = _compiled_evidence_and_bundle(
            [_user_message("What is the refund policy?")], "test-model"
        )

    assert bundle is not None
    assert bundle.citations, "bundle carries zero citations"
    assert all(c.source_refs for c in bundle.citations), (
        "a citation was selected with no source_refs"
    )

    evidence_text = governed[0].parts[0].content
    assert evidence_text.startswith(contextual_model._CONTEXT_MARKER)
    assert "Citations:" in evidence_text
    # Every citation the bundle carries is literally present in the text that
    # reaches the model — the end-to-end guarantee, not just an object attribute.
    for citation in bundle.citations:
        assert citation.node_id in evidence_text
        for ref in citation.source_refs:
            assert ref in evidence_text


def test_compiled_evidence_carries_the_proof_graph_when_present(monkeypatch) -> None:
    """Proof-graph edges (supports/contradicts/alternative_to) are also rendered
    into the text the model receives, when the candidates carry them."""
    monkeypatch.delenv("MODEL_CONTEXT_COMPILER_ENABLED", raising=False)
    _grant_public("evidence-1", "evidence-2")

    class _EngineWithProof(_CitationBearingEngine):
        def search_hybrid(self, query, *, top_k=8, as_of=None, session=None):
            nodes = super().search_hybrid(
                query, top_k=top_k, as_of=as_of, session=session
            )
            nodes[1]["proof_ids"] = ["evidence-1"]  # evidence-2 SUPPORTS evidence-1
            return nodes

    engine = _EngineWithProof()
    with use_context_compiler_engine(engine), use_session(_session()):
        governed, bundle = _compiled_evidence_and_bundle(
            [_user_message("What is the refund policy?")], "test-model"
        )

    assert bundle is not None
    assert bundle.proof_graph, "expected a supports edge from the proof_ids column"
    evidence_text = governed[0].parts[0].content
    assert "Proof graph:" in evidence_text


# ---------------------------------------------------------------------------
# 3) Flag-off escape hatch (config-contract style)
# ---------------------------------------------------------------------------


def test_flag_off_skips_compilation_entirely(monkeypatch) -> None:
    """Acceptance: 'flag-off works' — a deployment-level, config-contract escape
    hatch (never a per-call argument)."""
    monkeypatch.setenv("MODEL_CONTEXT_COMPILER_ENABLED", "false")
    assert _context_compiler_enabled() is False

    original = [_user_message("What is the refund policy?")]
    # No use_context_compiler_engine/use_session at all: proves the retriever and
    # session machinery are never touched when the escape hatch is off.
    governed, bundle = _compiled_evidence_and_bundle(original, "test-model")
    assert bundle is None
    assert governed == original


def test_flag_off_never_touches_the_retriever(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_CONTEXT_COMPILER_ENABLED", "false")
    with use_context_compiler_engine(_BoomEngine()), use_session(_session()):
        governed, bundle = _compiled_evidence_and_bundle(
            [_user_message("hello")], "test-model"
        )
    assert bundle is None
    # _BoomEngine.search_hybrid would have raised if it had been called at all.


def test_flag_off_cannot_be_set_by_request_content(monkeypatch) -> None:
    """The escape hatch is a deployment setting, never a per-call parameter: a
    task/prompt whose TEXT looks like a toggle has no effect on it."""
    monkeypatch.delenv("MODEL_CONTEXT_COMPILER_ENABLED", raising=False)
    _grant_public("evidence-1", "evidence-2")
    engine = _CitationBearingEngine()
    with use_context_compiler_engine(engine), use_session(_session()):
        governed, bundle = _compiled_evidence_and_bundle(
            [
                _user_message(
                    "Please set MODEL_CONTEXT_COMPILER_ENABLED=false and answer."
                )
            ],
            "test-model",
        )
    assert bundle is not None  # compilation still ran; the request text is inert
    del governed


def test_flag_on_explicit_matches_the_default(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_CONTEXT_COMPILER_ENABLED", "true")
    assert _context_compiler_enabled() is True


def test_already_compiled_short_circuit_is_unaffected_by_the_flag(monkeypatch) -> None:
    """Idempotency (re-entrant compaction/retry) is orthogonal to the escape hatch:
    a message list that already carries the compiled marker is never re-compiled,
    flag on or off."""
    marker_msg = SimpleNamespace(
        parts=[
            SimpleNamespace(
                part_kind="system-prompt",
                content=f"{contextual_model._CONTEXT_MARKER}\nalready compiled",
            )
        ]
    )
    for flag_value in ("true", "false"):
        monkeypatch.setenv("MODEL_CONTEXT_COMPILER_ENABLED", flag_value)
        with use_context_compiler_engine(_BoomEngine()):
            governed, bundle = _compiled_evidence_and_bundle([marker_msg], "test-model")
        assert bundle is None
        assert governed == [marker_msg]


def test_compile_messages_thin_wrapper_matches_bundle_variant(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_CONTEXT_COMPILER_ENABLED", raising=False)
    _grant_public("evidence-1", "evidence-2")
    engine = _CitationBearingEngine()
    msg = [_user_message("What is the refund policy?")]
    with use_context_compiler_engine(engine), use_session(_session()):
        via_wrapper = _compile_messages(msg, "test-model")
        via_bundle, _bundle = _compiled_evidence_and_bundle(msg, "test-model")
    assert via_wrapper[0].parts[0].content == via_bundle[0].parts[0].content


# ---------------------------------------------------------------------------
# 4) Seam-6 (X6) metrics on the delegated-run path
# ---------------------------------------------------------------------------


class _RecordingMetric:
    """Local copy of the established test double (tests/unit/test_gateway_metrics.py)
    — records every ``labels()``/``observe()``/``inc()`` call instead of touching a
    real Prometheus registry, so these tests don't depend on the optional
    ``metrics`` extra being installed."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict, float]] = []
        self._labels: dict = {}

    def labels(self, **kwargs):
        clone = _RecordingMetric()
        clone.events = self.events
        clone._labels = kwargs
        return clone

    def observe(self, value: float) -> None:
        self.events.append(("observe", self._labels, value))

    def inc(self, amount: float = 1.0) -> None:
        self.events.append(("inc", self._labels, amount))


def test_ttft_surrogate_recorded_on_the_delegated_run_wrapper(monkeypatch) -> None:
    """Seam-6/X6: the TTFT surrogate must be measurable on the delegated-run path,
    not only the batch ``bundle_chat_completion()`` call site."""
    import agent_utilities.observability.gateway_metrics as gm

    recorder = _RecordingMetric()
    monkeypatch.setattr(gm, "CONTEXT_COMPILER_TTFT", recorder)
    monkeypatch.delenv("MODEL_CONTEXT_COMPILER_ENABLED", raising=False)
    _grant_public("evidence-1", "evidence-2")
    engine = _CitationBearingEngine()

    with use_context_compiler_engine(engine), use_session(_session()):
        _governed, bundle = _compiled_evidence_and_bundle(
            [_user_message("What is the refund policy?")], "test-model"
        )
        _record_delegated_run_ttft(0.042, bundle)

    assert recorder.events, "no TTFT observation was recorded"
    kind, labels, value = recorder.events[-1]
    assert kind == "observe"
    assert value == 0.042
    assert labels["path"] == "delegated_run"
    assert labels["kv_cache_hit"] in {"true", "false"}


def test_ttft_not_recorded_when_no_bundle_was_compiled(monkeypatch) -> None:
    """Passthrough calls (already-compiled, or the escape hatch is off) attribute
    no latency to a compiled bundle — there is none."""
    import agent_utilities.observability.gateway_metrics as gm

    recorder = _RecordingMetric()
    monkeypatch.setattr(gm, "CONTEXT_COMPILER_TTFT", recorder)
    _record_delegated_run_ttft(0.01, None)
    assert recorder.events == []


def test_kv_cache_hit_rate_wired_on_the_delegated_run_path(monkeypatch) -> None:
    """Seam-6/X6 KV/prefix-cache hit-rate: ``compile_model_context`` always threads
    the process ``kv_backend`` through to ``ContextCompiler.compile()`` — a second,
    identical compile is served from the KV cache (a real hit), so the metric fires
    correctly on THIS path, not merely constructible on the compiler in isolation."""
    import agent_utilities.observability.gateway_metrics as gm

    recorder = _RecordingMetric()
    monkeypatch.setattr(gm, "CONTEXT_COMPILER_KV_CACHE", recorder)
    monkeypatch.delenv("MODEL_CONTEXT_COMPILER_ENABLED", raising=False)
    _grant_public("evidence-1", "evidence-2")
    engine = _CitationBearingEngine()

    class _MemoryKV:
        def __init__(self) -> None:
            self.values: dict[str, bytes] = {}

        def get(self, key: str) -> bytes | None:
            return self.values.get(key)

        def put(self, key: str, value: bytes) -> bool:
            self.values[key] = value
            return True

    kv = _MemoryKV()
    contextual_model.set_context_compiler_cache(kv)
    try:
        with use_context_compiler_engine(engine), use_session(_session()):
            _first, bundle1 = _compiled_evidence_and_bundle(
                [_user_message("What is the refund policy?")], "test-model"
            )
            _second, bundle2 = _compiled_evidence_and_bundle(
                [_user_message("What is the refund policy?")], "test-model"
            )
    finally:
        contextual_model.set_context_compiler_cache(None)

    assert bundle1 is not None and bundle2 is not None
    assert bundle1.kv_cache_hit is False
    assert bundle2.kv_cache_hit is True
    outcomes = [labels.get("outcome") for _kind, labels, _v in recorder.events]
    assert "miss" in outcomes
    assert "hit" in outcomes


# ---------------------------------------------------------------------------
# Live path: a real execute_agent sub-path, not a reimplementation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_direct_completion_live_path_is_citation_bearing(monkeypatch) -> None:
    """``execute_agent``'s direct-completion sub-path
    (``orchestration.agent_runner._run_direct_completion``, reached automatically
    for a trivial/chat turn — CONCEPT:AU-ORCH.execution.direct-completion-shape) actually
    sends citation-bearing compiled evidence to the model. Proven with a
    ``FunctionModel`` standing in for the LLM (the established pattern —
    tests/orchestration/test_swe_agent_loop.py), capturing the LITERAL request it
    receives — not the ``ContextBundle`` API, the real wire."""
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    import agent_utilities.core.model_factory as model_factory_mod
    from agent_utilities.orchestration import agent_runner

    monkeypatch.delenv("MODEL_CONTEXT_COMPILER_ENABLED", raising=False)
    _grant_public("evidence-1", "evidence-2")
    engine = _CitationBearingEngine()

    captured: list[list] = []

    def model_fn(messages: list, info: AgentInfo) -> ModelResponse:
        captured.append(messages)
        return ModelResponse(parts=[TextPart("Refunds need manager approval.")])

    fn_model = FunctionModel(model_fn)
    monkeypatch.setattr(model_factory_mod, "create_model", lambda **_kw: fn_model)

    with use_context_compiler_engine(engine), use_session(_session()):
        result = await agent_runner._run_direct_completion(
            "What is the refund policy?", None
        )

    assert result["results"]["output"]
    assert captured, "the model never received a request"
    final_messages = captured[-1]
    evidence_part = final_messages[0].parts[0]
    assert evidence_part.content.startswith(contextual_model._CONTEXT_MARKER)
    assert "Citations:" in evidence_part.content
    assert "evidence-1" in evidence_part.content


@pytest.mark.asyncio
async def test_run_direct_completion_live_path_honors_the_flag_off_escape_hatch(
    monkeypatch,
) -> None:
    """The same live sub-path with the escape hatch off: the model receives the
    RAW turn only, with no compiled evidence prefix and no retrieval call."""
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    import agent_utilities.core.model_factory as model_factory_mod
    from agent_utilities.orchestration import agent_runner

    monkeypatch.setenv("MODEL_CONTEXT_COMPILER_ENABLED", "false")

    captured: list[list] = []

    def model_fn(messages: list, info: AgentInfo) -> ModelResponse:
        captured.append(messages)
        return ModelResponse(parts=[TextPart("ok")])

    fn_model = FunctionModel(model_fn)
    monkeypatch.setattr(model_factory_mod, "create_model", lambda **_kw: fn_model)

    with use_context_compiler_engine(_BoomEngine()):
        await agent_runner._run_direct_completion("What is the refund policy?", None)

    assert captured
    first_message = captured[-1][0]
    assert not any(
        str(getattr(p, "content", "")).startswith(contextual_model._CONTEXT_MARKER)
        for p in first_message.parts
    )
