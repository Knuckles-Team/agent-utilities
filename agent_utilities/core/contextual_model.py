#!/usr/bin/env python3
from __future__ import annotations

"""Mandatory evidence compilation at the model transport boundary.

Every Pydantic AI agent is built by :func:`create_context_agent`, which passes its
model through :func:`wrap_model_with_context` before construction.  The wrapper
resolves the verified ``GraphSession`` before touching a retriever, compiles an
evidence bundle, and prepends that bundle to both streaming and non-streaming
requests.  There is no ``skip_context`` argument: code that genuinely has no
evidence source receives an explicit empty bundle in local/test profiles and fails
closed in authenticated production profiles.

The module owns no provider client and stores no prompt, identity, endpoint, or
filesystem value.  Persisted bundle caching remains the ContextCompiler's job and
uses only privacy-safe opaque references.
"""

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from agent_utilities.core.config import setting
from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    graph_session_required,
    resolve_session,
)

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.retrieval.context_compiler import ContextBundle

__all__ = [
    "ContextCompilationError",
    "compile_model_context",
    "create_context_agent",
    "disable_context_agent_instrumentation",
    "instrument_context_agents",
    "set_context_compiler_engine",
    "set_context_compiler_cache",
    "get_context_compiler_cache",
    "use_context_compiler_engine",
    "wrap_model_with_context",
]

_CONTEXT_MARKER = "[agent-utilities:compiled-evidence:v1]"
_compiler_engine: Any | None = None
_compiler_cache: Any | None = None
_WRAPPER_CLASS: Any | None = None
_MISSING_MODEL = object()


class ContextCompilationError(PermissionError):
    """A model invocation could not establish its governed evidence context."""


class _EmptyEvidenceSource:
    """Development-only explicit empty source; never used in an auth-required profile."""

    def search_hybrid(
        self, query: str, *, top_k: int = 8, as_of: str | None = None
    ) -> list[dict[str, Any]]:
        del query, top_k, as_of
        return []

    def retrieve_epistemic_view(self, query: str, *, top_k: int = 8) -> dict[str, Any]:
        del query, top_k
        return {}


def set_context_compiler_engine(engine: Any | None) -> None:
    """Install the process evidence source during graph/runtime bootstrap."""

    global _compiler_engine
    _compiler_engine = engine


@contextmanager
def use_context_compiler_engine(engine: Any) -> Iterator[None]:
    """Temporarily install an evidence source and restore the exact prior source.

    The compiler source is process-wide bootstrap state. Callers that keep this
    scope open across an ``await`` must serialize those scopes; the bundled-skill
    validator does so with its direct-case execution lock.
    """

    previous = _compiler_engine
    set_context_compiler_engine(engine)
    try:
        yield
    finally:
        set_context_compiler_engine(previous)


def set_context_compiler_cache(cache: Any | None) -> None:
    """Install an optional shared KV backend for compiled bundles."""

    global _compiler_cache
    _compiler_cache = cache


def get_context_compiler_cache() -> Any | None:
    """Return the process's configured KV-cache backend, or ``None`` (CONCEPT:
    AU-KG.retrieval.context-compiler-kv-seam).

    Read-side counterpart to :func:`set_context_compiler_cache` — lets the
    ``tms_revalidation`` maintenance task (W3.2 TMS live-wiring) reach the SAME
    backend :func:`compile_model_context` stores compiled bundles into, so a
    bundle the engine's TMS marks stale can be dropped from the shared cache.
    """

    return _compiler_cache


def _active_engine() -> Any | None:
    if _compiler_engine is not None:
        return _compiler_engine
    try:
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

        return IntelligenceGraphEngine.get_active()
    except Exception:
        return None


def compile_model_context(
    query: str,
    *,
    session: GraphSession | None = None,
    engine: Any | None = None,
    model_version: str = "",
    snapshot: str = "",
) -> ContextBundle:
    """Compile the sole evidence bundle allowed to reach a model invocation."""

    from agent_utilities.knowledge_graph.retrieval.context_compiler import (
        ContextCompiler,
    )

    authority = resolve_session(session, required_scope="kg:read")
    source = engine or _active_engine()
    if source is None:
        if graph_session_required():
            raise ContextCompilationError(
                "authenticated model invocation requires a configured ContextCompiler engine"
            )
        source = _EmptyEvidenceSource()

    try:
        budget = max(64, int(setting("MODEL_CONTEXT_TOKEN_BUDGET", "2000") or 2000))
    except (TypeError, ValueError):
        budget = 2000
    return ContextCompiler(source).compile(
        str(query or ""),
        authority,
        token_budget=budget,
        kv_backend=_compiler_cache,
        model_version=str(model_version or ""),
        redaction_version=str(
            setting("MODEL_CONTEXT_REDACTION_VERSION", "permissioning-v1")
            or "permissioning-v1"
        ),
        evidence_ordering_version=str(
            setting("MODEL_CONTEXT_ORDERING_VERSION", "context-mmr-v1")
            or "context-mmr-v1"
        ),
        snapshot=str(snapshot or authority.catalog_epoch or ""),
    )


def _part_text(part: Any) -> str:
    content = getattr(part, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list | tuple):
        return " ".join(str(item) for item in content if isinstance(item, str))
    return ""


def _query_from_messages(messages: list[Any]) -> str:
    for message in reversed(messages):
        for part in reversed(list(getattr(message, "parts", ()) or ())):
            kind = str(getattr(part, "part_kind", ""))
            if kind in {"user-prompt", "system-prompt"}:
                text = _part_text(part).strip()
                # Marker text is not a capability: a caller is allowed to ask
                # about it and that complete turn must still drive retrieval.
                # A genuinely compiled request is returned early by
                # ``_already_compiled`` before this query extractor runs.
                if text:
                    return text
    return ""


def _already_compiled(messages: list[Any]) -> bool:
    if not messages:
        return False
    parts = list(getattr(messages[0], "parts", ()) or ())
    if not parts:
        return False
    # The wrapper always prepends exactly this leading system part. Trusting a
    # marker anywhere else would let caller-controlled history suppress the
    # mandatory compiler. The marker is an idempotency sentinel, not an auth
    # token, and its position is therefore part of the boundary contract.
    first = parts[0]
    return str(getattr(first, "part_kind", "")) == "system-prompt" and _part_text(
        first
    ).lstrip().startswith(_CONTEXT_MARKER)


def _compile_messages(messages: list[Any], model_name: str) -> list[Any]:
    if _already_compiled(messages):
        return messages
    query = _query_from_messages(messages)
    bundle = compile_model_context(query, model_version=model_name)
    from pydantic_ai.messages import ModelRequest, SystemPromptPart

    evidence = ModelRequest(
        parts=[
            SystemPromptPart(
                content=(
                    f"{_CONTEXT_MARKER}\n"
                    "This governed evidence bundle is the only factual context "
                    "for the request. Cite it or state that evidence is absent.\n\n"
                    f"{bundle.as_text()}"
                )
            )
        ]
    )
    return [evidence, *messages]


def _wrapper_class() -> Any | None:
    global _WRAPPER_CLASS
    if _WRAPPER_CLASS is not None:
        return _WRAPPER_CLASS
    try:
        from pydantic_ai.models.wrapper import WrapperModel
    except Exception:
        return None

    class _ContextCompiledModel(WrapperModel):  # type: ignore[misc, valid-type]
        _agent_utilities_context_wrapper = True

        async def request(
            self, messages: list[Any], model_settings: Any, mrp: Any
        ) -> Any:
            governed = _compile_messages(messages, self.model_name)
            return await super().request(governed, model_settings, mrp)

        async def count_tokens(
            self, messages: list[Any], model_settings: Any, mrp: Any
        ) -> Any:
            governed = _compile_messages(messages, self.model_name)
            return await super().count_tokens(governed, model_settings, mrp)

        async def compact_messages(
            self, request_context: Any, *, instructions: str | None = None
        ) -> Any:
            governed = _compile_messages(request_context.messages, self.model_name)
            return await super().compact_messages(
                replace(request_context, messages=governed),
                instructions=instructions,
            )

        @asynccontextmanager
        async def request_stream(
            self,
            messages: list[Any],
            model_settings: Any,
            mrp: Any,
            run_context: Any | None = None,
        ) -> AsyncIterator[Any]:
            governed = _compile_messages(messages, self.model_name)
            async with super().request_stream(
                governed, model_settings, mrp, run_context
            ) as response:
                yield response

    _WRAPPER_CLASS = _ContextCompiledModel
    return _WRAPPER_CLASS


def wrap_model_with_context(model: Any) -> Any:
    """Return the mandatory ContextCompiler wrapper, idempotently."""

    if isinstance(model, str):
        # Pydantic AI accepts convenient model-name strings, but they otherwise
        # bypass the model-factory transport choke point entirely. Resolve them
        # through the factory before wrapping; the second call is a real Model
        # and therefore cannot recurse through this branch.
        from agent_utilities.core.model_factory import create_model

        provider, model_id = (None, model)
        if ":" in model:
            prefix, candidate = model.split(":", 1)
            if prefix.lower() in {
                "openai",
                "anthropic",
                "google",
                "gemini",
                "groq",
                "mistral",
                "huggingface",
                "ollama",
                "deepseek",
            }:
                provider, model_id = prefix.lower(), candidate
                if provider == "gemini":
                    provider = "google"
        return create_model(provider=provider, model_id=model_id)
    if getattr(model, "_agent_utilities_context_wrapper", False):
        return model
    wrapper = _wrapper_class()
    if wrapper is None:
        raise ContextCompilationError(
            "pydantic-ai WrapperModel is required for governed model invocation"
        )
    return wrapper(model)


def create_context_agent(
    model: Any = _MISSING_MODEL,
    *,
    permissions_kernel: Any | None = None,
    agent_identity: Any | None = None,
    permission_engine: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Construct a Pydantic AI agent behind the mandatory context boundary.

    This is the sole application constructor for Pydantic AI agents.  It requires
    an explicit model, idempotently installs the transport wrapper, and only then
    instantiates the agent.  Runtime model injection therefore cannot bypass the
    compiler by handing an arbitrary model object directly to ``Agent``.
    """

    if model is _MISSING_MODEL or model is None:
        raise ContextCompilationError(
            "governed agent construction requires an explicit model"
        )
    toolsets = list(kwargs.get("toolsets", ()) or ())
    raw_mcp_bound = any(
        hasattr(toolset, "list_tools") or hasattr(toolset, "direct_call_tool")
        for toolset in toolsets
    )
    if raw_mcp_bound:
        if permissions_kernel is None or agent_identity is None:
            raise PermissionError(
                "raw MCP toolsets require an explicitly injected permission context"
            )
        from agent_utilities.security.permissions_kernel import (
            verify_permission_context,
        )
        from agent_utilities.security.tool_guard import flag_mcp_tool_definitions

        permission_context = verify_permission_context(
            permissions_kernel,
            agent_identity,
        )
        if permission_context is None:  # defensive: an explicit pair was supplied
            raise PermissionError("MCP permission context is required")
        kwargs["toolsets"] = flag_mcp_tool_definitions(
            toolsets,
            permissions_kernel=permission_context.kernel,
            agent_identity=permission_context.identity,
            engine=permission_engine,
        )
    elif (permissions_kernel is None) != (agent_identity is None):
        raise PermissionError(
            "permission kernel and agent identity must be injected together"
        )
    elif permissions_kernel is not None and agent_identity is not None:
        from agent_utilities.security.permissions_kernel import (
            verify_permission_context,
        )

        verify_permission_context(permissions_kernel, agent_identity)
    from pydantic_ai import Agent as _PydanticAgent

    return _PydanticAgent(model=wrap_model_with_context(model), **kwargs)


def instrument_context_agents(settings: Any) -> Any:
    """Install process-wide instrumentation on the governed agent runtime."""

    from pydantic_ai import Agent as _PydanticAgent

    _PydanticAgent.instrument_all(settings)
    return getattr(_PydanticAgent, "_instrument_default", None)


def disable_context_agent_instrumentation() -> None:
    """Disable process-wide instrumentation on the governed agent runtime."""

    from pydantic_ai import Agent as _PydanticAgent

    _PydanticAgent.instrument_all(False)
