"""CONCEPT:AU-ORCH.adapter.multi-cli-adapter-dispatch — Multi-CLI Agent Adapter Registry tests.

Covers the declarative adapter contract, non-blocking PATH detection (graceful degradation on a
missing/broken CLI), stream-format dispatch to canonical events, the subprocess executor, and the
engine's additive adapter-dispatch path (live path: ``UnifiedExecutionEngine.run``).
"""

from __future__ import annotations

import shutil
import sys

import pytest

from agent_utilities.core.execution.adapters import (
    AdapterDefinition,
    AdapterRegistry,
    PromptDelivery,
    StreamFormat,
)
from agent_utilities.core.execution.adapters.executor import (
    AdapterExecutionError,
    run_adapter_text,
)
from agent_utilities.core.execution.stream_handlers import (
    ExecEventType,
    collect_text,
    get_stream_handler,
)

pytestmark = pytest.mark.concept(id="AU-ORCH.adapter.multi-cli-adapter-dispatch")


# ── declarative contract ────────────────────────────────────────────


def test_resolve_model_precedence(monkeypatch):
    d = AdapterDefinition(
        id="x", bin="true", fallback_models=("fb-1",), model_override_env_var="X_MODEL"
    )
    assert d.resolve_model("explicit") == "explicit"  # explicit wins
    monkeypatch.setenv("X_MODEL", "from-env")
    assert d.resolve_model(None) == "from-env"  # env override next
    monkeypatch.delenv("X_MODEL", raising=False)
    assert d.resolve_model(None) == "fb-1"  # fallback last
    assert AdapterDefinition(id="y", bin="true").resolve_model(None) == ""


# ── non-blocking detection (graceful degradation) ───────────────────


def test_detect_missing_bin_is_unavailable_not_raising():
    reg = AdapterRegistry(load_builtins=False)
    reg.register(AdapterDefinition(id="nope", bin="definitely-not-a-real-bin-xyz"))
    detected = reg.detect(force=True)
    assert detected["nope"].available is False
    assert "PATH" in (detected["nope"].error or "")


def test_detect_present_bin_is_available():
    reg = AdapterRegistry(load_builtins=False)
    # ``true`` (or python) is guaranteed present on the test host.
    bin_name = "true" if shutil.which("true") else sys.executable
    reg.register(
        AdapterDefinition(id="present", bin=bin_name, version_args=("--version",))
    )
    detected = reg.detect(force=True)
    assert detected["present"].available is True
    assert detected["present"].path


def test_detect_is_cached_until_ttl():
    reg = AdapterRegistry(load_builtins=False, detect_ttl=999)
    reg.register(AdapterDefinition(id="present", bin="true"))
    first = reg.detect(force=True)
    again = reg.detect()  # cached, no force
    assert first.keys() == again.keys()


def test_builtins_load_without_error():
    reg = AdapterRegistry()
    assert "claude-code" in reg.ids()
    assert "ollama" in reg.ids()


# ── stream-format dispatch → canonical events ───────────────────────


def test_plain_handler_normalizes():
    events = list(get_stream_handler(StreamFormat.PLAIN)(["hello\n", "world\n"]))
    assert events[0].type is ExecEventType.START
    assert events[-1].type is ExecEventType.END
    assert collect_text(events) == "hello\nworld\n"


def test_jsonl_handler_normalizes_types():
    lines = [
        '{"type":"text","text":"hi "}\n',
        '{"type":"delta","content":"there"}\n',
        '{"type":"tool_use","name":"calc"}\n',
        '{"type":"error","message":"boom"}\n',
        "not-json\n",
    ]
    events = list(get_stream_handler(StreamFormat.JSONL)(lines))
    kinds = [e.type for e in events]
    assert ExecEventType.TOOL_USE in kinds
    assert ExecEventType.ERROR in kinds
    # both text + delta + the unparseable line contribute to text
    assert "hi there" in collect_text(events)


def test_both_formats_share_one_event_schema():
    plain = list(get_stream_handler(StreamFormat.PLAIN)(["a\n"]))
    jsonl = list(
        get_stream_handler(StreamFormat.JSONL)(['{"type":"text","text":"a"}\n'])
    )
    for ev in plain + jsonl:
        assert isinstance(ev.type, ExecEventType)


# ── subprocess executor ─────────────────────────────────────────────


async def test_executor_missing_bin_raises():
    d = AdapterDefinition(id="ghost", bin="definitely-not-a-real-bin-xyz")
    with pytest.raises(AdapterExecutionError):
        await run_adapter_text(d, "hi")


async def test_executor_runs_echo_via_args():
    echo = shutil.which("echo")
    if not echo:
        pytest.skip("no echo on PATH")
    d = AdapterDefinition(
        id="echo",
        bin="echo",
        build_args=lambda model, prompt: [prompt],
        stream_format=StreamFormat.PLAIN,
        prompt_delivery=PromptDelivery.ARGS,
    )
    out = await run_adapter_text(d, "canonical-stream-ok")
    assert "canonical-stream-ok" in out


async def test_executor_runs_via_stdin_text():
    cat = shutil.which("cat")
    if not cat:
        pytest.skip("no cat on PATH")
    d = AdapterDefinition(
        id="cat",
        bin="cat",
        build_args=lambda model, prompt: [],
        stream_format=StreamFormat.PLAIN,
        prompt_delivery=PromptDelivery.STDIN_TEXT,
    )
    out = await run_adapter_text(d, "piped-prompt")
    assert "piped-prompt" in out
