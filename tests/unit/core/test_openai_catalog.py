"""CONCEPT:AU-ORCH.adapter.openai-catalog-verification — OpenAI model catalogue discovery/verification.

Covers ``verify_openai_model``'s exists/missing/no-key/no-client paths, and proves the
credential-leak boundary: a failed verification's ``error`` is a class name only, never
the underlying exception's message (which the ``openai`` SDK can populate with the
request URL/headers, i.e. the API key) — the repo's ``PersistencePrivacyGuard`` api_token
pattern (``sk-...``) is used as the independent proof that nothing sk-shaped survives.
"""

from __future__ import annotations

import pytest

from agent_utilities.core.openai_catalog import verify_openai_model
from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

pytestmark = pytest.mark.concept(id="AU-ORCH.adapter.openai-catalog-verification")

# Self-identifying synthetic fixture: the literal carries the "example"
# marker `scripts/check_fleet_supply_chain.py` (SC-SEC-002 /
# SYNTHETIC_TOKEN_MARKERS) uses to tell a test double from live credential
# material, while keeping the exact `sk-proj-` shape the redactors match
# (`agent_utilities/security/persistence_privacy.py`,
# `agent_utilities/httpsupport/redaction.py`), so the test still exercises the
# real redaction path.
_FAKE_KEY = "sk-proj-example-abcdefghijklmnopqrstuvwxyz0123456789"


@pytest.mark.asyncio
async def test_verify_openai_model_no_api_key():
    result = await verify_openai_model("gpt-4o-mini", api_key=None)
    assert result.exists is False
    assert result.error == "no API key configured"
    assert _FAKE_KEY not in (result.error or "")


@pytest.mark.asyncio
async def test_verify_openai_model_empty_model_id():
    result = await verify_openai_model("", api_key=_FAKE_KEY)
    assert result.exists is False
    assert result.error == "empty model_id"


class _FakeModelsResource:
    def __init__(self, *, raise_exc: Exception | None, record: object | None):
        self._raise_exc = raise_exc
        self._record = record

    async def retrieve(self, model_id: str):
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._record


class _FakeAsyncOpenAI:
    last_init_kwargs: dict | None = None

    def __init__(self, **kwargs):
        _FakeAsyncOpenAI.last_init_kwargs = kwargs
        self.models = kwargs.pop("_models_resource", None) or _FakeModelsResource(
            raise_exc=kwargs.get("_raise_exc"), record=kwargs.get("_record")
        )

    async def close(self):
        return None


class _FakeRecord:
    def __init__(self, owned_by: str, created: int):
        self.owned_by = owned_by
        self.created = created


@pytest.mark.asyncio
async def test_verify_openai_model_exists(monkeypatch):
    record = _FakeRecord(owned_by="openai", created=1700000000)

    class _Client(_FakeAsyncOpenAI):
        def __init__(self, **kwargs):
            self.models = _FakeModelsResource(raise_exc=None, record=record)

    monkeypatch.setitem(
        __import__("sys").modules,
        "openai",
        type("M", (), {"AsyncOpenAI": _Client}),
    )
    result = await verify_openai_model(
        "gpt-4o-mini", api_key=_FAKE_KEY, base_url="https://api.openai.com/v1"
    )
    assert result.exists is True
    assert result.owned_by == "openai"
    assert result.created == 1700000000
    assert result.error is None


@pytest.mark.asyncio
async def test_verify_openai_model_empty_base_url_uses_sdk_default(monkeypatch):
    record = _FakeRecord(owned_by="openai", created=1700000000)

    class _Client(_FakeAsyncOpenAI):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.models = _FakeModelsResource(raise_exc=None, record=record)

    monkeypatch.setitem(
        __import__("sys").modules,
        "openai",
        type("M", (), {"AsyncOpenAI": _Client}),
    )

    result = await verify_openai_model("gpt-4o-mini", api_key=_FAKE_KEY, base_url="")

    assert result.exists is True
    assert _FakeAsyncOpenAI.last_init_kwargs == {"api_key": _FAKE_KEY}


@pytest.mark.asyncio
async def test_verify_openai_model_missing_never_leaks_the_api_key(monkeypatch):
    class _AuthError(Exception):
        def __init__(self, key: str):
            # Simulates an SDK exception whose message embeds request context
            # (URL/headers) that can carry the credential — the real-world leak
            # vector this module's class-name-only error contract closes.
            super().__init__(f"401 Unauthorized: Bearer {key} rejected")

    class _Client(_FakeAsyncOpenAI):
        def __init__(self, **kwargs):
            self.models = _FakeModelsResource(
                raise_exc=_AuthError(_FAKE_KEY), record=None
            )

    monkeypatch.setitem(
        __import__("sys").modules,
        "openai",
        type("M", (), {"AsyncOpenAI": _Client}),
    )
    result = await verify_openai_model("gpt-4o-mini", api_key=_FAKE_KEY)
    assert result.exists is False
    assert result.error == "_AuthError"  # class name only
    assert _FAKE_KEY not in (result.error or "")

    # Independent proof via the repo's own privacy sanitizer: the persisted result
    # (as it would be embedded in a doctor report or graph property) contains no
    # sk-shaped token anywhere.
    guard = PersistencePrivacyGuard()
    payload = {
        "model_id": result.model_id,
        "exists": result.exists,
        "error": result.error,
        "checked_at": result.checked_at,
    }
    clean, report = guard.sanitize(payload)
    assert report.changed is False  # nothing needed redacting -> nothing was there
    assert clean == payload
