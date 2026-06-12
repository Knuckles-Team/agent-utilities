"""Tests for the centralized live env accessor ``config.setting`` (config discipline).

``setting()`` is the sanctioned replacement for bare ``os.environ.get``/``os.getenv``
in modules: typed, defaulted, config.json-driven, and read live at call time.
"""

from __future__ import annotations

import pytest

from agent_utilities.core.config import setting


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in ("AU_T_STR", "AU_T_INT", "AU_T_FLOAT", "AU_T_BOOL", "AU_T_LIST"):
        monkeypatch.delenv(k, raising=False)
    yield


def test_unset_returns_default():
    assert setting("AU_T_STR") is None
    assert setting("AU_T_STR", "fallback") == "fallback"
    assert setting("AU_T_INT", 7) == 7


def test_empty_string_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv("AU_T_STR", "")
    assert setting("AU_T_STR", "fallback") == "fallback"


def test_type_inferred_from_default(monkeypatch):
    monkeypatch.setenv("AU_T_INT", "42")
    monkeypatch.setenv("AU_T_FLOAT", "1.5")
    monkeypatch.setenv("AU_T_BOOL", "true")
    monkeypatch.setenv("AU_T_LIST", "a,b,c")
    assert setting("AU_T_INT", 0) == 42
    assert isinstance(setting("AU_T_INT", 0), int)
    assert setting("AU_T_FLOAT", 0.0) == 1.5
    assert setting("AU_T_BOOL", False) is True
    assert setting("AU_T_LIST", []) == ["a", "b", "c"]


def test_explicit_cast_overrides_inference(monkeypatch):
    monkeypatch.setenv("AU_T_INT", "9")
    assert setting("AU_T_INT", cast=int) == 9
    assert setting("AU_T_STR", "x", cast=str) == "x"


def test_bad_cast_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("AU_T_INT", "not-a-number")
    assert setting("AU_T_INT", 5) == 5


def test_reads_are_live(monkeypatch):
    # The whole point: a value set AFTER import is still observed (unlike a
    # frozen AgentConfig field).
    assert setting("AU_T_STR", "d") == "d"
    monkeypatch.setenv("AU_T_STR", "live")
    assert setting("AU_T_STR", "d") == "live"
