"""Tests for the deterministic ZERO-LLM ``a2a.json`` generator
(CONCEPT:AU-KG.ontology.a2a-card-generation).

Closes D-OB-1: ``a2a.json`` was 47 hand-maintained files that had already
drifted from what their certification ledger signed. These tests prove the
generator (``build_a2a_card``/``write_a2a_card`` in
``generate_connector_manifests.py``) derives every field from the connector's
own package metadata, is byte-stable across repeated runs, supports the
minimal ``[tool.a2a]`` residue for genuinely per-connector content, and that
``write_a2a_card(check=True)`` is a fail-closed drift gate.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "generate_connector_manifests",
    Path(__file__).resolve().parents[3] / "scripts" / "generate_connector_manifests.py",
)
gen = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(gen)


def _write_pyproject(root: Path, text: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(text, encoding="utf-8")
    return root


def test_build_a2a_card_derives_every_field_from_pyproject(tmp_path: Path):
    root = _write_pyproject(
        tmp_path / "acme-api",
        '[project]\nname = "acme-api"\nversion = "3.2.1"\n'
        'description = "Acme REST API + MCP Server"\n'
        '[project.license]\ntext = "MIT"\n'
        '[project.urls]\nHomepage = "https://github.com/Acme-Org/acme-api"\n',
    )
    card = gen.build_a2a_card(root)
    assert card == {
        "name": "acme-api-agent",
        "type": "agent",
        "version": "3.2.1",
        "description": "Acme REST API + MCP Server",
        "url": "https://github.com/Acme-Org/acme-api/tree/main",
        "license": "MIT",
        "capabilities": [dict(c) for c in gen.DEFAULT_A2A_CAPABILITIES],
        "tools": [dict(t) for t in gen.DEFAULT_A2A_TOOLS],
    }


def test_build_a2a_card_resolves_dynamic_version_attr(tmp_path: Path):
    root = _write_pyproject(
        tmp_path / "acme-api",
        '[project]\nname = "acme-api"\ndynamic = ["version"]\n'
        'description = "Acme"\n'
        "[tool.setuptools.dynamic]\n"
        'version = {attr = "acme_api._version.__version__"}\n',
    )
    pkg = root / "acme_api"
    pkg.mkdir()
    (pkg / "_version.py").write_text('__version__ = "9.9.9"\n', encoding="utf-8")

    card = gen.build_a2a_card(root)
    assert card["version"] == "9.9.9"


def test_build_a2a_card_falls_back_to_generic_description_with_todo(tmp_path: Path):
    root = _write_pyproject(
        tmp_path / "acme-api",
        '[project]\nname = "acme-api"\nversion = "1.0.0"\n',
    )
    todos: list[str] = []
    card = gen.build_a2a_card(root, todos=todos)
    assert card["description"] == "Agent package for acme-api"
    assert any("description" in t for t in todos)


def test_build_a2a_card_url_falls_back_to_heuristic_default_with_todo(tmp_path: Path):
    root = _write_pyproject(
        tmp_path / "acme-api",
        '[project]\nname = "acme-api"\nversion = "1.0.0"\ndescription = "Acme"\n',
    )
    todos: list[str] = []
    card = gen.build_a2a_card(root, todos=todos)
    assert card["url"] == "https://github.com/Knuckles-Team/acme-api/tree/main"
    assert any("url" in t for t in todos)


def test_build_a2a_card_tool_a2a_residue_appends_bespoke_entries(tmp_path: Path):
    root = _write_pyproject(
        tmp_path / "freshrss-agent",
        '[project]\nname = "freshrss-agent"\nversion = "2.0.0"\n'
        'description = "FreshRSS API + MCP Server + A2A Server"\n'
        "[[tool.a2a.capabilities]]\n"
        'id = "read_feeds"\nname = "Read RSS Feed Contents"\n'
        'description = "Fetch stream contents"\n'
        "[[tool.a2a.capabilities]]\n"
        'id = "curate_feeds"\nname = "Curate RSS Subscriptions"\n'
        'description = "Subscribe/unsubscribe"\n',
    )
    card = gen.build_a2a_card(root)
    ids = [c["id"] for c in card["capabilities"]]
    # defaults first, then the residue, in declared order — additive-only.
    assert ids == [c["id"] for c in gen.DEFAULT_A2A_CAPABILITIES] + [
        "read_feeds",
        "curate_feeds",
    ]


def test_build_a2a_card_tool_a2a_description_override(tmp_path: Path):
    root = _write_pyproject(
        tmp_path / "okta-agent",
        '[project]\nname = "okta-agent"\nversion = "2.0.0"\n'
        'description = "Okta CIAM/SSO MCP Server and Agent for Agentic AI!"\n'
        "[tool.a2a]\n"
        'description = "Okta agent: users, groups, applications, policies."\n',
    )
    card = gen.build_a2a_card(root)
    assert card["description"] == "Okta agent: users, groups, applications, policies."


def test_build_a2a_card_is_byte_stable_across_repeated_calls(tmp_path: Path):
    root = _write_pyproject(
        tmp_path / "acme-api",
        '[project]\nname = "acme-api"\nversion = "1.0.0"\ndescription = "Acme"\n',
    )
    first = gen._canonical_a2a_bytes(gen.build_a2a_card(root))
    second = gen._canonical_a2a_bytes(gen.build_a2a_card(root))
    assert first == second


def test_canonical_bytes_have_no_trailing_whitespace_ambiguity(tmp_path: Path):
    root = _write_pyproject(
        tmp_path / "acme-api",
        '[project]\nname = "acme-api"\nversion = "1.0.0"\ndescription = "Acme"\n',
    )
    payload = gen._canonical_a2a_bytes(gen.build_a2a_card(root))
    text = payload.decode("utf-8")
    assert text.endswith("\n")
    assert not text.endswith("\n\n")
    # round-trips through json.loads with no surprises
    json.loads(text)


def test_write_a2a_card_writes_the_canonical_bytes(tmp_path: Path):
    root = _write_pyproject(
        tmp_path / "acme-api",
        '[project]\nname = "acme-api"\nversion = "1.0.0"\ndescription = "Acme"\n',
    )
    card, changed = gen.write_a2a_card(root)
    assert changed is True
    on_disk = (root / "a2a.json").read_bytes()
    assert on_disk == gen._canonical_a2a_bytes(card)

    # idempotent re-run: no diff
    _, changed_again = gen.write_a2a_card(root)
    assert changed_again is False


def test_write_a2a_card_check_mode_never_writes_and_fails_closed_on_drift(
    tmp_path: Path,
):
    root = _write_pyproject(
        tmp_path / "acme-api",
        '[project]\nname = "acme-api"\nversion = "1.0.0"\ndescription = "Acme"\n',
    )
    (root / "a2a.json").write_text('{"hand": "edited"}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="drift"):
        gen.write_a2a_card(root, check=True)
    # check=True never writes, even on drift
    assert (root / "a2a.json").read_text(encoding="utf-8") == '{"hand": "edited"}'


def test_write_a2a_card_check_mode_passes_once_regenerated(tmp_path: Path):
    root = _write_pyproject(
        tmp_path / "acme-api",
        '[project]\nname = "acme-api"\nversion = "1.0.0"\ndescription = "Acme"\n',
    )
    gen.write_a2a_card(root)
    card, changed = gen.write_a2a_card(root, check=True)
    assert changed is False
    assert card["name"] == "acme-api-agent"
