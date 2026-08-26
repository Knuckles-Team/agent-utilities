"""Freshness + determinism contract for the published OpenAPI reference.

``scripts/generate_openapi.py`` projects the live Agent Web Dashboard app's
own ``app.openapi()`` into two committed artifacts under ``docs/reference/``
(``openapi.json`` + the rendered ``api.md`` catalog page). These tests prove:

* the generator is deterministic -- the same code produces a byte-identical
  spec across independent app builds (CONCEPT: this repo's byte-identical
  wheel gate applies the same discipline to generated docs artifacts);
* the committed artifacts are not stale relative to what the generator
  produces right now, mirroring the ``docs_contract.py --check`` pattern
  already used for the other generated catalogs under ``docs/reference/``;
* the honesty note about schema-less mounted routes is derived generically
  from the live app (never a hardcoded route list) and actually fires when
  a route really does carry no schema.

Building the real app is comparable in cost to agent-webui's own
``test_canonical_gateway_mount.py`` drift guard, so this is marked
``integration`` like that test.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "generate_openapi", ROOT / "scripts" / "generate_openapi.py"
)
gen = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(gen)


def test_openapi_spec_matches_the_served_app():
    """The generator's app is the real one: same title/paths agent-webui pins."""
    spec = gen.openapi_spec()
    assert spec["info"]["title"] == "Agent Web Dashboard"
    assert spec["openapi"].startswith("3.")
    assert spec["paths"], "expected a non-empty OpenAPI path set"


def test_render_spec_is_deterministic_across_independent_app_builds():
    """Two independent app builds must yield a byte-identical serialized spec."""
    spec_a = gen.openapi_spec()
    spec_b = gen.openapi_spec()
    rendered_a = gen.render_spec(spec_a)
    rendered_b = gen.render_spec(spec_b)
    assert rendered_a == rendered_b
    # And it must actually be valid, re-parseable JSON with sorted top-level keys.
    parsed = json.loads(rendered_a)
    assert parsed == spec_a


def test_render_spec_has_no_trailing_whitespace_or_timestamp_drift():
    rendered = gen.render_spec(gen.openapi_spec())
    assert rendered.endswith("\n")
    assert not rendered.endswith("\n\n")
    # No obvious embedded build-time timestamp/host markers.
    for banned in ("localhost", "127.0.0.1", "0.0.0.0"):
        assert banned not in rendered


def test_schemaless_routes_is_generic_not_hardcoded():
    """The coverage note must be computed from the live route table, not a list.

    ``schemaless_routes`` must not reference any specific path literal in its
    own source -- the whole point is that it shrinks on its own as parallel
    lanes add typed schemas, with nothing here to hand-maintain.
    """
    source = Path(gen.__file__).read_text(encoding="utf-8")
    # The docstring is allowed to name example surfaces (for a human reader);
    # only the executable body after it must not special-case one.
    def_start = source.index("def schemaless_routes")
    def_end = source.index("\ndef _group_key")
    def_block = source[def_start:def_end]
    doc_start = def_block.index('"""')
    doc_end = def_block.index('"""', doc_start + 3) + 3
    body = def_block[doc_end:]
    for banned in ("/api/graph", "/api/engine", "\"graph\"", "'graph'"):
        assert banned not in body, f"schemaless_routes() must not hardcode {banned!r}"


def test_schemaless_routes_detects_a_real_gap():
    """A raw Starlette route with no schema must actually be reported."""
    app = gen.build_app()
    spec = gen.openapi_spec()
    count, prefixes = gen.schemaless_routes(app, spec)
    # This repo mounts the canonical KG REST surface as raw Starlette routes
    # (register_graph_routes / kg_server._mount_rest_routes) specifically so
    # the same route code serves webui and gateway clients -- it is real,
    # live, schema-less surface today, not a fixture. If this ever goes to
    # zero, the honesty note on the docs page correctly stops appearing, and
    # this assertion should be revisited alongside it.
    assert count > 0
    assert any(prefix.startswith("/api/") for prefix in prefixes)


def test_meta_and_websocket_routes_are_excluded_from_the_gap_count():
    """FastAPI's own doc endpoints and websockets are not a "coverage gap"."""
    app = gen.build_app()
    spec = gen.openapi_spec()
    _, prefixes = gen.schemaless_routes(app, spec)
    for path in spec.get("paths", {}):
        assert path not in ("/openapi.json", "/docs", "/redoc")
    assert "/openapi.json" not in prefixes
    assert "/docs" not in prefixes
    assert "/redoc" not in prefixes
    assert "/ws" not in prefixes and "/ws/dashboard" not in prefixes


def test_render_page_states_the_incompleteness_honestly():
    spec = gen.openapi_spec()
    app = gen.build_app()
    count, prefixes = gen.schemaless_routes(app, spec)
    page = gen.render_page(spec, schemaless_count=count, schemaless_prefixes=prefixes)
    assert "GENERATED" in page
    assert "do not edit by hand" in page
    if count:
        assert "does not yet cover the whole live API" in page
        assert str(count) in page
    else:  # pragma: no cover - only true once every surface is typed
        assert "covers the whole served surface" in page


class TestFailClosedOnTruncatedSurface:
    """Regression tests for the "silent partial write, exit 0" defect.

    `create_agent_web_app` mounts several distinct route groups (the
    optional service dashboard API, the mandatory canonical KG REST
    surface via ``register_graph_routes``) behind their own
    import/registration guards. A guard that fails soft still returns a
    *working* app -- just one missing an entire surface -- so
    ``build_app()``/``openapi_spec()`` here can succeed with a
    well-formed spec that is nonetheless missing 50+ paths. Before this
    fix, that truncated spec got written to
    ``docs/reference/openapi.json`` and the script exited 0, so a lane
    running the documented refresh command without diffing the output
    would commit a large, plausible-looking documentation regression.
    """

    def test_assert_surface_complete_raises_below_the_floor(self):
        spec = {"paths": {f"/x{i}": {} for i in range(gen._MIN_EXPECTED_PATHS - 1)}}
        with pytest.raises(gen.IncompleteSurfaceError):
            gen._assert_surface_complete(spec)

    def test_assert_surface_complete_passes_at_the_floor(self):
        spec = {"paths": {f"/x{i}": {} for i in range(gen._MIN_EXPECTED_PATHS)}}
        gen._assert_surface_complete(spec)  # must not raise

    def test_write_refuses_a_truncated_spec_and_leaves_the_artifact_untouched(
        self, monkeypatch
    ):
        """Force the exact condition Defect 2 exploited: a spec with far
        fewer paths than the served app actually carries (as happens when
        a mounted surface silently fails to register) -- ``write()`` must
        raise instead of committing it, and the real, already-committed
        artifact on disk must be left completely unmodified.
        """
        original = gen.SPEC_PATH.read_text(encoding="utf-8")

        def _truncated_spec():
            return {
                "info": {"title": "Agent Web Dashboard"},
                "openapi": "3.1.0",
                "paths": {f"/x{i}": {} for i in range(5)},
            }

        monkeypatch.setattr(gen, "openapi_spec", _truncated_spec)

        with pytest.raises(gen.IncompleteSurfaceError):
            gen.write()

        with pytest.raises(gen.IncompleteSurfaceError):
            gen.check()

        assert gen.SPEC_PATH.read_text(encoding="utf-8") == original, (
            "a failed --write must never modify the committed artifact"
        )

    def test_main_exits_nonzero_and_reports_failure_on_a_truncated_surface(
        self, monkeypatch, capsys
    ):
        def _truncated_spec():
            return {
                "info": {"title": "Agent Web Dashboard"},
                "openapi": "3.1.0",
                "paths": {f"/x{i}": {} for i in range(3)},
            }

        monkeypatch.setattr(gen, "openapi_spec", _truncated_spec)
        monkeypatch.setattr("sys.argv", ["generate_openapi.py", "--write"])

        exit_code = gen.main()

        assert exit_code == 1, (
            "the CLI must exit non-zero on a truncated surface, not silently "
            "report success"
        )
        out = capsys.readouterr().out
        assert "FAILED" in out
        assert "truncated" in out.lower() or "Refusing" in out

    def test_generator_errors_when_the_canonical_kg_surface_fails_to_import(
        self, monkeypatch
    ):
        """End-to-end: the SAME failure condition Defect 1 forces (a broken
        ``agent_utilities.gateway.graph_api`` import) must make the
        generator error out rather than silently writing whatever spec the
        (now headless) app happens to produce.

        Post-fix, ``agent_webui.server.create_agent_web_app`` itself
        refuses to build a headless app and raises ``RuntimeError`` --
        this proves the generator does not swallow that failure either.
        """
        monkeypatch.setitem(sys.modules, "agent_utilities.gateway.graph_api", None)

        original = gen.SPEC_PATH.read_text(encoding="utf-8")
        with pytest.raises(RuntimeError):
            gen.write()
        assert gen.SPEC_PATH.read_text(encoding="utf-8") == original


def test_committed_artifacts_are_not_stale():
    """``docs/reference/{openapi.json,api.md}`` must match the generator's output.

    This is the freshness gate: run in CI/pre-commit as
    ``python scripts/generate_openapi.py --check`` (script mode) or here, as
    a test, so a code change that alters the served API surface without
    regenerating the committed artifact fails loudly instead of silently
    drifting from the code that documents it.
    """
    errors = gen.check()
    assert not errors, (
        "docs/reference/openapi.json and/or api.md are stale; run "
        "`python scripts/generate_openapi.py --write` and commit the result:\n"
        + "\n".join(errors)
    )
