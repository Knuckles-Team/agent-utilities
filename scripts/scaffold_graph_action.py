#!/usr/bin/python
"""Scaffold a new graph_* tool action — kill the 6-file wiring drudgery (CONCEPT:AU-OS.deployment.os-2).

Adding one action means touching ~6 places (dispatch branch, REST twin, route,
manifest, description, test) + 3 generators. This emits ready-to-paste snippets for
all of them and writes a test stub, so a new surface is one command and can't drift.
It deliberately PRINTS the code-edit snippets (rather than mutating the big tool
files) so the author reviews each insertion; it only writes the new test file.

Usage:
    python scripts/scaffold_graph_action.py --tool graph_analyze --action my_action \\
        --concept KG-2.999 --summary "what it does"
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

_BRANCH = """            elif action == "{action}":
                # CONCEPT:{concept} — {summary}
                import json as _json

                # TODO: implement; `query`/`target`/`node_id`/`top_k`/`depth` are in scope.
                result = {{"status": "ok", "action": "{action}"}}
                return _json.dumps(result, default=str)"""

_REST = '''async def graph_analyze_{action}_endpoint(request: Request) -> JSONResponse:
    """REST twin of graph_analyze action={action} (CONCEPT:{concept}): {summary}."""
    try:
        body = await request.json()
    except Exception:
        body = {{}}
    try:
        res = await _execute_tool(
            "{tool}", action="{action}", query=body.get("query", ""),
            target=body.get("target", ""),
        )
        return JSONResponse({{"status": "success", "result": safe_json_load(res)}})
    except Exception as e:
        return JSONResponse({{"status": "error", "message": str(e)}}, status_code=500)'''

_ROUTE = (
    '    route("/graph/analyze/{dashed}", graph_analyze_{action}_endpoint, ["POST"])'
)

_TEST = '''"""Test for graph_analyze action={action} (CONCEPT:{concept})."""

from __future__ import annotations

import pytest


@pytest.mark.concept("{concept}")
def test_{action}_smoke():
    # TODO: exercise the action's core function with a fake engine and assert the
    # synthesized/structured result. Keep it engine-free (pure dispatch logic).
    assert True
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", default="graph_analyze")
    ap.add_argument("--action", required=True)
    ap.add_argument("--concept", required=True)
    ap.add_argument("--summary", default="TODO describe")
    args = ap.parse_args()
    ctx = {
        "tool": args.tool,
        "action": args.action,
        "concept": args.concept,
        "summary": args.summary,
        "dashed": args.action.replace("_", "-"),
    }

    print("=" * 72)
    print(f"Scaffold for {args.tool} action={args.action} (CONCEPT:{args.concept})")
    print("=" * 72)
    print("\n[1] Dispatch branch — paste BEFORE the `else:` in the action chain of")
    print(f"    agent_utilities/mcp/tools/analysis_tools.py ({args.tool}):\n")
    print(_BRANCH.format(**ctx))
    print("\n[2] REST twin — paste near the other graph_analyze_*_endpoint funcs in")
    print("    agent_utilities/mcp/kg_server.py:\n")
    print(_REST.format(**ctx))
    print(
        "\n[3] Route registration — paste with the other analyze routes in kg_server.py:\n"
    )
    print(_ROUTE.format(**ctx))
    print(
        "\n[4] Description — append a clause to the graph_analyze `description=` blob."
    )
    print(f"    '{args.action}' = {args.summary} (CONCEPT:{args.concept}).")

    test_path = REPO / "tests" / "unit" / f"test_{args.action}.py"
    if not test_path.exists():
        test_path.write_text(_TEST.format(**ctx), encoding="utf-8")
        print(f"\n[5] Wrote test stub: {test_path.relative_to(REPO)}")
    else:
        print(f"\n[5] Test already exists: {test_path.relative_to(REPO)} (left as-is)")

    print("\n[6] Regenerate the single-sources-of-truth:")
    print("    python scripts/gen_graphos_manifest.py   # verbose-op manifest")
    print("    python scripts/build_concepts_yaml.py     # concepts.yaml")
    print("    python scripts/validate_change.py          # diff-scoped quality bar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
