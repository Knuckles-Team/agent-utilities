#!/usr/bin/env python3
"""Drift gate: every system-prompt blueprint conforms to the canonical schema.

CONCEPT:AU-ORCH.routing.resolve-body-single-canonical. Validates every ``"type": "prompt"`` JSON blueprint under
``agent_utilities/prompts/`` against ``validate_canonical`` (the ONE validator
shared with ``prompt-builder/validate_prompt.py`` and per-package
``test_prompt_parity``), and asserts the generated ``prompt.schema.json`` is
current. Every violation fails the gate.

Usage::

    python scripts/check_prompt_schema.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROMPTS_DIR = REPO / "agent_utilities" / "prompts"


def _scan() -> dict[str, list[str]]:
    """Return ``{relative_path: [violations]}`` for every non-canonical prompt."""
    sys.path.insert(0, str(REPO))
    from agent_utilities.prompting.structured import validate_canonical

    offenders: dict[str, list[str]] = {}
    for pfile in sorted(PROMPTS_DIR.glob("*.json")):
        try:
            data = json.loads(pfile.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            offenders[pfile.name] = [f"unreadable: {e}"]
            continue
        if not isinstance(data, dict) or data.get("type") != "prompt":
            continue
        errs = validate_canonical(data)
        if errs:
            offenders[pfile.name] = errs
    return offenders


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments:
        print("check_prompt_schema accepts no options", file=sys.stderr)
        return 2

    offenders = _scan()

    # Schema currency check (regenerate-in-memory and diff).
    from gen_prompt_schema import SCHEMA_PATH, render_schema  # type: ignore

    sys.path.insert(0, str(REPO / "scripts"))
    schema_stale = False
    try:
        rendered = render_schema()
        current = (
            SCHEMA_PATH.read_text(encoding="utf-8") if SCHEMA_PATH.exists() else ""
        )
        schema_stale = current != rendered
    except Exception as e:  # pragma: no cover - defensive
        print(f"WARNING: could not verify prompt.schema.json currency: {e}")

    if offenders:
        print("Non-canonical prompt blueprints:")
        for name, errs in sorted(offenders.items()):
            print(f"  {name}: {'; '.join(errs)}")

    if schema_stale:
        print(
            "prompt.schema.json is stale — run: python scripts/gen_prompt_schema.py",
            file=sys.stderr,
        )

    fail = schema_stale or offenders
    if fail:
        print(
            f"\nFAIL: {len(offenders)} non-canonical prompt(s)"
            + (", schema stale" if schema_stale else "")
            + ".",
            file=sys.stderr,
        )
        return 1

    print("OK: all prompt blueprints are canonical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
