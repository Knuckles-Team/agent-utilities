#!/usr/bin/env python3
"""Generate the canonical prompt JSON Schema from the Pydantic model.

CONCEPT:AU-ORCH.routing.resolve-body-single-canonical. The ``StructuredPrompt`` model is the single source of truth
for the system-prompt format; this writes its JSON Schema to
``agent_utilities/prompting/prompt.schema.json`` for editors/CI to validate
against. It is regenerated, never hand-edited — the same generator+checker
discipline as ``build_concepts_yaml.py`` / ``concepts.yaml``.

Usage::

    python scripts/gen_prompt_schema.py            # write the schema
    python scripts/gen_prompt_schema.py --check    # fail if committed schema is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO / "agent_utilities" / "prompting" / "prompt.schema.json"


def render_schema() -> str:
    sys.path.insert(0, str(REPO))
    from agent_utilities.prompting.structured import StructuredPrompt

    schema = StructuredPrompt.model_json_schema()
    return json.dumps(schema, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the committed schema differs from regeneration.",
    )
    args = parser.parse_args(argv)

    rendered = render_schema()
    if args.check:
        current = (
            SCHEMA_PATH.read_text(encoding="utf-8") if SCHEMA_PATH.exists() else ""
        )
        if current != rendered:
            print(
                f"prompt.schema.json is stale — run: python scripts/gen_prompt_schema.py\n"
                f"  ({SCHEMA_PATH})",
                file=sys.stderr,
            )
            return 1
        print("prompt.schema.json is current.")
        return 0

    SCHEMA_PATH.write_text(rendered, encoding="utf-8")
    print(f"Wrote {SCHEMA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
