#!/usr/bin/env python3
"""Drift gate: every system-prompt blueprint conforms to the canonical schema.

CONCEPT:AU-ORCH.routing.resolve-body-single-canonical. Validates every ``"type": "prompt"`` JSON blueprint under
``agent_utilities/prompts/`` against ``validate_canonical`` (the ONE validator
shared with ``prompt-builder/validate_prompt.py`` and per-package
``test_prompt_parity``), and asserts the generated ``prompt.schema.json`` is
current. Baseline-gated like ``check_no_env_sprawl.py``: files listed in
``scripts/prompt_schema_baseline.txt`` are grandfathered (report-only) so a
migration can burn the list down to empty without breaking CI on day one.

Usage::

    python scripts/check_prompt_schema.py            # report mode (baseline-gated)
    python scripts/check_prompt_schema.py --strict   # fail on ANY non-canonical file
    python scripts/check_prompt_schema.py --update-baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROMPTS_DIR = REPO / "agent_utilities" / "prompts"
BASELINE = REPO / "scripts" / "prompt_schema_baseline.txt"


def _load_baseline() -> set[str]:
    if not BASELINE.exists():
        return set()
    return {
        line.strip()
        for line in BASELINE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


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
        errs = validate_canonical(data, strict=True)
        if errs:
            offenders[pfile.name] = errs
    return offenders


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args(argv)

    offenders = _scan()

    if args.update_baseline:
        BASELINE.write_text(
            "# Non-canonical prompt blueprints grandfathered by check_prompt_schema.\n"
            "# Burn this down to empty. CONCEPT:AU-ORCH.routing.resolve-body-single-canonical\n"
            + "".join(f"{name}\n" for name in sorted(offenders)),
            encoding="utf-8",
        )
        print(f"Wrote baseline with {len(offenders)} entries.")
        return 0

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

    baseline = _load_baseline()
    new_offenders = {k: v for k, v in offenders.items() if k not in baseline}

    if offenders:
        print("Non-canonical prompt blueprints:")
        for name, errs in sorted(offenders.items()):
            tag = "" if name not in baseline else "  (baseline)"
            print(f"  {name}{tag}: {'; '.join(errs)}")

    if schema_stale:
        print(
            "prompt.schema.json is stale — run: python scripts/gen_prompt_schema.py",
            file=sys.stderr,
        )

    fail = schema_stale or (offenders if args.strict else new_offenders)
    if fail:
        print(
            f"\nFAIL: {len(new_offenders)} new non-canonical prompt(s)"
            + (", schema stale" if schema_stale else "")
            + ".",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {len(offenders)} grandfathered, 0 new non-canonical prompts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
