"""Shared fakes for the L4 harness port tests: a fake ``claude`` executable."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

#: Behaviours the fake executable supports, chosen by the ``FAKE_MODE`` env value.
_SCRIPT = """#!{python}
import json, os, sys, time
prompt = sys.stdin.read()
record = {{"argv": sys.argv[1:], "cwd": os.getcwd(), "prompt": prompt,
          "env": sorted(os.environ)}}
with open({record!r}, "w") as fh:
    json.dump(record, fh)
mode = os.environ.get("FAKE_MODE", "ok")
if mode == "sleep":
    time.sleep(30)
if mode == "garbage":
    print("not json at all")
    sys.exit(1)
result = {{
    "type": "result",
    "subtype": "success" if mode == "ok" else "error_during_execution",
    "is_error": mode != "ok",
    "result": "done: " + prompt.strip(),
    "session_id": "sess-123",
    "total_cost_usd": 0.0421,
    "usage": {{"input_tokens": 11, "output_tokens": 7,
              "cache_read_input_tokens": 3, "cache_creation_input_tokens": 2}},
    "modelUsage": {{"fake-model-1": {{"inputTokens": 11}}}},
}}
print(json.dumps({{"type": "system", "subtype": "init"}}))
print(json.dumps(result))
sys.exit(0 if mode == "ok" else 2)
"""


def write_fake_claude(directory: Path) -> tuple[Path, Path]:
    """Write the fake executable; return ``(executable, record_file)``."""
    record = directory / "fake-claude-record.json"
    executable = directory / "fake-claude"
    executable.write_text(_SCRIPT.format(python=sys.executable, record=str(record)))
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    return executable, record


def read_record(record: Path) -> dict:
    return json.loads(record.read_text())
