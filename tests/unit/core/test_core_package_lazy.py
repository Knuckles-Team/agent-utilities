"""Cold-import contract for the core package boundary."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_core_package_loads_only_the_requested_runtime() -> None:
    root = Path(__file__).resolve().parents[3]
    code = """
import sys
import agent_utilities.core
assert 'agent_utilities.core.cognitive_scheduler' not in sys.modules
assert 'agent_utilities.core.release_channel' not in sys.modules
assert 'agent_utilities.core.wasm_runner' not in sys.modules
from agent_utilities.core import config
assert config.__name__ == 'agent_utilities.core.config'
assert 'agent_utilities.core.wasm_runner' not in sys.modules
from agent_utilities.core import CognitiveScheduler
assert CognitiveScheduler.__name__ == 'CognitiveScheduler'
assert 'agent_utilities.core.cognitive_scheduler' in sys.modules
assert 'agent_utilities.core.wasm_runner' not in sys.modules
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    environment["PYDANTIC_DISABLE_PLUGINS"] = "__all__"
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=root,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
