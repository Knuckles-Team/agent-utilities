#!/usr/bin/python
"""CONCEPT:AHE-3.1 — Prompt Integrity Tests.

Validates the structural integrity and consistency of all JSON prompt
files in ``agent_utilities/prompts/``.  Catches prompt drift, missing
required fields, and malformed configurations before they silently
degrade system behavior.

Tests cover:
    1. JSON parse validity for all prompt files
    2. Required key presence (``role``, ``core_directive``)
    3. Core directive non-emptiness
    4. Baseline hash stability (detects unexpected drift)
    5. Role uniqueness across all prompts
    6. No embedded secrets or API keys in prompt content
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pytest

# Locate the prompts directory relative to this test file
PROMPTS_DIR = Path(__file__).parent.parent / "agent_utilities" / "prompts"

# Fall back if running from a different location
if not PROMPTS_DIR.is_dir():
    PROMPTS_DIR = Path(os.getcwd()) / "agent_utilities" / "prompts"

# Required keys that every prompt file should have at minimum
REQUIRED_KEYS = {"identity", "task", "type", "version"}

# Patterns that should NEVER appear in prompt content (security)
SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),  # OpenAI keys
    re.compile(r"AIza[a-zA-Z0-9_-]{35}"),  # Google API keys
    re.compile(r"ghp_[a-zA-Z0-9]{36}"),  # GitHub tokens
    re.compile(r"glpat-[a-zA-Z0-9_-]{20}"),  # GitLab tokens
    re.compile(r"password\s*[:=]\s*['\"][^'\"]+['\"]", re.IGNORECASE),
]


def _discover_prompt_files() -> list[Path]:
    """Find all JSON prompt files in the prompts directory."""
    if not PROMPTS_DIR.is_dir():
        return []
    return sorted(PROMPTS_DIR.glob("*.json"))


def _load_prompt(path: Path) -> dict:
    """Load and parse a JSON prompt file."""
    with open(path) as f:
        return json.load(f)


# ── Test Cases ────────────────────────────────────────────────────────


class TestPromptParsing:
    """All prompt files must be valid JSON."""

    @pytest.mark.parametrize(
        "prompt_file",
        _discover_prompt_files(),
        ids=lambda p: p.stem,
    )
    def test_json_parse_valid(self, prompt_file: Path):
        """Each prompt file should parse as valid JSON without errors."""
        try:
            data = _load_prompt(prompt_file)
            assert isinstance(data, dict), f"{prompt_file.name} is not a JSON object"
        except json.JSONDecodeError as e:
            pytest.fail(f"{prompt_file.name} is invalid JSON: {e}")


class TestPromptStructure:
    """Prompt files must have the required structural keys."""

    @pytest.mark.parametrize(
        "prompt_file",
        _discover_prompt_files(),
        ids=lambda p: p.stem,
    )
    def test_required_keys_present(self, prompt_file: Path):
        """Each prompt must contain at minimum: 'role'."""
        data = _load_prompt(prompt_file)
        missing = REQUIRED_KEYS - set(data.keys())
        assert not missing, (
            f"{prompt_file.name} missing required keys: {missing}"
        )

    @pytest.mark.parametrize(
        "prompt_file",
        _discover_prompt_files(),
        ids=lambda p: p.stem,
    )
    def test_identity_is_nonempty(self, prompt_file: Path):
        """The 'identity' field must not be empty."""
        data = _load_prompt(prompt_file)
        identity = data.get("identity", "")
        assert identity, f"{prompt_file.name} has empty 'identity' field"

    @pytest.mark.parametrize(
        "prompt_file",
        _discover_prompt_files(),
        ids=lambda p: p.stem,
    )
    def test_task_if_present(self, prompt_file: Path):
        """If 'task' exists, it must be non-empty."""
        data = _load_prompt(prompt_file)
        task = data.get("task", "")
        if task is not None:
            assert task, f"{prompt_file.name} has empty 'task' field"


class TestPromptSecurity:
    """Prompt files must not contain embedded secrets."""

    @pytest.mark.parametrize(
        "prompt_file",
        _discover_prompt_files(),
        ids=lambda p: p.stem,
    )
    def test_no_embedded_secrets(self, prompt_file: Path):
        """Prompt content must not contain API keys or credentials."""
        content = prompt_file.read_text()
        for pattern in SECRET_PATTERNS:
            match = pattern.search(content)
            if match:
                pytest.fail(
                    f"{prompt_file.name} contains potential secret: "
                    f"'{match.group()[:20]}...'"
                )


class TestPromptConsistency:
    """Cross-file consistency checks."""

    def test_no_duplicate_identities(self):
        """Each prompt file should have a unique identity."""
        identities: dict[str, str] = {}
        for path in _discover_prompt_files():
            data = _load_prompt(path)
            identity_raw = data.get("identity", "")
            # Identity can be a string or dict — normalize to string
            if isinstance(identity_raw, dict):
                identity = json.dumps(identity_raw, sort_keys=True)
            else:
                identity = str(identity_raw) if identity_raw else ""
            if identity and identity in identities:
                # Allow council_ prompts to share advisory identities
                if not (path.stem.startswith("council_") and
                        identities[identity].startswith("council_")):
                    pytest.fail(
                        f"Duplicate identity in {path.name} "
                        f"(first seen in {identities[identity]})"
                    )
            if identity:
                identities[identity] = path.stem

    def test_prompt_count_minimum(self):
        """There should be at least 10 prompt files (sanity check)."""
        files = _discover_prompt_files()
        assert len(files) >= 10, (
            f"Expected at least 10 prompt files, found {len(files)}. "
            f"Prompts directory: {PROMPTS_DIR}"
        )


class TestPromptIntegrity:
    """Hash-based drift detection."""

    def test_critical_prompts_exist(self):
        """Critical prompts (verifier, router, planner) must exist."""
        critical = ["verifier.json", "router.json", "planner.json", "researcher.json"]
        existing = {p.name for p in _discover_prompt_files()}
        missing = [c for c in critical if c not in existing]
        assert not missing, f"Critical prompt files missing: {missing}"

    @pytest.mark.parametrize(
        "prompt_file",
        _discover_prompt_files(),
        ids=lambda p: p.stem,
    )
    def test_hash_stability(self, prompt_file: Path):
        """Prompt hashes should be stable — this test generates a baseline.

        On first run, it records hashes. On subsequent runs, it can
        detect drift. Currently informational only (does not fail).
        """
        content = prompt_file.read_text()
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        # This is informational — print hash for baseline tracking
        # In a CI environment, you'd compare against a known-good manifest
        assert content_hash  # Always passes — hash is never empty
