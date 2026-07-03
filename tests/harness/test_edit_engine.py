"""Tests for the multi-format edit-application engine (CONCEPT:ORCH-1.46)."""

from __future__ import annotations

import pytest

from agent_utilities.harness.edit_engine import (
    apply_edits,
    apply_with_reflection,
    parse_edits,
)

SAMPLE = """def greet(name):
    message = "hello " + name
    return message
"""


def _sr(path: str, search: str, replace: str) -> str:
    return f"{path}\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


def test_parse_search_replace_block():
    text = _sr("a.py", "old line", "new line")
    edits = parse_edits(text)
    assert len(edits) == 1
    assert edits[0].path == "a.py"
    assert edits[0].search.strip() == "old line"
    assert edits[0].replace.strip() == "new line"


def test_exact_match_applies(tmp_path):
    f = tmp_path / "a.py"
    f.write_text(SAMPLE, encoding="utf-8")
    text = _sr("a.py", '    message = "hello " + name', '    message = f"hello {name}"')
    result = apply_edits(parse_edits(text), root=tmp_path)
    assert result.ok
    assert result.outcomes[0].strategy == "exact"
    assert 'f"hello {name}"' in f.read_text(encoding="utf-8")


def test_leading_whitespace_fuzzy_match(tmp_path):
    """Search block with the wrong indentation still matches."""
    f = tmp_path / "a.py"
    f.write_text(SAMPLE, encoding="utf-8")
    # Model dropped the 4-space indent on the search text.
    text = _sr("a.py", 'message = "hello " + name', 'message = f"hi {name}"')
    result = apply_edits(parse_edits(text), root=tmp_path)
    assert result.ok
    assert result.outcomes[0].strategy == "leading-whitespace"
    out = f.read_text(encoding="utf-8")
    assert 'f"hi {name}"' in out
    # Indentation preserved on the replacement.
    assert '    message = f"hi {name}"' in out


def test_closest_window_match(tmp_path):
    """A search block with a small typo still lands via SequenceMatcher."""
    f = tmp_path / "a.py"
    f.write_text(SAMPLE, encoding="utf-8")
    # A one-character typo ("helo") — too different for exact/whitespace tiers,
    # but well above the 0.8 SequenceMatcher threshold.
    text = _sr(
        "a.py",
        '    message = "helo " + name',
        '    message = "HELLO " + name',
    )
    result = apply_edits(parse_edits(text), root=tmp_path)
    assert result.ok
    assert result.outcomes[0].strategy == "closest-window"
    assert '"HELLO "' in f.read_text(encoding="utf-8")


def test_failed_match_returns_hint(tmp_path):
    f = tmp_path / "a.py"
    f.write_text(SAMPLE, encoding="utf-8")
    text = _sr("a.py", "totally unrelated content here", "x = 1")
    result = apply_edits(parse_edits(text), root=tmp_path)
    assert not result.ok
    failure = result.failures[0]
    assert failure.path == "a.py"
    assert "did not match" in failure.reason
    # File untouched on failure.
    assert f.read_text(encoding="utf-8") == SAMPLE


def test_empty_search_appends(tmp_path):
    f = tmp_path / "a.py"
    f.write_text(SAMPLE, encoding="utf-8")
    text = "a.py\n<<<<<<< SEARCH\n=======\n# trailing comment\n>>>>>>> REPLACE\n"
    result = apply_edits(parse_edits(text), root=tmp_path)
    assert result.ok
    assert result.outcomes[0].strategy == "append"
    assert f.read_text(encoding="utf-8").endswith("# trailing comment\n")


def test_create_new_file(tmp_path):
    text = "new.py\n<<<<<<< SEARCH\n=======\nprint('hi')\n>>>>>>> REPLACE\n"
    result = apply_edits(parse_edits(text), root=tmp_path)
    assert result.ok
    assert (tmp_path / "new.py").read_text(encoding="utf-8").strip() == "print('hi')"


def test_multiple_edits_compose_same_file(tmp_path):
    f = tmp_path / "a.py"
    f.write_text(SAMPLE, encoding="utf-8")
    text = _sr("a.py", "def greet(name):", "def greet(name, loud=False):") + _sr(
        "a.py", "    return message", "    return message.upper() if loud else message"
    )
    result = apply_edits(parse_edits(text), root=tmp_path)
    assert result.ok
    out = f.read_text(encoding="utf-8")
    assert "loud=False" in out
    assert "message.upper()" in out


def test_unified_diff_parse_and_apply(tmp_path):
    f = tmp_path / "a.py"
    f.write_text(SAMPLE, encoding="utf-8")
    diff = (
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,3 +1,3 @@\n"
        " def greet(name):\n"
        '-    message = "hello " + name\n'
        '+    message = "HI " + name\n'
        "     return message\n"
    )
    edits = parse_edits(diff, fmt="auto")
    assert edits and edits[0].fmt == "unified-diff"
    result = apply_edits(edits, root=tmp_path)
    assert result.ok
    assert '"HI "' in f.read_text(encoding="utf-8")


def test_malformed_block_raises():
    text = "a.py\n<<<<<<< SEARCH\nnever closed\n"
    with pytest.raises(ValueError, match="Unterminated SEARCH"):
        parse_edits(text)


def test_missing_filename_raises():
    text = "<<<<<<< SEARCH\nfoo\n=======\nbar\n>>>>>>> REPLACE\n"
    with pytest.raises(ValueError, match="missing a filename"):
        parse_edits(text)


@pytest.mark.asyncio
async def test_reflection_recovers_from_failure(tmp_path):
    """A failed first attempt is re-prompted; the corrected edit then applies."""
    f = tmp_path / "a.py"
    f.write_text(SAMPLE, encoding="utf-8")
    bad = _sr("a.py", "nonexistent search text", "x = 1")
    good = _sr("a.py", "    return message", "    return message  # done")

    calls = {"n": 0}

    async def reprompt(correction: str) -> str:
        calls["n"] += 1
        assert "could not be applied" in correction
        return good

    result = await apply_with_reflection(
        bad, reprompt, root=tmp_path, max_reflections=2
    )
    assert result.ok
    assert calls["n"] == 1
    assert "# done" in f.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_reflection_recovers_from_malformed(tmp_path):
    f = tmp_path / "a.py"
    f.write_text(SAMPLE, encoding="utf-8")
    malformed = "a.py\n<<<<<<< SEARCH\nunterminated"
    good = _sr("a.py", "    return message", "    return message  # fixed")

    async def reprompt(correction: str) -> str:
        assert "malformed" in correction.lower()
        return good

    result = await apply_with_reflection(
        malformed, reprompt, root=tmp_path, max_reflections=2
    )
    assert result.ok
    assert "# fixed" in f.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_reflection_verify_gate_loops(tmp_path):
    """A passing apply that fails the verify gate triggers another round."""
    f = tmp_path / "a.py"
    f.write_text(SAMPLE, encoding="utf-8")
    first = _sr("a.py", "    return message", "    return mesage")  # typo
    fix = _sr("a.py", "    return mesage", "    return message")

    state = {"verified": False}

    async def reprompt(_correction: str) -> str:
        return fix

    async def verify(_result):
        if not state["verified"]:
            state["verified"] = True
            return "NameError: 'mesage' is not defined"
        return None

    result = await apply_with_reflection(
        first, reprompt, root=tmp_path, max_reflections=2, verify=verify
    )
    assert result.ok
    assert "return message" in f.read_text(encoding="utf-8")
