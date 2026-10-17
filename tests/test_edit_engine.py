"""Multi-format edit application with graduated matching and reflection."""

from __future__ import annotations

import pytest

from agent_utilities.harness.edit_engine import (
    apply_edits,
    apply_with_reflection,
    parse_edits,
)

SAMPLE = 'def greet(name):\n    message = "hello " + name\n    return message\n'


def _sr(path: str, search: str, replace: str) -> str:
    return f"{path}\n<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


@pytest.fixture
def sample(tmp_path):
    f = tmp_path / "a.py"
    f.write_text(SAMPLE, encoding="utf-8")
    return f


@pytest.mark.parametrize(
    ("search", "replace", "strategy", "expected"),
    [
        (
            '    message = "hello " + name',
            '    message = f"hello {name}"',
            "exact",
            'f"hello {name}"',
        ),
        (
            'message = "hello " + name',
            'message = f"hi {name}"',
            "leading-whitespace",
            '    message = f"hi {name}"',
        ),
        (
            '    message = "helo " + name',
            '    message = "HELLO " + name',
            "closest-window",
            '"HELLO "',
        ),
        ("", "# trailing comment", "append", "# trailing comment\n"),
    ],
)
def test_match_strategies(sample, search, replace, strategy, expected):
    result = apply_edits(
        parse_edits(_sr("a.py", search, replace).replace("SEARCH\n\n", "SEARCH\n")),
        root=sample.parent,
    )
    assert result.ok
    assert result.outcomes[0].strategy == strategy
    assert expected in sample.read_text(encoding="utf-8")


def test_unmatched_edit_reports_and_leaves_file(sample):
    result = apply_edits(
        parse_edits(_sr("a.py", "unrelated content", "x = 1")), root=sample.parent
    )
    assert not result.ok
    assert result.failures[0].path == "a.py"
    assert "did not match" in result.failures[0].reason
    assert sample.read_text(encoding="utf-8") == SAMPLE


def test_creates_file_and_composes_edits(sample):
    text = (
        "new.py\n<<<<<<< SEARCH\n=======\nprint('hi')\n>>>>>>> REPLACE\n"
        + _sr("a.py", "def greet(name):", "def greet(name, loud=False):")
        + _sr(
            "a.py",
            "    return message",
            "    return message.upper() if loud else message",
        )
    )
    assert apply_edits(parse_edits(text), root=sample.parent).ok
    assert (sample.parent / "new.py").read_text(
        encoding="utf-8"
    ).strip() == "print('hi')"
    out = sample.read_text(encoding="utf-8")
    assert "loud=False" in out and "message.upper()" in out


def test_unified_diff(sample):
    diff = (
        "--- a/a.py\n+++ b/a.py\n@@ -1,3 +1,3 @@\n def greet(name):\n"
        '-    message = "hello " + name\n+    message = "HI " + name\n     return message\n'
    )
    edits = parse_edits(diff, fmt="auto")
    assert edits[0].fmt == "unified-diff"
    assert apply_edits(edits, root=sample.parent).ok
    assert '"HI "' in sample.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("a.py\n<<<<<<< SEARCH\nnever closed\n", "Unterminated SEARCH"),
        ("<<<<<<< SEARCH\nfoo\n=======\nbar\n>>>>>>> REPLACE\n", "missing a filename"),
    ],
)
def test_malformed_input_raises(text, message):
    with pytest.raises(ValueError, match=message):
        parse_edits(text)


@pytest.mark.parametrize(
    ("first", "needle"),
    [
        (_sr("a.py", "nonexistent search text", "x = 1"), "could not be applied"),
        ("a.py\n<<<<<<< SEARCH\nunterminated", "malformed"),
    ],
)
async def test_reflection_reprompts_until_applied(sample, first, needle):
    seen: list[str] = []

    async def reprompt(correction: str) -> str:
        seen.append(correction.lower())
        return _sr("a.py", "    return message", "    return message  # done")

    result = await apply_with_reflection(
        first, reprompt, root=sample.parent, max_reflections=2
    )
    assert result.ok
    assert len(seen) == 1 and needle in seen[0]
    assert "# done" in sample.read_text(encoding="utf-8")


async def test_reflection_verify_gate_loops(sample):
    verdicts = iter(["NameError: 'mesage' is not defined", None])

    async def reprompt(_correction: str) -> str:
        return _sr("a.py", "    return mesage", "    return message")

    async def verify(_result):
        return next(verdicts)

    result = await apply_with_reflection(
        _sr("a.py", "    return message", "    return mesage"),
        reprompt,
        root=sample.parent,
        max_reflections=2,
        verify=verify,
    )
    assert result.ok
    assert "return message" in sample.read_text(encoding="utf-8")
