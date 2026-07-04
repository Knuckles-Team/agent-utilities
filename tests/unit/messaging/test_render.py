"""Markdown → Telegram-HTML rendering (CONCEPT:AU-ECO.messaging.native-backend-abstraction entrypoint rendering)."""

from __future__ import annotations

from agent_utilities.messaging.render import markdown_to_telegram_html as render


def test_inline_emphasis_and_code() -> None:
    out = render("**bold** and *italic* and `code`")
    assert out == "<b>bold</b> and <i>italic</i> and <code>code</code>"


def test_headers_become_bold_and_bullets() -> None:
    out = render("## Stacks\n- portainer\n- graph-os")
    assert "<b>Stacks</b>" in out
    assert "• portainer" in out and "• graph-os" in out
    assert "#" not in out and out.count("- ") == 0


def test_code_block_is_escaped_verbatim() -> None:
    out = render("```python\nprint(1 < 2 & 3)\n```")
    # HTML specials inside code are escaped; the body is wrapped in <pre>, no inner markdown.
    assert "<pre>" in out and "</pre>" in out
    assert "1 &lt; 2 &amp; 3" in out


def test_links_render_as_anchors() -> None:
    out = render("see [the repo](https://example.com/x?a=1)")
    assert '<a href="https://example.com/x?a=1">the repo</a>' in out


def test_plain_text_html_specials_are_escaped() -> None:
    out = render("a < b & c > d, no markdown")
    assert out == "a &lt; b &amp; c &gt; d, no markdown"


def test_empty_is_passthrough() -> None:
    assert render("") == ""


# ── The two Telegram validation cases (docs/architecture/ontology-native-classification.md §6) ──
# These capture the FORMATTING half of each case: a realistic tool-reply in Markdown must render
# to valid Telegram HTML (no raw markers leak). The classification/routing half is the manual
# runbook in §6 (needs the live KG/daemon).


def test_case1_portainer_reply_renders() -> None:
    """Case 1 — 'list my portainer stacks' → a stack listing with bold/bullets/code."""
    reply = (
        "**Your Portainer stacks:**\n"
        "- `graph-os` — running\n"
        "- `agent-utilities` — running\n"
        "- *paradedb* — stopped"
    )
    out = render(reply)
    assert "<b>Your Portainer stacks:</b>" in out
    assert "• <code>graph-os</code> — running" in out
    assert "<i>paradedb</i>" in out
    # No raw markdown markers leak through.
    assert "**" not in out and "`" not in out and "\n- " not in out


def test_case2_github_reply_renders() -> None:
    """Case 2 — 'github open issues' → a heading + bold repo + linked issue numbers."""
    reply = (
        "## Open issues — Knuckles-Team\n"
        "**agent-utilities** (2 open):\n"
        "- [#42](https://github.com/Knuckles-Team/agent-utilities/issues/42) dispatcher fix\n"
        "- #43 add tests"
    )
    out = render(reply)
    assert "<b>Open issues — Knuckles-Team</b>" in out  # header → bold
    assert "<b>agent-utilities</b> (2 open):" in out
    assert (
        '<a href="https://github.com/Knuckles-Team/agent-utilities/issues/42">#42</a>'
        in out
    )
    assert "• #43 add tests" in out
    assert "##" not in out and "](" not in out
