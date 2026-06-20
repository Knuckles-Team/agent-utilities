"""Render the universal agent's Markdown reply for a specific messaging medium.

CONCEPT:ECO-4.0 / Universal-capability — the orchestrator emits ONE Markdown answer; each
messaging entrypoint only adapts how it *renders* that answer for its medium (it adds no
capability). Telegram renders a small HTML subset (``<b> <i> <u> <s> <code> <pre> <a>``) — NOT
generic Markdown — so a reply sent verbatim with ``parse_mode=HTML`` shows raw ``**bold**`` /
``## heading`` / `` `code` `` markers instead of formatting. This converts the model's standard
Markdown into that subset; anything Telegram can't represent (headings, tables, list markup)
degrades to bold/bullets rather than leaking syntax.

Kept dependency-free (pure ``re`` + stdlib ``html``) per the core dependency discipline — this
is rendering glue, not a Markdown engine. Malformed output can only ever *under*-format: the
backend falls back to plain text if Telegram rejects the HTML, so a render bug never blocks a
reply.
"""

from __future__ import annotations

import html as _html
import re

# Fenced code block: ```lang\n<body>\n``` — body is rendered verbatim (no inner Markdown).
_CODE_BLOCK_RE = re.compile(r"```[^\n`]*\n?(.*?)```", re.DOTALL)
# Inline code: `code` (single line, no nested backtick).
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
# [text](http(s)://url)
_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)")
# # .. ###### heading → bold (Telegram has no headings).
_HEADER_RE = re.compile(r"^[ \t]*#{1,6}[ \t]+(.+?)[ \t]*$", re.MULTILINE)
# **bold** or __bold__
_BOLD_RE = re.compile(r"\*\*([^*\n]+)\*\*|__([^_\n]+)__")
# ~~strike~~
_STRIKE_RE = re.compile(r"~~([^~\n]+)~~")
# *italic* or _italic_ (single marker, not the bold doubles handled above).
_ITALIC_RE = re.compile(r"(?<![*\w])[*_]([^*_\n]+)[*_](?![*\w])")
# Leading list bullets (-, *, +) → •
_BULLET_RE = re.compile(r"^[ \t]*[-*+][ \t]+", re.MULTILINE)
# A placeholder no real reply contains, used to protect code spans during escaping.
_SENTINEL = "\x00"


def markdown_to_telegram_html(text: str) -> str:
    """Convert standard Markdown to the HTML subset Telegram's ``parse_mode=HTML`` renders."""
    if not text:
        return text

    # 1) Stash code spans BEFORE escaping/markdown so their contents are rendered verbatim.
    blocks: list[str] = []
    inlines: list[str] = []

    def _stash_block(m: re.Match[str]) -> str:
        blocks.append(m.group(1))
        return f"{_SENTINEL}B{len(blocks) - 1}{_SENTINEL}"

    def _stash_inline(m: re.Match[str]) -> str:
        inlines.append(m.group(1))
        return f"{_SENTINEL}I{len(inlines) - 1}{_SENTINEL}"

    text = _CODE_BLOCK_RE.sub(_stash_block, text)
    text = _INLINE_CODE_RE.sub(_stash_inline, text)

    # 2) Escape the remaining prose so stray <, >, & never break Telegram's HTML parser.
    text = _html.escape(text, quote=False)

    # 3) Markdown markers (which survive escaping) → Telegram tags. Links first so their URL/text
    #    are already escaped from step 2 (correct inside href + body).
    text = _LINK_RE.sub(lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', text)
    text = _HEADER_RE.sub(r"<b>\1</b>", text)
    text = _BOLD_RE.sub(lambda m: f"<b>{m.group(1) or m.group(2)}</b>", text)
    text = _STRIKE_RE.sub(r"<s>\1</s>", text)
    text = _ITALIC_RE.sub(r"<i>\1</i>", text)
    text = _BULLET_RE.sub("• ", text)

    # 4) Restore code spans, escaped and wrapped in the verbatim tags.
    text = re.sub(
        rf"{_SENTINEL}B(\d+){_SENTINEL}",
        lambda m: f"<pre>{_html.escape(blocks[int(m.group(1))], quote=False)}</pre>",
        text,
    )
    text = re.sub(
        rf"{_SENTINEL}I(\d+){_SENTINEL}",
        lambda m: f"<code>{_html.escape(inlines[int(m.group(1))], quote=False)}</code>",
        text,
    )
    return text
