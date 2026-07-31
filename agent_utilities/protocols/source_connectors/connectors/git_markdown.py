from __future__ import annotations

"""Git + markdown source connector — revision-scoped content, diff-driven change feed.

CONCEPT:AU-ECO.connector.git-markdown-revision-connector — Git-Markdown Revision Connector

**Preset vs. new native connector (the judgement call this module makes).** AGENTS.md
is explicit that a new *external* source is a declarative ``mcp_tool`` preset, never a
new connector module — but it just as explicitly reserves native connectors for
**zero-infra defaults** (filesystem, sqlite: "things that must work with nothing
deployed") because those substrates have no server/protocol/auth for a preset to
describe. A local git working tree is exactly that: no server, no credentials, no
network — a directory on disk, like ``filesystem`` already covers. So the question is
narrower than "preset or connector" — it's "does git need its own connector, or is it
already covered by ``filesystem``?"

It needs its own connector because **git has real revision semantics that a
filesystem walk cannot express**, and revision is the operator's specific ask
("the git commit SHA is the natural ``source_version``"):

* ``FilesystemConnector.poll`` waters on ``st_mtime_ns`` (or a caller-supplied
  content-hash snapshot file). Neither is a *revision* — mtime is reset by a fresh
  checkout, and there is no source-native "as of commit X" concept to cite.
* Content here is read via ``git show <sha>:<path>`` — bound to the exact git object
  at that revision — never the live, possibly-dirty working tree
  ``FilesystemConnector`` reads. That is what makes ``source_version`` a real,
  independently-verifiable fact rather than an opaque watermark.
* A **git diff between two revisions is a natural incremental change feed**
  (``git diff --name-status <old>..<new>``) — add/modify/copy/rename all map onto
  ``upsert``, delete maps onto a tombstone ``ChangeEnvelope``. This is a materially
  different poll algorithm from mtime-comparison, not a config knob on top of it.

So: one new, small, zero-infra native connector (this module), reusing
``FilesystemConnector``'s conventions (namespace hashing, fail-closed ACL defaults,
governed-envelope emission after ``graphql_document``'s precedent) rather than its
code — the walk algorithm (``git ls-tree``/``git show``/``git diff`` at a revision)
is different enough from a live directory walk that inheriting would be more
confusing than two small, independent implementations.

**What this module does NOT do.** It does not implement the domain-pack framework,
the declarative frontmatter/heading/table -> graph-facts mapping DSL, candidate-claim
extraction, or entity resolution (tracks 2-5 of the universal-ingestion program,
``reports/program/universal-ingestion.md``) — those are sibling lanes and had not
landed any code as of this connector (their worktrees exist at the same commit as
``main``). ``GIT_MARKDOWN_PRESETS`` below is therefore the *narrowest* thing that
lets "a corpus is configured, not coded" hold today: a config preset (root/subdir/
corpus slug/access), exactly mirroring ``FilesystemConnector.FILESYSTEM_PRESETS``'s
own precedent — not a mapping DSL. Frontmatter is read as opaque document text (it
flows into the same ``Document`` body every other connector's markdown does); only a
human-readable ``title`` is sniffed from it (mirroring what ``ReaderConnector``
already does for HTML), never turned into typed graph facts.

**Governed envelopes.** Like ``graphql_document.py``, this connector populates
``last_envelopes`` with real :class:`ChangeEnvelope` objects (not just
:class:`SourceDocument` metadata) so a delete is a genuine tombstone
(``operation="delete"``), and every upsert's ``source_version`` is the commit SHA
that content was read at. Its own governed-envelope identity
(:meth:`GitMarkdownConnector._revision_record_id`) is a *different* node from the
``DocumentProcessor``-managed ``Document``/``Chunk`` nodes engine.py's generic
connector-ingestion adaptor creates for the same file — NOT, as originally
designed, the same node under engine.py's own ``sha256(portable_uri)[:24]``
formula. That 24-hex-char (96-bit) digest isn't recognized by
``envelope_ingest``'s opaque-identifier exemption (only a bare/namespaced
32-or-64-hex digest qualifies), so embedding it in an identity-checked
``ChangeEnvelope`` field falls through to a full privacy-pattern scan whose
case-insensitive IBAN pattern intermittently false-positives on ordinary hex —
reproducibly hit on 3 of this repo's own real ``docs/pillars`` files while
building this connector (see ``D-GM-3``,
``reports/deferred/lane-git-markdown.md``, and this module's test suite).
``_revision_record_id`` instead uses a full 32-hex-char digest, which always
satisfies that exemption. The two nodes are joined by the ``relpath``/
``corpus``/``git_commit`` properties both carry, not by a shared primary key —
an ordinary graph-modeling join, not a defect.

**Evidence spine.** :meth:`GitMarkdownConnector.build_artifact` builds a real
``Artifact``/``Fragment`` graph slice (the pinned evidence-spine contract,
``knowledge_graph.ingestion.evidence_spine``, cherry-picked from
``feat/evidence-spine`` commit ``961698b8``) over the connector's own verbatim,
revision-scoped content: the artifact is keyed to the FILE (stable across
revisions), its ``content_hash``/its fragments' ``content_hash``es carry the git
revision. :func:`fragment_markdown` is a small ATX-heading + paragraph
fragmenter — enough to prove the stable-address/content-hash split holds across
a real git revision; not the evidence-spine lane's own (still in-flight, not
pinned here) full markdown fragmenter.
"""

import hashlib
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.ingestion.evidence_spine import Fragment

from agent_utilities.knowledge_graph.ingestion.change_envelope import ChangeEnvelope
from agent_utilities.models.company_brain import DataClassification

from ..base import (
    CheckpointedBatch,
    ConnectorCheckpoint,
    ExternalAccess,
    LoadConnector,
    PermSyncConnector,
    PollConnector,
    SlimDocument,
    SourceDocument,
)
from ..registry import register_source

__all__ = ["GitMarkdownConnector", "GitMarkdownError", "GIT_MARKDOWN_PRESETS"]

_GIT_TIMEOUT_S = 30.0
_DEFAULT_EXTENSIONS = frozenset({".md", ".markdown"})

# ── Domain-pack presets (CONCEPT:AU-ECO.connector.git-markdown-revision-connector) ──
#
# Two structurally different markdown corpora proving the connector (and, on top of
# it, epistemic-graph) generalizes across conventions rather than special-casing one:
#
# * ``au-pillars``: docs/pillars/**/*.md — no YAML frontmatter; a
#   "``# KG-2.1-Title_With_Underscores``" heading, a "``**Pillar:** N — Name ·
#   **Status:** live``" metadata line, and ``## What`` / ``## How / Wiring`` /
#   ``## Tests`` sections with inline ``CONCEPT:`` markers.
# * ``au-skills``: agent_utilities/skills/*/SKILL.md — real YAML frontmatter
#   (``name``/``skill_type``/``description``), then a title heading and free-form
#   prose/table sections (``## Action reference``, …). Restricted to the exact
#   ``SKILL.md`` filename so the corpus stays uniformly frontmatter-bearing (a
#   skill directory's ``references/*.md`` siblings carry no frontmatter at all and
#   would blur the contrast this preset exists to demonstrate).
#
# Mirrors ``FilesystemConnector.FILESYSTEM_PRESETS``: a preset supplies everything
# EXCEPT ``root`` (the local checkout path), which the caller always provides.
GIT_MARKDOWN_PRESETS: dict[str, dict[str, object]] = {
    "au-pillars": {
        "subdir": "docs/pillars",
        "corpus": "au-pillars",
        "doc_type": "pillar-doc",
        "public": True,
    },
    "au-skills": {
        "subdir": "agent_utilities/skills",
        "corpus": "au-skills",
        "doc_type": "skill-doc",
        "filenames": ["SKILL.md"],
        "public": True,
    },
}


class GitMarkdownError(RuntimeError):
    """Raised when the configured root is not a usable git working tree."""


def _run_git(repo_root: Path, *args: str) -> str:
    """Run ``git -C repo_root <args>`` and return stdout; never a shell, never raises past this."""
    try:
        result = subprocess.run(  # noqa: S603 — fixed argv, no shell, local git only
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=True,
        )
    except FileNotFoundError as exc:
        raise GitMarkdownError("git executable not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitMarkdownError(f"git {' '.join(args)} timed out") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise GitMarkdownError(f"git {' '.join(args)} failed: {stderr}") from exc
    return result.stdout


def _resolve_repo_root(path: Path) -> Path:
    """Resolve the git working-tree root containing ``path`` (fails closed)."""
    try:
        out = _run_git(path, "rev-parse", "--show-toplevel")
    except GitMarkdownError as exc:
        raise GitMarkdownError(f"{path} is not inside a git working tree") from exc
    return Path(out.strip()).resolve()


def _head_sha(repo_root: Path) -> str:
    return _run_git(repo_root, "rev-parse", "HEAD").strip()


def _ls_tree(repo_root: Path, sha: str, subdir: str) -> list[str]:
    """Tracked file paths (relative to repo root) at ``sha``, restricted to ``subdir``."""
    args = ["ls-tree", "-r", "--name-only", sha, "--"]
    if subdir:
        args.append(subdir)
    out = _run_git(repo_root, *args)
    return [line for line in out.splitlines() if line]


def _diff_name_status(
    repo_root: Path, old_sha: str, new_sha: str, subdir: str
) -> list[tuple[str, str, str | None]]:
    """``(status, path, old_path)`` triples for every change between two revisions.

    ``old_path`` is set only for a rename/copy (``status`` starting with ``R``/``C``),
    and is the path the record is filed under BEFORE the change (used to tombstone
    the prior id when a tracked markdown file is renamed).
    """
    args = [
        "diff",
        "--name-status",
        "-M",
        old_sha,
        new_sha,
        "--",
    ]
    if subdir:
        args.append(subdir)
    out = _run_git(repo_root, *args)
    changes: list[tuple[str, str, str | None]] = []
    for line in out.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        status = parts[0]
        if status[0] in ("R", "C") and len(parts) == 3:
            changes.append((status, parts[2], parts[1]))
        elif len(parts) == 2:
            changes.append((status, parts[1], None))
    return changes


def _show(repo_root: Path, sha: str, relpath: str) -> str | None:
    """``git show <sha>:<relpath>`` — the file's exact content at that revision."""
    try:
        return _run_git(repo_root, "show", f"{sha}:{relpath}")
    except GitMarkdownError:
        return None


def _privacy_safe_text(text: str) -> str:
    """Redact PII-shaped substrings before markdown body crosses the persistence
    boundary (mirrors ``source_sync._privacy_safe``'s exact rationale).

    Real internal documentation legitimately discusses local paths, IPs, and
    example secrets/tokens in prose (this repo's own ``docs/pillars/`` is full
    of them). The native ``ApplyChangeEnvelope`` commit REJECTS such inline
    text outright rather than redacting it, so a single flagged paragraph would
    otherwise fail an entire file's ingest with the unactionable message
    "persistence privacy policy rejected inline text" (discovered ingesting
    this repo's own real corpora — see the connector's CHANGELOG/test suite).
    Redacting here, exactly like ``source_sync`` already does for fleet-supplied
    tool/skill descriptions, degrades a flagged paragraph to a redacted one
    instead of dropping the whole document.
    """
    from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

    safe, _report = PersistencePrivacyGuard().sanitize_text(text)
    return safe


def _extract_title(text: str, relpath: str) -> str:
    """Sniff a human-readable title: YAML frontmatter ``name``/``title``, else the
    first Markdown heading, else the filename (mirrors ``ReaderConnector``'s HTML
    title-sniffing — no frontmatter-to-graph-facts mapping, just a display label).
    """
    stripped = text.lstrip()
    if stripped.startswith("---"):
        end = stripped.find("\n---", 3)
        if end != -1:
            frontmatter = stripped[3:end]
            for line in frontmatter.splitlines():
                line = line.strip()
                for key in ("title:", "name:"):
                    if line.lower().startswith(key):
                        value = line[len(key) :].strip().strip("'\"")
                        if value:
                            return value
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return Path(relpath).stem


_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

#: Fragmenter version tag stamped into ``Artifact.provenance["fragmenter"]`` —
#: bump if the segmentation algorithm below changes so a re-ingest under a new
#: version is distinguishable from a genuine content edit.
GIT_MARKDOWN_FRAGMENTER = "git_markdown.headings_paragraphs_v1"


def fragment_markdown(artifact_id: str, text: str) -> list[Fragment]:
    """Fragment VERBATIM markdown into ``heading``/``paragraph`` Fragments.

    CONCEPT:AU-KG.ingest.stable-fragment-address — built on the evidence-spine's
    pinned contract (``Fragment.at()``): each ATX heading (``#`` .. ``######``)
    becomes a ``heading`` fragment addressed by a slug of its own text (stable
    across a heading's body/sibling edits elsewhere in the file); each paragraph
    becomes a ``paragraph`` fragment addressed by its ordinal among paragraph
    siblings under the nearest enclosing heading (or the document root).
    Headings nest by ATX level, so a fragment's structural path mirrors the
    document's own section hierarchy.

    Deliberately minimal — headings + paragraphs only, no tables/lists/code
    blocks — the smallest fragmenter that lets the incremental proof show a
    typo-fix changing exactly one ``content_hash`` while every other fragment's
    ``fragment_id`` (and, for untouched ones, ``content_hash`` too) stays
    byte-identical. A fuller fragmenter is the evidence-spine lane's own
    follow-on wiring (not pinned here — see this module's docstring).
    """
    from agent_utilities.knowledge_graph.ingestion.evidence_spine import Fragment

    fragments: list[Any] = []
    heading_stack: list[tuple[int, tuple[str, ...], str]] = []
    ordinals: dict[tuple[tuple[str, ...], str], int] = {}
    sequence = 0
    paragraph_buf: list[str] = []

    def parent() -> tuple[tuple[str, ...], str | None]:
        if heading_stack:
            return heading_stack[-1][1], heading_stack[-1][2]
        return (), None

    def next_ordinal(parent_path: tuple[str, ...], kind: str) -> int:
        key = (parent_path, kind)
        ordinal = ordinals.get(key, 0)
        ordinals[key] = ordinal + 1
        return ordinal

    def flush_paragraph() -> None:
        nonlocal sequence
        if not paragraph_buf:
            return
        para_text = "\n".join(paragraph_buf).strip()
        paragraph_buf.clear()
        if not para_text:
            return
        parent_path, parent_id = parent()
        frag = Fragment.at(
            artifact_id=artifact_id,
            kind="paragraph",
            parent_path=parent_path,
            text=para_text,
            ordinal=next_ordinal(parent_path, "paragraph"),
            parent_fragment_id=parent_id,
            sequence=sequence,
        )
        sequence += 1
        fragments.append(frag)

    for line in text.splitlines():
        heading_match = _ATX_HEADING_RE.match(line)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            label = heading_match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            parent_path, parent_id = parent()
            frag = Fragment.at(
                artifact_id=artifact_id,
                kind="heading",
                parent_path=parent_path,
                text=label,
                label=label,
                ordinal=next_ordinal(parent_path, "heading"),
                parent_fragment_id=parent_id,
                sequence=sequence,
            )
            sequence += 1
            fragments.append(frag)
            heading_stack.append((level, frag.path, frag.fragment_id))
            continue
        if not line.strip():
            flush_paragraph()
            continue
        paragraph_buf.append(line)
    flush_paragraph()
    return fragments


@register_source("git_markdown")
class GitMarkdownConnector(LoadConnector, PollConnector, PermSyncConnector):
    """Ingest markdown files from a git working tree, revision-scoped.

    CONCEPT:AU-ECO.connector.git-markdown-revision-connector.

    Config:
        root: A path inside the git working tree to ingest from (required).
        preset: Optional ``GIT_MARKDOWN_PRESETS`` key supplying ``subdir``/
            ``corpus``/``doc_type``/``filenames``/``public`` defaults.
        subdir: Restrict to this subdirectory (relative to the repo root).
        extensions: Override the default ``{.md, .markdown}`` allow-list.
        filenames: Optional exact-basename allow-list (e.g. ``["SKILL.md"]``).
        public: Mark every document world-readable (default False — fail-closed).
        group_ids: Non-personal access groups when not public.
        corpus: A logical slug identifying this domain pack (stamped into
            document metadata as ``corpus`` for cross-corpus lineage).
        doc_type: Optional document type hint.
        source_id: Optional logical namespace (hashed before persistence).
    """

    provider = "Git Markdown Repository"

    def configure(
        self,
        *,
        root: str = "",
        preset: str = "",
        subdir: str = "",
        extensions: list[str] | None = None,
        filenames: list[str] | None = None,
        public: bool = False,
        group_ids: list[str] | None = None,
        corpus: str = "",
        doc_type: str = "",
        source_id: str = "",
        **_: object,
    ) -> None:
        if preset:
            base = GIT_MARKDOWN_PRESETS.get(preset)
            if base is None:
                raise ValueError(
                    f"Unknown git_markdown preset {preset!r}. "
                    f"Available: {', '.join(sorted(GIT_MARKDOWN_PRESETS)) or '(none)'}"
                )
            merged = {
                **base,
                **{
                    k: v
                    for k, v in self._config.items()
                    if k != "preset" and v not in ("", None)
                },
            }
            merged.pop("preset", None)
            self._config = merged
            self.configure(**merged)  # type: ignore[arg-type]
            return
        if not root:
            raise ValueError("GitMarkdownConnector requires a 'root' directory")
        base_path = Path(root).expanduser().resolve(strict=False)
        self.repo_root = _resolve_repo_root(base_path)
        # subdir is relative to the repo root, never to `root` — a caller may point
        # `root` anywhere inside the tree; the corpus boundary is always repo-relative.
        if subdir:
            resolved_subdir = (self.repo_root / subdir).resolve(strict=False)
            if not resolved_subdir.is_relative_to(self.repo_root):
                raise ValueError(
                    "GitMarkdownConnector subdir must stay within the repo"
                )
            self.subdir = str(resolved_subdir.relative_to(self.repo_root).as_posix())
        else:
            base_rel = base_path.resolve(strict=False)
            self.subdir = (
                ""
                if base_rel == self.repo_root
                else str(base_rel.relative_to(self.repo_root).as_posix())
            )
        namespace_material = source_id or str(self.repo_root)
        self.source_namespace = hashlib.sha256(
            namespace_material.encode("utf-8", errors="surrogatepass")
        ).hexdigest()[:16]
        self.extensions = (
            {e.lower() for e in extensions} if extensions else set(_DEFAULT_EXTENSIONS)
        )
        self.filenames = {str(f) for f in (filenames or [])}
        self.public = public
        self.group_ids = [
            value
            for value in (str(group).strip() for group in (group_ids or []))
            if value
        ]
        self.corpus = corpus or self.source_namespace
        self.doc_type_override = doc_type
        self.last_envelopes: list[ChangeEnvelope] = []

    def health_check(self) -> bool:
        try:
            _head_sha(self.repo_root)
        except GitMarkdownError:
            return False
        return True

    # -- path filtering ------------------------------------------------------

    def _matches(self, relpath: str) -> bool:
        p = Path(relpath)
        if p.suffix.lower() not in self.extensions:
            return False
        if self.filenames and p.name not in self.filenames:
            return False
        return True

    def _tracked_paths(self, sha: str) -> list[str]:
        return sorted(
            path
            for path in _ls_tree(self.repo_root, sha, self.subdir)
            if self._matches(path)
        )

    # -- identity / access ----------------------------------------------------

    def _portable_uri(self, relpath: str) -> str:
        return f"git-markdown://{self.source_namespace}/{quote(relpath, safe='/')}"

    def _object_key(self, uri: str) -> str:
        # Matches engine.py's `_ingest_connector` object_key formula (a 24-hex
        # truncation), so a caller cross-referencing the DocumentProcessor-owned
        # ``doc:git_markdown:<key>`` node (created independently by the engine's
        # generic connector adaptor, never by this connector) can derive its id.
        return hashlib.sha256(uri.encode("utf-8")).hexdigest()[:24]

    def _document_node_id(self, relpath: str) -> str:
        """The DocumentProcessor-owned node id a caller can cross-reference.

        NOT used as this connector's OWN governed-envelope identity (see
        :meth:`_revision_record_id`) — a 24-hex-char (96-bit) digest doesn't
        match ``envelope_ingest``'s opaque-identifier exemption (which only
        recognizes a bare or namespaced 32/64-hex digest), so embedding it in an
        identity-checked ``ChangeEnvelope`` field falls through to a full
        privacy-pattern scan of the raw digest text. That scan is a coincidence
        away from a false positive: the case-insensitive IBAN pattern
        (``[A-Z]{2}\\d{2}(?:[A-Z0-9]){11,30}``) matches whenever a namespaced id's
        digest happens to start with two letters then two digits — true for
        roughly 1 in 20 sha256 digests, and reproducibly hit 3 of this repo's
        own 96 real ``docs/pillars`` files in this connector's own live-engine
        proof (``tests/integration/knowledge_graph/
        test_git_markdown_domain_packs_live_engine.py``). Reported as D-GM-3;
        not fixed here — widening the opaque-identifier regex is shared,
        security-adjacent code affecting every connector's identity safety, out
        of proportion for a connector-adding lane.
        """
        return f"doc:git_markdown:{self._object_key(self._portable_uri(relpath))}"

    def _revision_record_id(self, relpath: str) -> str:
        """This connector's OWN governed-envelope identity — always opaque.

        A full 32-hex-char (128-bit) digest reliably matches
        ``envelope_ingest``'s namespaced opaque-identifier exemption (exactly
        32 or 64 hex chars after the last colon), so it is never subjected to
        the full-text privacy scan D-GM-3 describes — unlike the truncated
        24-hex :meth:`_document_node_id`. Deliberately a DIFFERENT node from
        the DocumentProcessor-owned Document (see that method's docstring);
        the two are joined by the shared ``relpath``/``corpus``/``git_commit``
        properties both carry, not by a shared primary key.
        """
        digest = hashlib.sha256(self._portable_uri(relpath).encode("utf-8")).hexdigest()
        return f"gitmd-revision:{digest[:32]}"

    def _access(self) -> ExternalAccess:
        if self.public:
            return ExternalAccess.public()
        if self.group_ids:
            return ExternalAccess(is_public=False, group_ids=list(self.group_ids))
        return ExternalAccess.quarantined()

    # -- document + envelope construction -------------------------------------

    def _to_document(self, sha: str, relpath: str) -> SourceDocument | None:
        text = _show(self.repo_root, sha, relpath)
        if text is None or not text.strip():
            return None
        portable_uri = self._portable_uri(relpath)
        doc_type = self.doc_type_override or "markdown"
        title = _extract_title(text, relpath)
        safe_text = _privacy_safe_text(text)
        if not safe_text.strip():
            return None
        return SourceDocument(
            id=portable_uri,
            source_uri=portable_uri,
            title=title,
            text=safe_text,
            doc_type=doc_type,
            metadata={
                "corpus": self.corpus,
                "relpath": relpath,
                "git_commit": sha,
                "git_repo": self.source_namespace,
            },
            external_access=self._access(),
            updated_at=sha,
        )

    def _upsert_envelope(self, sha: str, relpath: str) -> ChangeEnvelope:
        node_id = self._revision_record_id(relpath)
        access = self._access()
        classification = (
            DataClassification.PUBLIC
            if access.is_public
            else DataClassification.INTERNAL
        )
        return ChangeEnvelope(
            connector="git_markdown",
            operation="upsert",
            source_instance=self.corpus,
            source_object_id=node_id,
            source_version=sha,
            typed_payload={
                "id": node_id,
                "type": "GitMarkdownRevision",
                "corpus": self.corpus,
                "relpath": relpath,
                "source_kind": "git_markdown",
            },
            source_acl=access,
            classification=classification,
            provenance={"git_commit": sha, "corpus": self.corpus, "relpath": relpath},
        )

    def _delete_envelope(self, sha: str, relpath: str) -> ChangeEnvelope:
        node_id = self._revision_record_id(relpath)
        return ChangeEnvelope(
            connector="git_markdown",
            operation="delete",
            source_instance=self.corpus,
            source_object_id=node_id,
            source_version=sha,
            provenance={"git_commit": sha, "corpus": self.corpus, "relpath": relpath},
        )

    # -- LoadConnector ---------------------------------------------------------

    def _full_batch(self, sha: str) -> list[SourceDocument]:
        documents: list[SourceDocument] = []
        envelopes: list[ChangeEnvelope] = []
        for relpath in self._tracked_paths(sha):
            doc = self._to_document(sha, relpath)
            if doc is None:
                continue
            documents.append(doc)
            envelopes.append(self._upsert_envelope(sha, relpath))
        self.last_envelopes = envelopes
        return documents

    def load(self) -> Iterator[SourceDocument]:
        sha = _head_sha(self.repo_root)
        yield from self._full_batch(sha)

    # -- PollConnector -----------------------------------------------------

    def poll(self, checkpoint: ConnectorCheckpoint | None = None) -> CheckpointedBatch:
        """Diff-driven incremental batch (CONCEPT:AU-ECO.connector.git-markdown-revision-connector).

        First poll (no prior watermark): a full snapshot at HEAD, exactly like
        :meth:`load`. Unchanged HEAD: zero documents, zero envelopes — a re-run
        over an untouched repo is provably a no-op. Otherwise: ``git diff
        --name-status`` between the prior and current HEAD is the change feed —
        added/modified/copied/renamed-to paths become ``upsert`` documents +
        envelopes (content read at the NEW sha); deleted/renamed-from paths
        become tombstone ``delete`` envelopes only (no document to embed).
        """
        new_sha = _head_sha(self.repo_root)
        prior_sha = checkpoint.watermark if checkpoint and checkpoint.watermark else ""

        if not prior_sha:
            snapshot_documents = self._full_batch(new_sha)
            return CheckpointedBatch(
                documents=snapshot_documents,
                checkpoint=ConnectorCheckpoint(has_more=False, watermark=new_sha),
            )
        if prior_sha == new_sha:
            self.last_envelopes = []
            return CheckpointedBatch(
                documents=[],
                checkpoint=ConnectorCheckpoint(has_more=False, watermark=new_sha),
            )

        changes = _diff_name_status(self.repo_root, prior_sha, new_sha, self.subdir)
        documents: list[SourceDocument] = []
        envelopes: list[ChangeEnvelope] = []
        for status, path, old_path in changes:
            if old_path is not None and self._matches(old_path) and old_path != path:
                # A tracked markdown file moved — tombstone the id it used to be
                # filed under before (maybe) upserting the new one below.
                envelopes.append(self._delete_envelope(new_sha, old_path))
            if status.startswith("D"):
                if self._matches(path):
                    envelopes.append(self._delete_envelope(new_sha, path))
                continue
            if not self._matches(path):
                continue
            doc = self._to_document(new_sha, path)
            if doc is None:
                continue
            documents.append(doc)
            envelopes.append(self._upsert_envelope(new_sha, path))

        self.last_envelopes = envelopes
        return CheckpointedBatch(
            documents=documents,
            checkpoint=ConnectorCheckpoint(has_more=False, watermark=new_sha),
        )

    # -- SlimConnector / PermSyncConnector ---------------------------------

    def slim(self) -> Iterator[SlimDocument]:
        sha = _head_sha(self.repo_root)
        for relpath in self._tracked_paths(sha):
            yield SlimDocument(
                id=self._portable_uri(relpath),
                source_uri=self._portable_uri(relpath),
                external_access=self._access(),
            )

    def fetch_access(self) -> Iterator[tuple[str, ExternalAccess]]:
        sha = _head_sha(self.repo_root)
        for relpath in self._tracked_paths(sha):
            yield self._portable_uri(relpath), self._access()

    # -- evidence spine (CONCEPT:AU-KG.ingest.evidence-spine-artifact) --------

    def build_artifact(self, sha: str, relpath: str) -> Any | None:
        """Build the addressable ``Artifact`` + ``Fragment``s for one file.

        Not yet called by the generic ``ContentType.CONNECTOR`` adaptor (that
        wiring is the evidence-spine lane's own follow-on work, deliberately
        not taken here — see the module docstring's contract-pin note); this is
        the connector-side building block a caller commits directly via
        ``ingest_graph_slice`` (``Artifact.to_graph_slice()``) — see
        ``tests/integration/knowledge_graph/
        test_git_markdown_domain_packs_live_engine.py`` for the live proof.

        The artifact is keyed to the FILE (``_revision_record_id``, a stable
        function of the portable URI — never of content), and its
        ``content_hash``/its fragments' ``content_hash``es carry the git
        revision. Re-building the SAME file at a LATER commit therefore updates
        the same artifact rather than forking a new one.
        """
        from agent_utilities.knowledge_graph.ingestion.evidence_spine import (
            Artifact,
            artifact_id_for,
        )

        text = _show(self.repo_root, sha, relpath)
        if text is None or not text.strip():
            return None
        title = _extract_title(text, relpath)
        safe_text = _privacy_safe_text(text)
        if not safe_text.strip():
            return None
        envelope = self._upsert_envelope(sha, relpath)
        # Same 3 inputs Artifact.from_envelope will independently re-derive its
        # own artifact_id from — computed here only so the fragments below can
        # be built against the identical id ahead of the Artifact existing.
        artifact_id = artifact_id_for(
            envelope.connector, envelope.source_instance, envelope.source_object_id
        )
        fragments = fragment_markdown(artifact_id, safe_text)
        return Artifact.from_envelope(
            envelope,
            content=safe_text,
            media_type="text/markdown",
            fragments=tuple(fragments),
            title=title,
            fragmenter=GIT_MARKDOWN_FRAGMENTER,
        )
