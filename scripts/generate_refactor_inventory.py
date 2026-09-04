#!/usr/bin/env python3
"""Generate the RF-00 refactor inventory corpus from EG's native ``index_repository``.

Contract: ``plans/refactor/evidence/INVENTORY-TOOLING-DESIGN.md`` (``rf00-symbol-v1``).

**How the engine is reached.** ``Method::IndexRepository`` is dispatched over the
wire in EG's ``src/server/dispatch.rs``, but the work it performs is the pure,
non-mutating, idempotent ``eg_compute::parser::resolve::index_repository``. This
driver shells out to a release build of EG's
``crates/eg-compute/examples/index_repository_json.rs``, which calls that exact
function. No engine process, no transport, and no wheel is involved — deliberately,
because the eg wheel is resolved elsewhere by FILENAME version and can silently
serve a stale engine. ``--parser-binary`` is recorded verbatim in the manifest along
with its sha256, so the report can prove which binary produced the corpus.

**Universe.** ``git ls-files -z`` only — never a filesystem walk. A walk would pull
in ``target-isolated/`` build output.

**Resolution scope.** One ``index_repository`` call per repository, whole repository
in one batch: "the batch IS the resolution scope", so a partitioned run would leave
intra-repo calls unbound.

Usage::

    generate_refactor_inventory.py \
        --repo agent-utilities=/home/apps/workspace/agent-packages/agent-utilities \
        --repo epistemic-graph=/home/apps/workspace/agent-packages/epistemic-graph \
        --parser-binary /var/tmp/l9/eg-parser-target/release/examples/index_repository_json \
        --out /var/tmp/l9/index-pass-20260904
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import tomllib
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA_SYMBOL = "rf00-symbol-v1"
SCHEMA_RUN = "rf00-run-v1"
SOURCE_EXTENSIONS = (".py", ".pyi", ".rs")

# Fields the projection must carry for every symbol row. Absent evidence is an
# explicit null / unknown-Fact, never a guess.
REQUIRED_FIELDS = [
    "schema",
    "symbol_id",
    "repository",
    "path",
    "qualified_symbol",
    "symbol_kind",
    "language",
    "range",
    "ast_hash",
    "current_owner",
    "public_or_internal",
    "publicness_evidence",
    "generated",
    "generated_reason",
    "inbound_consumers",
    "outbound_dependencies",
    "entrypoints",
    "tests",
    "data_reads",
    "data_writes",
    "side_effects",
    "identity_and_policy",
    "configuration",
    "errors",
    "telemetry",
    "bug_ids",
    "scanner_findings",
    "target_owner",
    "disposition",
    "target_slice",
    "acceptance_evidence",
    "closure_commit",
    "evidence_status",
]

# The twelve+ behavioural fields the program tracks as empty. A field is listed
# here with the provider that can populate it, or None when nothing in
# ``IndexResult`` can. Honesty matters more than coverage: do not move a field out
# of ``UNAVAILABLE`` without an actual provider.
NATIVE_PROVIDERS = {
    "inbound_consumers": "eg-native-index/v1:calls-edges-reversed",
    "outbound_dependencies": "eg-native-index/v1:calls+depends_on-edges",
    "tests": "eg-native-index/v1:calls-edges-from-test-symbols",
}
UNAVAILABLE = {
    "entrypoints": "needs Cargo target metadata / Python packaging entry points; "
    "IndexResult carries neither",
    "data_reads": "no I/O model in IndexResult",
    "data_writes": "no I/O model in IndexResult",
    "side_effects": "no effect model in IndexResult",
    "identity_and_policy": "no policy/tenant model in IndexResult",
    "configuration": "no config-key model in IndexResult",
    "errors": "no raise/Result-error model in IndexResult",
    "telemetry": "no metric/span model in IndexResult",
    "bug_ids": "ledger evidence, not AST",
    "scanner_findings": "scanner evidence, not AST",
}

UNKNOWN_FACT = {"status": "unknown", "items": []}


# ─────────────────────────── deterministic primitives ───────────────────────────


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}:{sha256_hex(b'\x00'.join(p.encode('utf-8') for p in parts))}"


def jline(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# ────────────────────────────── universe discovery ──────────────────────────────


def git(root: Path, *args: str) -> str:
    """Run git in ``root`` with the ambient hook env stripped.

    A git hook exports ``GIT_DIR``/``GIT_INDEX_FILE``/``GIT_WORK_TREE`` into every
    subprocess, which makes ``git -C <dir> ls-files`` report paths relative to the
    ambient repo instead of ``<dir>`` — an empty or wrong universe, silently.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.setdefault("PATH", os.environ.get("PATH", ""))
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout


def tracked_sources(root: Path) -> list[str]:
    out = git(root, "ls-files", "-z")
    paths = [p for p in out.split("\0") if p]
    return sorted(nfc(p) for p in paths if p.endswith(SOURCE_EXTENSIONS))


# ───────────────────────────── module qualification ─────────────────────────────
#
# The parser stamps `qualified_symbol` as the LEXICAL ancestor chain only — the
# file/module/crate prefix is not in the AST. The driver composes it here, because
# only the driver knows the repository layout.


def python_module(root: Path, rel: str) -> str:
    """Dotted module path: walk up while ``__init__.py`` exists to find the root."""
    p = Path(rel)
    parts = list(p.parts)
    stem = p.stem  # drops .py / .pyi
    dirs = parts[:-1]
    # Deepest package root: the highest ancestor dir that is still a package.
    first_pkg = len(dirs)
    for i in range(len(dirs) - 1, -1, -1):
        if (root / Path(*dirs[: i + 1]) / "__init__.py").exists():
            first_pkg = i
        else:
            break
    mod_parts = dirs[first_pkg:]
    if stem != "__init__":
        mod_parts = [*mod_parts, stem]
    return ".".join(mod_parts)


def _cargo_name(cargo_toml: Path) -> str | None:
    try:
        data = tomllib.loads(cargo_toml.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    name = data.get("package", {}).get("name")
    return name.replace("-", "_") if isinstance(name, str) else None


# ``tests/``, ``benches/`` and ``examples/`` files are their own Cargo target, so they
# are qualified as ``<crate>::<target-kind>::<stem>`` rather than as crate modules.
RUST_TARGET_DIRS = ("tests", "benches", "examples")
#: Stems that ARE their enclosing module -- they contribute no segment of their own.
RUST_ROOT_STEMS = ("lib", "main", "mod")


def _rust_crate_dir(root: Path, p: Path) -> Path | None:
    """The nearest ancestor directory of ``p`` holding a ``Cargo.toml``."""
    for anc in [p.parent, *p.parents]:
        if (root / anc / "Cargo.toml").exists():
            return anc
    return None


def _rust_crate_name(root: Path, crate_dir: Path, cargo_cache: dict[str, str | None]) -> str | None:
    """The crate name for ``crate_dir``, read once and memoized per directory."""
    key = str(crate_dir)
    if key not in cargo_cache:
        cargo_cache[key] = _cargo_name(root / crate_dir / "Cargo.toml")
    return cargo_cache[key]


def _rust_target_path(crate: str, parts: list[str]) -> str | None:
    """``<crate>::<target-kind>::<stem>`` for a non-library target, else ``None``."""
    if parts and parts[0] in RUST_TARGET_DIRS:
        return "::".join([crate, parts[0], Path(parts[-1]).stem])
    return None


def _rust_mod_parts(parts: list[str]) -> list[str]:
    """Module segments for a library path: the directories under ``src/``, plus the
    stem unless the stem is itself the module (``lib``/``main``/``mod``)."""
    if parts and parts[0] == "src":
        parts = parts[1:]
    stem = Path(parts[-1]).stem if parts else ""
    mod_parts = parts[:-1]
    if stem in RUST_ROOT_STEMS:
        return mod_parts
    return [*mod_parts, stem]


def rust_module(root: Path, rel: str, cargo_cache: dict[str, str | None]) -> str:
    """``crate::mod::path`` from the nearest ``Cargo.toml`` plus the path under ``src/``.

    ``lib.rs``/``main.rs`` are the crate root; ``mod.rs`` is its directory;
    ``tests/``, ``benches/`` and ``examples/`` files are their own target and are
    qualified as ``<crate>::<target-kind>::<stem>``.
    """
    p = Path(rel)
    crate_dir = _rust_crate_dir(root, p)
    if crate_dir is None:
        return ""
    crate = _rust_crate_name(root, crate_dir, cargo_cache)
    if not crate:
        return ""
    parts = list(p.relative_to(crate_dir).parts)
    target = _rust_target_path(crate, parts)
    if target is not None:
        return target
    mod_parts = _rust_mod_parts(parts)
    return "::".join([crate, *mod_parts]) if mod_parts else crate


def module_prefix(root: Path, rel: str, language: str, cargo_cache: dict) -> tuple[str, str]:
    if language == "python":
        return python_module(root, rel), "."
    if language == "rust":
        return rust_module(root, rel, cargo_cache), "::"
    return "", "."


# ──────────────────────────────── classification ────────────────────────────────

GENERATED_MARKERS = (
    "@generated",
    "DO NOT EDIT",
    "Code generated by",
    "automatically generated",
    "autogenerated",
)


def generated_reason(root: Path, rel: str) -> str | None:
    """Checked-in generated source: a banner in the first 4 KiB, or a known path rule."""
    if "/generated/" in f"/{rel}" or rel.endswith("_pb2.py") or rel.endswith("_pb2.pyi"):
        return "path-rule"
    try:
        head = (root / rel).read_bytes()[:4096].decode("utf-8", "replace")
    except OSError:
        return None
    for marker in GENERATED_MARKERS:
        if marker in head:
            return f"banner:{marker}"
    return None


def publicness(qualified: str, name: str, language: str, path: str) -> tuple[str, list[str]]:
    """Conservative visibility. Rust visibility is not in the parser contract yet,
    so Rust is ``unknown`` rather than guessed from a name."""
    if language == "python":
        segments = [*qualified.split("."), name]
        if any(seg.startswith("_") and not seg.startswith("__") for seg in segments):
            return "internal", ["leading-underscore"]
        if name.startswith("__") and name.endswith("__"):
            return "internal", ["dunder"]
        if "/tests/" in f"/{path}" or Path(path).name.startswith("test_"):
            return "internal", ["test-module"]
        # A non-underscore name in a non-private module is a WEAK public signal;
        # the contract wants an explicit export signal (``__all__`` / re-export),
        # which is not available here.
        return "unknown", ["no-export-evidence"]
    return "unknown", ["rust-visibility-not-in-parser-contract"]


# ─────────────────────────────── the native call ────────────────────────────────


def run_index_repository(
    binary: Path, repository: str, root: Path, files: list[str], workdir: Path
) -> dict:
    request = workdir / f"{repository}-request.json"
    native = workdir / f"{repository}-native-result.json"
    request.write_text(
        json.dumps(
            {
                "repository": repository,
                "root": str(root),
                "files": files,
                "out": str(native),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    started = time.monotonic()
    proc = subprocess.run(
        [str(binary), str(request)], capture_output=True, text=True, check=False
    )
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        raise SystemExit(
            f"index_repository failed for {repository} (rc={proc.returncode}):\n{proc.stderr}"
        )
    sys.stderr.write(f"[{repository}] {proc.stderr.strip()}\n")
    return {
        "result": json.loads(native.read_bytes()),
        "native_result_path": str(native),
        "native_result_sha256": sha256_hex(native.read_bytes()),
        "wall_seconds": round(elapsed, 3),
        "stderr": proc.stderr.strip(),
    }


# ──────────────────────────────── the projection ────────────────────────────────


#: Engine counters copied verbatim into the run manifest, in this order.
COUNTER_KEYS = (
    "symbols_extracted",
    "files_parsed",
    "calls_resolved",
    "calls_unresolved",
    "calls_scope_resolved",
    "calls_type_resolved",
    "inherits_edges",
    "realizes_edges",
    "similar_edges",
    "imports_resolved",
    "imports_unresolved",
)

#: Behavioural fields with no provider in ``IndexResult`` (see ``UNAVAILABLE``).
#: Every one is emitted as an explicit unknown-Fact, never as a guess.
UNKNOWN_FACT_FIELDS = (
    "entrypoints",
    "data_reads",
    "data_writes",
    "side_effects",
    "identity_and_policy",
    "configuration",
    "errors",
    "telemetry",
    "bug_ids",
    "scanner_findings",
)

#: Program-plane fields the driver cannot observe; a later pass fills them in.
PROGRAM_FIELD_DEFAULTS = {
    "target_owner": None,
    "disposition": None,
    "target_slice": None,
    "closure_commit": None,
    "evidence_status": "structural-observed",
}


# ── file plane ──────────────────────────────────────────────────────────────


def _language_of(rel: str) -> str:
    if rel.endswith((".py", ".pyi")):
        return "python"
    if rel.endswith(".rs"):
        return "rust"
    return "other"


def _file_row(repository: str, root: Path, receipt: dict, lang: str) -> dict:
    rel = receipt["path"]
    reason = generated_reason(root, rel)
    return {
        "schema": "rf00-file-v1",
        "file_id": stable_id("file:v1", repository, rel),
        "repository": repository,
        "path": rel,
        "language": lang,
        "source_sha256": receipt["sha256"],
        "bytes": receipt["bytes"],
        "parse_status": receipt["parse_status"],
        "symbols": receipt["symbols"],
        "generated": reason is not None,
        "generated_reason": reason,
        "resolution_scope": repository,
    }


def _file_plane(
    repository: str, root: Path, receipts: list[dict]
) -> tuple[list[dict], dict[str, str]]:
    """One ``rf00-file-v1`` row per receipt, in receipt order, plus path -> language."""
    file_rows: list[dict] = []
    lang_of_path: dict[str, str] = {}
    for r in receipts:
        lang = _language_of(r["path"])
        lang_of_path[r["path"]] = lang
        file_rows.append(_file_row(repository, root, r, lang))
    return file_rows, lang_of_path


# ── symbol plane ────────────────────────────────────────────────────────────


def _int_prop(p: dict, key: str) -> int | None:
    v = p.get(key)
    return int(v) if v is not None and v.lstrip("-").isdigit() else None


#: Row field -> the parser property carrying it, in emission order. ``start_line``
#: is the one rename: the parser stamps a declaration's first line as ``line``.
RANGE_PROPERTIES = (
    ("start_byte", "start_byte"),
    ("end_byte", "end_byte"),
    ("start_line", "line"),
    ("start_col", "start_col"),
    ("end_line", "end_line"),
    ("end_col", "end_col"),
)


def _range_of(p: dict) -> dict:
    """Byte/line/column extent. A property the parser did not stamp stays ``None``."""
    return {field: _int_prop(p, key) for field, key in RANGE_PROPERTIES}


def _identity_fields(repository: str, rel: str, lang: str, prefix: str, sep: str,
                     node: dict) -> dict:
    """Identity + provenance: what the row IS and where it came from.

    ``qualified_symbol`` is the driver's composition -- the parser stamps only the
    LEXICAL ancestor chain, because the file/module/crate prefix is not in the AST.
    """
    p = node["properties"]
    lexical = p.get("qualified_symbol") or p.get("name", "")
    qualified = f"{prefix}{sep}{lexical}" if prefix else lexical
    kind = p.get("kind_detail") or p.get("symbol_type", "")
    vis, evidence = publicness(lexical, p.get("name", ""), lang, rel)
    return {
        "schema": SCHEMA_SYMBOL,
        "symbol_id": stable_id("sym:v1", repository, rel, qualified, kind),
        "repository": repository,
        "path": rel,
        "qualified_symbol": qualified,
        "lexical_qualified_symbol": lexical,
        "name": p.get("name"),
        "symbol_kind": kind,
        "language": lang,
        "range": _range_of(p),
        "ast_hash": p.get("ast_hash"),
        "native_node_id": node["node_id"],
        "current_owner": str(Path(rel).parent) if "/" in rel else repository,
        "public_or_internal": vis,
        "publicness_evidence": evidence,
    }


def _empty_behavioural_fields() -> dict:
    """The behavioural half of the schema, before back-fill: back-fillable fields as
    empty collections, unprovidable ones as explicit unknown-Facts."""
    fields: dict = {
        "generated": None,
        "generated_reason": None,
        "inbound_consumers": [],
        "outbound_dependencies": [],
        "tests": [],
        "acceptance_evidence": [],
    }
    fields.update(PROGRAM_FIELD_DEFAULTS)
    fields.update({f: dict(UNKNOWN_FACT) for f in UNKNOWN_FACT_FIELDS})
    return fields


def _symbol_row(repository: str, rel: str, lang: str, prefix: str, sep: str,
                node: dict) -> dict:
    return {
        **_identity_fields(repository, rel, lang, prefix, sep, node),
        **_empty_behavioural_fields(),
    }


def _symbol_plane(
    repository: str,
    root: Path,
    nodes: list[dict],
    lang_of_path: dict[str, str],
    cargo_cache: dict[str, str | None],
) -> tuple[list[dict], dict[str, list[dict]]]:
    """Symbol rows in node order, plus the ``node_id`` -> rows index.

    ``IndexResult.nodes`` is NOT deduplicated by node_id despite its doc comment
    (resolve.rs ``collect_result_nodes`` pushes unconditionally), and node_id is
    ``symbol:<sha256 of the declaration bytes>`` -- so byte-identical declarations
    in different files SHARE a node id. Rows are therefore keyed by
    (path, qualified_symbol, kind); node_id ambiguity is reported, not hidden.
    """
    by_node_id: dict[str, list[dict]] = defaultdict(list)
    symbol_rows: list[dict] = []
    prefix_cache: dict[str, tuple[str, str]] = {}
    for n in nodes:
        if n.get("node_type") != "SYMBOL":
            continue
        p = n["properties"]
        rel = p.get("file_path", "")
        lang = p.get("language", lang_of_path.get(rel, "other"))
        if rel not in prefix_cache:
            prefix_cache[rel] = module_prefix(root, rel, lang, cargo_cache)
        prefix, sep = prefix_cache[rel]
        row = _symbol_row(repository, rel, lang, prefix, sep, n)
        symbol_rows.append(row)
        by_node_id[n["node_id"]].append(row)
    return symbol_rows, by_node_id


def _apply_generated(symbol_rows: list[dict], file_rows: list[dict]) -> None:
    """Push each file's generated verdict down onto its symbols."""
    gen_by_path = {f["path"]: (f["generated"], f["generated_reason"]) for f in file_rows}
    for row in symbol_rows:
        g, reason = gen_by_path.get(row["path"], (False, None))
        row["generated"] = g
        row["generated_reason"] = reason


# ── edge plane + behavioural back-fill ──────────────────────────────────────


class BehaviourIndex:
    """The behavioural evidence accumulated while walking the edge plane.

    ``calls`` gives inbound/outbound and -- when the caller is a test symbol --
    ``tests``. ``inherits``/``realizes`` are structural dependencies in both
    directions. ``depends_on`` is file-level and is kept apart from the symbol
    indexes because it back-fills by PATH, not by symbol id.
    """

    def __init__(self) -> None:
        self.inbound: dict[str, set[str]] = defaultdict(set)
        self.outbound: dict[str, set[str]] = defaultdict(set)
        self.tests_of: dict[str, set[str]] = defaultdict(set)
        self.file_deps: dict[str, set[str]] = defaultdict(set)

    def record_call(self, s: dict, t: dict) -> None:
        self.inbound[t["symbol_id"]].add(s["symbol_id"])
        self.outbound[s["symbol_id"]].add(t["symbol_id"])
        if _is_test(s):
            self.tests_of[t["symbol_id"]].add(s["symbol_id"])

    def record_hierarchy(self, s: dict, t: dict) -> None:
        self.outbound[s["symbol_id"]].add(t["symbol_id"])
        self.inbound[t["symbol_id"]].add(s["symbol_id"])

    def record(self, etype: str, s: dict, t: dict) -> None:
        recorder = EDGE_BEHAVIOUR.get(etype)
        if recorder is not None:
            recorder(self, s, t)

    def record_file_dependency(self, src: str, tgt: str) -> None:
        self.file_deps[src].add(tgt)


#: Edge type -> the accumulation it drives. An unlisted type contributes a row but
#: no behavioural evidence.
EDGE_BEHAVIOUR = {
    "calls": BehaviourIndex.record_call,
    "inherits": BehaviourIndex.record_hierarchy,
    "realizes": BehaviourIndex.record_hierarchy,
}


def _file_depends_row(repository: str, e: dict, src: str, tgt: str) -> dict:
    return {
        "schema": "rf00-edge-v1",
        "edge_kind": "file_depends_on",
        "edge_id": stable_id("edge:v1", "file_depends_on", src, tgt,
                             e["properties"].get("module", "")),
        "repository": repository,
        "source_file": src,
        "target_file": tgt,
        "module": e["properties"].get("module"),
        "resolution_status": "resolved",
    }


def _symbol_edge_row(repository: str, etype: str, e: dict, s: dict, t: dict,
                     ambiguous: bool) -> dict:
    return {
        "schema": "rf00-edge-v1",
        "edge_kind": etype,
        "edge_id": stable_id(
            "edge:v1", etype, s["symbol_id"], t["symbol_id"],
            e["properties"].get("name", "")
        ),
        "repository": repository,
        "source_symbol_id": s["symbol_id"],
        "target_symbol_id": t["symbol_id"],
        "site_key": e["properties"].get("name"),
        "strategy": e["properties"].get("strategy"),
        "confidence": e["properties"].get("confidence"),
        "score": e["properties"].get("score"),
        "resolution_status": "resolved",
        "identity_ambiguous": ambiguous,
    }


def _symbol_edge_rows(repository: str, etype: str, e: dict, srcs: list[dict],
                      tgts: list[dict], ambiguous: bool,
                      behaviour: BehaviourIndex) -> list[dict]:
    """The cross product of candidate endpoints -- one row per (src, tgt) pair."""
    rows = []
    for s in srcs:
        for t in tgts:
            rows.append(_symbol_edge_row(repository, etype, e, s, t, ambiguous))
            behaviour.record(etype, s, t)
    return rows


def _edge_plane(
    repository: str, edges: list[dict], by_node_id: dict[str, list[dict]]
) -> tuple[list[dict], BehaviourIndex, int]:
    """Edge rows in edge order, the behavioural indexes, and the ambiguity count."""
    edge_rows: list[dict] = []
    behaviour = BehaviourIndex()
    ambiguous_edges = 0
    for e in edges:
        etype = e["edge_type"]
        if etype == "IMPLEMENTS":
            continue
        if etype == "depends_on":
            src = e["source"].removeprefix("file:")
            tgt = e["target"].removeprefix("file:")
            behaviour.record_file_dependency(src, tgt)
            edge_rows.append(_file_depends_row(repository, e, src, tgt))
            continue
        srcs = by_node_id.get(e["source"], [])
        tgts = by_node_id.get(e["target"], [])
        ambiguous = len(srcs) > 1 or len(tgts) > 1
        if ambiguous:
            ambiguous_edges += 1
        edge_rows.extend(
            _symbol_edge_rows(repository, etype, e, srcs, tgts, ambiguous, behaviour)
        )
    return edge_rows, behaviour, ambiguous_edges


def _apply_behaviour(symbol_rows: list[dict], behaviour: BehaviourIndex) -> None:
    """Back-fill the three natively-provided behavioural fields onto every symbol."""
    for row in symbol_rows:
        sid = row["symbol_id"]
        row["inbound_consumers"] = sorted(behaviour.inbound.get(sid, ()))
        row["outbound_dependencies"] = sorted(
            set(behaviour.outbound.get(sid, ()))
            | {f"file:{d}" for d in behaviour.file_deps.get(row["path"], ())}
        )
        row["tests"] = sorted(behaviour.tests_of.get(sid, ()))


# ── output ──────────────────────────────────────────────────────────────────


def _write_streams(
    out_dir: Path, repository: str, planes: tuple[tuple[str, list[dict]], ...]
) -> dict[str, dict]:
    written = {}
    for name, rows in planes:
        path = out_dir / f"{repository}.{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(jline(r) + "\n")
        data = path.read_bytes()
        written[name] = {
            "path": str(path),
            "rows": len(rows),
            "bytes": len(data),
            "sha256": sha256_hex(data),
        }
    return written


def _repo_provenance(root: Path, result: dict) -> dict:
    """Where the corpus came from: the tree state and the engine's own counters."""
    return {
        "root": str(root),
        "head": git(root, "rev-parse", "HEAD").strip(),
        "dirty_paths": len(
            [l for l in git(root, "status", "--porcelain=v1").splitlines() if l]
        ),
        "files_requested": result["files_requested"],
        "files_read": result["files_read"],
        "elapsed_index_ms": result["elapsed_index_ms"],
        "elapsed_total_ms": result["elapsed_total_ms"],
        "counters": {k: result["index"][k] for k in COUNTER_KEYS},
    }


def _identity_stats(
    file_rows: list[dict], by_node_id: dict[str, list[dict]], ambiguous_edges: int
) -> dict:
    """How much node_id ambiguity the run carried -- reported, never hidden."""
    return {
        "parse_status": dict(Counter(f["parse_status"].split(":")[0] for f in file_rows)),
        "distinct_native_node_ids": len(by_node_id),
        "colliding_native_node_ids": sum(1 for v in by_node_id.values() if len(v) > 1),
        "identity_ambiguous_edges": ambiguous_edges,
    }


def project(repository: str, root: Path, native: dict, out_dir: Path) -> dict:
    """Project one native ``IndexResult`` onto the file, symbol and edge planes,
    write the four JSONL streams, and return this repository's manifest entry."""
    result = native["result"]
    index = result["index"]
    cargo_cache: dict[str, str | None] = {}

    file_rows, lang_of_path = _file_plane(repository, root, result["receipts"])
    symbol_rows, by_node_id = _symbol_plane(
        repository, root, index["nodes"], lang_of_path, cargo_cache
    )
    _apply_generated(symbol_rows, file_rows)
    edge_rows, behaviour, ambiguous_edges = _edge_plane(
        repository, index["edges"], by_node_id
    )
    _apply_behaviour(symbol_rows, behaviour)

    written = _write_streams(
        out_dir,
        repository,
        (
            ("files", file_rows),
            ("symbols", symbol_rows),
            ("edges", edge_rows),
            ("deletions", []),
        ),
    )
    return {
        "repository": repository,
        **_repo_provenance(root, result),
        "wall_seconds": native["wall_seconds"],
        "native_result_sha256": native["native_result_sha256"],
        "native_result_path": native["native_result_path"],
        **_identity_stats(file_rows, by_node_id, ambiguous_edges),
        "streams": written,
        "field_population": _field_population(symbol_rows),
    }


def _is_test(row: dict) -> bool:
    """Test-symbol evidence.

    Name/path heuristics alone miss Rust's dominant idiom — a ``#[cfg(test)] mod
    tests`` inline in the implementation file, whose functions carry descriptive
    (non ``test_``-prefixed) names in a source file that is not under ``tests/``.
    The parser's lexical chain makes that module visible, which is the single
    largest concrete gain from `qualified_symbol` in this corpus.
    """
    name = row.get("name") or ""
    path = row["path"]
    lexical = row.get("lexical_qualified_symbol") or ""
    segments = lexical.replace("::", ".").split(".")[:-1]
    return (
        name.startswith("test")
        or Path(path).name.startswith("test_")
        or "/tests/" in f"/{path}"
        or any(seg in ("tests", "test") for seg in segments)
    )


def _populated(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        if "status" in value:
            return value["status"] != "unknown"
        return any(v is not None for v in value.values())
    if isinstance(value, (list, str)):
        return len(value) > 0
    return True


def _field_population(rows: list[dict]) -> dict[str, int]:
    counts = {f: 0 for f in REQUIRED_FIELDS}
    for r in rows:
        for f in REQUIRED_FIELDS:
            if _populated(r.get(f)):
                counts[f] += 1
    return counts


# ──────────────────────────────────── main ──────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="repository to index; repeatable",
    )
    ap.add_argument("--parser-binary", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    binary: Path = args.parser_binary
    if not binary.is_file():
        raise SystemExit(f"parser binary not found: {binary}")

    started = time.monotonic()
    repos = []
    for spec in args.repo:
        name, _, path = spec.partition("=")
        root = Path(path).resolve()
        files = tracked_sources(root)
        sys.stderr.write(f"[{name}] {len(files)} tracked source files\n")
        native = run_index_repository(binary, name, root, files, out_dir)
        repos.append(project(name, root, native, out_dir))

    manifest = {
        "schema": SCHEMA_RUN,
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256_hex(Path(__file__).read_bytes()),
        "parser_binary": str(binary.resolve()),
        "parser_binary_sha256": sha256_hex(binary.read_bytes()),
        "engine_entrypoint": "eg_compute::parser::resolve::index_repository "
        "(the body of Method::IndexRepository)",
        "source_policy": "git ls-files -z, extensions " + ",".join(SOURCE_EXTENSIONS),
        "required_fields": REQUIRED_FIELDS,
        "native_providers": NATIVE_PROVIDERS,
        "unavailable_fields": UNAVAILABLE,
        "repositories": repos,
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    mpath = out_dir / "run.json"
    mpath.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    sys.stderr.write(f"manifest: {mpath}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
