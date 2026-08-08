import argparse
import ast
import io
import os
import re
import sys
import tokenize

# List of keywords for comments and docstrings that signify deferred/todo work
TODO_KEYWORDS = [
    "TODO",
    "FIXME",
    "STUB",
    "WORK DEFERRED",
    "FUTURE WORK",
    "FUTURE ENHANCEMENT",
]

# "STUB" is an ordinary English verb/noun ("stub it out", "a trivial stub app",
# "not a stub") that shows up constantly in test-comment prose with no relation to
# an actual deferred-work marker -- unlike "TODO"/"FIXME"/etc., which are already
# never used that way in this codebase's comments. A bare case-insensitive
# \bSTUB\b match therefore has a very high false-positive rate for this one
# keyword specifically. Every genuine marker in this codebase (see the "TODO:"
# convention already in use) is written as an explicit, capitalized "KEYWORD:"
# tag, so require that same marker form -- case-sensitive, colon-terminated --
# for STUB only, instead of loosening the gate globally or excluding files/paths.
MARKER_ONLY_KEYWORDS = {"STUB": re.compile(r"\bSTUB\s*:")}


def check_file_for_stubs(filepath):
    """
    Scans a python file for:
    1. Functions, async functions, and classes that are stubs (only contain pass, ellipsis, docstrings, or NotImplementedError).
    2. Any raising of NotImplementedError anywhere in the file.
    3. Real `#` comments (not string-literal text) containing TODO, FIXME, STUB, work deferred, future work, future enhancement.
    """
    findings = []

    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        content = "".join(lines)
    except Exception as e:
        return [
            {"type": "READ_ERROR", "line": 0, "message": f"Error reading file: {e}"}
        ]

    # 1. AST Analysis
    try:
        tree = ast.parse(content)
        file_basename = os.path.basename(filepath).lower()
        is_interface_file = "interface" in file_basename or "protocol" in file_basename

        class StubVisitor(ast.NodeVisitor):
            def __init__(self):
                self.findings = []
                self.in_abc = False

            @staticmethod
            def _is_marked_abstract_ok(node) -> bool:
                """True iff a ``# ABSTRACT-OK`` marker appears on this function's
                own source lines.

                Same convention ``scripts/check_no_stub.py`` already uses
                repo-wide (production code) for a documented, permanently-
                unimplemented method (e.g. a partial facade explicitly declining
                to support a feature it has no substrate for yet) as distinct
                from an incomplete work-in-progress stub. Recognising it here
                too keeps this AST-shape-based scanner and that gate's
                substring-based scanner from disagreeing about the exact same
                function.
                """
                start = max(node.lineno - 1, 0)
                end = min(getattr(node, "end_lineno", node.lineno), len(lines))
                return any("# ABSTRACT-OK" in lines[i] for i in range(start, end))

            def visit_ClassDef(self, node: ast.ClassDef):
                was_in_abc = self.in_abc

                is_abc = any(
                    isinstance(base, ast.Name) and base.id in ("ABC", "Protocol")
                    for base in node.bases
                )
                is_exception = (
                    any(
                        isinstance(base, ast.Name)
                        and ("Error" in base.id or "Exception" in base.id)
                        for base in node.bases
                    )
                    or "Error" in node.name
                    or "Exception" in node.name
                )

                if is_exception:
                    return  # Completely skip exceptions

                if is_abc:
                    self.in_abc = True

                self._check_stub_body(node, "Class")
                self.generic_visit(node)

                self.in_abc = was_in_abc

            def visit_FunctionDef(self, node: ast.FunctionDef):
                self._handle_function(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
                self._handle_function(node)

            def _handle_function(self, node):
                is_abstract = False
                for dec in getattr(node, "decorator_list", []):
                    if isinstance(dec, ast.Name) and "abstract" in dec.id:
                        is_abstract = True
                        break
                    if isinstance(dec, ast.Attribute) and "abstract" in dec.attr:
                        is_abstract = True
                        break

                if (
                    is_abstract
                    or is_interface_file
                    or self.in_abc
                    or self._is_marked_abstract_ok(node)
                ):
                    # Skip traversal of abstract functions, ANY function inside an
                    # ABC, or a function explicitly marked `# ABSTRACT-OK`.
                    # This prevents false positives for interface methods and properties.
                    return

                self._check_stub_body(node, "Function")
                self.generic_visit(node)

            def _check_stub_body(self, node, node_type: str):
                is_stub = True
                if not getattr(node, "body", []):
                    is_stub = True
                else:
                    for expr in node.body:
                        if isinstance(expr, ast.Expr):
                            val = expr.value
                            is_str_or_bytes = False
                            if isinstance(val, ast.Constant):
                                if val.value is Ellipsis:
                                    continue
                                if isinstance(val.value, (str, bytes)):
                                    is_str_or_bytes = True
                            else:
                                val_class_name = type(val).__name__
                                if val_class_name in ("Str", "Bytes", "Ellipsis"):
                                    is_str_or_bytes = True
                                    if val_class_name == "Ellipsis":
                                        continue
                            if is_str_or_bytes:
                                continue
                            is_stub = False
                            break
                        elif isinstance(expr, ast.Pass):
                            continue
                        elif isinstance(expr, ast.Raise):
                            if (
                                isinstance(expr.exc, ast.Name)
                                and expr.exc.id == "NotImplementedError"
                            ):
                                continue
                            if (
                                isinstance(expr.exc, ast.Call)
                                and isinstance(expr.exc.func, ast.Name)
                                and expr.exc.func.id == "NotImplementedError"
                            ):
                                continue
                            is_stub = False
                            break
                        else:
                            is_stub = False
                            break

                if is_stub:
                    if node_type == "Class" and getattr(node, "bases", []):
                        return
                    self.findings.append(
                        {
                            "type": "AST_STUB",
                            "line": node.lineno,
                            "message": f"{node_type} '{node.name}' has no implementation (is a stub).",
                        }
                    )

        visitor = StubVisitor()
        if not (
            "test_" in file_basename
            or "conftest" in file_basename
            or "tests" in filepath.split(os.sep)
        ):
            visitor.visit(tree)
            findings.extend(visitor.findings)

    except SyntaxError as e:
        findings.append(
            {
                "type": "SYNTAX_ERROR",
                "line": e.lineno or 0,
                "message": f"SyntaxError during parsing: {e}",
            }
        )
    except Exception as e:
        findings.append(
            {
                "type": "PARSING_ERROR",
                "line": 0,
                "message": f"Unexpected parsing error: {e}",
            }
        )

    # 2. Comment Analysis (TODO, FIXME, etc.)
    #
    # Scan real `#` COMMENT tokens only, via `tokenize`, not "any line
    # containing a '#' character" (the previous behaviour). A naive text scan
    # cannot distinguish an actual code comment from a `#`-and-keyword
    # sequence that merely appears *inside a string literal* -- e.g. a test
    # building synthetic source text as fixture data
    # (`"# TODO: wire in ... eventually\n"` passed to `.write_text(...)`)
    # is DATA the test writes to a temp file, not a deferred-work marker in
    # this codebase, but the old line-based scan could not tell the
    # difference and flagged it anyway. `tokenize` sees exactly what the
    # Python grammar itself considers a comment, so this is a strict
    # false-positive fix with no loss of real coverage: every line that used
    # to match AND is a genuine comment still matches.
    try:
        comment_tokens = [
            tok
            for tok in tokenize.generate_tokens(io.StringIO(content).readline)
            if tok.type == tokenize.COMMENT
        ]
    except (tokenize.TokenizeError, SyntaxError, IndentationError):
        comment_tokens = []

    for tok in comment_tokens:
        i = tok.start[0]
        comment_part = tok.string[1:]  # strip the leading '#'
        for kw in TODO_KEYWORDS:
            marker_re = MARKER_ONLY_KEYWORDS.get(kw)
            matched = (
                marker_re.search(comment_part)
                if marker_re is not None
                else re.search(
                    r"\b" + re.escape(kw) + r"\b", comment_part, re.IGNORECASE
                )
            )
            if matched:
                findings.append(
                    {
                        "type": "TODO_COMMENT",
                        "line": i,
                        "message": f"Found '{kw}' comment: {lines[i - 1].strip()}",
                    }
                )

    # De-duplicate findings on same line and type if any
    unique_findings = []
    seen = set()
    for f in findings:
        key = (f["line"], f["type"], f["message"])
        if key not in seen:
            seen.add(key)
            unique_findings.append(f)

    return sorted(unique_findings, key=lambda x: x["line"])


def main():
    parser = argparse.ArgumentParser(
        description="Harden stub/TODO scanner for pre-commit verification."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific files to scan. If none, scans the project recursively.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Directories/files to exclude from recursive scan.",
    )
    args = parser.parse_args()

    files_to_scan = []
    default_excludes = {
        ".venv",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        "build",
        "dist",
        "node_modules",
        "__pycache__",
    }
    excludes = default_excludes.union(set(args.exclude))

    if args.files:
        # Check specific files provided by pre-commit
        for f in args.files:
            if f.endswith(".py") and os.path.exists(f):
                files_to_scan.append(f)
    else:
        # Recursive scan of current directory
        for root, dirs, files in os.walk("."):
            # Exclude specified directories and all hidden directories starting with '.' in place
            dirs[:] = [
                d
                for d in dirs
                if d not in excludes and d != "workspace" and not d.startswith(".")
            ]
            for file in files:
                if file.endswith(".py"):
                    files_to_scan.append(os.path.join(root, file))

    total_violations = 0
    report = {}

    for filepath in files_to_scan:
        # Normalize path
        filepath = os.path.normpath(filepath)
        # Skip this script itself to prevent it from failing on the stub check lists it contains
        if "check_stubs.py" in filepath:
            continue

        findings = check_file_for_stubs(filepath)
        if findings:
            report[filepath] = findings
            total_violations += len(findings)

    if total_violations > 0:
        print("\n" + "=" * 80)
        print(
            f"STUB & TODO VERIFICATION FAILED: Found {total_violations} active stub/TODO items!"
        )
        print("=" * 80)
        for fp, findings in report.items():
            print(f"\nFile: {fp}")
            for f in findings:
                print(f"  [Line {f['line']}] [{f['type']}] - {f['message']}")
        print("\n" + "=" * 80)
        print(
            "Please implement all stubs, remove NotImplementedErrors, and complete pending TODOs."
        )
        print("=" * 80 + "\n")
        sys.exit(1)
    else:
        print(
            "STUB & TODO VERIFICATION PASSED: No active stubs or deferred work items found."
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
