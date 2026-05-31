import argparse
import ast
import os
import sys
import re

# List of keywords for comments and docstrings that signify deferred/todo work
TODO_KEYWORDS = [
    "TODO",
    "FIXME",
    "STUB",
    "WORK DEFERRED",
    "FUTURE WORK",
    "FUTURE ENHANCEMENT"
]

def check_file_for_stubs(filepath):
    """
    Scans a python file for:
    1. Functions, async functions, and classes that are stubs (only contain pass, ellipsis, docstrings, or NotImplementedError).
    2. Any raising of NotImplementedError anywhere in the file.
    3. Comments/docstrings containing TODO, FIXME, STUB, work deferred, future work, future enhancement.
    """
    findings = []
    
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        content = "".join(lines)
    except Exception as e:
        return [{
            "type": "READ_ERROR",
            "line": 0,
            "message": f"Error reading file: {e}"
        }]

    # 1. AST Analysis
    try:
        tree = ast.parse(content)
        file_basename = os.path.basename(filepath).lower()
        is_interface_file = "interface" in file_basename or "protocol" in file_basename
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Smart Exclusions:
                # 1. Skip exceptions and ABC base classes
                if isinstance(node, ast.ClassDef):
                    is_exception = any(
                        isinstance(base, ast.Name) and (
                            "Error" in base.id or "Exception" in base.id or base.id == "ABC"
                        ) for base in node.bases
                    ) or "Error" in node.name or "Exception" in node.name
                    if is_exception:
                        continue
                
                # 2. Skip abstract methods or interface files
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    is_abstract = False
                    for dec in node.decorator_list:
                        if isinstance(dec, ast.Name) and "abstract" in dec.id:
                            is_abstract = True
                            break
                        if isinstance(dec, ast.Attribute) and "abstract" in dec.attr:
                            is_abstract = True
                            break
                    if is_abstract or is_interface_file:
                        continue
                
                # Check if body consists only of pass, docstring, ellipsis, or NotImplementedError
                is_stub = True
                if not node.body:
                    is_stub = True
                else:
                    for expr in node.body:
                        # docstring or ellipsis
                        if isinstance(expr, ast.Expr):
                            val = expr.value
                            is_str_or_bytes = False
                            if isinstance(val, ast.Constant):
                                if val.value is Ellipsis:
                                    continue
                                if isinstance(val.value, (str, bytes)):
                                    is_str_or_bytes = True
                            else:
                                # Legacy Python compatibility for ast.Str, ast.Bytes, ast.Ellipsis
                                val_class_name = type(val).__name__
                                if val_class_name in ("Str", "Bytes", "Ellipsis"):
                                    is_str_or_bytes = True
                                    if val_class_name == "Ellipsis":
                                        continue
                            if is_str_or_bytes:
                                continue
                            is_stub = False
                            break
                        # pass
                        elif isinstance(expr, ast.Pass):
                            continue
                        # raise NotImplementedError
                        elif isinstance(expr, ast.Raise):
                            if isinstance(expr.exc, ast.Name) and expr.exc.id == "NotImplementedError":
                                continue
                            if isinstance(expr.exc, ast.Call) and isinstance(expr.exc.func, ast.Name) and expr.exc.func.id == "NotImplementedError":
                                continue
                            is_stub = False
                            break
                        else:
                            is_stub = False
                            break
                
                if is_stub:
                    node_type = "Class" if isinstance(node, ast.ClassDef) else "Function"
                    findings.append({
                        "type": "AST_STUB",
                        "line": node.lineno,
                        "message": f"{node_type} '{node.name}' has no implementation (is a stub)."
                    })
            
            # Check for any raise NotImplementedError in the AST (even inside a non-stub body)
            elif isinstance(node, ast.Raise):
                is_nie = False
                if isinstance(node.exc, ast.Name) and node.exc.id == "NotImplementedError":
                    is_nie = True
                elif isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name) and node.exc.func.id == "NotImplementedError":
                    is_nie = True
                
                if is_nie:
                    findings.append({
                        "type": "NOT_IMPLEMENTED_ERROR",
                        "line": node.lineno,
                        "message": "Raises NotImplementedError representing incomplete work."
                    })
    except SyntaxError as e:
        findings.append({
            "type": "SYNTAX_ERROR",
            "line": e.lineno or 0,
            "message": f"SyntaxError during parsing: {e}"
        })
    except Exception as e:
        findings.append({
            "type": "PARSING_ERROR",
            "line": 0,
            "message": f"Unexpected parsing error: {e}"
        })

    # 2. Text/Comment Analysis (TODO, FIXME, etc.)
    for i, line in enumerate(lines, 1):
        clean_line = line.strip()
        if "#" in clean_line:
            comment_part = clean_line.split("#", 1)[1]
            for kw in TODO_KEYWORDS:
                if re.search(r'\b' + re.escape(kw) + r'\b', comment_part, re.IGNORECASE):
                    findings.append({
                        "type": "TODO_COMMENT",
                        "line": i,
                        "message": f"Found '{kw}' comment: {clean_line}"
                    })
        
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
    parser = argparse.ArgumentParser(description="Harden stub/TODO scanner for pre-commit verification.")
    parser.add_argument("files", nargs="*", help="Specific files to scan. If none, scans the project recursively.")
    parser.add_argument("--exclude", nargs="*", default=[], help="Directories/files to exclude from recursive scan.")
    args = parser.parse_args()

    files_to_scan = []
    default_excludes = {".venv", ".git", ".mypy_cache", ".pytest_cache", "build", "dist", "node_modules", "__pycache__"}
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
            dirs[:] = [d for d in dirs if d not in excludes and d != "workspace" and not d.startswith(".")]
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
        print(f"STUB & TODO VERIFICATION FAILED: Found {total_violations} active stub/TODO items!")
        print("=" * 80)
        for fp, findings in report.items():
            print(f"\nFile: {fp}")
            for f in findings:
                print(f"  [Line {f['line']}] [{f['type']}] - {f['message']}")
        print("\n" + "=" * 80)
        print("Please implement all stubs, remove NotImplementedErrors, and complete pending TODOs.")
        print("=" * 80 + "\n")
        sys.exit(1)
    else:
        print("STUB & TODO VERIFICATION PASSED: No active stubs or deferred work items found.")
        sys.exit(0)


if __name__ == "__main__":
    main()
