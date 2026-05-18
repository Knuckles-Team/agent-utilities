import os
import ast


def check_file_for_stubs(filepath):
    try:
        with open(filepath, "r") as f:
            content = f.read()
        tree = ast.parse(content)

        stubs = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Check if body consists only of pass, ..., or docstring
                is_stub = True
                for expr in node.body:
                    if isinstance(expr, ast.Pass):
                        continue
                    if isinstance(expr, ast.Expr) and isinstance(
                        expr.value, (ast.Str, ast.Constant)
                    ):  # Docstring
                        continue
                    if isinstance(expr, ast.Expr) and isinstance(
                        expr.value, ast.Ellipsis
                    ):
                        continue
                    if (
                        isinstance(expr, ast.Raise)
                        and isinstance(expr.exc, ast.Name)
                        and expr.exc.id == "NotImplementedError"
                    ):
                        continue
                    if (
                        isinstance(expr, ast.Raise)
                        and isinstance(expr.exc, ast.Call)
                        and getattr(expr.exc.func, "id", "") == "NotImplementedError"
                    ):
                        continue
                    is_stub = False
                    break
                if is_stub:
                    stubs.append(node.name)
        return stubs
    except Exception as e:
        return []


stub_report = {}
for root, _, files in os.walk("agent_utilities"):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            stubs = check_file_for_stubs(filepath)
            if stubs:
                stub_report[filepath] = stubs

for fp, stubs in stub_report.items():
    print(f"{fp}: {len(stubs)} stubs -> {stubs[:5]}")
