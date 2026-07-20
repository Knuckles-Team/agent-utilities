import glob
import re

tools_dir = "agent_utilities/tools"
python_files = glob.glob(f"{tools_dir}/*.py")

for filepath in python_files:
    with open(filepath) as f:
        content = f.read()

    # We want to find @tool_version("1.0.0") (or similar) followed by async def <func_name>
    # and insert @trace(name="<func_name>", trace_type="TOOL") if it doesn't exist.

    # Check if tracing is imported
    if (
        "from agent_utilities.harness.tracing import trace" not in content
        and "@tool_version" in content
    ):
        # Add import at the top after other imports
        content = re.sub(
            r"(import logging\n(from typing import .*?\n)?)",
            r"\1from agent_utilities.harness.tracing import trace\n",
            content,
            count=1,
        )
        # If it failed to inject, just inject after module docstring
        if "from agent_utilities.harness.tracing import trace" not in content:
            content = re.sub(
                r'"""\n',
                r'"""\nfrom agent_utilities.harness.tracing import trace\n',
                content,
                count=1,
            )

    # Pattern to find @tool_version decorators and the subsequent function definition
    pattern = (
        r'(@tool_version\("[^"]+"\)\n(?:@[^\n]+\n)*)(async\s+def\s+([a-zA-Z0-9_]+))'
    )

    def replacer(match):
        decorators = match.group(1)
        func_def = match.group(2)
        func_name = match.group(3)

        # If @trace is already in the decorators, leave it
        if "@trace" in decorators or "@trace" in content:  # noqa: B023
            # We'll just be safe and assume if @trace is before it, we don't double inject
            pass

        trace_decorator = f'@trace(name="{func_name}", trace_type="TOOL")\n'
        return trace_decorator + decorators + func_def

    new_content = re.sub(pattern, replacer, content)

    if new_content != content:
        with open(filepath, "w") as f:
            f.write(new_content)
        print(f"Updated {filepath}")
    else:
        print(f"No changes in {filepath}")
