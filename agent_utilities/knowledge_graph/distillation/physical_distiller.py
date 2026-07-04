#!/usr/bin/python
from __future__ import annotations

"""Physical Knowledge Distillation Engine.

CONCEPT:AU-AHE.optimization.physical-distillation-engine — Physical Knowledge Distillation Engine.

Translates KG-native self-evolutionary adaptations of skills, MCP tools (descriptions,
docstrings, and input schemas), and optimized system prompts back into concrete physical
filesystem changes (Python modules, YAML/JSON configurations, and Markdown files) and
commits them to Git (CONCEPT:AU-AHE.optimization.gitops-commit-automation).
"""

import ast
import logging
import os
import re
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


class PhysicalDistillationEngine:
    """CONCEPT:AU-AHE.optimization.physical-distillation-engine — Manages translation of KG changes to the physical filesystem."""

    def __init__(self, workspace_root: str = "/home/apps/workspace") -> None:
        self.workspace_root = workspace_root

    def distill_skill(
        self,
        skill_id: str,
        new_name: str,
        new_description: str,
        skill_code_path: str,
        tags: list[str] | None = None,
        requires: list[str] | None = None,
    ) -> bool:
        """CONCEPT:AU-AHE.optimization.physical-distillation-engine — Distills updated skill properties back to its physical SKILL.md.

        Args:
            skill_id: Unique skill identifier.
            new_name: Evolved name of the skill.
            new_description: Evolved description.
            skill_code_path: Absolute or workspace-relative path to the skill.
            tags: Updated tags for frontmatter.
            requires: Updated requirements/dependencies.

        Returns:
            True if distillation was successful, False otherwise.
        """
        try:
            # Resolve physical path
            target_path = skill_code_path
            if not os.path.isabs(target_path):
                target_path = os.path.join(self.workspace_root, target_path)

            # If path is a directory, check for SKILL.md inside it
            if os.path.isdir(target_path):
                target_path = os.path.join(target_path, "SKILL.md")

            if not os.path.exists(target_path):
                logger.warning(
                    f"Skill file not found at {target_path}, creating standard template."
                )
                # Create parent directories if missing
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write("---\nname: temp\ndescription: temp\n---\n")

            with open(target_path, encoding="utf-8") as f:
                content = f.read()

            # Robust Frontmatter Parsing
            frontmatter_match = re.match(
                r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL
            )
            frontmatter_data: dict[str, Any] = {}
            if not frontmatter_match:
                body = content
            else:
                frontmatter_text = frontmatter_match.group(1)
                body = frontmatter_match.group(2)
                # Simple line-by-line parsing to avoid yaml package dependencies
                for line in frontmatter_text.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        # Clean key and value
                        k = k.strip()
                        v = v.strip()
                        # Clean quotes and brackets
                        if v.startswith("[") and v.endswith("]"):
                            # Parse list of strings
                            v = [
                                item.strip(" '\"")
                                for item in v[1:-1].split(",")
                                if item.strip()
                            ]
                        else:
                            v = v.strip(" '\"")
                        frontmatter_data[k] = v

            # Update frontmatter values
            frontmatter_data["name"] = new_name
            frontmatter_data["description"] = new_description
            if tags is not None:
                frontmatter_data["tags"] = tags
            if requires is not None:
                frontmatter_data["requires"] = requires

            # Format Frontmatter back
            new_frontmatter_lines = ["---"]
            for k, v in frontmatter_data.items():
                if isinstance(v, list):
                    v_str = ", ".join(f"'{item}'" for item in v)
                    new_frontmatter_lines.append(f"{k}: [{v_str}]")
                else:
                    new_frontmatter_lines.append(f"{k}: {v}")
            new_frontmatter_lines.append("---")
            new_frontmatter = "\n".join(new_frontmatter_lines)

            # Re-compile SKILL.md content
            new_content = f"{new_frontmatter}\n\n{body.lstrip()}"
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            logger.info(
                f"Successfully distilled skill '{skill_id}' to '{target_path}'."
            )
            return True

        except Exception as e:
            logger.error(f"Error distilling skill '{skill_id}': {e}")
            return False

    def distill_mcp_tool(
        self,
        tool_name: str,
        new_description: str,
        file_path: str,
        function_name: str,
    ) -> bool:
        """CONCEPT:AU-AHE.optimization.physical-distillation-engine — Distills updated tool description back into Python function docstring.

        Uses AST parsing to precisely locate the function and replace/inject the docstring
        without altering any surrounding code or comments.

        Args:
            tool_name: Name of the MCP tool.
            new_description: Evolved description.
            file_path: Path to Python source file containing tool definition.
            function_name: Name of the python function.

        Returns:
            True if distillation was successful, False otherwise.
        """
        try:
            target_path = file_path
            if not os.path.isabs(target_path):
                target_path = os.path.join(self.workspace_root, target_path)

            if not os.path.exists(target_path):
                logger.error(f"Source file not found: {target_path}")
                return False

            with open(target_path, encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source)

            class DocstringUpdater(ast.NodeVisitor):
                def __init__(self) -> None:
                    self.start_line: int | None = None
                    self.end_line: int | None = None
                    self.node_to_inject: ast.FunctionDef | None = None

                def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                    if node.name == function_name:
                        doc = ast.get_docstring(node)
                        if doc is not None:
                            # Locate the docstring node
                            for body_node in node.body:
                                if (
                                    isinstance(body_node, ast.Expr)
                                    and isinstance(body_node.value, ast.Constant)
                                    and isinstance(body_node.value.value, str)
                                ):
                                    self.start_line = body_node.lineno
                                    self.end_line = body_node.end_lineno
                                    break
                        else:
                            # Prepare for docstring injection
                            self.node_to_inject = node
                    self.generic_visit(node)

            updater = DocstringUpdater()
            updater.visit(tree)

            lines = source.splitlines(keepends=True)

            if updater.start_line is not None and updater.end_line is not None:
                # We found an existing docstring. Replace it.
                # Get the indentation level of the docstring start line
                orig_line = lines[updater.start_line - 1]
                indent = len(orig_line) - len(orig_line.lstrip())
                indent_str = " " * indent

                new_doc_lines = [f'"""{new_description}"""\n']
                # Replace the range (start_line - 1) to (end_line)
                lines[updater.start_line - 1 : updater.end_line] = [
                    indent_str + line for line in new_doc_lines
                ]
            elif updater.node_to_inject is not None:
                # Inject docstring as the first statement in the function body
                first_stmt = updater.node_to_inject.body[0]
                indent = first_stmt.col_offset
                indent_str = " " * indent
                lines.insert(
                    first_stmt.lineno - 1, indent_str + f'"""{new_description}"""\n'
                )
            else:
                logger.warning(
                    f"Function '{function_name}' not found in {target_path}."
                )
                return False

            with open(target_path, "w", encoding="utf-8") as f:
                f.write("".join(lines))

            logger.info(
                f"Successfully distilled MCP tool '{tool_name}' docstring in '{target_path}'."
            )
            return True

        except Exception as e:
            logger.error(f"Error distilling MCP tool '{tool_name}': {e}")
            return False

    def distill_system_prompt(self, file_path: str, new_content: str) -> bool:
        """CONCEPT:AU-AHE.optimization.physical-distillation-engine — Overwrites the concrete prompt file with optimized text.

        Args:
            file_path: Workspace-relative or absolute path to target prompt file.
            new_content: Fully optimized, compiled prompt text.

        Returns:
            True if distillation was successful, False otherwise.
        """
        try:
            target_path = file_path
            if not os.path.isabs(target_path):
                target_path = os.path.join(self.workspace_root, target_path)

            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            logger.info(
                f"Successfully distilled optimized system prompt to '{target_path}'."
            )
            return True
        except Exception as e:
            logger.error(f"Error distilling system prompt to '{file_path}': {e}")
            return False

    def commit_distilled_changes(
        self, file_paths: list[str], concept_id: str = "AU-AHE.optimization.gitops-commit-automation"
    ) -> bool:
        """CONCEPT:AU-AHE.optimization.gitops-commit-automation — GitOps Git Commit Automation.

        Stages, commits, and logs distilled adaptations using native git subprocess calls.

        Args:
            file_paths: List of absolute or relative paths to commit.
            concept_id: Concept identifier for audit trail.

        Returns:
            True if commit succeeded, False otherwise.
        """
        try:
            # Stage changed files
            for fp in file_paths:
                cmd_add = ["git", "add", fp]
                subprocess.run(
                    cmd_add, check=True, capture_output=True, cwd=self.workspace_root
                )

            # Create message and commit
            msg = f"[CONCEPT:{concept_id}] Dynamic self-evolution: Distilled KG adaptations to filesystem."
            cmd_commit = ["git", "commit", "-m", msg]
            res = subprocess.run(
                cmd_commit, check=True, capture_output=True, cwd=self.workspace_root
            )
            logger.info(f"Git commit successful: {res.stdout.decode().strip()}")
            return True
        except subprocess.CalledProcessError as e:
            # If nothing to commit, return True
            if b"nothing to commit" in e.stderr or b"no changes added" in e.stderr:
                logger.info("No modifications to commit.")
                return True
            logger.error(f"Git commit failed: {e.stderr.decode().strip()}")
            return False
        except Exception as e:
            logger.error(f"Failed to commit distilled changes: {e}")
            return False
