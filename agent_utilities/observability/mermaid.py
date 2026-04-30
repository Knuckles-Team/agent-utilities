#!/usr/bin/python
"""Mermaid Diagram Generation Utilities.

This module provides a structured way to generate Mermaid.js diagram code
from various internal data structures, ensuring consistent styling and
escaping of special characters.
"""

from __future__ import annotations

import re
from enum import Enum


class MermaidTheme(Enum):
    DEFAULT = "default"
    NEUTRAL = "neutral"
    DARK = "dark"
    FOREST = "forest"
    BASE = "base"


class MermaidBuilder:
    """Base class for building Mermaid diagrams."""

    def __init__(
        self, title: str | None = None, theme: MermaidTheme = MermaidTheme.DARK
    ):
        self.title = title
        self.theme = theme
        self.lines: list[str] = []

    def _sanitize(self, text: str) -> str:
        """Sanitize text for Mermaid labels, escaping brackets and quotes."""
        if not text:
            return ""
        # Remove markdown-style links or other complex structures that break Mermaid
        text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
        # Escape quotes
        text = text.replace('"', "&quot;")
        # Replace characters that break labels
        text = text.replace("(", "&#40;").replace(")", "&#41;")
        text = text.replace("[", "&#91;").replace("]", "&#93;")
        text = text.replace("{", "&#123;").replace("}", "&#125;")
        return text

    def render(self) -> str:
        """Render the complete Mermaid diagram as a string."""
        header = []
        if self.title:
            header.append("---")
            header.append(f"title: {self.title}")
            header.append("config:")
            header.append(f"  theme: {self.theme.value}")
            header.append("---")

        return "\n".join(header + self.lines)


class FlowchartBuilder(MermaidBuilder):
    """Builder for Mermaid Flowcharts."""

    def __init__(
        self,
        title: str | None = None,
        direction: str = "TD",
        theme: MermaidTheme = MermaidTheme.DARK,
    ):
        super().__init__(title, theme)
        self.lines.append(f"flowchart {direction}")

    def add_node(
        self,
        node_id: str,
        label: str | None = None,
        shape: str = "round",
        css_class: str | None = None,
    ):
        """Add a node to the flowchart.

        Shapes:
        - round: (label)
        - box: [label]
        - diamond: {label}
        - circle: ((label))
        - cylinder: [(label)]
        """
        label = label or node_id
        safe_label = self._sanitize(label)
        safe_id = node_id.replace("-", "_").replace(":", "_").replace(".", "_")

        if shape == "round":
            line = f'  {safe_id}("{safe_label}")'
        elif shape == "box":
            line = f'  {safe_id}["{safe_label}"]'
        elif shape == "diamond":
            line = f'  {safe_id}{{"{safe_label}"}}'
        elif shape == "circle":
            line = f'  {safe_id}(("{safe_label}"))'
        elif shape == "cylinder":
            line = f'  {safe_id}[("{safe_label}")]'
        else:
            line = f'  {safe_id}("{safe_label}")'

        self.lines.append(line)
        if css_class:
            self.lines.append(f"  class {safe_id} {css_class}")

    def add_edge(
        self,
        source: str,
        target: str,
        label: str | None = None,
        edge_type: str = "-->",
    ):
        """Add an edge between two nodes.

        Edge Types:
        - --> (arrow)
        - --- (line)
        - -.-> (dotted arrow)
        - ==> (thick arrow)
        """
        src_id = source.replace("-", "_").replace(":", "_").replace(".", "_")
        tgt_id = target.replace("-", "_").replace(":", "_").replace(".", "_")

        if label:
            safe_label = self._sanitize(label)
            self.lines.append(f'  {src_id} {edge_type} |"{safe_label}"| {tgt_id}')
        else:
            self.lines.append(f"  {src_id} {edge_type} {tgt_id}")

    def add_subgraph(self, title: str, nodes: list[str], direction: str = "TB"):
        """Group nodes into a subgraph."""
        self.lines.append(f"  subgraph {self._sanitize(title)}")
        self.lines.append(f"    direction {direction}")
        for node in nodes:
            safe_id = node.replace("-", "_").replace(":", "_").replace(".", "_")
            self.lines.append(f"    {safe_id}")
        self.lines.append("  end")


class ClassDiagramBuilder(MermaidBuilder):
    """Builder for Mermaid Class Diagrams (useful for SDD/TDD)."""

    def __init__(
        self, title: str | None = None, theme: MermaidTheme = MermaidTheme.DARK
    ):
        super().__init__(title, theme)
        self.lines.append("classDiagram")

    def add_class(
        self,
        name: str,
        attributes: list[str] | None = None,
        methods: list[str] | None = None,
        annotation: str | None = None,
    ):
        safe_name = name.replace("-", "_").replace(":", "_").replace(".", "_")
        self.lines.append(f"  class {safe_name} {{")
        if annotation:
            self.lines.append(f"    <<{annotation}>>")
        if attributes:
            for attr in attributes:
                self.lines.append(f"    {attr}")
        if methods:
            for meth in methods:
                self.lines.append(f"    {meth}")
        self.lines.append("  }")

    def add_relationship(
        self, source: str, target: str, rel_type: str, label: str | None = None
    ):
        """Add a relationship between classes.

        Rel Types:
        - <|-- (Inheritance)
        - *-- (Composition)
        - o-- (Aggregation)
        - --> (Association)
        - -- (Link)
        - .. (Dependency)
        """
        src_id = source.replace("-", "_").replace(":", "_").replace(".", "_")
        tgt_id = target.replace("-", "_").replace(":", "_").replace(".", "_")

        rel = f"{src_id} {rel_type} {tgt_id}"
        if label:
            rel += f" : {self._sanitize(label)}"
        self.lines.append(f"  {rel}")


class EntityRelationshipBuilder(MermaidBuilder):
    """Builder for Mermaid Entity Relationship diagrams."""

    def __init__(
        self, title: str | None = None, theme: MermaidTheme = MermaidTheme.DARK
    ):
        super().__init__(title, theme)
        self.lines.append("erDiagram")

    def add_entity(self, name: str, attributes: list[tuple[str, str]] | None = None):
        safe_name = name.replace("-", "_").replace(":", "_").replace(".", "_")
        if attributes:
            self.lines.append(f"  {safe_name} {{")
            for attr_type, attr_name in attributes:
                self.lines.append(f"    {attr_type} {attr_name}")
            self.lines.append("  }")
        else:
            self.lines.append(f"  {safe_name}")

    def add_relationship(
        self, source: str, target: str, rel_type: str, label: str | None = None
    ):
        """Add a relationship.

        Rel Types:
        - }|..|| (one to many)
        - ||--|| (one to one)
        - }|--|{ (many to many)
        """
        src_id = source.replace("-", "_").replace(":", "_").replace(".", "_")
        tgt_id = target.replace("-", "_").replace(":", "_").replace(".", "_")

        label_str = f' : "{self._sanitize(label)}"' if label else ""
        self.lines.append(f"  {src_id} {rel_type} {tgt_id}{label_str}")
