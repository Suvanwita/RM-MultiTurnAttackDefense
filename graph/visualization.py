"""
graph/visualization.py
=======================
Visualization utilities for the Intent Evolution Graph.

Provides:
- Mermaid diagram generation (for artifacts and Markdown rendering)
- Matplotlib PNG rendering (saved to results directory)
- Text/ASCII graph summary
"""

from __future__ import annotations

import os
from typing import Optional
import networkx as nx

from graph.intent_graph import IntentEvolutionGraph, SUSPICIOUS_EDGE_TYPES


class GraphVisualizer:
    """Exports and visualizes Intent Evolution Graphs."""

    @staticmethod
    def to_mermaid(graph: IntentEvolutionGraph, title: str = "Intent Evolution Graph") -> str:
        """Converts the graph to a Mermaid flowchart for markdown rendering."""
        lines = ["flowchart TD"]
        lines.append(f"    subgraph Conversation[\"{title}\"]")

        # 1. Add Turn and Intent Nodes
        for turn_node in graph.turn_nodes:
            t_data = graph.graph.nodes.get(turn_node, {})
            t_id = t_data.get("turn_id", 0)
            text_preview = t_data.get("raw_text", "")[:35].replace('"', "'")
            if len(t_data.get("raw_text", "")) > 35:
                text_preview += "..."

            intent_node = f"intent_{t_id}"
            i_data = graph.graph.nodes.get(intent_node, {})
            intent_desc = i_data.get("description", "")[:30].replace('"', "'")

            lines.append(f"        {turn_node}[\"Turn {t_id}: {text_preview}\"]")
            lines.append(f"        {intent_node}([\"Intent {t_id}: {intent_desc}\"])")
            lines.append(f"        {turn_node} --> {intent_node}")

        # 2. Add Cross-Turn Evolution Edges
        for u, v, data in graph.graph.edges(data=True):
            if u.startswith("turn_") and v.startswith("turn_"):
                etype = data.get("edge_type", "")
                conf = data.get("confidence", 1.0)
                if etype == "FOLLOWS":
                    continue  # Keep diagram uncluttered

                label = f"{etype} ({conf:.2f})"
                if etype in SUSPICIOUS_EDGE_TYPES:
                    lines.append(f"        {u} == \"{label}\" ==> {v}")
                else:
                    lines.append(f"        {u} -. \"{label}\" .-> {v}")

        lines.append("    end")
        lines.append("    classDef suspicious fill:#ffdddd,stroke:#cc0000,stroke-width:2px;")
        lines.append("    classDef benign fill:#ddffdd,stroke:#00aa00,stroke-width:1px;")
        return "\n".join(lines)

    @staticmethod
    def render_to_png(graph: IntentEvolutionGraph, output_path: str, title: str = "Intent Evolution Graph") -> bool:
        """Renders the graph to a PNG image file using matplotlib and networkx."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[GraphVisualizer] matplotlib not available; skipping PNG export.")
            return False

        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            sub = graph.get_evolution_subgraph()
            if sub.number_of_nodes() == 0:
                return False

            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            pos = {}
            # Layout turns chronologically on an x-axis progression
            for idx, turn_node in enumerate(graph.turn_nodes):
                pos[turn_node] = (float(idx) * 2.5, 0.0)

            node_colors = []
            for n in sub.nodes():
                data = sub.nodes[n]
                t_id = data.get("turn_id", 1)
                intent = graph.get_turn_intent(t_id)
                if intent and intent.has_high_risk:
                    node_colors.append("#ff6b6b")  # red
                else:
                    node_colors.append("#51cf66")  # green

            nx.draw_networkx_nodes(sub, pos, ax=ax, node_color=node_colors, node_size=1600, alpha=0.9)

            labels = {n: f"Turn {sub.nodes[n].get('turn_id', n)}" for n in sub.nodes()}
            nx.draw_networkx_labels(sub, pos, labels=labels, ax=ax, font_size=9, font_weight="bold")

            # Draw edges with distinct colors based on type
            for u, v, data in sub.edges(data=True):
                etype = data.get("edge_type", "")
                conf = data.get("confidence", 1.0)
                if etype in SUSPICIOUS_EDGE_TYPES:
                    ecolor = "#d90429"
                    style = "solid"
                    rad = 0.25
                    width = 2.2
                elif etype == "DEPENDS_ON":
                    ecolor = "#3a86ff"
                    style = "dashed"
                    rad = -0.2
                    width = 1.6
                else:
                    ecolor = "#6c757d"
                    style = "dotted"
                    rad = 0.1
                    width = 1.0

                nx.draw_networkx_edges(
                    sub,
                    pos,
                    edgelist=[(u, v)],
                    ax=ax,
                    edge_color=ecolor,
                    style=style,
                    connectionstyle=f"arc3,rad={rad}",
                    arrowsize=14,
                    width=width,
                )

            ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close(fig)
            return True
        except Exception as e:
            print(f"[GraphVisualizer] Error rendering PNG: {e}")
            return False
