"""
================================================
visualization/mindmap_viz.py
Mind-map style radial graph using Plotly
================================================
"""

from typing import Dict, List, Optional, Tuple
import math

import plotly.graph_objects as go

from utils.logger import get_logger

log = get_logger(__name__)

_PALETTE = [
    "#6C63FF", "#FF6584", "#43BCCD", "#F9A826",
    "#00C49A", "#FF8C42", "#845EC2", "#D65DB1",
]


class MindMapViz:
    """
    Generates a radial mind-map visualization using Plotly.
    The paper title sits at the centre; domains radiate outwards;
    concepts/keywords are leaves.
    """

    def __init__(self) -> None:
        self._bg = "#0F0F1A"

    def build(
        self,
        title: str,
        tree: Dict,
        max_depth: int = 3,
    ) -> go.Figure:
        """
        Build a mind-map figure.

        Args:
            title:     Paper title (centre node).
            tree:      Concept hierarchy dict.
            max_depth: How many levels to visualise.

        Returns:
            Plotly Figure.
        """
        nodes, edges = [], []
        self._traverse(title, tree, 0, 0, 2 * math.pi, nodes, edges, max_depth)

        if not nodes:
            return self._empty_fig(f"Mind Map – {title[:40]}")

        # ── Edge traces ───────────────────────────────────────────────────────
        edge_x, edge_y = [], []
        for (x0, y0), (x1, y1) in edges:
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(color="#334", width=1.2),
            hoverinfo="none",
        )

        # ── Node trace ────────────────────────────────────────────────────────
        xs     = [n["x"] for n in nodes]
        ys     = [n["y"] for n in nodes]
        texts  = [n["label"] for n in nodes]
        colors = [n["color"] for n in nodes]
        sizes  = [n["size"] for n in nodes]

        node_trace = go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            text=texts,
            textposition="top center",
            textfont=dict(size=9, color="#ddd"),
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=1.5, color="#111"),
                opacity=0.92,
            ),
            hovertemplate="<b>%{text}</b><extra></extra>",
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=dict(
                text=f"Mind Map – {title[:50]}",
                font=dict(size=17, color="#6C63FF"),
            ),
            showlegend=False,
            paper_bgcolor=self._bg,
            plot_bgcolor=self._bg,
            font=dict(color="#E0E0E0"),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=10, r=10, t=60, b=10),
            height=700,
        )
        return fig

    # ──────────────────────────────────────────────────────────────────────────

    def _traverse(
        self,
        label: str,
        children: Dict,
        depth: int,
        angle_start: float,
        angle_end: float,
        nodes: List,
        edges: List,
        max_depth: int,
        parent_pos: Optional[Tuple] = None,
    ) -> Tuple[float, float]:
        """
        Recursively place nodes on concentric circles.

        Returns: (x, y) position of this node.
        """
        angle_mid = (angle_start + angle_end) / 2
        radius = depth * 2.0  # each level further out

        x = radius * math.cos(angle_mid)
        y = radius * math.sin(angle_mid)

        color = _PALETTE[depth % len(_PALETTE)]
        size  = max(8, 26 - depth * 5)

        nodes.append({"x": x, "y": y, "label": label[:25], "color": color, "size": size})
        if parent_pos:
            edges.append((parent_pos, (x, y)))

        if depth < max_depth and isinstance(children, dict) and children:
            n_children = len(children)
            arc = (angle_end - angle_start) / max(n_children, 1)
            for i, (child_label, grandchildren) in enumerate(children.items()):
                child_start = angle_start + i * arc
                child_end   = child_start + arc
                self._traverse(
                    child_label, grandchildren,
                    depth + 1, child_start, child_end,
                    nodes, edges, max_depth, (x, y),
                )

        return x, y

    @staticmethod
    def _empty_fig(title: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title=title, paper_bgcolor="#0F0F1A")
        return fig
