"""
================================================
visualization/hierarchy_viz.py
Plotly-based hierarchy visualizations:
  • Sunburst chart
  • Treemap
  • Icicle / top-down tree
================================================
"""

from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go
import plotly.express as px

from utils.logger import get_logger

log = get_logger(__name__)


class HierarchyViz:
    """Creates sunburst, treemap, and icicle charts from a concept tree."""

    def __init__(self) -> None:
        self._bg = "#0F0F1A"
        self._font_color = "#E0E0E0"

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def sunburst(self, tree: Dict, title: str = "Concept Hierarchy") -> go.Figure:
        """
        Build a Plotly Sunburst chart from a nested hierarchy dict.
        """
        ids, labels, parents, values = self._flatten_tree(tree, root="Root")
        if not ids:
            return self._empty_fig(title)

        fig = go.Figure(
            go.Sunburst(
                ids=ids,
                labels=labels,
                parents=parents,
                values=values,
                branchvalues="total",
                hovertemplate="<b>%{label}</b><br>Concepts: %{value}<extra></extra>",
                insidetextorientation="radial",
                maxdepth=4,
            )
        )
        self._style(fig, title)
        return fig

    def treemap(self, tree: Dict, title: str = "Concept Treemap") -> go.Figure:
        """
        Build a Plotly Treemap from a nested hierarchy dict.
        """
        ids, labels, parents, values = self._flatten_tree(tree, root="Root")
        if not ids:
            return self._empty_fig(title)

        fig = go.Figure(
            go.Treemap(
                ids=ids,
                labels=labels,
                parents=parents,
                values=values,
                branchvalues="total",
                hovertemplate="<b>%{label}</b><br>%{id}<extra></extra>",
                maxdepth=4,
                textinfo="label+percent entry",
            )
        )
        self._style(fig, title)
        return fig

    def icicle(self, tree: Dict, title: str = "Concept Icicle") -> go.Figure:
        """
        Build a Plotly Icicle chart (top-down hierarchy).
        """
        ids, labels, parents, values = self._flatten_tree(tree, root="Root")
        if not ids:
            return self._empty_fig(title)

        fig = go.Figure(
            go.Icicle(
                ids=ids,
                labels=labels,
                parents=parents,
                values=values,
                branchvalues="total",
                hovertemplate="<b>%{label}</b><extra></extra>",
                maxdepth=4,
            )
        )
        self._style(fig, title)
        return fig

    def tree_diagram(
        self, flat_nodes: List[Tuple[str, str, int]], title: str = "Concept Tree"
    ) -> go.Figure:
        """
        Build a top-down tree diagram using Plotly scatter + annotations.

        Args:
            flat_nodes: List of (node_name, parent_name, depth) tuples.
        """
        import networkx as nx

        G = nx.DiGraph()
        for node, parent, _ in flat_nodes:
            G.add_node(node)
            if parent:
                G.add_edge(parent, node)

        if not G.nodes:
            return self._empty_fig(title)

        # Compute positions (hierarchical layout)
        try:
            pos = self._hierarchy_pos(G)
        except Exception:
            pos = nx.spring_layout(G, seed=42)

        # Draw edges
        edge_x, edge_y = [], []
        for u, v in G.edges():
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(width=1, color="#444466"),
            hoverinfo="none",
        )

        # Draw nodes
        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        depths = {n: d for n, _, d in flat_nodes}

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=list(G.nodes()),
            textposition="top center",
            marker=dict(
                size=[16 - depths.get(n, 0) * 2 for n in G.nodes()],
                color=[depths.get(n, 0) for n in G.nodes()],
                colorscale="Viridis",
                line=dict(width=1, color="#222"),
            ),
            hoverinfo="text",
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color="#6C63FF")),
            showlegend=False,
            paper_bgcolor=self._bg,
            plot_bgcolor=self._bg,
            font=dict(color=self._font_color),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=60, b=20),
            height=700,
        )
        return fig

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _flatten_tree(
        self,
        tree: Dict,
        root: str = "Root",
        parent: str = "",
        ids: List | None = None,
        labels: List | None = None,
        parents: List | None = None,
        values: List | None = None,
        id_counter: List | None = None,
    ) -> Tuple[List, List, List, List]:
        """Recursively flatten nested dict into Plotly hierarchy arrays."""
        if ids is None:
            ids, labels, parents, values = [], [], [], []
            id_counter = [0]
            # Add virtual root
            ids.append(root)
            labels.append(root)
            parents.append("")
            values.append(1)

        for key, children in tree.items():
            id_counter[0] += 1
            node_id = f"{key}_{id_counter[0]}"
            ids.append(node_id)
            labels.append(key[:40])
            parents.append(parent if parent else root)
            leaf_count = self._count_leaves(children)
            values.append(max(1, leaf_count))

            if isinstance(children, dict) and children:
                self._flatten_tree(
                    children, root, node_id, ids, labels, parents, values, id_counter
                )

        return ids, labels, parents, values  # type: ignore

    def _count_leaves(self, node: Dict) -> int:
        """Count leaf nodes in a subtree."""
        if not isinstance(node, dict) or not node:
            return 1
        return sum(self._count_leaves(v) for v in node.values())

    def _hierarchy_pos(
        self, G, root=None, width: float = 2.0, vert_gap: float = 0.4,
        vert_loc: float = 0, xcenter: float = 0.5
    ) -> Dict:
        """Compute hierarchical layout positions for a directed graph."""
        import networkx as nx

        if root is None:
            roots = [n for n in G.nodes() if G.in_degree(n) == 0]
            root = roots[0] if roots else list(G.nodes())[0]

        pos: Dict = {}

        def _recurse(node, left, right, depth):
            children = list(G.successors(node))
            pos[node] = ((left + right) / 2, -depth * vert_gap)
            if children:
                dx = (right - left) / len(children)
                for i, child in enumerate(children):
                    _recurse(child, left + i * dx, left + (i + 1) * dx, depth + 1)

        _recurse(root, 0, width, 0)
        return pos

    def _style(self, fig: go.Figure, title: str) -> None:
        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color="#6C63FF")),
            paper_bgcolor=self._bg,
            font=dict(color=self._font_color, size=12),
            margin=dict(t=60, l=10, r=10, b=10),
            height=600,
        )

    @staticmethod
    def _empty_fig(title: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(
            text="No data to visualize",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#888"),
        )
        fig.update_layout(
            title=title,
            paper_bgcolor="#0F0F1A",
            font=dict(color="#E0E0E0"),
        )
        return fig
