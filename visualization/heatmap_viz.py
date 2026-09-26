"""
================================================
visualization/heatmap_viz.py
Plotly-based similarity heatmap visualization
================================================
"""

from typing import List, Optional

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.logger import get_logger

log = get_logger(__name__)


class HeatmapViz:
    """Generates an interactive Plotly heatmap for a similarity matrix."""

    def __init__(self, colorscale: str = "Viridis") -> None:
        self.colorscale = colorscale

    def build(
        self,
        matrix: np.ndarray,
        labels: List[str],
        title: str = "Paper Similarity Heatmap",
        annotate: bool = True,
    ) -> go.Figure:
        """
        Build a symmetric heatmap.

        Args:
            matrix:   Square similarity matrix (values 0–1).
            labels:   Paper name labels.
            title:    Chart title.
            annotate: Whether to show cell values.

        Returns:
            Plotly Figure.
        """
        # Shorten labels for display
        short_labels = [self._short_label(l) for l in labels]

        # Mask upper triangle for cleaner look
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
        display_matrix = matrix.copy()
        display_matrix[mask] = None  # type: ignore

        z_text = None
        if annotate:
            z_text = [
                [f"{v:.2f}" if v is not None else "" for v in row]  # type: ignore
                for row in display_matrix.tolist()
            ]

        fig = go.Figure(
            data=go.Heatmap(
                z=display_matrix,
                x=short_labels,
                y=short_labels,
                colorscale=self.colorscale,
                zmin=0,
                zmax=1,
                text=z_text,
                texttemplate="%{text}" if annotate else None,
                hovertemplate=(
                    "<b>%{y}</b> ↔ <b>%{x}</b><br>"
                    "Similarity: %{z:.4f}<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color="#6C63FF")),
            paper_bgcolor="#0F0F1A",
            plot_bgcolor="#0F0F1A",
            font=dict(color="#E0E0E0"),
            xaxis=dict(tickangle=-45),
            margin=dict(l=60, r=20, t=60, b=120),
            height=max(400, 80 * len(labels)),
        )

        return fig

    def build_clustermap(
        self,
        matrix: np.ndarray,
        labels: List[str],
        title: str = "Clustered Similarity Heatmap",
    ) -> go.Figure:
        """
        Heatmap with hierarchical clustering ordering.
        Uses scipy linkage to reorder rows/cols.
        """
        try:
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import squareform

            dist = 1 - matrix
            np.fill_diagonal(dist, 0)
            # Ensure symmetric and non-negative
            dist = (dist + dist.T) / 2
            dist = np.clip(dist, 0, None)
            condensed = squareform(dist)
            Z = linkage(condensed, method="ward")
            order = leaves_list(Z)

            reordered = matrix[np.ix_(order, order)]
            reordered_labels = [labels[i] for i in order]
            return self.build(reordered, reordered_labels, title=title)
        except ImportError:
            log.warning("scipy not available; returning standard heatmap.")
            return self.build(matrix, labels, title=title)
        except Exception as e:
            log.error(f"Clustermap failed: {e}")
            return self.build(matrix, labels, title=title)

    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _short_label(label: str, max_chars: int = 30) -> str:
        return label if len(label) <= max_chars else label[:max_chars - 1] + "…"
