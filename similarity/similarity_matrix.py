"""
================================================
similarity/similarity_matrix.py
Utility class for similarity matrix operations
and pretty-printing / comparison helpers.
================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger

log = get_logger(__name__)


class SimilarityMatrix:
    """
    Wraps a square similarity matrix with labelled rows/columns.
    Provides analysis and formatting utilities.
    """

    def __init__(self, matrix: np.ndarray, labels: List[str]) -> None:
        assert matrix.shape[0] == matrix.shape[1] == len(labels), \
            "Matrix dimensions must match label count."
        self.matrix = matrix
        self.labels = labels
        self.n      = len(labels)

    # ──────────────────────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Return matrix as a labelled Pandas DataFrame."""
        return pd.DataFrame(
            self.matrix,
            index=self.labels,
            columns=self.labels,
        )

    def top_pairs(self, n: int = 10) -> List[Dict]:
        """Return top-N most similar pairs (excluding diagonal)."""
        pairs: List[Dict] = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                pairs.append({
                    "paper_a": self.labels[i],
                    "paper_b": self.labels[j],
                    "score":   round(float(self.matrix[i, j]), 4),
                })
        return sorted(pairs, key=lambda x: x["score"], reverse=True)[:n]

    def cluster_labels(self) -> List[int]:
        """
        Assign each paper to a cluster using hierarchical clustering.
        Returns a list of cluster IDs (one per paper).
        """
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform

            dist = 1.0 - self.matrix
            np.fill_diagonal(dist, 0)
            dist = np.clip((dist + dist.T) / 2, 0, None)
            Z = linkage(squareform(dist), method="ward")
            # Dynamic threshold: ~30% of max distance
            threshold = 0.3 * max(Z[:, 2])
            return fcluster(Z, threshold, criterion="distance").tolist()
        except Exception as e:
            log.warning(f"Clustering failed: {e}")
            return list(range(self.n))

    def average_similarity(self) -> float:
        """Mean off-diagonal similarity."""
        if self.n < 2:
            return 0.0
        mask = ~np.eye(self.n, dtype=bool)
        return float(self.matrix[mask].mean())

    def most_central_paper(self) -> Tuple[str, float]:
        """
        Return the paper with the highest average similarity to all others
        (i.e., the most central / representative paper).
        """
        avg_sims = [
            self.matrix[i, :].sum() - self.matrix[i, i]  # exclude self
            for i in range(self.n)
        ]
        idx = int(np.argmax(avg_sims))
        return self.labels[idx], round(avg_sims[idx] / max(self.n - 1, 1), 4)

    def __repr__(self) -> str:
        return (
            f"<SimilarityMatrix n={self.n} "
            f"avg_sim={self.average_similarity():.3f}>"
        )
