"""
================================================
similarity/similarity_engine.py
Computes paper-to-paper similarity using:
  • TF-IDF + Cosine Similarity
  • Sentence Transformer embeddings + Cosine
Enforces minimum paper requirements and prevents stale matrix reuse.
================================================
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger

log = get_logger(__name__)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False
    log.warning("sentence-transformers not installed; semantic similarity disabled.")


class SimilarityEngine:
    """
    Computes pairwise similarity between research papers.
    """

    def __init__(self, sentence_model: str = "all-MiniLM-L6-v2") -> None:
        self._st_model_name = sentence_model
        self._st_model: Optional[object] = None

    def compute(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None,
        method: str = "both",
    ) -> Dict:
        """
        Compute similarity matrices across papers.
        If len(texts) < 2, returns 'similarity_status': 'insufficient_papers'.
        """
        if not texts or len(texts) < 2:
            log.info("Insufficient papers (<2) provided for similarity calculation.")
            res = self._empty_result()
            res["similarity_status"] = "insufficient_papers"
            if texts and labels:
                res["labels"] = labels
            return res

        labels = labels or [f"Paper {i+1}" for i in range(len(texts))]

        tfidf_matrix    = None
        semantic_matrix = None
        embeddings      = None

        if method in ("tfidf", "both") and HAS_SKLEARN:
            tfidf_matrix = self._tfidf_similarity(texts)
            log.info("TF-IDF similarity matrix computed.")

        if method in ("semantic", "both") and HAS_ST:
            semantic_matrix, embeddings = self._semantic_similarity(texts)
            log.info("Semantic similarity matrix computed.")

        available = [m for m in [tfidf_matrix, semantic_matrix] if m is not None]
        if len(available) >= 2:
            combined_matrix = np.mean(available, axis=0)
        elif available:
            combined_matrix = available[0]
        else:
            combined_matrix = None

        return {
            "similarity_status": "completed",
            "tfidf_matrix":    tfidf_matrix,
            "semantic_matrix": semantic_matrix,
            "combined_matrix": combined_matrix,
            "labels":          labels,
            "embeddings":      embeddings,
        }

    def get_top_similar_pairs(
        self,
        matrix: np.ndarray,
        labels: List[str],
        top_n: int = 10,
    ) -> List[Dict]:
        """
        Return top-N most similar paper pairs. Excludes self-similarity (i == j).
        """
        if matrix is None or len(labels) < 2:
            return []

        n = len(labels)
        pairs: List[Dict] = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append({
                    "paper_a": labels[i],
                    "paper_b": labels[j],
                    "score":   round(float(matrix[i][j]), 4),
                })
        return sorted(pairs, key=lambda x: x["score"], reverse=True)[:top_n]

    def _tfidf_similarity(self, texts: List[str]) -> Optional[np.ndarray]:
        try:
            vec = TfidfVectorizer(
                max_features=10_000,
                ngram_range=(1, 2),
                stop_words="english",
                sublinear_tf=True,
            )
            tfidf = vec.fit_transform(texts)
            matrix = cosine_similarity(tfidf)
            return matrix.astype(np.float32)
        except Exception as e:
            log.error(f"TF-IDF similarity failed: {e}")
            return None

    def _semantic_similarity(
        self, texts: List[str]
    ) -> Tuple[Optional[np.ndarray], Optional[List]]:
        try:
            if self._st_model is None:
                log.info(f"Loading SentenceTransformer: {self._st_model_name}")
                self._st_model = SentenceTransformer(self._st_model_name)

            truncated = [t[:8000] for t in texts]
            embeddings = self._st_model.encode(truncated, show_progress_bar=False)
            matrix = cosine_similarity(embeddings)
            return matrix.astype(np.float32), embeddings.tolist()
        except Exception as e:
            log.error(f"Semantic similarity failed: {e}")
            return None, None

    @staticmethod
    def _empty_result() -> Dict:
        return {
            "similarity_status": "insufficient_papers",
            "tfidf_matrix": None,
            "semantic_matrix": None,
            "combined_matrix": None,
            "labels": [],
            "embeddings": None,
        }
