"""
================================================
extraction/keyword_extractor.py
Multi-algorithm keyword extraction:
  • TF-IDF  (scikit-learn)
  • KeyBERT (bert-based)
  • YAKE    (statistical)
Merges results with confidence scores.
================================================
"""

import math
import re
from typing import Dict, List, Optional

from utils.logger import get_logger
from utils.helpers import top_n, merge_keyword_dicts

log = get_logger(__name__)

# ─── Optional imports with graceful fallbacks ─────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    log.warning("scikit-learn not installed; TF-IDF disabled.")

try:
    from keybert import KeyBERT
    HAS_KEYBERT = True
except ImportError:
    HAS_KEYBERT = False
    log.warning("KeyBERT not installed; KeyBERT extraction disabled.")

try:
    import yake
    HAS_YAKE = True
except ImportError:
    HAS_YAKE = False
    log.warning("YAKE not installed; YAKE extraction disabled.")


class KeywordExtractor:
    """
    Extracts keywords using TF-IDF, KeyBERT, and YAKE then
    merges results, returning unified confidence-scored lists.
    """

    def __init__(
        self,
        tfidf_max_features: int = 5000,
        tfidf_ngram: tuple = (1, 3),
        keybert_top_n: int = 20,
        keybert_ngram: tuple = (1, 3),
        yake_max_ngram: int = 3,
        yake_top_n: int = 20,
        final_top_n: int = 30,
    ) -> None:
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram        = tfidf_ngram
        self.keybert_top_n      = keybert_top_n
        self.keybert_ngram      = keybert_ngram
        self.yake_max_ngram     = yake_max_ngram
        self.yake_top_n         = yake_top_n
        self.final_top_n        = final_top_n

        # Lazy-loaded models
        self._keybert_model: Optional[object] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def extract(self, text: str, corpus: Optional[List[str]] = None, vocab_text: Optional[str] = None) -> Dict[str, List[Dict]]:
        """
        Run all available keyword extractors.

        Args:
            text:       Processed natural text of the current document.
            corpus:     List of processed texts of all docs (for TF-IDF IDF weighting).
            vocab_text: Optional lemmatized token text specifically for TF-IDF vocabulary.

        Returns:
            {
              "tfidf":   [{word, score}, ...],
              "keybert": [{word, score}, ...],
              "yake":    [{word, score}, ...],
              "merged":  [{word, score}, ...],
            }
        """
        results: Dict[str, List[Dict]] = {
            "tfidf": [], "keybert": [], "yake": [], "merged": []
        }

        tfidf_input = vocab_text or text
        if HAS_SKLEARN:
            results["tfidf"] = self._filter_keywords(self._run_tfidf(tfidf_input, corpus or [tfidf_input]))

        if HAS_KEYBERT:
            results["keybert"] = self._filter_keywords(self._run_keybert(text))

        if HAS_YAKE:
            results["yake"] = self._filter_keywords(self._run_yake(text))

        # Merge
        active = [v for v in [results["tfidf"], results["keybert"], results["yake"]] if v]
        if active:
            results["merged"] = merge_keyword_dicts(*active, top=self.final_top_n)

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Keyword quality filter
    # ──────────────────────────────────────────────────────────────────────────

    # Common English stopwords that should never appear as keywords on their own
    _STOPWORDS = {
        "which", "that", "this", "with", "from", "into", "over", "then",
        "also", "such", "they", "them", "when", "were", "been", "have",
        "will", "more", "some", "each", "both", "used", "using", "based",
        "paper", "show", "shows", "shown", "results", "method", "methods",
        "approach", "proposed", "presents", "use", "model", "models",
        "task", "tasks", "data", "given", "work", "works", "study",
    }

    def _filter_keywords(self, kws: List[Dict]) -> List[Dict]:
        """Remove low-quality keyword candidates."""
        filtered = []
        for kw in kws:
            word = kw.get("word", "").strip()
            # Minimum 4 characters, must contain at least one letter
            if len(word) < 4 or not re.search(r"[a-zA-Z]", word):
                continue
            # Single token stopword check
            if word.lower() in self._STOPWORDS:
                continue
            # Skip strings starting with digits (table rows, page numbers)
            if re.match(r"^[\d\W]", word):
                continue
            # For 2-3 letter all-caps (USA, GER) require a min score
            if re.match(r"^[A-Z]{2,4}$", word) and kw.get("score", 0) < 0.35:
                continue
            filtered.append(kw)
        return filtered

    def extract_tfidf_corpus(self, texts: List[str]) -> List[List[Dict]]:
        """
        Batch TF-IDF extraction across a corpus.
        Each document gets its own ranked list.
        """
        if not HAS_SKLEARN or not texts:
            return [[] for _ in texts]

        try:
            vec = TfidfVectorizer(
                max_features=self.tfidf_max_features,
                ngram_range=self.tfidf_ngram,
                stop_words="english",
                sublinear_tf=True,
            )
            tfidf_matrix = vec.fit_transform(texts)
            feature_names = vec.get_feature_names_out()

            all_keywords: List[List[Dict]] = []
            for row in tfidf_matrix:
                scores = row.toarray().flatten()
                top_idx = scores.argsort()[::-1][: self.final_top_n]
                kws = [
                    {"word": feature_names[i], "score": round(float(scores[i]), 4)}
                    for i in top_idx
                    if scores[i] > 0
                ]
                all_keywords.append(kws)
            return all_keywords
        except Exception as e:
            log.error(f"Batch TF-IDF failed: {e}")
            return [[] for _ in texts]

    # ──────────────────────────────────────────────────────────────────────────
    # Internal extractors
    # ──────────────────────────────────────────────────────────────────────────

    def _run_tfidf(self, text: str, corpus: List[str]) -> List[Dict]:
        """Single-document TF-IDF relative to corpus."""
        try:
            vec = TfidfVectorizer(
                max_features=self.tfidf_max_features,
                ngram_range=self.tfidf_ngram,
                stop_words="english",
                sublinear_tf=True,
            )
            corpus_unique = list({text} | set(corpus))
            matrix = vec.fit_transform(corpus_unique)
            feature_names = vec.get_feature_names_out()

            # Row index of `text` in corpus_unique
            doc_idx = corpus_unique.index(text)
            scores = matrix[doc_idx].toarray().flatten()
            top_idx = scores.argsort()[::-1][: self.final_top_n]

            return [
                {"word": feature_names[i], "score": round(float(scores[i]), 4)}
                for i in top_idx
                if scores[i] > 0
            ]
        except Exception as e:
            log.error(f"TF-IDF extraction failed: {e}")
            return []

    def _run_keybert(self, text: str) -> List[Dict]:
        """KeyBERT extraction using cached model."""
        try:
            if self._keybert_model is None:
                log.info("Loading KeyBERT model…")
                self._keybert_model = KeyBERT(model="all-MiniLM-L6-v2")

            kws = self._keybert_model.extract_keywords(
                text,
                keyphrase_ngram_range=self.keybert_ngram,
                stop_words="english",
                top_n=self.keybert_top_n,
                use_mmr=True,
                diversity=0.5,
            )
            return [{"word": w, "score": round(s, 4)} for w, s in kws]
        except Exception as e:
            log.error(f"KeyBERT extraction failed: {e}")
            return []

    def _run_yake(self, text: str) -> List[Dict]:
        """YAKE statistical keyword extraction (lower score = more relevant)."""
        try:
            extractor = yake.KeywordExtractor(
                lan="en",
                n=self.yake_max_ngram,
                dedupLim=0.7,
                top=self.yake_top_n,
                features=None,
            )
            raw = extractor.extract_keywords(text)

            # Invert and normalise scores (YAKE: lower is better)
            if not raw:
                return []
            max_score = max(s for _, s in raw) or 1.0
            return [
                {"word": kw, "score": round(1.0 - (s / max_score), 4)}
                for kw, s in raw
            ]
        except Exception as e:
            log.error(f"YAKE extraction failed: {e}")
            return []
