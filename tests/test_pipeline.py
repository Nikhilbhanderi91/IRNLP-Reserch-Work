"""
================================================
tests/test_pipeline.py
Basic smoke tests for the analysis pipeline.
Run: python -m pytest tests/ -v
================================================
"""

import sys
import io
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ─── Test text processor ─────────────────────────────────────────────────────
class TestTextProcessor:
    def test_clean(self):
        from extraction.text_processor import TextProcessor
        tp = TextProcessor()
        text = "Hello   World! [1] https://example.com doi:10.1234/abc"
        cleaned = tp.clean(text)
        assert "https" not in cleaned
        assert "doi" not in cleaned
        assert "\n" not in cleaned

    def test_tokenise(self):
        from extraction.text_processor import TextProcessor
        tp = TextProcessor()
        tokens = tp.tokenise("Machine learning is great")
        assert "machine" in tokens
        assert "is" in tokens  # before stopword removal

    def test_remove_stopwords(self):
        from extraction.text_processor import TextProcessor
        tp = TextProcessor()
        tokens = ["machine", "learning", "is", "the", "best"]
        filtered = tp.remove_stopwords(tokens)
        assert "is" not in filtered
        assert "the" not in filtered
        assert "machine" in filtered

    def test_lemmatise(self):
        from extraction.text_processor import TextProcessor
        tp = TextProcessor()
        lemmatised = tp.lemmatise(["running", "models", "classifiers"])
        assert "run" in lemmatised or "running" in lemmatised
        assert "model" in lemmatised or "models" in lemmatised


# ─── Test keyword extractor ───────────────────────────────────────────────────
class TestKeywordExtractor:
    SAMPLE_TEXT = (
        "Deep learning and convolutional neural networks have revolutionized "
        "computer vision. BERT and transformer models dominate NLP tasks. "
        "Machine learning algorithms like SVM and random forest are widely used "
        "for classification and regression in data science."
    )

    def test_tfidf_extraction(self):
        from extraction.keyword_extractor import KeywordExtractor, HAS_SKLEARN
        if not HAS_SKLEARN:
            pytest.skip("sklearn not installed")
        ke = KeywordExtractor()
        results = ke.extract(self.SAMPLE_TEXT)
        assert "tfidf" in results
        assert len(results["tfidf"]) > 0
        first = results["tfidf"][0]
        assert "word" in first and "score" in first

    def test_yake_extraction(self):
        from extraction.keyword_extractor import KeywordExtractor, HAS_YAKE
        if not HAS_YAKE:
            pytest.skip("YAKE not installed")
        ke = KeywordExtractor()
        results = ke.extract(self.SAMPLE_TEXT)
        assert "yake" in results
        # YAKE should return something
        if results["yake"]:
            assert results["yake"][0]["score"] >= 0


# ─── Test concept hierarchy ───────────────────────────────────────────────────
class TestConceptHierarchy:
    def test_build_basic(self):
        from hierarchy.concept_hierarchy import ConceptHierarchy
        ch = ConceptHierarchy()
        tree = ch.build(
            concepts=["machine learning", "deep learning", "neural network"],
            domains=["Artificial Intelligence"],
        )
        assert isinstance(tree, dict)
        assert len(tree) > 0

    def test_flat_concepts(self):
        from hierarchy.concept_hierarchy import ConceptHierarchy
        ch = ConceptHierarchy()
        tree = {"AI": {"ML": {"DL": {}}}}
        flat = ch.get_flat_concepts(tree)
        assert len(flat) == 3
        assert flat[0] == ("AI", 0)
        assert flat[1] == ("ML", 1)
        assert flat[2] == ("DL", 2)


# ─── Test hierarchy builder ───────────────────────────────────────────────────
class TestHierarchyBuilder:
    def test_detect_domains(self):
        from hierarchy.hierarchy_builder import HierarchyBuilder
        hb = HierarchyBuilder()
        domains = hb.detect_domains(
            ["machine learning", "neural network", "bert", "nlp"],
            text="",
        )
        # Should detect AI and/or NLP
        assert len(domains) >= 1

    def test_build_returns_tuple(self):
        from hierarchy.hierarchy_builder import HierarchyBuilder
        hb = HierarchyBuilder()
        result = hb.build(
            concepts=["classification", "regression"],
            keywords=["machine learning"],
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        tree, domains = result
        assert isinstance(tree, dict)
        assert isinstance(domains, list)


# ─── Test search engine ───────────────────────────────────────────────────────
class TestSearchEngine:
    PAPERS = [
        {
            "title": "Deep Learning for NLP",
            "keywords": ["deep learning", "transformer", "bert"],
            "technical_terms": ["BERT", "Transformer", "Attention"],
            "research_domains": ["NLP", "AI"],
            "abstract": "We present a deep learning approach for NLP.",
            "full_text": "Deep learning using transformers.",
        }
    ]

    def test_search_keyword_match(self):
        from search.search_engine import SearchEngine
        se = SearchEngine()
        results = se.search("transformer", self.PAPERS, search_in="keywords")
        assert len(results) > 0
        assert any(r.matched_term.lower() == "transformer" for r in results)

    def test_search_no_results(self):
        from search.search_engine import SearchEngine
        se = SearchEngine()
        results = se.search("quantum_computing_xyz", self.PAPERS)
        assert len(results) == 0

    def test_search_hierarchy(self):
        from search.search_engine import SearchEngine
        se = SearchEngine()
        tree = {
            "Artificial Intelligence": {
                "Machine Learning": {
                    "Deep Learning": {}
                }
            }
        }
        matches = se.search_hierarchy("deep learning", tree)
        assert any("Deep Learning" in m for m in matches)


# ─── Test helpers ─────────────────────────────────────────────────────────────
class TestHelpers:
    def test_deduplicate(self):
        from utils.helpers import deduplicate
        result = deduplicate(["a", "B", "a", "c", "b"])
        assert result == ["a", "B", "c"]

    def test_merge_keyword_dicts(self):
        from utils.helpers import merge_keyword_dicts
        list1 = [{"word": "ml", "score": 0.8}, {"word": "ai", "score": 0.6}]
        list2 = [{"word": "ml", "score": 0.7}, {"word": "nlp", "score": 0.5}]
        merged = merge_keyword_dicts(list1, list2, top=5)
        assert any(d["word"] == "ml" for d in merged)
        ml_entry = next(d for d in merged if d["word"] == "ml")
        assert abs(ml_entry["score"] - 0.75) < 0.01  # average of 0.8 and 0.7

    def test_top_n(self):
        from utils.helpers import top_n
        items = [{"score": 0.3}, {"score": 0.9}, {"score": 0.1}]
        top = top_n(items, n=2)
        assert top[0]["score"] == 0.9
        assert top[1]["score"] == 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
