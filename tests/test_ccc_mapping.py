"""
================================================
tests/test_ccc_mapping.py
Unit tests for Concept-to-Concept & Cross-Domain Mapping (CCC)
================================================
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hierarchy.ccc_mapper import CCCMapper
from models.paper import Paper
from app.pipeline import AnalysisPipeline


class TestCCCMapping(unittest.TestCase):

    def setUp(self):
        self.ccc = CCCMapper()

    def test_paper_ccc_generation(self):
        """Test concept-to-concept edge extraction and cross-domain bridging."""
        concepts = [
            "Convolutional Neural Networks",
            "Deep Learning",
            "Image Segmentation",
            "Protein Structure Prediction",
            "Supervised Learning"
        ]
        keywords = ["CNN", "ResNet", "Proteins", "Biomedical"]
        domains = ["Computer Vision", "Biomedical & Genomics", "Artificial Intelligence"]
        text = "Deep learning and convolutional neural networks are used for image segmentation and protein structure prediction."

        result = self.ccc.map_paper_ccc(
            concepts=concepts,
            keywords=keywords,
            domains=domains,
            text=text,
            min_similarity=0.20
        )

        self.assertIn("nodes", result)
        self.assertIn("c2c_edges", result)
        self.assertIn("cross_domain_bridges", result)
        self.assertIn("domain_alignments", result)
        self.assertGreater(len(result["nodes"]), 0)
        self.assertGreater(len(result["c2c_edges"]), 0)

    def test_corpus_ccc_mapping(self):
        """Test global corpus level bridge extraction."""
        p1 = Paper(
            paper_id="p1",
            title="Transformer in Vision and Genomics",
            research_domains=["Computer Vision", "Biomedical & Genomics"],
            research_concepts=["Transformer", "Attention Mechanism", "Protein Sequence"],
            all_keywords=["transformer", "genomics"]
        )
        p2 = Paper(
            paper_id="p2",
            title="Transformer in NLP",
            research_domains=["Natural Language Processing", "Artificial Intelligence"],
            research_concepts=["Transformer", "Self-Attention", "BERT"],
            all_keywords=["transformer", "nlp"]
        )

        corpus_result = self.ccc.map_corpus_ccc([p1, p2])
        self.assertIn("cross_domain_bridge_concepts", corpus_result)
        self.assertIn("top_cooccurring_concept_pairs", corpus_result)
        
        # "Transformer" should be identified as a bridge concept across multiple domains
        bridge_names = [b["concept"] for b in corpus_result["cross_domain_bridge_concepts"]]
        self.assertIn("Transformer", bridge_names)


if __name__ == "__main__":
    unittest.main()
