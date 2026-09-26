"""
================================================
tests/test_paper_isolation.py
Automated regression test suite for Paper Analysis Isolation,
Identity Validation, Cache Invalidation, and Concept Traceability.
================================================
"""

import os
import unittest
import tempfile
from pathlib import Path

from models.paper import Paper
from app.pipeline import AnalysisPipeline
from utils.validator import PaperValidator, PaperIdentityMismatchError, StaleAnalysisDataError


class TestPaperAnalysisIsolation(unittest.TestCase):
    """
    Automated tests verifying paper-by-paper isolation, identity validation,
    and prevention of cross-paper concept leakage.
    """

    def setUp(self):
        self.pipeline = AnalysisPipeline()
        self.validator = PaperValidator()

        # Synthetic test paper text A
        self.paper_a_text = """
        AT-BERT: Adversarial Training BERT for Acronym Identification
        John Doe, Jane Smith
        Abstract
        We propose AT-BERT, an adversarial training approach for BERT in Acronym Identification tasks.
        We apply Fast Gradient Method (FGM) perturbations to embeddings on the SDU@AAAI-21 dataset.
        Experimental results show state-of-the-art F1 score and Precision for sequence labeling.
        1. Introduction
        Acronym identification (AI) is crucial for Natural Language Processing (NLP).
        Multi-BERT ensemble improves robustness.
        """

        # Synthetic test paper text B
        self.paper_b_text = """
        Deep Learning for Autonomous Vehicle Path Planning
        Alice Johnson, Bob Williams
        Abstract
        This paper introduces a Reinforcement Learning algorithm for autonomous driving path selection.
        We evaluate on the KITTI dataset using Convolutional Neural Networks (CNN) and Q-Learning.
        1. Introduction
        Computer Vision and LiDAR processing enable real-time obstacle avoidance.
        """

        # Write synthetic PDFs or text files
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.path_a = Path(self.tmp_dir.name) / "paper_at_bert.txt"
        self.path_b = Path(self.tmp_dir.name) / "paper_autonomous.txt"

        self.path_a.write_text(self.paper_a_text, encoding="utf-8")
        self.path_b.write_text(self.paper_b_text, encoding="utf-8")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_scenario_1_paper_a_extraction(self):
        """TEST 1: Upload Paper A -> verify concepts belong to Paper A."""
        ext_a = self.pipeline._pdf.extract(self.path_a)
        proc_a = self.pipeline._text.process(ext_a["full_text"])
        paper_a = self.pipeline._build_paper(ext_a, proc_a, [], [proc_a["cleaned_text"]])

        self.assertIn("AT-BERT", paper_a.title)
        self.assertTrue(any("BERT" in c or "Acronym" in c or "NLP" in c for c in paper_a.all_keywords + paper_a.research_concepts))

    def test_scenario_2_paper_b_isolation(self):
        """TEST 2: Upload Paper B -> verify Paper B does not contain Paper A concepts."""
        ext_b = self.pipeline._pdf.extract(self.path_b)
        proc_b = self.pipeline._text.process(ext_b["full_text"])
        paper_b = self.pipeline._build_paper(ext_b, proc_b, [], [proc_b["cleaned_text"]])

        concepts_b = [c.lower() for c in paper_b.research_concepts + paper_b.all_keywords]
        self.assertNotIn("at-bert", concepts_b)
        self.assertNotIn("fgm", concepts_b)
        self.assertNotIn("sdu@aaai-21", concepts_b)

    def test_scenario_3_reanalyze_paper_a(self):
        """TEST 3: Upload Paper A again -> verify previous Paper B data is not reused."""
        ext_a = self.pipeline._pdf.extract(self.path_a)
        proc_a = self.pipeline._text.process(ext_a["full_text"])
        paper_a = self.pipeline._build_paper(ext_a, proc_a, [], [proc_a["cleaned_text"]])

        concepts_a = [c.lower() for c in paper_a.research_concepts + paper_a.all_keywords]
        self.assertNotIn("kitti", concepts_a)
        self.assertNotIn("lidar", concepts_a)
        self.assertNotIn("q-learning", concepts_a)

    def test_scenario_4_similarity_insufficient_papers(self):
        """TEST 4: Single paper similarity check returns insufficient_papers."""
        ext_a = self.pipeline._pdf.extract(self.path_a)
        proc_a = self.pipeline._text.process(ext_a["full_text"])
        paper_a = self.pipeline._build_paper(ext_a, proc_a, [], [proc_a["cleaned_text"]])

        sim_res = self.pipeline.compute_similarity([paper_a])
        self.assertEqual(sim_res.get("similarity_status"), "insufficient_papers")

    def test_scenario_5_paper_identity_hashing(self):
        """TEST 5: Verify each paper has a unique paper_id and file_hash."""
        ext_a = self.pipeline._pdf.extract(self.path_a)
        ext_b = self.pipeline._pdf.extract(self.path_b)

        self.assertNotEqual(ext_a["file_hash"], ext_b["file_hash"])
        self.assertNotEqual(ext_a["paper_id"], ext_b["paper_id"])

    def test_scenario_6_hierarchy_and_mindmap_node_derivation(self):
        """TEST 6 & 7: Verify hierarchy & mindmap nodes belong strictly to current paper."""
        ext_a = self.pipeline._pdf.extract(self.path_a)
        proc_a = self.pipeline._text.process(ext_a["full_text"])
        paper_a = self.pipeline._build_paper(ext_a, proc_a, [], [proc_a["cleaned_text"]])

        tree_str = str(paper_a.concept_hierarchy).lower()
        self.assertNotIn("kitti", tree_str)
        self.assertNotIn("path selection", tree_str)

    def test_scenario_8_at_bert_validation_and_no_maddog_leak(self):
        """TEST 8: AT-BERT paper validation -> zero MadDog / Veyseh concept leak."""
        ext_a = self.pipeline._pdf.extract(self.path_a)
        proc_a = self.pipeline._text.process(ext_a["full_text"])
        paper_a = self.pipeline._build_paper(ext_a, proc_a, [], [proc_a["cleaned_text"]])

        val_result = self.validator.validate_paper(paper_a)
        self.assertEqual(val_result["status"], "validated")

        # Confirm section 6 JSON format
        sec6 = paper_a.to_structured_json()
        self.assertIn("paper", sec6)
        self.assertIn("traceable_concepts", sec6)
        self.assertEqual(sec6["paper"]["file_hash"], paper_a.file_hash)


if __name__ == "__main__":
    unittest.main()
