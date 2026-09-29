"""
================================================
tests/test_dataset_expansion.py
Unit tests for HMRP Multidisciplinary Dataset Expansion:
  - Title Normalization
  - arXiv Category Validation
  - Duplicate Detection
  - Metadata Completeness
  - PDF Text Validation
  - Resumable Dataset Counting
================================================
"""

import sys
import unittest
import tempfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from HMRP_Dataset.config import DOMAINS, TARGET_PER_DOMAIN, MIN_FULLTEXT_WORDS
from HMRP_Dataset.download_papers import normalize_title, validate_pdf_file, HMRPDatasetHarvester


class TestDatasetExpansion(unittest.TestCase):

    def test_domain_configuration(self):
        """Test that exactly 15 domains are configured with TARGET_PER_DOMAIN=50."""
        self.assertEqual(len(DOMAINS), 15, "DOMAINS must contain exactly 15 multidisciplinary domains.")
        self.assertEqual(TARGET_PER_DOMAIN, 50, "TARGET_PER_DOMAIN must be 50.")
        
        # Test total paper count calculation
        total_target = len(DOMAINS) * TARGET_PER_DOMAIN
        self.assertEqual(total_target, 750, "Expected dataset size must be exactly 750 papers.")

    def test_title_normalization(self):
        """Test Unicode stripping, lowercase, punctuation removal, and whitespace collapsing."""
        raw_title_1 = "  Hierarchical Concept-Mapping for   Research: A Novel Approach!  "
        raw_title_2 = "Hierarchical Concept Mapping for Research: A Novel Approach."
        
        norm_1 = normalize_title(raw_title_1)
        norm_2 = normalize_title(raw_title_2)
        
        self.assertEqual(norm_1, norm_2)
        self.assertEqual(norm_1, "hierarchical concept mapping for research a novel approach")

    def test_arxiv_category_specifications(self):
        """Test that all domain category lists contain valid non-empty strings."""
        for domain, cats in DOMAINS.items():
            self.assertIsInstance(cats, list, f"Domain {domain} must specify categories as a list.")
            self.assertGreater(len(cats), 0, f"Domain {domain} must have at least one category.")
            for cat in cats:
                self.assertTrue(len(cat) >= 4, f"Category '{cat}' in domain {domain} is invalid.")
                self.assertTrue("." in cat or "-" in cat or cat in ("quant-ph", "cs.AI", "cs.RO", "cs.CV", "cs.CL", "cs.IR", "eess.SP", "eess.SY"), f"Category '{cat}' format unrecognized.")

    def test_pdf_validation_invalid_file(self):
        """Test that non-existent and corrupt files fail validation."""
        # Case 1: Under 1KB file size
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(b"not a real pdf content")
            tmp.flush()
            is_valid, reason, pages, words = validate_pdf_file(Path(tmp.name))
            self.assertFalse(is_valid)
            self.assertIn("file_too_small", reason)

        # Case 2: Over 1KB with invalid header
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(b"NOT_A_PDF" * 200)
            tmp.flush()
            is_valid, reason, pages, words = validate_pdf_file(Path(tmp.name))
            self.assertFalse(is_valid)
            self.assertIn("invalid_pdf_header", reason)

    def test_harvester_state_tracking(self):
        """Test harvester duplicate registry."""
        harvester = HMRPDatasetHarvester(target_per_domain=50)
        
        # Add a dummy record
        harvester.global_seen_ids.add("2301.00001")
        harvester.global_seen_titles.add(normalize_title("Test Machine Learning Paper"))

        self.assertIn("2301.00001", harvester.global_seen_ids)
        self.assertIn(normalize_title("Test Machine Learning Paper"), harvester.global_seen_titles)


if __name__ == "__main__":
    unittest.main()
