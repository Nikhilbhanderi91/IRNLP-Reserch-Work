import sys
import unittest
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.scibert_embedder import SciBertEmbedder

class TestSciBertEmbedder(unittest.TestCase):
    def test_scibert_loading_and_similarity(self):
        embedder = SciBertEmbedder()
        
        # Test getting embedding
        emb = embedder.get_embedding("Deep Learning")
        self.assertIsNotNone(emb, "Embedding should not be None")
        self.assertEqual(emb.shape, (768,), "SciBERT base embeddings should be 768-dimensional")
        
        # Test semantic mapping to anchor categories
        anchors = ["Computer Vision", "Natural Language Processing", "Robotics"]
        
        best_anchor, score = embedder.find_best_anchor("Convolutional Neural Networks", anchors, threshold=0.5)
        self.assertIsNotNone(best_anchor)
        self.assertTrue(score >= 0.5)
        
        best_anchor2, score2 = embedder.find_best_anchor("Transformers and Language Models", anchors, threshold=0.5)
        self.assertIsNotNone(best_anchor2)
        self.assertTrue(score2 >= 0.5)

if __name__ == "__main__":
    unittest.main()
