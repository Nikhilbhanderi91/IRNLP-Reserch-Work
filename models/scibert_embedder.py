"""
================================================
models/scibert_embedder.py
Loads SciBERT and computes CLS token embeddings
for semantic concept mapping and similarity.
================================================
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from utils.logger import get_logger

log = get_logger(__name__)

class SciBertEmbedder:
    """
    Lazy-loaded singleton class for extracting SciBERT [CLS] embeddings
    and finding matching anchor concepts.
    """
    _instance: Optional['SciBertEmbedder'] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SciBertEmbedder, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (useful for testing)."""
        cls._instance = None

    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased") -> None:
        if self._initialized:
            return
        
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        self._initialized = True

    def _load_model(self) -> None:
        """Lazy load SciBERT model and tokenizer."""
        if self.model is not None:
            return
        
        try:
            from transformers import AutoTokenizer, AutoModel
            log.info(f"Loading SciBERT model and tokenizer: {self.model_name}")

            # Try loading from local cache first to avoid network calls
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, local_files_only=True
                )
                self.model = AutoModel.from_pretrained(
                    self.model_name, local_files_only=True
                )
                log.info("SciBERT loaded from local cache.")
            except Exception:
                log.info("Local cache not found – downloading from HuggingFace...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
            
            # Set device
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
                
            self.model.to(self.device)
            self.model.eval()
            log.info(f"SciBERT successfully loaded on device: {self.device}")
        except Exception as e:
            log.error(f"Failed to load SciBERT model '{self.model_name}': {e}")
            raise e

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Compute a mean-pooled embedding for the given text.
        Mean pooling over all non-padding token embeddings produces more
        reliable cosine similarity scores than the raw [CLS] token alone.
        """
        try:
            self._load_model()
            if self.model is None or self.tokenizer is None:
                return None

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                token_embeddings = outputs.last_hidden_state  # (1, seq_len, 768)
                attention_mask = inputs["attention_mask"]     # (1, seq_len)

                # Expand mask to match embedding dimensions
                mask_expanded = (
                    attention_mask.unsqueeze(-1)
                    .expand(token_embeddings.size())
                    .float()
                )
                # Masked mean pooling
                sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                mean_embedding = (sum_embeddings / sum_mask)[0].cpu().numpy()

            return mean_embedding
        except Exception as e:
            log.error(f"Error computing embedding for text '{text}': {e}")
            return None

    def find_best_anchor(
        self, 
        term: str, 
        anchors: List[str], 
        threshold: float = 0.5
    ) -> Tuple[Optional[str], float]:
        """
        Compare term embedding to anchor embeddings using cosine similarity.
        Returns the best matching anchor (or None if below threshold) and similarity score.
        """
        term_emb = self.get_embedding(term)
        if term_emb is None:
            return None, 0.0
            
        term_norm = np.linalg.norm(term_emb)
        if term_norm == 0:
            return None, 0.0

        best_anchor = None
        best_score = -1.0

        for anchor in anchors:
            anchor_emb = self.get_embedding(anchor)
            if anchor_emb is None:
                continue
                
            anchor_norm = np.linalg.norm(anchor_emb)
            if anchor_norm == 0:
                continue
                
            sim = np.dot(term_emb, anchor_emb) / (term_norm * anchor_norm)
            if sim > best_score:
                best_score = float(sim)
                best_anchor = anchor

        if best_anchor and best_score >= threshold:
            return best_anchor, best_score
        return None, best_score
