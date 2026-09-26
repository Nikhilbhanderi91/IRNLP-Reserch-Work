"""
================================================
extraction/text_processor.py
Cleaning, tokenisation, lemmatisation, stopword
removal, and sentence segmentation using NLTK.
================================================
"""

import re
import string
from typing import List

import nltk

from utils.logger import get_logger

log = get_logger(__name__)

# ─── Download required NLTK resources (once) ─────────────────────────────────
_NLTK_RESOURCES = [
    ("tokenizers/punkt",                 "punkt"),
    ("tokenizers/punkt_tab",             "punkt_tab"),
    ("corpora/stopwords",                "stopwords"),
    ("corpora/wordnet",                  "wordnet"),
    ("taggers/averaged_perceptron_tagger","averaged_perceptron_tagger"),
]

for _path, _pkg in _NLTK_RESOURCES:
    try:
        nltk.data.find(_path)
    except LookupError:
        log.info(f"Downloading NLTK package: {_pkg}")
        nltk.download(_pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

_STOP_WORDS  = set(stopwords.words("english"))
_LEMMATIZER  = WordNetLemmatizer()

# Extra domain-specific stopwords
_EXTRA_STOPS = {
    "et", "al", "fig", "figure", "table", "eq", "equation",
    "using", "used", "based", "also", "however", "therefore",
    "thus", "result", "results", "paper", "method", "proposed",
    "show", "shown", "section", "approach", "model",
}
_STOP_WORDS |= _EXTRA_STOPS


class TextProcessor:
    """
    Handles all text pre-processing steps:
      1. Unicode / whitespace normalisation
      2. Sentence segmentation
      3. Tokenisation
      4. Lowercasing & punctuation removal
      5. Stopword removal
      6. Lemmatisation
    """

    def __init__(self, language: str = "english") -> None:
        self.language = language

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def process(self, text: str) -> dict:
        """
        Full processing pipeline.

        Args:
            text: Raw extracted text from a paper.

        Returns:
            dict with keys: cleaned_text, tokens, sentences,
                            lemmatised_tokens, filtered_tokens
        """
        cleaned    = self.clean(text)
        sentences  = self.segment_sentences(cleaned)
        tokens     = self.tokenise(cleaned)
        filtered   = self.remove_stopwords(tokens)
        lemmatised = self.lemmatise(filtered)

        return {
            "cleaned_text":    cleaned,
            "sentences":       sentences,
            "tokens":          tokens,
            "filtered_tokens": filtered,
            "lemmatised_tokens": lemmatised,
        }

    def clean(self, text: str) -> str:
        """Normalise unicode, remove noise, collapse whitespace."""
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        # Remove email addresses
        text = re.sub(r"\S+@\S+", " ", text)
        # Remove DOI / arXiv refs
        text = re.sub(r"\b(doi|arxiv)[:\s]\S+", " ", text, flags=re.I)
        # Remove stray numbers (page numbers, refs like [1], (2))
        text = re.sub(r"\[\d+\]|\(\d+\)", " ", text)
        # Remove lone single chars except 'a', 'i'
        text = re.sub(r"\b(?![ai])[b-z]\b", " ", text, flags=re.I)
        # Collapse whitespace
        text = re.sub(r"[\t\r\n]+", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def segment_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK's Punkt tokenizer."""
        try:
            return sent_tokenize(text, language=self.language)
        except Exception:
            # Fallback: split on period-space
            return [s.strip() for s in re.split(r"\.\s+", text) if s.strip()]

    def tokenise(self, text: str) -> List[str]:
        """Word-tokenise and lowercase; keep only alphabetic tokens."""
        try:
            tokens = word_tokenize(text.lower())
        except Exception:
            tokens = text.lower().split()
        return [t for t in tokens if t.isalpha() and len(t) > 1]

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Filter out stopwords from a token list."""
        return [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]

    def lemmatise(self, tokens: List[str]) -> List[str]:
        """Lemmatise tokens using WordNet lemmatizer."""
        return [_LEMMATIZER.lemmatize(t) for t in tokens]

    def get_vocab_text(self, tokens: List[str]) -> str:
        """Join processed tokens back to a single string (for TF-IDF)."""
        return " ".join(tokens)
