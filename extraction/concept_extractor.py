"""
================================================
extraction/concept_extractor.py
Extracts technical concepts, named entities, and
research topics using spaCy (+ SciSpacy fallback).
Includes paper-specific concept traceability with evidence.
================================================
"""

import re
from typing import Dict, List, Set, Any

from utils.logger import get_logger
from utils.helpers import deduplicate

log = get_logger(__name__)

# ─── spaCy / SciSpacy imports ────────────────────────────────────────────────
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    log.warning("spaCy not installed.")

# ─── Technical term patterns (regex) ─────────────────────────────────────────
_ACRONYM_RE   = re.compile(r"\b[A-Z]{3,8}(?:-\d+)?\b")
_TECH_RE      = re.compile(
    r"\b(?:[A-Z][a-z]+ )*[A-Z][a-z]+(?:[- ][A-Z][a-z]+)+\b"  # Multi-word capitals
    r"|\b[A-Z]{3,8}\b"                                           # Acronyms (≥3 caps)
    r"|\b\w+-(?:based|driven|aware|free|agnostic|efficient)\b",  # compound adjectives
    re.UNICODE,
)

# Research-paper specific noise to ignore
_IGNORE_TERMS: Set[str] = {
    "Figure", "Table", "Section", "Algorithm", "Equation",
    "Appendix", "Theorem", "Proof", "Note",
    "IEEE", "ACM", "ICML", "NeurIPS", "CVPR", "ICLR",
}

# Two/three-letter codes that commonly appear in result tables but are not terms
_ACRONYM_DENYLIST: Set[str] = {
    "USA", "GER", "CHN", "IPA", "PRA", "AVG", "MAX", "MIN",
    "TOP", "ENG", "JPN", "KOR", "FRA", "ESP", "RUS", "AUS",
    "ACC", "FMT", "SYS", "REF", "DEF", "OBJ", "VAR", "ARG",
    "NA", "OK", "GT", "FP", "FN", "TP", "TN",
}


class ConceptExtractor:
    """
    Extracts:
      - Named entities (PERSON, ORG, GPE, PRODUCT, WORK_OF_ART)
      - Technical terms via regex + POS tagging
      - Research concepts (noun chunks filtered by relevance)
      - Traceable concepts with page/sentence evidence
    """

    _ENTITY_LABELS = {"ORG", "PRODUCT", "WORK_OF_ART", "GPE", "EVENT", "LAW"}

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        self.nlp = None
        self._load_spacy(spacy_model)

    def _load_spacy(self, model: str) -> None:
        """Load spaCy model; fall back to smaller model on failure."""
        if not HAS_SPACY:
            return
        for m in [model, "en_core_web_sm", "en_core_web_md"]:
            try:
                self.nlp = spacy.load(m)
                if "sentencizer" not in self.nlp.pipe_names and "parser" not in self.nlp.pipe_names:
                    self.nlp.add_pipe("sentencizer")
                log.info(f"Loaded spaCy model: {m}")
                return
            except OSError:
                log.warning(f"spaCy model '{m}' not found, trying next…")
        log.error("No spaCy model could be loaded. Falling back to regex only.")

    def extract(self, text: str, keywords: List[str] | None = None, pages: List[str] | None = None) -> Dict:
        """
        Full concept extraction.
        """
        keywords = keywords or []
        pages = pages or []

        named_entities   = self._extract_entities(text)
        technical_terms  = self._extract_technical_terms(text)
        noun_chunks      = self._extract_noun_chunks(text)
        research_concepts = self._build_concept_list(
            technical_terms, noun_chunks, keywords
        )

        dedup_concepts = deduplicate(research_concepts)[:60]
        traceable = self.build_traceable_concepts(text, dedup_concepts, pages)

        return {
            "named_entities":    named_entities,
            "technical_terms":   deduplicate(technical_terms)[:50],
            "research_concepts": dedup_concepts,
            "noun_chunks":       deduplicate(noun_chunks)[:80],
            "traceable_concepts": traceable,
        }

    def build_traceable_concepts(self, text: str, concepts: List[str], pages: List[str]) -> List[Dict[str, Any]]:
        """
        Attaches evidence sentence and page index to each concept for source traceability.
        """
        traceable = []
        sentences = [s.strip() for s in re.split(r"\.\s+", text) if len(s.strip()) > 15]

        for concept in concepts:
            c_lower = concept.lower()
            evidence = ""
            page_num = 1

            # Find matching sentence
            for s in sentences:
                if c_lower in s.lower():
                    evidence = s[:200]
                    break

            # Find matching page
            if pages:
                for idx, page_text in enumerate(pages):
                    if c_lower in page_text.lower():
                        page_num = idx + 1
                        break

            confidence = 0.95 if evidence else 0.80
            traceable.append({
                "concept": concept,
                "source": "paper",
                "confidence": confidence,
                "evidence": evidence or f"Extracted from paper text matching '{concept}'.",
                "page": page_num
            })

        return traceable

    def _extract_entities(self, text: str) -> List[Dict]:
        if not self.nlp:
            return []
        try:
            chunk_size = 100_000
            entities: List[Dict] = []
            seen: Set[str] = set()

            for start in range(0, len(text), chunk_size):
                chunk = text[start: start + chunk_size]
                doc = self.nlp(chunk)
                for ent in doc.ents:
                    key = ent.text.strip().lower()
                    if (
                        ent.label_ in self._ENTITY_LABELS
                        and key not in seen
                        and len(ent.text.strip()) > 2
                        and ent.text.strip() not in _IGNORE_TERMS
                    ):
                        seen.add(key)
                        entities.append({"text": ent.text.strip(), "label": ent.label_})

            return entities[:60]
        except Exception as e:
            log.error(f"Entity extraction failed: {e}")
            return []

    def _extract_technical_terms(self, text: str) -> List[str]:
        terms: Set[str] = set()

        for m in _ACRONYM_RE.finditer(text):
            t = m.group()
            if t not in _IGNORE_TERMS and t not in _ACRONYM_DENYLIST and len(t) >= 3:
                terms.add(t)

        for m in _TECH_RE.finditer(text):
            t = m.group().strip()
            if t not in _IGNORE_TERMS and t not in _ACRONYM_DENYLIST and 3 <= len(t) <= 60:
                terms.add(t)

        return list(terms)

    def _extract_noun_chunks(self, text: str) -> List[str]:
        if not self.nlp:
            return []
        try:
            sample = text[:50_000]
            doc = self.nlp(sample)
            chunks: List[str] = []
            seen: Set[str] = set()

            for chunk in doc.noun_chunks:
                c = chunk.text.strip()
                key = c.lower()
                if (
                    key not in seen
                    and 3 <= len(c) <= 60
                    and len(c.split()) >= 2
                    and chunk.root.pos_ in {"NOUN", "PROPN"}
                ):
                    if not any(n in c.lower() for n in ["figure", "table", "section"]):
                        seen.add(key)
                        chunks.append(c)

            return chunks[:100]
        except Exception as e:
            log.error(f"Noun chunk extraction failed: {e}")
            return []

    def _build_concept_list(
        self,
        technical_terms: List[str],
        noun_chunks: List[str],
        keywords: List[str],
    ) -> List[str]:
        all_concepts: List[str] = []
        all_concepts.extend(keywords)
        all_concepts.extend(
            [nc for nc in noun_chunks if 2 <= len(nc.split()) <= 5]
        )
        all_concepts.extend(
            [t for t in technical_terms if len(t) > 3]
        )
        return deduplicate(all_concepts)
