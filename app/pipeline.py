"""
================================================
app/pipeline.py
Orchestrates the full paper analysis pipeline:
  PDF → Extract → Process → Keywords → Concepts
  → Hierarchy → Similarity → Validation → Results
================================================
"""

import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from models.paper import Paper
from extraction.pdf_extractor import PDFExtractor
from extraction.text_processor import TextProcessor
from extraction.keyword_extractor import KeywordExtractor
from extraction.concept_extractor import ConceptExtractor
from hierarchy.hierarchy_builder import HierarchyBuilder
from similarity.similarity_engine import SimilarityEngine
from utils.validator import PaperValidator
from utils.logger import get_logger

log = get_logger(__name__)


class AnalysisPipeline:
    """
    High-level orchestrator that processes one or many PDF files
    and returns fully populated & validated Paper objects.
    """

    def __init__(self) -> None:
        self._pdf       = PDFExtractor()
        self._text      = TextProcessor()
        self._keywords  = KeywordExtractor()
        self._concepts  = ConceptExtractor()
        self._hierarchy = HierarchyBuilder()
        self._similarity = SimilarityEngine()
        self._validator = PaperValidator()

    def process_papers(
        self,
        pdf_paths: List[str | Path],
        progress_callback=None,
    ) -> List[Paper]:
        """
        Process a list of PDF files end-to-end with strict paper isolation.
        """
        papers: List[Paper] = []
        total = len(pdf_paths)

        all_raw_texts: List[str] = []
        raw_extractions: List[Dict] = []

        for i, path in enumerate(pdf_paths):
            if progress_callback:
                progress_callback(i, total * 3, f"Extracting PDF {i+1}/{total}…")
            ext = self._pdf.extract(path)
            raw_extractions.append(ext)
            all_raw_texts.append(ext.get("full_text", ""))

        processed_texts: List[str] = []
        all_processed: List[Dict] = []
        for ext in raw_extractions:
            proc = self._text.process(ext.get("full_text", ""))
            all_processed.append(proc)
            processed_texts.append(proc.get("cleaned_text", ""))

        tfidf_batch = self._keywords.extract_tfidf_corpus(processed_texts)

        for i, (ext, proc, tfidf_kws) in enumerate(
            zip(raw_extractions, all_processed, tfidf_batch)
        ):
            if progress_callback:
                progress_callback(
                    total + i, total * 3,
                    f"Analysing paper {i+1}/{total}…",
                )
            paper = self._build_paper(ext, proc, tfidf_kws, processed_texts)
            
            # Validate paper before adding
            self._validator.validate_paper(paper)
            papers.append(paper)
            log.info(f"Paper processed & validated: {paper.title[:60]}")

        return papers

    def process_single(self, pdf_path: str | Path) -> Paper:
        results = self.process_papers([pdf_path])
        return results[0] if results else Paper()

    def compute_similarity(
        self, papers: List[Paper], method: str = "both"
    ) -> Dict:
        """
        Compute pairwise similarity across papers.
        Returns 'similarity_status': 'insufficient_papers' if len(papers) < 2.
        """
        if len(papers) < 2:
            log.warning("Fewer than 2 papers provided for similarity analysis.")
            return {
                "similarity_status": "insufficient_papers",
                "labels": [p.title[:50] or p.file_name for p in papers],
                "tfidf_matrix": None,
                "semantic_matrix": None,
                "combined_matrix": None,
                "embeddings": None
            }

        texts  = [p.cleaned_text or p.full_text for p in papers]
        labels = [p.title[:50] or p.file_name for p in papers]

        results = self._similarity.compute(texts, labels=labels, method=method)

        if results.get("embeddings"):
            for paper, emb in zip(papers, results["embeddings"]):
                paper.embedding = emb
                paper.similarity_data = {
                    "labels": results.get("labels", []),
                    "status": results.get("similarity_status", "completed")
                }

        return results

    def _build_paper(
        self,
        ext: Dict,
        proc: Dict,
        tfidf_kws: List[Dict],
        corpus_texts: List[str],
    ) -> Paper:
        paper = Paper()

        # Identity
        paper.paper_id     = ext.get("paper_id", f"paper_{uuid.uuid4().hex[:12]}")
        paper.file_hash    = ext.get("file_hash", "")
        paper.analysis_id  = f"analysis_{uuid.uuid4().hex[:8]}"
        paper.file_name    = ext.get("file_name", "")
        paper.file_path    = ext.get("file_path", "")
        paper.year         = ext.get("year", None)

        # Raw & Processed text
        paper.raw_text     = ext.get("raw_text", "")
        paper.full_text    = ext.get("full_text", "")
        paper.title        = ext.get("title", "Untitled")
        paper.authors      = ext.get("authors", [])
        paper.abstract     = ext.get("abstract", "")
        paper.introduction = ext.get("introduction", "")
        paper.page_count   = ext.get("page_count", 0)

        paper.cleaned_text = proc.get("cleaned_text", "")
        paper.tokens       = proc.get("filtered_tokens", [])
        paper.sentences    = proc.get("sentences", [])
        paper.word_count   = len(proc.get("tokens", []))
        paper.char_count   = len(paper.full_text)

        # Extracted paper sections
        paper.problem            = ext.get("problem")
        paper.objective          = ext.get("objective")
        paper.existing_methods   = ext.get("existing_methods", [])
        paper.proposed_method    = ext.get("proposed_method", {"name": paper.title, "description": ""})
        paper.models             = ext.get("models", [])
        paper.algorithms         = ext.get("algorithms", [])
        paper.datasets           = ext.get("datasets", [])
        paper.evaluation_metrics = ext.get("evaluation_metrics", [])
        paper.results            = ext.get("results", [])
        paper.advantages         = ext.get("advantages", [])
        paper.limitations        = ext.get("limitations", [])
        paper.research_gap       = ext.get("research_gap", [])
        paper.future_work        = ext.get("future_work", [])

        # Keywords - Pass natural cleaned text for KeyBERT & YAKE, vocab tokens for TF-IDF
        kw_vocab = self._text.get_vocab_text(proc.get("lemmatised_tokens", []))
        natural_text = (proc.get("cleaned_text", "") or paper.full_text)[:30000]
        kw_results = self._keywords.extract(natural_text, corpus=corpus_texts, vocab_text=kw_vocab)

        paper.tfidf_keywords  = tfidf_kws
        paper.keybert_keywords = kw_results.get("keybert", [])
        paper.yake_keywords    = kw_results.get("yake", [])
        merged = kw_results.get("merged", [])
        paper.all_keywords = [d["word"] for d in merged]

        raw_kws = ext.get("keywords_raw", [])
        for kw in raw_kws:
            if kw not in paper.all_keywords:
                paper.all_keywords.insert(0, kw)

        # Paper-specific Concept Extraction & Traceability
        concept_result = self._concepts.extract(
            paper.full_text[:80_000],
            keywords=paper.all_keywords[:20],
            pages=ext.get("pages", []),
        )
        paper.named_entities    = concept_result.get("named_entities", [])
        paper.technical_terms   = concept_result.get("technical_terms", [])
        paper.research_concepts  = concept_result.get("research_concepts", [])
        paper.traceable_concepts = concept_result.get("traceable_concepts", [])

        # Hierarchy
        tree, domains = self._hierarchy.build(
            concepts=paper.research_concepts,
            keywords=paper.all_keywords,
            text=paper.full_text[:20_000],
        )
        paper.concept_hierarchy = tree
        paper.research_domains  = domains
        paper.mindmap = {"root": paper.title, "children": tree}

        return paper
