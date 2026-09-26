"""
================================================
models/paper.py
Data model representing a single research paper
with complete identity tracking & Section 6 JSON schema.
================================================
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class Paper:
    """
    Represents a parsed and processed research paper.
    Holds all extracted fields, identity hashes, and computed attributes.
    """

    # ── Identity ──────────────────────────────────────
    paper_id: str = ""
    file_hash: str = ""
    analysis_id: str = ""
    analysis_version: str = "2.0"
    file_name: str = ""
    file_path: str = ""
    year: Optional[str] = None

    # ── Extracted raw text ─────────────────────────────
    raw_text: str = ""
    title: str = ""
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    introduction: str = ""
    full_text: str = ""

    # ── Processed text ─────────────────────────────────
    cleaned_text: str = ""
    tokens: List[str] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)

    # ── Keywords ───────────────────────────────────────
    tfidf_keywords: List[Dict] = field(default_factory=list)   # [{word, score}]
    keybert_keywords: List[Dict] = field(default_factory=list) # [{word, score}]
    yake_keywords: List[Dict] = field(default_factory=list)    # [{word, score}]
    all_keywords: List[str] = field(default_factory=list)      # merged unique list

    # ── Paper Analysis Details ────────────────────────
    problem: Optional[str] = None
    objective: Optional[str] = None
    existing_methods: List[str] = field(default_factory=list)
    proposed_method: Dict[str, str] = field(default_factory=lambda: {"name": "", "description": ""})
    models: List[str] = field(default_factory=list)
    algorithms: List[str] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)
    evaluation_metrics: List[str] = field(default_factory=list)
    results: List[str] = field(default_factory=list)
    advantages: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    research_gap: List[str] = field(default_factory=list)
    future_work: List[str] = field(default_factory=list)

    # ── Concepts & Traceability ───────────────────────
    named_entities: List[Dict] = field(default_factory=list)   # [{text, label}]
    technical_terms: List[str] = field(default_factory=list)
    research_concepts: List[str] = field(default_factory=list)
    research_domains: List[str] = field(default_factory=list)
    traceable_concepts: List[Dict] = field(default_factory=list) # [{concept, source, confidence, evidence, page}]

    # ── Hierarchy & Visualizations ────────────────────
    concept_hierarchy: Dict = field(default_factory=dict)      # nested dict tree
    mindmap: Dict = field(default_factory=dict)

    # ── Metadata ───────────────────────────────────────
    page_count: int = 0
    word_count: int = 0
    char_count: int = 0

    # ── Embeddings & Similarity ───────────────────────
    embedding: Optional[List[float]] = None
    similarity_data: Dict[str, Any] = field(default_factory=dict)

    def summary_dict(self) -> Dict:
        """Return a lightweight JSON-serialisable summary."""
        return {
            "paper_id": self.paper_id,
            "file_hash": self.file_hash,
            "file_name": self.file_name,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "abstract": self.abstract[:2000],
            "keywords": self.all_keywords[:20],
            "technical_terms": self.technical_terms[:20],
            "research_domains": self.research_domains,
            "page_count": self.page_count,
            "word_count": self.word_count,
        }

    def to_structured_json(self) -> Dict[str, Any]:
        """
        Generates the standard paper-specific JSON structure conforming to Section 6 schema.
        """
        return {
            "paper": {
                "paper_id": self.paper_id or None,
                "file_hash": self.file_hash or None,
                "file_name": self.file_name or None,
                "title": self.title or None,
                "authors": self.authors,
                "year": self.year or None,
                "abstract": self.abstract or None,
                "keywords": self.all_keywords,
                "research_domain": self.research_domains,
                "page_count": self.page_count,
                "word_count": self.word_count,
            },
            "problem": self.problem or None,
            "objective": self.objective or None,
            "existing_methods": self.existing_methods,
            "proposed_method": self.proposed_method,
            "models": self.models,
            "algorithms": self.algorithms,
            "datasets": self.datasets,
            "evaluation_metrics": self.evaluation_metrics,
            "results": self.results,
            "advantages": self.advantages,
            "limitations": self.limitations,
            "research_gap": self.research_gap,
            "future_work": self.future_work,
            "technical_concepts": self.technical_terms,
            "entities": self.named_entities,
            "traceable_concepts": self.traceable_concepts,
            "embeddings": {"cls_embedding": self.embedding} if self.embedding else {},
            "similarity": self.similarity_data,
            "hierarchy": self.concept_hierarchy,
            "mindmap": self.mindmap if self.mindmap else {"root": self.title, "children": self.concept_hierarchy}
        }

    def __repr__(self) -> str:
        return f"<Paper paper_id={self.paper_id[:8]!r} title={self.title!r} pages={self.page_count}>"
