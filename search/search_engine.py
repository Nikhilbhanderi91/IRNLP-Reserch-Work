"""
================================================
search/search_engine.py
Keyword / concept / topic search with
fuzzy matching and result highlighting.
================================================
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from utils.logger import get_logger

log = get_logger(__name__)

try:
    from difflib import SequenceMatcher
    HAS_DIFFLIB = True
except ImportError:
    HAS_DIFFLIB = False


@dataclass
class SearchResult:
    """A single search match."""
    paper_title: str
    match_type: str        # "keyword" | "concept" | "entity" | "text"
    matched_term: str
    score: float
    context: str = ""      # surrounding text snippet
    highlighted: str = ""  # HTML-highlighted version


class SearchEngine:
    """
    Searches across all loaded papers for a query string.
    Supports:
      - Exact keyword/concept match
      - Fuzzy substring match
      - Full-text context search with snippet extraction
    """

    def __init__(self, fuzzy_threshold: float = 0.7) -> None:
        self.fuzzy_threshold = fuzzy_threshold

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        papers_data: List[Dict],
        search_in: str = "all",  # "keywords" | "concepts" | "text" | "all"
    ) -> List[SearchResult]:
        """
        Search query across all papers.

        Args:
            query:       Search string.
            papers_data: List of paper dicts (from Paper.summary_dict()).
            search_in:   Which fields to search.

        Returns:
            List of SearchResult sorted by score descending.
        """
        if not query or not papers_data:
            return []

        query = query.strip()
        results: List[SearchResult] = []

        for paper in papers_data:
            title = paper.get("title", "Unknown")

            if search_in in ("keywords", "all"):
                results.extend(
                    self._match_list(query, paper.get("keywords", []),
                                     title, "keyword")
                )

            if search_in in ("concepts", "all"):
                results.extend(
                    self._match_list(query, paper.get("technical_terms", []),
                                     title, "concept")
                )
                results.extend(
                    self._match_list(query, paper.get("research_domains", []),
                                     title, "domain")
                )

            if search_in in ("text", "all"):
                text_results = self._match_text(
                    query,
                    paper.get("abstract", "") + " " + paper.get("full_text", "")[:5000],
                    title,
                )
                results.extend(text_results)

        # Deduplicate and sort
        seen = set()
        unique: List[SearchResult] = []
        for r in results:
            key = (r.paper_title, r.matched_term.lower())
            if key not in seen:
                seen.add(key)
                unique.append(r)

        return sorted(unique, key=lambda r: r.score, reverse=True)

    def search_hierarchy(
        self,
        query: str,
        tree: Dict,
        prefix: str = "",
    ) -> List[str]:
        """
        Search a concept hierarchy tree, returning matching node paths.

        Returns:
            List of strings like "AI > Machine Learning > BERT"
        """
        results: List[str] = []
        query_l = query.lower()

        for key, children in tree.items():
            current_path = f"{prefix} > {key}" if prefix else key
            if query_l in key.lower() or self._fuzzy_score(query_l, key.lower()) >= self.fuzzy_threshold:
                results.append(current_path)
            if isinstance(children, dict):
                results.extend(self.search_hierarchy(query, children, current_path))

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Internal methods
    # ──────────────────────────────────────────────────────────────────────────

    def _match_list(
        self,
        query: str,
        terms: List[str],
        paper_title: str,
        match_type: str,
    ) -> List[SearchResult]:
        """Match query against a list of terms (exact + fuzzy)."""
        results: List[SearchResult] = []
        query_l = query.lower()

        for term in terms:
            term_l = term.lower()
            exact   = query_l in term_l or term_l in query_l
            if exact:
                score = 1.0
            else:
                score = self._fuzzy_score(query_l, term_l)

            if score >= self.fuzzy_threshold:
                highlighted = self._highlight(term, query)
                results.append(
                    SearchResult(
                        paper_title=paper_title,
                        match_type=match_type,
                        matched_term=term,
                        score=round(score, 3),
                        context="",
                        highlighted=highlighted,
                    )
                )

        return results

    def _match_text(
        self,
        query: str,
        text: str,
        paper_title: str,
        context_chars: int = 150,
    ) -> List[SearchResult]:
        """Search full text with context extraction."""
        if not text:
            return []

        results: List[SearchResult] = []
        pattern = re.compile(re.escape(query), re.I)

        for match in list(pattern.finditer(text))[:5]:  # max 5 snippets
            start = max(0, match.start() - context_chars)
            end   = min(len(text), match.end() + context_chars)
            context = "…" + text[start:end].strip() + "…"
            highlighted = self._highlight(context, query)

            results.append(
                SearchResult(
                    paper_title=paper_title,
                    match_type="text",
                    matched_term=match.group(),
                    score=0.85,
                    context=context,
                    highlighted=highlighted,
                )
            )

        return results

    @staticmethod
    def _fuzzy_score(a: str, b: str) -> float:
        """Compute similarity ratio between two strings."""
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def _highlight(text: str, query: str) -> str:
        """Wrap query matches with HTML <mark> tags."""
        if not query:
            return text
        pattern = re.compile(re.escape(query), re.I)
        return pattern.sub(
            lambda m: f"<mark style='background:#6C63FF;color:#fff;padding:0 2px;border-radius:3px'>{m.group()}</mark>",
            text,
        )
