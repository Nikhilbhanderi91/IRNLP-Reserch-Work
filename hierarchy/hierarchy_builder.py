"""
================================================
hierarchy/hierarchy_builder.py
Orchestrates domain detection and hierarchy
construction for a Paper object.
================================================
"""

import re
from typing import Dict, List, Tuple

from utils.logger import get_logger
from hierarchy.concept_hierarchy import ConceptHierarchy

log = get_logger(__name__)

# Import domain taxonomy from config
try:
    from app.config import DOMAIN_TAXONOMY
except ImportError:
    DOMAIN_TAXONOMY: Dict[str, List[str]] = {}


class HierarchyBuilder:
    """
    High-level orchestrator that:
      1. Detects research domains from concepts/keywords
      2. Delegates hierarchy construction to ConceptHierarchy
      3. Returns the final tree + flat list
    """

    def __init__(self) -> None:
        self._hierarchy = ConceptHierarchy()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def build(
        self,
        concepts: List[str],
        keywords: List[str],
        text: str = "",
    ) -> Tuple[Dict, List[str]]:
        """
        Build a concept hierarchy for a single paper.

        Args:
            concepts: All extracted research concepts.
            keywords: Extracted keywords.
            text:     Full text (used for domain scoring).

        Returns:
            (hierarchy_tree, detected_domains)
        """
        domains = self.detect_domains(concepts + keywords, text)
        log.info(f"Detected domains: {domains}")

        tree = self._hierarchy.build(concepts, domains, keywords)
        return tree, domains

    def detect_domains(
        self,
        terms: List[str],
        text: str = "",
        top_n: int = 5,
    ) -> List[str]:
        """
        Score each domain from DOMAIN_TAXONOMY by how many of its
        seed keywords appear in the paper's terms/text.

        Args:
            terms:  Combined concept + keyword list.
            text:   Full text for broader matching.
            top_n:  Max domains to return.

        Returns:
            List of domain names sorted by relevance score.
        """
        combined = " ".join(terms).lower()
        if text:
            combined += " " + text[:20_000].lower()

        scores: Dict[str, int] = {}
        for domain, seed_keywords in DOMAIN_TAXONOMY.items():
            score = sum(
                combined.count(kw.lower()) for kw in seed_keywords
            )
            # Require at least 3 keyword occurrences to prevent false positives (like Bioinformatics)
            if score >= 3:
                scores[domain] = score

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:top_n]]

    def get_hierarchy_paths(self, tree: Dict, prefix: str = "") -> List[str]:
        """
        Return all root-to-leaf paths as strings like:
          "AI > Machine Learning > Deep Learning > CNN"
        """
        paths: List[str] = []
        for key, children in tree.items():
            current_path = f"{prefix} > {key}" if prefix else key
            if not children:
                paths.append(current_path)
            else:
                paths.extend(self.get_hierarchy_paths(children, current_path))
        return paths

    def flatten_tree(self, tree: Dict, depth: int = 0) -> List[Tuple[str, str, int]]:
        """
        Flatten to list of (node, parent, depth) tuples for graph building.
        """
        result: List[Tuple[str, str, int]] = []
        for key, children in tree.items():
            result.append((key, "", depth))
            if isinstance(children, dict):
                for child, grandchildren in children.items():
                    result.append((child, key, depth + 1))
                    if isinstance(grandchildren, dict):
                        for gc, ggc in grandchildren.items():
                            result.append((gc, child, depth + 2))
                            if isinstance(ggc, dict):
                                for ggc_k in ggc:
                                    result.append((ggc_k, gc, depth + 3))
        return result
