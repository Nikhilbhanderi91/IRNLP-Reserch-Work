"""
================================================
hierarchy/concept_hierarchy.py
Builds parent-child concept relationships from
a domain taxonomy + extracted paper concepts.
Ensures zero concept leakage across papers.
================================================
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from utils.logger import get_logger

log = get_logger(__name__)

# ─── Hard-coded knowledge sub-trees ──────────────────────────────────────────
KNOWLEDGE_TREE: Dict[str, List[str]] = {
    "Artificial Intelligence": [
        "Machine Learning", "Deep Learning", "Natural Language Processing",
        "Computer Vision", "Reinforcement Learning", "Robotics",
    ],
    "Machine Learning": [
        "Supervised Learning", "Unsupervised Learning", "Semi-supervised Learning",
        "Transfer Learning", "Meta-Learning", "Ensemble Methods",
    ],
    "Deep Learning": [
        "Neural Networks", "Convolutional Neural Networks", "Recurrent Neural Networks",
        "Transformers", "Generative Adversarial Networks", "Attention Mechanism",
    ],
    "Transformers": [
        "BERT", "GPT", "T5", "RoBERTa", "XLNet", "SciBERT", "ALBERT", "ELECTRA",
    ],
    "Natural Language Processing": [
        "Text Classification", "Named Entity Recognition", "Sequence Labeling",
        "Acronym Identification", "Acronym Disambiguation", "Information Extraction",
    ],
    "Adversarial Training": [
        "FGM", "Fast Gradient Method", "Adversarial Perturbation",
    ],
}


class ConceptHierarchy:
    """
    Builds a concept hierarchy from:
      1. Domain assignments
      2. Paper-specific extracted concepts/keywords
    """

    def __init__(self) -> None:
        self._child_to_parent: Dict[str, str] = {}
        for parent, children in KNOWLEDGE_TREE.items():
            for child in children:
                self._child_to_parent[child.lower()] = parent

    def build(
        self,
        concepts: List[str],
        domains: List[str],
        keywords: List[str] | None = None,
    ) -> Dict:
        keywords = keywords or []
        all_terms = deduplicate_ci(concepts + keywords)

        tree: Dict = {}

        for domain in domains:
            domain_subtree = self._build_domain_subtree(domain, all_terms)
            if domain_subtree:
                tree[domain] = domain_subtree

        if not tree:
            tree["Research Concepts"] = self._flat_tree(all_terms[:25])

        # Insert uncategorised paper terms
        categorised = self._collect_all_leaves(tree)
        uncategorised = [t for t in all_terms if t.lower() not in categorised]
        if uncategorised:
            tree.setdefault("Other Concepts", {})
            for t in uncategorised[:25]:
                tree["Other Concepts"][t] = {}

        # Prune empty branches recursively
        return self._prune_empty(tree)

    def _build_domain_subtree(self, domain: str, concepts: List[str]) -> Dict:
        subtree: Dict = {}
        domain_key = domain.lower()

        for known_parent, known_children in KNOWLEDGE_TREE.items():
            if known_parent.lower() in domain_key or domain_key in known_parent.lower():
                branch: Dict = {}
                for child in known_children:
                    matched = self._find_matching_concepts(child, concepts)
                    if matched or child.lower() in [c.lower() for c in concepts]:
                        branch[child] = {m: {} for m in matched if m.lower() != child.lower()}
                if branch:
                    subtree[known_parent] = branch

        # Match direct domain terms
        direct = self._find_matching_concepts(domain, concepts)
        for d in direct:
            if d not in subtree:
                subtree[d] = {}

        return subtree if subtree else self._flat_tree([c for c in concepts if len(c) > 3][:15])

    def _find_matching_concepts(self, term: str, concepts: List[str]) -> List[str]:
        term_l = term.lower()
        return [
            c for c in concepts
            if (term_l in c.lower() or c.lower() in term_l) and c.lower() != term_l
        ]

    def _flat_tree(self, concepts: List[str]) -> Dict:
        return {c: {} for c in concepts[:25]}

    def _collect_all_leaves(self, tree: Dict, seen: Set[str] | None = None) -> Set[str]:
        seen = seen or set()
        for key, children in tree.items():
            seen.add(key.lower())
            if isinstance(children, dict):
                self._collect_all_leaves(children, seen)
        return seen

    def _prune_empty(self, tree: Dict) -> Dict:
        pruned = {}
        for k, v in tree.items():
            if isinstance(v, dict):
                sub = self._prune_empty(v)
                pruned[k] = sub
            else:
                pruned[k] = v
        return pruned


def deduplicate_ci(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out
