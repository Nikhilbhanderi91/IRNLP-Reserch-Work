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

# ─── Multi-level Taxonomy Knowledge Graph ────────────────────────────────────
TAXONOMY_PARENTS: Dict[str, str] = {
    # AI -> ML -> DL
    "Machine Learning": "Artificial Intelligence",
    "Deep Learning": "Machine Learning",
    "Reinforcement Learning": "Machine Learning",
    "Supervised Learning": "Machine Learning",
    "Unsupervised Learning": "Machine Learning",
    "Semi-supervised Learning": "Machine Learning",
    "Transfer Learning": "Machine Learning",
    "Federated Learning": "Machine Learning",

    # DL -> Models & Architectures
    "Neural Networks": "Deep Learning",
    "Convolutional Neural Networks": "Deep Learning",
    "Recurrent Neural Networks": "Deep Learning",
    "Transformers": "Deep Learning",
    "Generative Adversarial Networks": "Deep Learning",
    "Attention Mechanism": "Deep Learning",
    "Diffusion Models": "Deep Learning",
    "State Space Models": "Deep Learning",

    # Transformers -> Specific Models
    "BERT": "Transformers",
    "RoBERTa": "Transformers",
    "SciBERT": "Transformers",
    "BioBERT": "Transformers",
    "ALBERT": "Transformers",
    "ELECTRA": "Transformers",
    "GPT": "Transformers",
    "T5": "Transformers",
    "LLaMA": "Transformers",
    "Transformer": "Transformers",

    # NLP -> Tasks & Techniques
    "Text Classification": "Natural Language Processing",
    "Named Entity Recognition": "Natural Language Processing",
    "Sequence Labeling": "Natural Language Processing",
    "Acronym Identification": "Natural Language Processing",
    "Acronym Disambiguation": "Natural Language Processing",
    "Information Extraction": "Natural Language Processing",
    "Sentiment Analysis": "Natural Language Processing",
    "Machine Translation": "Natural Language Processing",
    "Question Answering": "Natural Language Processing",
    "Adversarial Training": "Deep Learning",
    "FGM": "Adversarial Training",
    "Fast Gradient Method": "Adversarial Training",

    # Computer Vision
    "Object Detection": "Computer Vision",
    "Image Segmentation": "Computer Vision",
    "Point Tracking": "Computer Vision",
    "Optical Flow": "Computer Vision",
    "Visual Representation": "Computer Vision",
    "Visuomotor Control": "Robotics",
}

CONCEPT_SYNONYMS: Dict[str, str] = {
    "bert": "BERT",
    "roberta": "RoBERTa",
    "scibert": "SciBERT",
    "transformer": "Transformers",
    "transformers": "Transformers",
    "deep learning": "Deep Learning",
    "machine learning": "Machine Learning",
    "neural network": "Neural Networks",
    "neural networks": "Neural Networks",
    "cnn": "Convolutional Neural Networks",
    "rnn": "Recurrent Neural Networks",
    "gan": "Generative Adversarial Networks",
    "attention": "Attention Mechanism",
    "ner": "Named Entity Recognition",
    "named entity recognition": "Named Entity Recognition",
    "acronym identification": "Acronym Identification",
    "acronym disambiguation": "Acronym Disambiguation",
    "information extraction": "Information Extraction",
    "adversarial training": "Adversarial Training",
    "fgm": "FGM",
    "fast gradient method": "Fast Gradient Method",
    "point tracking": "Point Tracking",
    "visuomotor control": "Visuomotor Control",
}


class ConceptHierarchy:
    """
    Builds a structured concept hierarchy from:
      1. Detected domain assignments
      2. Paper-specific extracted concepts and keywords
    """

    def __init__(self) -> None:
        pass

    def build(
        self,
        concepts: List[str],
        domains: List[str],
        keywords: List[str] | None = None,
    ) -> Dict:
        keywords = keywords or []
        all_terms = deduplicate_ci(concepts + keywords)

        if not domains:
            domains = ["Artificial Intelligence"]

        tree: Dict = {}
        assigned_terms: Set[str] = set()

        for domain in domains:
            # Build subtree for each detected domain
            dom_branch: Dict = {}
            for raw_term in all_terms:
                canonical = self._canonical_name(raw_term)
                chain = self._get_lineage(canonical, target_root=domain)
                if chain:
                    self._insert_chain(dom_branch, chain[1:])  # skip domain itself
                    assigned_terms.add(raw_term.lower())
                elif domain.lower() in raw_term.lower() and raw_term.lower() != domain.lower():
                    dom_branch.setdefault(raw_term, {})
                    assigned_terms.add(raw_term.lower())

            if dom_branch:
                tree[domain] = dom_branch

        # Collect unassigned terms into domain or Other Concepts
        unassigned = [t for t in all_terms if t.lower() not in assigned_terms]
        if unassigned:
            # Attach general concepts to first domain if possible or Other Concepts
            if tree and len(tree) == 1:
                first_dom = list(tree.keys())[0]
                for t in unassigned[:20]:
                    tree[first_dom].setdefault(t, {})
            else:
                tree.setdefault("Other Concepts", {})
                for t in unassigned[:20]:
                    tree["Other Concepts"].setdefault(t, {})

        if not tree:
            tree["Research Concepts"] = {c: {} for c in all_terms[:25]}

        return tree

    def _canonical_name(self, term: str) -> str:
        term_clean = term.strip().lower()
        if term_clean in CONCEPT_SYNONYMS:
            return CONCEPT_SYNONYMS[term_clean]
        for syn_key, canon in CONCEPT_SYNONYMS.items():
            if syn_key in term_clean or term_clean in syn_key:
                return canon
        return term.strip()

    def _get_lineage(self, concept: str, target_root: str) -> List[str] | None:
        """Returns [target_root, ..., concept] if concept belongs to target_root, else None."""
        curr = concept
        chain = [curr]
        visited = set()
        while curr in TAXONOMY_PARENTS and curr not in visited:
            visited.add(curr)
            parent = TAXONOMY_PARENTS[curr]
            chain.insert(0, parent)
            curr = parent

        if chain[0].lower() == target_root.lower() or (
            target_root == "Artificial Intelligence" and chain[0] in ["Natural Language Processing", "Computer Vision", "Machine Learning", "Deep Learning"]
        ):
            if chain[0] != target_root:
                chain.insert(0, target_root)
            return chain
        return None

    def _insert_chain(self, node: Dict, chain: List[str]) -> None:
        if not chain:
            return
        head, *tail = chain
        node.setdefault(head, {})
        if tail:
            self._insert_chain(node[head], tail)


def deduplicate_ci(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out
