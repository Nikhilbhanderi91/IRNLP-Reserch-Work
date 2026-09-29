"""
================================================
hierarchy/ccc_mapper.py
Concept-to-Concept & Cross-Domain Mapping (CCC)
================================================
Implements:
  1. Concept Extraction & Semantic Association (Co-occurrence + Dense Embedding Cosine)
  2. Concept-to-Concept (C2C) Directed & Weighted Graph
  3. Cross-Domain Alignment (Projecting domain bridge concepts)
  4. Global CCC Matrix & Similarity Graph for HMRP
"""

from __future__ import annotations

import re
import numpy as np
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict

from utils.logger import get_logger

log = get_logger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


class CCCMapper:
    """
    Constructs Concept-to-Concept and Cross-Domain Semantic Mappings (CCC)
    across individual papers and multi-domain corpora.
    """

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = embedding_model_name
        self._embedder: Optional[Any] = None

    def _get_embedder(self):
        if self._embedder is None and HAS_ST:
            try:
                self._embedder = SentenceTransformer(self.model_name)
            except Exception as e:
                log.warning(f"Could not load SentenceTransformer in CCCMapper: {e}")
                self._embedder = None
        return self._embedder

    def map_paper_ccc(
        self,
        concepts: List[str],
        keywords: List[str],
        domains: List[str],
        text: str = "",
        min_similarity: float = 0.40,
        max_edges_per_concept: int = 4,
    ) -> Dict[str, Any]:
        """
        Generates Concept-to-Concept (C2C) associations and Cross-Domain connections for a single paper.

        Returns:
            Dict with:
              - nodes: List of concepts with domain, type, frequency
              - c2c_edges: List of {source, target, weight, relation_type}
              - cross_domain_bridges: List of {concept, connected_domains, bridge_score}
              - domain_alignments: Dict mapping domain -> list of mapped concepts
        """
        all_concepts = []
        seen = set()
        for c in (concepts + keywords):
            c_clean = c.strip()
            if c_clean and len(c_clean) > 2 and c_clean.lower() not in seen:
                seen.add(c_clean.lower())
                all_concepts.append(c_clean)

        # Cap top concepts for clear visualization
        top_concepts = all_concepts[:30]

        if not top_concepts:
            return {
                "nodes": [],
                "c2c_edges": [],
                "cross_domain_bridges": [],
                "domain_alignments": {d: [] for d in domains}
            }

        # 1. Compute Semantic Similarity Matrix for concepts
        embedder = self._get_embedder()
        sim_matrix = np.zeros((len(top_concepts), len(top_concepts)))

        if embedder is not None and len(top_concepts) > 1:
            try:
                embeddings = embedder.encode(top_concepts, convert_to_numpy=True, normalize_embeddings=True)
                sim_matrix = np.dot(embeddings, embeddings.T)
            except Exception as e:
                log.warning(f"Dense concept embedding failed in CCCMapper: {e}")
                sim_matrix = self._compute_lexical_overlap_matrix(top_concepts)
        else:
            sim_matrix = self._compute_lexical_overlap_matrix(top_concepts)

        # 2. Text Co-occurrence Boost
        text_lower = text.lower() if text else ""
        edges: List[Dict[str, Any]] = []

        for i in range(len(top_concepts)):
            # Find top-K most similar partner concepts
            ranked_indices = np.argsort(-sim_matrix[i])
            edge_count = 0
            for j in ranked_indices:
                if i == j:
                    continue
                score = float(sim_matrix[i, j])

                # Co-occurrence verification
                c1_in = top_concepts[i].lower() in text_lower
                c2_in = top_concepts[j].lower() in text_lower
                if c1_in and c2_in:
                    score = min(1.0, score + 0.15)

                if score >= min_similarity:
                    edges.append({
                        "source": top_concepts[i],
                        "target": top_concepts[j],
                        "weight": round(score, 3),
                        "relation": "semantic_association" if score < 0.75 else "hierarchical_subsumption"
                    })
                    edge_count += 1
                    if edge_count >= max_edges_per_concept:
                        break

        # 3. Domain Alignments & Cross-Domain Bridges
        domain_alignments = {d: [] for d in domains}
        cross_domain_bridges = []

        for c_idx, concept in enumerate(top_concepts):
            c_lower = concept.lower()
            matching_domains = []
            for d in domains:
                d_lower = d.lower()
                if c_lower in d_lower or d_lower in c_lower:
                    domain_alignments[d].append(concept)
                    matching_domains.append(d)
                else:
                    # Semantic connection between concept and domain
                    if embedder is not None:
                        d_emb = embedder.encode([d], convert_to_numpy=True, normalize_embeddings=True)
                        c_emb = embedder.encode([concept], convert_to_numpy=True, normalize_embeddings=True)
                        d_sim = float(np.dot(c_emb, d_emb.T)[0, 0])
                        if d_sim >= 0.45:
                            domain_alignments[d].append(concept)
                            matching_domains.append(d)

            if len(matching_domains) > 1:
                cross_domain_bridges.append({
                    "concept": concept,
                    "connected_domains": matching_domains,
                    "bridge_score": len(matching_domains)
                })

        nodes = [
            {
                "id": c,
                "label": c,
                "type": "concept",
                "in_degree": sum(1 for e in edges if e["target"] == c),
                "out_degree": sum(1 for e in edges if e["source"] == c),
            }
            for c in top_concepts
        ]

        return {
            "nodes": nodes,
            "c2c_edges": edges,
            "cross_domain_bridges": cross_domain_bridges,
            "domain_alignments": domain_alignments
        }

    def _compute_lexical_overlap_matrix(self, concepts: List[str]) -> np.ndarray:
        """Fallback token-overlap Jaccard matrix when embeddings are unavailable."""
        n = len(concepts)
        matrix = np.zeros((n, n))
        for i in range(n):
            tokens_i = set(re.findall(r"\w+", concepts[i].lower()))
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                    continue
                tokens_j = set(re.findall(r"\w+", concepts[j].lower()))
                union = tokens_i | tokens_j
                if union:
                    matrix[i, j] = len(tokens_i & tokens_j) / len(union)
        return matrix

    def map_corpus_ccc(self, papers: List[Any]) -> Dict[str, Any]:
        """
        Builds a global Corpus-Level Concept-to-Concept and Cross-Domain network across all papers.
        """
        global_concept_domains: Dict[str, Set[str]] = defaultdict(set)
        global_concept_freq: Dict[str, int] = defaultdict(int)
        global_cooccurrences: Dict[Tuple[str, str], float] = defaultdict(float)

        for paper in papers:
            p_domains = getattr(paper, "research_domains", [])
            if not p_domains and hasattr(paper, "get"):
                p_domains = paper.get("research_domains", [])
                
            p_concepts = getattr(paper, "research_concepts", []) or getattr(paper, "all_keywords", [])
            if not p_concepts and hasattr(paper, "get"):
                p_concepts = paper.get("research_concepts", []) or paper.get("all_keywords", [])

            top_p_concepts = p_concepts[:15]
            for c in top_p_concepts:
                c_clean = c.strip()
                if c_clean:
                    global_concept_freq[c_clean] += 1
                    for d in p_domains:
                        global_concept_domains[c_clean].add(d)

            # Record co-occurrences
            for i in range(len(top_p_concepts)):
                for j in range(i + 1, len(top_p_concepts)):
                    c1, c2 = sorted([top_p_concepts[i], top_p_concepts[j]])
                    global_cooccurrences[(c1, c2)] += 1.0

        # Filter top corpus bridge concepts
        bridges = []
        for c, doms in global_concept_domains.items():
            if len(doms) >= 2 and global_concept_freq[c] >= 2:
                bridges.append({
                    "concept": c,
                    "frequency": global_concept_freq[c],
                    "domains": list(doms),
                    "domain_span": len(doms)
                })

        bridges.sort(key=lambda x: (x["domain_span"], x["frequency"]), reverse=True)

        return {
            "total_unique_concepts": len(global_concept_freq),
            "cross_domain_bridge_concepts": bridges[:50],
            "top_cooccurring_concept_pairs": [
                {"concept_a": k[0], "concept_b": k[1], "cooccurrence_count": v}
                for k, v in sorted(global_cooccurrences.items(), key=lambda x: x[1], reverse=True)[:50]
            ]
        }
