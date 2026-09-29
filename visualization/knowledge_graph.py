"""
================================================
visualization/knowledge_graph.py
Interactive knowledge graph using NetworkX + PyVis
and Plotly for static rendering.
================================================
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import plotly.graph_objects as go

from utils.logger import get_logger

log = get_logger(__name__)

try:
    from pyvis.network import Network as PyVisNetwork
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False
    log.warning("pyvis not installed; interactive HTML graph disabled.")

# Node colour by type
_TYPE_COLORS = {
    "paper":   "#FF6584",
    "domain":  "#6C63FF",
    "concept": "#43BCCD",
    "keyword": "#00C49A",
    "entity":  "#F9A826",
}


class KnowledgeGraph:
    """
    Builds a NetworkX graph from paper metadata and
    renders it as a Plotly chart or interactive PyVis HTML.
    """

    def __init__(self) -> None:
        self.G: nx.Graph = nx.Graph()

    # ──────────────────────────────────────────────────────────────────────────
    # Graph construction
    # ──────────────────────────────────────────────────────────────────────────

    def build_from_papers(self, papers_data: List[Dict]) -> "KnowledgeGraph":
        """
        Construct an accurate semantic graph from paper metadata.
        Builds a structured taxonomy network: Paper -> Domain -> Subtopics & Models -> Key Concepts.
        """
        self.G = nx.Graph()

        for paper in papers_data:
            title = paper.get("title", "Unknown")[:50]
            self.G.add_node(title, node_type="paper", label=title)

            domains = paper.get("research_domains", [])
            primary_domain = domains[0] if domains else "Artificial Intelligence"

            # Connect paper to verified domains
            for domain in domains:
                self.G.add_node(domain, node_type="domain", label=domain)
                self.G.add_edge(title, domain, weight=3, rel="belongs_to")

            # Structured concept attachment: attach concepts to domains, not directly to paper
            concepts = paper.get("technical_terms", []) + paper.get("keywords", [])
            seen_concepts = set()

            for c in concepts[:15]:
                c_clean = c.strip()
                if not c_clean or len(c_clean) < 3 or c_clean.lower() in seen_concepts:
                    continue
                seen_concepts.add(c_clean.lower())

                # Determine if concept relates directly to a domain
                matched_domain = None
                for d in domains:
                    if d.lower() in c_clean.lower() or c_clean.lower() in d.lower():
                        matched_domain = d
                        break

                target_parent = matched_domain if matched_domain else (domains[0] if domains else title)
                self.G.add_node(c_clean, node_type="concept", label=c_clean)
                self.G.add_edge(target_parent, c_clean, weight=2, rel="has_concept")

        # Add cross-paper shared concept links
        self._add_shared_concept_edges(papers_data)

        log.info(
            f"Knowledge graph: {self.G.number_of_nodes()} nodes, "
            f"{self.G.number_of_edges()} edges."
        )
        return self

    def build_from_hierarchy(
        self, tree: Dict, paper_title: str = "Paper"
    ) -> "KnowledgeGraph":
        """Build structured hierarchical graph from concept tree with proper taxonomic lineage."""
        self.G = nx.Graph()
        p_title = paper_title[:50]
        self.G.add_node(p_title, node_type="paper", label=p_title)
        self._add_tree_edges(tree, p_title)
        return self

    def build_from_ccc(self, ccc_data: Dict, paper_title: str = "Paper") -> "KnowledgeGraph":
        """Build semantic association graph directly from Concept-to-Concept (CCC) mapping."""
        self.G = nx.Graph()
        p_title = paper_title[:50]
        self.G.add_node(p_title, node_type="paper", label=p_title)

        edges = ccc_data.get("c2c_edges", [])
        bridges = ccc_data.get("cross_domain_bridges", [])
        alignments = ccc_data.get("domain_alignments", {})

        # Add domain nodes and align verified concepts
        for domain, d_concepts in alignments.items():
            if not d_concepts:
                continue
            self.G.add_node(domain, node_type="domain", label=domain)
            self.G.add_edge(p_title, domain, weight=2.5, rel="domain")
            for c in d_concepts[:8]:
                self.G.add_node(c, node_type="concept", label=c)
                self.G.add_edge(domain, c, weight=1.5, rel="domain_concept")

        # Add concept-to-concept direct semantic association edges
        for e in edges:
            u, v = e.get("source"), e.get("target")
            w = float(e.get("weight", 0.5))
            if u and v and u != v:
                self.G.add_node(u, node_type="concept", label=u)
                self.G.add_node(v, node_type="concept", label=v)
                self.G.add_edge(u, v, weight=w, rel=e.get("relation", "association"))

        # Highlight cross-domain bridge concepts
        for b in bridges:
            b_concept = b.get("concept")
            if b_concept:
                self.G.add_node(b_concept, node_type="entity", label=f"🌉 {b_concept}")
                for d in b.get("connected_domains", []):
                    self.G.add_node(d, node_type="domain", label=d)
                    self.G.add_edge(d, b_concept, weight=2.0, rel="cross_domain_bridge")

        return self

    # ──────────────────────────────────────────────────────────────────────────
    # Rendering
    # ──────────────────────────────────────────────────────────────────────────

    def to_plotly(self, title: str = "Knowledge Graph") -> go.Figure:
        """Render graph as a Plotly scatter figure."""
        if not self.G.nodes():
            return self._empty_fig(title)

        pos = self._layout()

        # ── Edge traces ───────────────────────────────────────────────────────
        edge_x, edge_y = [], []
        for u, v in self.G.edges():
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(width=0.8, color="#334"),
            hoverinfo="none",
        )

        # ── Node traces (grouped by type) ────────────────────────────────────
        node_traces = []
        type_groups: Dict[str, List] = {}
        for node in self.G.nodes(data=True):
            ntype = node[1].get("node_type", "concept")
            type_groups.setdefault(ntype, []).append(node[0])

        for ntype, nodes in type_groups.items():
            x_vals = [pos[n][0] for n in nodes if n in pos]
            y_vals = [pos[n][1] for n in nodes if n in pos]
            color  = _TYPE_COLORS.get(ntype, "#888")
            size   = {"paper": 20, "domain": 16, "concept": 12, "keyword": 9, "entity": 10}.get(ntype, 10)

            node_traces.append(
                go.Scatter(
                    x=x_vals, y=y_vals,
                    mode="markers+text",
                    name=ntype.capitalize(),
                    text=nodes,
                    textposition="top center",
                    textfont=dict(size=8, color="#ccc"),
                    marker=dict(
                        size=size,
                        color=color,
                        line=dict(width=1, color="#111"),
                        opacity=0.9,
                    ),
                    hovertemplate="<b>%{text}</b><br>Type: " + ntype + "<extra></extra>",
                )
            )

        fig = go.Figure(data=[edge_trace] + node_traces)
        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color="#6C63FF")),
            showlegend=True,
            legend=dict(
                title="Node Type",
                bgcolor="#1A1A2E",
                bordercolor="#444",
                font=dict(color="#ccc"),
            ),
            paper_bgcolor="#0F0F1A",
            plot_bgcolor="#0F0F1A",
            font=dict(color="#E0E0E0"),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=10, r=10, t=60, b=10),
            height=700,
        )
        return fig

    def to_pyvis_html(
        self, output_path: Optional[str] = None, title: str = "Knowledge Graph"
    ) -> str:
        """
        Render an interactive PyVis HTML file.

        Returns:
            Path to the generated HTML file.
        """
        if not HAS_PYVIS:
            raise ImportError("pyvis is not installed.")

        net = PyVisNetwork(
            notebook=False,
            height="700px",
            width="100%",
            bgcolor="#0F0F1A",
            font_color="#E0E0E0",
            directed=False,
        )
        net.set_options(json.dumps({
            "physics": {
                "enabled": True,
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 120,
                },
                "solver": "forceAtlas2Based",
            },
            "nodes": {"font": {"size": 12}},
            "edges": {"smooth": {"type": "continuous"}},
        }))

        for node, attrs in self.G.nodes(data=True):
            ntype = attrs.get("node_type", "concept")
            color = _TYPE_COLORS.get(ntype, "#888")
            size  = {"paper": 30, "domain": 22, "concept": 16, "keyword": 12, "entity": 14}.get(ntype, 14)
            net.add_node(node, label=node[:30], color=color, size=size, title=node)

        for u, v, data in self.G.edges(data=True):
            net.add_edge(u, v, width=data.get("weight", 1) * 0.5)

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
            output_path = tmp.name
            tmp.close()

        net.save_graph(output_path)
        log.info(f"PyVis graph saved to {output_path}")
        return output_path

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _layout(self) -> Dict:
        """Choose clean, non-overlapping layout with optimal node spacing."""
        try:
            # Try Kamada-Kawai for clean path and cluster separation
            if len(self.G.nodes) > 1:
                return nx.kamada_kawai_layout(self.G)
            return nx.spring_layout(self.G, seed=42, k=0.8, iterations=100)
        except Exception:
            try:
                return nx.spring_layout(self.G, seed=42, k=0.8, iterations=100)
            except Exception:
                return nx.random_layout(self.G, seed=42)

    def _add_tree_edges(self, tree: Dict, parent: str) -> None:
        """Recursively add tree edges to graph."""
        for key, children in tree.items():
            self.G.add_node(key, node_type="concept", label=key)
            self.G.add_edge(parent, key, weight=1)
            if isinstance(children, dict) and children:
                self._add_tree_edges(children, key)

    def _add_shared_concept_edges(self, papers_data: List[Dict]) -> None:
        """Add edges between papers sharing keywords/concepts."""
        paper_keywords: Dict[str, set] = {}
        for p in papers_data:
            title = p.get("title", "")[:60]
            kws = set(p.get("keywords", [])[:15])
            paper_keywords[title] = kws

        titles = list(paper_keywords.keys())
        for i in range(len(titles)):
            for j in range(i + 1, len(titles)):
                shared = paper_keywords[titles[i]] & paper_keywords[titles[j]]
                if shared:
                    weight = len(shared)
                    self.G.add_edge(titles[i], titles[j], weight=weight, rel="shared_concept")

    @staticmethod
    def _empty_fig(title: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(
            text="No graph data",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#888"),
        )
        fig.update_layout(title=title, paper_bgcolor="#0F0F1A")
        return fig
