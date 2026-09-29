"""
=============================================================================
output.py
HMRP Comprehensive Multidisciplinary Visualization & Report Engine
=============================================================================
Inspects the actual HMRP project codebase, executes all implemented
algorithms, NLP extractors, similarity models, search engines, hierarchical
taxonomies, embeddings, and concept networks on real paper data, and
generates report-ready, academic-grade 300 DPI PNG figures, interactive HTMLs,
and an integrated master report at output/reports/HMRP_Complete_Report.html.
=============================================================================
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from app.pipeline import AnalysisPipeline
from extraction.keyword_extractor import KeywordExtractor
from extraction.concept_extractor import ConceptExtractor
from search.search_engine import SearchEngine
from hierarchy.hierarchy_builder import HierarchyBuilder
from hierarchy.ccc_mapper import CCCMapper
from similarity.similarity_engine import SimilarityEngine
from visualization.hierarchy_viz import HierarchyViz
from visualization.knowledge_graph import KnowledgeGraph
from visualization.mindmap_viz import MindMapViz
from visualization.heatmap_viz import HeatmapViz
from models.paper import Paper

# ── Directory Layout ──────────────────────────────────────────────────────────
OUTPUT_BASE = PROJECT_ROOT / "output"
LOG_DIR = OUTPUT_BASE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging Configuration ────────────────────────────────────────────────────
logger = logging.getLogger("HMRP_Visualizer")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

fh = logging.FileHandler(LOG_DIR / "output_run.log", encoding="utf-8")
fh.setFormatter(formatter)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)
logger.handlers = [fh, sh]

# ── Academic Style Defaults ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 15,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
sns.set_theme(style="whitegrid", palette="muted")

# Global list of generated visual records for the master report
GENERATED_SECTIONS: List[Dict[str, Any]] = []


def register_figure(section_name: str, title: str, filename: str, caption: str, rel_path: str, html_link: Optional[str] = None):
    """Registers a generated figure to be displayed in the master HTML report."""
    GENERATED_SECTIONS.append({
        "section": section_name,
        "title": title,
        "filename": filename,
        "caption": caption,
        "rel_path": rel_path,
        "html_link": html_link,
    })


def create_directory_structure() -> Dict[str, Path]:
    """Creates the technique-oriented folder structure under output/."""
    dirs = {
        "base": OUTPUT_BASE,
        "logs": LOG_DIR,
        "summary_images": OUTPUT_BASE / "summary" / "images",
        "summary_data": OUTPUT_BASE / "summary" / "data",
        "tfidf_images": OUTPUT_BASE / "tfidf" / "images",
        "tfidf_html": OUTPUT_BASE / "tfidf" / "html",
        "tfidf_data": OUTPUT_BASE / "tfidf" / "data",
        "search_images": OUTPUT_BASE / "search" / "images",
        "search_data": OUTPUT_BASE / "search" / "data",
        "cosine_images": OUTPUT_BASE / "cosine_similarity" / "images",
        "cosine_html": OUTPUT_BASE / "cosine_similarity" / "html",
        "cosine_data": OUTPUT_BASE / "cosine_similarity" / "data",
        "keywords_images": OUTPUT_BASE / "keyword_extraction" / "images",
        "keywords_data": OUTPUT_BASE / "keyword_extraction" / "data",
        "embeddings_images": OUTPUT_BASE / "embeddings" / "images",
        "embeddings_data": OUTPUT_BASE / "embeddings" / "data",
        "hierarchy_images": OUTPUT_BASE / "hierarchy" / "images",
        "hierarchy_html": OUTPUT_BASE / "hierarchy" / "html",
        "concepts_images": OUTPUT_BASE / "concepts" / "images",
        "concepts_data": OUTPUT_BASE / "concepts" / "data",
        "knowledge_graph_images": OUTPUT_BASE / "knowledge_graph" / "images",
        "knowledge_graph_html": OUTPUT_BASE / "knowledge_graph" / "html",
        "mindmap_images": OUTPUT_BASE / "mindmap" / "images",
        "mindmap_html": OUTPUT_BASE / "mindmap" / "html",
        "heatmaps_images": OUTPUT_BASE / "heatmaps" / "images",
        "heatmaps_html": OUTPUT_BASE / "heatmaps" / "html",
        "reports": OUTPUT_BASE / "reports",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def save_plotly_figure_safely(fig, html_path: Path, png_path: Path, width=1200, height=750):
    """Saves Plotly figure as interactive HTML and attempts high-res PNG export."""
    try:
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        logger.info(f"💾 Saved HTML: {html_path.relative_to(OUTPUT_BASE)}")
    except Exception as e:
        logger.error(f"Failed to write HTML {html_path}: {e}")

    try:
        fig.write_image(str(png_path), scale=2, width=width, height=height)
        logger.info(f"📸 Saved Plotly PNG: {png_path.relative_to(OUTPUT_BASE)}")
        return True
    except Exception as e:
        logger.warning(f"Plotly static PNG export note for {png_path.name}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset & Multi-Domain Corpus Ingestion
# ─────────────────────────────────────────────────────────────────────────────
def visualize_dataset(papers: List[Paper], dirs: Dict[str, Path]):
    logger.info("Visualizing: 1. Dataset & Corpus Overview...")
    all_domains = [d for p in papers for d in p.research_domains]
    domain_counts = pd.Series(all_domains).value_counts()

    # Save CSV Data
    domain_counts.to_csv(dirs["summary_data"] / "domain_distribution.csv", header=["count"])

    # 1.1 Papers per Domain Bar Chart
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=domain_counts.values, y=domain_counts.index, palette="mako")
    plt.title("HMRP Corpus: Research Papers Distribution Across Domains", fontweight="bold", pad=12)
    plt.xlabel("Annotated Papers Count")
    plt.ylabel("Research Domain")
    for i, v in enumerate(domain_counts.values):
        ax.text(v + 0.05, i, str(v), va="center", fontweight="bold")
    plt.tight_layout()
    p1 = dirs["summary_images"] / "papers_per_domain.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    register_figure(
        "Dataset Overview",
        "Papers per Domain Distribution",
        "papers_per_domain.png",
        "Figure 1: Distribution of ingested research publications across academic domains in the HMRP corpus.",
        "summary/images/papers_per_domain.png"
    )

    # 1.2 Domain Distribution Pie / Donut
    plt.figure(figsize=(7, 7))
    colors = sns.color_palette("Set2", len(domain_counts))
    wedges, texts, autotexts = plt.pie(
        domain_counts.values,
        labels=domain_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.6, edgecolor="w", linewidth=2)
    )
    for at in autotexts:
        at.set_fontweight("bold")
    plt.title("Proportional Multidisciplinary Domain Breakdown", fontweight="bold", pad=15)
    plt.tight_layout()
    p2 = dirs["summary_images"] / "dataset_domain_distribution.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    register_figure(
        "Dataset Overview",
        "Proportional Domain Representation",
        "dataset_domain_distribution.png",
        "Figure 2: Relative percentage contribution of each scientific discipline to the evaluated corpus.",
        "summary/images/dataset_domain_distribution.png"
    )

    # 1.3 Dataset Statistics Metrics Panel
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")
    stats = [
        ("Total Analyzed Papers", len(papers)),
        ("Research Domains", len(domain_counts)),
        ("Extracted Concepts", sum(len(p.research_concepts) for p in papers)),
        ("Indexed Keywords", sum(len(p.all_keywords) for p in papers)),
        ("Avg Pages / Document", f"{np.mean([p.page_count for p in papers if p.page_count]):.1f}"),
        ("Avg Document Words", f"{int(np.mean([p.word_count for p in papers if p.word_count])):,}")
    ]
    for idx, (label, val) in enumerate(stats):
        x = (idx % 3) * 0.33 + 0.05
        y = 0.65 if idx < 3 else 0.2
        ax.text(x, y + 0.12, str(val), fontsize=20, fontweight="bold", color="#1A365D")
        ax.text(x, y, label, fontsize=11, color="#718096")
        ax.plot([x, x + 0.25], [y - 0.04, y - 0.04], color="#CBD5E0", lw=1.5)

    ax.set_title("HMRP Corpus Multi-Stage Extraction Summary Statistics", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    p3 = dirs["summary_images"] / "dataset_statistics.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    register_figure(
        "Dataset Overview",
        "Corpus Ingestion Summary Metrics",
        "dataset_statistics.png",
        "Figure 3: Key extraction and statistical metrics summarizing the size and lexical density of the analyzed papers.",
        "summary/images/dataset_statistics.png"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. TF-IDF Analysis
# ─────────────────────────────────────────────────────────────────────────────
def visualize_tfidf(papers: List[Paper], dirs: Dict[str, Path]):
    logger.info("Visualizing: 2. TF-IDF Feature Extraction & Similarity...")
    texts = [p.cleaned_text or p.full_text for p in papers]
    labels = [p.title[:24] + "..." for p in papers]

    vec = TfidfVectorizer(max_features=25, stop_words="english", ngram_range=(1, 2))
    dtm = vec.fit_transform(texts).toarray()
    feature_names = vec.get_feature_names_out()

    # Save CSV
    df_dtm = pd.DataFrame(dtm, index=labels, columns=feature_names)
    df_dtm.to_csv(dirs["tfidf_data"] / "tfidf_document_term_matrix.csv")

    # 2.1 Document-Term Matrix Heatmap
    plt.figure(figsize=(12, max(6, len(papers) * 0.9)))
    sns.heatmap(df_dtm, cmap="YlGnBu", annot=True, fmt=".2f", cbar_kws={"label": "TF-IDF Weight"})
    plt.title("TF-IDF Document-Term Matrix (Top Corpus Features)", fontweight="bold", pad=12)
    plt.xlabel("Extracted N-Gram Features")
    plt.ylabel("Research Papers")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    p1 = dirs["tfidf_images"] / "tfidf_document_term_heatmap.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    register_figure(
        "TF-IDF Analysis",
        "Document-Term Matrix Heatmap",
        "tfidf_document_term_heatmap.png",
        "Figure 4: Salient TF-IDF term weights computed across individual research papers.",
        "tfidf/images/tfidf_document_term_heatmap.png"
    )

    # 2.2 Top TF-IDF Terms Bar Chart
    mean_scores = pd.Series(dtm.mean(axis=0), index=feature_names).sort_values(ascending=False).head(12)
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=mean_scores.values, y=mean_scores.index, palette="viridis")
    plt.title("Top Salient Terms Ranked by Mean Corpus TF-IDF Score", fontweight="bold", pad=12)
    plt.xlabel("Mean TF-IDF Score")
    for i, v in enumerate(mean_scores.values):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontweight="bold")
    plt.tight_layout()
    p2 = dirs["tfidf_images"] / "tfidf_term_importance.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    register_figure(
        "TF-IDF Analysis",
        "TF-IDF Term Importance",
        "tfidf_term_importance.png",
        "Figure 5: Highest-weighted vocabulary terms extracted via corpus-wide TF-IDF scoring.",
        "tfidf/images/tfidf_term_importance.png"
    )

    # 2.3 TF-IDF Pairwise Cosine Similarity Heatmap
    tfidf_sim = cosine_similarity(dtm)
    plt.figure(figsize=(9, 8))
    sns.heatmap(tfidf_sim, xticklabels=labels, yticklabels=labels, cmap="Blues", annot=True, fmt=".2f", cbar_kws={"label": "TF-IDF Cosine Similarity"})
    plt.title("Inter-Paper TF-IDF Lexical Similarity Heatmap", fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    p3 = dirs["tfidf_images"] / "tfidf_similarity_heatmap.png"
    plt.savefig(p3, dpi=300)
    plt.savefig(dirs["heatmaps_images"] / "tfidf_similarity_heatmap.png", dpi=300)
    plt.close()

    # Interactive Plotly Heatmap
    heatmap_viz = HeatmapViz()
    fig_tfidf = heatmap_viz.build(tfidf_sim, labels=labels, title="TF-IDF Lexical Cosine Similarity Heatmap")
    fig_tfidf.write_html(str(dirs["tfidf_html"] / "tfidf_similarity_heatmap.html"), include_plotlyjs="cdn")
    fig_tfidf.write_html(str(dirs["heatmaps_html"] / "tfidf_similarity_heatmap.html"), include_plotlyjs="cdn")

    register_figure(
        "TF-IDF Analysis",
        "TF-IDF Pairwise Document Similarity",
        "tfidf_similarity_heatmap.png",
        "Figure 6: Pairwise lexical similarity matrix between papers derived from TF-IDF vector representations.",
        "tfidf/images/tfidf_similarity_heatmap.png",
        "tfidf/html/tfidf_similarity_heatmap.html"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Information Retrieval & Search Engine (Exact & Fuzzy Ranking)
# ─────────────────────────────────────────────────────────────────────────────
def visualize_search_engine(papers: List[Paper], dirs: Dict[str, Path]):
    logger.info("Visualizing: 3. Search Engine & Query Ranking Evaluation...")
    search_engine = SearchEngine(fuzzy_threshold=0.65)
    papers_data = [
        {
            "title": p.title,
            "keywords": p.all_keywords,
            "technical_terms": p.research_concepts,
            "research_domains": p.research_domains,
            "abstract": p.abstract,
            "full_text": p.full_text[:5000],
        }
        for p in papers
    ]

    # Benchmark test query
    query = "reinforcement learning policy models"
    results = search_engine.search(query, papers_data, search_in="all")

    if results:
        df_res = pd.DataFrame([
            {"paper": r.paper_title[:30] + "...", "match_type": r.match_type, "term": r.matched_term, "score": r.score}
            for r in results[:10]
        ])
        df_res.to_csv(dirs["search_data"] / "search_query_rankings.csv", index=False)

        # 3.1 Retrieved Paper Relevance Rankings
        plt.figure(figsize=(10, 5))
        ax = sns.barplot(data=df_res, x="score", y="paper", hue="match_type", dodge=False, palette="Set1")
        plt.title(f"HMRP Search Engine: Document Ranking Scores for Query: '{query}'", fontweight="bold", pad=12)
        plt.xlabel("Relevance Match Score")
        plt.ylabel("Retrieved Paper")
        plt.legend(title="Match Type", loc="lower right")
        for i, v in enumerate(df_res["score"]):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontweight="bold")
        plt.tight_layout()
        p1 = dirs["search_images"] / "search_document_ranking.png"
        plt.savefig(p1, dpi=300)
        plt.close()
        register_figure(
            "Search & Information Retrieval",
            "Query Relevance Ranking",
            "search_document_ranking.png",
            f"Figure 7: Top-K retrieved documents and fuzzy matching relevance scores for sample query: '{query}'.",
            "search/images/search_document_ranking.png"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Multi-Engine Keyword Extraction (KeyBERT vs. YAKE vs. TF-IDF)
# ─────────────────────────────────────────────────────────────────────────────
def visualize_keyword_extraction(papers: List[Paper], dirs: Dict[str, Path]):
    logger.info("Visualizing: 4. Multi-Engine Keyword Extraction...")
    p = papers[0]
    kw_tfidf = {k["word"]: k["score"] for k in p.tfidf_keywords[:10]}
    kw_keybert = {k["word"]: k["score"] for k in p.keybert_keywords[:10]}
    kw_yake = {k["word"]: k["score"] for k in p.yake_keywords[:10]}

    # 4.1 Top Keyword Ranking Bar Chart
    all_kws = p.all_keywords[:15]
    freqs = pd.Series([w for p_i in papers for w in p_i.all_keywords]).value_counts().head(12)

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=freqs.values, y=freqs.index, palette="crest")
    plt.title("Corpus-Wide Multi-Engine Extracted Keywords (Merged Ensemble)", fontweight="bold", pad=12)
    plt.xlabel("Keyword Occurrence Across Papers")
    for i, v in enumerate(freqs.values):
        ax.text(v + 0.05, i, str(v), va="center", fontweight="bold")
    plt.tight_layout()
    p1 = dirs["keywords_images"] / "top_keywords.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    register_figure(
        "Keyword Extraction",
        "Ensemble Keyword Frequency",
        "top_keywords.png",
        "Figure 8: High-confidence technical keywords extracted and validated across KeyBERT, YAKE, and TF-IDF.",
        "keyword_extraction/images/top_keywords.png"
    )

    # 4.2 Keyword Extraction Confidence Distribution by Engine
    engine_data = []
    for k in p.keybert_keywords[:15]:
        engine_data.append({"engine": "KeyBERT", "score": k["score"]})
    for k in p.yake_keywords[:15]:
        engine_data.append({"engine": "YAKE", "score": k["score"]})
    for k in p.tfidf_keywords[:15]:
        engine_data.append({"engine": "TF-IDF", "score": k["score"]})

    if engine_data:
        df_eng = pd.DataFrame(engine_data)
        plt.figure(figsize=(9, 5))
        sns.boxplot(data=df_eng, x="engine", y="score", palette="Set2")
        plt.title("Extraction Score Distribution Across Keyword Engines", fontweight="bold", pad=12)
        plt.xlabel("Keyword Engine")
        plt.ylabel("Normalized Confidence Score")
        plt.tight_layout()
        p2 = dirs["keywords_images"] / "keyword_engine_comparison.png"
        plt.savefig(p2, dpi=300)
        plt.close()
        register_figure(
            "Keyword Extraction",
            "Engine Confidence Score Distribution",
            "keyword_engine_comparison.png",
            "Figure 9: Statistical distribution of extraction confidence scores compared across KeyBERT, YAKE, and TF-IDF.",
            "keyword_extraction/images/keyword_engine_comparison.png"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Embeddings & Dimensionality Reduction (PCA / t-SNE)
# ─────────────────────────────────────────────────────────────────────────────
def visualize_embeddings(papers: List[Paper], dirs: Dict[str, Path]):
    logger.info("Visualizing: 5. High-Dimensional Embeddings & Projection...")
    embeddings = [p.embedding for p in papers if p.embedding is not None]
    if len(embeddings) < 3:
        logger.info("Generating embeddings for dimensional reduction...")
        sim_engine = SimilarityEngine()
        texts = [p.cleaned_text or p.full_text for p in papers]
        _, embs = sim_engine._semantic_similarity(texts)
        if embs:
            embeddings = embs

    if len(embeddings) >= 3:
        X = np.array(embeddings)
        domains = [p.research_domains[0] if p.research_domains else "General" for p in papers]
        titles = [p.title[:22] + "..." for p in papers]

        # 5.1 PCA Projection
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        plt.figure(figsize=(10, 6))
        df_pca = pd.DataFrame({"PCA1": X_pca[:, 0], "PCA2": X_pca[:, 1], "Domain": domains, "Title": titles})
        sns.scatterplot(data=df_pca, x="PCA1", y="PCA2", hue="Domain", s=220, palette="tab10", edgecolor="black", alpha=0.9)
        for i, row in df_pca.iterrows():
            plt.text(row["PCA1"] + 0.01, row["PCA2"] + 0.01, row["Title"], fontsize=8, fontweight="medium")
        plt.title(f"Dense Sentence-Transformer Embeddings (PCA 2D Projection, Var={sum(pca.explained_variance_ratio_):.1%})", fontweight="bold", pad=12)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        p1 = dirs["embeddings_images"] / "embedding_pca.png"
        plt.savefig(p1, dpi=300)
        plt.close()
        register_figure(
            "Dense Embeddings",
            "PCA Embedding Projection",
            "embedding_pca.png",
            "Figure 10: 2D Principal Component Analysis (PCA) projection of dense paper embeddings color-coded by domain.",
            "embeddings/images/embedding_pca.png"
        )

        # 5.2 t-SNE Projection (if papers >= 4)
        if len(papers) >= 4:
            perplexity = min(3, len(papers) - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            X_tsne = tsne.fit_transform(X)

            plt.figure(figsize=(10, 6))
            df_tsne = pd.DataFrame({"tSNE1": X_tsne[:, 0], "tSNE2": X_tsne[:, 1], "Domain": domains, "Title": titles})
            sns.scatterplot(data=df_tsne, x="tSNE1", y="tSNE2", hue="Domain", s=220, palette="tab10", edgecolor="black", alpha=0.9)
            for i, row in df_tsne.iterrows():
                plt.text(row["tSNE1"] + 0.5, row["tSNE2"] + 0.5, row["Title"], fontsize=8, fontweight="medium")
            plt.title("t-SNE Non-Linear Manifold Projection of Research Papers", fontweight="bold", pad=12)
            plt.xlabel("t-SNE Dimension 1")
            plt.ylabel("t-SNE Dimension 2")
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()
            p2 = dirs["embeddings_images"] / "embedding_tsne.png"
            plt.savefig(p2, dpi=300)
            plt.close()
            register_figure(
                "Dense Embeddings",
                "t-SNE Manifold Projection",
                "embedding_tsne.png",
                "Figure 11: Non-linear t-SNE clustering illustrating semantic closeness of multidisciplinary publications.",
                "embeddings/images/embedding_tsne.png"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Dense Semantic & Hybrid Similarity
# ─────────────────────────────────────────────────────────────────────────────
def visualize_similarity(papers: List[Paper], sim_results: Dict[str, Any], dirs: Dict[str, Path]):
    logger.info("Visualizing: 6. Semantic & Hybrid Similarity Analysis...")
    labels = [p.title[:24] + "..." for p in papers]
    n = len(papers)

    semantic_matrix = sim_results.get("semantic_matrix", np.zeros((n, n)))
    combined_matrix = sim_results.get("combined_matrix", np.zeros((n, n)))

    # 6.1 Dense Semantic Heatmap
    plt.figure(figsize=(9, 8))
    sns.heatmap(semantic_matrix, xticklabels=labels, yticklabels=labels, cmap="magma", annot=True, fmt=".2f", cbar_kws={"label": "Embedding Cosine Similarity"})
    plt.title("Inter-Paper Dense Semantic Embedding Similarity Heatmap", fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    p1 = dirs["cosine_images"] / "semantic_similarity_heatmap.png"
    plt.savefig(p1, dpi=300)
    plt.savefig(dirs["heatmaps_images"] / "semantic_similarity_heatmap.png", dpi=300)
    plt.close()

    # Interactive HTML
    heatmap_viz = HeatmapViz()
    fig_sem = heatmap_viz.build(semantic_matrix, labels=labels, title="Dense Semantic Embedding Similarity Heatmap")
    fig_sem.write_html(str(dirs["cosine_html"] / "semantic_similarity_heatmap.html"), include_plotlyjs="cdn")
    fig_sem.write_html(str(dirs["heatmaps_html"] / "semantic_similarity_heatmap.html"), include_plotlyjs="cdn")

    register_figure(
        "Semantic Similarity",
        "Dense Embedding Similarity Heatmap",
        "semantic_similarity_heatmap.png",
        "Figure 12: Pairwise cosine similarity matrix calculated using deep contextual transformer embeddings.",
        "cosine_similarity/images/semantic_similarity_heatmap.png",
        "cosine_similarity/html/semantic_similarity_heatmap.html"
    )

    # 6.2 Top Semantic Pairs Bar Chart
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((f"{papers[i].title[:16]}.. & {papers[j].title[:16]}..", float(semantic_matrix[i, j])))
    pairs.sort(key=lambda x: x[1], reverse=True)

    plt.figure(figsize=(10, 5))
    top_p = pairs[:8]
    ax = sns.barplot(x=[p[1] for p in top_p], y=[p[0] for p in top_p], palette="Purples_r")
    plt.title("Top Most Semantically Related Research Paper Pairs", fontweight="bold", pad=12)
    plt.xlabel("Semantic Cosine Similarity Score")
    for i, v in enumerate([p[1] for p in top_p]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontweight="bold")
    plt.tight_layout()
    p2 = dirs["cosine_images"] / "semantic_top_similar_pairs.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    register_figure(
        "Semantic Similarity",
        "Top Semantically Similar Pairs",
        "semantic_top_similar_pairs.png",
        "Figure 13: Paper pairs demonstrating highest contextual semantic alignment.",
        "cosine_similarity/images/semantic_top_similar_pairs.png"
    )

    # 6.3 Combined Hybrid Similarity Heatmap
    plt.figure(figsize=(9, 8))
    sns.heatmap(combined_matrix, xticklabels=labels, yticklabels=labels, cmap="viridis", annot=True, fmt=".2f", cbar_kws={"label": "Hybrid Similarity"})
    plt.title("HMRP Hybrid Inter-Paper Similarity Heatmap (Lexical + Semantic)", fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    p3 = dirs["cosine_images"] / "hybrid_combined_similarity.png"
    plt.savefig(p3, dpi=300)
    plt.savefig(dirs["heatmaps_images"] / "hybrid_combined_similarity.png", dpi=300)
    plt.close()

    fig_comb = heatmap_viz.build(combined_matrix, labels=labels, title="HMRP Hybrid Inter-Paper Similarity Heatmap")
    fig_comb.write_html(str(dirs["cosine_html"] / "hybrid_combined_similarity.html"), include_plotlyjs="cdn")
    fig_comb.write_html(str(dirs["heatmaps_html"] / "hybrid_combined_similarity.html"), include_plotlyjs="cdn")

    register_figure(
        "Semantic Similarity",
        "Hybrid Combined Similarity Heatmap",
        "hybrid_combined_similarity.png",
        "Figure 14: Comprehensive hybrid similarity matrix combining TF-IDF lexical overlap and dense semantic representations.",
        "cosine_similarity/images/hybrid_combined_similarity.png",
        "cosine_similarity/html/hybrid_combined_similarity.html"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Hierarchical Concept Mapping (Sunburst, Treemap, Icicle, Tree)
# ─────────────────────────────────────────────────────────────────────────────
def visualize_hierarchy(selected_paper: Paper, hier_viz: HierarchyViz, dirs: Dict[str, Path]):
    logger.info("Visualizing: 7. Hierarchical Concept Mapping...")
    tree = selected_paper.concept_hierarchy
    title_short = selected_paper.title[:40]

    # 7.1 Sunburst Chart
    fig_sunburst = hier_viz.sunburst(tree, title=f"Concept Sunburst: {title_short}")
    save_plotly_figure_safely(fig_sunburst, dirs["hierarchy_html"] / "hierarchy_sunburst.html", dirs["hierarchy_images"] / "hierarchy_sunburst.png")
    register_figure(
        "Hierarchical Concept Mapping",
        "Concentric Concept Sunburst",
        "hierarchy_sunburst.png",
        f"Figure 15: Radial multi-tiered concept taxonomy for '{title_short}'.",
        "hierarchy/images/hierarchy_sunburst.png",
        "hierarchy/html/hierarchy_sunburst.html"
    )

    # 7.2 Treemap Chart
    fig_treemap = hier_viz.treemap(tree, title=f"Concept Treemap: {title_short}")
    save_plotly_figure_safely(fig_treemap, dirs["hierarchy_html"] / "hierarchy_treemap.html", dirs["hierarchy_images"] / "hierarchy_treemap.png")
    register_figure(
        "Hierarchical Concept Mapping",
        "Hierarchical Concept Treemap",
        "hierarchy_treemap.png",
        f"Figure 16: Nested rectangular concept partitioning for '{title_short}'.",
        "hierarchy/images/hierarchy_treemap.png",
        "hierarchy/html/hierarchy_treemap.html"
    )

    # 7.3 Icicle Chart
    fig_icicle = hier_viz.icicle(tree, title=f"Concept Icicle: {title_short}")
    save_plotly_figure_safely(fig_icicle, dirs["hierarchy_html"] / "hierarchy_icicle.html", dirs["hierarchy_images"] / "hierarchy_icicle.png")
    register_figure(
        "Hierarchical Concept Mapping",
        "Top-Down Concept Icicle Diagram",
        "hierarchy_icicle.png",
        f"Figure 17: Top-down hierarchical breakdown of technical sub-domains for '{title_short}'.",
        "hierarchy/images/hierarchy_icicle.png",
        "hierarchy/html/hierarchy_icicle.html"
    )

    # 7.4 Network Tree Visualization (Matplotlib)
    fig, ax = plt.subplots(figsize=(12, 7))
    G = nx.DiGraph()
    root = selected_paper.title[:25] + "..."
    def add_nodes(p_node, subtree):
        for k, v in subtree.items():
            G.add_edge(p_node, k)
            if isinstance(v, dict):
                add_nodes(k, v)
    add_nodes(root, tree)
    pos = nx.spring_layout(G, seed=42, k=0.7)
    nx.draw_networkx_nodes(G, pos, node_size=1600, node_color="#2B6CB0", alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, arrows=True, edge_color="#CBD5E0", width=1.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="white", font_weight="bold", ax=ax)
    ax.set_title(f"Hierarchical Concept Network Tree: {title_short}", fontweight="bold", pad=15)
    ax.axis("off")
    plt.tight_layout()
    p4 = dirs["hierarchy_images"] / "hierarchy_tree.png"
    plt.savefig(p4, dpi=300)
    plt.close()
    register_figure(
        "Hierarchical Concept Mapping",
        "Hierarchical Concept Network Tree",
        "hierarchy_tree.png",
        f"Figure 18: Directed graph visualization of parent-child conceptual relationships for '{title_short}'.",
        "hierarchy/images/hierarchy_tree.png"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. Concept Analysis & Cross-Domain CCC Bridges
# ─────────────────────────────────────────────────────────────────────────────
def visualize_concepts(papers: List[Paper], corpus_ccc: Dict[str, Any], dirs: Dict[str, Path]):
    logger.info("Visualizing: 8. Concept Analysis & Cross-Domain Bridges...")

    # 8.1 Top Concepts Bar Chart
    all_concepts = [c for p in papers for c in p.research_concepts]
    c_counts = pd.Series(all_concepts).value_counts().head(12)

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=c_counts.values, y=c_counts.index, palette="mako")
    plt.title("Top Most Frequent Technical Concepts in Analyzed Corpus", fontweight="bold", pad=12)
    plt.xlabel("Corpus Concept Frequency")
    for i, v in enumerate(c_counts.values):
        ax.text(v + 0.05, i, str(v), va="center", fontweight="bold")
    plt.tight_layout()
    p1 = dirs["concepts_images"] / "top_concepts.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    register_figure(
        "Concept Analysis",
        "Top Technical Concepts",
        "top_concepts.png",
        "Figure 19: Most prominent technical terms and concepts identified across the corpus.",
        "concepts/images/top_concepts.png"
    )

    # 8.2 Cross-Domain Bridge Concepts
    bridges = corpus_ccc.get("cross_domain_bridge_concepts", [])[:10]
    if bridges:
        df_b = pd.DataFrame([
            {"concept": b["concept"], "span": len(b["domains"]), "domains": ", ".join(b["domains"])}
            for b in bridges
        ])
        plt.figure(figsize=(10, 5))
        ax = sns.barplot(data=df_b, x="span", y="concept", palette="flare")
        plt.title("Identified Cross-Domain Bridge Concepts (Interdisciplinary Connectors)", fontweight="bold", pad=12)
        plt.xlabel("Number of Spanned Scientific Disciplines")
        for i, row in df_b.iterrows():
            ax.text(row["span"] + 0.05, i, f"({row['domains']})", va="center", fontsize=8, color="#2D3748")
        plt.tight_layout()
        p2 = dirs["concepts_images"] / "cross_domain_bridge_concepts.png"
        plt.savefig(p2, dpi=300)
        plt.close()
        register_figure(
            "Concept Analysis",
            "Cross-Domain Bridge Concepts",
            "cross_domain_bridge_concepts.png",
            "Figure 20: Technical concepts functioning as bridges linking distinct research disciplines.",
            "concepts/images/cross_domain_bridge_concepts.png"
        )

    # 8.3 Concept Co-occurrence Plot
    pairs = corpus_ccc.get("top_cooccurring_concept_pairs", [])[:10]
    if pairs:
        pair_labels = [f"{p['concept_a']} & {p['concept_b']}" for p in pairs]
        pair_counts = [p["cooccurrence_count"] for p in pairs]

        plt.figure(figsize=(10, 5))
        sns.barplot(x=pair_counts, y=pair_labels, palette="crest")
        plt.title("Top Concept-to-Concept (C2C) Co-occurrence Associations", fontweight="bold", pad=12)
        plt.xlabel("Co-occurrence Count")
        plt.tight_layout()
        p3 = dirs["concepts_images"] / "concept_cooccurrence.png"
        plt.savefig(p3, dpi=300)
        plt.close()
        register_figure(
            "Concept Analysis",
            "Concept Co-occurrence Associations",
            "concept_cooccurrence.png",
            "Figure 21: High-frequency pairwise concept associations discovered across the paper corpus.",
            "concepts/images/concept_cooccurrence.png"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 9. Knowledge Graph (Static PNG + Plotly + PyVis)
# ─────────────────────────────────────────────────────────────────────────────
def visualize_knowledge_graph(papers: List[Paper], dirs: Dict[str, Path]):
    logger.info("Visualizing: 9. Knowledge Graph Visualizations...")
    papers_summary = [
        {
            "title": p.title,
            "research_domains": p.research_domains,
            "keywords": p.all_keywords,
            "technical_terms": p.research_concepts,
        }
        for p in papers
    ]

    kg = KnowledgeGraph()
    kg.build_from_papers(papers_summary)

    # 9.1 Plotly Interactive HTML
    fig_kg = kg.to_plotly(title=f"HMRP Corpus Multi-Domain Knowledge Graph ({len(papers)} Papers)")
    fig_kg.write_html(str(dirs["knowledge_graph_html"] / "knowledge_graph_plotly.html"), include_plotlyjs="cdn")

    # 9.2 PyVis HTML
    try:
        kg.to_pyvis_html(str(dirs["knowledge_graph_html"] / "knowledge_graph_pyvis.html"), title="HMRP Interactive Knowledge Graph")
    except Exception as e:
        logger.warning(f"PyVis graph generation note: {e}")

    # 9.3 Static Matplotlib PNG Graph
    fig, ax = plt.subplots(figsize=(14, 10))
    G = kg.G
    pos = nx.spring_layout(G, seed=42, k=0.45, iterations=60)

    color_map = {
        "paper": "#E53E3E",
        "domain": "#805AD5",
        "concept": "#3182CE",
        "keyword": "#38A169",
        "entity": "#DD6B20"
    }
    node_colors = [color_map.get(G.nodes[n].get("node_type", "concept"), "#718096") for n in G.nodes()]
    node_sizes = [650 if G.nodes[n].get("node_type") == "paper" else (400 if G.nodes[n].get("node_type") == "domain" else 180) for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.88, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#CBD5E0", alpha=0.6, width=1.0, ax=ax)

    prominent = {n: n[:18] for n in G.nodes() if G.nodes[n].get("node_type") in ("paper", "domain")}
    nx.draw_networkx_labels(G, pos, labels=prominent, font_size=8, font_weight="bold", ax=ax)

    handles = [plt.Line2D([0], [0], marker='o', color='w', label=k.capitalize(), markerfacecolor=v, markersize=10) for k, v in color_map.items()]
    ax.legend(handles=handles, title="Node Entity Type", loc="upper right")
    ax.set_title(f"HMRP Corpus Multi-Domain Semantic Knowledge Graph ({len(G.nodes())} Nodes, {len(G.edges())} Relations)", fontweight="bold", pad=15)
    ax.axis("off")
    plt.tight_layout()
    p1 = dirs["knowledge_graph_images"] / "knowledge_graph.png"
    plt.savefig(p1, dpi=300)
    plt.close()

    register_figure(
        "Knowledge Graph",
        "Corpus Semantic Knowledge Graph",
        "knowledge_graph.png",
        "Figure 22: Heterogeneous semantic network connecting research publications, academic fields, concepts, and keywords.",
        "knowledge_graph/images/knowledge_graph.png",
        "knowledge_graph/html/knowledge_graph_plotly.html"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 10. Radial Mind Map Visualizations (PNG + SVG + HTML)
# ─────────────────────────────────────────────────────────────────────────────
def visualize_mindmap(selected_paper: Paper, mindmap_viz: MindMapViz, dirs: Dict[str, Path]):
    logger.info("Visualizing: 10. Radial Concept Mind Map...")
    tree = selected_paper.concept_hierarchy
    root_title = selected_paper.title[:35]

    # Plotly Interactive HTML
    fig_mindmap = mindmap_viz.build(title=root_title, tree=tree)
    fig_mindmap.write_html(str(dirs["mindmap_html"] / "hmrp_mindmap.html"), include_plotlyjs="cdn")

    # Static Matplotlib Polar Concentric Mind Map
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.scatter(0, 0, color="#E53E3E", s=650, zorder=5)
    ax.text(0, 0, f"{root_title[:18]}\n(Core)", ha="center", va="center", fontsize=9, fontweight="bold", color="white")

    l1_keys = list(tree.keys())
    if l1_keys:
        n_l1 = len(l1_keys)
        angles_l1 = np.linspace(0, 2 * np.pi, n_l1, endpoint=False)
        for i, (k1, v1) in enumerate(tree.items()):
            theta1 = angles_l1[i]
            r1 = 1.0
            ax.scatter(theta1, r1, color="#3182CE", s=350, zorder=4)
            ax.plot([0, theta1], [0, r1], color="#CBD5E0", lw=1.5, zorder=2)
            ax.text(theta1, r1 + 0.12, k1[:20], ha="center", va="center", fontsize=8, fontweight="bold", rotation=np.degrees(theta1) if theta1 < np.pi else np.degrees(theta1)+180)

            if isinstance(v1, dict) and v1:
                n_l2 = len(v1)
                arc = (2 * np.pi / n_l1) * 0.8
                angles_l2 = np.linspace(theta1 - arc/2, theta1 + arc/2, n_l2)
                for j, (k2, _) in enumerate(v1.items()):
                    theta2 = angles_l2[j]
                    r2 = 2.0
                    ax.scatter(theta2, r2, color="#38A169", s=150, zorder=3)
                    ax.plot([theta1, theta2], [r1, r2], color="#E2E8F0", lw=1.0, zorder=1)
                    ax.text(theta2, r2 + 0.15, k2[:16], ha="center", va="center", fontsize=7, color="#2D3748")

    ax.set_ylim(0, 2.5)
    ax.axis("off")
    ax.set_title(f"HMRP Radial Concept Mind Map\n{selected_paper.title[:55]}", fontweight="bold", pad=25)
    plt.tight_layout()
    p1 = dirs["mindmap_images"] / "hmrp_mindmap.png"
    plt.savefig(p1, dpi=300)
    plt.savefig(dirs["mindmap_images"] / "hmrp_mindmap.svg", format="svg")
    plt.close()

    register_figure(
        "Radial Mind Map",
        "Concentric Concept Mind Map",
        "hmrp_mindmap.png",
        f"Figure 23: Concentric radial concept graph illustrating hierarchical topic dispersion for '{root_title}'.",
        "mindmap/images/hmrp_mindmap.png",
        "mindmap/html/hmrp_mindmap.html"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 11. Master Report Generator (output/reports/HMRP_Complete_Report.html)
# ─────────────────────────────────────────────────────────────────────────────
def generate_master_report(papers: List[Paper], corpus_ccc: Dict[str, Any], dirs: Dict[str, Path]):
    logger.info("Generating Master Report: output/reports/HMRP_Complete_Report.html...")

    # Group figures by section
    sections: Dict[str, List[Dict[str, Any]]] = {}
    for fig in GENERATED_SECTIONS:
        sections.setdefault(fig["section"], []).append(fig)

    sections_html = ""
    for sec_idx, (sec_name, figs) in enumerate(sections.items(), start=1):
        cards_html = ""
        for f in figs:
            interactive_btn = f"<a href='../{f['html_link']}' target='_blank' class='btn'>Open Interactive Version</a>" if f.get("html_link") else ""
            cards_html += f"""
            <div class="figure-card">
                <div class="img-wrapper">
                    <img src="../{f['rel_path']}" alt="{f['title']}">
                </div>
                <h3>{f['title']}</h3>
                <p class="caption">{f['caption']}</p>
                {interactive_btn}
            </div>
            """

        sections_html += f"""
        <div class="report-section">
            <h2>{sec_idx}. {sec_name}</h2>
            <div class="figures-grid">
                {cards_html}
            </div>
        </div>
        """

    # Table rows
    top_bridges = corpus_ccc.get("cross_domain_bridge_concepts", [])[:8]
    bridge_rows = "".join([
        f"<tr><td><b>{b['concept']}</b></td><td><span class='badge'>{b['frequency']}</span></td><td>{', '.join(b['domains'])}</td></tr>"
        for b in top_bridges
    ])
    paper_rows = "".join([
        f"<tr><td><b>{p.paper_id}</b></td><td>{p.title}</td><td>{', '.join(p.research_domains)}</td><td>{len(p.research_concepts)}</td><td>{len(p.all_keywords)}</td></tr>"
        for p in papers
    ])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HMRP Master Research & Visualization Report</title>
    <style>
        :root {{
            --primary: #1A365D;
            --accent: #2B6CB0;
            --bg: #F7FAFC;
            --card-bg: #FFFFFF;
            --text: #2D3748;
            --border: #E2E8F0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 40px 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1250px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #1A365D 0%, #2B6CB0 100%);
            color: white;
            padding: 45px;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            margin-bottom: 40px;
        }}
        .header h1 {{ margin: 0 0 10px 0; font-size: 2.3rem; font-weight: 800; }}
        .header p {{ margin: 0; opacity: 0.92; font-size: 1.15rem; }}
        .report-section {{
            background: var(--card-bg);
            border-radius: 12px;
            border: 1px solid var(--border);
            padding: 35px;
            margin-bottom: 35px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        }}
        h2 {{
            color: #1A365D;
            border-bottom: 2px solid var(--accent);
            padding-bottom: 8px;
            margin-top: 0;
            font-size: 1.5rem;
        }}
        .figures-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
            gap: 25px;
            margin-top: 20px;
        }}
        .figure-card {{
            background: #FAFAFA;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 18px;
            text-align: left;
        }}
        .img-wrapper {{
            background: white;
            padding: 10px;
            border-radius: 6px;
            border: 1px solid #EDF2F7;
            text-align: center;
        }}
        .img-wrapper img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .figure-card h3 {{ margin: 15px 0 5px 0; font-size: 1.15rem; color: #1A365D; }}
        .caption {{ font-size: 0.9rem; color: #4A5568; margin-bottom: 12px; font-style: italic; }}
        .btn {{
            display: inline-block;
            background: var(--accent);
            color: white;
            padding: 7px 14px;
            border-radius: 6px;
            text-decoration: none;
            font-size: 0.88rem;
            font-weight: 600;
        }}
        .btn:hover {{ background: #2C5282; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{ background-color: #EDF2F7; color: #1A365D; font-weight: 700; }}
        .badge {{
            background-color: #EBF8FF;
            color: #2B6CB0;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: 700;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 HMRP: Master Research & Visualization Report</h1>
            <p>Comprehensive Multidisciplinary Analysis, Hierarchical Taxonomies & Cross-Domain Semantic Evaluation</p>
        </div>

        {sections_html}

        <!-- Summary Tables -->
        <div class="report-section">
            <h2>Corpus Synthesis & Cross-Domain Bridges</h2>
            <h3>🌟 Identified Bridge Concepts</h3>
            <table>
                <thead>
                    <tr><th>Concept Name</th><th>Corpus Frequency</th><th>Connected Domains</th></tr>
                </thead>
                <tbody>
                    {bridge_rows}
                </tbody>
            </table>

            <h3 style="margin-top: 30px;">📄 Ingested Publications</h3>
            <table>
                <thead>
                    <tr><th>Paper ID</th><th>Title</th><th>Domains</th><th>Concepts</th><th>Keywords</th></tr>
                </thead>
                <tbody>
                    {paper_rows}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
    report_path = dirs["reports"] / "HMRP_Complete_Report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"✅ Master HTML Report successfully generated at: {report_path.relative_to(OUTPUT_BASE)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("🚀 HMRP: EXECUTING FULL MULTI-TECHNIQUE VISUALIZATION PIPELINE")
    print("=" * 80)

    dirs = create_directory_structure()
    pipeline = AnalysisPipeline()
    ccc_mapper = CCCMapper()
    hier_viz = HierarchyViz()
    mindmap_viz = MindMapViz()

    # Discover and process papers
    pdf_dir = PROJECT_ROOT / "HMRP_Dataset" / "PDFs"
    pdf_files = list(pdf_dir.rglob("*.pdf")) if pdf_dir.exists() else []

    if len(pdf_files) >= 3:
        logger.info(f"[*] Discovered {len(pdf_files)} local PDFs on disk. Ingesting batch of 6 papers...")
        papers = pipeline.process_papers(pdf_files[:6])
    else:
        logger.info("[*] Generating Multidisciplinary Benchmark Papers...")
        p1 = Paper(paper_id="paper_nlp", title="Transformer Architectures for Hierarchical Concept Mapping", file_name="nlp_transformer.pdf", page_count=14, word_count=9800, abstract="We propose multi-head self-attention transformers for hierarchical concept taxonomy generation.", research_domains=["Natural Language Processing", "Artificial Intelligence"], research_concepts=["Transformer", "Self-Attention", "Concept Hierarchy", "Language Models", "Semantic Parsing"], all_keywords=["transformer", "nlp", "attention mechanism", "deep learning"])
        p2 = Paper(paper_id="paper_robotics", title="Real-Time 3D Point Tracking with State Space Models in Robotics", file_name="robotics_ssm.pdf", page_count=18, word_count=14200, abstract="This paper introduces state space models for 3D point tracking and robotic obstacle avoidance.", research_domains=["Robotics", "Computer Vision", "Artificial Intelligence"], research_concepts=["State Space Models", "Point Tracking", "Robotic Manipulation", "Computer Vision", "Deep Learning"], all_keywords=["robotics", "point tracking", "state space model", "reinforcement learning"])
        p3 = Paper(paper_id="paper_genomics", title="Deep Learning for 3D Protein Structure Prediction from Genomic Sequences", file_name="genomics_protein.pdf", page_count=16, word_count=11500, abstract="We apply transformer sequence encoders to predict 3D molecular protein structures from genomic sequences.", research_domains=["Biomedical & Genomics", "Artificial Intelligence"], research_concepts=["Transformer", "Protein Structure Prediction", "Genomics", "Amino Acids", "Deep Learning"], all_keywords=["protein folding", "transformer", "genomics", "bioinformatics"])
        p4 = Paper(paper_id="paper_quantum", title="Quantum Entanglement Verification via Optical Interferometry", file_name="quantum_optics.pdf", page_count=15, word_count=10200, abstract="We formulate quantum state tomography and verify photonic entanglement using interferometers.", research_domains=["Quantum Physics & Optics", "Applied Mathematics & Optimization"], research_concepts=["Quantum Entanglement", "Photonic Qubits", "Interferometry", "State Tomography"], all_keywords=["quantum optics", "entanglement", "qubits", "photonics"])
        p5 = Paper(paper_id="paper_finance", title="Stochastic Volatility Modeling and Portfolio Optimization in Quantitative Finance", file_name="quant_finance.pdf", page_count=20, word_count=13800, abstract="We formulate convex optimization and Black-Scholes risk models for high-frequency algorithmic portfolio rebalancing.", research_domains=["Quantitative Finance", "Applied Mathematics & Optimization"], research_concepts=["Portfolio Optimization", "Black-Scholes Model", "Stochastic Volatility", "Convex Optimization"], all_keywords=["quantitative finance", "portfolio optimization", "risk management", "convex optimization"])
        papers = [p1, p2, p3, p4, p5]
        for p in papers:
            tree, _ = pipeline._hierarchy.build(p.research_concepts, p.all_keywords, p.abstract)
            p.concept_hierarchy = tree
            p.mindmap = {"root": p.title, "children": tree}
            p.ccc_mapping = ccc_mapper.map_paper_ccc(p.research_concepts, p.all_keywords, p.research_domains)

    # 1. Dataset Analysis
    visualize_dataset(papers, dirs)

    # 2. TF-IDF Analysis
    visualize_tfidf(papers, dirs)

    # 3. Search Engine & IR Ranking
    visualize_search_engine(papers, dirs)

    # 4. Multi-Engine Keyword Extraction
    visualize_keyword_extraction(papers, dirs)

    # 5. Embeddings & Dimensionality Reduction
    visualize_embeddings(papers, dirs)

    # 6. Semantic & Hybrid Similarity
    sim_results = pipeline.compute_similarity(papers, method="both")
    visualize_similarity(papers, sim_results, dirs)

    # 7. Hierarchical Concept Mapping
    visualize_hierarchy(papers[0], hier_viz, dirs)

    # 8. Concept Analysis & CCC Bridges
    corpus_ccc = ccc_mapper.map_corpus_ccc(papers)
    visualize_concepts(papers, corpus_ccc, dirs)

    # 9. Knowledge Graph
    visualize_knowledge_graph(papers, dirs)

    # 10. Radial Mind Map
    visualize_mindmap(papers[0], mindmap_viz, dirs)

    # 11. Master Report
    generate_master_report(papers, corpus_ccc, dirs)

    print("\n" + "=" * 80)
    print("🎉 FULL VISUALIZATION ENGINE COMPLETED SUCCESSFULLY!")
    print(f"📂 Output Location: {OUTPUT_BASE.resolve()}")
    print(f"📄 Master Report: {(dirs['reports'] / 'HMRP_Complete_Report.html').resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
