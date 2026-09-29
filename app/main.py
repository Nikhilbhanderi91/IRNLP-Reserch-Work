"""
================================================
app/main.py  –  Streamlit Entry Point
Hierarchical Concept Mapping for Research Paper
Similarity Analysis (HMRP)
================================================
Run: streamlit run app/main.py
================================================
"""

import sys
import os
import time
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Optional

# ── Path fix so modules resolve correctly ─────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Local modules ─────────────────────────────────────────────────────────────
from app.config import APP_TITLE, APP_ICON, APP_LAYOUT, UPLOAD_DIR, EXPORT_DIR
from app.pipeline import AnalysisPipeline
from models.paper import Paper
from visualization.knowledge_graph import KnowledgeGraph
from visualization.hierarchy_viz import HierarchyViz
from visualization.heatmap_viz import HeatmapViz
from visualization.mindmap_viz import MindMapViz
from search.search_engine import SearchEngine
from export.exporter import Exporter
from hierarchy.hierarchy_builder import HierarchyBuilder
from utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Global background ── */
    .main {
        background: linear-gradient(135deg, #0a0a14 0%, #0f0f1e 50%, #12121f 100%);
        min-height: 100vh;
    }
    .block-container { padding: 2rem 2.5rem; }

    /* ── Hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #1a0a2e 0%, #16213e 40%, #0f3460 100%);
        border: 1px solid rgba(108,99,255,0.3);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(108,99,255,0.1) 0%, transparent 60%);
        animation: pulse 4s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50%       { transform: scale(1.1); opacity: 1; }
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6C63FF, #43BCCD, #00C49A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        position: relative;
    }
    .hero-sub {
        color: rgba(255,255,255,0.6);
        font-size: 1rem;
        margin-top: 0.5rem;
        position: relative;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, rgba(108,99,255,0.15), rgba(67,188,205,0.1));
        border: 1px solid rgba(108,99,255,0.25);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(108,99,255,0.5);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #6C63FF;
    }
    .metric-label {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #6C63FF;
        border-left: 4px solid #6C63FF;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* ── Keyword chips ── */
    .kw-chip {
        display: inline-block;
        background: linear-gradient(135deg, rgba(108,99,255,0.2), rgba(67,188,205,0.2));
        border: 1px solid rgba(108,99,255,0.4);
        border-radius: 20px;
        padding: 0.25rem 0.7rem;
        font-size: 0.78rem;
        color: #c8c3ff;
        margin: 0.2rem;
        transition: background 0.2s;
    }
    .kw-chip:hover {
        background: rgba(108,99,255,0.4);
        color: #fff;
    }

    /* ── Abstract box ── */
    .abstract-box {
        background: rgba(15,15,26,0.8);
        border: 1px solid rgba(108,99,255,0.2);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: rgba(255,255,255,0.8);
        font-size: 0.9rem;
        line-height: 1.7;
    }

    /* ── Hierarchy path ── */
    .hierarchy-path {
        background: rgba(108,99,255,0.08);
        border-left: 3px solid #6C63FF;
        border-radius: 0 8px 8px 0;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        color: #c8c3ff;
        font-size: 0.85rem;
        font-family: monospace;
    }

    /* ── Search result ── */
    .search-result {
        background: rgba(0,196,154,0.08);
        border: 1px solid rgba(0,196,154,0.25);
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin: 0.4rem 0;
    }

    /* ── Similarity pair ── */
    .sim-pair {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        background: rgba(249,168,38,0.08);
        border: 1px solid rgba(249,168,38,0.2);
        margin: 0.3rem 0;
    }
    .sim-score {
        font-weight: 700;
        color: #F9A826;
        min-width: 50px;
        text-align: right;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a18 0%, #0f0f24 100%);
        border-right: 1px solid rgba(108,99,255,0.2);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(108,99,255,0.1);
        border: 1px solid rgba(108,99,255,0.25);
        border-radius: 8px;
        color: #c8c3ff;
        font-weight: 500;
        padding: 0.5rem 1.2rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6C63FF, #43BCCD) !important;
        color: white !important;
        border-color: transparent !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF, #43BCCD);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: opacity 0.2s, transform 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.85;
        transform: translateY(-2px);
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0a14; }
    ::-webkit-scrollbar-thumb { background: #6C63FF; border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
def init_state() -> None:
    defaults = {
        "papers":        [],    # List[Paper]
        "sim_results":   {},    # similarity dict
        "pipeline":      None,
        "processed":     False,
        "active_file_hashes": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_pipeline() -> AnalysisPipeline:
    return AnalysisPipeline()


def render_chips(items: List[str], max_show: int = 30) -> str:
    chips = "".join(
        f'<span class="kw-chip">{w}</span>' for w in items[:max_show]
    )
    return chips


def render_metric(value, label: str) -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>'
    )


def short(text: str, n: int = 60) -> str:
    return text if len(text) <= n else text[:n] + "…"


# ─────────────────────────────────────────────────────────────────────────────
# Hero Banner
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">🔬 Hierarchical Concept Mapper</div>
        <div class="hero-sub">
            Paper-Isolated Analysis · Source Traceability · Similarity Matrix · Concept Hierarchy
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – upload & controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Upload Papers")
    uploaded_files = st.file_uploader(
        "Upload PDF research papers",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to analyse.",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("## ⚙️ Settings")

    sim_method = st.selectbox(
        "Similarity Method",
        ["both", "tfidf", "semantic"],
        help="Choose which similarity algorithm(s) to use.",
    )

    keyword_top_n = st.slider(
        "Keywords to extract", min_value=5, max_value=50, value=20
    )

    show_raw_entities = st.checkbox("Show named entities", value=True)

    st.markdown("---")
    analyse_btn = st.button(
        "🚀 Analyse Papers",
        use_container_width=True,
        disabled=not uploaded_files,
        type="primary",
    )

    if st.session_state.processed:
        st.success(f"✅ {len(st.session_state.papers)} paper(s) analysed")


# ─────────────────────────────────────────────────────────────────────────────
# Processing logic
# ─────────────────────────────────────────────────────────────────────────────
if analyse_btn and uploaded_files:
    pipeline = get_pipeline()
    tmp_paths: List[Path] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for uf in uploaded_files:
            p = Path(tmpdir) / uf.name
            p.write_bytes(uf.read())
            tmp_paths.append(p)

        progress_bar  = st.progress(0, text="Starting analysis…")
        progress_text = st.empty()

        def update_progress(step: int, total: int, msg: str) -> None:
            pct = min(int((step / max(total, 1)) * 100), 99)
            progress_bar.progress(pct, text=msg)
            progress_text.markdown(f"**{msg}**")

        with st.spinner("🔄 Processing papers with isolation check…"):
            try:
                papers = pipeline.process_papers(tmp_paths, progress_callback=update_progress)
                for p, uf in zip(papers, uploaded_files):
                    p.file_name = uf.name

                progress_bar.progress(90, text="Computing similarity…")
                sim_results = pipeline.compute_similarity(papers, method=sim_method)

                progress_bar.progress(100, text="Done!")
                time.sleep(0.4)
                progress_bar.empty()
                progress_text.empty()

                st.session_state.papers      = papers
                st.session_state.sim_results  = sim_results
                st.session_state.processed   = True
                st.session_state.active_file_hashes = [p.file_hash for p in papers]
                st.rerun()

            except Exception as e:
                log.exception("Pipeline error")
                st.error(f"❌ Processing failed: {e}")
                progress_bar.empty()


# ─────────────────────────────────────────────────────────────────────────────
# Main content (shown only after processing)
# ─────────────────────────────────────────────────────────────────────────────
papers: List[Paper] = st.session_state.papers
sim_results: Dict   = st.session_state.sim_results

if not papers:
    # ── Landing state ────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding: 4rem 2rem;">
            <div style="font-size:5rem">📄</div>
            <h2 style="color:#6C63FF; font-weight:700">Upload Research Papers to Begin</h2>
            <p style="color:rgba(255,255,255,0.5); max-width:600px; margin:1rem auto">
                Upload one or more PDF research papers. Every paper is analyzed in complete
                isolation with SHA-256 identity hashing, concept traceability, and strict validation.
            </p>
        </div>

        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:1rem; max-width:900px; margin: 0 auto;">
        """,
        unsafe_allow_html=True,
    )
    features = [
        ("📑", "PDF Identity Hashing", "SHA-256 paper isolation & verification"),
        ("🔑", "Keyword Analysis", "TF-IDF · KeyBERT · YAKE"),
        ("🌳", "Concept Hierarchy", "Dynamic parent-child tree mapping"),
        ("🔗", "Similarity Matrix", "Cosine · Sentence Transformers"),
        ("🕸️", "Knowledge Graph", "Interactive network visualization"),
        ("🧠", "Mind Map", "Radial paper concept tree"),
        ("🔍", "Source Traceability", "Concept evidence & page tracing"),
        ("💾", "Structured JSON Export", "Section 6 schema export"),
    ]
    cols = st.columns(4)
    for i, (icon, title_f, desc) in enumerate(features):
        with cols[i % 4]:
            st.markdown(
                f"""<div class="metric-card">
                    <div style="font-size:2rem">{icon}</div>
                    <div style="color:#c8c3ff;font-weight:600;margin-top:.5rem">{title_f}</div>
                    <div style="color:rgba(255,255,255,0.4);font-size:.75rem;margin-top:.3rem">{desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )

else:
    total_words   = sum(p.word_count for p in papers)
    total_kws     = sum(len(p.all_keywords) for p in papers)
    total_concepts = sum(len(p.research_concepts) for p in papers)
    all_domains   = list({d for p in papers for d in p.research_domains})

    cols = st.columns(5)
    metrics = [
        (len(papers),       "Papers"),
        (total_words,       "Total Words"),
        (total_kws,         "Keywords"),
        (total_concepts,    "Concepts"),
        (len(all_domains),  "Domains"),
    ]
    for col, (val, lbl) in zip(cols, metrics):
        with col:
            st.markdown(render_metric(f"{val:,}", lbl), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main tabs ─────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📄 Papers Overview",
        "🔍 Concept Traceability",
        "🔑 Keywords",
        "🌳 Hierarchy",
        "🕸️ Knowledge Graph",
        "🧠 Mind Map",
        "📊 Similarity",
        "🔍 Search",
        "📚 Literature Review",
        "💾 Export & JSON",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 – Papers Overview & Extracted Fields
    # ════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown('<div class="section-header">📄 Paper Details & Structure</div>', unsafe_allow_html=True)

        paper_selector = st.selectbox(
            "Select Paper",
            options=range(len(papers)),
            format_func=lambda i: f"{papers[i].file_name} – {short(papers[i].title)}",
        )
        p = papers[paper_selector]

        # Identity Card
        st.info(f"🆔 **Paper ID:** `{p.paper_id}` | 🔒 **File Hash (SHA-256):** `{p.file_hash[:20]}...` | 📅 **Year:** {p.year or 'N/A'}")

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Pages", p.page_count)
        with c2: st.metric("Words", f"{p.word_count:,}")
        with c3: st.metric("Authors", len(p.authors))
        with c4: st.metric("Domains", len(p.research_domains))

        st.markdown(f"### 📌 {p.title}")
        if p.authors:
            st.markdown("**Authors:** " + ", ".join(p.authors[:8]))

        if p.research_domains:
            st.markdown("**Research Domains:**")
            st.markdown(render_chips(p.research_domains), unsafe_allow_html=True)

        st.markdown('<div class="section-header">Abstract</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="abstract-box">{p.abstract or "Abstract not detected."}</div>', unsafe_allow_html=True)

        # Structured Paper Sections
        st.markdown('<div class="section-header">Paper Structural Analysis</div>', unsafe_allow_html=True)
        col_sec1, col_sec2 = st.columns(2)
        with col_sec1:
            st.markdown(f"**Research Problem:** {p.problem or 'Not explicitly extracted'}")
            st.markdown(f"**Research Objective:** {p.objective or 'Not explicitly extracted'}")
            st.markdown(f"**Proposed Method:** {p.proposed_method.get('name', 'N/A')}")
            st.write(p.proposed_method.get('description', ''))
            if p.models:
                st.markdown(f"**Models:** {', '.join(p.models)}")
            if p.algorithms:
                st.markdown(f"**Algorithms:** {', '.join(p.algorithms)}")

        with col_sec2:
            if p.datasets:
                st.markdown(f"**Datasets:** {', '.join(p.datasets)}")
            if p.evaluation_metrics:
                st.markdown(f"**Evaluation Metrics:** {', '.join(p.evaluation_metrics)}")
            if p.results:
                st.markdown("**Key Results:**")
                for r in p.results[:3]:
                    st.markdown(f"- {r}")
            if p.advantages:
                st.markdown(f"**Advantages:** {', '.join(p.advantages[:3])}")
            if p.limitations:
                st.markdown(f"**Limitations:** {', '.join(p.limitations[:3])}")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 – Source Traceability
    # ════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown('<div class="section-header">🔍 Concept Source Traceability</div>', unsafe_allow_html=True)
        paper_sel_tr = st.selectbox(
            "Select Paper to Inspect Traceability",
            options=range(len(papers)),
            format_func=lambda i: papers[i].file_name,
            key="trace_sel",
        )
        ptr = papers[paper_sel_tr]

        if ptr.traceable_concepts:
            st.write("Each concept below is verified against the uploaded PDF text with sentence evidence and page numbers.")
            trace_df = pd.DataFrame(ptr.traceable_concepts)
            st.dataframe(
                trace_df,
                use_container_width=True,
                column_config={
                    "concept": st.column_config.TextColumn("Extracted Concept", width="medium"),
                    "source": st.column_config.TextColumn("Source", width="small"),
                    "confidence": st.column_config.NumberColumn("Confidence", format="%.2f", width="small"),
                    "page": st.column_config.NumberColumn("Page #", width="small"),
                    "evidence": st.column_config.TextColumn("Paper Sentence Evidence", width="large"),
                }
            )
        else:
            st.info("No traceable concepts available for this paper.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 – Keywords
    # ════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown('<div class="section-header">🔑 Keyword Extraction</div>', unsafe_allow_html=True)

        paper_sel2 = st.selectbox(
            "Select Paper",
            options=range(len(papers)),
            format_func=lambda i: papers[i].file_name,
            key="kw_sel",
        )
        p2 = papers[paper_sel2]

        kw_tab1, kw_tab2, kw_tab3, kw_tab4 = st.tabs(["Merged", "TF-IDF", "KeyBERT", "YAKE"])

        def kw_table(kws: List[Dict], col_name: str = "Keyword") -> None:
            if not kws:
                st.info("No keywords extracted.")
                return
            df = pd.DataFrame(kws[:keyword_top_n])
            if "word" in df.columns:
                df.columns = [col_name, "Score"]
                df[col_name] = df[col_name].str.title()
                df["Score"] = df["Score"].round(4)
                st.dataframe(df, use_container_width=True, height=350)

                fig = go.Figure(
                    go.Bar(
                        x=df["Score"],
                        y=df[col_name],
                        orientation="h",
                        marker_color="#6C63FF",
                        text=df["Score"].astype(str),
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    paper_bgcolor="#0F0F1A",
                    plot_bgcolor="#0F0F1A",
                    font=dict(color="#E0E0E0"),
                    height=400,
                    margin=dict(l=150, r=20, t=20, b=30),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True)

        with kw_tab1:
            merged = [{"word": w, "score": 1.0} for w in p2.all_keywords[:keyword_top_n]]
            kw_table(merged, "Merged Keyword")

        with kw_tab2:
            kw_table(p2.tfidf_keywords[:keyword_top_n], "TF-IDF Keyword")

        with kw_tab3:
            kw_table(p2.keybert_keywords[:keyword_top_n], "KeyBERT Keyword")

        with kw_tab4:
            kw_table(p2.yake_keywords[:keyword_top_n], "YAKE Keyword")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 – Hierarchy
    # ════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown('<div class="section-header">🌳 Hierarchical Concept Map</div>', unsafe_allow_html=True)

        paper_sel3 = st.selectbox(
            "Select Paper",
            options=range(len(papers)),
            format_func=lambda i: papers[i].file_name,
            key="hier_sel",
        )
        p3 = papers[paper_sel3]

        hier_viz = HierarchyViz()
        builder  = HierarchyBuilder()

        hier_tab1, hier_tab2, hier_tab3, hier_tab4 = st.tabs(
            ["☀️ Sunburst", "🗂️ Treemap", "🌊 Icicle", "📋 Path List"]
        )

        tree = p3.concept_hierarchy

        with hier_tab1:
            if tree:
                fig_sb = hier_viz.sunburst(tree, title=f"Sunburst – {short(p3.title, 50)}")
                st.plotly_chart(fig_sb, use_container_width=True)

        with hier_tab2:
            if tree:
                fig_tm = hier_viz.treemap(tree, title=f"Treemap – {short(p3.title, 50)}")
                st.plotly_chart(fig_tm, use_container_width=True)

        with hier_tab3:
            if tree:
                fig_ic = hier_viz.icicle(tree, title=f"Icicle – {short(p3.title, 50)}")
                st.plotly_chart(fig_ic, use_container_width=True)

        with hier_tab4:
            paths = builder.get_hierarchy_paths(tree)
            if paths:
                for path in paths[:50]:
                    st.markdown(
                        f'<div class="hierarchy-path">📍 {path}</div>',
                        unsafe_allow_html=True,
                    )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 5 – Knowledge Graph
    # ════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown('<div class="section-header">🕸️ Knowledge Graph</div>', unsafe_allow_html=True)

        kg_mode = st.radio(
            "Graph mode",
            ["All Papers (combined)", "Single Paper hierarchy", "Concept-to-Concept (CCC) Semantic Network"],
            horizontal=True,
        )

        kg = KnowledgeGraph()

        if kg_mode == "All Papers (combined)":
            papers_data_list = [
                {
                    "title":           p.title,
                    "keywords":        p.all_keywords[:15],
                    "technical_terms": p.technical_terms[:10],
                    "research_domains":p.research_domains,
                }
                for p in papers
            ]
            kg.build_from_papers(papers_data_list)
            fig_kg = kg.to_plotly(title="Combined Knowledge Graph")
            st.plotly_chart(fig_kg, use_container_width=True)

        elif kg_mode == "Concept-to-Concept (CCC) Semantic Network":
            paper_sel_kg = st.selectbox(
                "Select Paper for CCC Analysis",
                options=range(len(papers)),
                format_func=lambda i: f"{papers[i].file_name} – {short(papers[i].title)}",
                key="ccc_sel",
            )
            pk = papers[paper_sel_kg]
            ccc_info = getattr(pk, "ccc_mapping", {})
            if ccc_info:
                kg.build_from_ccc(ccc_info, pk.title[:50])
                fig_kg = kg.to_plotly(title=f"CCC Concept Network – {short(pk.title)}")
                st.plotly_chart(fig_kg, use_container_width=True)

                # Show CCC stats
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**🌉 Cross-Domain Bridge Concepts:**")
                    bridges = ccc_info.get("cross_domain_bridges", [])
                    if bridges:
                        for b in bridges:
                            st.markdown(f"- **{b['concept']}** (Spans: {', '.join(b['connected_domains'])})")
                    else:
                        st.info("No cross-domain bridges detected for this single paper.")

                with c2:
                    st.markdown("**🔗 Top Concept-to-Concept (C2C) Associations:**")
                    edges = ccc_info.get("c2c_edges", [])
                    if edges:
                        for e in edges[:8]:
                            st.markdown(f"- `{e['source']}` ⟷ `{e['target']}` (Weight: {e['weight']:.2f})")
                    else:
                        st.info("No C2C edges extracted.")
            else:
                st.warning("CCC mapping data not available for this paper.")

        else:
            paper_sel_kg = st.selectbox(
                "Select Paper",
                options=range(len(papers)),
                format_func=lambda i: papers[i].file_name,
                key="kg_sel",
            )
            pk = papers[paper_sel_kg]
            kg.build_from_hierarchy(pk.concept_hierarchy, pk.title[:50])
            fig_kg = kg.to_plotly(title=f"Knowledge Graph – {short(pk.title)}")
            st.plotly_chart(fig_kg, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 6 – Mind Map
    # ════════════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown('<div class="section-header">🧠 Mind Map</div>', unsafe_allow_html=True)

        paper_sel_mm = st.selectbox(
            "Select Paper",
            options=range(len(papers)),
            format_func=lambda i: papers[i].file_name,
            key="mm_sel",
        )
        pm = papers[paper_sel_mm]
        mm_depth = st.slider("Depth", 1, 4, 3, key="mm_depth")

        mm_viz = MindMapViz()
        fig_mm = mm_viz.build(pm.title, pm.concept_hierarchy, max_depth=mm_depth)
        st.plotly_chart(fig_mm, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 7 – Similarity Analysis
    # ════════════════════════════════════════════════════════════════════════
    with tabs[6]:
        st.markdown('<div class="section-header">📊 Pairwise Similarity Analysis</div>', unsafe_allow_html=True)

        if len(papers) < 2 or sim_results.get("similarity_status") == "insufficient_papers":
            st.info("ℹ️ **similarity_status: insufficient_papers** — Upload at least 2 research papers to perform pairwise similarity analysis.")
        else:
            hm_viz = HeatmapViz()
            labels = sim_results.get("labels", [])

            sim_tab1, sim_tab2, sim_tab3, sim_tab4 = st.tabs(
                ["Combined", "TF-IDF", "Semantic", "Top Pairs"]
            )

            def show_heatmap(matrix, title):
                if matrix is None:
                    st.info("This method was not computed.")
                    return
                fig_hm = hm_viz.build(matrix, labels, title=title)
                st.plotly_chart(fig_hm, use_container_width=True)

            with sim_tab1:
                show_heatmap(sim_results.get("combined_matrix"), "Combined Similarity Heatmap")

            with sim_tab2:
                show_heatmap(sim_results.get("tfidf_matrix"), "TF-IDF Cosine Similarity")

            with sim_tab3:
                show_heatmap(sim_results.get("semantic_matrix"), "Semantic Similarity (Sentence Transformers)")

            with sim_tab4:
                st.markdown("**Most Similar Paper Pairs**")
                matrix = sim_results.get("combined_matrix") or sim_results.get("tfidf_matrix")
                if matrix is not None:
                    from similarity.similarity_engine import SimilarityEngine
                    se = SimilarityEngine()
                    pairs = se.get_top_similar_pairs(matrix, labels, top_n=20)
                    for pair in pairs:
                        score_color = "#00C49A" if pair["score"] > 0.7 else "#F9A826" if pair["score"] > 0.4 else "#FF6584"
                        st.markdown(
                            f"""<div class="sim-pair">
                                <span style="color:#ccc;flex:1">{pair['paper_a']}</span>
                                <span style="color:#666">↔</span>
                                <span style="color:#ccc;flex:1">{pair['paper_b']}</span>
                                <span class="sim-score" style="color:{score_color}">{pair['score']:.3f}</span>
                            </div>""",
                            unsafe_allow_html=True,
                        )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 8 – Search Concepts & Keywords
    # ════════════════════════════════════════════════════════════════════════
    with tabs[7]:
        st.markdown('<div class="section-header">🔍 Search Concepts & Keywords</div>', unsafe_allow_html=True)

        search_query = st.text_input(
            "Search",
            placeholder="e.g. 'transformer', 'adversarial training', 'BERT'…",
            label_visibility="collapsed",
        )
        search_in = st.radio(
            "Search in",
            ["all", "keywords", "concepts", "text"],
            horizontal=True,
        )

        if search_query:
            engine = SearchEngine()
            papers_data_search = [
                {
                    "title":           p.title,
                    "keywords":        p.all_keywords,
                    "technical_terms": p.technical_terms,
                    "research_domains":p.research_domains,
                    "abstract":        p.abstract,
                    "full_text":       p.full_text[:8000],
                }
                for p in papers
            ]
            results = engine.search(search_query, papers_data_search, search_in=search_in)

            if results:
                st.success(f"Found **{len(results)}** result(s)")
                for r in results[:25]:
                    badge_color = {"keyword": "#6C63FF", "concept": "#43BCCD",
                                   "domain": "#F9A826", "text": "#00C49A"}.get(r.match_type, "#888")
                    st.markdown(
                        f"""<div class="search-result">
                            <span style="background:{badge_color};color:white;border-radius:4px;padding:1px 8px;font-size:.72rem">
                                {r.match_type}
                            </span>
                            <strong style="color:#E0E0E0;margin-left:.5rem">
                                {short(r.paper_title, 40)}
                            </strong>
                            <span style="color:#F9A826;margin-left:.5rem">
                                Score: {r.score:.3f}
                            </span>
                            <div style="margin-top:.3rem">{r.highlighted or r.matched_term}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 9 – Literature Review
    # ════════════════════════════════════════════════════════════════════════
    with tabs[8]:
        st.markdown('<div class="section-header">📚 Literature Review & Comparison</div>', unsafe_allow_html=True)
        comparison_path = ROOT / "Consolidated_Paper_Comparison.xlsx"
        if comparison_path.exists():
            try:
                df_comp = pd.read_excel(comparison_path)
                st.dataframe(df_comp, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading comparison matrix: {e}")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 10 – Export & Section 6 Structured JSON
    # ════════════════════════════════════════════════════════════════════════
    with tabs[9]:
        st.markdown('<div class="section-header">💾 Export Results & Section 6 JSON</div>', unsafe_allow_html=True)
        paper_sel_ex = st.selectbox(
            "Select Paper to Export",
            options=range(len(papers)),
            format_func=lambda i: papers[i].file_name,
            key="exp_sel",
        )
        pex = papers[paper_sel_ex]

        st.markdown("### 🗄️ Standard Paper Analysis JSON (Section 6 Schema)")
        sec6_json = json.dumps(pex.to_structured_json(), indent=2)
        st.download_button(
            f"⬇️ Download {pex.file_name}_analysis.json",
            data=sec6_json,
            file_name=f"{pex.file_name}_analysis.json",
            mime="application/json",
            use_container_width=True,
        )

        with st.expander("👁️ Preview Section 6 Structured JSON"):
            st.json(pex.to_structured_json())

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center;padding:2rem 0 1rem;color:rgba(255,255,255,0.2);font-size:0.8rem">
        HMRP · Hierarchical Concept Mapping for Research Paper Similarity Analysis
    </div>
    """,
    unsafe_allow_html=True,
)
