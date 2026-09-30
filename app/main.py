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
import networkx as nx

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
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-dark: #090D16;
        --card-bg: rgba(15, 23, 42, 0.7);
        --card-border: rgba(56, 189, 248, 0.18);
        --card-hover-border: rgba(56, 189, 248, 0.5);
        --accent-cyan: #38BDF8;
        --accent-indigo: #818CF8;
        --accent-rose: #FB7185;
        --accent-emerald: #34D399;
        --accent-amber: #FBBF24;
    }

    * {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    code, pre, .hierarchy-path {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── Global Background & Container ── */
    .stApp {
        background: radial-gradient(ellipse 80% 60% at 50% -20%, rgba(99, 102, 241, 0.18), rgba(9, 13, 22, 1) 75%) !important;
        background-color: var(--bg-dark) !important;
        color: #F1F5F9;
    }
    .block-container { 
        padding: 2rem 2.5rem 3.5rem !important; 
        max-width: 1440px;
    }

    /* ── Futuristic Top Navigation Bar ── */
    .saas-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 0.9rem 1.6rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    .saas-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-icon {
        background: linear-gradient(135deg, #6366F1, #38BDF8);
        width: 38px;
        height: 38px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
    }
    .brand-title {
        font-weight: 800;
        font-size: 1.15rem;
        letter-spacing: -0.3px;
        background: linear-gradient(90deg, #FFFFFF 0%, #E2E8F0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .brand-tag {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        padding: 3px 8px;
        border-radius: 6px;
        border: 1px solid rgba(56, 189, 248, 0.25);
    }
    .nav-status {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.82rem;
        color: #94A3B8;
        font-weight: 500;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10B981;
        box-shadow: 0 0 10px #10B981;
        animation: pulse-dot 2s infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.25rem 1rem;
        text-align: center;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 16px 32px -8px rgba(56, 189, 248, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.76rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
        margin-top: 0.35rem;
    }

    /* ── Section Headers ── */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #F8FAFC;
        border-left: 3px solid #38BDF8;
        padding-left: 0.8rem;
        margin: 1.6rem 0 1.1rem 0;
        letter-spacing: -0.2px;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Feature Cards ── */
    .feature-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.4rem;
        transition: all 0.25s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 12px 28px -6px rgba(99, 102, 241, 0.18);
        background: rgba(20, 30, 55, 0.7);
    }
    .feature-icon-wrapper {
        width: 42px;
        height: 42px;
        border-radius: 10px;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.25);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        margin-bottom: 0.9rem;
    }

    /* ── Smart Keyword Chips ── */
    .kw-chip {
        display: inline-flex;
        align-items: center;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.2);
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
        border-radius: 8px;
        padding: 0.3rem 0.75rem;
        font-size: 0.8rem;
        font-weight: 500;
        color: #E2E8F0;
        margin: 0.2rem;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .kw-chip:hover {
        background: rgba(56, 189, 248, 0.15);
        border-color: rgba(56, 189, 248, 0.5);
        color: #38BDF8;
        transform: scale(1.04);
    }

    /* ── Abstract & Detail Boxes ── */
    .abstract-box {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.3rem 1.6rem;
        color: #E2E8F0;
        font-size: 0.92rem;
        line-height: 1.75;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
    }

    /* ── Hierarchy Pathways ── */
    .hierarchy-path {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 0.8rem 1.1rem;
        margin: 0.45rem 0;
        color: #E2E8F0;
        font-size: 0.86rem;
        transition: all 0.2s ease;
    }
    .hierarchy-path:hover {
        border-color: rgba(56, 189, 248, 0.4);
        background: rgba(20, 30, 55, 0.7);
    }

    /* ── Similarity Pair Cards ── */
    .sim-pair {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.8rem 1.2rem;
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.07);
        margin: 0.4rem 0;
        transition: all 0.2s ease;
    }
    .sim-pair:hover {
        border-color: rgba(56, 189, 248, 0.45);
        background: rgba(20, 30, 55, 0.8);
        transform: translateX(4px);
    }
    .sim-score {
        font-weight: 800;
        font-size: 1rem;
        min-width: 60px;
        text-align: right;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Sidebar Redesign ── */
    [data-testid="stSidebar"] {
        background: #0B0F19 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.07) !important;
    }

    /* ── Polished Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(12px);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.07);
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 8px;
        color: #94A3B8;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0.55rem 1.1rem;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #F1F5F9;
        background: rgba(255, 255, 255, 0.05);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.9), rgba(56, 189, 248, 0.9)) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    /* ── Button Glow ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #38BDF8 100%) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.6rem !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(56, 189, 248, 0.45) !important;
        filter: brightness(1.08);
    }

    /* ── DataFrames ── */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #090D16; }
    ::-webkit-scrollbar-thumb { 
        background: rgba(148, 163, 184, 0.25); 
        border-radius: 3px; 
    }
    ::-webkit-scrollbar-thumb:hover { 
        background: rgba(56, 189, 248, 0.5); 
    }
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
# Sidebar – upload & controls
# ─────────────────────────────────────────────────────────────────────────────


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
# Top Brand Bar
# ─────────────────────────────────────────────────────────────────────────────
papers: List[Paper] = st.session_state.papers
sim_results: Dict   = st.session_state.sim_results

status_badge = f'<span class="status-dot"></span> {len(papers)} Papers Loaded' if papers else '<span class="status-dot" style="background:#64748B;box-shadow:none;"></span> Ready for Ingestion'

st.markdown(
    f"""
    <div class="saas-nav">
        <div class="saas-brand">
            <div class="brand-icon">⚡</div>
            <div>
                <span class="brand-title">IRNLP Concept Intelligence</span>
                <span class="brand-tag">v2.4 Enterprise</span>
            </div>
        </div>
        <div class="nav-status">
            {status_badge}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not papers:
    # ── Landing state ────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding: 2rem 1rem 2.5rem;">
            <div style="display:inline-block; margin-bottom:1rem; background:rgba(56,189,248,0.1); border:1px solid rgba(56,189,248,0.3); padding:6px 18px; border-radius:9999px;">
                <span style="color:#38BDF8; font-size:0.85rem; font-weight:700; letter-spacing:0.5px;">✨ NEXT-GEN AGENTIC RESEARCH INTELLIGENCE</span>
            </div>
            <h1 style="font-size:2.8rem; font-weight:800; letter-spacing:-1px; margin:0 auto 1rem; max-width:850px; background:linear-gradient(135deg, #FFFFFF 30%, #94A3B8 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                Hierarchical Concept Extraction & Cross-Paper Semantic Discovery
            </h1>
            <p style="color:#94A3B8; max-width:680px; margin:0 auto 2.5rem; font-size:1.05rem; line-height:1.7;">
                Accelerate literature reviews with automated zero-leakage paper isolation, multi-tiered taxonomy mapping, tri-model keyword extraction, and dynamic semantic knowledge graphs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">⚡ Platform Capabilities</div>', unsafe_allow_html=True)
    features = [
        ("📑", "Identity & Isolation", "SHA-256 zero-leakage paper isolation & rigorous section verification."),
        ("🔑", "Tri-Model Keywords", "Ranked comparative keyword extraction via TF-IDF, KeyBERT, & YAKE algorithms."),
        ("🌳", "Taxonomy & Hierarchy", "Interactive Sunburst, Treemap, and Icicle visualizers with multi-tier concept lineages."),
        ("🔗", "Semantic Similarity", "Dense vector embeddings and cosine similarity heatmaps across paper corpuses."),
        ("🕸️", "Knowledge Graph", "Cross-domain bridge discovery and Concept-to-Concept (CCC) semantic networks."),
        ("🔍", "Source Traceability", "Grounded sentence citations with page numbers and confidence score validation."),
        ("📚", "Literature Review", "Unified comparative synthesis across objectives, proposed methods, and metrics."),
        ("💾", "Structured Export", "Instant Section 6 JSON exports formatted for academic publication pipelines."),
    ]
    cols = st.columns(4)
    for i, (icon, title_f, desc) in enumerate(features):
        with cols[i % 4]:
            st.markdown(
                f"""<div class="feature-card" style="margin-bottom:1rem;min-height:165px;">
                    <div class="feature-icon-wrapper">{icon}</div>
                    <div style="color:#F8FAFC;font-weight:700;font-size:0.98rem;margin-bottom:0.35rem;">{title_f}</div>
                    <div style="color:#94A3B8;font-size:0.82rem;line-height:1.5;">{desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hierarchy-path" style="padding:1.4rem 1.8rem;background:rgba(56,189,248,0.06);border-left:4px solid #38BDF8;border-radius:14px;">
            <div style="margin:0 0 8px 0;color:#38BDF8;font-size:1.05rem;font-weight:700;">🚀 Quick Start Workspace:</div>
            <div style="color:#CBD5E1;font-size:0.92rem;line-height:1.7;">
                1️⃣ Open the <b>Left Control Sidebar</b> and upload one or more research papers (PDF format).<br>
                2️⃣ Configure your similarity strategy (<b>TF-IDF</b>, <b>Semantic Embeddings</b>, or <b>Hybrid</b>).<br>
                3️⃣ Click <b>🚀 Analyse Papers</b> to launch the deep ingestion and visual mapping pipeline!
            </div>
        </div>
        """,
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
        "📊 Similarity",
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

        # Identity Card & Quick Stats
        st.markdown(
            f"""
            <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:1rem 1.4rem;margin-bottom:1.2rem;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
                <div style="font-size:0.88rem;color:#94A3B8;">
                    🆔 <span style="color:#CBD5E1;font-family:'JetBrains Mono';font-weight:600;">{p.paper_id}</span> &nbsp;|&nbsp; 
                    🔒 <span style="color:#64748B;font-family:'JetBrains Mono';">{p.file_hash[:22]}...</span>
                </div>
                <div style="display:flex;gap:10px;">
                    <span class="kw-chip" style="color:#38BDF8;background:rgba(56,189,248,0.1);border-color:rgba(56,189,248,0.25);">📅 {p.year or 'N/A'}</span>
                    <span class="kw-chip" style="color:#34D399;background:rgba(52,211,153,0.1);border-color:rgba(52,211,153,0.25);">📄 {p.page_count} Pages</span>
                    <span class="kw-chip" style="color:#FBBF24;background:rgba(251,191,36,0.1);border-color:rgba(251,191,36,0.25);">📝 {p.word_count:,} Words</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"### 📌 {p.title}")
        if p.authors:
            st.markdown("**Authors:** " + ", ".join(p.authors[:8]))

        if p.research_domains:
            st.markdown("**Research Domains:**")
            st.markdown(render_chips(p.research_domains), unsafe_allow_html=True)

        st.markdown('<div class="section-header">Abstract</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="abstract-box">{p.abstract or "Abstract not detected."}</div>', unsafe_allow_html=True)

        # Structured Paper Sections
        st.markdown('<div class="section-header">📑 Paper Structural Analysis</div>', unsafe_allow_html=True)
        col_sec1, col_sec2 = st.columns(2)
        with col_sec1:
            st.markdown(
                f"""<div class="hierarchy-path" style="margin-bottom:12px;">
                    <div style="font-weight:700;color:#A78BFA;font-size:0.95rem;margin-bottom:4px;">🎯 Research Problem</div>
                    <div style="color:#E5E7EB;font-size:0.9rem;">{p.problem or 'Not explicitly identified in text'}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""<div class="hierarchy-path" style="margin-bottom:12px;">
                    <div style="font-weight:700;color:#60A5FA;font-size:0.95rem;margin-bottom:4px;">🎯 Research Objective</div>
                    <div style="color:#E5E7EB;font-size:0.9rem;">{p.objective or 'Not explicitly identified in text'}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            proposed_name = p.proposed_method.get('name', 'N/A') if isinstance(p.proposed_method, dict) else str(p.proposed_method)
            proposed_desc = p.proposed_method.get('description', '') if isinstance(p.proposed_method, dict) else ''
            st.markdown(
                f"""<div class="hierarchy-path" style="margin-bottom:12px;">
                    <div style="font-weight:700;color:#34D399;font-size:0.95rem;margin-bottom:4px;">🛠️ Proposed Method</div>
                    <div style="color:#FFFFFF;font-weight:600;font-size:0.92rem;">{proposed_name}</div>
                    <div style="color:rgba(229,231,235,0.85);font-size:0.86rem;margin-top:2px;">{proposed_desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            if p.models:
                st.markdown(
                    f"""<div class="hierarchy-path" style="margin-bottom:12px;">
                        <div style="font-weight:700;color:#FBBF24;font-size:0.95rem;margin-bottom:4px;">🧠 Models & Architectures</div>
                        <div style="color:#E5E7EB;font-size:0.9rem;">{', '.join(p.models)}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        with col_sec2:
            if p.datasets:
                st.markdown(
                    f"""<div class="hierarchy-path" style="margin-bottom:12px;">
                        <div style="font-weight:700;color:#38BDF8;font-size:0.95rem;margin-bottom:4px;">📂 Datasets Used</div>
                        <div style="color:#E5E7EB;font-size:0.9rem;">{', '.join(p.datasets)}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            if p.evaluation_metrics:
                st.markdown(
                    f"""<div class="hierarchy-path" style="margin-bottom:12px;">
                        <div style="font-weight:700;color:#F472B6;font-size:0.95rem;margin-bottom:4px;">📊 Evaluation Metrics</div>
                        <div style="color:#E5E7EB;font-size:0.9rem;">{', '.join(p.evaluation_metrics)}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            if p.results:
                res_bullets = "".join(f"<li style='margin-bottom:3px;'>{r}</li>" for r in p.results[:3])
                st.markdown(
                    f"""<div class="hierarchy-path" style="margin-bottom:12px;">
                        <div style="font-weight:700;color:#4ADE80;font-size:0.95rem;margin-bottom:4px;">📈 Key Quantitative Results</div>
                        <ul style="color:#E5E7EB;font-size:0.88rem;padding-left:18px;margin-top:4px;">{res_bullets}</ul>
                    </div>""",
                    unsafe_allow_html=True,
                )
            if p.advantages or p.limitations:
                adv_str = ', '.join(p.advantages[:2]) if p.advantages else 'None listed'
                lim_str = ', '.join(p.limitations[:2]) if p.limitations else 'None listed'
                st.markdown(
                    f"""<div class="hierarchy-path" style="margin-bottom:12px;">
                        <div style="font-weight:700;color:#C084FC;font-size:0.95rem;margin-bottom:4px;">⚖️ Advantages & Limitations</div>
                        <div style="color:#E5E7EB;font-size:0.88rem;"><b>Pros:</b> {adv_str}</div>
                        <div style="color:#E5E7EB;font-size:0.88rem;margin-top:2px;"><b>Cons:</b> {lim_str}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

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
                df[col_name] = df[col_name].astype(str).str.title()
                df["Score"] = pd.to_numeric(df["Score"], errors="coerce").fillna(1.0).round(4)

                col_t1, col_t2 = st.columns([1, 1])
                with col_t1:
                    st.dataframe(df, use_container_width=True, height=350)
                with col_t2:
                    colors = ["#38BDF8" if i % 2 == 0 else "#818CF8" for i in range(len(df))]
                    fig = go.Figure(
                        go.Bar(
                            x=df["Score"],
                            y=df[col_name],
                            orientation="h",
                            marker=dict(color=colors, line=dict(width=0)),
                            text=df["Score"].astype(str),
                            textposition="outside",
                        )
                    )
                    fig.update_layout(
                        paper_bgcolor="#0B0F19",
                        plot_bgcolor="#0B0F19",
                        font=dict(color="#F1F5F9", family="Plus Jakarta Sans"),
                        height=350,
                        margin=dict(l=120, r=30, t=20, b=20),
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with kw_tab1:
            merged = [{"word": w, "score": 1.0 - (i * 0.03)} for i, w in enumerate(p2.all_keywords[:keyword_top_n])]
            kw_table(merged, "Merged Keyword")

        with kw_tab2:
            kw_table(p2.tfidf_keywords[:keyword_top_n], "TF-IDF Keyword")

        with kw_tab3:
            kw_table(p2.keybert_keywords[:keyword_top_n], "KeyBERT Keyword")

        with kw_tab4:
            kw_table(p2.yake_keywords[:keyword_top_n], "YAKE Keyword")

        # ── Below: Keyword Co-occurrence & Semantic Network Graph ─────────────
        st.markdown("---")
        st.markdown("#### 🕸️ Keyword Semantic & Co-occurrence Network")
        st.caption("Visualizing semantic relationships and text associations between the top extracted keywords for this paper:")

        kw_nodes = p2.all_keywords[:15]
        if kw_nodes:
            kw_graph = nx.Graph()
            # Add center paper node
            center_label = short(p2.title, 35)
            kw_graph.add_node(center_label, node_type="paper", label=center_label)

            for i, kw_w in enumerate(kw_nodes):
                kw_title = kw_w.title()
                kw_graph.add_node(kw_title, node_type="keyword", label=kw_title)
                kw_graph.add_edge(center_label, kw_title, weight=1.5)

                # Connect keywords that appear in same sentences or share tokens
                for j in range(i + 1, len(kw_nodes)):
                    other_kw = kw_nodes[j].title()
                    w1_tokens = set(kw_w.lower().split())
                    w2_tokens = set(kw_nodes[j].lower().split())
                    if w1_tokens & w2_tokens or (len(kw_w) > 3 and kw_w.lower() in kw_nodes[j].lower()):
                        kw_graph.add_edge(kw_title, other_kw, weight=2.0)

            # Render Plotly network for keywords
            try:
                pos_kw = nx.kamada_kawai_layout(kw_graph)
            except Exception:
                pos_kw = nx.spring_layout(kw_graph, seed=42)

            k_edge_x, k_edge_y = [], []
            for u, v in kw_graph.edges():
                if u in pos_kw and v in pos_kw:
                    k_edge_x += [pos_kw[u][0], pos_kw[v][0], None]
                    k_edge_y += [pos_kw[u][1], pos_kw[v][1], None]

            k_edge_trace = go.Scatter(
                x=k_edge_x, y=k_edge_y,
                mode="lines",
                line=dict(width=1.2, color="rgba(148, 163, 184, 0.35)"),
                hoverinfo="none",
            )

            # Node markers
            k_node_x, k_node_y, k_node_text, k_node_colors, k_node_sizes = [], [], [], [], []
            for n, attrs in kw_graph.nodes(data=True):
                if n in pos_kw:
                    k_node_x.append(pos_kw[n][0])
                    k_node_y.append(pos_kw[n][1])
                    k_node_text.append(n)
                    if attrs.get("node_type") == "paper":
                        k_node_colors.append("#F43F5E")
                        k_node_sizes.append(22)
                    else:
                        k_node_colors.append("#06B6D4")
                        k_node_sizes.append(14)

            k_node_trace = go.Scatter(
                x=k_node_x, y=k_node_y,
                mode="markers+text",
                text=k_node_text,
                textposition="top center",
                textfont=dict(size=10, color="#CBD5E1", family="Plus Jakarta Sans"),
                marker=dict(
                    size=k_node_sizes,
                    color=k_node_colors,
                    line=dict(width=1.5, color="#0B0F19"),
                ),
                hovertemplate="<b>%{text}</b><extra></extra>",
            )

            fig_kw_net = go.Figure(data=[k_edge_trace, k_node_trace])
            fig_kw_net.update_layout(
                title=dict(text=f"<b style='color:#38BDF8;'>Keyword Association Network – {short(p2.title, 40)}</b>", font=dict(size=15, color="#38BDF8", family="Plus Jakarta Sans")),
                paper_bgcolor="#0B0F19",
                plot_bgcolor="#0B0F19",
                font=dict(color="#F1F5F9", family="Plus Jakarta Sans"),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                margin=dict(l=10, r=10, t=40, b=10),
                height=500,
            )
            st.plotly_chart(fig_kw_net, use_container_width=True)

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
        if not tree and (p3.research_concepts or p3.all_keywords):
            tree, _ = builder.build(
                concepts=p3.research_concepts or [],
                keywords=p3.all_keywords or [],
                text=p3.full_text[:20_000] if p3.full_text else "",
            )
            p3.concept_hierarchy = tree

        if not tree:
            st.info("ℹ️ No concept hierarchy data available for this paper. Make sure paper analysis is completed.")
        else:
            with hier_tab1:
                fig_sb = hier_viz.sunburst(tree, title=f"Sunburst – {short(p3.title, 50)}")
                st.plotly_chart(fig_sb, use_container_width=True)

            with hier_tab2:
                fig_tm = hier_viz.treemap(tree, title=f"Treemap – {short(p3.title, 50)}")
                st.plotly_chart(fig_tm, use_container_width=True)

            with hier_tab3:
                fig_ic = hier_viz.icicle(tree, title=f"Icicle – {short(p3.title, 50)}")
                st.plotly_chart(fig_ic, use_container_width=True)

            with hier_tab4:
                col_p1, col_p2 = st.columns([2, 1])
                with col_p2:
                    only_main = st.checkbox("Only Main Concepts", value=True, help="Hide miscellaneous/uncategorized terms and show only structured domain concept taxonomies.")
                with col_p1:
                    filter_txt = st.text_input("🔍 Filter paths", "", placeholder="Type concept name (e.g. BERT, NLP, Attention)...", key="hier_path_filter")

                paths = builder.get_hierarchy_paths(tree, only_main_concepts=only_main)
                if paths:
                    st.caption(f"Showing {len(paths)} main concept lineage pathways:")
                    filtered_paths = [p for p in paths if filter_txt.lower() in p.lower()] if filter_txt else paths
                    
                    for path in filtered_paths[:100]:
                        parts = [p.strip() for p in path.split(">")]
                        badge_html = " <span style='color:#6C63FF;font-weight:bold;'>→</span> ".join(
                            f"<span style='background:rgba(108,99,255,0.15);border:1px solid rgba(108,99,255,0.3);padding:2px 8px;border-radius:6px;font-size:13px;color:#E0E0E0;'>{p}</span>"
                            for p in parts
                        )
                        st.markdown(
                            f'<div class="hierarchy-path" style="margin-bottom:8px;padding:8px 12px;background:rgba(255,255,255,0.03);border-radius:8px;">📍 {badge_html}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No main hierarchy paths found for this selection.")

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
    # TAB 6 – Similarity Analysis
    # ════════════════════════════════════════════════════════════════════════
    with tabs[5]:
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
                matrix = sim_results.get("combined_matrix")
                if matrix is None:
                    matrix = sim_results.get("tfidf_matrix")
                if matrix is not None:
                    from similarity.similarity_engine import SimilarityEngine
                    se = SimilarityEngine()
                    pairs = se.get_top_similar_pairs(matrix, labels, top_n=20)
                    for pair in pairs:
                        score_color = "#34D399" if pair["score"] > 0.7 else "#FBBF24" if pair["score"] > 0.4 else "#FB7185"
                        st.markdown(
                            f"""<div class="sim-pair">
                                <span style="color:#E2E8F0;flex:1;font-weight:500;">{pair['paper_a']}</span>
                                <span style="color:#64748B;font-weight:700;">⟷</span>
                                <span style="color:#E2E8F0;flex:1;font-weight:500;">{pair['paper_b']}</span>
                                <span class="sim-score" style="color:{score_color};background:rgba(255,255,255,0.05);padding:4px 8px;border-radius:6px;">{pair['score']:.4f}</span>
                            </div>""",
                            unsafe_allow_html=True,
                        )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 7 – Literature Review
    # ════════════════════════════════════════════════════════════════════════
    with tabs[6]:
        st.markdown('<div class="section-header">📚 Literature Review & Comparison</div>', unsafe_allow_html=True)
        comparison_path = ROOT / "Consolidated_Paper_Comparison.xlsx"
        if comparison_path.exists():
            try:
                df_comp = pd.read_excel(comparison_path)
                st.dataframe(df_comp, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading comparison matrix: {e}")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 8 – Export & Section 6 Structured JSON
    # ════════════════════════════════════════════════════════════════════════
    with tabs[7]:
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
    <div style="text-align:center;padding:2.5rem 0 1.5rem;color:#64748B;font-size:0.82rem;line-height:1.6;border-top:1px solid rgba(255,255,255,0.06);margin-top:3.5rem;">
        <div style="font-weight:600;color:#94A3B8;">IRNLP Intelligent Concept Discovery Engine · Enterprise Research Edition</div>
        <div style="font-size:0.75rem;color:#64748B;margin-top:4px;">
            Marwadi University · Department of Information and Communication Technology (ICT) · Research Paper Ingestion Pipeline
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
