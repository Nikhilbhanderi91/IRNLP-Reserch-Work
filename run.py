#!/usr/bin/env python3
"""
================================================
run.py  –  Launch script for the HMRP application
================================================
Usage:
    python run.py          # launch Streamlit app
    python run.py --port 8502
    python run.py --demo   # quick CLI demo without UI
================================================
"""

import sys
import argparse
from pathlib import Path


def launch_streamlit(port: int = 8501) -> None:
    """Start the Streamlit web application."""
    import subprocess

    app_path = Path(__file__).parent / "app" / "main.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(port),
        "--server.headless", "false",
        "--theme.base", "dark",
        "--theme.primaryColor", "#6C63FF",
        "--theme.backgroundColor", "#0F0F1A",
        "--theme.secondaryBackgroundColor", "#1A1A2E",
        "--theme.textColor", "#E0E0E0",
    ]
    print(f"\n🚀 Launching HMRP on http://localhost:{port}\n")
    subprocess.run(cmd)


def demo_cli(pdf_path: str) -> None:
    """
    Quick CLI demo: process a single PDF and print a summary.
    """
    from app.pipeline import AnalysisPipeline

    pipeline = AnalysisPipeline()
    print(f"\n📄 Processing: {pdf_path}")
    paper = pipeline.process_single(pdf_path)

    print(f"\n{'='*60}")
    print(f"Title   : {paper.title}")
    print(f"Authors : {', '.join(paper.authors[:5])}")
    print(f"Pages   : {paper.page_count}")
    print(f"Words   : {paper.word_count:,}")
    print(f"\nAbstract (first 300 chars):")
    print(f"  {paper.abstract[:300]}…")
    print(f"\nKeywords (top 15):")
    print(f"  {', '.join(paper.all_keywords[:15])}")
    print(f"\nResearch Domains:")
    print(f"  {', '.join(paper.research_domains)}")
    print(f"\nTechnical Terms (top 15):")
    print(f"  {', '.join(paper.technical_terms[:15])}")
    print(f"\nHierarchy root nodes:")
    for k in list(paper.concept_hierarchy.keys())[:5]:
        print(f"  • {k}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HMRP – Hierarchical Concept Mapping for Research Paper Similarity Analysis"
    )
    parser.add_argument(
        "--port", type=int, default=8501,
        help="Streamlit server port (default: 8501)"
    )
    parser.add_argument(
        "--demo", metavar="PDF_PATH",
        help="Run a quick CLI demo on a single PDF without the UI"
    )
    args = parser.parse_args()

    if args.demo:
        demo_cli(args.demo)
    else:
        launch_streamlit(port=args.port)
