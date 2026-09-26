"""
================================================
export/exporter.py
Exports analysis results to multiple formats:
  CSV, Excel, JSON, PNG (chart), HTML (graph)
================================================
"""

import json
import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.logger import get_logger
from utils.helpers import ensure_dir

log = get_logger(__name__)

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class Exporter:
    """
    Handles all export operations for the HMRP project.
    Writes files to data/exports/ and returns file paths.
    """

    def __init__(self, export_dir: str | Path = "data/exports") -> None:
        self.export_dir = ensure_dir(export_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # Paper summary exports
    # ──────────────────────────────────────────────────────────────────────────

    def export_papers_csv(self, papers_data: List[Dict], filename: str = "papers_summary.csv") -> str:
        """Export paper summaries to CSV."""
        rows = []
        for p in papers_data:
            rows.append({
                "Title":          p.get("title", ""),
                "Authors":        ", ".join(p.get("authors", [])),
                "Abstract":       p.get("abstract", "")[:500],
                "Keywords":       ", ".join(p.get("keywords", [])[:15]),
                "Domains":        ", ".join(p.get("research_domains", [])),
                "Technical Terms":"; ".join(p.get("technical_terms", [])[:15]),
                "Word Count":     p.get("word_count", ""),
                "Page Count":     p.get("page_count", ""),
            })

        df = pd.DataFrame(rows)
        path = self.export_dir / filename
        df.to_csv(path, index=False, encoding="utf-8-sig")
        log.info(f"CSV exported: {path}")
        return str(path)

    def export_papers_excel(
        self, papers_data: List[Dict], filename: str = "papers_analysis.xlsx"
    ) -> str:
        """Export paper summaries to multi-sheet Excel."""
        path = self.export_dir / filename
        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                # Sheet 1: Summary
                summary_rows = []
                for p in papers_data:
                    summary_rows.append({
                        "Title":    p.get("title", ""),
                        "Authors":  ", ".join(p.get("authors", [])),
                        "Domains":  ", ".join(p.get("research_domains", [])),
                        "Pages":    p.get("page_count", ""),
                        "Words":    p.get("word_count", ""),
                    })
                pd.DataFrame(summary_rows).to_excel(
                    writer, sheet_name="Summary", index=False
                )

                # Sheet 2: Keywords
                kw_rows = []
                for p in papers_data:
                    for kw in p.get("keywords", [])[:20]:
                        kw_rows.append({"Paper": p.get("title", "")[:60], "Keyword": kw})
                if kw_rows:
                    pd.DataFrame(kw_rows).to_excel(
                        writer, sheet_name="Keywords", index=False
                    )

                # Sheet 3: Technical Terms
                tt_rows = []
                for p in papers_data:
                    for tt in p.get("technical_terms", [])[:20]:
                        tt_rows.append({"Paper": p.get("title", "")[:60], "Term": tt})
                if tt_rows:
                    pd.DataFrame(tt_rows).to_excel(
                        writer, sheet_name="Technical Terms", index=False
                    )

            log.info(f"Excel exported: {path}")
        except Exception as e:
            log.error(f"Excel export failed: {e}")
        return str(path)

    def export_json(
        self, data: Any, filename: str = "analysis.json"
    ) -> str:
        """Export arbitrary data as JSON."""
        path = self.export_dir / filename
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        log.info(f"JSON exported: {path}")
        return str(path)

    # ──────────────────────────────────────────────────────────────────────────
    # Similarity matrix export
    # ──────────────────────────────────────────────────────────────────────────

    def export_similarity_csv(
        self, matrix, labels: List[str], filename: str = "similarity_matrix.csv"
    ) -> str:
        """Export similarity matrix as CSV."""
        import numpy as np
        path = self.export_dir / filename
        df = pd.DataFrame(matrix, index=labels, columns=labels)
        df.to_csv(path, float_format="%.4f")
        log.info(f"Similarity CSV exported: {path}")
        return str(path)

    # ──────────────────────────────────────────────────────────────────────────
    # Figure exports
    # ──────────────────────────────────────────────────────────────────────────

    def export_figure_png(
        self, fig: "go.Figure", filename: str = "figure.png"
    ) -> Optional[str]:
        """Export a Plotly figure to PNG."""
        if not HAS_PLOTLY:
            return None
        path = self.export_dir / filename
        try:
            fig.write_image(str(path), scale=2)
            log.info(f"PNG exported: {path}")
            return str(path)
        except Exception as e:
            log.error(f"PNG export failed: {e}")
            return None

    def export_figure_svg(
        self, fig: "go.Figure", filename: str = "figure.svg"
    ) -> Optional[str]:
        """Export a Plotly figure to SVG."""
        if not HAS_PLOTLY:
            return None
        path = self.export_dir / filename
        try:
            fig.write_image(str(path))
            log.info(f"SVG exported: {path}")
            return str(path)
        except Exception as e:
            log.error(f"SVG export failed: {e}")
            return None

    def export_figure_html(
        self, fig: "go.Figure", filename: str = "figure.html"
    ) -> Optional[str]:
        """Export a Plotly figure to standalone interactive HTML."""
        if not HAS_PLOTLY:
            return None
        path = self.export_dir / filename
        try:
            fig.write_html(str(path), include_plotlyjs="cdn")
            log.info(f"HTML exported: {path}")
            return str(path)
        except Exception as e:
            log.error(f"HTML export failed: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # In-memory bytes (for Streamlit download buttons)
    # ──────────────────────────────────────────────────────────────────────────

    def to_csv_bytes(self, papers_data: List[Dict]) -> bytes:
        """Return CSV as bytes for Streamlit download."""
        rows = []
        for p in papers_data:
            rows.append({
                "Title":    p.get("title", ""),
                "Authors":  ", ".join(p.get("authors", [])),
                "Keywords": ", ".join(p.get("keywords", [])[:15]),
                "Domains":  ", ".join(p.get("research_domains", [])),
            })
        df = pd.DataFrame(rows)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")

    def to_json_bytes(self, data: Any) -> bytes:
        """Return JSON as bytes for Streamlit download."""
        return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")

    def to_excel_bytes(self, papers_data: List[Dict]) -> bytes:
        """Return Excel as bytes for Streamlit download."""
        buf = io.BytesIO()
        rows = [
            {
                "Title":   p.get("title", ""),
                "Authors": ", ".join(p.get("authors", [])),
                "Domains": ", ".join(p.get("research_domains", [])),
                "Keywords":"; ".join(p.get("keywords", [])[:15]),
            }
            for p in papers_data
        ]
        df = pd.DataFrame(rows)
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Papers")
        return buf.getvalue()
