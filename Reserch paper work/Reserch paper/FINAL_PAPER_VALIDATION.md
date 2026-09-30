# FINAL_PAPER_VALIDATION.md
## Validation Report — HMRP Final Research Paper

This document records the checks performed before delivering `FINAL_RESEARCH_PAPER.pdf`, `FINAL_RESEARCH_PAPER.docx`, and the `/figures/` directory, derived from `DRAFT_RESEARCH_PAPER.pdf`.

---

### 1. Format Compliance

| Check | Status | Notes |
|---|---|---|
| IEEE two-column conference format (`IEEEtran.cls`) | ✅ Pass | Compiled with `pdflatex` using the official IEEEtran class (`texlive-publishers` package) |
| Page count within 12–15 pages | ✅ Pass | Final PDF is **12 pages** |
| Title, authors/affiliation block | ✅ Pass | Author names/affiliation are placeholders (see Section 6 — *Known Limitations of This Deliverable*), clearly marked as such rather than fabricated |
| Abstract | ✅ Pass | Condensed from the draft's abstract, unchanged in substance |
| Keywords / Index Terms | ✅ Pass | Carried over from the draft |
| All 19 requested sections present | ✅ Pass | Title, Authors, Abstract, Keywords, Introduction, Related Work, Research Gap, Objectives, Methodology, System Architecture, Dataset, Experimental Setup, Results & Evaluation, Comparative Analysis, Discussion, Limitations, Future Work, Conclusion, References |

### 2. Figures

| # | Figure | Source | Referenced in text | Numbered caption |
|---|---|---|---|---|
| 1 | System architecture (8-layer stack) | Generated with Graphviz from Section 10 of the draft | ✅ (System Architecture) | ✅ Fig. 1 |
| 2 | End-to-end nine-step inference pipeline | Generated with Graphviz from Section 14.1 of the draft | ✅ (System Architecture) | ✅ Fig. 2 |
| 3 | Dataset construction / preprocessing pipeline | Generated with Graphviz from Section 12 of the draft | ✅ (Dataset) | ✅ Fig. 3 |
| 4 | Ten-module design with implementation-status color coding | Generated with Graphviz from Section 9.2 of the draft | ✅ (Methodology) | ✅ Fig. 4 |
| 5 | Baseline vs. enhanced pipeline bar chart | Generated with Matplotlib from Table in Section 16.1 of the draft | ✅ (Results) | ✅ Fig. 5 |
| 6 | Pairwise semantic similarity statistics | Generated with Matplotlib from Section 16.3 of the draft | ✅ (Results) | ✅ Fig. 6 |

All figures were generated programmatically from data and structure already present in the draft. No figure contains fabricated results, invented data points, or decorative-only content. Each figure is legible at print resolution (300 DPI raster charts; vector PDF diagrams) and is cited by number from the body text.

### 3. Tables

Eight numbered tables were created, all populated directly from figures/data stated in the draft (no invented numbers):
1. Core Literature Review (six papers)
2. Proposed Ten-Module Design and Implementation Status
3. Mapping of Proposed Modules to Implemented Layers
4. Models and Algorithms in the Implemented System
5. HMRP Corpus Statistics
6. Baseline vs. Enhanced Pipeline Comparison
7. Automated Test Suite Results
8. Pipeline-Stage Coverage: Literature vs. Design vs. HMRP

### 4. Content Integrity Checks

| Check | Status | Notes |
|---|---|---|
| No fabricated datasets, experiments, or metrics | ✅ Pass | Every figure/statistic traces to a specific section of the draft |
| No fabricated or altered references | ✅ Pass | All 27 references reproduced from the draft's reference list; no reference added, removed, or reworded beyond citation-style formatting |
| Implemented vs. future work clearly distinguished | ✅ Pass | Explicit "Implemented / Partial / Not Implemented" labeling maintained across Methodology, System Architecture, Comparative Analysis, Limitations, and Future Work sections |
| Novelty not overstated | ✅ Pass | Discussion and Conclusion explicitly state that CCC mapping, typed relationships, and multi-candidate disambiguation remain unimplemented, and that no SciAI/SciAD benchmarking was performed |
| "Not Available" markers preserved | ✅ Pass | Retained wherever the draft explicitly marked a figure/claim as unmeasured (e.g., hardware/training configuration, Tier-1/Tier-2 evaluation, AD accuracy figure) |

### 5. Layout / Rendering QA

Performed by rendering the compiled PDF to page images (`pdftoppm`) and visually inspecting every page:
- No orphan headings (all headings immediately followed by body text or a figure/table)
- No broken equations (only `O(n·d)`/`O(V)` complexity notation used, rendered correctly via `amsmath`)
- Two initial table-column overflow bugs (Table I, Table IV) were caught during QA and corrected before finalization
- No overlapping text/figure elements in the final render
- No excessive whitespace (only the expected end-of-references trailing space on the last page, which is normal for a references section that does not fill the final page)
- Figure/table numbering is sequential and consistent with in-text references throughout

### 6. Known Limitations of This Deliverable

- **Author names and institutional affiliation are placeholders.** The source draft did not specify individual author names; the byline reads "Research Project Team — Department of Computer Science and Engineering" with an explicit placeholder email. The submitting team should replace this before submission.
- **The `.docx` is a single-column, Word-native rendering** of the same content (tables, figures, and section structure preserved) rather than a two-column IEEE layout — Word does not natively replicate the IEEEtran class, so the two-column PDF should be treated as the authoritative formatted version, and the `.docx` as the editable source.
- **No new empirical claims were introduced.** Where the draft explicitly flagged a result as "Not Available" or a module as unimplemented, this paper preserves that status rather than filling the gap.

### 7. File Manifest

```
FINAL_RESEARCH_PAPER.pdf     — IEEE two-column formatted paper (12 pages)
FINAL_RESEARCH_PAPER.docx    — Editable Word version (same content/figures/tables)
FINAL_RESEARCH_PAPER.tex     — LaTeX source for the PDF (IEEEtran class)
figures/
  fig1_architecture.pdf/.png
  fig2_pipeline.pdf/.png
  fig3_dataset.pdf/.png
  fig4_modules.pdf/.png
  fig5_baseline_comparison.pdf/.png
  fig6_similarity_stats.pdf/.png
FINAL_PAPER_VALIDATION.md    — this report
```

**Overall status: ✅ Ready for supervisor/reviewer evaluation**, subject to the author/affiliation placeholder being filled in by the submitting team.
