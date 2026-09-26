# 🔬 HMRP — Hierarchical Concept Mapping for Research Paper Similarity Analysis

A production-ready Python application that automatically reads research papers (PDFs), extracts concepts, builds hierarchical concept maps, computes paper similarity, and visualizes everything through an interactive Streamlit dashboard.

---

## ✨ Features

| Feature | Details |
|---|---|
| **PDF Extraction** | Title · Authors · Abstract · Keywords · Introduction · Full Text |
| **Text Processing** | Cleaning · Tokenization · Lemmatization · Stopword Removal |
| **Keyword Extraction** | TF-IDF · KeyBERT · YAKE with confidence scores |
| **Concept Extraction** | spaCy NER · Technical Terms · Noun Chunks |
| **Hierarchy Building** | Auto parent-child tree from domain taxonomy |
| **Similarity Analysis** | TF-IDF Cosine · Sentence Transformers |
| **Visualizations** | Sunburst · Treemap · Icicle · Mind Map · Knowledge Graph · Heatmap |
| **Search** | Keyword / concept / text search with fuzzy matching |
| **Export** | CSV · Excel · JSON · PNG · SVG · Interactive HTML |

---

## 📁 Project Structure

```
HMRP/
├── app/
│   ├── config.py           # Global configuration
│   ├── main.py             # Streamlit UI entry point
│   └── pipeline.py         # End-to-end analysis orchestrator
├── extraction/
│   ├── pdf_extractor.py    # PyMuPDF + pdfplumber
│   ├── text_processor.py   # NLTK cleaning & tokenization
│   ├── keyword_extractor.py # TF-IDF + KeyBERT + YAKE
│   └── concept_extractor.py # spaCy NER + technical terms
├── hierarchy/
│   ├── concept_hierarchy.py # Tree construction
│   └── hierarchy_builder.py # Domain detection + orchestration
├── similarity/
│   └── similarity_engine.py # TF-IDF cosine + Sentence Transformers
├── visualization/
│   ├── knowledge_graph.py   # NetworkX + PyVis + Plotly
│   ├── hierarchy_viz.py     # Sunburst, Treemap, Icicle
│   ├── heatmap_viz.py       # Similarity heatmap
│   └── mindmap_viz.py       # Radial mind map
├── search/
│   └── search_engine.py     # Fuzzy + exact search
├── export/
│   └── exporter.py          # Multi-format export
├── models/
│   └── paper.py             # Paper dataclass
├── utils/
│   ├── logger.py            # Centralized logging
│   └── helpers.py           # Shared utilities
├── data/
│   ├── uploads/             # PDF input files
│   ├── processed/           # Processed JSON outputs
│   └── exports/             # Exported results
├── requirements.txt
├── setup.py
└── run.py                   # Launch script
```

---

## 🚀 Quick Start

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
# OR
venv\Scripts\activate          # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 4. Run the application

```bash
python run.py
# OR directly:
streamlit run app/main.py
```

Open your browser at **http://localhost:8501**

---

## 🖥️ Usage

1. **Upload PDFs** via the sidebar (one or multiple)
2. Click **🚀 Analyse Papers**
3. Explore the tabs:
   - **Papers** — Metadata, abstract, authors, entities
   - **Keywords** — TF-IDF / KeyBERT / YAKE results
   - **Hierarchy** — Sunburst, Treemap, Icicle, Path list
   - **Knowledge Graph** — Interactive network (Plotly + PyVis)
   - **Mind Map** — Radial concept visualization
   - **Similarity** — Heatmap + top similar pairs
   - **Search** — Keyword/concept search with highlighting
   - **Export** — Download CSV, Excel, JSON, HTML

---

## ⚙️ CLI Demo Mode

Run a quick pipeline test on a single PDF without the Streamlit UI:

```bash
python run.py --demo /path/to/paper.pdf
```

---

## 🔧 Configuration

Edit `app/config.py` to customize:

- **spaCy model** — switch to `en_core_sci_sm` for scientific text
- **Sentence model** — change embedding model name
- **Domain taxonomy** — add/edit research domain seed keywords
- **Top-N settings** — control keyword/concept counts
- **Color palette** — customize visualization colors

---

## 📦 Technology Stack

| Layer | Libraries |
|---|---|
| **Web UI** | Streamlit |
| **PDF** | PyMuPDF (fitz), pdfplumber |
| **NLP** | spaCy, NLTK |
| **Keyword Extraction** | scikit-learn (TF-IDF), KeyBERT, YAKE |
| **Similarity** | scikit-learn, sentence-transformers |
| **Visualization** | Plotly, NetworkX, PyVis |
| **Data** | NumPy, Pandas |
| **Export** | openpyxl, kaleido |

---

## 📝 License

MIT License — free for academic and commercial use.

---

## 🎓 Academic Use

This project is designed as a final-year engineering project / research tool. It supports:

- Literature review automation
- Research gap identification via concept maps
- Paper clustering and grouping
- Citation graph-style keyword analysis
