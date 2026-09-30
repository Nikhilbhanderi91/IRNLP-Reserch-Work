# Phase 4 — Proposed Methodology

**Research Area:** Acronym Identification (AI), Disambiguation (AD), and CCC-Based Concept Mapping
**Builds on:** Phase 1 (Problem), Phase 2 (Literature Review), Phase 3 (Research Gap Analysis)

---

## 4.1 — Input Specification

**Input:** A research paper, accepted as either:
- `paper.pdf` (raw PDF file), or
- Raw research paper text (if PDF is already converted)

**Example:** `paper.pdf`

The system must handle the multi-column, figure/table-heavy layout typical of scientific PDFs (as noted implicitly by all six reviewed papers, which all source from arXiv-style scientific documents).

---

## 4.2 — Output Specification

The system produces a layered output, extending the flat (short form → long form) output of Papers 1–6 into a structured concept representation:

- ✓ Abbreviation list
- ✓ Full forms (disambiguated long forms)
- ✓ Categories (concept classification)
- ✓ CCC mapping (Concept–Category–Context)
- ✓ Hierarchical tree
- ✓ Knowledge graph
- ✓ Interactive visualization

**Example output chain:**

```
CNN
 ↓
Convolutional Neural Network
 ↓
Deep Learning
 ↓
Machine Learning
 ↓
Artificial Intelligence
```

This is the concrete realization of the novelty defined in Phase 3: existing systems (Papers 1, 2, 4, 5, 6) stop at "CNN → Convolutional Neural Network." The proposed system continues the chain upward into a category hierarchy.

---

## 4.3 — System Modules

The system is divided into 10 independent modules so each can be developed, tested, and evaluated separately, and so that modules 1–6 can reuse/benchmark against existing techniques from Papers 1–6 while modules 7–10 implement the novel contribution.

| # | Module | Grounded in (prior work) |
|---|---|---|
| 1 | PDF Processing | Common preprocessing step across Papers 1–6 |
| 2 | Text Extraction | Common preprocessing step across Papers 1–6 |
| 3 | Preprocessing (cleaning, sentence/token segmentation) | Common preprocessing step across Papers 1–6 |
| 4 | Abbreviation Detection | Papers 1 (LSTM-CRF), 2 (AT-BERT), 6 (XLNet ensemble) |
| 5 | Long Form Extraction | Papers 1, 3 (MadDog rule-based + DOG glossary) |
| 6 | Disambiguation | Papers 1 (GAD), 3 (MadDog), 4 (multi-strategy BERT), 5 (hdBERT), 6 (Siamese retrieval) |
| 7 | Concept Extraction | **New** — not covered in reviewed literature |
| 8 | CCC Mapping | **New** — not covered in reviewed literature |
| 9 | Hierarchy Generation | **New** — not covered in reviewed literature |
| 10 | Knowledge Graph + Visualization | **New** — not covered in reviewed literature |

---

## 4.4 — Data Flow

```
PDF
 ↓
Raw text
 ↓
Clean text
 ↓
Sentences
 ↓
Tokens
 ↓
Abbreviations
 ↓
Full forms
 ↓
Concepts
 ↓
Hierarchy
 ↓
Knowledge graph
 ↓
Visualization
```

Each arrow represents a module boundary (Section 4.3) — data is transformed incrementally and each intermediate output is independently inspectable/testable, which supports the modular evaluation plan in Section 4.8.

---

## 4.5 — Module Design

### Module 1 — PDF Processing
- **Input:** `paper.pdf`
- **Output:** Raw text
- **Notes:** Handle multi-column layout, figures/tables/references stripped or flagged separately.

### Module 2 — Text Extraction
- **Input:** Raw text (from Module 1)
- **Output:** Structured raw text (title, abstract, sections)

### Module 3 — Text Cleaning / Preprocessing
- **Input:** Raw text
- **Output:** Clean text (normalized whitespace, de-hyphenation, sentence and token segmentation)

### Module 4 — Abbreviation Detection
- **Input:** Clean text
- **Output:** Candidate short forms, e.g. `CNN`, `RNN`, `GAN`, `LLM`
- **Approach:** Sequence-labeling model (BIO tagging, following Paper 1's LSTM-CRF and Paper 2's BERT-ensemble approach) or rule-based candidate detection (capital-letter/pattern heuristics as in Papers 1 and 3), depending on resource constraints.

### Module 5 — Full Form / Long Form Detection
- **Input:** `CNN`
- **Output:** `Convolutional Neural Network`
- **Approach:** Dictionary lookup against a glossary (à la MadDog's DOG) plus rule-based/statistical extraction of nearby candidate definitions in-text.

### Module 6 — Context Extraction & Disambiguation
- **Input:** Sentence context, e.g. *"CNN is widely used for image classification."*
- **Output:** Surrounding sentence context; disambiguated correct long form
- **Approach:** Context-based classification using a domain-fused transformer approach (following hdBERT's dual-path general + domain-specific fusion, or Pan et al.'s multi-strategy binary classifier), selected as the disambiguation backbone since these two showed the strongest reported accuracy (93.73–94.05 F1) among reviewed methods.

### Module 7 — Concept Extraction *(new)*
- **Input:** Disambiguated long form, e.g. `CNN` → `Convolutional Neural Network`
- **Output:** Associated broader concept, e.g. `Deep Learning`
- **Approach:** Map each resolved long form to a concept using a domain taxonomy/ontology lookup (e.g., a curated AI/CS concept taxonomy) combined with embedding-similarity clustering of long forms that share semantic neighborhoods.

### Module 8 — CCC Mapping *(new)*
- **Input:** Concepts from Module 7
- **Output:** A Concept–Category–Context chain
- **Example:**
```
Artificial Intelligence
 ↓
Machine Learning
 ↓
Deep Learning
 ↓
CNN
```
- **Approach:** Assign each concept to a category (from a predefined or learned category taxonomy) and retain the originating sentence/document context as provenance for each mapping.

### Module 9 — Relationship Extraction + Hierarchy Generation *(new)*
- **Input:** CCC-mapped concepts
- **Output:** Directed relationships between concepts and a hierarchical tree
- **Example relationship:**
```
CNN
 --used for-->
Image Classification
 --applied in-->
Medical Imaging
 --applied in-->
Cancer Detection
```
- **Example hierarchy:**
```
Artificial Intelligence
│
├── Machine Learning
│
├── Deep Learning
│      ├── CNN
│      ├── RNN
│      ├── GAN
│      └── Transformer
```
- **Approach:** Use dependency-parse or LLM-prompted relation extraction to label edges (is-a, used-for, applied-in), then assemble a tree/DAG.

### Module 10 — Knowledge Graph + Visualization *(new)*
- **Input:** Hierarchy + relationships
- **Output:** Knowledge graph (nodes = concepts/acronyms, edges = relationships) and an interactive visualization (tree view, mind map, or graph view)

---

## 4.6 — Complete System Architecture

```
                 Research Paper (PDF)
                         │
                         ▼
               PDF Processing Module
                         │
                         ▼
                Text Extraction Module
                         │
                         ▼
               Text Cleaning Module
                         │
                         ▼
              Sentence Segmentation
                         │
                         ▼
                Tokenization & NLP
                         │
                         ▼
             Abbreviation Detection
                         │
                         ▼
              Long Form Extraction
                         │
                         ▼
             Acronym Disambiguation
                         │
                         ▼
               Concept Identification
                         │
                         ▼
                 CCC Mapping Engine
                         │
                         ▼
              Hierarchy Construction
                         │
                         ▼
             Knowledge Graph Builder
                         │
                         ▼
             Interactive Visualization
```

*(The first six stages consolidate proven techniques from Papers 1–6; the final five stages — Concept Identification through Interactive Visualization — implement the novel contribution identified in Phase 3.)*

---

## 4.7 — Technology Stack

| Module | Suggested Technology | Rationale (tied to literature) |
|---|---|---|
| PDF Processing | PyMuPDF, pdfplumber | Standard scientific PDF parsing |
| NLP (cleaning, tokenization, sentence segmentation) | spaCy, SciSpaCy | SciSpaCy is domain-tuned for scientific text, aligning with the scientific-domain focus of Papers 1–6 |
| Transformer models (backbone) | SciBERT, BERT | SciBERT was the strongest single backbone across Papers 4 and 5 |
| Abbreviation detection | SciAI-based sequence-labeling model or rule-based methods | Directly follows Paper 1 (LSTM-CRF, SciAI dataset) and Paper 3 (rule-based detector) |
| Disambiguation | SciBERT / hdBERT-style dual-path model | Follows Paper 5's dual-path fusion (93.73 F1) and Paper 4's multi-strategy training (94.05 F1), the two strongest reported AD approaches |
| Concept/relationship & graph processing | NetworkX | Lightweight graph construction and analysis |
| Knowledge graph storage (optional, for scale) | Neo4j | For persistent, queryable graph storage if the corpus grows beyond single-paper scope |
| Visualization | PyVis, D3.js, Cytoscape | Interactive tree/graph rendering — the layer entirely absent from Papers 1–6 |

---

## 4.8 — Evaluation Plan

Two evaluation tiers, matching the two halves of the pipeline (Section 4.6):

**Tier 1 — Reused/benchmarked stages (Modules 1–6), evaluated against the literature (Phase 2):**
- Does the system correctly detect abbreviations? (Precision/Recall/Macro-F1 on SciAI, compared to Papers 1, 2, 6)
- Does it extract the correct full form? (Accuracy against SciAI/SciAD gold long forms)
- Does it resolve ambiguous abbreviations correctly? (Macro-F1 on SciAD / SciAD-dedupe, compared to Papers 1, 3, 4, 5, 6, with human performance ~96% F1 as the ceiling reference)

**Tier 2 — Novel stages (Modules 7–10), evaluated with new criteria not present in prior work:**
- Does the system generate a meaningful concept hierarchy? (Manual/expert judgment of hierarchy correctness and depth; agreement rate against a small human-curated gold hierarchy for a sample of papers)
- Is the CCC mapping semantically accurate? (Precision of concept-to-category assignment, spot-checked against domain taxonomies, e.g. ACM CCS or a curated AI/CS taxonomy)
- Are extracted relationships correct and useful? (Sampled manual relationship-correctness review, similar in spirit to the manual error analysis performed in Paper 6)
- Is the visualization understandable to users? (Small user study — task completion time and comprehension score for finding a concept's relationships, comparing the CCC/knowledge-graph view against a flat acronym list baseline representing the Papers 1–6 status quo)

This two-tier plan ensures the system is validated both against established, comparable benchmarks (grounding it in the literature) and against the new capabilities that define its contribution (validating the research gap addressed in Phase 3).

---

## Deliverables Checklist

- ✅ Complete system architecture (Step 4.6)
- ✅ Module descriptions (Steps 4.3, 4.5)
- ✅ Data flow diagram (Step 4.4)
- ✅ Input and output specifications (Steps 4.1, 4.2)
- ✅ Technology stack (Step 4.7)
- ✅ High-level implementation plan (Modules 1–10, sequenced)
- ✅ Evaluation plan (Step 4.8)
