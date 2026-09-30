# Phase 3 — Research Gap Analysis

**Research Area:** Acronym Identification (AI) and Acronym Disambiguation (AD) in Scientific Text
**Builds on:** Phase 1 (Problem Understanding), Phase 2 (Literature Review)

---

## Step 3.1–3.2 — Paper-wise Limitations ("What still remains unsolved?")

| Paper | Main Contribution | What It Solves | What Remains Unsolved (Limitation) |
|---|---|---|---|
| Paper 1 — Veyseh et al. 2020 (SciAI/SciAD, GAD) | Large annotated AI/AD datasets + dependency-tree GAD model | Detects short/long form spans; disambiguates using syntactic context | No concept hierarchy; treats each acronym independently — no cross-acronym or cross-sentence relationship modeling |
| Paper 2 — Zhu et al. 2021 (AT-BERT) | Adversarially-trained, ensembled BERT for identification | Improves raw identification accuracy (94.12 F1) via robustness training | No semantic mapping between acronyms; purely a tagging system — output is a flat list of spans, not organized knowledge |
| Paper 3 — Veyseh et al. 2021 (MadDog) | First public, multi-domain AI+AD web system | Detects and expands acronyms across domains at scale (DOG + MAD) | No visualization of results beyond direct lookup; no way to see how concepts in a document relate to one another |
| Paper 4 — Pan et al. 2021 (BERT-AD Multi-Strategy) | Combined training strategies (TAPT, adversarial, pseudo-labeling) for AD | Pushes disambiguation accuracy close to human level (0.9405 F1) | No relationship extraction between disambiguated concepts; each acronym-candidate pair is judged in isolation |
| Paper 5 — Zhong et al. 2021 (hdBERT) | Dual-path general + domain-specific transformer fusion | Improves AD by combining complementary pretrained knowledge (93.73 F1) | No knowledge graph or structured output; result is still a single predicted label per acronym, not a connected concept map |
| Paper 6 — Egan & Bohannon 2021 (Primer AI) | XLNet ensemble (AI) + Siamese retrieval-based AD + SciAD-dedupe | Improves both tasks and fixes benchmark data-quality issues | No concept organization; retrieval returns a single nearest-neighbor label, with no attempt to structure or group related acronym meanings |

**Common thread:** every paper improves *accuracy* on a flat, sentence-level task (detect a span, or pick one label from candidates). None of them ask what happens *after* an acronym is correctly resolved — i.e., how the now-understood concept relates to other concepts in the same document or across documents.

---

## Step 3.3 — Common Missing Features (Capability Checklist)

What the reviewed literature **does** cover:

- ✓ Acronym Detection (span-level identification)
- ✓ Acronym Expansion (long-form extraction)
- ✓ Acronym Disambiguation (context-based meaning resolution)
- ✓ Robustness improvements (adversarial training, ensembling, domain fusion)
- ✓ Benchmark data-quality auditing (deduplication)

What is **consistently missing** across all six papers:

- ✗ Concept Mapping (organizing resolved acronyms as concepts, not just labels)
- ✗ CCC (Concept–Category–Context) Mapping / structured grouping of related concepts
- ✗ Hierarchical Knowledge Representation (parent/child or broader/narrower concept relations)
- ✗ Semantic Relationship Extraction between acronyms/concepts (e.g., "is-a," "used-in," "related-to")
- ✗ Interactive Visualization of a document's concept structure
- ✗ Cross-Paper / Cross-Document Concept Linking (connecting the same or related acronyms across a corpus)

This checklist is the direct output of Phase 2's "common limitations" analysis and is the raw material for the research gap statement below.

---

## Step 3.4 — Research Gap Statement

> Existing studies on scientific acronym understanding focus almost entirely on two flat, sentence-level sub-tasks: **identifying** acronym/long-form spans and **disambiguating** an acronym's correct meaning from a set of candidates. State-of-the-art methods — ranging from syntactic graph models (GAD) to adversarially-trained transformer ensembles (AT-BERT), rule-based multi-domain systems (MadDog), multi-strategy binary classifiers (Pan et al.), dual-path domain-fused transformers (hdBERT), and retrieval-based disambiguation (Primer AI) — have pushed accuracy close to human level (~91–94% F1 vs. ~96% human). However, **none of these systems organize resolved acronyms into higher-level concepts, establish semantic relationships between them, build a hierarchical or graph-based knowledge structure, or provide an interactive way for a reader to explore how the concepts in a paper relate to one another.** Once an acronym is expanded and disambiguated, the existing pipelines simply stop — the output is a flat label, not structured, explorable knowledge. This leaves a clear gap between *accurate acronym resolution* and *usable, structured document/corpus-level understanding*.

This becomes the Research Gap section of the final paper.

---

## Step 3.5 — Novelty Statement

**Core question:** *What will the proposed system do that none of the six reviewed systems do?*

**Answer:** It will not stop at expansion/disambiguation. It will take the resolved acronyms and organize them into a **CCC (Concept–Category–Context) map**, build a **hierarchical concept structure**, connect related concepts into a **knowledge graph**, and present the result through an **interactive visualization** — turning flat acronym resolution into structured, explorable document understanding.

**Existing systems (pipeline):**

```
PDF
 ↓
Abbreviation Detection
 ↓
Expansion
 ↓
(Disambiguation, in some systems)
 ↓
END — flat list of (short form, long form) pairs
```

**Proposed system (pipeline):**

```
PDF
 ↓
Abbreviation Detection
 ↓
Expansion
 ↓
Disambiguation
 ↓
Concept Identification
 ↓
CCC Mapping (Concept – Category – Context)
 ↓
Hierarchy Generation
 ↓
Knowledge Graph Construction
 ↓
Interactive Visualization
```

**Novelty in one sentence:** the proposed system extends the resolved-acronym output of prior work (Papers 1–6) with a concept-organization layer — CCC mapping, hierarchy generation, knowledge-graph construction, and interactive visualization — that no reviewed paper currently provides.

---

## Step 3.6 — Research Questions (RQs)

- **RQ1:** How can abbreviations/acronyms be automatically and accurately identified in scientific research papers?
- **RQ2:** How can abbreviation expansions (long forms) be accurately extracted and disambiguated when multiple candidate meanings exist?
- **RQ3:** How can disambiguated abbreviations be organized into a CCC-based (Concept–Category–Context) hierarchical concept structure?
- **RQ4:** How can semantic relationships between concepts be extracted and represented as a knowledge graph?
- **RQ5:** Can interactive concept visualization improve a reader's understanding and navigation of scientific papers compared to flat acronym lists?

---

## Step 3.7 — Research Objectives

1. Develop an automated abbreviation/acronym extraction module for scientific PDF documents.
2. Accurately extract candidate long-form expansions for detected acronyms.
3. Resolve ambiguous abbreviations using sentence/document context (disambiguation).
4. Design and implement a CCC-based (Concept–Category–Context) concept mapping framework.
5. Generate hierarchical concept structures from the mapped concepts.
6. Construct a knowledge graph representing semantic relationships between concepts.
7. Build an interactive visualization interface for exploring the generated concept structure.
8. Evaluate the system both on standard AI/AD accuracy metrics (Precision, Recall, Macro-F1, benchmarked against Papers 1–6) and on the added value of the concept-organization layer (e.g., readability/usability improvement, coverage of relationships, or user-study based evaluation).

---

## Step 3.8 — Proposed Framework (High-Level Workflow)

```
 ┌───────────────────────┐
 │     Research Paper      │
 │        (PDF)             │
 └───────────┬───────────┘
             │
             ▼
 ┌───────────────────────┐
 │    PDF Processing        │
 └───────────┬───────────┘
             │
             ▼
 ┌───────────────────────┐
 │    Text Extraction       │
 └───────────┬───────────┘
             │
             ▼
 ┌───────────────────────┐
 │ Abbreviation Identification│
 └───────────┬───────────┘
             │
             ▼
 ┌───────────────────────┐
 │  Long Form Extraction    │
 └───────────┬───────────┘
             │
             ▼
 ┌───────────────────────┐
 │     Disambiguation       │
 └───────────┬───────────┘
             │
             ▼
 ┌───────────────────────┐
 │    CCC Mapping            │
 │ (Concept–Category–Context)│
 └───────────┬───────────┘
             │
             ▼
 ┌───────────────────────┐
 │   Hierarchy Generation   │
 └───────────┬───────────┘
             │
             ▼
 ┌───────────────────────┐
 │    Knowledge Graph       │
 └───────────┬───────────┘
             │
             ▼
 ┌───────────────────────┐
 │ Interactive Visualization│
 └───────────────────────┘
```

**Stage-by-stage mapping to prior work:**

| Stage | Covered by prior literature? | Reference |
|---|---|---|
| PDF Processing / Text Extraction | Implicit preprocessing step in all six papers | Papers 1–6 |
| Abbreviation Identification | Yes — core focus | Papers 1, 2, 6 |
| Long Form Extraction | Yes — core focus | Papers 1, 3 |
| Disambiguation | Yes — core focus | Papers 1, 3, 4, 5, 6 |
| Concept Identification | **Not covered** | — |
| CCC Mapping | **Not covered** | — |
| Hierarchy Generation | **Not covered** | — |
| Knowledge Graph | **Not covered** | — |
| Interactive Visualization | **Not covered** (MadDog offers lookup only, not concept visualization) | — |

This table makes explicit that the first half of the proposed pipeline consolidates established, well-benchmarked techniques from Papers 1–6, while the second half (from Concept Identification onward) is the genuinely novel contribution addressing the research gap identified in Step 3.4.

---

## Deliverables Checklist

- ✅ Paper-wise limitations (Step 3.1–3.2)
- ✅ Common limitations (Step 3.3)
- ✅ Research gap statement (Step 3.4)
- ✅ Novelty statement (Step 3.5)
- ✅ Research questions — RQ1–RQ5 (Step 3.6)
- ✅ Research objectives (Step 3.7)
- ✅ Proposed framework diagram (Step 3.8)
