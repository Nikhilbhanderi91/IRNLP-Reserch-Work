# Phase 2 — Literature Review

**Research Area:** Acronym Identification (AI) and Acronym Disambiguation (AD) in Scientific Text

*Note: This phase is built primarily from the six papers already summarized (Veyseh et al. 2020; Zhu et al. 2021; Veyseh et al. 2021; Pan et al. 2021; Zhong et al. 2021; Egan & Bohannon 2021). Steps 2.1 and 2.2 below show how these six fit the required extraction format, and steps 2.3–2.7 build the comparative and analytical deliverables. If you add papers from the other focus-keyword areas (Knowledge Graph, Concept Mapping, Semantic Relationship Extraction, Hierarchical Concept Mapping), append them to each table using the same structure.*

---

## Step 2.1 — Collected Papers (Base Set)

| # | Source |
|---|---|
| 1 | COLING 2020 / arXiv:2010.14678 |
| 2 | AAAI-21 SDU Workshop / arXiv:2101.03700 / CEUR-WS Vol-2831 |
| 3 | EACL 2021 (System Demonstrations) / arXiv:2101.09893 |
| 4 | AAAI-21 SDU Workshop / arXiv:2103.00488 / CEUR-WS Vol-2831 |
| 5 | AAAI-21 SDU Workshop / arXiv:2107.00316 / CEUR-WS Vol-2831 |
| 6 | AAAI-21 SDU Workshop / arXiv:2012.08013 / CEUR-WS Vol-2831 |

All six were sourced via arXiv, with parallel publication through ACL Anthology (COLING/EACL) or CEUR-WS/AAAI proceedings — consistent with the required search across arXiv, ACM/IEEE-adjacent workshop venues, and Google Scholar. To extend this set, targeted searches on Springer, IEEE Xplore, ACM Digital Library, and Google Scholar using the focus keywords (Acronym Identification, Acronym Disambiguation, Abbreviation Extraction, Scientific Document Understanding, Scientific NLP, Knowledge Graph, Concept Mapping, Semantic Relationship Extraction, Hierarchical Concept Mapping) are recommended next steps to broaden coverage, particularly toward the Knowledge Representation category (Step 2.3, Category 4), which the current six papers do not cover.

---

## Step 2.2 — Paper Extraction Table

| Field | Paper 1: SciAI/SciAD (Veyseh et al., 2020) | Paper 2: AT-BERT (Zhu et al., 2021) | Paper 3: MadDog (Veyseh et al., 2021) | Paper 4: BERT-AD Multi-Strategy (Pan et al., 2021) | Paper 5: hdBERT (Zhong et al., 2021) | Paper 6: Primer AI (Egan & Bohannon, 2021) |
|---|---|---|---|---|---|---|
| **Authors** | Veyseh, Dernoncourt, Tran, Nguyen | Zhu, Lin, Zhang, Zhong, Zeng, Wu, Tang | Veyseh, Dernoncourt, Chang, Nguyen | Pan, Song, Wang, Luo | Zhong, Zeng, Zhu, Zhang, Lin, Chen, Tang | Egan, Bohannon |
| **Year** | 2020 | 2021 | 2021 | 2021 | 2021 | 2021 |
| **Publisher** | COLING (ACL Anthology) / arXiv | AAAI-21 SDU Workshop / CEUR-WS / arXiv | EACL 2021 (ACL Anthology) / arXiv | AAAI-21 SDU Workshop / CEUR-WS / arXiv | AAAI-21 SDU Workshop / CEUR-WS / arXiv | AAAI-21 SDU Workshop / CEUR-WS / arXiv |
| **Research Problem** | Existing AI/AD datasets too small, noisy, or medical-only | Small AI datasets cause overfitting in deep models | No unified, public, multi-domain AI+AD system exists | Scientific-domain AD underexplored with combined PLM training strategies | General vs. domain-specific PLMs alone are insufficient for AD | AD/AI models can be improved with more data; SciAD has label-quality issues |
| **Objective** | Build large annotated AI/AD datasets + strong AD baseline | Build robust AI system (won SDU@AAAI-21) | Build first web-based multi-domain AI+AD system | Build top binary-classification AD model (won SDU@AAAI-21) | Fuse general + domain-specific knowledge for AD | Improve AI/AD + fix SciAD data quality |
| **Dataset** | SciAI (17,506 sent.), SciAD (62,441 samples) | SciAI (14,006/1,717/1,750 split) | DOG glossary (426K acronyms), MAD (46M records); tested on SciAI/SciAD | SciAD (62,441 samples) | SciAD → SciAD_BI (352K/28K/28K) | SciAI, SciAD + new AuxAI, AuxAD, SciAD-dedupe |
| **Methodology** | LSTM-CRF (AI); BiLSTM+GCN over dependency tree = GAD (AD) | Fine-tuned BERT/SciBERT/RoBERTa/ALBERT/ELECTRA + FGM adversarial training + ensembling | Rule-based AI pipeline (extended Schwartz–Hearst) + BiLSTM disambiguation model | BERT/SciBERT binary classifier + dynamic negative sampling + TAPT + adversarial training + pseudo-labeling | Dual-path RoBERTa + SciBERT → MLP (hdBERT) | XLNet ensemble (AI); Siamese/Twin networks for retrieval-based AD |
| **Results** | LSTM-CRF: 86.55 F1 (AI); GAD: 81.90 F1 (AD) | AT-BERT-E: 94.12 F1 (AI) | 88.12 F1 (AI, SciAI); 88.49 F1 (AD, SciAD) | P 0.9695 / R 0.9132 / F1 0.9405 (AD) | 93.73 F1 (AD) | 92.60 F1 (AI, test); 91.58 F1 (AD, test) |
| **Advantages** | Large, human-annotated; captures long-range syntax | SOTA robustness; won shared task | Multi-domain; public; high recall + precision | Systematic ablation; near-human results | Fuses complementary knowledge; simple integration | Novel IR framing; fixed a major data-quality bug |
| **Limitations** | Far below human performance; annotation noise | Still below human level; ~2x training cost | High storage/compute (64GB); domain generalization untested | Struggles with near-synonymous candidates | Still below human recall; noisy conflicting labels persist | AuxAI alone weak (F1 66.96); retrieval fails without similar examples |
| **Future Work** | Advanced models to close human-performance gap | Try PGD/FreeLB adversarial methods, Dice/Focal loss | Integrate into e-readers | Not detailed beyond comparisons | Self-training, adversarial/contrastive learning | Larger corpora to reduce retrieval failures |

---

## Step 2.3 — Categorization of Papers

**Category 1: Acronym Identification (AI)**
- Paper 1 — SciAI/SciAD (Veyseh et al., 2020) — dataset + LSTM-CRF baseline
- Paper 2 — AT-BERT (Zhu et al., 2021) — BERT ensemble + adversarial training
- Paper 6 (AI component) — Primer AI (Egan & Bohannon, 2021) — XLNet ensemble + AuxAI pretraining

**Category 2: Acronym Disambiguation (AD)**
- Paper 1 (AD component) — GAD (Veyseh et al., 2020) — BiLSTM + GCN
- Paper 4 — BERT-AD Multi-Strategy (Pan et al., 2021) — binary classification + combined training tricks
- Paper 5 — hdBERT (Zhong et al., 2021) — dual-path domain fusion
- Paper 6 (AD component) — Primer AI (Egan & Bohannon, 2021) — Siamese retrieval-based AD

**Category 3: Rule-Based / Hybrid Systems**
- Paper 3 — MadDog (Veyseh et al., 2021) — rule-based AI + BiLSTM AD, deployed as a live multi-domain web system

**Category 4: Knowledge Representation (Knowledge Graph / Ontology / Concept Mapping)**
- No paper in the current set falls into this category. This is a notable gap (see Step 2.6 and the research-gap discussion below) — none of the six papers build a knowledge graph, ontology, or concept-hierarchy layer on top of identified/disambiguated acronyms. Papers covering this category should be added from targeted searches (e.g., "acronym knowledge graph," "concept mapping scientific NLP," "hierarchical concept extraction").

---

## Step 2.4 — Evolution of the Research (Chronological)

```
2020
│
├── Paper 1: SciAI / SciAD datasets + GAD model
│     (Establishes the first large, high-quality, scientific-domain
│      benchmark; sets the human-performance target at ~96% F1)
│
2021
│
├── Paper 2: AT-BERT
│     (Introduces BERT-family fine-tuning + adversarial training +
│      ensembling for AI; F1 jumps from 86.55 → 94.12)
│
├── Paper 3: MadDog
│     (Moves from benchmark models to a deployed, multi-domain,
│      rule-based + deep learning hybrid system)
│
├── Paper 4: BERT-based AD (Multiple Training Strategies)
│     (Reframes AD as binary classification; stacks TAPT, dynamic
│      sampling, adversarial training, pseudo-labeling; F1 → 0.9405)
│
├── Paper 5: hdBERT
│     (Combines general-domain (RoBERTa) + domain-specific (SciBERT)
│      representations in a dual-path architecture; F1 → 93.73)
│
├── Paper 6: Primer AI
│     (Introduces XLNet ensembling + retrieval-based/Siamese AD;
│      critically identifies and fixes duplicate/conflicting labels
│      in SciAD, producing SciAD-dedupe)
│
2022+ (open direction, not covered by current paper set)
│
├── Larger/newer transformer and LLM-based approaches
├── Broader Scientific NLP applications (definition extraction, QA)
└── Knowledge-graph / concept-hierarchy layers on top of AI+AD output
```

**Observed trend:** research moves from (a) *dataset construction and a first syntactic baseline* (2020) → (b) *transformer fine-tuning with robustness tricks (adversarial training, ensembling)* → (c) *task reframing (binary classification, retrieval/IR-based disambiguation)* → (d) *combining domain-agnostic and domain-specific pretrained knowledge* → (e) *auditing and correcting benchmark data quality itself*. By late 2021, model performance on both AI (~92–94% F1) and AD (~91–94% F1) had closed most, but not all, of the gap to human performance (~96% F1), and the field's remaining bottlenecks had shifted from "raw model capacity" toward "data quality" and "hard residual cases" (long-distance dependency, near-synonymous candidates, insufficient context).

---

## Step 2.5 — Comparison Table

| Paper | Method | Dataset | Reported Score (F1) | Limitation |
|---|---|---|---|---|
| Veyseh et al. 2020 | GAD (BiLSTM + GCN) | SciAD | 81.90 | Dependency-based context modeling only; well below human performance |
| Veyseh et al. 2020 | LSTM-CRF | SciAI | 86.55 | Sequence-labeling baseline; no deep contextual pretraining |
| Zhu et al. 2021 | AT-BERT-E (BERT ensemble + FGM) | SciAI | 94.12 | High training cost (~2x); still below human level |
| Veyseh et al. 2021 | MadDog (rule-based + BiLSTM) | DOG + MAD (tested on SciAI/SciAD) | 88.12 (AI) / 88.49 (AD) | Large storage/compute footprint (64GB); limited evaluation outside benchmark domains |
| Pan et al. 2021 | SciBERT + TAPT + adversarial + pseudo-labeling | SciAD | 94.05 | Struggles with near-synonymous candidate expansions |
| Zhong et al. 2021 | hdBERT (RoBERTa + SciBERT dual-path) | SciAD (SciAD_BI) | 93.73 | Still below human recall; sensitive to noisy/conflicting labels |
| Egan & Bohannon 2021 | XLNet ensemble (AI) / Siamese retrieval (AD) | SciAI / SciAD, SciAD-dedupe | 92.60 (AI) / 91.58 (AD) | Retrieval-based AD depends on availability of similar training examples |

*(This directly extends the example table given in the assignment, using the actual reported numbers from the six summarized papers rather than the illustrative placeholders.)*

---

## Step 2.6 — Common Limitations Across Papers

Looking across all six papers, several limitations recur:

1. **Performance ceiling below human level.** Every single paper, regardless of method (rule-based, BiLSTM+GCN, BERT ensembles, dual-path transformers, retrieval-based), reports final scores below the ~96% human-performance benchmark established in Paper 1.
2. **Task fragmentation.** Most papers focus on *either* AI *or* AD in isolation (Papers 2 and 4 are AD/AI-only respectively); only Papers 1, 3, and 6 address both, and even then largely as two separate sub-systems rather than a jointly optimized pipeline.
3. **No concept hierarchy or knowledge-graph generation.** None of the six papers attempt to organize identified/disambiguated acronyms into a structured knowledge representation (ontology, concept hierarchy, or knowledge graph) — they stop at flat span detection or single-label classification.
4. **No semantic relationship mapping beyond acronym↔long-form.** The papers resolve the *identity* relationship (short form = long form) but do not model broader semantic relationships between concepts (e.g., "is-a," "part-of," or cross-acronym relatedness).
5. **No interactive visualization layer.** Even MadDog (Paper 3), the only deployed system, is presented as a lookup/expansion tool rather than an interactive exploration or visualization interface for understanding a document's overall concept structure.
6. **Data quality problems.** Papers 1, 5, and especially 6 note noisy, duplicated, or conflicting-label examples in the shared SciAD benchmark, meaning reported "state-of-the-art" numbers on the un-deduplicated dataset may be partly inflated.
7. **High resource cost for marginal gains.** Papers 2, 3, and 4 all report that their performance improvements come with meaningfully higher computational cost (adversarial training ~2x runtime; MadDog's 64GB storage; multi-strategy stacking in Paper 4), suggesting diminishing returns from simply scaling up training tricks.
8. **Residual hard cases are semantic, not architectural.** Papers 4 and 5 both specifically report failures on near-synonymous or highly similar candidate long forms and cases with insufficient sentence-level context — suggesting current sentence-level context windows may be an intrinsic ceiling.

---

## Step 2.7 — Literature Review (Narrative Summary / Related Work Draft)

Research on automated acronym understanding in scientific text has progressed rapidly since 2020. The field's foundation was laid by Veyseh et al. (2020), who introduced large, manually annotated datasets for acronym identification and disambiguation in the scientific domain, which addressed the earlier shortage of high-quality, non-medical benchmark data and proposed GAD, a graph-based model that used dependency-tree structure to resolve acronyms whose correct meaning depends on distant context. This established both a benchmark (SciAI/SciAD) and a human-performance ceiling (~96% F1) against which all subsequent work has been measured.

Building on this foundation, several 2021 papers — most produced for the SDU@AAAI-21 shared task — pursued complementary directions. One line of work focused on strengthening transformer-based identification and disambiguation through training-time robustness techniques: fine-tuning BERT-family models with adversarial perturbations and ensembling multiple architectures pushed acronym identification F1 from the mid-80s into the mid-90s, while a separate binary-classification reformulation of disambiguation — combining dynamic negative sampling, task-adaptive pretraining, adversarial training, and pseudo-labeling — achieved near-human disambiguation accuracy. A second line of work explored architectural fusion of complementary knowledge sources, proposing a dual-path model that combines a general-domain transformer with a scientific-domain transformer, showing that neither general nor domain-specific pretraining alone is sufficient. A third line of work moved beyond pure benchmark modeling toward practical deployment and data auditing: one system combined refined rule-based detection with a large-scale automatically labeled disambiguation dataset to build the first public, multi-domain acronym web tool, while another combined a transformer ensemble with a retrieval/embedding-based disambiguation strategy and, critically, discovered and corrected substantial duplicate and conflicting-label examples in the widely used SciAD benchmark.

Taken together, these six papers show a field that has made significant methodological progress — from syntactic graph modeling, to adversarial and ensemble transformer fine-tuning, to domain-knowledge fusion, to retrieval-based reasoning and benchmark auditing — while converging on a shared, still-unresolved gap: all reported systems remain a few points below human-level accuracy, and the remaining errors are concentrated in semantically close candidates and long-range or insufficient context, rather than in architecture choice alone. Furthermore, every method addresses either identification or disambiguation as flat, sentence-level classification tasks; none of the surveyed work organizes recognized acronyms and their meanings into a broader knowledge structure (e.g., a knowledge graph or concept hierarchy) that could support document-level or corpus-level understanding, relationship discovery, or interactive exploration. This gap — between accurate flat acronym resolution and structured, relational, document/corpus-level concept understanding — motivates the need for a new approach that goes beyond span-level identification and single-label disambiguation.

---

## Initial Research Gap (carried into Phase 3)

Based on the above, the identified research gap is:

> Existing work solves acronym identification and disambiguation reasonably well as isolated, sentence-level tasks, achieving near-human accuracy (~91–94% F1) through transformer fine-tuning, adversarial training, ensembling, and domain-knowledge fusion. However, **no reviewed work connects the *output* of acronym identification/disambiguation into a structured, relational, or hierarchical knowledge representation** (e.g., knowledge graph, concept map, or ontology) that would allow a reader or downstream system to understand how the concepts behind acronyms relate to one another across a document or corpus, nor is there an **interactive visualization** layer for this purpose. Additionally, **residual errors remain concentrated in semantically similar candidates and context-insufficient sentences**, suggesting that purely sentence-level context modeling has diminishing returns and that document-level or knowledge-grounded context could be the next meaningful improvement.

This gap forms the basis for the proposed method to be developed in Phase 3.

---

## Deliverables Checklist

- ✅ Collection of relevant research papers (six core papers; extension sources listed for Knowledge Representation category)
- ✅ Paper summaries (Step 2.2 extraction table)
- ✅ Comparison table (Step 2.5)
- ✅ Chronological evolution of methods (Step 2.4)
- ✅ Common strengths and weaknesses (Step 2.6)
- ✅ Literature review chapter (Step 2.7)
- ✅ Initial understanding of the research gap (above)
