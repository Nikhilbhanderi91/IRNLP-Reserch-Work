# Phase 1 — Problem Understanding

**Research Area:** Acronym Identification (AI) and Acronym Disambiguation (AD) in Scientific Text

---

## 1. Problem Statement

Scientific and technical documents make heavy use of acronyms and abbreviations, but automatically recognizing these short forms and correctly resolving what they mean is still an unsolved problem for NLP systems. Two coupled sub-problems exist:

- **Acronym Identification (AI):** given a sentence, detect which spans are acronyms (short forms) and which spans are their corresponding long forms (definitions).
- **Acronym Disambiguation (AD):** given an acronym that has multiple possible expansions (e.g., "CNN" → *Convolutional Neural Network* or *Cable News Network*), determine the correct long form based on the surrounding context.

Historically, research on these tasks was held back by datasets that were too small, generated through noisy rule-based heuristics, or narrowly confined to the medical/biomedical domain. This made it difficult to train and fairly evaluate deep learning models capable of handling the scale, ambiguity, and domain diversity of real scientific writing (arXiv papers, general text, financial text, etc.). Even after large benchmark datasets (SciAI, SciAD) were introduced, subsequent work revealed additional problems — annotation noise, duplicate examples with conflicting labels, and a persistent performance gap between the best models (~81–94% Macro F1) and human-level accuracy (~96% F1).

**Core problem:** existing models and resources for acronym identification and disambiguation in scientific text are limited by data quality/scale issues and still fall short of human performance, making reliable automated acronym understanding an open challenge.

---

## 2. Motivation

- Acronyms are extremely pervasive in technical writing — roughly 15% of PubMed queries and a large share of scientific/clinical text contain abbreviations.
- Misinterpreting an acronym can silently propagate errors into downstream NLP applications such as question answering, information retrieval, document/glossary generation, and definition extraction.
- Acronyms are frequently ambiguous (the same short form maps to different long forms across domains or even within the same domain), and their correct meaning often depends on distant contextual or syntactic cues rather than the immediately adjacent words.
- Prior benchmark datasets were flawed (small, noisy, single-domain, or containing duplicated/conflicting labels), which both limited model training and biased reported evaluation results.
- No unified, publicly available, multi-domain tool existed that could both detect and resolve acronyms for a general reader or downstream system — a real practical gap for reading assistance and text-understanding applications.
- Closing the remaining gap to human-level performance (~96% F1) is still an active, well-defined, and measurable research target.

---

## 3. Objectives

1. To build/use large-scale, high-quality, human-annotated datasets for acronym identification and disambiguation in the scientific domain (e.g., SciAI, SciAD, and their refined variants such as SciAD-dedupe).
2. To design or apply deep learning models that effectively identify acronym–long form pairs in text (e.g., sequence labeling with LSTM-CRF, transformer-based taggers such as XLNet/BERT variants).
3. To design or apply models that correctly disambiguate acronyms with multiple candidate meanings by leveraging sentence-level context, syntactic structure (dependency trees), and/or pre-trained language model representations (general-domain and domain-specific).
4. To evaluate and benchmark these models using standard metrics (Precision, Recall, Macro-F1) against existing baselines and human performance.
5. To identify and, where possible, correct data quality issues (e.g., duplication, conflicting labels) that bias evaluation.
6. To explore techniques — such as adversarial training, ensembling, task-adaptive pretraining, and dual-path (general + domain-specific) architectures — that improve robustness and push performance closer to human-level accuracy.
7. (Optionally) To package the resulting approach into a usable, multi-domain, deployable system for real-world acronym lookup/disambiguation.

---

## 4. Expected Output

- A working acronym identification model that, given raw scientific text, outputs the spans of short forms and their long forms.
- A working acronym disambiguation model that, given an ambiguous acronym and its sentence context, predicts the correct long-form meaning from a set of candidates.
- Quantitative evaluation results (Precision, Recall, Macro-F1) benchmarked against established baselines (e.g., LSTM-CRF, GAD, BERT/SciBERT/RoBERTa variants, hdBERT, XLNet ensembles) and against human-level performance.
- A comparative analysis highlighting which techniques (syntactic/dependency information, adversarial training, ensembling, domain-specific vs. domain-agnostic pretraining, retrieval-based disambiguation) contribute most to performance gains.
- Documentation of any data quality issues found (e.g., duplicate or conflicting examples) and how they were handled.
- (Optionally) A demonstrable end-to-end pipeline or interface for acronym detection and disambiguation.

---

## 5. Scope

**In scope:**
- Acronym identification and disambiguation for scientific/academic text (primarily English, arXiv-style papers), following the SciAI/SciAD benchmark framing established in the surveyed literature.
- Use of established datasets (SciAI, SciAD, SciAD-dedupe, and related auxiliary sets like AuxAI/AuxAD) and/or construction of similar resources.
- Application and comparison of deep learning approaches: sequence labeling (LSTM-CRF), graph-based context modeling (BiLSTM+GCN), transformer fine-tuning (BERT, SciBERT, RoBERTa, ALBERT, ELECTRA, XLNet), adversarial training (FGM), model ensembling, dual-path architectures (hdBERT), and retrieval/embedding-based disambiguation (Siamese networks).
- Standard intrinsic evaluation using Precision, Recall, and Macro-F1 against benchmark test sets and reported human performance.

**Out of scope:**
- Full multi-lingual acronym handling (the surveyed work is English-centric).
- Deep exploration of non-scientific domains (financial, general Wikipedia, Reddit, biomedical) beyond what is needed for comparison, unless explicitly extended.
- Building a production-grade, large-scale deployed web service (this is a possible extension, not a core requirement, unless the project scope specifically calls for a system demo similar to MadDog).
- Real-time or streaming acronym resolution use cases.

---

## 6. Expected Contributions

1. A clear synthesis of the current state of the art in scientific acronym identification and disambiguation, based on six key papers spanning dataset creation, graph-based modeling, transformer fine-tuning with adversarial training, multi-domain systems, dual-path domain fusion, and data-quality auditing.
2. Implementation and/or reproduction of one or more competitive baseline/state-of-the-art models for AI and/or AD.
3. An evaluation and comparison of the trade-offs between different modeling strategies (syntactic structure vs. pure transformer context vs. retrieval-based approaches; domain-agnostic vs. domain-specific pretraining).
4. Insight into how dataset quality (duplication, annotation noise) affects reported model performance, and best practices for mitigating this.
5. A consolidated understanding of the remaining gap to human-level performance and the specific error types (e.g., long-distance dependencies, near-synonymous candidate meanings, insufficient context) that still challenge current models.
6. A foundation/roadmap for subsequent phases of the project (e.g., model design, implementation, experimentation, and evaluation phases that follow this Phase 1 problem understanding).

---

*This document is based on a synthesis of six papers: (1) Veyseh et al. 2020 — SciAI/SciAD & GAD; (2) Zhu et al. 2021 — AT-BERT; (3) Veyseh et al. 2021 — MadDog; (4) Pan et al. 2021 — BERT-based AD with multiple training strategies; (5) Zhong et al. 2021 — hdBERT; (6) Egan & Bohannon 2021 — Primer AI's XLNet/Siamese systems and SciAD-dedupe.*
