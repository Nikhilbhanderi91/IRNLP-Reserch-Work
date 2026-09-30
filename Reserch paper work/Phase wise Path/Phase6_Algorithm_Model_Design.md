# Phase 6 — Algorithm & Model Design

**Builds on:** Phase 1–5 (Problem, Literature Review, Research Gap, Methodology, Dataset)

This phase specifies, module by module, the exact algorithm and model each stage of the Phase 4 architecture will use — so implementation in a later phase is a direct translation of this document, not a design exercise.

---

## Step 6.1 — Overall Processing Pipeline

```
Research Paper (PDF)
        │
        ▼
PDF Text Extraction
        │
        ▼
Text Cleaning
        │
        ▼
Sentence Segmentation
        │
        ▼
Tokenization
        │
        ▼
Abbreviation Identification
        │
        ▼
Long Form Extraction
        │
        ▼
Acronym Disambiguation
        │
        ▼
Concept Extraction
        │
        ▼
CCC Concept Mapping
        │
        ▼
Hierarchy Generation
        │
        ▼
Knowledge Graph Construction
        │
        ▼
Visualization
```

Same 12-stage flow as Phase 4's architecture (Step 4.6); this phase fills in the algorithm/model detail behind each arrow.

---

## Step 6.2 — Algorithm: PDF Processing

**Input:** `paper.pdf`

**Process:**
```
1. Read PDF file (PyMuPDF / pdfplumber)
2. FOR each page in PDF:
       extract text blocks in reading order (handle multi-column layout)
       extract embedded metadata (title, authors, if present in PDF properties)
3. Concatenate page texts → raw_text
4. Store raw_text → Raw_Text/P00X.txt
5. Store metadata → Metadata/P00X.json (cross-check against Phase 5 metadata schema)
```

**Tools:** PyMuPDF, pdfplumber, Apache Tika (fallback for scanned/complex layouts)

---

## Step 6.3 — Algorithm: Text Preprocessing

**Input:** Raw text

```
1. Preserve original case for abbreviation detection (do NOT lowercase globally —
   case is a primary signal for acronym candidates, e.g. all-caps "CNN")
2. Remove non-textual artifacts: page numbers, running headers/footers, URLs,
   figure/table captions (regex + layout-position heuristics from Phase 5, Step 5.6)
3. Normalize whitespace, de-hyphenate line-wrapped words
4. Sentence split (SciSpaCy sentencizer)
5. Tokenize (SciSpaCy tokenizer)
6. Lemmatize + POS tag (used as auxiliary features for Step 6.4 rule-based candidates
   and for Step 6.7 concept extraction, not for the acronym spans themselves)
```

**Output:** Clean, sentence-segmented, tokenized, POS-tagged text → `Clean_Text/`, `Sentences/`, `Tokens/`

---

## Step 6.4 — Algorithm: Abbreviation Identification (Module 4)

**Input:** Sentence, e.g. *"Convolutional Neural Network (CNN) is widely used..."*
**Output:** Span `CNN`, tagged with BIO labels (`B-short`, `I-short`, `O`)

**Selected approach: BERT-family sequence labeling, following Paper 2 (AT-BERT)**

```
MODEL: SciBERT encoder + linear classification head (BIO tagging)
TRAINING (optional, adversarial): apply FGM perturbation to token embeddings
  during fine-tuning (Paper 2's approach) to improve robustness on the
  relatively small SciAI-scale training set.

INFERENCE:
1. tokens = tokenize(sentence)
2. embeddings = SciBERT(tokens)
3. logits = Linear(embeddings)
4. tags = argmax(logits) per token   # B-short / I-short / O
5. spans = decode_BIO(tags)          # e.g. ["CNN"]
RETURN spans
```

**Fallback (low-resource / rule-based path, following Paper 1 & Paper 3):**
```
IF token is all-uppercase AND length 2-6 AND appears in parentheses:
    mark as abbreviation candidate
```

**Rationale:** Paper 2's adversarially-trained BERT ensemble is the strongest reported AI approach (94.12 F1); the rule-based fallback (from Papers 1 and 3) provides a lightweight, dependency-free baseline for early testing before a model is trained/fine-tuned.

---

## Step 6.5 — Algorithm: Long Form Extraction (Module 5)

**Input:** `CNN`
**Output:** `Convolutional Neural Network` → added to `Abbreviations.csv` (Phase 5)

```
1. Search sentence window around the abbreviation for a candidate long form:
   a. Parenthesis rule: "Long Form (SHORT)" or "SHORT (Long Form)"
   b. Schwartz & Hearst rule: align capital letters of SHORT with the initial
      letters of a candidate phrase preceding/following it
2. IF no local match found:
   Look up SHORT in the pre-built glossary (Phase 5's Abbreviations.csv,
   extendable the way Paper 3's DOG glossary works — built from prior papers)
3. Normalize the extracted long form (strip trailing punctuation,
   Levenshtein-distance dedup against existing dictionary entries, per Paper 1)
RETURN long_form
```

**Rationale:** Combines the classic Schwartz & Hearst rule (used as the seed detector in Paper 3's MadDog) with dictionary lookup, avoiding the need for a trained extraction model for this sub-step.

---

## Step 6.6 — Algorithm: Acronym Disambiguation (Module 6)

**Input:** Sentence + ambiguous acronym (e.g. `CNN` → candidates: *Convolutional Neural Network* / *Cable News Network*)
**Output:** Correct meaning

**Selected approach: dual-path fusion, following Paper 5 (hdBERT), with training strategies from Paper 4**

```
MODEL: hdBERT-style dual encoder
  branch_general  = RoBERTa(sentence + candidate)   # domain-agnostic
  branch_domain   = SciBERT(sentence + candidate)    # domain-specific
  combined        = concat(branch_general[CLS], branch_domain[CLS])
  score           = sigmoid(MLP(combined))           # match probability

TRAINING STRATEGIES (from Paper 4, layered on top if resources allow):
  - dynamic negative sampling (reported single largest F1 gain, +4%)
  - task-adaptive pretraining (TAPT) on in-domain unlabeled text
  - adversarial training (FGM)
  - pseudo-labeling of high-confidence (>0.95) predictions

INFERENCE (for an ambiguous acronym with N candidate long forms):
1. FOR each candidate c in candidates:
       score[c] = hdBERT_score(sentence, acronym, c)
2. RETURN argmax(score)
```

**Fallback (retrieval-based, following Paper 6):**
```
1. Embed sentence via a Siamese/sentence-transformer encoder
2. Find nearest-neighbor sentence in the training set with the same acronym
3. RETURN that neighbor's gold label
   (used only when candidate long forms aren't explicitly enumerated)
```

**Rationale:** hdBERT (93.73 F1) and Pan et al.'s multi-strategy SciBERT classifier (94.05 F1) are the two strongest reported AD methods reviewed in Phase 2; combining their core ideas (dual-path fusion + dynamic sampling) is the primary model, with Paper 6's retrieval approach as a fallback when no fixed candidate set exists.

---

## Step 6.7 — Algorithm: Concept Extraction (Module 7) — *novel*

This is where the proposed contribution begins (per Phase 3's research gap). Instead of stopping at the full form, the system links the full form to a broader concept.

**Input:** `CNN` → `Convolutional Neural Network`
**Output:** `Deep Learning`

```
1. Look up (abbreviation, full_form) in the seeded Concept dictionary
   (Phase 5's Concepts.csv) — direct hit if the term is already known.
2. IF not found:
   a. Compute an embedding for full_form (SciBERT [CLS] embedding)
   b. Compare against embeddings of known concept-anchor terms
      (e.g. "Deep Learning", "Machine Learning", "Natural Language Processing")
      using cosine similarity
   c. Assign the concept whose anchor embedding is most similar,
      above a confidence threshold theta
   d. IF below threshold: flag for manual/human-in-the-loop labeling
3. Cache the new (full_form -> concept) mapping into Concepts.csv for reuse
RETURN concept
```

**Rationale:** No reviewed paper performs this step (Phase 3, Step 3.3) — it is implemented as a hybrid of dictionary lookup (fast, precise, reuses Phase 5's seed data) and embedding-similarity fallback (generalizes to unseen abbreviations).

---

## Step 6.8 — CCC Mapping Algorithm (Module 8) — *core contribution*

**Goal:** Instead of storing a flat `abbreviation → full_form` pair, store a full chain: `abbreviation → concept → parent_concept → root_concept`.

**Example:**
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

**Algorithm:**
```
FUNCTION ccc_map(abbreviation, full_form):
    concept = concept_extraction(abbreviation, full_form)      # Step 6.7
    chain = [abbreviation, full_form, concept]

    current = concept
    WHILE current has a parent in the category taxonomy:
        parent = taxonomy_lookup(current)      # e.g. Concepts.csv "Category" field,
                                                 # or a curated CS/AI taxonomy (ACM CCS)
        chain.append(parent)
        current = parent

    RETURN chain    # Concept -> Category -> Context (CCC)
```

**Taxonomy source:** initially the `Category` column already present in Phase 5's `Concepts.csv` (e.g. `CNN → Deep Learning → Machine Learning`); extendable with a standard external taxonomy (e.g. ACM Computing Classification System) once the corpus grows beyond the AI/CS seed domain.

**Context component:** each CCC chain also stores the originating sentence/section (from Step 6.3) as provenance, so a concept mapping can be traced back to the specific claim in the specific paper that produced it — this is the "Context" in CCC.

---

## Step 6.9 — Hierarchy Generation Algorithm (Module 9)

**Input:** List of CCC chains (Step 6.8)
**Output:** A tree

```
Artificial Intelligence
│
├── Machine Learning
│
├── Deep Learning
│      ├── CNN
│      ├── GAN
│      ├── Transformer
│      └── LSTM
```

**Algorithm (parent-child linking + tree construction):**
```
FUNCTION build_hierarchy(ccc_chains):
    tree = {}   # adjacency map: parent -> set(children)
    FOR chain in ccc_chains:
        FOR i in range(len(chain) - 1, 0, -1):     # walk chain root -> leaf
            child, parent = chain[i-1], chain[i]
            tree[parent].add(child)
    root = find_node_with_no_parent(tree)          # e.g. "Artificial Intelligence"
    RETURN render_tree(tree, root)                  # recursive DFS print/serialize
```

**Complexity:** with `n` CCC chains of average depth `d`, building the adjacency map is `O(n·d)`; tree traversal for rendering is `O(V)` where `V` is the number of unique concept nodes.

---

## Step 6.10 — Relationship Extraction Algorithm (Module 8b, feeding Module 10)

Beyond the vertical CCC hierarchy, extract lateral relationships between concepts (following Phase 5's `Relationships.csv` schema).

```
RELATION_PATTERNS = {
  "used_for":   [" is used for ", " used to ", " applied to "],
  "belongs_to": ["is a type of", "is a kind of", "belongs to"],
  "part_of":    ["is part of", "component of"],
  "extends":    ["extends", "builds on", "based on"],
  "improves":   ["improves", "outperforms", "outperformed"],
  "depends_on": ["depends on", "relies on", "requires"],
  "similar_to": ["similar to", "comparable to"]
}

FUNCTION extract_relationships(sentence, concept_A, concept_B):
    FOR relation, patterns in RELATION_PATTERNS:
        IF any pattern found in sentence between concept_A and concept_B:
            RETURN (concept_A, relation, concept_B)
    # fallback: transformer-based relation classifier if no pattern matches
    RETURN classify_relation_with_model(sentence, concept_A, concept_B)
```

**Relation types:** `belongs_to`, `used_for`, `part_of`, `extends`, `improves`, `depends_on`, `similar_to` — matching Phase 5's seed `Relationships.csv` (e.g. `hdBERT, outperforms, GAD` uses the `improves`-family pattern).

---

## Step 6.11 — Knowledge Graph Construction (Module 10a)

```
FUNCTION build_knowledge_graph(ccc_chains, relationships):
    G = Graph()   # NetworkX DiGraph
    FOR chain in ccc_chains:
        FOR i in range(len(chain) - 1):
            G.add_edge(chain[i], chain[i+1], relation="belongs_to")
    FOR (source, relation, target) in relationships:
        G.add_edge(source, target, relation=relation)
    RETURN G
```

**Example (matches Phase 5 seed data):**
```
CNN --belongs_to--> Deep Learning --part_of--> Machine Learning --part_of--> Artificial Intelligence
```

**Nodes:** every abbreviation, full form, and concept (CNN, Deep Learning, Machine Learning, Artificial Intelligence, ...)
**Edges:** `belongs_to` (from CCC hierarchy), plus `used_for`, `related_to`, `improves`, etc. (from Step 6.10)

---

## Step 6.12 — Visualization Algorithm (Module 10b)

```
FUNCTION visualize(G, mode):
    IF mode == "tree":       RETURN render_tree_layout(G)        # PyVis hierarchical layout
    IF mode == "graph":      RETURN render_force_directed(G)      # Cytoscape / D3.js force layout
    IF mode == "mindmap":    RETURN render_radial_layout(G)       # radial tree, root at center
    ADD interactivity: click node -> expand/collapse children,
                        hover edge -> show relation label + source sentence (context)
```

**Libraries:** NetworkX (graph object + basic layout), PyVis (quick interactive HTML export), Cytoscape.js / D3.js (richer, production interactive front-end) — matching the stack chosen in Phase 4, Step 4.7.

---

## Step 6.13 — Complexity Analysis

| Module | Time Complexity | Space Complexity | Notes |
|---|---|---|---|
| PDF Extraction | O(n) | O(n) | n = number of characters/pages |
| Text Cleaning | O(n) | O(n) | n = text length |
| Tokenization | O(n) | O(n) | n = text length |
| Abbreviation Detection (SciBERT tagging) | O(n · L) | O(n) | n = number of tokens, L = transformer sequence-processing cost per token (effectively O(n²) within a single attention window of fixed max length, O(n) across windows) |
| Long Form Extraction | O(n · w) | O(k) | w = local search window size, k = dictionary size (O(1) amortized with hash lookup) |
| Disambiguation (hdBERT) | O(m · L) | O(m) | m = number of ambiguous acronym instances × candidate count |
| Concept Extraction | O(m) amortized (dict lookup), O(m · c) worst case | O(c) | c = number of concept anchors (embedding comparison fallback) |
| CCC Mapping | O(m · d) | O(m · d) | d = average chain depth (typically 3-5) |
| Hierarchy Generation | O(V + E) | O(V + E) | V = unique concepts, E = parent-child edges |
| Relationship Extraction | O(n · r) | O(r) | r = number of relation patterns checked per sentence pair |
| Knowledge Graph Construction | O(V + E) | O(V + E) | standard graph build |
| Visualization | O(V + E) | O(V + E) | rendering cost scales with graph size |

*(These are first-pass estimates per the assignment's own guidance — Step 6.13 notes they should be refined once implementation and profiling are done in a later phase.)*

---

## Step 6.14 — Final Algorithm Flow

```
PDF
 ↓
Text
 ↓
Sentences
 ↓
Tokens
 ↓
Abbreviations
 ↓
Full Forms
 ↓
Disambiguation
 ↓
Concept Detection
 ↓
CCC Mapping
 ↓
Hierarchy
 ↓
Knowledge Graph
 ↓
Visualization
```

---

## Model Selection Summary

| Module | Selected Model/Algorithm | Justification (from Phase 2 literature) |
|---|---|---|
| Abbreviation Identification | SciBERT + BIO tagging (+ optional FGM adversarial training) | Paper 2 (AT-BERT): 94.12 F1, strongest reviewed AI result |
| Long Form Extraction | Schwartz & Hearst rule + dictionary lookup | Paper 3 (MadDog): high-precision rule-based approach, no training data needed |
| Disambiguation | hdBERT-style dual-path fusion + Paper 4's training strategies | Papers 4 & 5: 94.05 and 93.73 F1, the two strongest reviewed AD results |
| Concept Extraction | Dictionary lookup + embedding-similarity fallback | Novel — no direct precedent; designed to reuse Phase 5 seed data |
| CCC Mapping | Recursive taxonomy-chain construction | Novel — core proposed contribution (Phase 3) |
| Hierarchy Generation | Parent-child adjacency + tree traversal | Novel — standard tree-construction algorithm applied to a new data type |
| Relationship Extraction | Pattern matching + transformer fallback classifier | Novel — pattern set matches Phase 5's `Relationships.csv` relation types |
| Knowledge Graph | NetworkX directed graph | Standard, matches Phase 4 tech stack |
| Visualization | PyVis / Cytoscape.js / D3.js | Standard, matches Phase 4 tech stack; addresses the "no interactive visualization" gap from Phase 3 |

---

## Deliverables Checklist

- ✅ Overall processing algorithm (Step 6.1, 6.14)
- ✅ Module-wise algorithms (Steps 6.2–6.12)
- ✅ Model selection for each module (Model Selection Summary)
- ✅ CCC mapping algorithm — core contribution (Step 6.8)
- ✅ Hierarchy generation algorithm (Step 6.9)
- ✅ Knowledge graph algorithm (Step 6.11)
- ✅ Visualization strategy (Step 6.12)
- ✅ Complexity analysis (Step 6.13)
