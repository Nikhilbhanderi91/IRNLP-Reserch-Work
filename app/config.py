"""
================================================
app/config.py
Global configuration and constants
================================================
"""

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT_DIR / "data"
UPLOAD_DIR  = DATA_DIR / "uploads"
PROC_DIR    = DATA_DIR / "processed"
EXPORT_DIR  = DATA_DIR / "exports"
LOG_DIR     = ROOT_DIR / "logs"

for _d in (UPLOAD_DIR, PROC_DIR, EXPORT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─── NLP Models ───────────────────────────────────────────────────────────────
SPACY_MODEL         = "en_core_web_sm"   # fallback if scispacy not installed
SCISPACY_MODEL      = "en_core_sci_sm"
SENTENCE_MODEL      = "all-MiniLM-L6-v2"  # lightweight yet accurate
USE_SCIBERT         = True
SCIBERT_MODEL       = "allenai/scibert_scivocab_uncased"

# ─── Keyword extraction settings ──────────────────────────────────────────────
TFIDF_MAX_FEATURES  = 5000
TFIDF_NGRAM_RANGE   = (1, 3)
KEYBERT_TOP_N       = 20
KEYBERT_NGRAM       = (1, 3)
YAKE_MAX_NGRAM      = 3
YAKE_TOP_N          = 20
KEYWORD_TOP_N       = 30          # final merged keyword count

# ─── Similarity ───────────────────────────────────────────────────────────────
SIM_CHUNK_SIZE      = 512         # max tokens for sentence transformer
SIM_METHODS         = ["tfidf_cosine", "sentence_transformer"]

# ─── Visualization colours ────────────────────────────────────────────────────
COLOR_PALETTE = [
    "#6C63FF", "#FF6584", "#43BCCD", "#F9A826",
    "#00C49A", "#FF8C42", "#845EC2", "#D65DB1",
]

NODE_COLOR_MAP = {
    "domain":    "#6C63FF",
    "concept":   "#43BCCD",
    "keyword":   "#00C49A",
    "entity":    "#F9A826",
    "paper":     "#FF6584",
}

# ─── Domain keyword taxonomy ──────────────────────────────────────────────────
# Used by hierarchy builder to assign broad research domains.
DOMAIN_TAXONOMY = {
    "Artificial Intelligence": [
        "artificial intelligence", "ai", "machine learning", "deep learning",
        "neural network", "reinforcement learning", "transfer learning",
        "generative model", "foundation model", "large language model", "llm",
    ],
    "Natural Language Processing": [
        "natural language processing", "nlp", "text mining", "sentiment analysis",
        "named entity recognition", "ner", "question answering", "summarization",
        "language model", "bert", "gpt", "transformer",
    ],
    "Computer Vision": [
        "computer vision", "image classification", "object detection",
        "segmentation", "convolutional neural network", "cnn", "resnet",
        "image recognition", "visual", "optical flow",
    ],
    "Data Science": [
        "data science", "data mining", "feature extraction", "feature engineering",
        "dimensionality reduction", "clustering", "classification", "regression",
        "random forest", "gradient boosting", "xgboost",
    ],
    "Cybersecurity": [
        "cybersecurity", "intrusion detection", "malware", "anomaly detection",
        "network security", "cryptography", "privacy", "adversarial",
    ],
    "Bioinformatics": [
        "bioinformatics", "genomics", "protein", "dna", "rna", "drug discovery",
        "molecular", "biological", "clinical",
    ],
    "Robotics": [
        "robotics", "autonomous", "path planning", "localization", "mapping",
        "slam", "manipulation", "control system",
    ],
    "Cloud & Systems": [
        "cloud computing", "distributed system", "microservice", "kubernetes",
        "docker", "scalability", "latency", "edge computing",
    ],
}

# ─── App UI ───────────────────────────────────────────────────────────────────
APP_TITLE   = "HMRP – Research Paper Concept Mapper"
APP_ICON    = "🔬"
APP_LAYOUT  = "wide"
