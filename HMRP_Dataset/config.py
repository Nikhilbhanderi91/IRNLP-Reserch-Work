"""
================================================
HMRP Multidisciplinary Dataset Configuration
Target: 15 Domains x 50 Papers = 750 Valid Papers
================================================
"""

TARGET_PER_DOMAIN = 50

# Thresholds for strict paper validation
MIN_TITLE_WORDS = 3
MIN_ABSTRACT_WORDS = 40
MIN_FULLTEXT_WORDS = 300

# Directory folder name sanitization mapping
DOMAIN_DIR_NAMES = {
    "Natural Language Processing": "Natural_Language_Processing",
    "Information Retrieval": "Information_Retrieval",
    "Computer Vision": "Computer_Vision",
    "Robotics": "Robotics",
    "Artificial Intelligence": "Artificial_Intelligence",
    "Statistics & Machine Learning": "Statistics_Machine_Learning",
    "Applied Mathematics & Optimization": "Applied_Mathematics_Optimization",
    "Astrophysics & Cosmology": "Astrophysics_Cosmology",
    "Quantum Physics & Optics": "Quantum_Physics_Optics",
    "Biomedical & Genomics": "Biomedical_Genomics",
    "Computational Neuroscience": "Computational_Neuroscience",
    "Signal Processing": "Signal_Processing",
    "Systems & Control Engineering": "Systems_Control_Engineering",
    "Quantitative Finance": "Quantitative_Finance",
    "Econometrics & Economics": "Econometrics_Economics",
}

DOMAINS = {
    # ── 1. Computer Science & AI ──────────────────────────────────
    "Natural Language Processing": ["cs.CL"],
    "Information Retrieval": ["cs.IR"],
    "Computer Vision": ["cs.CV"],
    "Robotics": ["cs.RO"],
    "Artificial Intelligence": ["cs.AI"],

    # ── 2. Mathematics & Statistics ──────────────────────────────
    "Statistics & Machine Learning": ["stat.ML", "cs.LG"],
    "Applied Mathematics & Optimization": ["math.OC", "math.PR"],

    # ── 3. Physical Sciences ──────────────────────────────────────
    "Astrophysics & Cosmology": ["astro-ph.CO", "astro-ph.GA"],
    "Quantum Physics & Optics": ["quant-ph", "physics.optics"],

    # ── 4. Biological & Medical Sciences ──────────────────────────
    "Biomedical & Genomics": ["q-bio.BM", "q-bio.GN"],
    "Computational Neuroscience": ["q-bio.NC"],

    # ── 5. Electrical & Systems Engineering ───────────────────────
    "Signal Processing": ["eess.SP"],
    "Systems & Control Engineering": ["eess.SY"],

    # ── 6. Economics & Finance ────────────────────────────────────
    "Quantitative Finance": ["q-fin.PM", "q-fin.RM"],
    "Econometrics & Economics": ["econ.EM", "econ.TH"],
}
