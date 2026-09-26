#!/usr/bin/env bash
# =====================================================
# install.sh  –  One-shot setup script for HMRP
# =====================================================
# Usage:  bash install.sh
# =====================================================

set -e

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   HMRP – Research Paper Concept Mapper           ║"
echo "║   Installation Script                             ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── 1. Check Python version ───────────────────────────
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✓ Python $python_version detected"
if [[ "$(python3 -c 'import sys; print(sys.version_info >= (3, 9))')" == "False" ]]; then
    echo "✗ Python 3.9+ required."
    exit 1
fi

# ── 2. Create virtualenv if not present ──────────────
if [ ! -d "venv" ]; then
    echo "→ Creating virtual environment…"
    python3 -m venv venv
fi
source venv/bin/activate
echo "✓ Virtual environment activated"

# ── 3. Upgrade pip ───────────────────────────────────
pip install --upgrade pip --quiet

# ── 4. Install requirements ──────────────────────────
echo "→ Installing dependencies (this may take a few minutes)…"
pip install -r requirements.txt

# ── 5. Download spaCy model ──────────────────────────
echo "→ Downloading spaCy English model…"
python -m spacy download en_core_web_sm || true

# ── 6. Download NLTK data ────────────────────────────
echo "→ Downloading NLTK resources…"
python -c "
import nltk
for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
    nltk.download(pkg, quiet=True)
print('NLTK resources ready.')
"

# ── 7. Create directories ─────────────────────────────
mkdir -p data/uploads data/processed data/exports logs

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   ✅  Installation complete!                      ║"
echo "║                                                   ║"
echo "║   Launch:  python run.py                          ║"
echo "║   OR:      streamlit run app/main.py              ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
