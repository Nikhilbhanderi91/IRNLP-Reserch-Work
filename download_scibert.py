"""
download_scibert.py
-------------------
Run this script ONCE (with internet access) to download and cache the
SciBERT model locally. After this, tests and the app will work offline.

Usage:
    python3 download_scibert.py
"""

from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "allenai/scibert_scivocab_uncased"

print(f"Downloading '{MODEL_NAME}' from HuggingFace Hub...")
print("This is ~440 MB and only needs to be done once.\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

print("\n✅ SciBERT successfully downloaded and cached!")
print("You can now run tests offline:")
print("    python3 -m pytest tests/test_scibert.py -v")
