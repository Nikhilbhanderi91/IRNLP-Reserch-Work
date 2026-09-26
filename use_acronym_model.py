"""
Acronym Model Inference Script

This script loads `acronym_model.pkl` and performs inference using both sub-modules:
1. Module 4: Token Classification (SciBERT for Acronym Identification - BIO tagging)
2. Module 6: Sequence Classification (SciBERT for Acronym/Sentence Classification)
"""

import pickle
import torch
from transformers import BertForTokenClassification, BertForSequenceClassification

MODEL_PATH = "acronym_model.pkl"

def load_acronym_model(pkl_path: str = MODEL_PATH):
    """Loads tokenizer, Module 4, and Module 6 from acronym_model.pkl."""
    print(f"Loading model file: {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    tokenizer = data["tokenizer"]
    model_name = data.get("model_name", "allenai/scibert_scivocab_uncased")

    # Load Module 4 (Token Classifier for Acronym Identification)
    m4_config = data["module4"]["config"]
    m4_state = data["module4"]["model_state_dict"]
    model4 = BertForTokenClassification(m4_config)
    model4.load_state_dict(m4_state)
    model4.eval()

    # Load Module 6 (Sequence Classifier)
    m6_config = data["module6"]["config"]
    m6_state = data["module6"]["model_state_dict"]
    model6 = BertForSequenceClassification(m6_config)
    model6.load_state_dict(m6_state)
    model6.eval()

    id2label4 = data["module4"].get("id2label", {0: "O", 1: "B-ABBR", 2: "I-ABBR"})

    return {
        "tokenizer": tokenizer,
        "model_name": model_name,
        "module4": model4,
        "module6": model6,
        "id2label4": id2label4
    }


def analyze_text(text: str, model_dict: dict):
    """
    Performs inference on the input text using both Module 4 and Module 6.
    """
    tokenizer = model_dict["tokenizer"]
    model4 = model_dict["module4"]
    model6 = model_dict["module6"]
    id2label4 = model_dict["id2label4"]

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    with torch.no_grad():
        # Module 4 Inference (Token Classification / Acronym Extraction)
        out4 = model4(**inputs)
        preds4 = torch.argmax(out4.logits, dim=-1)[0]

        # Module 6 Inference (Sequence Classification)
        out6 = model6(**inputs)
        logits6 = out6.logits[0]
        probs6 = torch.softmax(logits6, dim=-1)

    # Format Module 4 Predictions
    token_labels = []
    acronym_spans = []
    current_acronym = ""

    for tok, p in zip(tokens, preds4):
        label = id2label4.get(p.item(), str(p.item()))
        token_labels.append((tok, label))

        if label in ("B-ABBR", "I-ABBR"):
            clean_tok = tok.replace("##", "")
            current_acronym += clean_tok
        else:
            if current_acronym:
                acronym_spans.append(current_acronym)
                current_acronym = ""
    if current_acronym:
        acronym_spans.append(current_acronym)

    return {
        "input_text": text,
        "module4_predictions": {
            "token_tags": [(t, l) for t, l in token_labels if l != "O"],
            "extracted_acronyms": acronym_spans
        },
        "module6_predictions": {
            "logits": logits6.tolist(),
            "probabilities": {
                "class_0": float(probs6[0]),
                "class_1": float(probs6[1])
            },
            "predicted_class": int(torch.argmax(probs6).item())
        }
    }


if __name__ == "__main__":
    models = load_acronym_model(MODEL_PATH)
    
    sample_texts = [
        "Acronym Identification (AI) and Acronym Disambiguation (AD) in NLP.",
        "We present AT-BERT model for Natural Language Processing.",
        "The Convolutional Neural Network (CNN) is widely used in Computer Vision (CV)."
    ]
    
    print("\n" + "="*70)
    print("ACRONYM MODEL INFERENCE RESULTS")
    print("="*70)

    for text in sample_texts:
        result = analyze_text(text, models)
        print(f"\nText: \"{result['input_text']}\"")
        print("  [Module 4 - Token Classification]")
        print(f"    - Extracted Acronym Tokens: {result['module4_predictions']['extracted_acronyms']}")
        print(f"    - Tagged Sub-tokens: {result['module4_predictions']['token_tags']}")
        print("  [Module 6 - Sequence Classification]")
        print(f"    - Class Probabilities: Class 0 = {result['module6_predictions']['probabilities']['class_0']:.4f}, Class 1 = {result['module6_predictions']['probabilities']['class_1']:.4f}")
        print(f"    - Predicted Class: {result['module6_predictions']['predicted_class']}")
        print("-" * 70)
