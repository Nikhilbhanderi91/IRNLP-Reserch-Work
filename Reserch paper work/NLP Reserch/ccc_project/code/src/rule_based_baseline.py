"""
Rule-based Acronym Identification baseline, reproducing the paper's
`Baseline (Rule-based)` row in Table 3 (P=91.31 R=77.93 F1=84.09).

Method (paper, "Participating Systems & Results"):
  - a token is an acronym CANDIDATE if >60% of its characters are uppercase
    AND it is inside or right before parentheses (relaxed rule also allows
    acronyms without parentheses, to match SciAI's annotation policy).
  - given an acronym candidate A, search a window of size
        min(len(A) + 5, 2 * len(A))
    tokens before/after it for a sequence of words whose initials can spell A.

Usage:
    python rule_based_baseline.py --data data/test.json
"""
import argparse
import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


def upper_ratio(token: str) -> float:
    letters = [c for c in token if c.isalpha()]
    if not letters:
        return 0.0
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters)


def is_acronym_candidate(token: str, relaxed: bool = True) -> bool:
    """relaxed=True matches SciAI's policy of labeling acronyms even without
    parentheses (paper section 'Acronym Identification' baseline description)."""
    if len(token) < 2:
        return False
    return upper_ratio(token) > 0.6


def window_size(acronym: str) -> int:
    a = len(acronym)
    return min(a + 5, 2 * a)


def initials_can_spell(words: List[str], acronym: str) -> bool:
    """Check whether the initial letters of `words` (in order) form the
    acronym as a subsequence (case-insensitive), matching the paper's
    'sequence of characters ... can form the upper-cased characters' rule."""
    target = re.sub(r"[^A-Za-z0-9]", "", acronym).lower()
    if not target:
        return False
    initials = "".join(w[0].lower() for w in words if w and w[0].isalnum())
    # subsequence check: every char of target must appear in order in initials
    i = 0
    for ch in initials:
        if i < len(target) and ch == target[i]:
            i += 1
    return i == len(target)


@dataclass
class Span:
    start: int
    end: int  # inclusive
    label: str  # "short" or "long"


def find_long_form(tokens: List[str], acr_idx: int, acronym: str) -> Optional[Tuple[int, int]]:
    """Search before and after the acronym for a candidate long-form span."""
    w = window_size(acronym)

    # try candidate spans of length 1..w, both before and after acr_idx
    for direction in ("after", "before"):
        for length in range(1, w + 1):
            if direction == "after":
                start = acr_idx + 1
                end = start + length - 1
            else:
                end = acr_idx - 1
                start = end - length + 1
            if start < 0 or end >= len(tokens) or start > end:
                continue
            span_tokens = tokens[start:end + 1]
            if initials_can_spell(span_tokens, acronym):
                return (start, end)
    return None


def predict_sentence(tokens: List[str]) -> List[str]:
    """Return BIO tags: B-long/I-long/B-short/I-short/O."""
    labels = ["O"] * len(tokens)

    for i, tok in enumerate(tokens):
        stripped = tok.strip("()")
        if not is_acronym_candidate(stripped):
            continue
        if labels[i] != "O":
            continue
        span = find_long_form(tokens, i, stripped)
        labels[i] = "B-short"
        if span:
            start, end = span
            if all(labels[j] == "O" for j in range(start, end + 1)):
                labels[start] = "B-long"
                for j in range(start + 1, end + 1):
                    labels[j] = "I-long"
    return labels


def run(data_path: str, out_path: str = "preds_ai_rule.json"):
    with open(data_path) as f:
        data = json.load(f)

    # accepts either a list of {"id","tokens",...} or {"id": {...}} dict
    items = data if isinstance(data, list) else list(data.values())

    predictions = []
    for item in items:
        tokens = item["tokens"]
        pred_labels = predict_sentence(tokens)
        predictions.append({"id": item.get("id"), "tokens": tokens, "pred_labels": pred_labels})

    with open(out_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Wrote {len(predictions)} predictions to {out_path}")
    print("Now run: python evaluate.py --predictions", out_path, "--gold", data_path, "--task ai")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to SciAI-format json (list of {id,tokens,labels})")
    ap.add_argument("--out", default="preds_ai_rule.json")
    args = ap.parse_args()
    run(args.data, args.out)
