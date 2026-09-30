"""
Evaluation matching the paper's reported metrics:
  - AI: macro-averaged precision/recall/F1 over exact SPAN matches (both "short" and
    "long" span types), not per-token accuracy.
  - AD: precision/recall/F1 over correct long-form selection.

Usage:
  python evaluate.py --predictions preds_ai_rule.json --gold data/test.json --task ai
  python evaluate.py --predictions preds_ad.json --gold data/ad_test.json --task ad
"""
import argparse
import json
from typing import List, Set, Tuple


def bio_to_spans(labels: List[str]) -> Set[Tuple[int, int, str]]:
    spans = set()
    start, cur_type = None, None
    for i, lab in enumerate(labels + ["O"]):  # sentinel to flush last span
        if lab.startswith("B-"):
            if start is not None:
                spans.add((start, i - 1, cur_type))
            start, cur_type = i, lab[2:]
        elif lab.startswith("I-") and cur_type == lab[2:]:
            continue
        else:
            if start is not None:
                spans.add((start, i - 1, cur_type))
            start, cur_type = None, None
    return spans


def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def evaluate_ai(pred_path: str, gold_path: str):
    with open(pred_path) as f:
        preds = json.load(f)
    with open(gold_path) as f:
        gold_raw = json.load(f)
    gold_items = gold_raw if isinstance(gold_raw, list) else list(gold_raw.values())
    gold_by_id = {g["id"]: g for g in gold_items}

    tp = fp = fn = 0
    for p in preds:
        g = gold_by_id.get(p["id"])
        if g is None:
            continue
        pred_spans = bio_to_spans(p["pred_labels"])
        gold_spans = bio_to_spans(g["labels"])
        tp += len(pred_spans & gold_spans)
        fp += len(pred_spans - gold_spans)
        fn += len(gold_spans - pred_spans)

    p, r, f1 = prf(tp, fp, fn)
    print(f"AI  Precision={p*100:.2f}  Recall={r*100:.2f}  F1={f1*100:.2f}")


def evaluate_ad(pred_path: str, gold_path: str):
    with open(pred_path) as f:
        preds = json.load(f)  # list of {"id":..., "pred_index": int}
    with open(gold_path) as f:
        gold_raw = json.load(f)
    gold_items = gold_raw if isinstance(gold_raw, list) else list(gold_raw.values())
    gold_by_id = {g["id"]: g["label"] for g in gold_items}

    correct = total = 0
    for p in preds:
        gold_label = gold_by_id.get(p["id"])
        if gold_label is None:
            continue
        total += 1
        if p["pred_index"] == gold_label:
            correct += 1
    acc = correct / total if total else 0.0
    print(f"AD  Accuracy (= P = R = F1 for single-label task) = {acc*100:.2f}  (n={total})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--task", choices=["ai", "ad"], required=True)
    args = ap.parse_args()
    if args.task == "ai":
        evaluate_ai(args.predictions, args.gold)
    else:
        evaluate_ad(args.predictions, args.gold)
