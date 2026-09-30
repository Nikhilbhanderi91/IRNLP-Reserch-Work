"""
HuggingFace-compatible Dataset classes for SciAI (AI task) and SciAD (AD task).

Expected file formats
----------------------
SciAI (train/dev/test.json) — list of:
    {"id": "...", "tokens": ["The","key",...], "labels": ["O","B-long",...]}

SciAD (ad_train/ad_dev/ad_test.json) — list of:
    {"id": "...", "acronym": "CNN", "sentence": ["We","use","a","CNN",...],
     "acronym_index": 3, "candidates": ["Convolutional Neural Network", "Cable News Network"],
     "label": 0}
"""
import json
from typing import Dict, List

import torch
from torch.utils.data import Dataset

AI_LABELS = ["O", "B-short", "I-short", "B-long", "I-long"]
LABEL2ID = {l: i for i, l in enumerate(AI_LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


class SciAIDataset(Dataset):
    """Token classification dataset for Acronym Identification."""

    def __init__(self, path: str, tokenizer, max_length: int = 128):
        with open(path) as f:
            raw = json.load(f)
        self.items = raw if isinstance(raw, list) else list(raw.values())
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        tokens, word_labels = item["tokens"], item.get("labels")

        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        word_ids = enc.word_ids(batch_index=0)

        label_ids = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)  # special tokens: ignored in loss
            elif wid != prev_word_id:
                label_ids.append(LABEL2ID[word_labels[wid]] if word_labels else -100)
            else:
                # sub-word continuation: convert B- to I- of same type, else ignore
                if word_labels:
                    lab = word_labels[wid]
                    if lab.startswith("B-"):
                        lab = "I-" + lab[2:]
                    label_ids.append(LABEL2ID[lab])
                else:
                    label_ids.append(-100)
            prev_word_id = wid

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


class SciADDataset(Dataset):
    """Candidate-scoring (binary classification) dataset for Acronym Disambiguation.
    One example is produced PER CANDIDATE, following the DeepBlueAI formulation:
        [CLS] candidate_long_form [SEP] ... <START> acronym <END> ... [SEP]
    """

    START_TOK, END_TOK = "[START]", "[END]"

    def __init__(self, path: str, tokenizer, max_length: int = 160):
        with open(path) as f:
            raw = json.load(f)
        items = raw if isinstance(raw, list) else list(raw.values())
        self.tokenizer = tokenizer
        self.max_length = max_length

        # flatten into one (sentence, candidate, label) row per candidate
        self.rows: List[Dict] = []
        for item in items:
            sent = list(item["sentence"])
            acr_idx = item["acronym_index"]
            marked = sent[:acr_idx] + [self.START_TOK, sent[acr_idx], self.END_TOK] + sent[acr_idx + 1:]
            marked_text = " ".join(marked)
            for cand_i, cand in enumerate(item["candidates"]):
                self.rows.append({
                    "candidate": cand,
                    "sentence": marked_text,
                    "label": 1 if cand_i == item["label"] else 0,
                    "group_id": item["id"],
                })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = self.tokenizer(
            row["candidate"],
            row["sentence"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])).squeeze(0),
            "labels": torch.tensor(row["label"], dtype=torch.float),
        }
