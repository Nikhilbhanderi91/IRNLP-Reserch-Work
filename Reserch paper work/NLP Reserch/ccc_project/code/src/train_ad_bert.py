"""
Fine-tune BERT for Acronym Disambiguation using the DeepBlueAI (SOTA) formulation:
one binary classification example per (sentence, candidate-long-form) pair.

Usage:
    python train_ad_bert.py --model_name bert-base-uncased --train data/ad_train.json --dev data/ad_dev.json
"""
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from dataset import SciADDataset


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[START]", "[END]"]})

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    train_ds = SciADDataset(args.train, tokenizer, max_length=args.max_length)
    dev_ds = SciADDataset(args.dev, tokenizer, max_length=args.max_length)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(**batch).logits.squeeze(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={total_loss/len(train_dl):.4f}")
        evaluate_loss(model, dev_dl, device, loss_fn)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved model to {args.output_dir}")


@torch.no_grad()
def evaluate_loss(model, dl, device, loss_fn):
    model.eval()
    total = 0.0
    for batch in dl:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits.squeeze(-1)
        total += loss_fn(logits, labels).item()
    print(f"  dev_loss={total/len(dl):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--train", default="data/ad_train.json")
    ap.add_argument("--dev", default="data/ad_dev.json")
    ap.add_argument("--output_dir", default="checkpoints/ad_model")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=160)
    train(ap.parse_args())
