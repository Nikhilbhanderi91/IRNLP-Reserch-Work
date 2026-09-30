"""
Fine-tune BERT/SciBERT/RoBERTa for Acronym Identification (token classification).

Baseline usage:
    python train_ai_bert.py --model_name bert-base-uncased --train data/train.json --dev data/dev.json

SciBERT (recommended, closest to paper's strong systems):
    python train_ai_bert.py --model_name allenai/scibert_scivocab_uncased --train data/train.json --dev data/dev.json

With adversarial training (toward AT-BERT-E style, ~+1-2 F1 typically):
    python train_ai_bert.py --model_name allenai/scibert_scivocab_uncased --adversarial --train data/train.json --dev data/dev.json
"""
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup

from dataset import SciAIDataset, AI_LABELS, LABEL2ID, ID2LABEL


class FGM:
    """Fast Gradient Method adversarial training (Miyato et al. 2017), the core
    trick behind AT-BERT-E: perturb the embedding layer in the direction of the
    loss gradient, then take one extra optimizer step on the perturbed input."""

    def __init__(self, model, emb_name="word_embeddings", epsilon=1.0):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_adv = self.epsilon * param.grad / norm
                    param.data.add_(r_adv)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=len(AI_LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    ).to(device)

    train_ds = SciAIDataset(args.train, tokenizer, max_length=args.max_length)
    dev_ds = SciAIDataset(args.dev, tokenizer, max_length=args.max_length)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    fgm = FGM(model) if args.adversarial else None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()

            if fgm is not None:
                fgm.attack()
                out_adv = model(**batch)
                out_adv.loss.backward()
                fgm.restore()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()

        print(f"epoch {epoch+1}/{args.epochs}  train_loss={total_loss/len(train_dl):.4f}")
        evaluate_loss(model, dev_dl, device)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved model to {args.output_dir}")


@torch.no_grad()
def evaluate_loss(model, dl, device):
    model.eval()
    total = 0.0
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total += out.loss.item()
    print(f"  dev_loss={total/len(dl):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--train", default="data/train.json")
    ap.add_argument("--dev", default="data/dev.json")
    ap.add_argument("--output_dir", default="checkpoints/ai_model")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--adversarial", action="store_true", help="enable FGM adversarial training (AT-BERT-E style)")
    train(ap.parse_args())
