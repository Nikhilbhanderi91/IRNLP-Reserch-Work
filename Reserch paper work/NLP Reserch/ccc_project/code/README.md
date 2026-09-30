# SciAI/SciAD Reproduction Codebase

Companion code to `../docs/00_MASTER_GUIDE.md`. Implements:
- `src/rule_based_baseline.py` — paper's rule-based AI baseline (tested, runs as-is).
- `src/dataset.py` — SciAI (token classification) + SciAD (candidate scoring) dataset loaders.
- `src/train_ai_bert.py` — fine-tune BERT/SciBERT/RoBERTa for AI, with optional `--adversarial` (FGM) flag toward AT-BERT-E.
- `src/train_ad_bert.py` — fine-tune BERT for AD using the DeepBlueAI candidate-scoring formulation.
- `src/evaluate.py` — span-level macro P/R/F1 for AI, accuracy for AD (matches paper's Table 3/4 methodology).

## 1. Get the data
Download the official SciAI/SciAD release (search "SDU AAAI-21 acronym identification
shared task github" — the `amirveyseh` org repos host `train.json/dev.json/test.json`
for AI and the AD equivalents). Place them under `data/`. `data/sample.json` shows the
exact expected format (one real example from the paper).

## 2. Rule-based baseline (no GPU needed, runs immediately)
```bash
pip install -r requirements.txt
python src/rule_based_baseline.py --data data/test.json --out preds_ai_rule.json
python src/evaluate.py --predictions preds_ai_rule.json --gold data/test.json --task ai
```
Expect roughly P≈91 R≈78 F1≈84, matching the paper's baseline row.

## 3. Transformer AI model
```bash
python src/train_ai_bert.py --model_name allenai/scibert_scivocab_uncased \
    --train data/train.json --dev data/dev.json --epochs 5 --adversarial
```
Then generate predictions with `src/infer.py` (write this next — stub described in the
guide, section 11) and re-run `evaluate.py`.

## 4. Transformer AD model
```bash
python src/train_ad_bert.py --model_name bert-base-uncased \
    --train data/ad_train.json --dev data/ad_dev.json --epochs 3
```

## Notes
- All scripts use HuggingFace `transformers` + plain PyTorch — no exotic dependencies.
- `--adversarial` in `train_ai_bert.py` implements FGM (Miyato et al. 2017), the core
  idea behind the shared task's winning AT-BERT-E system.
- Metrics are macro/span-level to match the paper exactly — token-level accuracy will
  look artificially high (O-tag dominates) and should NOT be reported as your result.
