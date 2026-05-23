# Korean Wiki Augmentation Strategy Experiment

Experiment comparing three training strategies for Korean lexical normalization using Wikipedia-based augmentation data.

---

## Overview

This experiment investigates whether Wikipedia corruption augmentation improves Korean (ko) normalization performance, and which training strategy is most effective.

### Research Question
> Does pre-training on Wikipedia-corrupted data improve ByT5 fine-tuning performance on Korean SNS text normalization?

---

## Three Strategies

| Strategy | Description | Training Order |
|---|---|---|
| **A** | Original ko data only | ko train → evaluate |
| **B** | Merged + shuffled | (ko + wiki) shuffled → evaluate |
| **C** | Sequential (논문 방식) | wiki pretrain → ko finetune → evaluate |

All strategies use the same hyperparameters: `lr=5e-4`, `4 epochs`, `ByT5-base`.

---

## Results

### ERR per Epoch

| Epoch | A (original) | B (merged) | C (wiki→ft) |
|---|---|---|---|
| 1 | -9.64 | 1.20 | 0.00 |
| 2 | -2.41 | **5.42** | 3.61 |
| 3 | 2.41 | 0.60 | **5.42** |
| 4 | 3.61 | 1.20 | 4.82 |

### Final ERR

| Strategy | Final ERR | Best Epoch ERR |
|---|---|---|
| A (original only) | 3.61 | 3.61 (ep4) |
| B (merged) | 1.20 | 5.42 (ep2) |
| C (wiki pretrain → ft) | 4.82 | 5.42 (ep3) |

---

## Key Findings

### 1. Wiki augmentation has limited effect on Korean
All three strategies plateaued around ERR 5%, well below other languages. This is due to the structural characteristics of the Korean dataset:
- Words are **uniquely sampled** — training words rarely appear in validation
- Data source is **dcinside** (Korean online community) — heavy use of slang and profanity not present in Wikipedia

### 2. Strategy C starts at ERR=0 (ep1)
After wiki pre-training, the model is overfit to "standard → standard" patterns. It takes several fine-tuning epochs to recover performance on non-standard Korean text. This highlights the **domain gap** between Wikipedia and SNS text.

### 3. Strategy B peaks early then drops
The merged dataset causes the model to learn conflicting patterns (Wikipedia-style corruption vs real SNS slang), leading to instability after ep2.

### 4. Strategy A still rising at ep4
A's ERR is still increasing at epoch 4, suggesting more epochs may yield higher performance — unlike B and C which peak and decline.

---

## Conclusion

> Wikipedia-based augmentation is **not effective for Korean** due to the domain mismatch between Wikipedia text and dcinside slang. All three strategies converge around ERR ~5%, indicating a structural data limitation rather than a training strategy issue.

This is documented as a **limitation** in the final report.

---

## File Structure

```
strategy_results/
├── results_A.json          # A: final ERR + epoch history
├── results_B.json          # B: final ERR + epoch history
├── results_C.json          # C: final ERR + epoch history
├── err_history_A.json      # A: ERR per epoch (real-time)
├── err_history_B.json      # B: ERR per epoch (real-time)
├── err_history_C.json      # C: ERR per epoch (real-time)
├── predictions_A.jsonl     # A: raw / gold / pred pairs
├── predictions_B.jsonl     # B: raw / gold / pred pairs
└── predictions_C.jsonl     # C: raw / gold / pred pairs

runs/
├── A//logs_strategy_A/        # TensorBoard logs (Strategy A)
├── B//logs_strategy_B/        # TensorBoard logs (Strategy B)
├── c-pre//logs_strategy_C_pre/    # TensorBoard logs (C: wiki pretrain)
└── c-fine//logs_strategy_C_ft/     # TensorBoard logs (C: ko finetune)
```

---

## TensorBoard

View training loss and eval loss for all strategies:

```bash
# All strategies at once
tensorboard --logdir ./runs

# Individual strategy
tensorboard --logdir ./runs/logs_strategy_A
tensorboard --logdir ./runs/logs_strategy_B
tensorboard --logdir ./runs/logs_strategy_C_pre
tensorboard --logdir ./runs/logs_strategy_C_ft
```

Open `http://localhost:6006` in your browser.

| Metric | Tab | Description |
|---|---|---|
| `eval/loss` | SCALARS | Validation loss per epoch |
| `train/loss` | SCALARS | Training loss (should decrease) |
| `train/learning_rate` | SCALARS | LR schedule |
| `train/grad_norm` | SCALARS | Gradient stability |

> ⚠️ Note: `eval/loss` and ERR do not always move together. Always check ERR directly using `err_history_*.json`.

---

## Scripts

| Script | Description |
|---|---|
| `strategy_experiment.py` | Run all 3 strategies (A → B → C) |
| `strategy_C_only.py` | Run Strategy C only (skips if pretrained model exists) |

```bash
# Run all
python strategy_experiment.py

```

---

## Config

```python
MODEL_NAME = "google/byt5-base"
NUM_EPOCHS = 4
LR         = 5e-4
WIKI_AUG   = "./data/wiki_aug.jsonl"   # ko only (1000 pairs)
```
