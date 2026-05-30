# Grouped Language Models — Experiment Results

Language-specialized ByT5-base models trained per linguistic group.  
This approach outperformed the single multilingual model, achieving **41+ ERR on CodaBench**.

---

## Motivation

### Step 1: Korean wiki augmentation experiment
While running the Korean wiki augmentation experiment (Strategy A: ko original only), we observed that training on Korean alone — without other languages — outperformed the baseline ByT5-base model trained on all languages for 10 epochs. This suggested that **reducing the number of languages per model improves per-language performance**.

### Step 2: Language grouping hypothesis
Based on this finding, we hypothesized that training separate models per linguistic family would further improve performance. Languages in the same family share morphological and orthographic patterns, so a specialized model should generalize better within that group.

### Step 3: Validation
Training grouped models confirmed the hypothesis. The grouped approach achieved higher ERR on the CodaBench test set compared to the single multilingual model trained on all languages simultaneously.

> **Key insight**: When all languages share one model, they compete for model capacity. Separating by linguistic family allows each model to specialize.

---

## Language Groups

| Group | Languages | Val Available | Best ERR |
|---|---|---|---|
| germanic | en, de, nl, da | ✅ | 32.09 (ep4) |
| slav | hr, sl, sr | ✅ | 53.85 (ep7) |
| sea | id, vi, iden | ✅ | 74.14 (ep6) |
| ja | ja | ✅ | 15.67 (ep5) |
| ko | ko | ✅ | 5.42 (ep4) |
| th | th | ✅ | 11.45 (ep4) |
| romance | es, it | ❌ (no val) | N/A |
| turkic | tr, trde | ❌ (no val) | N/A |

> romance and turkic have no validation split in the dataset — performance measured only via CodaBench submission.

---

## Training Config

```python
MODEL_NAME = "google/byt5-base"
NUM_EPOCHS = 8
LR         = 5e-4
BATCH_SIZE = 16 (x2 grad accum = effective 32)
```

---

## Validation ERR per Epoch

### germanic (en, de, nl, da)
| ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | ep7 | ep8 |
|---|---|---|---|---|---|---|---|
| 18.05 | 28.05 | 31.51 | **32.09** | 31.55 | 31.13 | 31.59 | 31.47 |

### slav (hr, sl, sr)
| ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | ep7 | ep8 |
|---|---|---|---|---|---|---|---|
| 41.71 | 49.88 | 51.89 | 53.06 | 53.84 | 53.78 | **53.85** | 53.82 |

### sea (id, vi, iden)
| ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | ep7 | ep8 |
|---|---|---|---|---|---|---|---|
| 66.58 | 71.25 | 73.19 | 73.86 | 74.03 | **74.14** | 74.07 | 74.07 |

### ja
| ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | ep7 | ep8 |
|---|---|---|---|---|---|---|---|
| 8.35 | 9.08 | 10.40 | 12.01 | **15.67** | 14.20 | 15.23 | 15.08 |

### ko
| ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | ep7 | ep8 |
|---|---|---|---|---|---|---|---|
| -16.27 | -3.01 | 1.81 | **5.42** | 3.61 | 3.61 | 3.61 | 3.61 |

### th
| ep1 | ep2 | ep3 | ep4 | ep5 | ep6 | ep7 | ep8 |
|---|---|---|---|---|---|---|---|
| 3.94 | 7.25 | 8.52 | **11.45** | 11.20 | 11.07 | 11.20 | 10.69 |

---

## Per-Language Final ERR (Validation)

| Language | ERR | TP | FP | FN |
|---|---|---|---|---|
| id | 85.12 | 1808 | 57 | 249 |
| vi | 73.65 | 1708 | 137 | 425 |
| sr | 61.20 | 995 | 162 | 366 |
| sl | 54.45 | 1487 | 185 | 904 |
| hr | 46.88 | 994 | 205 | 689 |
| en | 45.66 | 435 | 146 | 198 |
| iden | 37.09 | 387 | 170 | 198 |
| de | 27.15 | 316 | 79 | 557 |
| nl | 26.72 | 388 | 96 | 705 |
| ja | 15.23 | 252 | 148 | 431 |
| th | 10.81 | 489 | 404 | 297 |
| ko | 3.61 | 31 | 25 | 135 |

**Overall validation ERR: 51.76**

---

## CodaBench Results

| Submission | ERR | Notes |
|---|---|---|
| MFR Baseline | 39.02 | Official baseline |
| Single ByT5-base (all languages, 10ep) | ~28 | Previous approach |
| Grouped models (8 groups, 17 langs) | **41+** | This experiment (CodaBench) |
| Validation ERR (grouped, 12 langs) | **51.76** | Local validation |

---

## Key Findings

**1. Language grouping significantly improves ERR**  
Training separate models per linguistic family reduces cross-language interference. Germanic, Slavic, and SEA groups all showed substantial gains over the single multilingual model.

**2. Korean (ko) and Thai (th) have structural limitations**  
- ko: unique word sampling means training words rarely appear in validation
- th: high FP (404) — model over-normalizes standard words due to only 4% non-standard rate

**3. SEA group achieves highest ERR (74+)**  
Indonesian (id: 85.12) and Vietnamese (vi: 73.65) benefit greatly from focused training, likely due to abundant and consistent training data.

**4. romance and turkic — no local validation**  
es, it, tr, trde have training data but no validation split. Performance measured only via CodaBench.

---

## File Structure

```
grouped_results/
├── grouped_summary.json          # overall ERR + per-group + per-language stats
├── summary.json                  # epoch history per group
├── err_history_germanic.json     # ERR per epoch (germanic)
├── err_history_slav.json
├── err_history_sea.json
├── err_history_ja.json
├── err_history_ko.json
├── err_history_th.json
├── predictions_germanic.jsonl    # raw / gold / pred pairs
├── predictions_slav.jsonl
├── predictions_sea.jsonl
├── predictions_ja.jsonl
├── predictions_ko.jsonl
├── predictions_th.jsonl
└── predictions_grouped.jsonl     # all groups combined

runs/
├── logs_germanic/                # TensorBoard logs
├── logs_slav/
├── logs_sea/
├── logs_romance/
├── logs_turkic/
├── logs_ja/
├── logs_ko/
└── logs_th/
```

---

## TensorBoard

View training loss and eval loss for all groups:

```bash
# All groups at once
tensorboard --logdir ./runs

# Individual group
tensorboard --logdir ./runs/logs_germanic
tensorboard --logdir ./runs/logs_slav
tensorboard --logdir ./runs/logs_sea
tensorboard --logdir ./runs/logs_ja
tensorboard --logdir ./runs/logs_ko
tensorboard --logdir ./runs/logs_th
```

Open `http://localhost:6006` in your browser.

| Metric | Tab | Description |
|---|---|---|
| `eval/loss` | SCALARS | Validation loss per epoch |
| `train/loss` | SCALARS | Training loss |
| `train/learning_rate` | SCALARS | LR schedule |
| `train/grad_norm` | SCALARS | Gradient stability |

> ⚠️ eval/loss and ERR do not always correlate — always check `err_history_*.json` for actual task performance.

---

## Scripts

| Script | Description |
|---|---|
| `finetune_grouped.py` | Train all language groups sequentially |
| `eval_grouped.py` | Evaluate all groups on validation set |
| `submit_grouped.py` | Generate predictions.json → zip for CodaBench |

```bash
# Train
python finetune_grouped.py

# Evaluate
python eval_grouped.py

# Submit
python submit_grouped.py
```
