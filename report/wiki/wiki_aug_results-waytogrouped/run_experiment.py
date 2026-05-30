# ===== run_experiment.py =====
# 실험 A/B/C 비교
# A: 원본만 4ep, lr=5e-4
# B: wiki+원본 합쳐서 4ep, lr=5e-4
# C: wiki 사전학습 3ep → 원본 파인튜닝 4ep, lr=5e-4 (논문 방식)

import os
import json
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from huggingface_hub import login
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

# ===== 1. Config =====
login(token="hf_")  # 토큰 바꿔줘

MODEL_NAME   = "google/byt5-base"
MAX_LEN      = 128
BATCH_SIZE   = 256
NUM_EPOCHS   = 4
LR           = 5e-4
TARGET_LANGS = ["ko", "en", "ja", "de"]
WIKI_AUG     = "./data/wiki_aug.jsonl"

DIR_A        = "./exp_A"
DIR_B        = "./exp_B"
DIR_C_PRE    = "./exp_C_pretrain"
DIR_C_FT     = "./exp_C_finetune"
LOG_A        = "./runs/logs_exp_A"
LOG_B        = "./runs/logs_exp_B"
LOG_C_PRE    = "./runs/logs_exp_C_pre"
LOG_C_FT     = "./runs/logs_exp_C_ft"
RESULTS_DIR  = "./experiment_results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ===== 2. 데이터 로드 =====
def build_original(split):
    dataset = load_dataset("weerayut/multilexnorm2026-dev-pub")
    rows = [
        {'raw': r, 'norm': n, 'lang': ex['lang']}
        for ex in dataset[split]
        if ex['lang'] in TARGET_LANGS
        for r, n in zip(ex['raw'], ex['norm'])
    ]
    print(f"  original {split}: {len(rows)}")
    return Dataset.from_list(rows)

def build_wiki():
    if not os.path.exists(WIKI_AUG):
        raise FileNotFoundError(f"wiki_aug.jsonl not found: {WIKI_AUG}")
    rows = []
    with open(WIKI_AUG, encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if item['lang'] in TARGET_LANGS:
                rows.append({'raw': item['raw'], 'norm': item['norm'], 'lang': item['lang']})
    print(f"  wiki_aug: {len(rows)}")
    return Dataset.from_list(rows)

# ===== 3. Tokenize =====
def tokenize_data(data, tokenizer):
    def preprocess(examples):
        inputs = tokenizer(examples['raw'], max_length=MAX_LEN, truncation=True, padding=False)
        labels = tokenizer(examples['norm'], max_length=MAX_LEN, truncation=True, padding=False)
        inputs['labels'] = labels['input_ids']
        return inputs
    return data.map(preprocess, batched=True, remove_columns=['raw', 'norm', 'lang'])

# ===== 4. 학습 =====
def train_model(train_tok, val_tok, output_dir, log_dir, tokenizer,
                num_epochs=NUM_EPOCHS, lr=LR, resume_from=None):
    model = T5ForConditionalGeneration.from_pretrained(
        resume_from if resume_from else MODEL_NAME,
        torch_dtype=torch.bfloat16
    )
    model = model.to("cuda")

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=num_epochs,
        load_best_model_at_end=False,
        predict_with_generate=True,
        generation_max_length=MAX_LEN,
        logging_steps=100,
        logging_dir=log_dir,
        report_to="tensorboard",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
    )
    trainer.train()
    return model

# ===== 5. ERR 평가 =====
def compute_err(model, tokenizer, val_data, label):
    model.eval()
    TP = FP = FN = 0
    results = []
    lang_stats = {}

    for i in range(0, len(val_data), BATCH_SIZE):
        batch = val_data[i:i+BATCH_SIZE]
        inputs = tokenizer(
            batch['raw'], return_tensors="pt",
            max_length=MAX_LEN, truncation=True, padding=True
        ).to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for raw, norm, pred, lang in zip(batch['raw'], batch['norm'], preds, batch['lang']):
            results.append({'raw': raw, 'gold': norm, 'pred': pred, 'lang': lang})
            if lang not in lang_stats:
                lang_stats[lang] = {'TP': 0, 'FP': 0, 'FN': 0}
            if raw == norm:
                if pred != raw:
                    FP += 1; lang_stats[lang]['FP'] += 1
            else:
                if pred == norm:
                    TP += 1; lang_stats[lang]['TP'] += 1
                else:
                    FN += 1; lang_stats[lang]['FN'] += 1

    ERR = (TP - FP) / (TP + FN) * 100 if (TP + FN) > 0 else 0
    print(f"\n[{label}] ERR: {ERR:.2f}  (TP={TP}, FP={FP}, FN={FN})")
    for lang in sorted(lang_stats):
        s = lang_stats[lang]
        d = s['TP'] + s['FN']
        e = (s['TP'] - s['FP']) / d * 100 if d > 0 else 0
        print(f"  {lang:6s}: ERR={e:6.2f}  (TP={s['TP']}, FP={s['FP']}, FN={s['FN']})")
    return ERR, results

def eval_checkpoints(output_dir, val_data, tokenizer, label):
    epoch_results = {}
    ckpts = sorted([
        d for d in os.listdir(output_dir)
        if d.startswith('checkpoint-')
    ], key=lambda x: int(x.split('-')[1]))

    for ckpt in ckpts:
        ckpt_path = os.path.join(output_dir, ckpt)
        model = T5ForConditionalGeneration.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16)
        model = model.to("cuda")
        err, _ = compute_err(model, tokenizer, val_data, f"{label}/{ckpt}")
        epoch_results[ckpt] = err
        del model
        torch.cuda.empty_cache()

    return epoch_results

def save_results(name, err, epoch_errs, results):
    with open(f"{RESULTS_DIR}/results_{name}.json", 'w', encoding='utf-8') as f:
        json.dump({'final_err': err, 'epoch_errs': epoch_errs,
                   'lr': LR, 'note': {
                       'A': 'original only, lr=5e-4, 4ep',
                       'B': 'wiki+original merged, lr=5e-4, 4ep',
                       'C': 'wiki pretrain(3ep) → original finetune(4ep), lr=5e-4',
                   }.get(name, '')}, f, ensure_ascii=False, indent=2)
    with open(f"{RESULTS_DIR}/predictions_{name}.jsonl", 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

# ===== 6. Main =====
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\nBuilding datasets...")
    original_train = build_original('train')
    original_val   = build_original('validation')
    wiki_data      = build_wiki()

    val_tok = tokenize_data(original_val, tokenizer)

    # ── Experiment A: 원본만, lr=5e-4, 4ep ───────────────────────────────────
    print("\n" + "="*50)
    print("EXPERIMENT A: original only | lr=5e-4 | 4ep")
    print("="*50)

    train_tok_A = tokenize_data(original_train, tokenizer)
    model_A = train_model(train_tok_A, val_tok, DIR_A, LOG_A, tokenizer,
                          num_epochs=4, lr=LR)
    err_A, results_A = compute_err(model_A, tokenizer, original_val, "A_final")
    epoch_errs_A = eval_checkpoints(DIR_A, original_val, tokenizer, "A")
    save_results("A", err_A, epoch_errs_A, results_A)
    del model_A; torch.cuda.empty_cache()

    # ── Experiment B: wiki+원본 합쳐서, lr=5e-4, 4ep ─────────────────────────
    print("\n" + "="*50)
    print("EXPERIMENT B: wiki+original merged | lr=5e-4 | 4ep")
    print("="*50)

    merged = concatenate_datasets([original_train, wiki_data])
    print(f"  merged train: {len(merged)} samples")
    train_tok_B = tokenize_data(merged, tokenizer)
    model_B = train_model(train_tok_B, val_tok, DIR_B, LOG_B, tokenizer,
                          num_epochs=4, lr=LR)
    err_B, results_B = compute_err(model_B, tokenizer, original_val, "B_final")
    epoch_errs_B = eval_checkpoints(DIR_B, original_val, tokenizer, "B")
    save_results("B", err_B, epoch_errs_B, results_B)
    del model_B; torch.cuda.empty_cache()

    # ── Experiment C: wiki 사전학습 → 원본 파인튜닝, lr=5e-4 (논문 방식) ──────
    print("\n" + "="*50)
    print("EXPERIMENT C: wiki pretrain(3ep) → original finetune(4ep) | lr=5e-4")
    print("="*50)

    # C-1: wiki 사전학습
    print("\n[C-1] Pre-training on wiki_aug (3ep, lr=5e-4)...")
    train_tok_C_pre = tokenize_data(wiki_data, tokenizer)
    model_C = train_model(train_tok_C_pre, val_tok, DIR_C_PRE, LOG_C_PRE, tokenizer,
                          num_epochs=3, lr=LR)
    pre_ckpt = os.path.join(DIR_C_PRE, "pretrained")
    model_C.save_pretrained(pre_ckpt)
    tokenizer.save_pretrained(pre_ckpt)
    del model_C; torch.cuda.empty_cache()

    # C-2: 원본 파인튜닝
    print("\n[C-2] Fine-tuning on original (4ep, lr=5e-4)...")
    train_tok_C_ft = tokenize_data(original_train, tokenizer)
    model_C = train_model(train_tok_C_ft, val_tok, DIR_C_FT, LOG_C_FT, tokenizer,
                          num_epochs=4, lr=LR, resume_from=pre_ckpt)
    err_C, results_C = compute_err(model_C, tokenizer, original_val, "C_final")
    epoch_errs_C = eval_checkpoints(DIR_C_FT, original_val, tokenizer, "C")
    save_results("C", err_C, epoch_errs_C, results_C)
    del model_C; torch.cuda.empty_cache()

    # ── 최종 비교 ─────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("FINAL COMPARISON (모든 변수 통제: lr=5e-4)")
    print("="*50)
    print(f"  A (original only):           ERR = {err_A:.2f}")
    print(f"  B (wiki+original merged):    ERR = {err_B:.2f}")
    print(f"  C (wiki pretrain → ft):      ERR = {err_C:.2f}")
    print(f"\n  B vs A: {err_B - err_A:+.2f}  ← wiki 데이터 추가 효과")
    print(f"  C vs A: {err_C - err_A:+.2f}  ← wiki 사전학습 효과")
    print(f"  C vs B: {err_C - err_B:+.2f}  ← 순차 학습 vs 합치기")
    print(f"\n✅ Results saved to {RESULTS_DIR}/")
