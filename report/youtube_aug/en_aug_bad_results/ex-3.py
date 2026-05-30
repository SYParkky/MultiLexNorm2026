# ===== ex-3.py =====
# English wiki augmentation 실험
# A: en 원본만 4ep
# B: en 원본 + wiki_aug_en 셔플 4ep

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
    TrainerCallback,
)

# ===== 1. Config =====
login(token=" ")  # 토큰 바꿔줘

MODEL_NAME  = "google/byt5-base"
MAX_LEN     = 128
BATCH_SIZE  = 256
NUM_EPOCHS  = 4
LR          = 5e-4
WIKI_EN_AUG = "./data/english_normalized_7500.jsonl"

DIR_A       = "./ex3_A-2"
DIR_B       = "./ex3_B-2"
RESULTS_DIR = "./ex3_results-2(youtub`;e_comments.json)"
LOG_A       = "./runs/logs_ex3_A-2"
LOG_B       = "./runs/logs_ex3_B-2"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ===== 2. ERR Callback =====
class ERRCallback(TrainerCallback):
    def __init__(self, model, tokenizer, val_data, label, results_dir):
        self.model       = model
        self.tokenizer   = tokenizer
        self.val_data    = val_data
        self.label       = label
        self.results_dir = results_dir
        self.history     = []

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        err, _ = compute_err(self.model, self.tokenizer, self.val_data,
                              f"{self.label}/ep{epoch}")
        self.history.append({'epoch': epoch, 'err': err})
        path = os.path.join(self.results_dir, f"err_history_{self.label}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        print(f"\n[ERR] {self.label} epoch {epoch}: ERR = {err:.2f}")

# ===== 3. ERR 계산 =====
def compute_err(model, tokenizer, val_data, label):
    model.eval()
    TP = FP = FN = 0
    results = []

    for i in range(0, len(val_data), BATCH_SIZE):
        batch  = val_data[i:i+BATCH_SIZE]
        inputs = tokenizer(
            batch['raw'], return_tensors="pt",
            max_length=MAX_LEN, truncation=True, padding=True
        ).to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for raw, norm, pred in zip(batch['raw'], batch['norm'], preds):
            results.append({'raw': raw, 'gold': norm, 'pred': pred})
            if raw == norm:
                if pred != raw:  FP += 1
            else:
                if pred == norm: TP += 1
                else:            FN += 1

    ERR = (TP - FP) / (TP + FN) * 100 if (TP + FN) > 0 else 0
    print(f"[{label}] ERR={ERR:.2f}  TP={TP} FP={FP} FN={FN}")
    model.train()
    return ERR, results

# ===== 4. 데이터 로드 =====
def load_en_original():
    ds = load_dataset("weerayut/multilexnorm2026-dev-pub")
    train = Dataset.from_list([
        {'raw': r, 'norm': n}
        for ex in ds['train'] if ex['lang'] == 'en'
        for r, n in zip(ex['raw'], ex['norm'])
    ])
    val = Dataset.from_list([
        {'raw': r, 'norm': n}
        for ex in ds['validation'] if ex['lang'] == 'en'
        for r, n in zip(ex['raw'], ex['norm'])
    ])
    print(f"  en original — train: {len(train)}, val: {len(val)}")
    return train, val

def load_wiki_en():
    rows = []
    with open(WIKI_EN_AUG, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows.append({'raw': item['raw'], 'norm': item['norm']})
    ds = Dataset.from_list(rows)
    print(f"  wiki_aug_en: {len(ds)}")
    return ds

# ===== 5. Tokenize =====
def tokenize_data(data, tokenizer):
    def preprocess(examples):
        inputs = tokenizer(examples['raw'], max_length=MAX_LEN, truncation=True, padding=False)
        labels = tokenizer(examples['norm'], max_length=MAX_LEN, truncation=True, padding=False)
        inputs['labels'] = labels['input_ids']
        return inputs
    cols = [c for c in data.column_names if c in ['raw', 'norm']]
    return data.map(preprocess, batched=True, remove_columns=cols)

# ===== 6. 학습 =====
def train_model(train_tok, val_tok, output_dir, log_dir, tokenizer, err_callback=None):
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    model = model.to("cuda")

    if err_callback:
        err_callback.model = model

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=NUM_EPOCHS,
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
        callbacks=[err_callback] if err_callback else [],
    )
    trainer.train()
    return model

def save_results(label, err, history, results):
    with open(f"{RESULTS_DIR}/results_{label}.json", 'w', encoding='utf-8') as f:
        json.dump({'final_err': err, 'epoch_history': history,
                   'config': {'lr': LR, 'epochs': NUM_EPOCHS}},
                  f, ensure_ascii=False, indent=2)
    with open(f"{RESULTS_DIR}/predictions_{label}.jsonl", 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

# ===== 7. Main =====
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\nLoading data...")
    en_train, en_val = load_en_original()
    wiki_en          = load_wiki_en()
    val_tok          = tokenize_data(en_val, tokenizer)

    # ── A: en 원본만 ──────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print(f"A: en original only ({len(en_train)} samples)")
    print("="*50)
    cb_A    = ERRCallback(None, tokenizer, en_val, "A", RESULTS_DIR)
    train_A = tokenize_data(en_train, tokenizer)
    model_A = train_model(train_A, val_tok, DIR_A, LOG_A, tokenizer, err_callback=cb_A)
    err_A, results_A = compute_err(model_A, tokenizer, en_val, "A_final")
    save_results("A", err_A, cb_A.history, results_A)
    del model_A; torch.cuda.empty_cache()

    # ── B: en 원본 + wiki 셔플 ───────────────────────────────────────────────
    print("\n" + "="*50)
    print(f"B: en original ({len(en_train)}) + wiki_aug_en ({len(wiki_en)}) shuffled")
    print("="*50)
    merged_B = concatenate_datasets([en_train, wiki_en]).shuffle(seed=42)
    print(f"  merged: {len(merged_B)} samples")
    cb_B     = ERRCallback(None, tokenizer, en_val, "B", RESULTS_DIR)
    train_B  = tokenize_data(merged_B, tokenizer)
    model_B  = train_model(train_B, val_tok, DIR_B, LOG_B, tokenizer, err_callback=cb_B)
    err_B, results_B = compute_err(model_B, tokenizer, en_val, "B_final")
    save_results("B", err_B, cb_B.history, results_B)
    del model_B; torch.cuda.empty_cache()

    # ── 최종 비교 ─────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("FINAL COMPARISON")
    print("="*50)
    print(f"  A (en only):              ERR = {err_A:.2f}")
    print(f"  B (en + wiki_aug_en):     ERR = {err_B:.2f}")
    print(f"  Improvement:              {err_B - err_A:+.2f}")
    print(f"\n✅ Results saved to {RESULTS_DIR}/")
    print(f"   TensorBoard: tensorboard --logdir ./runs")
