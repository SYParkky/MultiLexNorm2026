# ===== finetune_grouped.py =====
# 언어 그룹별 특화 학습
# epoch마다 체크포인트 저장 + ERR 즉석 평가 + TensorBoard
# val 없는 그룹(romance, turkic)은 eval_strategy="no"

import os
import json
import torch
from datasets import load_dataset, Dataset
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
login(token="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")  # 토큰 바꿔줘

MODEL_NAME = "google/byt5-base"
MAX_LEN    = 128
BATCH_SIZE = 256
NUM_EPOCHS = 8

LANG_GROUPS = {
    "germanic": ["en", "de", "nl", "da"],   # da 추가 (val 있는 언어들로 평가)
    "slav":     ["hr", "sl", "sr"],
    "sea":      ["id", "vi", "iden"],
    "romance":  ["es", "it"],               # val 없음 → eval 안 함
    "turkic":   ["tr", "trde"],             # val 없음 → eval 안 함
    "ja":       ["ja"],
    "ko":       ["ko"],
    "th":       ["th"],
}

# val이 없는 그룹 (train만 있음)
NO_VAL_GROUPS = {"romance", "turkic"}

BASE_DIR    = "./grouped_models"
RESULTS_DIR = "./grouped_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===== 2. ERR Callback =====
class ERRCallback(TrainerCallback):
    def __init__(self, model, tokenizer, val_data, group_name, results_dir):
        self.model       = model
        self.tokenizer   = tokenizer
        self.val_data    = val_data
        self.group_name  = group_name
        self.results_dir = results_dir
        self.history     = []

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        err, _ = compute_err(self.model, self.tokenizer, self.val_data,
                              f"{self.group_name}/ep{epoch}")
        self.history.append({'epoch': epoch, 'err': err})
        path = os.path.join(self.results_dir, f"err_history_{self.group_name}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        print(f"\n[ERR Callback] {self.group_name} epoch {epoch}: ERR = {err:.2f}")

# ===== 3. ERR 계산 =====
def compute_err(model, tokenizer, val_data, label):
    model.eval()
    TP = FP = FN = 0
    results = []
    lang_stats = {}

    for i in range(0, len(val_data), BATCH_SIZE):
        batch = val_data[i:i+BATCH_SIZE]
        raws  = batch['raw']
        norms = batch['norm']
        langs = batch['lang']

        inputs = tokenizer(raws, return_tensors="pt", max_length=MAX_LEN,
                           truncation=True, padding=True).to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for raw, norm, pred, lang in zip(raws, norms, preds, langs):
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

    model.train()
    return ERR, results

# ===== 4. 데이터 로드 =====
def build_dataset(langs, split):
    dataset = load_dataset("weerayut/multilexnorm2026-dev-pub")
    rows = [
        {'raw': r, 'norm': n, 'lang': ex['lang']}
        for ex in dataset[split]
        if ex['lang'] in langs
        for r, n in zip(ex['raw'], ex['norm'])
    ]
    print(f"  {split} {langs}: {len(rows)} samples")
    return Dataset.from_list(rows)

# ===== 5. Tokenize =====
def tokenize_data(data, tokenizer):
    def preprocess(examples):
        inputs = tokenizer(examples['raw'], max_length=MAX_LEN, truncation=True, padding=False)
        labels = tokenizer(examples['norm'], max_length=MAX_LEN, truncation=True, padding=False)
        inputs['labels'] = labels['input_ids']
        return inputs
    return data.map(preprocess, batched=True, remove_columns=['raw', 'norm', 'lang'])

# ===== 6. 학습 =====
def train_group(group_name, langs, tokenizer):
    print(f"\n{'='*50}")
    print(f"Training: {group_name} ({', '.join(langs)})")
    print(f"{'='*50}")

    has_val    = group_name not in NO_VAL_GROUPS
    output_dir = os.path.join(BASE_DIR, group_name)
    log_dir    = f"./runs/logs_{group_name}"
    os.makedirs(output_dir, exist_ok=True)

    # 데이터
    train_data = build_dataset(langs, 'train')
    train_tok  = tokenize_data(train_data, tokenizer)

    val_data = None
    val_tok  = None
    if has_val:
        val_data = build_dataset(langs, 'validation')
        val_tok  = tokenize_data(val_data, tokenizer)
    else:
        print(f"  ⚠️  No validation data for {group_name} — skipping eval")

    # 모델
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    model = model.to("cuda")

    callbacks = []
    if has_val:
        err_callback = ERRCallback(model, tokenizer, val_data, group_name, RESULTS_DIR)
        callbacks.append(err_callback)

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=5e-4,
        bf16=True,
        eval_strategy="epoch" if has_val else "no",
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
        callbacks=callbacks,
    )

    trainer.train()

    # 최종 모델 저장
    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Saved: {final_path}")

    # ERR 측정 (val 있는 그룹만)
    err_final = None
    history   = []
    if has_val:
        err_final, results = compute_err(model, tokenizer, val_data, f"{group_name}/final")
        with open(os.path.join(RESULTS_DIR, f"predictions_{group_name}.jsonl"), 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        history = err_callback.history
    else:
        print(f"  [{group_name}] No val → ERR not measured locally. Check CodaBench.")

    del model
    torch.cuda.empty_cache()

    return err_final, history

# ===== 7. Main =====
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    summary = {}

    for group_name, langs in LANG_GROUPS.items():
        err_final, history = train_group(group_name, langs, tokenizer)
        summary[group_name] = {'final_err': err_final, 'history': history}

    with open(os.path.join(RESULTS_DIR, "summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "="*50)
    print("ALL GROUPS DONE")
    print("="*50)
    for group, data in summary.items():
        err_str = f"{data['final_err']:.2f}" if data['final_err'] is not None else "N/A (no val)"
        print(f"  {group:12s}: ERR = {err_str}")
