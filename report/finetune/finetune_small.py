# ===== finetune_small.py =====
# ByT5-small 전체 언어 학습
# 12 epochs, 체크포인트 저장 안 함, ERR 측정 + TensorBoard

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
login(token="")  # 토큰 바꿔줘

MODEL_NAME  = "google/byt5-small"
MAX_LEN     = 128
BATCH_SIZE  = 256
NUM_EPOCHS  = 12
LR          = 5e-4

OUTPUT_DIR  = "./byt5-small-finetuned"
RESULTS_DIR = "./small_results"
LOG_DIR     = "./runs/logs_small"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ===== 2. ERR Callback =====
class ERRCallback(TrainerCallback):
    def __init__(self, model, tokenizer, val_data, results_dir):
        self.model       = model
        self.tokenizer   = tokenizer
        self.val_data    = val_data
        self.results_dir = results_dir
        self.history     = []

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        err, lang_stats = compute_err(self.model, self.tokenizer,
                                      self.val_data, f"small/ep{epoch}")
        self.history.append({'epoch': epoch, 'err': err, 'lang_stats': lang_stats})
        path = os.path.join(self.results_dir, "err_history_small.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        print(f"\n[ERR] small epoch {epoch}: ERR = {err:.2f}")

# ===== 3. ERR 계산 =====
def compute_err(model, tokenizer, val_data, label):
    model.eval()
    TP = FP = FN = 0
    lang_stats = {}

    for i in range(0, len(val_data), BATCH_SIZE):
        batch  = val_data[i:i+BATCH_SIZE]
        inputs = tokenizer(
            batch['raw'], return_tensors="pt",
            max_length=MAX_LEN, truncation=True, padding=True
        ).to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for raw, norm, pred, lang in zip(batch['raw'], batch['norm'], preds, batch['lang']):
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
    return ERR, {lang: {**s, 'err': (s['TP']-s['FP'])/(s['TP']+s['FN'])*100
                        if (s['TP']+s['FN']) > 0 else 0}
                 for lang, s in lang_stats.items()}

# ===== 4. 데이터 로드 =====
print("Loading dataset...")
dataset = load_dataset("weerayut/multilexnorm2026-dev-pub")

train_data = Dataset.from_list([
    {'raw': r, 'norm': n, 'lang': ex['lang']}
    for ex in dataset['train']
    for r, n in zip(ex['raw'], ex['norm'])
])
val_data = Dataset.from_list([
    {'raw': r, 'norm': n, 'lang': ex['lang']}
    for ex in dataset['validation']
    for r, n in zip(ex['raw'], ex['norm'])
])
print(f"  train: {len(train_data)}, val: {len(val_data)}")

# ===== 5. Tokenize =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(examples):
    inputs = tokenizer(examples['raw'], max_length=MAX_LEN, truncation=True, padding=False)
    labels = tokenizer(examples['norm'], max_length=MAX_LEN, truncation=True, padding=False)
    inputs['labels'] = labels['input_ids']
    return inputs

print("Tokenizing...")
train_tok = train_data.map(preprocess, batched=True, remove_columns=['raw', 'norm', 'lang'])
val_tok   = val_data.map(preprocess, batched=True, remove_columns=['raw', 'norm', 'lang'])

# ===== 6. 모델 =====
print(f"\nLoading {MODEL_NAME}...")
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
model = model.to("cuda")

err_callback = ERRCallback(model, tokenizer, val_data, RESULTS_DIR)

# ===== 7. 학습 설정 =====
args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,      # small이라 batch 크게
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,        # small이라 불필요
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    bf16=True,
    eval_strategy="epoch",
    save_strategy="no",                  # 체크포인트 저장 안 함
    predict_with_generate=True,
    generation_max_length=MAX_LEN,
    logging_steps=100,
    logging_dir=LOG_DIR,
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
    callbacks=[err_callback],
)

# ===== 8. 학습 =====
print(f"\nTraining ByT5-small ({NUM_EPOCHS} epochs, no checkpoint saving)...")
trainer.train()

# ===== 9. 최종 모델 저장 =====
final_path = "./byt5-small-final"
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
print(f"\nSaved to {final_path}")

# ===== 10. 최종 ERR =====
err_final, _ = compute_err(model, tokenizer, val_data, "small_final")
with open(f"{RESULTS_DIR}/results_small.json", 'w', encoding='utf-8') as f:
    json.dump({'final_err': err_final, 'epoch_history': err_callback.history}, f,
              ensure_ascii=False, indent=2)

print(f"\n✅ Done! Final ERR: {err_final:.2f}")
print(f"   TensorBoard: tensorboard --logdir ./runs/logs_small")
