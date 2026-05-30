# ===== aug_experiment.py =====
# 증강 데이터 실험: A(기본) vs B(기본+증강 셔플)
# 여러 언어 연달아 실행 가능
# ================================================================
# ↓↓↓ 여기서 옵션 변경 ↓↓↓

EXPERIMENT_NAME = "exp_aug_all"   # 결과 최상위 폴더 이름

EXPERIMENTS = [
    {"lang": "de", "aug": "./data/youtube_augmented_de.jsonl"},
    {"lang": "th", "aug": "./data/youtube_augmented_th.jsonl"},
    {"lang": "ja", "aug": "./data/youtube_augmented_ja.jsonl"},
    {"lang": "ko", "aug": "./data/youtube_augmented_ko.jsonl"},
    {"lang": "en", "aug": "./data/youtube_augmented_en.jsonl"},
]

NUM_EPOCHS = 4
LR         = 5e-4

# ↑↑↑ 여기까지 ↑↑↑
# ================================================================

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

login(token=" ")  # 토큰 바꿔줘

MODEL_NAME = "google/byt5-base"
MAX_LEN    = 128
BATCH_SIZE = 256

# ===== ERR Callback =====
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

# ===== ERR 계산 =====
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

# ===== 데이터 로드 =====
def load_original(lang, ds):
    train = Dataset.from_list([
        {'raw': r, 'norm': n}
        for ex in ds['train'] if ex['lang'] == lang
        for r, n in zip(ex['raw'], ex['norm'])
    ])
    val = Dataset.from_list([
        {'raw': r, 'norm': n}
        for ex in ds['validation'] if ex['lang'] == lang
        for r, n in zip(ex['raw'], ex['norm'])
    ])
    print(f"  [{lang}] train: {len(train)}, val: {len(val)}")
    return train, val

def load_aug(path):
    rows = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows.append({'raw': item['raw'], 'norm': item['norm']})
    ds = Dataset.from_list(rows)
    print(f"  aug: {len(ds)} from {path}")
    return ds

# ===== Tokenize =====
def tokenize_data(data, tokenizer):
    def preprocess(examples):
        inputs = tokenizer(examples['raw'], max_length=MAX_LEN, truncation=True, padding=False)
        labels = tokenizer(examples['norm'], max_length=MAX_LEN, truncation=True, padding=False)
        inputs['labels'] = labels['input_ids']
        return inputs
    cols = [c for c in data.column_names if c in ['raw', 'norm']]
    return data.map(preprocess, batched=True, remove_columns=cols)

# ===== 학습 =====
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
        save_strategy="no",
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

def save_results(label, err, history, results, results_dir, lang, aug_path):
    with open(os.path.join(results_dir, f"results_{label}.json"), 'w', encoding='utf-8') as f:
        json.dump({'final_err': err, 'epoch_history': history,
                   'lang': lang, 'aug': aug_path,
                   'config': {'lr': LR, 'epochs': NUM_EPOCHS}},
                  f, ensure_ascii=False, indent=2)
    with open(os.path.join(results_dir, f"predictions_{label}.jsonl"), 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

# ===== Main =====
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"\nLoading dataset...")
    ds = load_dataset("weerayut/multilexnorm2026-dev-pub")

    all_summary = {}

    for exp in EXPERIMENTS:
        lang     = exp['lang']
        aug_path = exp['aug']

        print(f"\n{'='*50}")
        print(f"EXPERIMENT: {EXPERIMENT_NAME} / {lang}")
        print(f"Aug: {aug_path}")
        print(f"{'='*50}")

        # 폴더 생성
        lang_dir    = os.path.join(EXPERIMENT_NAME, lang)
        dir_A       = os.path.join(lang_dir, "A")
        dir_B       = os.path.join(lang_dir, "B")
        results_dir = os.path.join(lang_dir, "results")
        log_A       = os.path.join(lang_dir, "logs_A")
        log_B       = os.path.join(lang_dir, "logs_B")
        for d in [dir_A, dir_B, results_dir, log_A, log_B]:
            os.makedirs(d, exist_ok=True)

        # 데이터
        orig_train, orig_val = load_original(lang, ds)
        aug_data             = load_aug(aug_path)
        val_tok              = tokenize_data(orig_val, tokenizer)

        # A: 기본만
        print(f"\n[A] original only ({len(orig_train)} samples)")
        cb_A    = ERRCallback(None, tokenizer, orig_val, f"{lang}_A", results_dir)
        train_A = tokenize_data(orig_train, tokenizer)
        model_A = train_model(train_A, val_tok, dir_A, log_A, tokenizer, err_callback=cb_A)
        err_A, results_A = compute_err(model_A, tokenizer, orig_val, f"{lang}_A_final")
        save_results(f"{lang}_A", err_A, cb_A.history, results_A, results_dir, lang, aug_path)
        del model_A; torch.cuda.empty_cache()

        # B: 기본 + 증강 셔플
        merged = concatenate_datasets([orig_train, aug_data]).shuffle(seed=42)
        print(f"\n[B] original + aug shuffled ({len(merged)} samples)")
        cb_B    = ERRCallback(None, tokenizer, orig_val, f"{lang}_B", results_dir)
        train_B = tokenize_data(merged, tokenizer)
        model_B = train_model(train_B, val_tok, dir_B, log_B, tokenizer, err_callback=cb_B)
        err_B, results_B = compute_err(model_B, tokenizer, orig_val, f"{lang}_B_final")
        save_results(f"{lang}_B", err_B, cb_B.history, results_B, results_dir, lang, aug_path)
        del model_B; torch.cuda.empty_cache()

        all_summary[lang] = {'A': err_A, 'B': err_B, 'diff': err_B - err_A}
        print(f"\n[{lang}] A={err_A:.2f} / B={err_B:.2f} / diff={err_B-err_A:+.2f}")

    # 최종 요약
    print(f"\n{'='*50}")
    print(f"FINAL SUMMARY: {EXPERIMENT_NAME}")
    print(f"{'='*50}")
    print(f"{'lang':6} {'A':>8} {'B':>8} {'diff':>8}")
    for lang, r in all_summary.items():
        print(f"{lang:6} {r['A']:8.2f} {r['B']:8.2f} {r['diff']:+8.2f}")

    with open(os.path.join(EXPERIMENT_NAME, "summary.json"), 'w', encoding='utf-8') as f:
        json.dump(all_summary, f, ensure_ascii=False, indent=2)
    print(f"\n✅ All results saved to ./{EXPERIMENT_NAME}/")
    print(f"   TensorBoard: tensorboard --logdir ./{EXPERIMENT_NAME}")
