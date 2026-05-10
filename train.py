"""
train.py
ByT5 fine-tuning for MultiLexNorm2026 (token-level approach)
Usage: python train.py
"""

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

# ── Config ───────────────────────────────────────────────
MODEL_NAME     = "google/byt5-small"
OUTPUT_DIR     = "./byt5-multilexnorm"
MAX_INPUT_LEN  = 64    # 토큰 하나라서 짧아도 충분
MAX_TARGET_LEN = 64
BATCH_SIZE     = 64    # 토큰 단위라 샘플 수 많아짐 → 배치 크게
NUM_EPOCHS     = 3
LR             = 5e-4
# ─────────────────────────────────────────────────────────

LANGS = ["en", "de", "nl", "es", "tr", "id", "th", "vi", "ko", "ja",
         "da", "sl", "sr", "hr", "bg", "ar", "hi"]


def build_token_pairs(split):
    """
    데이터셋에서 (raw 토큰, norm 토큰) 쌍을 추출.
    raw == norm인 경우(정규화 불필요)도 포함 —
    모델이 '건드리지 말아야 할 것'도 학습하게.
    """
    pub_data = load_dataset("weerayut/multilexnorm2026-dev-pub")
    data = pub_data[split]

    inputs, targets = [], []
    for row in data:
        if row["lang"] not in LANGS:
            continue
        for raw_tok, norm_tok in zip(row["raw"], row["norm"]):
            inputs.append(raw_tok)
            targets.append(norm_tok)

    return inputs, targets


def main():
    print("=== 데이터 로딩 ===")
    train_inputs, train_targets = build_token_pairs("train")
    val_inputs,   val_targets   = build_token_pairs("validation")
    print(f"Train 토큰 쌍: {len(train_inputs)}개")
    print(f"Val   토큰 쌍: {len(val_inputs)}개")

    print("=== 토크나이저 & 모델 로딩 ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    print("=== 토크나이징 ===")
    train_ds = Dataset.from_dict({"input": train_inputs, "target": train_targets})
    val_ds   = Dataset.from_dict({"input": val_inputs,   "target": val_targets})

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["input"],
            max_length=MAX_INPUT_LEN,
            truncation=True,
        )
        labels = tokenizer(
            batch["target"],
            max_length=MAX_TARGET_LEN,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = train_ds.map(preprocess, batched=True, remove_columns=["input", "target"])
    val_ds   = val_ds.map(preprocess,   batched=True, remove_columns=["input", "target"])

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    print("=== 학습 시작 ===")
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=200,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"=== 학습 완료! 모델 저장: {OUTPUT_DIR} ===")


if __name__ == "__main__":
    main()
