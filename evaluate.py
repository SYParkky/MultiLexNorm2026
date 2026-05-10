"""
evaluate.py
로컬에서 ERR 점수 확인 (토큰 단위)
Usage: python evaluate.py
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── Config ───────────────────────────────────────────────
MODEL_DIR      = "./byt5-multilexnorm"
MAX_INPUT_LEN  = 64
MAX_TARGET_LEN = 64
BATCH_SIZE     = 128
# ─────────────────────────────────────────────────────────

LANGS = ["en", "de", "nl", "es", "tr", "id", "th", "vi", "ko", "ja",
         "da", "sl", "sr", "hr", "bg", "ar", "hi"]


def predict_tokens(model, tokenizer, tokens, device):
    results = []
    for i in range(0, len(tokens), BATCH_SIZE):
        batch = tokens[i: i + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_TARGET_LEN,
                num_beams=4,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded)
    return results


def evaluate(raw, gold, pred):
    total, correct, changed_total, changed_correct = 0, 0, 0, 0

    for raw_sent, gold_sent, pred_sent in zip(raw, gold, pred):
        for r, g, p in zip(raw_sent, gold_sent, pred_sent):
            total += 1
            if p == g:
                correct += 1
            if r != g:
                changed_total += 1
                if p == g:
                    changed_correct += 1

    accuracy     = correct / total if total > 0 else 0
    baseline_acc = (total - changed_total) / total if total > 0 else 0
    err          = changed_correct / changed_total if changed_total > 0 else 0

    print(f"Baseline acc.(LAI): {baseline_acc * 100:.2f}")
    print(f"Accuracy:           {accuracy * 100:.2f}")
    print(f"ERR:                {err * 100:.2f}")
    return baseline_acc, accuracy, err


def main():
    print("=== 모델 로딩 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    print("=== 데이터 로딩 (validation) ===")
    pub_data = load_dataset("weerayut/multilexnorm2026-pub")
    val_data = pub_data["validation"]

    all_raw, all_gold, all_pred = [], [], []

    for lang in LANGS:
        lang_data = val_data.filter(lambda x: x["lang"] == lang)
        if len(lang_data) == 0:
            continue

        print(f"[{lang}] {len(lang_data)}개 문장 추론 중...")

        for row in lang_data:
            raw_tokens  = row["raw"]
            gold_tokens = row["norm"]
            pred_tokens = predict_tokens(model, tokenizer, raw_tokens, device)

            all_raw.append(raw_tokens)
            all_gold.append(gold_tokens)
            all_pred.append(pred_tokens)

    print("\n=== 전체 평가 결과 ===")
    evaluate(all_raw, all_gold, all_pred)


if __name__ == "__main__":
    main()
