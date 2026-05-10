"""
predict.py
학습된 ByT5 모델로 추론 (토큰 단위)
Usage: python predict.py
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── Config ───────────────────────────────────────────────
MODEL_DIR      = "./byt5-multilexnorm"
OUTPUT_DIR     = "./outputs"
MAX_INPUT_LEN  = 64
MAX_TARGET_LEN = 64
BATCH_SIZE     = 128   # 토큰 단위라 배치 크게 가능
# ─────────────────────────────────────────────────────────

LANGS = ["en", "de", "nl", "es", "tr", "id", "th", "vi", "ko", "ja",
         "da", "sl", "sr", "hr", "bg", "ar", "hi"]


def predict_tokens(model, tokenizer, tokens, device):
    """
    토큰 리스트를 받아 정규화된 토큰 리스트 반환.
    배치로 묶어서 처리.
    """
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== 모델 로딩 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    print("=== 데이터 로딩 ===")
    pub_data = load_dataset("weerayut/multilexnorm2026-pub")

    for split in ["validation", "train"]:
        print(f"\n--- [{split}] 추론 중 ---")
        data = pub_data[split]

        all_preds = {}  # lang → list[list[str]]

        for lang in LANGS:
            lang_data = data.filter(lambda x: x["lang"] == lang)
            if len(lang_data) == 0:
                continue

            print(f"  [{lang}] {len(lang_data)}개 문장")

            pred_sentences = []
            for row in lang_data:
                raw_tokens = row["raw"]
                # 토큰 하나씩 모델에 넣고 결과 받기
                pred_tokens = predict_tokens(model, tokenizer, raw_tokens, device)
                pred_sentences.append(pred_tokens)

            all_preds[lang] = pred_sentences

        # 결과 저장 (언어별 .txt 파일)
        suffix = "dev" if split == "validation" else "full"
        split_dir = os.path.join(OUTPUT_DIR, suffix)
        os.makedirs(split_dir, exist_ok=True)

        for lang, preds in all_preds.items():
            out_path = os.path.join(split_dir, f"{lang}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                for pred_tokens in preds:
                    f.write("\t".join(pred_tokens) + "\n")

        print(f"저장 완료: {split_dir}/")

    print("\n=== 추론 완료! ===")


if __name__ == "__main__":
    main()
