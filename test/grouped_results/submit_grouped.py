# ===== submit_grouped.py =====
# 언어 그룹별 모델로 test split 예측 → predictions.json → zip → CodaBench

from transformers import T5ForConditionalGeneration, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import login
import pandas as pd
import torch
import os
import zipfile
import json

# ===== 1. Login =====
login(token="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")  # 토큰 바꿔줘

# ===== 2. Config =====
BATCH_SIZE = 256
MAX_LEN    = 128
BASE_DIR   = "./grouped_models"
SAVE_PATH  = "./outputs/submission_dev"

LANG_GROUPS = {
    "germanic": ["en", "de", "nl", "da"],
    "slav":     ["hr", "sl", "sr"],
    "sea":      ["id", "vi", "iden"],
    "ja":       ["ja"],
    "ko":       ["ko"],
    "th":       ["th"],
    "romance":  ["es", "it"],
    "turkic":   ["tr", "trde"],
}

# 언어 → 그룹 매핑
LANG_TO_GROUP = {
    lang: group
    for group, langs in LANG_GROUPS.items()
    for lang in langs
}

# ===== 3. Data Load =====
print("Loading dataset...")
data = load_dataset("weerayut/multilexnorm2026-dev-pub")
test_df = data['test'].to_pandas()
print(f"  test: {len(test_df)} sentences")
print(f"  languages: {sorted(test_df['lang'].unique())}")

# ===== 4. Batch Prediction =====
def normalize_sent(raw_sent, model, tokenizer):
    pred_sent = []
    for i in range(0, len(raw_sent), BATCH_SIZE):
        batch = raw_sent[i:i+BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            max_length=MAX_LEN,
            truncation=True,
            padding=True
        ).to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        pred_sent.extend(preds)
    return pred_sent

# ===== 5. Group-based Inference =====
print("\nRunning grouped inference...")
tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")

# 결과 저장용
all_preds = {}  # idx → pred

for group_name, langs in LANG_GROUPS.items():
    # 해당 그룹 언어 문장만 필터
    mask = test_df['lang'].isin(langs)
    group_df = test_df[mask]

    if len(group_df) == 0:
        print(f"[{group_name}] No test sentences — skipping")
        continue

    # 모델 로드
    model_path = os.path.join(BASE_DIR, group_name, "final")
    if not os.path.exists(model_path):
        # final 없으면 마지막 체크포인트
        ckpts = sorted([
            d for d in os.listdir(os.path.join(BASE_DIR, group_name))
            if d.startswith('checkpoint-')
        ], key=lambda x: int(x.split('-')[1]))
        if not ckpts:
            print(f"[{group_name}] ⚠️  No model found — skipping")
            continue
        model_path = os.path.join(BASE_DIR, group_name, ckpts[-1])

    print(f"\n[{group_name}] Loading from {model_path}...")
    print(f"  sentences: {len(group_df)} ({', '.join(langs)})")

    model = T5ForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    )
    model = model.to("cuda")
    model.eval()

    # 예측
    done = 0
    for idx, row in group_df.iterrows():
        pred = normalize_sent(list(row['raw']), model, tokenizer)
        all_preds[idx] = pred
        done += 1
        if done % 500 == 0:
            print(f"  {done}/{len(group_df)} ({done/len(group_df)*100:.1f}%)")

    print(f"  ✅ {done} sentences done")

    del model
    torch.cuda.empty_cache()

# ===== 6. Merge predictions =====
test_df['pred'] = test_df.index.map(lambda i: all_preds.get(i, test_df.loc[i, 'raw']))

# ===== 7. Save predictions.json =====
os.makedirs(SAVE_PATH, exist_ok=True)
out = test_df[['raw', 'norm', 'lang', 'pred']]
out.to_json(f"{SAVE_PATH}/predictions.json", orient="records")
print(f"\nSaved: {SAVE_PATH}/predictions.json")

# sanity check
print("\nSample (first 2 per group):")
for group_name, langs in LANG_GROUPS.items():
    sample = out[out['lang'].isin(langs)].head(1)
    for _, row in sample.iterrows():
        print(f"  [{row['lang']}] raw:{row['raw'][:3]} → pred:{row['pred'][:3]}")

# ===== 8. Zip =====
zip_path = f"{SAVE_PATH}.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(f"{SAVE_PATH}/predictions.json", arcname="predictions.json")
print(f"\nCreated: {zip_path}")
print(f"\n✅ Upload to CodaBench:")
print(f"   https://www.codabench.org/competitions/14162/?secret_key=33d4b8ec-4951-478b-8132-474e458409c3")
