# ===== submit_grouped_selective.py =====
# grouped 모델 + MFR 후처리 (th/ja/ko만)
# test split 예측 → predictions.json → zip → CodaBench 제출용

from transformers import T5ForConditionalGeneration, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import login
from collections import defaultdict, Counter
import pandas as pd
import torch
import os
import zipfile
import json

# ===== 1. Login =====
login(token=" ")

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

# th/ja/ko만 MFR ON
MFR_GROUPS = {"ja", "ko", "th"}
MFR_THRESHOLD = 0.3

# ===== 3. Data Load =====
print("Loading dataset...")
data = load_dataset("weerayut/multilexnorm2026-dev-pub")
test_df    = data['test'].to_pandas()
train_data = data['train']
print(f"  test: {len(test_df)} sentences")

# ===== 4. MFR 빌드 =====
def build_mfr(train_data, langs):
    counts = defaultdict(Counter)
    for ex in train_data:
        if ex['lang'] not in langs:
            continue
        for r, n in zip(ex['raw'], ex['norm']):
            counts[r][n] += 1
    mfr, mfr_ratio = {}, {}
    for raw, counter in counts.items():
        total = sum(counter.values())
        best_norm, best_count = counter.most_common(1)[0]
        mfr[raw] = best_norm
        mfr_ratio[raw] = best_count / total
    return mfr, mfr_ratio

# ===== 5. 예측 =====
def normalize_sent(raw_sent, model, tokenizer, mfr=None, mfr_ratio=None):
    pred_sent = []
    for i in range(0, len(raw_sent), BATCH_SIZE):
        batch = raw_sent[i:i+BATCH_SIZE]
        inputs = tokenizer(
            batch, return_tensors="pt",
            max_length=MAX_LEN, truncation=True, padding=True
        ).to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for raw, pred in zip(batch, preds):
            final = pred
            if mfr and final != raw:
                mfr_norm = mfr.get(raw, raw)
                ratio    = mfr_ratio.get(raw, 0.0)
                if mfr_norm == raw or ratio < MFR_THRESHOLD:
                    final = raw
            pred_sent.append(final)
    return pred_sent

# ===== 6. 체크포인트 탐색 =====
def find_model_path(group_name):
    group_dir = os.path.join(BASE_DIR, group_name)
    final = os.path.join(group_dir, "final")
    if os.path.exists(final):
        return final
    ckpts = sorted([
        d for d in os.listdir(group_dir)
        if d.startswith('checkpoint-')
    ], key=lambda x: int(x.split('-')[1]))
    if ckpts:
        return os.path.join(group_dir, ckpts[-1])
    return None

# ===== 7. 그룹별 inference =====
print("\nRunning grouped inference...")
tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")
all_preds = {}

for group_name, langs in LANG_GROUPS.items():
    mask     = test_df['lang'].isin(langs)
    group_df = test_df[mask]

    if len(group_df) == 0:
        print(f"[{group_name}] No test sentences — skipping")
        continue

    model_path = find_model_path(group_name)
    if not model_path:
        print(f"[{group_name}] ⚠️  No model found — skipping")
        continue

    print(f"\n[{group_name}] {model_path}")
    print(f"  sentences: {len(group_df)} ({', '.join(langs)})")

    # MFR (th/ja/ko만)
    use_mfr = group_name in MFR_GROUPS
    if use_mfr:
        mfr, mfr_ratio = build_mfr(train_data, langs)
        print(f"  MFR dict: {len(mfr)} entries (ON)")
    else:
        mfr, mfr_ratio = None, None
        print(f"  MFR: OFF")

    model = T5ForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.eval()

    done = 0
    for idx, row in group_df.iterrows():
        pred = normalize_sent(list(row['raw']), model, tokenizer, mfr, mfr_ratio)
        all_preds[idx] = pred
        done += 1
        if done % 500 == 0:
            print(f"  {done}/{len(group_df)} ({done/len(group_df)*100:.1f}%)")

    print(f"  ✅ {done} sentences done")
    del model
    torch.cuda.empty_cache()

# ===== 8. 저장 (원래 포맷 그대로) =====
test_df['pred'] = test_df.index.map(lambda i: all_preds.get(i, test_df.loc[i, 'raw']))

os.makedirs(SAVE_PATH, exist_ok=True)
out = test_df[['raw', 'norm', 'lang', 'pred']]
out.to_json(f"{SAVE_PATH}/predictions.json", orient="records")
print(f"\nSaved: {SAVE_PATH}/predictions.json")

# ===== 9. Zip =====
zip_path = f"{SAVE_PATH}.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(f"{SAVE_PATH}/predictions.json", arcname="predictions.json")
print(f"\nCreated: {zip_path}")
print(f"\n✅ Upload to CodaBench:")
print(f"   https://www.codabench.org/competitions/14162/?secret_key=33d4b8ec-4951-478b-8132-474e458409c3")
