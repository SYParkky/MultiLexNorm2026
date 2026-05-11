from transformers import T5ForConditionalGeneration, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import login
import pandas as pd
import torch
import os
import zipfile

# ===== 1. Login =====
login(token="hf_")  # replace

# ===== 2. Config =====
MODEL_PATH = "./byt5-base-finetuned-v3-final"  # 제출할 모델 경로
SAVE_PATH  = "./outputs/submission_dev"
BATCH_SIZE = 256

# ===== 3. Model Load =====
print(f"Loading model: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16
)
model = model.to("cuda")
model.eval()
print("Model loaded!")

# ===== 4. Data Load =====
print("\nLoading dataset...")
data = load_dataset("weerayut/multilexnorm2026-dev-pub")
print(f"  test: {len(data['test'])} sentences")

# ===== 5. Batch Prediction =====
def normalize_sent(raw_sent):
    """문장(list[str]) → 예측 list[str]"""
    pred_sent = []
    for i in range(0, len(raw_sent), BATCH_SIZE):
        batch = raw_sent[i:i+BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding=True
        ).to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        pred_sent.extend(preds)
    return pred_sent

# ===== 6. Inference on test split =====
print("\nRunning inference on test split...")
test_df = data['test'].to_pandas()

preds = []
total = len(test_df)
for idx, row in test_df.iterrows():
    pred_sent = normalize_sent(row['raw'])
    preds.append(pred_sent)
    if (idx + 1) % 500 == 0:
        print(f"  {idx+1}/{total} ({(idx+1)/total*100:.1f}%)")

test_df['pred'] = preds
print(f"Inference complete!")

# ===== 7. Save predictions.json =====
# 형식: raw, norm, lang, pred (norm은 test라 빈 문자열)
os.makedirs(SAVE_PATH, exist_ok=True)
out = test_df[['raw', 'norm', 'lang', 'pred']]
out.to_json(f"{SAVE_PATH}/predictions.json", orient="records")
print(f"\nSaved: {SAVE_PATH}/predictions.json")

# sanity check
print("\nSample (first 2):")
for _, row in out.head(2).iterrows():
    print(f"  [{row['lang']}]")
    print(f"    raw:  {row['raw'][:3]}")
    print(f"    pred: {row['pred'][:3]}")

# ===== 8. Zip =====
zip_path = f"{SAVE_PATH}.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(f"{SAVE_PATH}/predictions.json", arcname="predictions.json")
print(f"\nCreated: {zip_path}")
print(f"\n✅ Upload to CodaBench:")
print(f"   https://www.codabench.org/competitions/14162/?secret_key=33d4b8ec-4951-478b-8132-474e458409c3")#change key
