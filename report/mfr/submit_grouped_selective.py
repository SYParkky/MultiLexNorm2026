# ===== submit_grouped_mfr.py =====
# grouped 모델 + MFR 하이브리드 + Confidence Threshold 후처리
# test split 예측 → predictions.json → zip → CodaBench 제출용

from transformers import T5ForConditionalGeneration, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import login
from collections import defaultdict, Counter
import torch
import torch.nn.functional as F
import os
import zipfile
import json

# ===== 1. Login =====
login(token="")

# ===== 2. Config =====
BATCH_SIZE = 256
MAX_LEN    = 128
BASE_DIR   = "./grouped_models"
SAVE_PATH  = "./outputs/submission_dev"

# 그룹별 후처리 파라미터
GROUP_POSTPROCESS = {
    "germanic": {"use_mfr": False, "use_conf": False, "mfr_threshold": 0.3, "conf_threshold": 0.5},
    "slav":     {"use_mfr": False, "use_conf": False, "mfr_threshold": 0.3, "conf_threshold": 0.5},
    "sea":      {"use_mfr": False, "use_conf": False, "mfr_threshold": 0.3, "conf_threshold": 0.5},
    "ja":       {"use_mfr": True,  "use_conf": False, "mfr_threshold": 0.3, "conf_threshold": 0.5},
    "ko":       {"use_mfr": True,  "use_conf": False, "mfr_threshold": 0.3, "conf_threshold": 0.5},
    "th":       {"use_mfr": True,  "use_conf": False, "mfr_threshold": 0.3, "conf_threshold": 0.5},
    "romance":  {"use_mfr": False, "use_conf": False, "mfr_threshold": 0.3, "conf_threshold": 0.5},
    "turkic":   {"use_mfr": False, "use_conf": False, "mfr_threshold": 0.3, "conf_threshold": 0.5},
}

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

# ===== 3. Data Load =====
print("Loading dataset...")
data = load_dataset("weerayut/multilexnorm2026-dev-pub")
test_data  = data['test']
train_data = data['train']
print(f"  test sentences: {len(test_data)}")

# ===== 4. MFR 딕셔너리 빌드 =====
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

# ===== 5. 후처리 적용 예측 =====
def normalize_with_postprocess(raw_sent, model, tokenizer, mfr, mfr_ratio,
                                use_mfr=True, use_conf=False,
                                mfr_threshold=0.3, conf_threshold=0.5):
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
            out = model.generate(
                **inputs,
                max_new_tokens=32,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # confidence
        if use_conf and out.scores:
            confs = F.softmax(out.scores[0], dim=-1).max(dim=-1).values.tolist()
        else:
            confs = [1.0] * len(batch)

        preds = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)

        for raw, pred, conf in zip(batch, preds, confs):
            final = pred

            # Confidence Threshold
            if use_conf and conf < conf_threshold:
                final = raw

            # MFR 하이브리드
            if use_mfr and final != raw:
                mfr_norm = mfr.get(raw, raw)
                ratio    = mfr_ratio.get(raw, 0.0)
                if mfr_norm == raw or ratio < mfr_threshold:
                    final = raw

            pred_sent.append(final)
    return pred_sent

# ===== 6. 최신 체크포인트 탐색 =====
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
print("\nRunning grouped inference with post-processing...")
tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")

# test 데이터를 리스트로 변환
test_list = list(test_data)
all_preds = [None] * len(test_list)

for group_name, langs in LANG_GROUPS.items():
    # 해당 그룹 인덱스 추출
    indices = [i for i, ex in enumerate(test_list) if ex['lang'] in langs]
    if not indices:
        print(f"[{group_name}] No test sentences — skipping")
        continue

    model_path = find_model_path(group_name)
    if not model_path:
        print(f"[{group_name}] ⚠️  No model found — skipping")
        continue

    print(f"\n[{group_name}] {model_path}")
    print(f"  {len(indices)} sentences | langs: {langs}")

    # MFR 빌드 (use_mfr=True인 그룹만)
    gp = GROUP_POSTPROCESS.get(group_name, {"use_mfr": False, "use_conf": False,
                                             "mfr_threshold": 0.3, "conf_threshold": 0.5})
    if gp["use_mfr"]:
        mfr, mfr_ratio = build_mfr(train_data, langs)
        print(f"  MFR dict: {len(mfr)} entries")
    else:
        mfr, mfr_ratio = None, None

    model = T5ForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.eval()

    for i, idx in enumerate(indices):
        ex   = test_list[idx]
        pred = normalize_with_postprocess(
            list(ex['raw']), model, tokenizer, mfr, mfr_ratio,
            use_mfr=gp["use_mfr"], use_conf=gp["use_conf"],
            mfr_threshold=gp["mfr_threshold"], conf_threshold=gp["conf_threshold"],
        )
        all_preds[idx] = pred
        if (i+1) % 200 == 0:
            print(f"  {i+1}/{len(indices)}")

    print(f"  ✅ done")
    del model
    torch.cuda.empty_cache()

# ===== 8. predictions.json 생성 =====
os.makedirs(SAVE_PATH, exist_ok=True)

output = []
for i, ex in enumerate(test_list):
    pred = all_preds[i] if all_preds[i] is not None else list(ex['raw'])
    output.append({
        "raw":  list(ex['raw']),
        "norm": list(ex['norm']) if 'norm' in ex else list(ex['raw']),
        "lang": ex['lang'],
        "pred": pred,
    })

out_path = os.path.join(SAVE_PATH, "predictions.json")
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False)
print(f"\nSaved: {out_path}")

# ===== 9. Zip =====
zip_path = f"{SAVE_PATH}.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(out_path, arcname="predictions.json")
print(f"Created: {zip_path}")

print(f"\n✅ Upload to CodaBench:")
print(f"   https://www.codabench.org/competitions/14162/?secret_key=33d4b8ec-4951-478b-8132-474e458409c3")
print(f"\n파라미터: th/ja/ko → MFR filter ON (mfr_threshold=0.3) | germanic/slav/sea → OFF")
