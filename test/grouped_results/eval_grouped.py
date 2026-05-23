# ===== eval_grouped.py =====
# 그룹별 모델로 전체 ERR 평가
# 각 언어에 맞는 모델 순차 로드 → 예측 → 합치기

import os
import json
import torch
from datasets import load_dataset, Dataset
from huggingface_hub import login
from transformers import T5ForConditionalGeneration, AutoTokenizer

# ===== 1. Config =====
login(token="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")  # 토큰 바꿔줘

MODEL_NAME = "google/byt5-base"
MAX_LEN    = 128
BATCH_SIZE = 256

LANG_GROUPS = {
    "germanic": ["en", "de", "nl"],
    "slav":     ["hr", "sl", "sr"],
    "sea":      ["id", "vi", "iden"],
    "ja":       ["ja"],
    "ko":       ["ko"],
    "th":       ["th"],
}

BASE_DIR    = "./grouped_models"
RESULTS_DIR = "./grouped_results"
OUTPUT_PATH = "./grouped_results/predictions_grouped.jsonl"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===== 2. 언어 → 그룹 매핑 =====
LANG_TO_GROUP = {}
for group, langs in LANG_GROUPS.items():
    for lang in langs:
        LANG_TO_GROUP[lang] = group

# ===== 3. 데이터 로드 =====
def load_val_by_group():
    dataset = load_dataset("weerayut/multilexnorm2026-dev-pub")
    groups = {}
    for ex in dataset['validation']:
        lang = ex['lang']
        group = LANG_TO_GROUP.get(lang)
        if group is None:
            continue
        if group not in groups:
            groups[group] = []
        for r, n in zip(ex['raw'], ex['norm']):
            groups[group].append({'raw': r, 'norm': n, 'lang': lang})
    return {g: Dataset.from_list(rows) for g, rows in groups.items()}

# ===== 4. ERR 계산 =====
def compute_err_from_results(results):
    TP = FP = FN = 0
    lang_stats = {}
    for r in results:
        raw, norm, pred, lang = r['raw'], r['gold'], r['pred'], r['lang']
        if lang not in lang_stats:
            lang_stats[lang] = {'TP': 0, 'FP': 0, 'FN': 0}
        if raw == norm:
            if pred != raw:
                FP += 1
                lang_stats[lang]['FP'] += 1
        else:
            if pred == norm:
                TP += 1
                lang_stats[lang]['TP'] += 1
            else:
                FN += 1
                lang_stats[lang]['FN'] += 1
    ERR = (TP - FP) / (TP + FN) * 100 if (TP + FN) > 0 else 0
    return ERR, TP, FP, FN, lang_stats

# ===== 5. 그룹별 예측 =====
def predict_group(group_name, val_data, tokenizer):
    model_path = os.path.join(BASE_DIR, group_name, "final")
    if not os.path.exists(model_path):
        # final 없으면 마지막 체크포인트 사용
        ckpts = sorted([
            d for d in os.listdir(os.path.join(BASE_DIR, group_name))
            if d.startswith('checkpoint-')
        ], key=lambda x: int(x.split('-')[1]))
        if not ckpts:
            print(f"⚠️  No model found for {group_name} — skipping")
            return []
        model_path = os.path.join(BASE_DIR, group_name, ckpts[-1])

    print(f"\nLoading {group_name} from {model_path}...")
    model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model = model.to("cuda")
    model.eval()

    results = []
    for i in range(0, len(val_data), BATCH_SIZE):
        batch = val_data[i:i+BATCH_SIZE]
        inputs = tokenizer(
            batch['raw'], return_tensors="pt",
            max_length=MAX_LEN, truncation=True, padding=True
        ).to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for raw, norm, pred, lang in zip(batch['raw'], batch['norm'], preds, batch['lang']):
            results.append({'raw': raw, 'gold': norm, 'pred': pred, 'lang': lang})

    del model
    torch.cuda.empty_cache()
    return results

# ===== 6. Main =====
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading validation data by group...")
    val_by_group = load_val_by_group()

    all_results = []
    group_errs  = {}

    for group_name in LANG_GROUPS.keys():
        if group_name not in val_by_group:
            print(f"⚠️  No validation data for {group_name}")
            continue

        results = predict_group(group_name, val_by_group[group_name], tokenizer)
        if not results:
            continue

        all_results.extend(results)

        # 그룹별 ERR
        err, tp, fp, fn, lang_stats = compute_err_from_results(results)
        group_errs[group_name] = err
        print(f"[{group_name}] ERR: {err:.2f}  (TP={tp}, FP={fp}, FN={fn})")
        for lang in sorted(lang_stats):
            s = lang_stats[lang]
            d = s['TP'] + s['FN']
            e = (s['TP'] - s['FP']) / d * 100 if d > 0 else 0
            print(f"  {lang:6s}: ERR={e:6.2f}  (TP={s['TP']}, FP={s['FP']}, FN={s['FN']})")

    # 전체 ERR
    print("\n" + "="*50)
    print("OVERALL ERR (grouped models)")
    print("="*50)
    overall_err, tp, fp, fn, lang_stats = compute_err_from_results(all_results)
    print(f"Overall: ERR = {overall_err:.2f}  (TP={tp}, FP={fp}, FN={fn})")
    print("\nPer language:")
    for lang in sorted(lang_stats):
        s = lang_stats[lang]
        d = s['TP'] + s['FN']
        e = (s['TP'] - s['FP']) / d * 100 if d > 0 else 0
        print(f"  {lang:6s}: ERR={e:6.2f}  (TP={s['TP']}, FP={s['FP']}, FN={s['FN']})")

    # 저장
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"\n✅ Predictions saved to {OUTPUT_PATH}")

    with open(os.path.join(RESULTS_DIR, "grouped_summary.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'overall_err': overall_err,
            'group_errs': group_errs,
            'lang_stats': {
                lang: {
                    'err': (s['TP']-s['FP'])/(s['TP']+s['FN'])*100 if (s['TP']+s['FN'])>0 else 0,
                    **s
                }
                for lang, s in lang_stats.items()
            }
        }, f, ensure_ascii=False, indent=2)
