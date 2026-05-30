# ===== mfr_hybrid_inference.py =====
# grouped 모델 + MFR 하이브리드 + Confidence Threshold 후처리
# 추가 학습 없음 — 기존 체크포인트에 후처리만 적용

import os
import json
import glob
import torch
import torch.nn.functional as F
from collections import defaultdict, Counter
from datasets import load_dataset, Dataset
from huggingface_hub import login
from transformers import T5ForConditionalGeneration, AutoTokenizer

login(token=" ")

# ================================================================
# ↓↓↓ 설정 ↓↓↓

MODELS_DIR = r"C:\Users\kj556\Downloads\multilexnorm\grouped_models"

# 그룹별 담당 언어
GROUP_LANGS = {
    "germanic": ["de", "en", "nl"],
    "slav":     ["hr", "sl", "sr"],
    "sea":      ["id", "iden", "vi"],
    "ja":       ["ja"],
    "ko":       ["ko"],
    "th":       ["th"],
    # turkic, romance는 val 데이터 없으면 스킵됨
}

# 후처리 파라미터 (tuning v1 — 덜 보수적)
MFR_THRESHOLD   = 0.15  # 낮춤: TP 손실 줄이기
CONF_THRESHOLD  = 0.3   # 낮춤: TP 손실 줄이기
USE_MFR         = True
USE_CONF        = True

MAX_LEN    = 128
BATCH_SIZE = 512

# ↑↑↑ 설정 끝 ↑↑↑
# ================================================================


# ===== 1. MFR 딕셔너리 빌드 (train 데이터에서) =====
def build_mfr(ds, langs):
    """각 언어별로 {raw: {norm: count}} 카운팅"""
    counts = defaultdict(Counter)  # counts[raw][norm] = n
    for ex in ds['train']:
        if ex['lang'] not in langs:
            continue
        for r, n in zip(ex['raw'], ex['norm']):
            counts[r][n] += 1
    # MFR: 가장 빈번한 norm
    mfr = {}
    mfr_ratio = {}
    for raw, counter in counts.items():
        total = sum(counter.values())
        best_norm, best_count = counter.most_common(1)[0]
        mfr[raw] = best_norm
        mfr_ratio[raw] = best_count / total  # 얼마나 일관되게 변환됐는지
    return mfr, mfr_ratio


# ===== 2. 최신 체크포인트 자동 탐색 =====
def find_latest_checkpoint(group_dir):
    checkpoints = glob.glob(os.path.join(group_dir, "checkpoint-*"))
    if not checkpoints:
        return group_dir  # checkpoint 없으면 group_dir 자체가 모델
    # 숫자 기준 최신
    checkpoints.sort(key=lambda x: int(x.split("checkpoint-")[-1]))
    return checkpoints[-1]


# ===== 3. ERR 계산 (후처리 포함) =====
def compute_err_with_postprocess(model, tokenizer, val_data, label,
                                  mfr=None, mfr_ratio=None):
    model.eval()
    TP = FP = FN = 0
    results = []

    for i in range(0, len(val_data), BATCH_SIZE):
        batch = val_data[i:i+BATCH_SIZE]
        inputs = tokenizer(
            batch['raw'], return_tensors="pt",
            max_length=MAX_LEN, truncation=True, padding=True
        ).to("cuda")

        with torch.no_grad():
            # beam=1로 logits도 같이 뽑기
            out = model.generate(
                **inputs,
                max_new_tokens=32,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # confidence 계산: 첫 토큰의 max softmax prob
        if USE_CONF and out.scores:
            first_logits = out.scores[0]  # (batch, vocab)
            confs = F.softmax(first_logits, dim=-1).max(dim=-1).values.tolist()
        else:
            confs = [1.0] * len(batch['raw'])

        preds = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)

        for raw, norm, pred, conf in zip(batch['raw'], batch['norm'], preds, confs):
            final_pred = pred

            # --- Confidence Threshold ---
            if USE_CONF and conf < CONF_THRESHOLD:
                final_pred = raw

            # --- MFR 하이브리드 ---
            if USE_MFR and mfr and final_pred != raw:
                mfr_norm = mfr.get(raw, raw)
                ratio    = mfr_ratio.get(raw, 0.0) if mfr_ratio else 0.0
                # 모델이 바꾸려 하는데 MFR은 그대로이거나 일관성이 낮으면 → 원본 유지
                if mfr_norm == raw or ratio < MFR_THRESHOLD:
                    final_pred = raw

            results.append({'raw': raw, 'gold': norm, 'pred': final_pred,
                             'model_pred': pred, 'conf': conf})

            if raw == norm:
                if final_pred != raw: FP += 1
            else:
                if final_pred == norm: TP += 1
                else:                  FN += 1

    ERR = (TP - FP) / (TP + FN) * 100 if (TP + FN) > 0 else 0
    print(f"[{label}] ERR={ERR:.2f}  TP={TP} FP={FP} FN={FN}")
    return ERR, results


# ===== 4. 메인 =====
if __name__ == "__main__":
    print("Loading dataset...")
    ds = load_dataset("weerayut/multilexnorm2026-dev-pub")

    # val 데이터 전체 (lang 포함)
    val_all = Dataset.from_list([
        {'raw': r, 'norm': n, 'lang': ex['lang']}
        for ex in ds['validation']
        for r, n in zip(ex['raw'], ex['norm'])
    ])

    overall_TP = overall_FP = overall_FN = 0
    group_results = {}

    for group, langs in GROUP_LANGS.items():
        group_dir = os.path.join(MODELS_DIR, group)
        if not os.path.exists(group_dir):
            print(f"[SKIP] {group} — 폴더 없음")
            continue

        ckpt = find_latest_checkpoint(group_dir)
        print(f"\n{'='*50}")
        print(f"Group: {group} | langs: {langs}")
        print(f"Checkpoint: {ckpt}")

        tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")
        model = T5ForConditionalGeneration.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16
        ).to("cuda")

        # 이 그룹 언어들의 val 데이터만
        val_group = val_all.filter(lambda x: x['lang'] in langs)
        if len(val_group) == 0:
            print(f"  val 데이터 없음, 스킵")
            continue

        # MFR 빌드
        mfr, mfr_ratio = build_mfr(ds, langs)
        print(f"  MFR dict: {len(mfr)} entries")

        err, results = compute_err_with_postprocess(
            model, tokenizer, val_group, group,
            mfr=mfr, mfr_ratio=mfr_ratio
        )

        # 언어별 분해
        lang_stats = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
        for r in results:
            lang = next((ex['lang'] for ex in ds['validation']
                         if r['raw'] in ex['raw']), 'unk')
        # 간단하게: val_group에 lang 있음
        val_list = val_group.to_list()
        for r, v in zip(results, val_list):
            lang = v['lang']
            raw, norm, pred = r['raw'], r['gold'], r['pred']
            if raw == norm:
                if pred != raw: lang_stats[lang]['FP'] += 1
            else:
                if pred == norm: lang_stats[lang]['TP'] += 1
                else:            lang_stats[lang]['FN'] += 1

        for lang in sorted(lang_stats):
            s = lang_stats[lang]
            d = s['TP'] + s['FN']
            e = (s['TP'] - s['FP']) / d * 100 if d > 0 else 0
            print(f"  {lang:6}: ERR={e:6.2f}  TP={s['TP']} FP={s['FP']} FN={s['FN']}")
            overall_TP += s['TP']
            overall_FP += s['FP']
            overall_FN += s['FN']

        group_results[group] = {'err': err, 'langs': langs}

        # 결과 저장
        out_dir = os.path.join(MODELS_DIR, group, "postprocess_results_tuning1")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "predictions.jsonl"), 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

        del model
        torch.cuda.empty_cache()

    # 전체 ERR
    overall_ERR = (overall_TP - overall_FP) / (overall_TP + overall_FN) * 100 \
        if (overall_TP + overall_FN) > 0 else 0

    print(f"\n{'='*50}")
    print(f"OVERALL ERR (with post-processing): {overall_ERR:.2f}")
    print(f"TP={overall_TP} FP={overall_FP} FN={overall_FN}")
    print(f"{'='*50}")
    print(f"\n파라미터: MFR_THRESHOLD={MFR_THRESHOLD}, CONF_THRESHOLD={CONF_THRESHOLD}")
    print(f"USE_MFR={USE_MFR}, USE_CONF={USE_CONF}")

    # 요약 저장
    summary = {
        "overall_err": overall_ERR,
        "params": {
            "mfr_threshold": MFR_THRESHOLD,
            "conf_threshold": CONF_THRESHOLD,
            "use_mfr": USE_MFR,
            "use_conf": USE_CONF,
        },
        "group_results": group_results,
        "overall_stats": {"TP": overall_TP, "FP": overall_FP, "FN": overall_FN}
    }
    with open(os.path.join(MODELS_DIR, "postprocess_summary_tuning1.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 완료! 요약: {MODELS_DIR}/postprocess_summary.json")
