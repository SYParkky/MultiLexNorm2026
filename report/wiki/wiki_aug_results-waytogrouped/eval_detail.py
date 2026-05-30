# ===== eval_detail.py =====
# 실험 A vs B 예측 결과 상세 분석 (ko/en/ja/de)

import json
import csv
import os
from collections import Counter

RESULTS_DIR = "./experiment_results"
OUTPUT_CSV  = f"{RESULTS_DIR}/comparison.csv"

def load_preds(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

print("Loading predictions...")
preds_A = {r['raw']: r for r in load_preds(f"{RESULTS_DIR}/predictions_A.jsonl")}
preds_B = {r['raw']: r for r in load_preds(f"{RESULTS_DIR}/predictions_B.jsonl")}

all_raws = sorted(set(preds_A.keys()) | set(preds_B.keys()))
print(f"Total unique words: {len(all_raws)}")

rows = []
for raw in all_raws:
    a = preds_A.get(raw, {})
    b = preds_B.get(raw, {})
    gold   = a.get('gold') or b.get('gold', '')
    pred_a = a.get('pred', '')
    pred_b = b.get('pred', '')
    lang   = a.get('lang') or b.get('lang', '')

    is_nonstandard = (raw != gold)
    a_correct = (pred_a == gold)
    b_correct = (pred_b == gold)

    if is_nonstandard:
        if not a_correct and b_correct:     change = 'B_only_correct'
        elif a_correct and not b_correct:   change = 'A_only_correct'
        elif a_correct and b_correct:       change = 'both_correct'
        else:                               change = 'both_wrong'
    else:
        a_fp = (pred_a != raw)
        b_fp = (pred_b != raw)
        if not a_fp and b_fp:               change = 'B_new_FP'
        elif a_fp and not b_fp:             change = 'B_fixed_FP'
        elif not a_fp and not b_fp:         change = 'both_TN'
        else:                               change = 'both_FP'

    rows.append({
        'lang': lang, 'raw': raw, 'gold': gold,
        'pred_A': pred_a, 'pred_B': pred_b,
        'is_nonstandard': is_nonstandard,
        'A_correct': a_correct, 'B_correct': b_correct,
        'change': change,
    })

# Save CSV
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'lang','raw','gold','pred_A','pred_B',
        'is_nonstandard','A_correct','B_correct','change'
    ])
    writer.writeheader()
    writer.writerows(rows)
print(f"\n✅ Saved to {OUTPUT_CSV}")

# Summary by language
print("\n" + "="*50)
print("SUMMARY BY LANGUAGE")
print("="*50)

langs = sorted(set(r['lang'] for r in rows))
for lang in langs:
    lang_rows = [r for r in rows if r['lang'] == lang]
    counts = Counter(r['change'] for r in lang_rows)
    nonstandard = [r for r in lang_rows if r['is_nonstandard']]
    print(f"\n[{lang}] total={len(lang_rows)}, nonstandard={len(nonstandard)}")
    print(f"  B_only_correct:  {counts['B_only_correct']:4d}  ← 증강으로 개선")
    print(f"  A_only_correct:  {counts['A_only_correct']:4d}  ← 증강으로 나빠짐")
    print(f"  both_correct:    {counts['both_correct']:4d}")
    print(f"  both_wrong:      {counts['both_wrong']:4d}")
    print(f"  B_new_FP:        {counts['B_new_FP']:4d}  ← 증강 후 과정규화")
    print(f"  B_fixed_FP:      {counts['B_fixed_FP']:4d}  ← 증강 후 FP 수정")

# 흥미로운 케이스
print("\n" + "="*50)
print("B_only_correct (증강으로 개선된 케이스)")
print("="*50)
for r in rows:
    if r['change'] == 'B_only_correct':
        print(f"  [{r['lang']}] '{r['raw']}' → gold:'{r['gold']}' | A:'{r['pred_A']}' | B:'{r['pred_B']}'")

print("\n" + "="*50)
print("A_only_correct (증강으로 나빠진 케이스)")
print("="*50)
for r in rows:
    if r['change'] == 'A_only_correct':
        print(f"  [{r['lang']}] '{r['raw']}' → gold:'{r['gold']}' | A:'{r['pred_A']}' | B:'{r['pred_B']}'")
