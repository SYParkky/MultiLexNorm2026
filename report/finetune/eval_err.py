from transformers import T5ForConditionalGeneration, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import login
import torch

# ===== 1. 로그인 =====
login(token="hf_")  # replace

# ===== 2. 모델 로드 =====
MODEL_PATH = "./byt5-large-finetuned-0511/checkpoint-20031"

print(f"모델 로딩: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16
)
model = model.to("cuda")
model.eval()
print("모델 로딩 완료!")

# ===== 3. 데이터 로드 =====
dataset = load_dataset("weerayut/multilexnorm2026-dev-pub")

def flatten(split):
    rows = []
    for example in split:
        for r, n in zip(example['raw'], example['norm']):
            rows.append({
                'raw':  r,
                'norm': n,
                'lang': example['lang']
            })
    return rows

val_data = flatten(dataset['validation'])
print(f"평가 데이터: {len(val_data)}개")

# ===== 4. 배치 예측 함수 =====
BATCH_SIZE = 256  # 한 번에 256개 처리

def normalize_batch(words):
    inputs = tokenizer(
        words,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding=True
    ).to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=32)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# ===== 5. ERR 계산 =====
TP = FP = FN = 0
lang_stats = {}

print(f"\n평가 중... (배치 크기: {BATCH_SIZE})")

for i in range(0, len(val_data), BATCH_SIZE):
    batch = val_data[i:i+BATCH_SIZE]
    words = [item['raw'] for item in batch]
    preds = normalize_batch(words)

    for item, pred in zip(batch, preds):
        raw  = item['raw']
        norm = item['norm']
        lang = item['lang']

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

    progress = min(i + BATCH_SIZE, len(val_data))
    print(f"  {progress}/{len(val_data)} ({progress/len(val_data)*100:.1f}%)")

# ===== 6. 결과 출력 =====
ERR = (TP - FP) / (TP + FN) if (TP + FN) > 0 else 0

print(f"\n{'='*40}")
print(f"전체 ERR 결과")
print(f"{'='*40}")
print(f"TP: {TP}, FP: {FP}, FN: {FN}")
print(f"ERR:            {ERR*100:.2f}")
print(f"MFR 베이스라인:  39.02")
print(f"개선:           {ERR*100 - 39.02:+.2f}")

print(f"\n{'='*40}")
print(f"언어별 ERR")
print(f"{'='*40}")
for lang, s in sorted(lang_stats.items()):
    tp, fp, fn = s['TP'], s['FP'], s['FN']
    if (tp + fn) > 0:
        err = (tp - fp) / (tp + fn) * 100
        print(f"{lang:4s}: ERR={err:6.2f}  (TP={tp}, FP={fp}, FN={fn})")
